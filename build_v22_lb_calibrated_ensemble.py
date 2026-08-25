"""Build leaderboard-score-calibrated log ensembles.

For predictions p_i in log1p space and public losses L_i = RMSLE_i ** 2:

    Cov(error_i, error_j) =
        (L_i + L_j - mean((p_i - p_j) ** 2)) / 2

When the public rows are a random sample of test.csv, pairwise distances on the
full test recover the public error covariance without knowing public targets.
The resulting quadratic form can be minimized for non-negative blend weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MODEL_SCORES = {
    "v8r": ("submission_v8r_verified_final.csv", 1.13312),
    "v9": ("submission_v9_cold_huber_mass.csv", 1.13784),
    "v12": ("submission_v12_cold_master.csv", 1.14332),
    "v16": ("submission_v16_standalone.csv", 1.32029),
}


@dataclass(frozen=True)
class BlendSpec:
    name: str
    weights: dict[str, float]
    predicted_public_rmsle: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def error_covariance(
    losses: np.ndarray,
    pairwise_distance: np.ndarray,
) -> np.ndarray:
    covariance = 0.5 * (
        losses[:, None] + losses[None, :] - pairwise_distance
    )
    return 0.5 * (covariance + covariance.T)


def optimize_nonnegative_weights(
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    count = len(covariance)
    result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        x0=np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
        },
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    return result.x, float(np.sqrt(max(0.0, result.fun)))


def pairwise_log_distance(log_predictions: np.ndarray) -> np.ndarray:
    count = log_predictions.shape[1]
    distance = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            value = float(
                np.mean(
                    np.square(
                        log_predictions[:, left] - log_predictions[:, right]
                    )
                )
            )
            distance[left, right] = value
            distance[right, left] = value
    return distance


def load_predictions(
    project_dir: Path,
) -> tuple[pd.Series, list[str], np.ndarray, np.ndarray]:
    names = list(MODEL_SCORES)
    ids: pd.Series | None = None
    logs: list[np.ndarray] = []
    for name in names:
        filename, _ = MODEL_SCORES[name]
        submission = pd.read_csv(project_dir / filename)
        if list(submission.columns) != ["id", "tuketim"]:
            raise ValueError(f"Unexpected columns in {filename}")
        if ids is None:
            ids = submission["id"]
        elif not ids.equals(submission["id"]):
            raise ValueError(f"ID/order mismatch in {filename}")
        values = submission["tuketim"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid predictions in {filename}")
        logs.append(np.log1p(values))
    assert ids is not None
    losses = np.square([MODEL_SCORES[name][1] for name in names])
    return ids, names, np.column_stack(logs), losses


def score_for_weights(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    return float(np.sqrt(max(0.0, weights @ covariance @ weights)))


def build_specs(
    names: list[str],
    losses: np.ndarray,
    distance: np.ndarray,
) -> list[BlendSpec]:
    covariance = error_covariance(losses, distance)
    full_weights, full_score = optimize_nonnegative_weights(covariance)

    safe_indices = [names.index("v8r"), names.index("v9")]
    safe_covariance = covariance[np.ix_(safe_indices, safe_indices)]
    safe_subweights, safe_score = optimize_nonnegative_weights(safe_covariance)
    safe_weights = np.zeros(len(names), dtype=float)
    safe_weights[safe_indices] = safe_subweights

    anchored_weights = 0.5 * full_weights
    anchored_weights[names.index("v8r")] += 0.5
    anchored_score = score_for_weights(anchored_weights, covariance)

    return [
        BlendSpec(
            name="safe",
            weights=dict(zip(names, safe_weights.tolist())),
            predicted_public_rmsle=safe_score,
        ),
        BlendSpec(
            name="anchored",
            weights=dict(zip(names, anchored_weights.tolist())),
            predicted_public_rmsle=anchored_score,
        ),
        BlendSpec(
            name="full_opt",
            weights=dict(zip(names, full_weights.tolist())),
            predicted_public_rmsle=full_score,
        ),
    ]


def write_blend(
    output_path: Path,
    ids: pd.Series,
    names: list[str],
    log_predictions: np.ndarray,
    weights: dict[str, float],
) -> str:
    weight_array = np.array([weights[name] for name in names], dtype=float)
    if (weight_array < -1e-12).any() or not np.isclose(weight_array.sum(), 1.0):
        raise ValueError(f"Invalid blend weights: {weights}")
    prediction = np.expm1(log_predictions @ weight_array)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise AssertionError("Blend produced invalid values")
    output = pd.DataFrame({"id": ids, "tuketim": prediction})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return sha256(output_path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent
    )
    args = parser.parse_args(argv)

    ids, names, log_predictions, losses = load_predictions(args.project_dir)
    distance = pairwise_log_distance(log_predictions)
    covariance = error_covariance(losses, distance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() < -1e-8:
        raise ValueError(
            "Implied error covariance is not positive semidefinite; "
            "the random-public assumption is inconsistent"
        )
    specs = build_specs(names, losses, distance)

    hashes: dict[str, str] = {}
    for spec in specs:
        output_path = args.output_dir / f"submission_v22_lbcal_{spec.name}.csv"
        hashes[spec.name] = write_blend(
            output_path, ids, names, log_predictions, spec.weights
        )
        print(
            f"{spec.name}: predicted_public_rmsle="
            f"{spec.predicted_public_rmsle:.6f} weights={spec.weights} "
            f"sha256={hashes[spec.name]}"
        )

    report = {
        "assumption": (
            "Public rows are a random/representative subset of test.csv. "
            "Scores are estimates, not guaranteed leaderboard results."
        ),
        "source_models": {
            name: {
                "file": MODEL_SCORES[name][0],
                "public_rmsle": MODEL_SCORES[name][1],
            }
            for name in names
        },
        "pairwise_log_distance": distance.tolist(),
        "implied_error_covariance": covariance.tolist(),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "blends": [
            {
                "name": spec.name,
                "weights": spec.weights,
                "predicted_public_rmsle": spec.predicted_public_rmsle,
                "sha256": hashes[spec.name],
            }
            for spec in specs
        ],
    }
    report_path = args.output_dir / "v22_lb_calibration_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
