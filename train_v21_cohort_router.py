"""Cohort-aware cold-start model and conservative submission router.

The model is limited to information available for a transformer with no target
history: capacity, location, calendar, first appearance in the target panel,
age since appearance, and observable commissioning-cohort structure.

It learns a residual around 2.5 * guc in log space from historically
commissioned transformers. For submission generation, that residual corrects
the proven V8R prediction only for train-unseen transformers. A divergence
gate shrinks large corrections, as required by rolling-origin validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold


LOGGER = logging.getLogger("v21")
SEED = 20260824
BASELINE_MULTIPLIER = 2.5
DEFAULT_GATE_SCALE = 0.90
DEFAULT_MAX_WEIGHT = 0.50

CAT_FEATURES = [
    "il", "ilce", "bolge", "guc_cat", "guc_bin", "age_week",
    "start_month", "start_dow",
]
NUM_FEATURES = [
    "guc", "log_guc", "sqrt_guc", "age_days", "log_age", "is_first_day",
    "month", "dow", "doy", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "age_sin", "age_cos", "cohort_size", "log_cohort_size",
    "cohort_ilce_size", "cohort_bolge_size", "cohort_guc_size",
    "is_mass_cohort",
]
FEATURES = CAT_FEATURES + NUM_FEATURES


@dataclass(frozen=True)
class FoldResult:
    name: str
    rows: int
    facilities: int
    baseline_rmsle: float
    gated_rmsle: float
    direct_rmsle: float
    mass_baseline_rmsle: float | None
    mass_gated_rmsle: float | None
    mean_gate_weight: float

    @property
    def gated_gain(self) -> float:
        return self.baseline_rmsle - self.gated_rmsle


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return RMSLE after enforcing the metric's non-negative domain."""
    true_log = np.log1p(np.clip(np.asarray(y_true, dtype=float), 0.0, None))
    pred_log = np.log1p(np.clip(np.asarray(y_pred, dtype=float), 0.0, None))
    return float(np.sqrt(np.mean(np.square(true_log - pred_log))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_location(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parts = out["lokasyon"].astype(str).str.split(">")
    out["il"] = parts.str[0].fillna("UNKNOWN")
    out["ilce"] = parts.str[-1].fillna("UNKNOWN")
    out["bolge"] = parts.apply(
        lambda values: values[-2]
        if isinstance(values, list) and len(values) >= 3
        else "DOGRUDAN"
    )
    return out


def prepare_panel(
    frame: pd.DataFrame,
    first_seen: pd.Series | None = None,
) -> pd.DataFrame:
    """Build label-free cohort features from one observable panel."""
    required = {"tanim", "guc", "tarih", "lokasyon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Panel columns missing: {sorted(missing)}")

    out = _parse_location(frame)
    if first_seen is None:
        first_seen = out.groupby("tanim", sort=False)["tarih"].min()
    out["first_seen"] = out["tanim"].map(first_seen)
    if out["first_seen"].isna().any():
        raise ValueError("At least one transformer has no first_seen date")

    out["age_days"] = (
        (out["tarih"] - out["first_seen"]).dt.days.clip(lower=0).astype(np.int16)
    )
    out["log_age"] = np.log1p(out["age_days"]).astype(np.float32)
    out["age_week"] = np.minimum(out["age_days"] // 7, 12).astype(str)
    out["is_first_day"] = (out["age_days"] == 0).astype(np.int8)
    out["month"] = out["tarih"].dt.month.astype(np.int8)
    out["dow"] = out["tarih"].dt.dayofweek.astype(np.int8)
    out["doy"] = out["tarih"].dt.dayofyear.astype(np.int16)
    out["start_month"] = out["first_seen"].dt.month.astype(str)
    out["start_dow"] = out["first_seen"].dt.dayofweek.astype(str)
    out["log_guc"] = np.log1p(out["guc"].clip(lower=0)).astype(np.float32)
    out["sqrt_guc"] = np.sqrt(out["guc"].clip(lower=0)).astype(np.float32)
    out["guc_cat"] = out["guc"].astype(str)
    out["guc_bin"] = (
        pd.cut(
            out["guc"],
            [-np.inf, 100, 250, 400, 630, 1000, 1600, 2500, np.inf],
            labels=False,
        )
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    for period, source, stem in [
        (7.0, out["dow"], "dow"),
        (365.25, out["doy"], "doy"),
        (30.4375, out["age_days"], "age"),
    ]:
        out[f"{stem}_sin"] = np.sin(2.0 * np.pi * source / period).astype(
            np.float32
        )
        out[f"{stem}_cos"] = np.cos(2.0 * np.pi * source / period).astype(
            np.float32
        )

    facility_meta = out[
        ["tanim", "first_seen", "ilce", "bolge", "guc_cat"]
    ].drop_duplicates("tanim")
    date_size = facility_meta.groupby("first_seen")["tanim"].nunique()
    date_ilce_size = facility_meta.groupby(["first_seen", "ilce"])["tanim"].nunique()
    date_bolge_size = facility_meta.groupby(["first_seen", "bolge"])["tanim"].nunique()
    date_guc_size = facility_meta.groupby(["first_seen", "guc_cat"])["tanim"].nunique()

    out["cohort_size"] = out["first_seen"].map(date_size).fillna(1).astype(np.float32)
    out["log_cohort_size"] = np.log1p(out["cohort_size"]).astype(np.float32)
    out["cohort_ilce_size"] = (
        pd.MultiIndex.from_frame(out[["first_seen", "ilce"]])
        .map(date_ilce_size).fillna(1).astype(np.float32)
    )
    out["cohort_bolge_size"] = (
        pd.MultiIndex.from_frame(out[["first_seen", "bolge"]])
        .map(date_bolge_size).fillna(1).astype(np.float32)
    )
    out["cohort_guc_size"] = (
        pd.MultiIndex.from_frame(out[["first_seen", "guc_cat"]])
        .map(date_guc_size).fillna(1).astype(np.float32)
    )
    out["is_mass_cohort"] = (out["cohort_size"] >= 20).astype(np.int8)
    for column in CAT_FEATURES:
        out[column] = out[column].fillna("UNKNOWN").astype(str)
    return out


def commissioned_training_rows(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp | None = None,
    initial_grace_days: int = 7,
) -> pd.DataFrame:
    """Select transformers whose commissioning is observed, not left-truncated."""
    grace_end = panel["tarih"].min() + pd.Timedelta(days=initial_grace_days)
    mask = panel["first_seen"] > grace_end
    if cutoff is not None:
        mask &= (panel["first_seen"] <= cutoff) & (panel["tarih"] <= cutoff)
    selected = panel.loc[mask].copy()
    if selected.empty:
        raise ValueError("No historically commissioned training rows were found")
    return selected


class CohortResidualModel:
    """Regularized CatBoost model for log-residual cold-start correction."""

    def __init__(self, iterations: int = 180) -> None:
        self.iterations = iterations
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            depth=5,
            learning_rate=0.045,
            l2_leaf_reg=30.0,
            random_seed=SEED,
            random_strength=0.4,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    def fit(self, frame: pd.DataFrame) -> "CohortResidualModel":
        baseline_log = np.log1p(
            BASELINE_MULTIPLIER * frame["guc"].clip(lower=0).to_numpy()
        )
        target_log = np.log1p(frame["tuketim"].clip(lower=0).to_numpy())
        self.model.fit(
            frame[FEATURES],
            target_log - baseline_log,
            cat_features=CAT_FEATURES,
        )
        return self

    def predict_log_delta(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(frame[FEATURES]), dtype=float)

    def predict_direct(self, frame: pd.DataFrame) -> np.ndarray:
        baseline_log = np.log1p(
            BASELINE_MULTIPLIER * frame["guc"].clip(lower=0).to_numpy()
        )
        return np.expm1(
            np.clip(baseline_log + self.predict_log_delta(frame), 0.0, None)
        )


def divergence_gate(
    log_delta: np.ndarray,
    scale: float = DEFAULT_GATE_SCALE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> np.ndarray:
    """Shrink corrections whose magnitude was unstable in rolling validation."""
    if scale <= 0:
        raise ValueError("Gate scale must be positive")
    delta = np.asarray(log_delta, dtype=float)
    return max_weight * np.exp(-np.square(np.abs(delta) / scale))


def apply_log_correction(
    base_prediction: np.ndarray,
    log_delta: np.ndarray,
    scale: float = DEFAULT_GATE_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    weights = divergence_gate(log_delta, scale=scale)
    base_log = np.log1p(np.clip(np.asarray(base_prediction, dtype=float), 0.0, None))
    corrected = np.expm1(np.clip(base_log + weights * log_delta, 0.0, None))
    return corrected, weights


def _evaluate_predictions(
    name: str,
    frame: pd.DataFrame,
    log_delta: np.ndarray,
    gate_scale: float,
) -> FoldResult:
    y_true = frame["tuketim"].to_numpy()
    baseline = BASELINE_MULTIPLIER * frame["guc"].to_numpy()
    direct = np.expm1(
        np.clip(np.log1p(baseline) + np.asarray(log_delta, dtype=float), 0.0, None)
    )
    gated, weights = apply_log_correction(baseline, log_delta, gate_scale)
    mass = frame["is_mass_cohort"].to_numpy(dtype=bool)
    return FoldResult(
        name=name,
        rows=len(frame),
        facilities=int(frame["tanim"].nunique()),
        baseline_rmsle=calculate_rmsle(y_true, baseline),
        gated_rmsle=calculate_rmsle(y_true, gated),
        direct_rmsle=calculate_rmsle(y_true, direct),
        mass_baseline_rmsle=(
            calculate_rmsle(y_true[mass], baseline[mass]) if mass.any() else None
        ),
        mass_gated_rmsle=(
            calculate_rmsle(y_true[mass], gated[mass]) if mass.any() else None
        ),
        mean_gate_weight=float(weights.mean()),
    )


def validate_cohort_dates(
    panel: pd.DataFrame,
    iterations: int,
    gate_scale: float,
) -> FoldResult:
    """Hold out whole commissioning dates inside the competition-season proxy."""
    validation = panel[
        (panel["first_seen"] > pd.Timestamp("2025-03-31"))
        & (panel["tarih"] >= pd.Timestamp("2025-04-01"))
        & (panel["tarih"] <= pd.Timestamp("2025-07-31"))
    ].copy()
    groups = validation["first_seen"].astype(str).to_numpy()
    oof_delta = np.zeros(len(validation), dtype=float)
    splitter = GroupKFold(n_splits=3)
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(validation, groups=groups), start=1
    ):
        LOGGER.info(
            "Commissioning-date fold %d/3: train=%s valid=%s",
            fold, f"{len(train_index):,}", f"{len(valid_index):,}",
        )
        model = CohortResidualModel(iterations).fit(validation.iloc[train_index])
        oof_delta[valid_index] = model.predict_log_delta(validation.iloc[valid_index])
    return _evaluate_predictions(
        "commissioning_date_groupkfold", validation, oof_delta, gate_scale
    )


def validate_rolling_origins(
    panel: pd.DataFrame,
    iterations: int,
    gate_scale: float,
) -> list[FoldResult]:
    """Strictly train before each cutoff and score later commissioned transformers."""
    folds = [
        ("2025-03-31", "2025-07-31"),
        ("2025-06-30", "2025-09-30"),
        ("2025-09-30", "2025-12-31"),
        ("2025-12-31", "2026-03-31"),
    ]
    results: list[FoldResult] = []
    for cutoff_text, end_text in folds:
        cutoff = pd.Timestamp(cutoff_text)
        end = pd.Timestamp(end_text)
        fit_frame = commissioned_training_rows(panel, cutoff=cutoff)
        validation = panel[
            (panel["first_seen"] > cutoff)
            & (panel["tarih"] > cutoff)
            & (panel["tarih"] <= end)
        ].copy()
        LOGGER.info(
            "Rolling %s -> %s: train_fac=%s valid_fac=%s",
            cutoff.date(), end.date(),
            f"{fit_frame['tanim'].nunique():,}",
            f"{validation['tanim'].nunique():,}",
        )
        model = CohortResidualModel(iterations).fit(fit_frame)
        log_delta = model.predict_log_delta(validation)
        results.append(
            _evaluate_predictions(
                f"rolling_{cutoff_text}_{end_text}",
                validation, log_delta, gate_scale,
            )
        )
    return results


def validate_full_cohorts(
    panel: pd.DataFrame,
    iterations: int,
    gate_scale: float,
) -> FoldResult:
    """Optional expensive CV over every observed commissioning-date cohort."""
    commissioned = commissioned_training_rows(panel)
    groups = commissioned["first_seen"].astype(str).to_numpy()
    oof_delta = np.zeros(len(commissioned), dtype=float)
    splitter = GroupKFold(n_splits=3)
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(commissioned, groups=groups), start=1
    ):
        LOGGER.info(
            "Full cohort fold %d/3: train=%s valid=%s",
            fold, f"{len(train_index):,}", f"{len(valid_index):,}",
        )
        model = CohortResidualModel(iterations).fit(commissioned.iloc[train_index])
        oof_delta[valid_index] = model.predict_log_delta(commissioned.iloc[valid_index])
    return _evaluate_predictions(
        "full_commissioning_date_groupkfold",
        commissioned, oof_delta, gate_scale,
    )


def run_validation(
    data_dir: Path,
    output_report: Path,
    iterations: int,
    gate_scale: float,
    deep: bool,
) -> list[FoldResult]:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["tarih"])
    panel = prepare_panel(train)
    results = [validate_cohort_dates(panel, iterations, gate_scale)]
    results.extend(validate_rolling_origins(panel, iterations, gate_scale))
    if deep:
        results.append(validate_full_cohorts(panel, iterations, gate_scale))

    payload = {
        "iterations": iterations,
        "gate_scale": gate_scale,
        "results": [
            {**asdict(result), "gated_gain": result.gated_gain}
            for result in results
        ],
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for result in results:
        LOGGER.info(
            "%-42s baseline=%.6f gated=%.6f gain=%+.6f direct=%.6f weight=%.3f",
            result.name, result.baseline_rmsle, result.gated_rmsle,
            result.gated_gain, result.direct_rmsle, result.mean_gate_weight,
        )
    LOGGER.info("Validation report: %s", output_report)
    return results


def _validate_submission_inputs(
    test: pd.DataFrame,
    base: pd.DataFrame,
) -> None:
    if list(base.columns) != ["id", "tuketim"]:
        raise ValueError("Base submission columns must be exactly ['id', 'tuketim']")
    if len(test) != len(base):
        raise ValueError("Test and base submission row counts differ")
    if not test["id"].equals(base["id"]):
        raise ValueError("Base submission IDs/order do not match test.csv")
    if base["tuketim"].isna().any() or not np.isfinite(base["tuketim"]).all():
        raise ValueError("Base submission contains non-finite predictions")
    if (base["tuketim"] < 0).any():
        raise ValueError("Base submission contains negative predictions")


def build_submission(
    data_dir: Path,
    base_submission_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    iterations: int,
    gate_scale: float,
) -> dict[str, object]:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["tarih"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["tarih"])
    base = pd.read_csv(base_submission_path)
    _validate_submission_inputs(test, base)

    train_panel = prepare_panel(train)
    known_transformers = set(train["tanim"].unique())
    cold_mask = ~test["tanim"].isin(known_transformers)
    cold_raw = test.loc[cold_mask].copy()
    cold_first_seen = cold_raw.groupby("tanim", sort=False)["tarih"].min()
    cold_panel = prepare_panel(cold_raw, first_seen=cold_first_seen)
    commissioned = commissioned_training_rows(train_panel)
    LOGGER.info(
        "Final fit: %s rows, %s transformers; cold test: %s rows, %s transformers",
        f"{len(commissioned):,}", f"{commissioned['tanim'].nunique():,}",
        f"{len(cold_panel):,}", f"{cold_panel['tanim'].nunique():,}",
    )

    model = CohortResidualModel(iterations).fit(commissioned)
    log_delta = model.predict_log_delta(cold_panel)
    base_cold = base.loc[cold_mask, "tuketim"].to_numpy()
    corrected_cold, weights = apply_log_correction(
        base_cold, log_delta, scale=gate_scale
    )
    ceiling = 36.0 * (cold_panel["guc"].to_numpy() + 1.0)
    corrected_cold = np.clip(corrected_cold, 0.0, ceiling)

    output = base.copy()
    original_warm = output.loc[~cold_mask, "tuketim"].to_numpy(copy=True)
    output.loc[cold_mask, "tuketim"] = corrected_cold
    if not np.array_equal(
        output.loc[~cold_mask, "tuketim"].to_numpy(), original_warm
    ):
        raise AssertionError("Warm predictions changed")
    if output["tuketim"].isna().any() or not np.isfinite(output["tuketim"]).all():
        raise AssertionError("Generated submission contains non-finite values")
    if (output["tuketim"] < 0).any():
        raise AssertionError("Generated submission contains negative values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    diagnostics = pd.DataFrame(
        {
            "id": cold_raw["id"].to_numpy(),
            "tanim": cold_raw["tanim"].to_numpy(),
            "tarih": cold_raw["tarih"].to_numpy(),
            "guc": cold_raw["guc"].to_numpy(),
            "first_seen": cold_panel["first_seen"].to_numpy(),
            "age_days": cold_panel["age_days"].to_numpy(),
            "cohort_size": cold_panel["cohort_size"].to_numpy(),
            "base_prediction": base_cold,
            "raw_log_delta": log_delta,
            "gate_weight": weights,
            "v21_prediction": corrected_cold,
        }
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_path, index=False)

    summary: dict[str, object] = {
        "rows": len(output),
        "cold_rows": int(cold_mask.sum()),
        "cold_transformers": int(cold_raw["tanim"].nunique()),
        "warm_rows_unchanged": int((~cold_mask).sum()),
        "base_cold_median": float(np.median(base_cold)),
        "v21_cold_median": float(np.median(corrected_cold)),
        "mean_gate_weight": float(weights.mean()),
        "p01_gate_weight": float(np.quantile(weights, 0.01)),
        "p99_gate_weight": float(np.quantile(weights, 0.99)),
        "zero_predictions": int((corrected_cold == 0).sum()),
        "output_sha256": sha256(output_path),
        "diagnostics_sha256": sha256(diagnostics_path),
    }
    LOGGER.info("Submission summary:\n%s", json.dumps(summary, indent=2))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, required=True)
    common.add_argument("--iterations", type=int, default=180)
    common.add_argument("--gate-scale", type=float, default=DEFAULT_GATE_SCALE)

    validate_parser = subparsers.add_parser(
        "validate", parents=[common],
        help="Run cohort and rolling-origin behavioral validation",
    )
    validate_parser.add_argument(
        "--report", type=Path, default=Path("v21_validation_results.json")
    )
    validate_parser.add_argument(
        "--deep", action="store_true",
        help="Also run CV over every historical commissioning cohort",
    )
    submit_parser = subparsers.add_parser(
        "submit", parents=[common],
        help="Train on full history and replace only train-unseen test rows",
    )
    submit_parser.add_argument("--base-submission", type=Path, required=True)
    submit_parser.add_argument(
        "--output", type=Path, default=Path("submission_v21_cohort_router.csv")
    )
    submit_parser.add_argument(
        "--diagnostics", type=Path, default=Path("v21_cold_diagnostics.csv")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    if args.command == "validate":
        run_validation(
            args.data_dir, args.report, args.iterations, args.gate_scale, args.deep
        )
    elif args.command == "submit":
        build_submission(
            args.data_dir, args.base_submission, args.output, args.diagnostics,
            args.iterations, args.gate_scale,
        )
    else:
        raise AssertionError(f"Unexpected command: {args.command}")


if __name__ == "__main__":
    main()
