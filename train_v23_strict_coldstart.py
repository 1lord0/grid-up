"""Strict rolling-origin cold-start experiments and submission builder.

This module deliberately excludes transformer target history from the cold models.
Every validation fold is fitted only on rows at or before its cutoff.  The final
submission changes only transformers absent from the training data and preserves
the official test ID order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


LOGGER = logging.getLogger("v23_strict_coldstart")
SEED = 20260828
DATA_START_GRACE_DAYS = 7

ROLLING_FOLDS = (
    ("2025-03-31", "2025-07-31"),
    ("2025-06-30", "2025-09-30"),
    ("2025-09-30", "2025-12-31"),
    ("2025-12-31", "2026-03-31"),
)

# These are deterministic calendar facts, not post-cutoff measured weather.
EVENT_DATES = {
    "2025-04-23": "national_holiday",
    "2025-05-01": "national_holiday",
    "2025-05-19": "national_holiday",
    "2025-06-05": "religious_eve",
    "2025-06-06": "religious_holiday",
    "2025-06-07": "religious_holiday",
    "2025-06-08": "religious_holiday",
    "2025-06-09": "religious_holiday",
    "2025-07-15": "national_holiday",
    "2025-08-30": "national_holiday",
    "2025-10-28": "national_eve",
    "2025-10-29": "national_holiday",
    "2026-01-01": "national_holiday",
    "2026-03-19": "religious_eve",
    "2026-03-20": "religious_holiday",
    "2026-03-21": "religious_holiday",
    "2026-03-22": "religious_holiday",
    "2026-04-23": "national_holiday",
    "2026-05-01": "national_holiday",
    "2026-05-19": "national_holiday",
    "2026-05-26": "religious_eve",
    "2026-05-27": "religious_holiday",
    "2026-05-28": "religious_holiday",
    "2026-05-29": "religious_holiday",
    "2026-05-30": "religious_holiday",
    "2026-07-15": "national_holiday",
}

STATIC_CAT = [
    "il",
    "bolge",
    "ilce",
    "guc_cat",
    "guc_bin",
    "month_cat",
    "dow_cat",
    "event_type",
    "event_window",
]
STATIC_NUM = [
    "log_guc",
    "sqrt_guc",
    "is_weekend",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "event_distance_abs",
    "event_distance_signed",
    "school_closed",
]
STATIC_FEATURES = STATIC_CAT + STATIC_NUM

COHORT_CAT = STATIC_CAT + ["age_week", "start_month", "start_dow"]
COHORT_NUM = STATIC_NUM + [
    "age_days",
    "log_age",
    "is_first_day",
    "log_cohort_size",
    "cohort_ilce_size",
    "cohort_bolge_size",
    "cohort_guc_size",
    "is_mass_cohort",
    "age_sin",
    "age_cos",
]
COHORT_FEATURES = COHORT_CAT + COHORT_NUM

EB_LEVELS = (
    (("month_cat",), 120.0),
    (("guc_bin",), 100.0),
    (("bolge",), 100.0),
    (("month_cat", "dow_cat"), 80.0),
    (("guc_bin", "month_cat"), 70.0),
    (("bolge", "month_cat"), 60.0),
    (("ilce", "guc_bin"), 50.0),
    (("ilce", "guc_bin", "month_cat"), 40.0),
    (("ilce", "guc_bin", "month_cat", "dow_cat"), 30.0),
)


@dataclass
class FoldMetric:
    fold: str
    rows: int
    facilities: int
    mass_rows: int
    rmsle_power: float
    rmsle_static: float
    rmsle_cohort: float
    blend_rmsle: dict[str, float]
    mass_rmsle_static: float | None
    mass_rmsle_cohort: float | None


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_log = np.log1p(np.maximum(0.0, np.asarray(y_true, dtype=float)))
    pred_log = np.log1p(np.maximum(0.0, np.asarray(y_pred, dtype=float)))
    return float(np.sqrt(np.mean(np.square(true_log - pred_log))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_location(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parts = out["lokasyon"].fillna("UNKNOWN").astype(str).str.split(">")
    out["il"] = parts.str[0].str.strip().fillna("UNKNOWN")
    out["ilce"] = parts.str[-1].str.strip().fillna("UNKNOWN")
    out["bolge"] = parts.apply(
        lambda values: values[-2].strip()
        if isinstance(values, list) and len(values) >= 3
        else "DOGRUDAN"
    )
    return out


def _nearest_event_features(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    event_days = np.array(
        [np.datetime64(pd.Timestamp(day).date(), "D") for day in EVENT_DATES],
        dtype="datetime64[D]",
    )
    row_days = dates.values.astype("datetime64[D]")
    signed = np.full(len(row_days), 32, dtype=np.int16)
    best_abs = np.full(len(row_days), 32, dtype=np.int16)
    for event_day in event_days:
        distance = (row_days - event_day).astype("timedelta64[D]").astype(np.int16)
        distance_abs = np.abs(distance)
        better = distance_abs < best_abs
        best_abs[better] = distance_abs[better]
        signed[better] = distance[better]
    return np.abs(signed).clip(0, 31).astype(np.int8), signed.clip(-31, 31).astype(np.int8)


def _school_closed(dates: pd.Series) -> np.ndarray:
    values = dates.values.astype("datetime64[D]")
    closed = (
        ((values >= np.datetime64("2025-06-21")) & (values < np.datetime64("2025-09-08")))
        | ((values >= np.datetime64("2026-01-17")) & (values < np.datetime64("2026-02-02")))
        | (values >= np.datetime64("2026-06-27"))
    )
    return closed.astype(np.int8)


def prepare_panel(frame: pd.DataFrame, first_seen: pd.Series | None = None) -> pd.DataFrame:
    required = {"tanim", "guc", "tarih", "lokasyon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Panel columns missing: {sorted(missing)}")

    out = _parse_location(frame)
    out["tarih"] = pd.to_datetime(out["tarih"])
    if first_seen is None:
        first_seen = out.groupby("tanim", sort=False)["tarih"].min()
    out["first_seen"] = out["tanim"].map(first_seen)
    if out["first_seen"].isna().any():
        raise ValueError("At least one transformer has no first_seen date")

    out["age_days"] = (out["tarih"] - out["first_seen"]).dt.days.clip(lower=0).astype(np.int16)
    out["log_age"] = np.log1p(out["age_days"]).astype(np.float32)
    out["age_week"] = np.minimum(out["age_days"] // 7, 12).astype(str)
    out["is_first_day"] = (out["age_days"] == 0).astype(np.int8)
    out["month"] = out["tarih"].dt.month.astype(np.int8)
    out["dow"] = out["tarih"].dt.dayofweek.astype(np.int8)
    out["doy"] = out["tarih"].dt.dayofyear.astype(np.int16)
    out["month_cat"] = out["month"].astype(str)
    out["dow_cat"] = out["dow"].astype(str)
    out["start_month"] = out["first_seen"].dt.month.astype(str)
    out["start_dow"] = out["first_seen"].dt.dayofweek.astype(str)
    out["log_guc"] = np.log1p(out["guc"].clip(lower=0)).astype(np.float32)
    out["sqrt_guc"] = np.sqrt(out["guc"].clip(lower=0)).astype(np.float32)
    out["guc_cat"] = out["guc"].astype(str)
    out["guc_bin"] = pd.cut(
        out["guc"],
        [-np.inf, 100, 250, 400, 630, 1000, 1600, 2500, np.inf],
        labels=False,
    ).fillna(-1).astype(int).astype(str)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)

    for period, source, stem in (
        (7.0, out["dow"], "dow"),
        (365.25, out["doy"], "doy"),
        (30.4375, out["age_days"], "age"),
    ):
        out[f"{stem}_sin"] = np.sin(2.0 * np.pi * source / period).astype(np.float32)
        out[f"{stem}_cos"] = np.cos(2.0 * np.pi * source / period).astype(np.float32)

    date_text = out["tarih"].dt.strftime("%Y-%m-%d")
    out["event_type"] = date_text.map(EVENT_DATES).fillna("regular")
    event_abs, event_signed = _nearest_event_features(out["tarih"])
    out["event_distance_abs"] = event_abs
    out["event_distance_signed"] = event_signed
    out["event_window"] = np.where(
        event_abs <= 3,
        pd.Series(event_signed, index=out.index).astype(str),
        "regular",
    )
    out["school_closed"] = _school_closed(out["tarih"])

    facility_meta = out[["tanim", "first_seen", "ilce", "bolge", "guc_cat"]].drop_duplicates("tanim")
    cohort_size = facility_meta.groupby("first_seen")["tanim"].nunique()
    cohort_ilce = facility_meta.groupby(["first_seen", "ilce"])["tanim"].nunique()
    cohort_bolge = facility_meta.groupby(["first_seen", "bolge"])["tanim"].nunique()
    cohort_guc = facility_meta.groupby(["first_seen", "guc_cat"])["tanim"].nunique()
    out["cohort_size"] = out["first_seen"].map(cohort_size).fillna(1).astype(np.float32)
    out["log_cohort_size"] = np.log1p(out["cohort_size"]).astype(np.float32)
    for target, keys, source in (
        ("cohort_ilce_size", ["first_seen", "ilce"], cohort_ilce),
        ("cohort_bolge_size", ["first_seen", "bolge"], cohort_bolge),
        ("cohort_guc_size", ["first_seen", "guc_cat"], cohort_guc),
    ):
        out[target] = pd.MultiIndex.from_frame(out[keys]).map(source).fillna(1).astype(np.float32)
    out["is_mass_cohort"] = (out["cohort_size"] >= 20).astype(np.int8)

    for column in set(STATIC_CAT + COHORT_CAT):
        out[column] = out[column].fillna("UNKNOWN").astype(str)
    return out


class LogCatBoost:
    def __init__(self, features: list[str], cat_features: list[str], iterations: int) -> None:
        self.features = features
        self.cat_features = cat_features
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            depth=6,
            learning_rate=0.045,
            l2_leaf_reg=35.0,
            random_seed=SEED,
            random_strength=0.5,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    def fit(self, frame: pd.DataFrame) -> "LogCatBoost":
        self.model.fit(
            frame[self.features],
            np.log1p(frame["tuketim"].clip(lower=0).to_numpy()),
            cat_features=self.cat_features,
        )
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, self.model.predict(frame[self.features]))


class HierarchicalResidualPrior:
    """Facility-equal, support-shrunk correction to the fixed power baseline."""

    def __init__(self, alpha_scale: float = 4.0) -> None:
        self.alpha_scale = alpha_scale
        self.global_mean = 0.0
        self.tables: list[tuple[tuple[str, ...], float, pd.DataFrame]] = []

    def fit(self, frame: pd.DataFrame) -> "HierarchicalResidualPrior":
        work = frame.copy()
        work["residual"] = (
            np.log1p(work["tuketim"].clip(lower=0))
            - np.log1p(2.5 * work["guc"].clip(lower=0))
        )
        self.global_mean = float(
            work.groupby("tanim", sort=False)["residual"].mean().mean()
        )
        self.tables = []
        for keys, alpha in EB_LEVELS:
            profiles = (
                work.groupby(["tanim", *keys], observed=True, sort=False)["residual"]
                .mean()
                .reset_index()
            )
            table = profiles.groupby(list(keys), observed=True)["residual"].agg(["mean", "count"])
            self.tables.append((keys, alpha * self.alpha_scale, table))
        return self

    def predict_correction(self, frame: pd.DataFrame) -> np.ndarray:
        correction = np.full(len(frame), self.global_mean, dtype=float)
        for keys, alpha, table in self.tables:
            index = (
                pd.Index(frame[keys[0]])
                if len(keys) == 1
                else pd.MultiIndex.from_frame(frame[list(keys)])
            )
            means = table["mean"].reindex(index).to_numpy()
            counts = table["count"].reindex(index).to_numpy()
            valid = np.isfinite(means) & np.isfinite(counts)
            weight = np.zeros(len(frame), dtype=float)
            weight[valid] = counts[valid] / (counts[valid] + alpha)
            correction = (1.0 - weight) * correction + weight * np.nan_to_num(means)
        return correction

    def predict(self, frame: pd.DataFrame, shrink: float = 0.5) -> np.ndarray:
        base_log = np.log1p(2.5 * frame["guc"].clip(lower=0).to_numpy(dtype=float))
        pred_log = np.maximum(0.0, base_log + shrink * self.predict_correction(frame))
        return np.expm1(pred_log)


def _commissioned_rows(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    grace_end = panel["tarih"].min() + pd.Timedelta(days=DATA_START_GRACE_DAYS)
    mask = (
        (panel["first_seen"] > grace_end)
        & (panel["first_seen"] <= cutoff)
        & (panel["tarih"] <= cutoff)
    )
    result = panel.loc[mask].copy()
    if result.empty:
        raise ValueError(f"No commissioned rows exist before {cutoff.date()}")
    return result


def _log_blend(static_log: np.ndarray, cohort_log: np.ndarray, weight: float) -> np.ndarray:
    return np.expm1((1.0 - weight) * static_log + weight * cohort_log).clip(min=0.0)


def _score_mass(y: np.ndarray, pred: np.ndarray, mass: np.ndarray) -> float | None:
    return calculate_rmsle(y[mass], pred[mass]) if mass.any() else None


def validate(
    data_dir: Path,
    report_path: Path,
    alpha_scale: float,
    shrink_values: Iterable[float],
) -> dict:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["tarih"])
    panel = prepare_panel(train)
    fold_results: list[dict[str, object]] = []
    pooled: list[pd.DataFrame] = []

    for cutoff_text, end_text in ROLLING_FOLDS:
        cutoff = pd.Timestamp(cutoff_text)
        end = pd.Timestamp(end_text)
        population = panel.loc[panel["tarih"] <= cutoff].copy()
        validation = panel.loc[
            (panel["first_seen"] > cutoff)
            & (panel["tarih"] > cutoff)
            & (panel["tarih"] <= end)
        ].copy()
        LOGGER.info(
            "Fold %s -> %s | population=%s validation=%s (%s facilities)",
            cutoff.date(), end.date(), f"{len(population):,}",
            f"{len(validation):,}", f"{validation['tanim'].nunique():,}",
        )
        model = HierarchicalResidualPrior(alpha_scale=alpha_scale).fit(population)
        correction = model.predict_correction(validation)
        y = validation["tuketim"].to_numpy(dtype=float)
        base_log = np.log1p(2.5 * validation["guc"].to_numpy(dtype=float))
        power_pred = np.expm1(base_log)
        mass = validation["is_mass_cohort"].to_numpy(dtype=bool)
        shrink_scores = {
            f"s_{shrink:.2f}": calculate_rmsle(
                y, np.expm1(np.maximum(0.0, base_log + shrink * correction))
            )
            for shrink in shrink_values
        }
        gated_shrink = np.where(mass, 0.20, 0.50)
        gated_pred = np.expm1(np.maximum(0.0, base_log + gated_shrink * correction))
        metric: dict[str, object] = {
            "fold": f"{cutoff_text}_{end_text}",
            "rows": len(validation),
            "facilities": int(validation["tanim"].nunique()),
            "mass_rows": int(mass.sum()),
            "baseline_rmsle": calculate_rmsle(y, power_pred),
            "shrink_rmsle": shrink_scores,
            "mass_baseline_rmsle": _score_mass(y, power_pred, mass),
            "mass_shrink_rmsle": {
                f"s_{shrink:.2f}": _score_mass(
                    y,
                    np.expm1(np.maximum(0.0, base_log + shrink * correction)),
                    mass,
                )
                for shrink in shrink_values
            },
            "segment_gated_rmsle": calculate_rmsle(y, gated_pred),
        }
        fold_results.append(metric)
        pooled.append(pd.DataFrame({
            "fold": metric["fold"],
            "y": y,
            "power": power_pred,
            "base_log": base_log,
            "correction": correction,
            "mass": mass,
            "segment_gated": gated_pred,
        }))
        LOGGER.info("Fold scores: %s", json.dumps(metric, ensure_ascii=False))

    oof = pd.concat(pooled, ignore_index=True)
    y_all = oof["y"].to_numpy()
    pooled_scores = {"baseline": calculate_rmsle(y_all, oof["power"].to_numpy())}
    pooled_scores.update({
        f"shrink_s_{shrink:.2f}": calculate_rmsle(
            y_all,
            np.expm1(np.maximum(
                0.0,
                oof["base_log"].to_numpy() + shrink * oof["correction"].to_numpy(),
            )),
        )
        for shrink in shrink_values
    })
    pooled_scores["segment_gated_mass_0.20_nonmass_0.50"] = calculate_rmsle(
        y_all, oof["segment_gated"].to_numpy()
    )
    fold_worst = {
        f"s_{shrink:.2f}": max(
            float(item["shrink_rmsle"][f"s_{shrink:.2f}"]) for item in fold_results
        )
        for shrink in shrink_values
    }
    selection = min(
        shrink_values,
        key=lambda shrink: (
            pooled_scores[f"shrink_s_{shrink:.2f}"] + 0.15 * fold_worst[f"s_{shrink:.2f}"],
            abs(shrink - 0.5),
        ),
    )
    report = {
        "method": "strict rolling origin; all targets and model fits are cutoff-bounded",
        "model": "facility-equal hierarchical empirical-Bayes correction to 2.5*guc",
        "alpha_scale": alpha_scale,
        "folds": fold_results,
        "pooled_scores": pooled_scores,
        "worst_fold_scores": fold_worst,
        "selected_shrink": selection,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Pooled scores: %s", json.dumps(pooled_scores, indent=2))
    LOGGER.info("Selected shrink: %.2f", selection)
    return report


def _validate_submission(test: pd.DataFrame, base: pd.DataFrame) -> None:
    if list(base.columns) != ["id", "tuketim"]:
        raise ValueError(f"Unexpected submission columns: {list(base.columns)}")
    if len(test) != len(base) or not np.array_equal(test["id"].to_numpy(), base["id"].to_numpy()):
        raise ValueError("Base submission IDs/order do not match official test.csv")


def submit(
    data_dir: Path,
    base_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    alpha_scale: float,
    eb_shrink: float,
    mass_eb_shrink: float,
    candidate_weight: float,
    mass_candidate_weight: float,
    mega_cohort_threshold: int,
    mega_candidate_weight: float,
    high_power_threshold: float,
) -> dict:
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["tarih"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["tarih"])
    base = pd.read_csv(base_path)
    _validate_submission(test, base)
    train_panel = prepare_panel(train)
    cutoff = train_panel["tarih"].max()
    population = train_panel.loc[train_panel["tarih"] <= cutoff].copy()
    known = set(train_panel["tanim"].unique())
    cold_mask = ~test["tanim"].isin(known)
    cold_raw = test.loc[cold_mask].copy()
    cold_first_seen = cold_raw.groupby("tanim", sort=False)["tarih"].min()
    cold_panel = prepare_panel(cold_raw, first_seen=cold_first_seen)

    LOGGER.info(
        "Final fit | population=%s cold=%s (%s facilities)",
        f"{len(population):,}", f"{len(cold_panel):,}",
        f"{cold_panel['tanim'].nunique():,}",
    )
    model = HierarchicalResidualPrior(alpha_scale=alpha_scale).fit(population)
    correction = model.predict_correction(cold_panel)
    power_log = np.log1p(2.5 * cold_panel["guc"].to_numpy(dtype=float))
    mass = cold_panel["is_mass_cohort"].to_numpy(dtype=bool)
    shrink = np.where(mass, mass_eb_shrink, eb_shrink)
    candidate_log = np.maximum(0.0, power_log + shrink * correction)
    base_cold = base.loc[cold_mask, "tuketim"].to_numpy(dtype=float)
    blend_weight = np.where(mass, mass_candidate_weight, candidate_weight)
    mega = cold_panel["cohort_size"].to_numpy(dtype=float) >= mega_cohort_threshold
    blend_weight = np.where(mega, mega_candidate_weight, blend_weight)
    blend_weight = np.where(
        cold_panel["guc"].to_numpy(dtype=float) > high_power_threshold,
        0.0,
        blend_weight,
    )
    final_log = (
        (1.0 - blend_weight) * np.log1p(np.maximum(0.0, base_cold))
        + blend_weight * candidate_log
    )
    final_cold = np.expm1(final_log).clip(min=0.0)
    ceiling = 36.0 * (cold_panel["guc"].to_numpy(dtype=float) + 1.0)
    final_cold = np.minimum(final_cold, ceiling)

    output = base.copy()
    warm_before = output.loc[~cold_mask, "tuketim"].to_numpy(copy=True)
    output.loc[cold_mask, "tuketim"] = final_cold
    if not np.array_equal(output.loc[~cold_mask, "tuketim"].to_numpy(), warm_before):
        raise AssertionError("Warm predictions changed")
    if output["tuketim"].isna().any() or not np.isfinite(output["tuketim"]).all():
        raise AssertionError("Non-finite predictions detected")
    if (output["tuketim"] < 0).any():
        raise AssertionError("Negative predictions detected")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    diagnostics = cold_raw[["id", "tanim", "guc", "tarih", "lokasyon"]].copy()
    diagnostics["first_seen"] = cold_panel["first_seen"].to_numpy()
    diagnostics["age_days"] = cold_panel["age_days"].to_numpy()
    diagnostics["cohort_size"] = cold_panel["cohort_size"].to_numpy()
    diagnostics["is_mass_cohort"] = mass.astype(np.int8)
    diagnostics["base_v8"] = base_cold
    diagnostics["power_baseline"] = np.expm1(power_log)
    diagnostics["eb_log_correction"] = correction
    diagnostics["v23_candidate"] = np.expm1(candidate_log)
    diagnostics["candidate_blend_weight"] = blend_weight
    diagnostics["v23_final"] = final_cold
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_path, index=False)

    summary = {
        "rows": len(output),
        "cold_rows": int(cold_mask.sum()),
        "cold_transformers": int(cold_raw["tanim"].nunique()),
        "warm_rows_unchanged": int((~cold_mask).sum()),
        "eb_alpha_scale": alpha_scale,
        "eb_shrink": eb_shrink,
        "mass_eb_shrink": mass_eb_shrink,
        "candidate_weight_against_v8": candidate_weight,
        "mass_candidate_weight_against_v8": mass_candidate_weight,
        "mega_cohort_threshold": mega_cohort_threshold,
        "mega_candidate_weight_against_v8": mega_candidate_weight,
        "high_power_threshold_preserved_from_v8": high_power_threshold,
        "base_cold_median": float(np.median(base_cold)),
        "candidate_cold_median": float(np.median(np.expm1(candidate_log))),
        "final_cold_median": float(np.median(final_cold)),
        "output_sha256": sha256(output_path),
        "diagnostics_sha256": sha256(diagnostics_path),
    }
    LOGGER.info("Submission summary: %s", json.dumps(summary, indent=2))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "submit"])
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--eb-alpha-scale", type=float, default=4.0)
    parser.add_argument("--eb-shrink", type=float, default=0.5)
    parser.add_argument("--mass-eb-shrink", type=float, default=0.2)
    parser.add_argument("--report", type=Path, default=Path("v23_strict_validation.json"))
    parser.add_argument("--base-submission", type=Path, default=Path("submission_v8r_verified_final.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission_v23_strict_coldstart.csv"))
    parser.add_argument("--diagnostics", type=Path, default=Path("v23_cold_diagnostics.csv"))
    parser.add_argument("--candidate-weight", type=float, default=0.25)
    parser.add_argument("--mass-candidate-weight", type=float, default=0.10)
    parser.add_argument("--mega-cohort-threshold", type=int, default=500)
    parser.add_argument("--mega-candidate-weight", type=float, default=0.03)
    parser.add_argument("--high-power-threshold", type=float, default=2500.0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parser().parse_args()
    if args.mode == "validate":
        validate(
            args.data_dir,
            args.report,
            args.eb_alpha_scale,
            shrink_values=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        )
    else:
        weights = (
            args.eb_shrink,
            args.mass_eb_shrink,
            args.candidate_weight,
            args.mass_candidate_weight,
            args.mega_candidate_weight,
        )
        if not all(0.0 <= weight <= 1.0 for weight in weights):
            raise ValueError("Blend weights must be in [0, 1]")
        submit(
            args.data_dir,
            args.base_submission,
            args.output,
            args.diagnostics,
            args.eb_alpha_scale,
            args.eb_shrink,
            args.mass_eb_shrink,
            args.candidate_weight,
            args.mass_candidate_weight,
            args.mega_cohort_threshold,
            args.mega_candidate_weight,
            args.high_power_threshold,
        )


if __name__ == "__main__":
    main()
