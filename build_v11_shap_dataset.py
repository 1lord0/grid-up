"""SHAP odaklı, rolling-origin ve hedef sızıntısız V11 eğitim ve test veri seti oluşturucu.

Train satırları üç ardışık 4 aylık tahmin bloğundan oluşur:
- Fold A (2025-04-01 - 2025-07-31), cutoff: 2025-03-31
- Fold B (2025-08-01 - 2025-11-30), cutoff: 2025-07-31
- Fold C (2025-12-01 - 2026-03-31), cutoff: 2025-11-30
- Test (2026-04-01 - 2026-07-31), cutoff: 2026-03-31

Her bloktaki özellikler strictly t <= cutoff_date geçmişiyle üretilir.
"""

from __future__ import annotations

import gc
import gzip
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
PIPELINE_DIR = Path(
    r"C:\Users\EREN\.gemini\antigravity\brain\3d312eb6-9f38-4f58-a766-70db2435f5c2\grid_up_pipeline"
)
OUTPUT_DIR = DATA_DIR / "features_v11_shap"
SHAP_REPORT = PIPELINE_DIR / "output" / "shap_importance_report.csv"
SEGMENTED_SHAP_REPORT = PIPELINE_DIR / "output" / "segmented_shap_report.csv"

sys.path.insert(0, str(PIPELINE_DIR))

from src.cutoff_history_engine import extract_cutoff_facility_history  # noqa: E402
from src.hierarchical_priors import HierarchicalPriorEngine  # noqa: E402
from src.roster_features import compute_targetless_roster_features  # noqa: E402
from src.seasonal_baseline_engine import SeasonalBaselineEngine  # noqa: E402

FOLDS = [
    ("fold_a_apr_jul_2025", "2025-03-31", "2025-04-01", "2025-07-31"),
    ("fold_b_aug_nov_2025", "2025-07-31", "2025-08-01", "2025-11-30"),
    ("fold_c_dec_mar_2026", "2025-11-30", "2025-12-01", "2026-03-31"),
]

# Section 8 & 9 Canonical Feature Definition
CORE_SHAP_FEATURES = [
    "mean_all_log",
    "mean_90_log",
    "day_of_year",
    "ewm_28_log",
    "mean_14_log",
    "mean_all_raw",
    "guc",
    "horizon_days",
    "roster_active_power_sum_loc",
    "loc_month",
    "median_14_log",
    "roster_hhi_loc",
    "district_month_factor",
    "last_value_log",
    "roster_active_facilities_loc",
    "median_all_raw",
    "loc_dow",
    "mean_14_raw",
    "month",
    "p10_log",
    "mean_56_raw",
    "p95_log",
    "day",
    "roster_active_power_mean_loc",
    "std_all_raw",
    "guc_month",
    "median_all_log",
    "mean_7_log",
    "std_all_log",
    "roster_facility_power_share_loc",
    "median_7_log",
    "day_of_week",
    "week_of_year",
    "district_dow_factor",
]

SAFETY_FEATURES = [
    "hist_count",
    "hist_tier",
    "facility_age_days",
    "days_since_last_seen",
    "is_cold_start",
    "is_warm_start",
    "is_annual_start",
    "log_guc",
    "lokasyon",
    "il",
    "bolge",
    "ilce",
    "guc_grup",
    "guc_bin",
    "zero_category",
    "prior_mean_log",
    "prior_load_ratio",
    "prior_estimated_raw",
    "lag_364",
    "lag_365",
    "lag_371",
    "lag_median",
    "lag_mean",
    "lag_std",
    "lag_count",
    "has_annual_lag",
    "yoy_trend_multiplier",
    "seasonal_baseline",
    "baseline_to_guc_ratio",
]

DERIVED_FEATURES = [
    "district_month",
    "district_dow",
    "doy_sin",
    "doy_cos",
    "horizon_fraction",
    "is_summer",
    "is_june",
    "is_july",
    "log_guc_x_summer",
    "recent_90_vs_all_log",
    "recent_28_vs_90_log",
    "last_vs_90_log",
    "history_coverage",
    "log_roster_power_sum",
]

CATEGORICAL_FEATURES = {
    "lokasyon",
    "il",
    "ilce",
    "bolge",
    "guc_grup",
    "guc_bin",
    "hist_tier",
    "zero_category",
    "loc_month",
    "loc_dow",
    "guc_month",
    "district_month",
    "district_dow",
}


def add_location_hierarchy(df: pd.DataFrame) -> None:
    """Parses raw lokasyon into il, bolge, ilce while retaining raw lokasyon.
    
    Examples:
    - İZMİR>METROPOL>KARABAĞLAR -> il=İZMİR, bolge=METROPOL, ilce=KARABAĞLAR
    - MANİSA>TURGUTLU -> il=MANİSA, bolge=DOGRUDAN, ilce=TURGUTLU
    """
    parts = df["lokasyon"].astype(str).str.split(">")
    df["il"] = parts.str[0].fillna("BILINMIYOR").astype(str)
    df["ilce"] = parts.str[-1].fillna("BILINMIYOR").astype(str)
    df["bolge"] = parts.apply(
        lambda val: val[-2] if isinstance(val, list) and len(val) >= 3 else "DOGRUDAN"
    ).astype(str)


def load_raw_data_clean() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw train and test datasets with clean location hierarchy and power bins."""
    logger.info("Loading raw train.csv and test.csv...")
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"tanim": str}, encoding="utf-8")
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"tanim": str}, encoding="utf-8")

    train["tarih"] = pd.to_datetime(train["tarih"])
    test["tarih"] = pd.to_datetime(test["tarih"])

    for df in [train, test]:
        df["guc"] = df["guc"].astype(np.float32)
        df["year"] = df["tarih"].dt.year.astype(np.int32)
        df["month"] = df["tarih"].dt.month.astype(np.int32)
        df["day"] = df["tarih"].dt.day.astype(np.int32)
        df["day_of_week"] = df["tarih"].dt.dayofweek.astype(np.int32)
        df["day_of_year"] = df["tarih"].dt.dayofyear.astype(np.int32)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.float32)
        df["is_sunday"] = (df["day_of_week"] == 6).astype(np.float32)
        df["is_saturday"] = (df["day_of_week"] == 5).astype(np.float32)
        df["quarter"] = df["tarih"].dt.quarter.astype(np.int32)
        df["week_of_year"] = df["tarih"].dt.isocalendar().week.astype(np.int32)

        # Precise location parsing
        add_location_hierarchy(df)

        # Power bins
        df["guc_grup"] = pd.cut(
            df["guc"],
            bins=[-np.inf, 50, 160, 400, 1000, 2500, np.inf],
            labels=["Micro", "Small", "Medium", "Large", "VeryLarge", "Mega"],
        ).astype(str)
        df["guc_bin"] = df["guc_grup"].astype(str)

    return train, test


def get_canonical_feature_list() -> List[str]:
    """Combines and deduplicates canonical features preserving order."""
    combined = []
    seen = set()
    for feat in CORE_SHAP_FEATURES + SAFETY_FEATURES + DERIVED_FEATURES:
        if feat not in seen:
            seen.add(feat)
            combined.append(feat)
    return combined


def build_one_block(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    roster_df: pd.DataFrame,
    feature_cols: List[str],
    fold_name: str,
    is_test: bool = False,
) -> pd.DataFrame:
    """Builds one leakage-free block for train fold or test."""
    t0 = time.time()
    logger.info(f"Building block '{fold_name}' (Target N={len(target_df):,d}, Cutoff={cutoff_date.date()})...")

    target = target_df.copy()
    target["_source_order"] = np.arange(len(target), dtype=np.int64)

    # 1. Roster features
    loc_roster = roster_df[
        [
            "tarih",
            "lokasyon",
            "roster_active_facilities_loc",
            "roster_active_power_sum_loc",
            "roster_active_power_mean_loc",
            "roster_active_power_max_loc",
            "roster_hhi_loc",
        ]
    ].drop_duplicates(["tarih", "lokasyon"])
    merged = pd.merge(target, loc_roster, on=["tarih", "lokasyon"], how="left")
    merged["roster_facility_power_share_loc"] = (
        merged["guc"] / (merged["roster_active_power_sum_loc"] + 1.0)
    ).astype(np.float32)

    # 2. Cutoff history aggregates
    hist_feats = extract_cutoff_facility_history(history_df, cutoff_date)
    merged = pd.merge(merged, hist_feats, on="tanim", how="left")
    merged["hist_count"] = merged["hist_count"].fillna(0).astype(np.float32)
    merged["days_since_last_seen"] = merged["days_since_last_seen"].fillna(999.0).astype(np.float32)
    merged["facility_age_days"] = merged["facility_age_days"].fillna(0.0).astype(np.float32)
    merged["hist_tier"] = merged["hist_tier"].fillna("0_Cold").astype(str)
    merged["zero_category"] = merged["zero_category"].fillna("Rarely_Zero (<5%)").astype(str)

    # 3. Empirical Bayes Priors (Fitted on history_df with cutoff)
    prior_engine = HierarchicalPriorEngine()
    prior_engine.fit(history_df, cutoff_date)
    merged = prior_engine.transform(merged)

    # 4. Seasonal Baseline & YoY Lag features (Fitted on history_df with cutoff)
    baseline_engine = SeasonalBaselineEngine(shrinkage_alpha=10.0)
    baseline_engine.fit(history_df, cutoff_date)
    baseline_feats = baseline_engine.compute_baseline_features(merged)
    merged = pd.concat([merged, baseline_feats], axis=1)

    # 5. Horizon, Interactions & Derived features
    merged["horizon_days"] = (merged["tarih"] - cutoff_date).dt.days.astype(np.float32)
    merged["horizon_fraction"] = (merged["horizon_days"] / 122.0).astype(np.float32)
    merged["log_guc"] = np.log1p(merged["guc"]).astype(np.float32)

    merged["loc_month"] = merged["lokasyon"].astype(str) + "_" + merged["month"].astype(str)
    merged["loc_dow"] = merged["lokasyon"].astype(str) + "_" + merged["day_of_week"].astype(str)
    merged["guc_month"] = merged["guc_grup"].astype(str) + "_" + merged["month"].astype(str)
    merged["district_month"] = merged["ilce"].astype(str) + "_" + merged["month"].astype(str)
    merged["district_dow"] = merged["ilce"].astype(str) + "_" + merged["day_of_week"].astype(str)

    merged["baseline_to_guc_ratio"] = (
        merged["seasonal_baseline"] / np.maximum(1.0, merged["guc"])
    ).fillna(0.0).astype(np.float32)

    angle = 2.0 * np.pi * merged["day_of_year"].astype(float) / 365.25
    merged["doy_sin"] = np.sin(angle).astype(np.float32)
    merged["doy_cos"] = np.cos(angle).astype(np.float32)

    merged["is_summer"] = merged["month"].isin([6, 7, 8]).astype(np.int8)
    merged["is_june"] = (merged["month"] == 6).astype(np.int8)
    merged["is_july"] = (merged["month"] == 7).astype(np.int8)
    merged["log_guc_x_summer"] = (merged["log_guc"] * merged["is_summer"]).astype(np.float32)

    merged["recent_90_vs_all_log"] = (merged["mean_90_log"] - merged["mean_all_log"]).astype(np.float32)
    merged["recent_28_vs_90_log"] = (merged["ewm_28_log"] - merged["mean_90_log"]).astype(np.float32)
    merged["last_vs_90_log"] = (merged["last_value_log"] - merged["mean_90_log"]).astype(np.float32)
    merged["history_coverage"] = (
        merged["hist_count"] / np.maximum(merged["facility_age_days"] + 1.0, 1.0)
    ).astype(np.float32)
    merged["log_roster_power_sum"] = np.log1p(
        merged["roster_active_power_sum_loc"].clip(lower=0)
    ).astype(np.float32)

    # 6. Segment definition
    merged["is_cold_start"] = (merged["hist_count"] == 0).astype(np.int8)
    merged["is_annual_start"] = (merged["has_annual_lag"].fillna(0) > 0).astype(np.int8)
    merged["is_warm_start"] = (
        (merged["hist_count"] > 0) & (merged["has_annual_lag"].fillna(0) == 0)
    ).astype(np.int8)
    merged["segment"] = np.select(
        [merged["is_annual_start"] == 1, merged["is_warm_start"] == 1],
        ["annual", "warm"],
        default="cold",
    )

    # 7. Metadata and ordering
    merged["cutoff_date"] = cutoff_date.strftime("%Y-%m-%d")
    merged["fold_id"] = fold_name
    merged["row_id"] = merged["tanim"].astype(str) + "_" + merged["tarih"].dt.strftime("%Y-%m-%d")

    # Restore exact initial order
    merged = merged.sort_values("_source_order").reset_index(drop=True)

    if is_test and "id" in merged:
        if not merged["row_id"].equals(merged["id"].astype(str)):
            raise AssertionError("Test row_id ile verilen id sırası birebir eşleşmiyor!")

    # Verify all feature columns exist
    missing_cols = [c for c in feature_cols if c not in merged.columns]
    if missing_cols:
        raise KeyError(f"Üretilemeyen kolonlar var: {missing_cols}")

    # Standardize categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in merged.columns:
            merged[col] = merged[col].fillna("__MISSING__").astype(str)

    metadata_cols = ["row_id", "tanim", "tarih", "cutoff_date", "fold_id", "segment"]
    if not is_test:
        metadata_cols.append("tuketim")

    out_df = merged[metadata_cols + feature_cols].copy()
    out_df["tarih"] = pd.to_datetime(out_df["tarih"]).dt.strftime("%Y-%m-%d")

    # Strict validations per block
    numeric_cols = out_df.select_dtypes(include=[np.number]).columns
    if np.isinf(out_df[numeric_cols].to_numpy(dtype=float, copy=False)).any():
        raise ValueError(f"[{fold_name}] Sonsuz (inf) sayısal değer tespit edildi!")
    if not is_test and (out_df["tuketim"] < 0).any():
        raise ValueError(f"[{fold_name}] Negatif hedef (tuketim < 0) tespit edildi!")
    if (pd.to_datetime(out_df["tarih"]) <= cutoff_date).any():
        raise AssertionError(f"[{fold_name}] Hedef tarihi cutoff tarihinden büyük değil!")
    if out_df["row_id"].duplicated().any():
        raise ValueError(f"[{fold_name}] Yinelenen row_id tespit edildi!")

    elapsed = time.time() - t0
    logger.info(f"✓ Block '{fold_name}' completed in {elapsed:.1f}s (N={len(out_df):,d}, Cols={len(out_df.columns)}).")
    return out_df


def build_manifest_table(
    feature_cols: List[str], sample_df: pd.DataFrame
) -> pd.DataFrame:
    """Builds feature manifest table with descriptions, roles, and types."""
    shap_df = pd.read_csv(SHAP_REPORT) if SHAP_REPORT.exists() else pd.DataFrame()
    segmented_df = pd.read_csv(SEGMENTED_SHAP_REPORT) if SEGMENTED_SHAP_REPORT.exists() else pd.DataFrame()

    shap_map = shap_df.set_index("feature").to_dict("index") if not shap_df.empty else {}
    seg_map = segmented_df.set_index("feature").to_dict("index") if not segmented_df.empty else {}

    rows = []
    for idx, feat in enumerate(feature_cols, start=1):
        global_info = shap_map.get(feat, {})
        seg_info = seg_map.get(feat, {})

        if feat in DERIVED_FEATURES:
            role = "derived_shap_interaction"
            rationale = "SHAP çekirdek değişkenlerinden sızıntısız doğrusal/trigonometrik etkileşim"
        elif feat in SAFETY_FEATURES:
            role = "segment_safety_and_priors"
            rationale = "Cold/Warm/Annual uzman yönlendirmesi, lokasyon hiyerarşisi veya güvenli öncül"
        else:
            role = "shap_core"
            rationale = "SHAP önem analizinde sıfırdan büyük katkı alan çekirdek değişken"

        rows.append(
            {
                "col_index": idx,
                "feature": feat,
                "role": role,
                "dtype": str(sample_df[feat].dtype),
                "is_categorical": feat in CATEGORICAL_FEATURES,
                "global_shap_rank": int(shap_df.index[shap_df["feature"] == feat][0] + 1) if feat in shap_map else None,
                "global_shap_pct": global_info.get("shap_pct"),
                "cold_shap_pct": seg_info.get("pct_share_Cold_Start"),
                "warm_shap_pct": seg_info.get("pct_share_Warm_Start"),
                "rationale": rationale,
            }
        )

    return pd.DataFrame(rows)


def main():
    logger.info("=" * 70)
    logger.info("STARTING V11 LEAKAGE-FREE DATASET GENERATION PIPELINE")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical_features = get_canonical_feature_list()
    logger.info(f"Canonical feature count: {len(canonical_features)}")

    train_raw, test_raw = load_raw_data_clean()

    # Pre-compute Targetless Roster on entire schedule
    logger.info("Computing targetless active roster schedule across all dates...")
    all_roster_input = pd.concat(
        [
            train_raw[["tarih", "lokasyon", "guc", "tanim", "ilce", "il"]],
            test_raw[["tarih", "lokasyon", "guc", "tanim", "ilce", "il"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["tarih", "tanim"])
    roster_df = compute_targetless_roster_features(all_roster_input)
    logger.info(f"Roster computed: {len(roster_df):,d} facility-date records.")

    # -------------------------------------------------------------------------
    # 1. BUILD TRAIN DATASET (3 ROLLING-ORIGIN FOLDS) TO PARTIAL FILE FIRST
    # -------------------------------------------------------------------------
    train_partial = OUTPUT_DIR / "train_features_v11.partial.csv.gz"
    train_final = OUTPUT_DIR / "train_features_v11.csv.gz"

    if train_partial.exists():
        train_partial.unlink()

    total_train_rows = 0
    fold_summaries = []
    sample_df: pd.DataFrame | None = None

    with gzip.open(train_partial, "wt", encoding="utf-8", newline="", compresslevel=6) as gz_out:
        for fold_idx, (name, cutoff_str, start_str, end_str) in enumerate(FOLDS):
            cutoff = pd.Timestamp(cutoff_str)
            hist = train_raw[train_raw["tarih"] <= cutoff].copy()
            target = train_raw[
                (train_raw["tarih"] >= pd.Timestamp(start_str))
                & (train_raw["tarih"] <= pd.Timestamp(end_str))
            ].copy()

            block = build_one_block(
                target,
                hist,
                cutoff,
                roster_df,
                canonical_features,
                name,
                is_test=False,
            )

            # Write header only for first fold
            block.to_csv(gz_out, index=False, header=fold_idx == 0)
            total_train_rows += len(block)

            summary = {
                "fold_id": name,
                "cutoff_date": cutoff_str,
                "target_start": start_str,
                "target_end": end_str,
                "total_rows": len(block),
                "annual_rows": int((block["segment"] == "annual").sum()),
                "warm_rows": int((block["segment"] == "warm").sum()),
                "cold_rows": int((block["segment"] == "cold").sum()),
                "mean_consumption": float(block["tuketim"].mean()),
                "median_consumption": float(block["tuketim"].median()),
            }
            fold_summaries.append(summary)
            logger.info(f"Fold Summary: {summary}")

            if sample_df is None:
                sample_df = block.head(1000).copy()

            del hist, target, block
            gc.collect()

    logger.info(f"All 3 train folds written to {train_partial} (Total rows: {total_train_rows:,d}).")

    # -------------------------------------------------------------------------
    # 2. BUILD TEST DATASET TO PARTIAL FILE FIRST
    # -------------------------------------------------------------------------
    test_partial = OUTPUT_DIR / "test_features_v11.partial.csv.gz"
    test_final = OUTPUT_DIR / "test_features_v11.csv.gz"

    if test_partial.exists():
        test_partial.unlink()

    full_cutoff = pd.Timestamp("2026-03-31")
    test_block = build_one_block(
        test_raw,
        train_raw,
        full_cutoff,
        roster_df,
        canonical_features,
        "test_apr_jul_2026",
        is_test=True,
    )

    test_block.to_csv(
        test_partial,
        index=False,
        compression={"method": "gzip", "compresslevel": 6},
    )
    logger.info(f"Test block written to {test_partial} (Total rows: {len(test_block):,d}).")

    # -------------------------------------------------------------------------
    # 3. STRICT QUALITY & INTEGRITY ASSERTIONS
    # -------------------------------------------------------------------------
    logger.info("Executing mandatory quality & integrity assertions...")

    # Test row count
    if len(test_block) != 714688:
        raise AssertionError(f"Test satır sayısı 714688 olmalı, fakat {len(test_block)} bulundu!")
    if not test_block["row_id"].equals(test_raw["id"].astype(str)):
        raise AssertionError("Test row_id sıralaması ham test.csv id ile birebir eşleşmiyor!")

    test_cold_count = int((test_block["segment"] == "cold").sum())
    test_cold_facilities = int(test_block.loc[test_block["segment"] == "cold", "tanim"].nunique())
    logger.info(f"Test cold rows: {test_cold_count:,d} (Expected: 158,369)")
    logger.info(f"Test cold facilities: {test_cold_facilities:,d} (Expected: 2,024)")

    if test_cold_count != 158369:
        raise AssertionError(f"Test cold satır sayısı 158369 olmalı, fakat {test_cold_count} bulundu!")
    if test_cold_facilities != 2024:
        raise AssertionError(f"Test cold tesis sayısı 2024 olmalı, fakat {test_cold_facilities} bulundu!")

    # Column alignment check
    train_meta_cols = ["row_id", "tanim", "tarih", "cutoff_date", "fold_id", "segment", "tuketim"]
    test_meta_cols = ["row_id", "tanim", "tarih", "cutoff_date", "fold_id", "segment"]

    train_feature_cols = [c for c in sample_df.columns if c not in train_meta_cols]
    test_feature_cols = [c for c in test_block.columns if c not in test_meta_cols]

    if train_feature_cols != test_feature_cols:
        raise AssertionError("Train ve Test feature kolonları veya sıralamaları birebir aynı değil!")

    # -------------------------------------------------------------------------
    # 4. ATOMIC PROMOTION TO FINAL GZ FILES
    # -------------------------------------------------------------------------
    logger.info("Promoting partial files atomically to final .csv.gz files...")
    if train_final.exists():
        train_final.unlink()
    if test_final.exists():
        test_final.unlink()

    train_partial.rename(train_final)
    test_partial.rename(test_final)
    logger.info(f"✓ Atomically promoted train -> {train_final}")
    logger.info(f"✓ Atomically promoted test  -> {test_final}")

    # -------------------------------------------------------------------------
    # 5. FEATURE MANIFEST & DATASET PROFILE
    # -------------------------------------------------------------------------
    manifest_df = build_manifest_table(canonical_features, sample_df)
    manifest_path = OUTPUT_DIR / "feature_manifest_v11.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"✓ Feature manifest saved to {manifest_path}")

    profile = {
        "dataset_version": "V11_SHAP_Leakage_Free",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_train_rows": total_train_rows,
        "total_test_rows": len(test_block),
        "total_features": len(canonical_features),
        "train_columns_count": len(sample_df.columns),
        "test_columns_count": len(test_block.columns),
        "target_leakage_rule": "Every target date is strictly > cutoff_date. History features computed strictly on t <= cutoff_date.",
        "location_parsing_rule": "il = parts[0], ilce = parts[-1], bolge = parts[-2] if len >= 3 else DOGRUDAN",
        "kmeans_included": False,
        "folds": fold_summaries,
        "test_breakdown": {
            "total_rows": len(test_block),
            "cold_rows": test_cold_count,
            "cold_facilities": test_cold_facilities,
            "annual_rows": int((test_block["segment"] == "annual").sum()),
            "warm_rows": int((test_block["segment"] == "warm").sum()),
            "date_min": str(test_block["tarih"].min()),
            "date_max": str(test_block["tarih"].max()),
        },
        "files": {
            "train": str(train_final),
            "test": str(test_final),
            "manifest": str(manifest_path),
        },
    }

    profile_path = OUTPUT_DIR / "dataset_profile_v11.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Dataset profile saved to {profile_path}")

    logger.info("=" * 70)
    logger.info("🎉 V11 LEAKAGE-FREE DATASET SUCCESSFULLY CREATED & VERIFIED! 🎉")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

