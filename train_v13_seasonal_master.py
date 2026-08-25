"""Grid Up Datathon — V13 Seasonal Master Pipeline.

Yaz Mevsimselliği ve Aylık Tüketim Çarpanları Mimarisi:
1. Monthly Network Seasonality Index (Nisan: 0.65, Mayıs: 0.56, Haziran: 1.00, Temmuz: 1.70)
2. Facility-Level Summer Surge Ratio (Temmuz 2025 / Kış 2025)
3. Hierarchical District x Capacity Monthly Multipliers
4. Seasonal Baseline Rescaling + Learned CatBoost/LightGBM Seasonal Residuals
5. Full 3-Fold Rolling Validation (Fold A, Fold B, Fold C)
6. Submission Generation: submission_v13_seasonal_master.csv
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
OUTPUT_SUB_PATH = DATA_DIR / "submission_v13_seasonal_master.csv"
OUTPUT_DIR = DATA_DIR / "features_v11_shap" / "v13_seasonal_master_results"


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.clip(y_true, 0, None)
    y_p = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_p) - np.log1p(y_t)) ** 2)))


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts = df["lokasyon"].astype(str).str.split(">")
    df["il"] = parts.str[0]
    df["ilce"] = parts.str[-1]
    df["bolge"] = parts.apply(lambda p: p[-2] if len(p) >= 3 else "DOGRUDAN")
    return df


def build_seasonal_features(train_df: pd.DataFrame, target_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sızıntısız şekilde cutoff_date öncesindeki verilerden mevsimsel katsayıları üretir."""
    # 1. Past data up to cutoff
    past_df = train_df[train_df["tarih"] <= cutoff_date].copy()

    # 2. Monthly Network Index (relative to mean month)
    m_totals = past_df.groupby(past_df["tarih"].dt.month)["tuketim"].sum()
    m_avg = m_totals.mean() if len(m_totals) > 0 else 1.0
    m_index = (m_totals / m_avg).to_dict()

    # Default fallback for missing months
    default_m_index = {1: 0.794, 2: 0.782, 3: 0.694, 4: 0.651, 5: 0.564, 6: 1.000, 7: 1.700, 8: 1.442, 9: 1.174, 10: 0.734, 11: 1.068, 12: 1.400}
    for m in range(1, 13):
        if m not in m_index:
            m_index[m] = default_m_index[m]

    # 3. Facility-Level Winter Baseline & Summer Surge
    past_2025 = past_df[past_df["tarih"].dt.year == 2025]
    fac_winter = past_2025[past_2025["tarih"].dt.month.isin([1, 2, 3])].groupby("tanim")["tuketim"].mean()
    fac_july = past_2025[past_2025["tarih"].dt.month == 7].groupby("tanim")["tuketim"].mean()
    fac_summer_surge = (fac_july / np.maximum(10.0, fac_winter)).clip(0.2, 8.0).to_dict()

    # Hierarchical Fallbacks for Summer Surge
    surge_df = past_df[["tanim", "ilce", "guc_bin", "il"]].drop_duplicates("tanim").set_index("tanim")
    surge_df["surge"] = surge_df.index.map(fac_summer_surge)
    global_surge = float(surge_df["surge"].dropna().median()) if len(surge_df["surge"].dropna()) > 0 else 1.40
    ilce_surge = surge_df.groupby("ilce")["surge"].median().to_dict()
    guc_surge = surge_df.groupby("guc_bin")["surge"].median().to_dict()

    # 4. Facility All-Time & Recent Level
    fac_recent_28 = past_df[past_df["tarih"] > (cutoff_date - pd.Timedelta(days=28))].groupby("tanim")["tuketim"].mean().to_dict()
    fac_mean_all = past_df.groupby("tanim")["tuketim"].mean().to_dict()
    fac_last_seen = past_df.groupby("tanim")["tarih"].max().to_dict()

    # 5. Exact 1-Year Lags
    lag_364_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=364)])["tuketim"].to_dict()
    lag_365_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=365)])["tuketim"].to_dict()
    lag_371_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=371)])["tuketim"].to_dict()

    def transform_df(df_target: pd.DataFrame) -> pd.DataFrame:
        df = df_target.copy()
        df["month"] = df["tarih"].dt.month
        df["day_of_week"] = df["tarih"].dt.dayofweek
        df["day_of_year"] = df["tarih"].dt.dayofyear
        df["day"] = df["tarih"].dt.day
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        df["is_june_july"] = df["month"].isin([6, 7]).astype(int)
        df["log_guc"] = np.log1p(np.maximum(1.0, df["guc"])).astype(np.float32)
        df["log_guc_x_summer"] = (df["log_guc"] * df["is_summer"]).astype(np.float32)

        # Feature: Monthly Network Index
        df["monthly_network_index"] = df["month"].map(m_index).fillna(1.0).astype(np.float32)

        # Fast Vectorized Surge
        direct_surge = df["tanim"].map(fac_summer_surge)
        fallback_ilce = df["ilce"].map(ilce_surge)
        fallback_guc = df["guc_bin"].map(guc_surge)
        df["facility_summer_surge"] = direct_surge.fillna(fallback_ilce).fillna(fallback_guc).fillna(global_surge).astype(np.float32)

        # Baseline Recency Level
        df["fac_recent_28"] = df["tanim"].map(fac_recent_28)
        df["fac_mean_all"] = df["tanim"].map(fac_mean_all)
        df["fac_level"] = df["fac_recent_28"].fillna(df["fac_mean_all"]).fillna(df["guc"] * 2.5).astype(np.float32)
        df["log_fac_level"] = np.log1p(df["fac_level"]).astype(np.float32)

        # Fast Vectorized Seasonal Scaled Baseline:
        m_arr = df["month"].values
        base_arr = df["fac_level"].values
        surge_arr = df["facility_summer_surge"].values

        conds = [
            m_arr == 4,
            m_arr == 5,
            m_arr == 6,
            m_arr == 7,
            m_arr == 8,
            m_arr == 9,
        ]
        choices = [
            base_arr * 0.85,
            base_arr * 0.70,
            base_arr * (0.90 + 0.35 * (surge_arr - 1.0)),
            base_arr * (1.10 + 0.85 * (surge_arr - 1.0)),
            base_arr * (1.05 + 0.70 * (surge_arr - 1.0)),
            base_arr * 0.95,
        ]
        df["seasonal_baseline"] = np.select(conds, choices, default=base_arr).astype(np.float32)
        df["log_seasonal_baseline"] = np.log1p(df["seasonal_baseline"]).astype(np.float32)

        # 1-Year Lags (fast tuple lookup)
        tanim_arr = df["tanim"].values
        tarih_arr = df["tarih"].values
        keys = list(zip(tanim_arr, tarih_arr))

        df["lag_364"] = [lag_364_map.get(k, np.nan) for k in keys]
        df["lag_365"] = [lag_365_map.get(k, np.nan) for k in keys]
        df["lag_371"] = [lag_371_map.get(k, np.nan) for k in keys]
        df["has_annual_lag"] = (~df["lag_365"].isna()).astype(int)
        df["annual_lag_val"] = df["lag_365"].fillna(df["lag_364"]).fillna(df["lag_371"]).fillna(df["seasonal_baseline"]).astype(np.float32)
        df["log_annual_lag"] = np.log1p(df["annual_lag_val"]).astype(np.float32)
        # Days since last seen
        df["last_seen_date"] = df["tanim"].map(fac_last_seen)
        df["days_since_last_seen"] = (cutoff_date - df["last_seen_date"]).dt.days.fillna(999).astype(np.float32)
        df["is_cold"] = (df["days_since_last_seen"] > 180).astype(int)

        # ID column
        if "id" not in df.columns:
            df["id"] = df["tanim"] + "_" + df["tarih"].dt.strftime("%Y-%m-%d")

        return df

    target_out = transform_df(target_df)
    past_out = transform_df(past_df)
    return past_out, target_out


def run_seasonal_master():
    logger.info("=" * 75)
    logger.info("STARTING V13 SEASONAL MASTER (SUMMER SURGE & MONTHLY PROFILE)")
    logger.info("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading raw train.csv and test.csv...")
    raw_train = pd.read_csv(DATA_DIR / "train.csv", dtype={"tanim": str, "row_id": str}, parse_dates=["tarih"])
    raw_test = pd.read_csv(DATA_DIR / "test.csv", dtype={"tanim": str, "row_id": str}, parse_dates=["tarih"])

    raw_train = parse_locations(raw_train)
    raw_test = parse_locations(raw_test)

    guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
    guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
    raw_train["guc_bin"] = pd.cut(raw_train["guc"], bins=guc_bins, labels=guc_labels).astype(str)
    raw_test["guc_bin"] = pd.cut(raw_test["guc"], bins=guc_bins, labels=guc_labels).astype(str)

    # -------------------------------------------------------------------------
    # 2. 3-FOLD ROLLING-ORIGIN VALIDATION (Fold A, Fold B, Fold C)
    # -------------------------------------------------------------------------
    folds = [
        ("fold_a_apr_jul_2025", pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
        ("fold_b_aug_nov_2025", pd.Timestamp("2025-07-31"), pd.Timestamp("2025-08-01"), pd.Timestamp("2025-11-30")),
        ("fold_c_dec_mar_2026", pd.Timestamp("2025-11-30"), pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-31")),
    ]

    oof_preds_list = []
    features_model = [
        "month", "day_of_week", "day_of_year", "day", "doy_sin", "doy_cos",
        "is_weekend", "is_summer", "is_june_july",
        "guc", "log_guc", "log_guc_x_summer",
        "monthly_network_index", "facility_summer_surge",
        "fac_level", "log_fac_level", "seasonal_baseline", "log_seasonal_baseline",
        "has_annual_lag", "annual_lag_val", "log_annual_lag", "days_since_last_seen", "is_cold",
        "il", "ilce", "bolge", "guc_bin"
    ]
    cat_cols = ["il", "ilce", "bolge", "guc_bin"]

    logger.info("\n--- EXECUTING ROLLING ORIGIN CROSS VALIDATION ---")
    for fold_name, cutoff, val_start, val_end in folds:
        logger.info(f"\nProcessing Fold: {fold_name} (Cutoff: {cutoff.strftime('%Y-%m-%d')})")
        val_raw = raw_train[(raw_train['tarih'] >= val_start) & (raw_train['tarih'] <= val_end)].copy()
        past_tr_feat, val_data = build_seasonal_features(raw_train, val_raw, cutoff)

        # Ensure types
        for c in cat_cols:
            past_tr_feat[c] = past_tr_feat[c].astype("category")
            val_data[c] = val_data[c].astype("category")

        # Train Target: log1p(tuketim) - log1p(seasonal_baseline)
        y_train_res = np.log1p(past_tr_feat["tuketim"].values) - np.log1p(past_tr_feat["seasonal_baseline"].values)

        # LightGBM Seasonal Residual Model
        model_lgb = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        model_lgb.fit(past_tr_feat[features_model], y_train_res)

        # Predict Residual on Val
        pred_res = model_lgb.predict(val_data[features_model])

        # Final prediction: expm1(log1p(seasonal_baseline) + residual)
        pred_val = np.maximum(0.0, np.expm1(np.log1p(val_data["seasonal_baseline"].values) + pred_res))

        # Physical ceiling guardrail: 36 * (guc + 1)
        pred_val = np.clip(pred_val, 0.0, 36.0 * (val_data["guc"].values + 1.0))

        val_data["pred_v13"] = pred_val
        val_rmsle = calculate_rmsle(val_data["tuketim"].values, pred_val)
        logger.info(f"✓ {fold_name} Total RMSLE: {val_rmsle:.5f} (N={len(val_data):,d})")

        # Print Month breakdown
        for m in sorted(val_data["month"].unique()):
            v_m = val_data[val_data["month"] == m]
            r_m = calculate_rmsle(v_m["tuketim"].values, v_m["pred_v13"].values)
            logger.info(f"   -> Month {m:2d}: RMSLE = {r_m:.5f} (N={len(v_m):,d})")

        oof_preds_list.append(val_data[["id", "tarih", "tanim", "tuketim", "pred_v13", "month", "has_annual_lag", "is_cold"]])

    oof_df = pd.concat(oof_preds_list, ignore_index=True)
    pooled_rmsle = calculate_rmsle(oof_df["tuketim"].values, oof_df["pred_v13"].values)

    logger.info("\n" + "=" * 75)
    logger.info(f"★ POOLED 3-FOLD OOF RMSLE (V13 SEASONAL MASTER): {pooled_rmsle:.5f}")
    logger.info("=" * 75)

    # Segment breakdowns
    ann_mask = oof_df["has_annual_lag"] == 1
    cold_mask = oof_df["is_cold"] == 1
    warm_mask = (~ann_mask) & (~cold_mask)

    logger.info(f" - Annual Segment RMSLE : {calculate_rmsle(oof_df.loc[ann_mask, 'tuketim'].values, oof_df.loc[ann_mask, 'pred_v13'].values):.5f} (N={ann_mask.sum():,d})")
    logger.info(f" - Warm Segment RMSLE   : {calculate_rmsle(oof_df.loc[warm_mask, 'tuketim'].values, oof_df.loc[warm_mask, 'pred_v13'].values):.5f} (N={warm_mask.sum():,d})")
    logger.info(f" - Cold Segment RMSLE   : {calculate_rmsle(oof_df.loc[cold_mask, 'tuketim'].values, oof_df.loc[cold_mask, 'pred_v13'].values):.5f} (N={cold_mask.sum():,d})")

    # -------------------------------------------------------------------------
    # 3. FULL RETRAINING ON 100% DATA & SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    logger.info("\nRetraining on 100% data (Cutoff = 2026-03-31) for Test Submission...")
    cutoff_full = pd.Timestamp("2026-03-31")
    full_tr_feat, test_feat = build_seasonal_features(raw_train, raw_test, cutoff_full)

    for c in cat_cols:
        full_tr_feat[c] = full_tr_feat[c].astype("category")
        test_feat[c] = test_feat[c].astype("category")

    y_full_res = np.log1p(full_tr_feat["tuketim"].values) - np.log1p(full_tr_feat["seasonal_baseline"].values)

    full_lgb = lgb.LGBMRegressor(
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    full_lgb.fit(full_tr_feat[features_model], y_full_res)

    test_res_pred = full_lgb.predict(test_feat[features_model])
    test_final_pred = np.maximum(0.0, np.expm1(np.log1p(test_feat["seasonal_baseline"].values) + test_res_pred))
    test_final_pred = np.clip(test_final_pred, 0.0, 36.0 * (test_feat["guc"].values + 1.0))

    sub = pd.DataFrame({"id": test_feat["id"], "tuketim": test_final_pred})
    sub.to_csv(OUTPUT_SUB_PATH, index=False)
    sha256_hash = get_sha256(OUTPUT_SUB_PATH)

    logger.info("=" * 75)
    logger.info("✓ V13 SEASONAL MASTER SUBMISSION READY!")
    logger.info(f"✓ Output Path: {OUTPUT_SUB_PATH}")
    logger.info(f"✓ File Size  : {OUTPUT_SUB_PATH.stat().st_size:,d} bytes")
    logger.info(f"✓ SHA256     : {sha256_hash}")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_seasonal_master()
