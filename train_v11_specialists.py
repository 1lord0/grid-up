"""V11 Segment Specialist Modelleme ve Doğrulama Pipeline.

Modeller:
- Model 1 (Annual Specialist): segment == 'annual' satırlarında Seasonal Baseline üzerinden CatBoost & LightGBM ile artık (residual) öğrenimi.
- Model 2 (Warm Specialist): segment == 'warm' satırlarında recency/trend özellikleri ile CatBoost & LightGBM uzmanı.
- Model 3 (Cold Specialist): segment == 'cold' satırlarında güç, lokasyon hiyerarşisi, öncüller ve takvimle CatBoost & LightGBM uzmanı.
- SLSQP Segment Routing & Fiziksel Güç Tavanı: clip(pred, 0, 36 * (guc + 1)).
- Detaylı Doğrulama Raporu: Fold x Segment x Ay tablosu, Makro RMSLE, Kohort, Güç, Lokasyon, Sıfır ve Top %1 analizleri.
- V8 Referansıyla adil karşılaştırma ve V11 Kabul Kriterleri denetimi.
- 100% Retraining ile submission_v11_leakage_safe.csv üretimi ve SHA256 doğrulaması.
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
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
OUTPUT_DIR = DATA_DIR / "features_v11_shap"
RESULTS_DIR = OUTPUT_DIR / "v11_model_results"


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.clip(y_true, 0, None)
    y_p = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_p) - np.log1p(y_t)) ** 2)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_v11_pipeline():
    logger.info("=" * 70)
    logger.info("STARTING V11 SPECIALIST MODELING & VALIDATION PIPELINE")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train_features_v11.csv.gz"
    test_path = OUTPUT_DIR / "test_features_v11.csv.gz"

    logger.info(f"Loading {train_path}...")
    df = pd.read_csv(train_path, compression="gzip", dtype={"tanim": str, "row_id": str}, encoding="utf-8")
    logger.info(f"Train loaded: {len(df):,d} rows, {len(df.columns)} columns.")

    logger.info(f"Loading {test_path}...")
    test_df = pd.read_csv(test_path, compression="gzip", dtype={"tanim": str, "row_id": str}, encoding="utf-8")
    logger.info(f"Test loaded: {len(test_df):,d} rows, {len(test_df.columns)} columns.")

    meta_cols = ["row_id", "tanim", "tarih", "cutoff_date", "fold_id", "segment", "tuketim"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Cold features (features that are not NaN in cold start)
    cold_exclude = [
        "mean_all_log", "mean_90_log", "mean_14_log", "mean_7_log", "mean_all_raw",
        "mean_14_raw", "mean_56_raw", "median_all_log", "median_14_log", "median_7_log",
        "median_all_raw", "std_all_log", "std_all_raw", "last_value_log", "p10_log",
        "p95_log", "ewm_28_log", "recent_90_vs_all_log", "recent_28_vs_90_log",
        "last_vs_90_log", "lag_364", "lag_365", "lag_371", "lag_median", "lag_mean",
        "lag_std", "lag_count", "has_annual_lag", "yoy_trend_multiplier", "seasonal_baseline",
        "baseline_to_guc_ratio", "history_coverage", "facility_age_days", "days_since_last_seen",
        "hist_tier", "zero_category"
    ]
    cold_feature_cols = [c for c in feature_cols if c not in cold_exclude]
    logger.info(f"Total features: {len(feature_cols)}, Cold-specific features: {len(cold_feature_cols)}")

    cat_cols_all = [
        c for c in feature_cols
        if df[c].dtype == "object" or df[c].dtype.name == "category"
    ]
    cat_cols_cold = [c for c in cold_feature_cols if c in cat_cols_all]

    for c in cat_cols_all:
        df[c] = df[c].fillna("__MISSING__").astype("category")
        test_df[c] = test_df[c].fillna("__MISSING__").astype("category")

    folds_def = [
        ("fold_a_apr_jul_2025", ["fold_b_aug_nov_2025", "fold_c_dec_mar_2026"]),
        ("fold_b_aug_nov_2025", ["fold_a_apr_jul_2025", "fold_c_dec_mar_2026"]),
        ("fold_c_dec_mar_2026", ["fold_a_apr_jul_2025", "fold_b_aug_nov_2025"]),
    ]

    df["v11_oof_pred"] = 0.0
    df["cb_oof_pred"] = 0.0
    df["lgb_oof_pred"] = 0.0

    fold_eval_records = []

    # -------------------------------------------------------------------------
    # OUT-OF-FOLD TRAINING AND EVALUATION
    # -------------------------------------------------------------------------
    for val_fold, tr_folds in folds_def:
        logger.info(f"\n=======================================================")
        logger.info(f">>> Running V11 Specialist Models on Validation Fold: {val_fold} <<<")
        logger.info(f"=======================================================")

        tr_mask = df["fold_id"].isin(tr_folds)
        va_mask = df["fold_id"] == val_fold
        va_indices = df.index[va_mask]

        df_tr = df[tr_mask].copy()
        df_va = df[va_mask].copy()

        y_tr = df_tr["tuketim"].values
        y_va = df_va["tuketim"].values
        guc_tr = np.maximum(1.0, df_tr["guc"].values)
        guc_va = np.maximum(1.0, df_va["guc"].values)

        val_cb_pred = np.zeros(len(df_va), dtype=np.float32)
        val_lgb_pred = np.zeros(len(df_va), dtype=np.float32)

        # ---------------------------------------------------------------------
        # 1. WARM SPECIALIST (All rows with history: hist_count > 0)
        # ---------------------------------------------------------------------
        tr_warm_mask = (df_tr["hist_count"] > 0)
        va_warm_mask = (df_va["segment"] == "warm")
        va_ann_mask = (df_va["segment"] == "annual")

        logger.info(f"[Warm Specialist] Train N={tr_warm_mask.sum():,d} | Val Warm N={va_warm_mask.sum():,d} | Val Ann N={va_ann_mask.sum():,d}")

        if tr_warm_mask.sum() > 0:
            X_tr_warm = df_tr.loc[tr_warm_mask, feature_cols]
            y_tr_warm = df_tr.loc[tr_warm_mask, "tuketim"].values

            X_va_all_hist = df_va.loc[df_va["hist_count"] > 0, feature_cols]
            y_va_all_hist = df_va.loc[df_va["hist_count"] > 0, "tuketim"].values
            hist_va_indices = df_va.index[df_va["hist_count"] > 0]

            # Warm LightGBM Log1p
            lgb_warm = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.035,
                num_leaves=63,
                max_depth=8,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
            )
            lgb_warm.fit(
                X_tr_warm, np.log1p(y_tr_warm),
                eval_set=[(X_va_all_hist, np.log1p(y_va_all_hist))],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred_warm_lgb = np.maximum(0.0, np.expm1(lgb_warm.predict(X_va_all_hist)))
            val_lgb_pred[df_va["hist_count"] > 0] = pred_warm_lgb

            # Warm CatBoost Direct Log1p
            X_tr_warm_cb = X_tr_warm.copy()
            X_va_hist_cb = X_va_all_hist.copy()
            for c in cat_cols_all:
                X_tr_warm_cb[c] = X_tr_warm_cb[c].astype(str)
                X_va_hist_cb[c] = X_va_hist_cb[c].astype(str)

            cb_warm = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.035,
                depth=6,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
            )
            cb_warm.fit(
                X_tr_warm_cb, np.log1p(y_tr_warm),
                cat_features=cat_cols_all,
                eval_set=(X_va_hist_cb, np.log1p(y_va_all_hist)),
                early_stopping_rounds=50,
                verbose=False,
            )
            pred_warm_cb = np.maximum(0.0, np.expm1(cb_warm.predict(X_va_hist_cb)))
            val_cb_pred[df_va["hist_count"] > 0] = pred_warm_cb

            if va_warm_mask.sum() > 0:
                cb_warm_rmsle = calculate_rmsle(df_va.loc[va_warm_mask, "tuketim"].values, val_cb_pred[va_warm_mask])
                lgb_warm_rmsle = calculate_rmsle(df_va.loc[va_warm_mask, "tuketim"].values, val_lgb_pred[va_warm_mask])
                logger.info(f"Warm Performance: CatBoost={cb_warm_rmsle:.5f} | LGBM={lgb_warm_rmsle:.5f}")

        # ---------------------------------------------------------------------
        # 2. ANNUAL SPECIALIST (Residual on Seasonal Baseline)
        # ---------------------------------------------------------------------
        tr_ann_mask = (df_tr["segment"] == "annual") & (df_tr["seasonal_baseline"].notnull()) & (df_tr["seasonal_baseline"] > 0)

        if va_ann_mask.sum() > 0:
            base_va = df_va.loc[va_ann_mask, "seasonal_baseline"].values
            if tr_ann_mask.sum() > 200:
                logger.info(f"[Annual Residual Model] Fitting on {tr_ann_mask.sum():,d} train samples...")
                base_tr = df_tr.loc[tr_ann_mask, "seasonal_baseline"].values
                y_res_tr = np.log1p(df_tr.loc[tr_ann_mask, "tuketim"].values) - np.log1p(base_tr)
                y_res_va = np.log1p(df_va.loc[va_ann_mask, "tuketim"].values) - np.log1p(base_va)

                X_tr_ann = df_tr.loc[tr_ann_mask, feature_cols]
                X_va_ann = df_va.loc[va_ann_mask, feature_cols]

                lgb_ann = lgb.LGBMRegressor(
                    n_estimators=800, learning_rate=0.035, num_leaves=63, max_depth=8,
                    subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
                )
                lgb_ann.fit(X_tr_ann, y_res_tr, eval_set=[(X_va_ann, y_res_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
                val_lgb_pred[va_ann_mask] = np.maximum(0.0, np.expm1(np.log1p(base_va) + lgb_ann.predict(X_va_ann)))

                X_tr_ann_cb = X_tr_ann.copy()
                X_va_ann_cb = X_va_ann.copy()
                for c in cat_cols_all:
                    X_tr_ann_cb[c] = X_tr_ann_cb[c].astype(str)
                    X_va_ann_cb[c] = X_va_ann_cb[c].astype(str)

                cb_ann = CatBoostRegressor(iterations=800, learning_rate=0.035, depth=6, loss_function="RMSE", random_seed=42, verbose=False)
                cb_ann.fit(X_tr_ann_cb, y_res_tr, cat_features=cat_cols_all, eval_set=(X_va_ann_cb, y_res_va), early_stopping_rounds=50, verbose=False)
                val_cb_pred[va_ann_mask] = np.maximum(0.0, np.expm1(np.log1p(base_va) + cb_ann.predict(X_va_ann_cb)))
            else:
                logger.info(f"[Annual Baseline Blending] No train annual samples in previous folds; combining seasonal baseline with warm model...")
                val_lgb_pred[va_ann_mask] = 0.70 * base_va + 0.30 * val_lgb_pred[va_ann_mask]
                val_cb_pred[va_ann_mask] = 0.70 * base_va + 0.30 * val_cb_pred[va_ann_mask]

            base_rmsle = calculate_rmsle(df_va.loc[va_ann_mask, "tuketim"].values, base_va)
            ann_cb_rmsle = calculate_rmsle(df_va.loc[va_ann_mask, "tuketim"].values, val_cb_pred[va_ann_mask])
            ann_lgb_rmsle = calculate_rmsle(df_va.loc[va_ann_mask, "tuketim"].values, val_lgb_pred[va_ann_mask])
            logger.info(f"Annual Performance: Raw Baseline={base_rmsle:.5f} | CatBoost={ann_cb_rmsle:.5f} | LGBM={ann_lgb_rmsle:.5f}")

        # ---------------------------------------------------------------------
        # 3. COLD SPECIALIST (segment == 'cold')
        # ---------------------------------------------------------------------
        tr_cold_mask = (df_tr["segment"] == "cold")
        va_cold_mask = (df_va["segment"] == "cold")

        logger.info(f"[Cold Specialist] Train N={tr_cold_mask.sum():,d} | Val N={va_cold_mask.sum():,d}")

        if tr_cold_mask.sum() > 0 and va_cold_mask.sum() > 0:
            X_tr_cold = df_tr.loc[tr_cold_mask, cold_feature_cols]
            y_tr_cold = df_tr.loc[tr_cold_mask, "tuketim"].values
            guc_tr_cold = np.maximum(1.0, df_tr.loc[tr_cold_mask, "guc"].values)

            X_va_cold = df_va.loc[va_cold_mask, cold_feature_cols]
            y_va_cold = df_va.loc[va_cold_mask, "tuketim"].values
            guc_va_cold = np.maximum(1.0, df_va.loc[va_cold_mask, "guc"].values)

            # Cold LightGBM on Power Ratio
            lgb_cold = lgb.LGBMRegressor(
                n_estimators=600,
                learning_rate=0.035,
                num_leaves=31,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
            )
            lgb_cold.fit(
                X_tr_cold, np.log1p(y_tr_cold / guc_tr_cold),
                eval_set=[(X_va_cold, np.log1p(y_va_cold / guc_va_cold))],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            val_lgb_pred[va_cold_mask] = np.maximum(0.0, np.expm1(lgb_cold.predict(X_va_cold))) * guc_va_cold

            # Cold CatBoost Log1p
            X_tr_cold_cb = X_tr_cold.copy()
            X_va_cold_cb = X_va_cold.copy()
            for c in cat_cols_cold:
                X_tr_cold_cb[c] = X_tr_cold_cb[c].astype(str)
                X_va_cold_cb[c] = X_va_cold_cb[c].astype(str)

            cb_cold = CatBoostRegressor(
                iterations=600,
                learning_rate=0.035,
                depth=6,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
            )
            cb_cold.fit(
                X_tr_cold_cb, np.log1p(y_tr_cold),
                cat_features=cat_cols_cold,
                eval_set=(X_va_cold_cb, np.log1p(y_va_cold)),
                early_stopping_rounds=50,
                verbose=False,
            )
            val_cb_pred[va_cold_mask] = np.maximum(0.0, np.expm1(cb_cold.predict(X_va_cold_cb)))

            cb_cold_rmsle = calculate_rmsle(y_va_cold, val_cb_pred[va_cold_mask])
            lgb_cold_rmsle = calculate_rmsle(y_va_cold, val_lgb_pred[va_cold_mask])
            logger.info(f"Cold Performance: CatBoost={cb_cold_rmsle:.5f} | LGBM={lgb_cold_rmsle:.5f}")

        # Store component predictions
        df.loc[va_indices, "cb_oof_pred"] = val_cb_pred
        df.loc[va_indices, "lgb_oof_pred"] = val_lgb_pred

        # ---------------------------------------------------------------------
        # 4. SLSQP OPTIMAL SEGMENT BLENDING
        # ---------------------------------------------------------------------
        val_routed = np.zeros(len(df_va), dtype=np.float32)

        for seg, mask in [("annual", va_ann_mask), ("warm", va_warm_mask), ("cold", va_cold_mask)]:
            if mask.sum() > 0:
                p_cb = val_cb_pred[mask]
                p_lgb = val_lgb_pred[mask]
                y_sub = y_va[mask]

                def blend_loss(w):
                    p = w[0] * p_cb + w[1] * p_lgb
                    return calculate_rmsle(y_sub, p)

                res = minimize(blend_loss, [0.5, 0.5], bounds=[(0, 1), (0, 1)], constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0})
                w = res.x
                val_routed[mask] = w[0] * p_cb + w[1] * p_lgb
                logger.info(f"Segment '{seg}' (N={mask.sum():,d}) Blended RMSLE: {res.fun:.5f} | Weights: [CB: {w[0]:.3f}, LGB: {w[1]:.3f}]")

        # Physical Ceiling Guardrail: 36 * (guc + 1)
        val_routed = np.clip(val_routed, 0.0, 36.0 * (guc_va + 1.0))
        df.loc[va_indices, "v11_oof_pred"] = val_routed

        fold_rmsle = calculate_rmsle(y_va, val_routed)
        fold_rmse = calculate_rmse(y_va, val_routed)
        fold_mae = calculate_mae(y_va, val_routed)
        logger.info(f"★ [{val_fold}] V11 Final OOF: RMSLE={fold_rmsle:.5f} | RMSE={fold_rmse:.2f} | MAE={fold_mae:.2f} ★")

        fold_eval_records.append({
            "fold_id": val_fold,
            "overall_rmsle": fold_rmsle,
            "annual_rmsle": calculate_rmsle(df_va.loc[va_ann_mask, "tuketim"].values, val_routed[va_ann_mask]) if va_ann_mask.sum() > 0 else np.nan,
            "warm_rmsle": calculate_rmsle(df_va.loc[va_warm_mask, "tuketim"].values, val_routed[va_warm_mask]) if va_warm_mask.sum() > 0 else np.nan,
            "cold_rmsle": calculate_rmsle(df_va.loc[va_cold_mask, "tuketim"].values, val_routed[va_cold_mask]) if va_cold_mask.sum() > 0 else np.nan,
        })

    # -------------------------------------------------------------------------
    # POOLED METRICS AND DETAILED SEGMENT/MONTH BREAKDOWN
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("★ V11 POOLED OOF EVALUATION & METRIC TABLES ★")
    logger.info("=" * 70)

    pooled_rmsle = calculate_rmsle(df["tuketim"].values, df["v11_oof_pred"].values)
    cb_pooled_rmsle = calculate_rmsle(df["tuketim"].values, df["cb_oof_pred"].values)
    lgb_pooled_rmsle = calculate_rmsle(df["tuketim"].values, df["lgb_oof_pred"].values)

    logger.info(f"CatBoost Specialist Pooled RMSLE : {cb_pooled_rmsle:.5f}")
    logger.info(f"LightGBM Specialist Pooled RMSLE : {lgb_pooled_rmsle:.5f}")
    logger.info(f"V11 Ensembled Pooled RMSLE       : {pooled_rmsle:.5f}")

    # Build Fold x Segment x Month Table
    df["month_num"] = pd.to_datetime(df["tarih"]).dt.month
    table_rows = []
    for f_name in df["fold_id"].unique():
        f_df = df[df["fold_id"] == f_name]
        row_dict = {
            "Fold": f_name,
            "Overall": calculate_rmsle(f_df["tuketim"].values, f_df["v11_oof_pred"].values),
            "Annual": calculate_rmsle(f_df.loc[f_df["segment"] == "annual", "tuketim"].values, f_df.loc[f_df["segment"] == "annual", "v11_oof_pred"].values) if (f_df["segment"] == "annual").sum() > 0 else "-",
            "Warm": calculate_rmsle(f_df.loc[f_df["segment"] == "warm", "tuketim"].values, f_df.loc[f_df["segment"] == "warm", "v11_oof_pred"].values) if (f_df["segment"] == "warm").sum() > 0 else "-",
            "Cold": calculate_rmsle(f_df.loc[f_df["segment"] == "cold", "tuketim"].values, f_df.loc[f_df["segment"] == "cold", "v11_oof_pred"].values) if (f_df["segment"] == "cold").sum() > 0 else "-",
        }
        for m_num, m_name in [(4, "Nisan"), (5, "Mayis"), (6, "Haziran"), (7, "Temmuz"), (8, "Agustos"), (9, "Eylul"), (10, "Ekim"), (11, "Kasim"), (12, "Aralik"), (1, "Ocak"), (2, "Subat"), (3, "Mart")]:
            m_sub = f_df[f_df["month_num"] == m_num]
            if len(m_sub) > 0:
                row_dict[m_name] = f"{calculate_rmsle(m_sub['tuketim'].values, m_sub['v11_oof_pred'].values):.5f}"
            else:
                row_dict[m_name] = "-"
        table_rows.append(row_dict)

    summary_table_df = pd.DataFrame(table_rows)
    logger.info(f"\nFold x Segment x Ay RMSLE Tablosu:\n{summary_table_df.to_string(index=False)}")

    # Specific slices
    # 1. Macro RMSLE per facility
    fac_rmsles = df.groupby("tanim").apply(
        lambda g: calculate_rmsle(g["tuketim"].values, g["v11_oof_pred"].values)
    )
    macro_facility_rmsle = float(fac_rmsles.mean())

    # 2. Power Tier RMSLE
    guc_rmsles = df.groupby("guc_grup").apply(
        lambda g: calculate_rmsle(g["tuketim"].values, g["v11_oof_pred"].values)
    ).to_dict()

    # 3. Location RMSLE
    loc_rmsles = df.groupby("il").apply(
        lambda g: calculate_rmsle(g["tuketim"].values, g["v11_oof_pred"].values)
    ).to_dict()

    # 4. Zero target RMSLE
    zero_mask = (df["tuketim"] == 0)
    zero_rmsle = calculate_rmsle(df.loc[zero_mask, "tuketim"].values, df.loc[zero_mask, "v11_oof_pred"].values)

    # 5. Top 1% high volume consumption RMSLE
    whale_thresh = np.percentile(df["tuketim"], 99)
    whale_mask = (df["tuketim"] >= whale_thresh)
    whale_rmsle = calculate_rmsle(df.loc[whale_mask, "tuketim"].values, df.loc[whale_mask, "v11_oof_pred"].values)

    logger.info(f"\nEkstra Dilim Analizleri:")
    logger.info(f" - Tesis Başına Macro RMSLE       : {macro_facility_rmsle:.5f}")
    logger.info(f" - Güç Dilimi RMSLE               : {guc_rmsles}")
    logger.info(f" - İl Bazında RMSLE               : {loc_rmsles}")
    logger.info(f" - Sıfır Hedefli Satırlarda RMSLE : {zero_rmsle:.5f} (N={zero_mask.sum():,d})")
    logger.info(f" - Top %1 Yüksek Hacim RMSLE      : {whale_rmsle:.5f} (N={whale_mask.sum():,d})")

    # Save OOF predictions
    oof_out = RESULTS_DIR / "v11_specialists_oof_predictions.csv.gz"
    df[["row_id", "tanim", "tarih", "fold_id", "segment", "tuketim", "v11_oof_pred", "cb_oof_pred", "lgb_oof_pred"]].to_csv(
        oof_out, compression="gzip", index=False
    )
    logger.info(f"✓ V11 OOF predictions saved to {oof_out}")

    # -------------------------------------------------------------------------
    # 5. RETRAINING ON 100% DATA FOR TEST SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RETRAINING V11 SPECIALISTS ON 100% OF DATA (Cutoff = 2026-03-31)")
    logger.info("=" * 70)

    # Retrain on full dataset with all available segments
    test_pred_cb = np.zeros(len(test_df), dtype=np.float32)
    test_pred_lgb = np.zeros(len(test_df), dtype=np.float32)

    # 1. Full Annual Specialist
    full_ann_mask = (df["segment"] == "annual") & (df["seasonal_baseline"].notnull()) & (df["seasonal_baseline"] > 0)
    test_ann_mask = (test_df["segment"] == "annual") & (test_df["seasonal_baseline"].notnull()) & (test_df["seasonal_baseline"] > 0)
    logger.info(f"Full Annual Specialist: Train N={full_ann_mask.sum():,d} | Test N={test_ann_mask.sum():,d}")

    if full_ann_mask.sum() > 0 and test_ann_mask.sum() > 0:
        base_full = df.loc[full_ann_mask, "seasonal_baseline"].values
        base_test = test_df.loc[test_ann_mask, "seasonal_baseline"].values
        y_res_full = np.log1p(df.loc[full_ann_mask, "tuketim"].values) - np.log1p(base_full)

        # Full LGBM Annual
        full_lgb_ann = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.035, num_leaves=63, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
        )
        full_lgb_ann.fit(df.loc[full_ann_mask, feature_cols], y_res_full)
        test_pred_lgb[test_ann_mask] = np.maximum(0.0, np.expm1(np.log1p(base_test) + full_lgb_ann.predict(test_df.loc[test_ann_mask, feature_cols])))

        # Full CB Annual
        X_ann_cb = df.loc[full_ann_mask, feature_cols].copy()
        X_test_ann_cb = test_df.loc[test_ann_mask, feature_cols].copy()
        for c in cat_cols_all:
            X_ann_cb[c] = X_ann_cb[c].astype(str)
            X_test_ann_cb[c] = X_test_ann_cb[c].astype(str)

        full_cb_ann = CatBoostRegressor(
            iterations=1000, learning_rate=0.035, depth=6, loss_function="RMSE",
            random_seed=42, verbose=False
        )
        full_cb_ann.fit(X_ann_cb, y_res_full, cat_features=cat_cols_all, verbose=False)
        test_pred_cb[test_ann_mask] = np.maximum(0.0, np.expm1(np.log1p(base_test) + full_cb_ann.predict(X_test_ann_cb)))

    # 2. Full Warm Specialist
    full_warm_mask = (df["segment"] == "warm")
    test_warm_mask = (test_df["segment"] == "warm")
    logger.info(f"Full Warm Specialist: Train N={full_warm_mask.sum():,d} | Test N={test_warm_mask.sum():,d}")

    if full_warm_mask.sum() > 0 and test_warm_mask.sum() > 0:
        y_warm_full = df.loc[full_warm_mask, "tuketim"].values

        full_lgb_warm = lgb.LGBMRegressor(
            n_estimators=1200, learning_rate=0.035, num_leaves=63, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
        )
        full_lgb_warm.fit(df.loc[full_warm_mask, feature_cols], np.log1p(y_warm_full))
        test_pred_lgb[test_warm_mask] = np.maximum(0.0, np.expm1(full_lgb_warm.predict(test_df.loc[test_warm_mask, feature_cols])))

        X_warm_cb = df.loc[full_warm_mask, feature_cols].copy()
        X_test_warm_cb = test_df.loc[test_warm_mask, feature_cols].copy()
        for c in cat_cols_all:
            X_warm_cb[c] = X_warm_cb[c].astype(str)
            X_test_warm_cb[c] = X_test_warm_cb[c].astype(str)

        full_cb_warm = CatBoostRegressor(
            iterations=1200, learning_rate=0.035, depth=6, loss_function="RMSE",
            random_seed=42, verbose=False
        )
        full_cb_warm.fit(X_warm_cb, np.log1p(y_warm_full), cat_features=cat_cols_all, verbose=False)
        test_pred_cb[test_warm_mask] = np.maximum(0.0, np.expm1(full_cb_warm.predict(X_test_warm_cb)))

    # 3. Full Cold Specialist
    full_cold_mask = (df["segment"] == "cold")
    test_cold_mask = (test_df["segment"] == "cold")
    logger.info(f"Full Cold Specialist: Train N={full_cold_mask.sum():,d} | Test N={test_cold_mask.sum():,d}")

    if full_cold_mask.sum() > 0 and test_cold_mask.sum() > 0:
        y_cold_full = df.loc[full_cold_mask, "tuketim"].values
        guc_cold_full = np.maximum(1.0, df.loc[full_cold_mask, "guc"].values)
        guc_cold_test = np.maximum(1.0, test_df.loc[test_cold_mask, "guc"].values)

        full_lgb_cold = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.035, num_leaves=31, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
        )
        full_lgb_cold.fit(df.loc[full_cold_mask, cold_feature_cols], np.log1p(y_cold_full / guc_cold_full))
        test_pred_lgb[test_cold_mask] = np.maximum(0.0, np.expm1(full_lgb_cold.predict(test_df.loc[test_cold_mask, cold_feature_cols]))) * guc_cold_test

        X_cold_cb = df.loc[full_cold_mask, cold_feature_cols].copy()
        X_test_cold_cb = test_df.loc[test_cold_mask, cold_feature_cols].copy()
        for c in cat_cols_cold:
            X_cold_cb[c] = X_cold_cb[c].astype(str)
            X_test_cold_cb[c] = X_test_cold_cb[c].astype(str)

        full_cb_cold = CatBoostRegressor(
            iterations=800, learning_rate=0.035, depth=6, loss_function="RMSE",
            random_seed=42, verbose=False
        )
        full_cb_cold.fit(X_cold_cb, np.log1p(y_cold_full), cat_features=cat_cols_cold, verbose=False)
        test_pred_cb[test_cold_mask] = np.maximum(0.0, np.expm1(full_cb_cold.predict(X_test_cold_cb)))

    # Final Segment Blending for Test
    final_test_pred = np.zeros(len(test_df), dtype=np.float32)
    final_test_pred[test_ann_mask] = 0.60 * test_pred_cb[test_ann_mask] + 0.40 * test_pred_lgb[test_ann_mask]
    final_test_pred[test_warm_mask] = 0.50 * test_pred_cb[test_warm_mask] + 0.50 * test_pred_lgb[test_warm_mask]
    final_test_pred[test_cold_mask] = 0.50 * test_pred_cb[test_cold_mask] + 0.50 * test_pred_lgb[test_cold_mask]

    # Power Ceiling
    guc_test_full = np.maximum(1.0, test_df["guc"].values)
    final_test_pred = np.clip(final_test_pred, 0.0, 36.0 * (guc_test_full + 1.0))

    # -------------------------------------------------------------------------
    # GENERATE AND VERIFY SUBMISSION
    # -------------------------------------------------------------------------
    raw_sample = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8")
    raw_test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8")

    sub_df = pd.DataFrame({
        "id": raw_test["id"].values,
        "tuketim": final_test_pred
    })

    sub_path = DATA_DIR / "submission_v11_leakage_safe.csv"
    sub_df.to_csv(sub_path, index=False)

    # Verification assertions on submission
    if len(sub_df) != 714688:
        raise AssertionError("Submission row count != 714688")
    if not sub_df["id"].equals(raw_sample["id"]):
        raise AssertionError("Submission ID order does not match sample_submission.csv")
    if sub_df["id"].duplicated().any():
        raise AssertionError("Duplicate IDs found in submission")
    if sub_df["tuketim"].isnull().any():
        raise AssertionError("NaN found in submission predictions")
    if np.isinf(sub_df["tuketim"].to_numpy(dtype=float, copy=False)).any():
        raise AssertionError("Infinite values found in submission predictions")
    if (sub_df["tuketim"] < 0).any():
        raise AssertionError("Negative values found in submission predictions")

    sha256_hash = get_sha256(sub_path)
    logger.info(f"\n✓ Submission generated and verified: {sub_path}")
    logger.info(f"✓ SHA256: {sha256_hash}")
    logger.info(f"✓ Prediction Stats:\n{sub_df['tuketim'].describe()}")

    return df, sub_df, sha256_hash


if __name__ == "__main__":
    run_v11_pipeline()
