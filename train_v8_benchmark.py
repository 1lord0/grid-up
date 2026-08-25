"""V8 Referans Benchmark Pipeline.

Yeni V11 sızıntısız veri setinin 3 ardışık ileri-zaman foldunda (Fold A, B, C)
birebir V8 mimarisini çalıştırarak adil karşılaştırma tabanını (benchmark) belirler.
"""

from __future__ import annotations

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
RESULTS_DIR = OUTPUT_DIR / "benchmark_results"


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.clip(y_true, 0, None)
    y_p = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_p) - np.log1p(y_t)) ** 2)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def run_v8_benchmark():
    logger.info("=" * 70)
    logger.info("STARTING V8 BENCHMARK TRAINING ON V11 ROLLING FOLDS")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train_features_v11.csv.gz"

    logger.info(f"Loading {train_path}...")
    df = pd.read_csv(train_path, compression="gzip", dtype={"tanim": str, "row_id": str}, encoding="utf-8")
    logger.info(f"Loaded: {len(df):,d} rows, {len(df.columns)} columns.")

    meta_cols = ["row_id", "tanim", "tarih", "cutoff_date", "fold_id", "segment", "tuketim"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or df[c].dtype.name == "category"
    ]
    for c in cat_cols:
        df[c] = df[c].fillna("__MISSING__").astype("category")

    folds_def = [
        ("fold_a_apr_jul_2025", ["fold_b_aug_nov_2025", "fold_c_dec_mar_2026"]),
        ("fold_b_aug_nov_2025", ["fold_a_apr_jul_2025", "fold_c_dec_mar_2026"]),
        ("fold_c_dec_mar_2026", ["fold_a_apr_jul_2025", "fold_b_aug_nov_2025"]),
    ]

    df["v8_oof_pred"] = 0.0
    fold_metrics = []

    for val_fold, tr_folds in folds_def:
        logger.info(f"\n>>> Running V8 Benchmark for Validation Fold: {val_fold} <<<")
        tr_mask = df["fold_id"].isin(tr_folds)
        va_mask = df["fold_id"] == val_fold

        X_tr = df.loc[tr_mask, feature_cols].copy()
        y_tr = df.loc[tr_mask, "tuketim"].values
        guc_tr = np.maximum(1.0, df.loc[tr_mask, "guc"].values)

        X_va = df.loc[va_mask, feature_cols].copy()
        y_va = df.loc[va_mask, "tuketim"].values
        guc_va = np.maximum(1.0, df.loc[va_mask, "guc"].values)
        va_indices = df.index[va_mask]

        logger.info(f"Train N={len(X_tr):,d} | Val N={len(X_va):,d}")

        # 1. LightGBM Log1p (Universal Base)
        logger.info("Fitting LightGBM Log1p...")
        lgb_log = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        lgb_log.fit(
            X_tr, np.log1p(y_tr),
            eval_set=[(X_va, np.log1p(y_va))],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_lgb_log_va = np.maximum(0.0, np.expm1(lgb_log.predict(X_va)))

        # 2. Power-Normalized LightGBM
        logger.info("Fitting Power-Normalized LightGBM...")
        lgb_pnorm = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        lgb_pnorm.fit(
            X_tr, np.log1p(y_tr / guc_tr),
            eval_set=[(X_va, np.log1p(y_va / guc_va))],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred_pnorm_va = np.maximum(0.0, np.expm1(lgb_pnorm.predict(X_va))) * guc_va

        # 3. CatBoost Residual / Direct
        logger.info("Fitting CatBoost...")
        cb_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.04,
            depth=6,
            random_seed=42,
            verbose=False,
        )
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(str)

        cb_model.fit(
            X_tr_cb, np.log1p(y_tr),
            cat_features=cat_cols,
            eval_set=(X_va_cb, np.log1p(y_va)),
            early_stopping_rounds=50,
            verbose=False,
        )
        pred_cb_va = np.maximum(0.0, np.expm1(cb_model.predict(X_va_cb)))

        # 4. Prior Baseline
        prior_pred_va = np.maximum(0.0, df.loc[va_mask, "prior_estimated_raw"].values)

        # 5. SLSQP Routing by Tier
        va_has_lag = (df.loc[va_mask, "has_annual_lag"] == 1).values & (df.loc[va_mask, "seasonal_baseline"].notnull()).values
        va_has_hist_no_lag = (df.loc[va_mask, "has_annual_lag"] == 0).values & (df.loc[va_mask, "hist_count"] > 0).values
        va_cold = (df.loc[va_mask, "hist_count"] == 0).values

        val_routed = np.zeros(len(X_va), dtype=np.float32)

        # Tier 1 (Annual/Lag-covered)
        if va_has_lag.sum() > 0:
            def loss_t1(w):
                p = w[0] * pred_lgb_log_va[va_has_lag] + w[1] * pred_pnorm_va[va_has_lag] + w[2] * pred_cb_va[va_has_lag]
                return calculate_rmsle(y_va[va_has_lag], p)
            res1 = minimize(loss_t1, [0.2, 0.2, 0.6], bounds=[(0, 1), (0, 1), (0, 1)], constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0})
            w1 = res1.x
            val_routed[va_has_lag] = w1[0] * pred_lgb_log_va[va_has_lag] + w1[1] * pred_pnorm_va[va_has_lag] + w1[2] * pred_cb_va[va_has_lag]
            logger.info(f"Tier 1 Lag (N={va_has_lag.sum():,d}) -> RMSLE: {res1.fun:.5f} | Weights: {w1.round(3)}")

        # Tier 2 (Warm No-Lag)
        if va_has_hist_no_lag.sum() > 0:
            def loss_t2(w):
                p = w[0] * pred_lgb_log_va[va_has_hist_no_lag] + w[1] * pred_pnorm_va[va_has_hist_no_lag] + w[2] * pred_cb_va[va_has_hist_no_lag]
                return calculate_rmsle(y_va[va_has_hist_no_lag], p)
            res2 = minimize(loss_t2, [0.33, 0.33, 0.34], bounds=[(0, 1), (0, 1), (0, 1)], constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0})
            w2 = res2.x
            val_routed[va_has_hist_no_lag] = w2[0] * pred_lgb_log_va[va_has_hist_no_lag] + w2[1] * pred_pnorm_va[va_has_hist_no_lag] + w2[2] * pred_cb_va[va_has_hist_no_lag]
            logger.info(f"Tier 2 Warm (N={va_has_hist_no_lag.sum():,d}) -> RMSLE: {res2.fun:.5f} | Weights: {w2.round(3)}")

        # Tier 3 (Cold-Start)
        if va_cold.sum() > 0:
            def loss_t3(w):
                p = w[0] * pred_lgb_log_va[va_cold] + w[1] * pred_pnorm_va[va_cold] + w[2] * prior_pred_va[va_cold]
                return calculate_rmsle(y_va[va_cold], p)
            res3 = minimize(loss_t3, [0.2, 0.2, 0.6], bounds=[(0, 1), (0, 1), (0, 1)], constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0})
            w3 = res3.x
            val_routed[va_cold] = w3[0] * pred_lgb_log_va[va_cold] + w3[1] * pred_pnorm_va[va_cold] + w3[2] * prior_pred_va[va_cold]
            logger.info(f"Tier 3 Cold (N={va_cold.sum():,d}) -> RMSLE: {res3.fun:.5f} | Weights: {w3.round(3)}")

        # Physical Ceiling Guardrail: 36 * (guc + 1)
        val_routed = np.clip(val_routed, 0.0, 36.0 * (guc_va + 1.0))
        df.loc[va_indices, "v8_oof_pred"] = val_routed

        fold_rmsle = calculate_rmsle(y_va, val_routed)
        fold_rmse = calculate_rmse(y_va, val_routed)
        fold_mae = calculate_mae(y_va, val_routed)
        logger.info(f"★ [{val_fold}] V8 OOF: RMSLE={fold_rmsle:.5f} | RMSE={fold_rmse:.2f} | MAE={fold_mae:.2f} ★")

        fold_metrics.append({
            "fold_id": val_fold,
            "v8_rmsle": fold_rmsle,
            "v8_rmse": fold_rmse,
            "v8_mae": fold_mae,
        })

    # Pooled Overall Metrics
    pooled_rmsle = calculate_rmsle(df["tuketim"].values, df["v8_oof_pred"].values)
    pooled_rmse = calculate_rmse(df["tuketim"].values, df["v8_oof_pred"].values)
    pooled_mae = calculate_mae(df["tuketim"].values, df["v8_oof_pred"].values)

    logger.info("\n" + "=" * 70)
    logger.info("★ V8 BENCHMARK POOLED OOF SUMMARY ★")
    logger.info("=" * 70)
    logger.info(f"V8 Pooled OOF RMSLE: {pooled_rmsle:.5f}")
    logger.info(f"V8 Pooled OOF RMSE : {pooled_rmse:.2f}")
    logger.info(f"V8 Pooled OOF MAE  : {pooled_mae:.2f}")
    logger.info(pd.DataFrame(fold_metrics).to_string(index=False))

    # Save OOF predictions
    oof_out = RESULTS_DIR / "v8_benchmark_oof_predictions.csv.gz"
    df[["row_id", "tanim", "tarih", "fold_id", "segment", "tuketim", "v8_oof_pred"]].to_csv(
        oof_out, compression="gzip", index=False
    )
    logger.info(f"✓ V8 OOF predictions saved to {oof_out}")

    return df


if __name__ == "__main__":
    run_v8_benchmark()
