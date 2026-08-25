"""Cold-Start Master (V12) Pipeline.

Hedef: Test log-hatasının %59'unu üreten 158.369 cold-start satırını
iki aşamalı Hurdle modeli, 6 kademeli ayrık yaş ramp-up'ı, kohort dayanıklılığı,
SVD mevsimsellik ayrışımı ve Bayes hiyerarşik öncülleriyle optimize etmek.

V8R resmi gönderimindeki (1.13312) Annual (243.839) ve Warm (312.480) satırları
%100 bit-seviyesinde korunacak; yalnızca 158.369 cold-start satırı güncellenecektir.
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
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
BASE_SUB_PATH = DATA_DIR / "submission_v8r_verified_final.csv"
OUTPUT_SUB_PATH = DATA_DIR / "submission_v12_cold_master.csv"
OUTPUT_DIR = DATA_DIR / "features_v11_shap" / "cold_master_results"


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


def run_cold_master():
    logger.info("=" * 75)
    logger.info("STARTING COLD-START MASTER (V12) TRAINING & SUBMISSION PIPELINE")
    logger.info("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Datasets
    logger.info("Loading raw train.csv and test.csv...")
    train_df = pd.read_csv(DATA_DIR / "train.csv", dtype={"tanim": str, "row_id": str}, parse_dates=["tarih"])
    test_df = pd.read_csv(DATA_DIR / "test.csv", dtype={"tanim": str, "row_id": str}, parse_dates=["tarih"])

    train_df = parse_locations(train_df)
    test_df = parse_locations(test_df)

    train_facs = set(train_df["tanim"].unique())
    test_facs = set(test_df["tanim"].unique())
    cold_facs = test_facs - train_facs
    logger.info(f"Total train facilities: {len(train_facs):,d}")
    logger.info(f"Total test facilities : {len(test_facs):,d} | Cold-start facilities: {len(cold_facs):,d}")

    # 2. Opening Cohort Detection & Discrete Age Bins
    train_starts = train_df.groupby("tanim")["tarih"].min().reset_index()
    train_starts.columns = ["tanim", "start_date"]
    train_df = train_df.merge(train_starts, on="tanim", how="left")
    train_df["age_days"] = (train_df["tarih"] - train_df["start_date"]).dt.days

    test_starts = test_df.groupby("tanim")["tarih"].min().reset_index()
    test_starts.columns = ["tanim", "start_date"]
    test_df = test_df.merge(test_starts, on="tanim", how="left")
    test_df["age_days"] = (test_df["tarih"] - test_df["start_date"]).dt.days

    # Age Bins: 0-3, 4-7, 8-14, 15-30, 31-60, 60+
    age_bins = [-1, 3, 7, 14, 30, 60, 10000]
    age_labels = ["age_0_3", "age_4_7", "age_8_14", "age_15_30", "age_31_60", "age_60_plus"]
    train_df["age_bin"] = pd.cut(train_df["age_days"], bins=age_bins, labels=age_labels).astype(str)
    test_df["age_bin"] = pd.cut(test_df["age_days"], bins=age_bins, labels=age_labels).astype(str)

    # Capacity Bins
    guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
    guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
    train_df["guc_bin"] = pd.cut(train_df["guc"], bins=guc_bins, labels=guc_labels).astype(str)
    test_df["guc_bin"] = pd.cut(test_df["guc"], bins=guc_bins, labels=guc_labels).astype(str)

    # Cohort Size & Structure
    train_cohort_sizes = train_starts.groupby("start_date")["tanim"].count().to_dict()
    test_cohort_sizes = test_starts.groupby("start_date")["tanim"].count().to_dict()

    train_df["cohort_size"] = train_df["start_date"].map(train_cohort_sizes).fillna(1)
    test_df["cohort_size"] = test_df["start_date"].map(test_cohort_sizes).fillna(1)
    train_df["is_mass_cohort"] = (train_df["cohort_size"] >= 100).astype(int)
    test_df["is_mass_cohort"] = (test_df["cohort_size"] >= 100).astype(int)

    # Time features
    for d in [train_df, test_df]:
        d["month"] = d["tarih"].dt.month
        d["day_of_week"] = d["tarih"].dt.dayofweek
        d["day"] = d["tarih"].dt.day
        d["day_of_year"] = d["tarih"].dt.dayofyear
        d["doy_sin"] = np.sin(2 * np.pi * d["day_of_year"] / 365.25)
        d["doy_cos"] = np.cos(2 * np.pi * d["day_of_year"] / 365.25)
        d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
        d["is_summer"] = d["month"].isin([6, 7, 8]).astype(int)
        d["log_guc"] = np.log1p(np.maximum(1.0, d["guc"]))
        d["log_guc_x_summer"] = d["log_guc"] * d["is_summer"]

    # Filter to new facility openings in train (opened after 2025-01-01) for cold model training
    train_cold_df = train_df[train_df["start_date"] > "2025-01-01"].copy()
    test_cold_df = test_df[test_df["tanim"].isin(cold_facs)].copy()

    logger.info(f"Train cold opening samples: {len(train_cold_df):,d} across {train_cold_df['tanim'].nunique():,d} facilities.")
    logger.info(f"Test cold-start samples    : {len(test_cold_df):,d} across {test_cold_df['tanim'].nunique():,d} facilities.")

    # -------------------------------------------------------------------------
    # 3. FACILITY-BALANCED HIERARCHICAL EMPIRICAL BAYES PRIORS
    # -------------------------------------------------------------------------
    logger.info("Computing Facility-Balanced Hierarchical Empirical Bayes Priors...")

    # Step A: Facility-level daily mean and power density
    fac_summary = train_df.groupby("tanim").agg(
        guc=("guc", "first"),
        il=("il", "first"),
        ilce=("ilce", "first"),
        bolge=("bolge", "first"),
        guc_bin=("guc_bin", "first"),
        mean_tuketim=("tuketim", "mean"),
    ).reset_index()
    fac_summary["density"] = fac_summary["mean_tuketim"] / np.maximum(1.0, fac_summary["guc"])

    global_density = float(fac_summary["density"].median())
    guc_density = fac_summary.groupby("guc_bin")["density"].median().to_dict()
    ilce_density = fac_summary.groupby(["ilce", "guc_bin"])["density"].median().to_dict()
    il_density = fac_summary.groupby("il")["density"].median().to_dict()

    # Step B: Facility Seasonal & Day-of-Week Multipliers
    train_df["facility_mean"] = train_df["tanim"].map(fac_summary.set_index("tanim")["mean_tuketim"])
    train_df["relative_consumption"] = train_df["tuketim"] / np.maximum(1.0, train_df["facility_mean"])

    month_factors = train_df.groupby("month")["relative_consumption"].median().to_dict()
    dow_factors = train_df.groupby("day_of_week")["relative_consumption"].median().to_dict()
    il_month_factors = train_df.groupby(["il", "month"])["relative_consumption"].median().to_dict()

    def get_bayes_prior(row) -> float:
        g = float(row["guc"])
        ilce_key = (row["ilce"], row["guc_bin"])
        il_key = (row["il"], row["month"])

        d_ilce = ilce_density.get(ilce_key, None)
        d_guc = guc_density.get(row["guc_bin"], global_density)
        d_il = il_density.get(row["il"], global_density)

        if d_ilce is not None:
            base_d = 0.50 * d_ilce + 0.35 * d_guc + 0.15 * d_il
        else:
            base_d = 0.70 * d_guc + 0.30 * d_il

        m_factor = il_month_factors.get(il_key, month_factors.get(row["month"], 1.0))
        d_factor = dow_factors.get(row["day_of_week"], 1.0)
        return float(base_d * g * m_factor * d_factor)

    train_cold_df["bayes_prior"] = train_cold_df.apply(get_bayes_prior, axis=1)
    test_cold_df["bayes_prior"] = test_cold_df.apply(get_bayes_prior, axis=1)

    # -------------------------------------------------------------------------
    # 4. LOW-RANK SVD SEASONAL MATRIX DECOMPOSITION
    # -------------------------------------------------------------------------
    logger.info("Decomposing seasonal patterns with Low-Rank SVD...")

    # Build facility x (month, dow) profile matrix on all established train facilities
    estab_train = train_df[train_df["start_date"] <= "2025-06-01"].copy()
    profile_pivot = estab_train.pivot_table(
        index="tanim",
        columns=["month", "day_of_week"],
        values="relative_consumption",
        aggfunc="median",
    ).fillna(1.0)

    svd = TruncatedSVD(n_components=4, random_state=42)
    svd_components = svd.fit_transform(profile_pivot.values)  # N_fac x 4
    V_matrix = svd.components_  # 4 x 84

    # Fit Ridge regressor from (log_guc, il, guc_bin) to predict SVD components
    estab_fac_meta = fac_summary.set_index("tanim").loc[profile_pivot.index]
    X_svd_train = pd.get_dummies(estab_fac_meta[["il", "guc_bin"]], drop_first=True)
    X_svd_train["log_guc"] = np.log1p(estab_fac_meta["guc"])

    ridge_svd = Ridge(alpha=10.0)
    ridge_svd.fit(X_svd_train, svd_components)

    # Predict SVD profile multipliers for cold facilities
    def get_svd_multiplier(df_sub: pd.DataFrame) -> np.ndarray:
        fac_m = df_sub.groupby("tanim").agg(
            guc=("guc", "first"),
            il=("il", "first"),
            guc_bin=("guc_bin", "first"),
        ).reset_index()

        X_m = pd.get_dummies(fac_m[["il", "guc_bin"]], drop_first=True)
        # align columns
        for col in X_svd_train.columns:
            if col not in X_m.columns:
                X_m[col] = 0.0
        X_m = X_m[X_svd_train.columns]
        X_m["log_guc"] = np.log1p(fac_m["guc"])

        u_pred = ridge_svd.predict(X_m)  # N_cold x 4
        # map back
        u_df = pd.DataFrame(u_pred, index=fac_m["tanim"])

        # Construct multipliers
        cols_map = {(m, d): idx for idx, (m, d) in enumerate(profile_pivot.columns)}
        multipliers = np.zeros(len(df_sub), dtype=np.float32)

        tanim_arr = df_sub["tanim"].values
        m_arr = df_sub["month"].values
        d_arr = df_sub["day_of_week"].values

        for idx, (t, m, d) in enumerate(zip(tanim_arr, m_arr, d_arr)):
            if t in u_df.index:
                u_vec = u_df.loc[t].values
                col_idx = cols_map.get((m, d), 0)
                reconstructed = 1.0 + float(np.dot(u_vec, V_matrix[:, col_idx]))
                multipliers[idx] = max(0.2, min(3.0, reconstructed))
            else:
                multipliers[idx] = 1.0
        return multipliers

    train_cold_df["svd_multiplier"] = get_svd_multiplier(train_cold_df)
    test_cold_df["svd_multiplier"] = get_svd_multiplier(test_cold_df)
    train_cold_df["svd_prior"] = train_cold_df["bayes_prior"] * train_cold_df["svd_multiplier"]
    test_cold_df["svd_prior"] = test_cold_df["bayes_prior"] * test_cold_df["svd_multiplier"]

    # -------------------------------------------------------------------------
    # 5. STEP 1: HURDLE ZERO-INFLATION CLASSIFIER (P(tuketim > 0))
    # -------------------------------------------------------------------------
    logger.info("Training Hurdle Zero-Inflation Classifier P(y > 0)...")

    features_hurdle = [
        "guc", "log_guc", "age_days", "cohort_size", "is_mass_cohort",
        "month", "day_of_week", "day", "day_of_year", "doy_sin", "doy_cos",
        "is_weekend", "is_summer", "log_guc_x_summer", "bayes_prior", "svd_prior",
        "il", "ilce", "bolge", "guc_bin", "age_bin"
    ]
    cat_cols_hurdle = ["il", "ilce", "bolge", "guc_bin", "age_bin"]

    for c in cat_cols_hurdle:
        train_cold_df[c] = train_cold_df[c].astype("category")
        test_cold_df[c] = test_cold_df[c].astype("category")

    y_hurdle_tr = (train_cold_df["tuketim"] > 0).astype(int).values

    clf_hurdle = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    clf_hurdle.fit(train_cold_df[features_hurdle], y_hurdle_tr)

    p_pos_train = clf_hurdle.predict_proba(train_cold_df[features_hurdle])[:, 1]
    p_pos_test = clf_hurdle.predict_proba(test_cold_df[features_hurdle])[:, 1]
    logger.info(f"Hurdle Train P(pos) mean: {p_pos_train.mean():.4f} | Test P(pos) mean: {p_pos_test.mean():.4f}")

    # -------------------------------------------------------------------------
    # 6. STEP 2: POSITIVE CONSUMPTION REGRESSION (CATBOOST HUBER + LIGHTGBM)
    # -------------------------------------------------------------------------
    logger.info("Training Positive Cold Regressors on (tuketim > 0)...")

    pos_mask = (train_cold_df["tuketim"] > 0)
    X_pos_tr = train_cold_df.loc[pos_mask, features_hurdle]
    y_pos_tr = train_cold_df.loc[pos_mask, "tuketim"].values
    guc_pos_tr = np.maximum(1.0, train_cold_df.loc[pos_mask, "guc"].values)

    # 1. LightGBM Power-Ratio Regressor
    lgb_cold = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    lgb_cold.fit(X_pos_tr, np.log1p(y_pos_tr / guc_pos_tr))

    # 2. CatBoost Power-Ratio Regressor (Predicts log1p(y / guc) to prevent extrapolation overflow)
    X_pos_cb = X_pos_tr.copy()
    for c in cat_cols_hurdle:
        X_pos_cb[c] = X_pos_cb[c].astype(str)

    cb_cold = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        loss_function="Huber:delta=1.5",
        random_seed=42,
        verbose=False,
    )
    cb_cold.fit(X_pos_cb, np.log1p(y_pos_tr / guc_pos_tr), cat_features=cat_cols_hurdle, verbose=False)

    # -------------------------------------------------------------------------
    # 7. LEAVE-ONE-OPENING-COHORT-OUT CROSS VALIDATION AUDIT
    # -------------------------------------------------------------------------
    logger.info("\nEvaluating Leave-One-Opening-Cohort-Out CV on Major Train Cohorts...")
    major_cohorts = train_cold_df["start_date"].value_counts()[lambda s: s >= 2000].index.tolist()
    cohort_evals = []

    for c_date in major_cohorts[:5]:
        val_mask = train_cold_df["start_date"] == c_date
        tr_mask = train_cold_df["start_date"] != c_date

        X_c_tr = train_cold_df.loc[tr_mask & (train_cold_df["tuketim"] > 0), features_hurdle]
        y_c_tr = train_cold_df.loc[tr_mask & (train_cold_df["tuketim"] > 0), "tuketim"].values
        guc_c_tr = np.maximum(1.0, train_cold_df.loc[tr_mask & (train_cold_df["tuketim"] > 0), "guc"].values)

        X_c_va = train_cold_df.loc[val_mask, features_hurdle]
        y_c_va = train_cold_df.loc[val_mask, "tuketim"].values
        guc_c_va = np.maximum(1.0, train_cold_df.loc[val_mask, "guc"].values)

        # Fit CV LightGBM on Power Ratio
        m_lgb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=6, random_state=42, n_jobs=-1)
        m_lgb.fit(X_c_tr, np.log1p(y_c_tr / guc_c_tr))
        p_lgb = np.maximum(0.0, np.expm1(m_lgb.predict(X_c_va))) * guc_c_va

        # CV Bayes Prior & SVD
        p_bayes = train_cold_df.loc[val_mask, "bayes_prior"].values
        p_svd = train_cold_df.loc[val_mask, "svd_prior"].values

        # Gated by Hurdle
        p_gated = p_pos_train[val_mask] * (0.50 * p_lgb + 0.30 * p_svd + 0.20 * p_bayes)
        p_gated = np.clip(p_gated, 0.0, 36.0 * (guc_c_va + 1.0))

        c_rmsle = calculate_rmsle(y_c_va, p_gated)
        cohort_evals.append({
            "cohort_start": str(c_date)[:10],
            "n_samples": val_mask.sum(),
            "n_facilities": train_cold_df.loc[val_mask, "tanim"].nunique(),
            "cohort_rmsle": c_rmsle,
        })
        logger.info(f"Cohort {str(c_date)[:10]} (N={val_mask.sum():,d} rows, {train_cold_df.loc[val_mask, 'tanim'].nunique():,d} facs) -> RMSLE: {c_rmsle:.5f}")

    cohort_df = pd.DataFrame(cohort_evals)
    logger.info(f"\nCohort CV Summary:\n{cohort_df.to_string(index=False)}")
    logger.info(f"Worst Cohort RMSLE: {cohort_df['cohort_rmsle'].max():.5f} | Mean Cohort RMSLE: {cohort_df['cohort_rmsle'].mean():.5f}")

    # -------------------------------------------------------------------------
    # 8. TEST COLD PREDICTIONS AND CONTROLLED SUBMISSION ASSEMBLY
    # -------------------------------------------------------------------------
    logger.info("\nGenerating Test Cold Predictions...")

    guc_cold_test = np.maximum(1.0, test_cold_df["guc"].values)
    pred_pos_lgb = np.maximum(0.0, np.expm1(lgb_cold.predict(test_cold_df[features_hurdle]))) * guc_cold_test

    X_test_cb = test_cold_df[features_hurdle].copy()
    for c in cat_cols_hurdle:
        X_test_cb[c] = X_test_cb[c].astype(str)
    pred_pos_cb = np.maximum(0.0, np.expm1(cb_cold.predict(X_test_cb))) * guc_cold_test

    pred_bayes_test = test_cold_df["bayes_prior"].values
    pred_svd_test = test_cold_df["svd_prior"].values

    # Blended Magnitude
    blended_magnitude = 0.40 * pred_pos_cb + 0.35 * pred_pos_lgb + 0.15 * pred_svd_test + 0.10 * pred_bayes_test

    # Hurdle Gating: E[y] = P(y > 0) * E[y | y > 0]
    final_cold_preds = p_pos_test * blended_magnitude

    # Physical Capacity Ceiling Guardrail: 36 * (guc + 1)
    final_cold_preds = np.clip(final_cold_preds, 0.0, 36.0 * (guc_cold_test + 1.0))

    logger.info(f"Cold Test Prediction Summary:\n{pd.Series(final_cold_preds).describe()}")

    # -------------------------------------------------------------------------
    # 9. ASSEMBLE CONTROLLED SUBMISSION (V8R Base + V12 Cold)
    # -------------------------------------------------------------------------
    logger.info(f"\nLoading baseline submission: {BASE_SUB_PATH}...")
    base_sub = pd.read_csv(BASE_SUB_PATH)

    cold_ids = set(test_cold_df["id"].values)
    cold_mask = base_sub["id"].isin(cold_ids)

    logger.info(f"Total submission rows : {len(base_sub):,d}")
    logger.info(f"Cold-start rows to update: {cold_mask.sum():,d} (Expected: 158,369)")
    logger.info(f"Untouched non-cold rows  : {(~cold_mask).sum():,d} (Expected: 556,319)")

    assert cold_mask.sum() == 158369, f"Cold rows {cold_mask.sum()} != 158369"
    assert (~cold_mask).sum() == 556319, f"Non-cold rows {(~cold_mask).sum()} != 556319"

    # Create mapping from ID to new cold prediction
    cold_pred_map = dict(zip(test_cold_df["id"].values, final_cold_preds))

    # Update only cold rows
    base_sub.loc[cold_mask, "tuketim"] = base_sub.loc[cold_mask, "id"].map(cold_pred_map)

    # Final sanity checks
    assert len(base_sub) == 714688, "Row count != 714688"
    assert not base_sub["id"].duplicated().any(), "Duplicate IDs found"
    assert not base_sub["tuketim"].isnull().any(), "NaN values found"
    assert not np.isinf(base_sub["tuketim"].to_numpy(dtype=float, copy=False)).any(), "Inf values found"
    assert (base_sub["tuketim"] >= 0).all(), "Negative values found"

    base_sub.to_csv(OUTPUT_SUB_PATH, index=False)
    sha256_hash = get_sha256(OUTPUT_SUB_PATH)

    logger.info("=" * 75)
    logger.info(f"✓ V12 COLD-MASTER SUBMISSION SUCCESSFULLY GENERATED!")
    logger.info(f"✓ Output Path: {OUTPUT_SUB_PATH}")
    logger.info(f"✓ File Size  : {OUTPUT_SUB_PATH.stat().st_size:,d} bytes")
    logger.info(f"✓ SHA256     : {sha256_hash}")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_cold_master()
