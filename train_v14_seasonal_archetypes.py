"""Grid Up Datathon — V14 Seasonal Archetypes & Multi-Month Transfer.

Temel Yenilikler:
1. Dinamik Lookback ve De-noised July+August Summer Envelope (Ağustos->Temmuz Transfer)
2. Log-Space Düzenlenmiş Ampirik Bayes Shrinkage (Sıfır/Düşük kış paydası koruması)
3. 3 Kümeli K-Means Sürekli Arketip Uzaklıkları ve Yumuşak Olasılıklar (Soft Probabilities)
4. CatBoost + LightGBM Mevsimsel Artık (Residual) Çoklu Model Topluluğu
5. Veri Odaklı Kapasite Tavanı Denetimi ve Risk-Hedged Blend (0.75 V14 + 0.25 V8R)
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
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
BASE_SUB_PATH = DATA_DIR / "submission_v8r_verified_final.csv"
OUTPUT_SUB_PATH = DATA_DIR / "submission_v14_seasonal_archetypes.csv"
OUTPUT_BLEND_PATH = DATA_DIR / "submission_v14_blend_75_25.csv"
OUTPUT_DIR = DATA_DIR / "features_v11_shap" / "v14_archetype_results"


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


def build_seasonal_archetype_features(train_df: pd.DataFrame, target_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cutoff öncesi verileri kullanarak sızıntısız arketip ve mevsim katsayılarını hesaplar."""
    past_df = train_df[train_df["tarih"] <= cutoff_date].copy()

    # 1. Monthly Network Index
    m_totals = past_df.groupby(past_df["tarih"].dt.month)["tuketim"].sum()
    m_avg = m_totals.mean() if len(m_totals) > 0 else 1.0
    m_index = (m_totals / m_avg).to_dict()
    default_m_index = {1: 0.794, 2: 0.782, 3: 0.694, 4: 0.651, 5: 0.564, 6: 1.000, 7: 1.700, 8: 1.442, 9: 1.174, 10: 0.734, 11: 1.068, 12: 1.400}
    for m in range(1, 13):
        if m not in m_index:
            m_index[m] = default_m_index[m]

    # 2. Dynamic Lookback: Find available past summer (July + August)
    past_july = past_df[past_df["tarih"].dt.month == 7].groupby("tanim")["tuketim"].mean()
    past_aug = past_df[past_df["tarih"].dt.month == 8].groupby("tanim")["tuketim"].mean()
    past_winter = past_df[past_df["tarih"].dt.month.isin([1, 2, 3])].groupby("tanim")["tuketim"].mean()

    guc_map = past_df.groupby("tanim")["guc"].first().to_dict()

    fac_summer_surge = {}
    if len(past_july) > 0 and len(past_winter) > 0:
        july_mean = past_july.mean()
        aug_mean = past_aug.mean() if len(past_aug) > 0 else july_mean
        aug_to_july_ratio = (july_mean / max(1.0, aug_mean)) if aug_mean > 0 else 1.18

        all_summer_facs = set(past_winter.index)
        for t in all_summer_facs:
            w_val = past_winter.get(t, np.nan)
            j_val = past_july.get(t, np.nan)
            a_val = past_aug.get(t, np.nan)
            g_val = guc_map.get(t, 630.0)
            prior_c = max(5.0, g_val * 0.10)

            if not np.isnan(j_val) and not np.isnan(a_val):
                denoised_summer = 0.55 * j_val + 0.45 * (a_val * aug_to_july_ratio)
            elif not np.isnan(j_val):
                denoised_summer = j_val
            elif not np.isnan(a_val):
                denoised_summer = a_val * aug_to_july_ratio
            else:
                denoised_summer = np.nan

            if not np.isnan(w_val) and not np.isnan(denoised_summer):
                surge = (denoised_summer + prior_c) / (w_val + prior_c)
                fac_summer_surge[t] = float(np.clip(surge, 0.1, 10.0))

    # Hierarchical Fallbacks for Summer Surge
    surge_df = past_df[["tanim", "ilce", "guc_bin", "il"]].drop_duplicates("tanim").set_index("tanim")
    surge_df["surge"] = surge_df.index.map(fac_summer_surge)
    global_surge = float(surge_df["surge"].dropna().median()) if len(surge_df["surge"].dropna()) > 0 else 1.40
    ilce_surge = surge_df.groupby("ilce")["surge"].median().to_dict()
    guc_surge = surge_df.groupby("guc_bin")["surge"].median().to_dict()

    # 3. K-Means Continuous Archetype Embeddings (100% Vectorized)
    profile_pivot = past_df.pivot_table(index="tanim", columns=past_df["tarih"].dt.month, values="tuketim", aggfunc="mean")
    profile_norm = profile_pivot.div(profile_pivot.mean(axis=1), axis=0).fillna(1.0)

    for m in range(1, 13):
        if m not in profile_norm.columns:
            profile_norm[m] = default_m_index[m]
    profile_norm = profile_norm[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(profile_norm.values)
    centers = kmeans.cluster_centers_  # 3 x 12

    # Vectorized computation of soft probabilities across all facilities in train & test
    all_known_facs = list(set(train_df["tanim"].unique()).union(set(target_df["tanim"].unique())))
    synth_vectors = []
    for t in all_known_facs:
        if t in profile_norm.index:
            v = profile_norm.loc[t].values
        else:
            s_val = fac_summer_surge.get(t, global_surge)
            v = np.array([1.0, 1.0, 1.0, 0.85, 0.70, 1.0 + 0.35 * (s_val - 1.0), 1.1 + 0.85 * (s_val - 1.0), 1.05 + 0.70 * (s_val - 1.0), 0.95, 0.85, 1.0, 1.1])
            v = v / max(0.1, v.mean())
        synth_vectors.append(v)

    synth_mat = np.array(synth_vectors, dtype=np.float32)  # (N_fac, 12)
    diffs = synth_mat[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # (N_fac, 3)
    exp_neg = np.exp(-dists)
    probs = exp_neg / exp_neg.sum(axis=1, keepdims=True)

    prob_dict_0 = dict(zip(all_known_facs, probs[:, 0]))
    prob_dict_1 = dict(zip(all_known_facs, probs[:, 1]))
    prob_dict_2 = dict(zip(all_known_facs, probs[:, 2]))

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

        # Monthly Network Index
        df["monthly_network_index"] = df["month"].map(m_index).fillna(1.0).astype(np.float32)

        # Vectorized Regularized Surge
        direct_surge = df["tanim"].map(fac_summer_surge)
        fallback_ilce = df["ilce"].map(ilce_surge)
        fallback_guc = df["guc_bin"].map(guc_surge)
        df["facility_summer_surge"] = direct_surge.fillna(fallback_ilce).fillna(fallback_guc).fillna(global_surge).astype(np.float32)

        # Fast Vectorized Archetype Soft Probabilities
        df["arch_prob_0"] = df["tanim"].map(prob_dict_0).fillna(0.33).astype(np.float32)
        df["arch_prob_1"] = df["tanim"].map(prob_dict_1).fillna(0.33).astype(np.float32)
        df["arch_prob_2"] = df["tanim"].map(prob_dict_2).fillna(0.33).astype(np.float32)

        # Baseline Recency Level
        df["fac_recent_28"] = df["tanim"].map(fac_recent_28)
        df["fac_mean_all"] = df["tanim"].map(fac_mean_all)
        df["fac_level"] = df["fac_recent_28"].fillna(df["fac_mean_all"]).fillna(df["guc"] * 2.5).astype(np.float32)
        df["log_fac_level"] = np.log1p(df["fac_level"]).astype(np.float32)

        # Seasonal Scaled Baseline with Archetype Blending
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

        # 1-Year Lags
        keys = list(zip(df["tanim"].values, df["tarih"].values))
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

        # Integer encodings for fast CatBoost/LGBM
        df["il_code"] = df["il"].astype("category").cat.codes.astype(np.int32)
        df["ilce_code"] = df["ilce"].astype("category").cat.codes.astype(np.int32)
        df["bolge_code"] = df["bolge"].astype("category").cat.codes.astype(np.int32)
        df["guc_bin_code"] = df["guc_bin"].astype("category").cat.codes.astype(np.int32)

        if "id" not in df.columns:
            df["id"] = df["tanim"] + "_" + df["tarih"].dt.strftime("%Y-%m-%d")

        return df

    target_out = transform_df(target_df)
    past_out = transform_df(past_df)
    return past_out, target_out


def run_v14_pipeline():
    logger.info("=" * 75)
    logger.info("STARTING V14 SEASONAL ARCHETYPES & MULTI-MODEL ENSEMBLE PIPELINE")
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

    features_model = [
        "month", "day_of_week", "day_of_year", "day", "doy_sin", "doy_cos",
        "is_weekend", "is_summer", "is_june_july",
        "guc", "log_guc", "log_guc_x_summer",
        "monthly_network_index", "facility_summer_surge",
        "arch_prob_0", "arch_prob_1", "arch_prob_2",
        "fac_level", "log_fac_level", "seasonal_baseline", "log_seasonal_baseline",
        "has_annual_lag", "annual_lag_val", "log_annual_lag", "days_since_last_seen", "is_cold",
        "il_code", "ilce_code", "bolge_code", "guc_bin_code"
    ]

    oof_preds_list = []

    logger.info("\n--- EXECUTING ROLLING ORIGIN CROSS VALIDATION ---")
    for fold_name, cutoff, val_start, val_end in folds:
        logger.info(f"\nProcessing Fold: {fold_name} (Cutoff: {cutoff.strftime('%Y-%m-%d')})")
        val_raw = raw_train[(raw_train['tarih'] >= val_start) & (raw_train['tarih'] <= val_end)].copy()
        past_tr_feat, val_data = build_seasonal_archetype_features(raw_train, val_raw, cutoff)

        y_train_res = np.log1p(past_tr_feat["tuketim"].values) - np.log1p(past_tr_feat["seasonal_baseline"].values)

        # 1. Fast LightGBM
        m_lgb = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        m_lgb.fit(past_tr_feat[features_model], y_train_res)
        pred_res_lgb = m_lgb.predict(val_data[features_model])

        # 2. Fast Multi-threaded CatBoost
        m_cb = CatBoostRegressor(
            iterations=350,
            learning_rate=0.06,
            depth=6,
            loss_function="RMSE",
            thread_count=-1,
            random_seed=42,
            verbose=False,
        )
        m_cb.fit(past_tr_feat[features_model], y_train_res, verbose=False)
        pred_res_cb = m_cb.predict(val_data[features_model])

        # Ensemble Residual (50% LGB + 50% CB)
        pred_res = 0.50 * pred_res_lgb + 0.50 * pred_res_cb

        # Final prediction
        pred_val = np.maximum(0.0, np.expm1(np.log1p(val_data["seasonal_baseline"].values) + pred_res))
        pred_val = np.clip(pred_val, 0.0, 36.0 * (val_data["guc"].values + 1.0))

        val_data["pred_v14"] = pred_val
        val_rmsle = calculate_rmsle(val_data["tuketim"].values, pred_val)
        logger.info(f"✓ {fold_name} Total RMSLE (LGB+CB Ensemble): {val_rmsle:.5f} (N={len(val_data):,d})")

        for m in sorted(val_data["month"].unique()):
            v_m = val_data[val_data["month"] == m]
            r_m = calculate_rmsle(v_m["tuketim"].values, v_m["pred_v14"].values)
            logger.info(f"   -> Month {m:2d}: RMSLE = {r_m:.5f} (N={len(v_m):,d})")

        oof_preds_list.append(val_data[["id", "tarih", "tanim", "tuketim", "pred_v14", "month", "has_annual_lag", "is_cold"]])

    oof_df = pd.concat(oof_preds_list, ignore_index=True)
    pooled_rmsle = calculate_rmsle(oof_df["tuketim"].values, oof_df["pred_v14"].values)

    logger.info("\n" + "=" * 75)
    logger.info(f"★ POOLED 3-FOLD OOF RMSLE (V14 SEASONAL ARCHETYPES): {pooled_rmsle:.5f}")
    logger.info("=" * 75)

    # -------------------------------------------------------------------------
    # 3. FULL RETRAINING ON 100% DATA & SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    logger.info("\nRetraining on 100% data (Cutoff = 2026-03-31) for Test Submission...")
    cutoff_full = pd.Timestamp("2026-03-31")
    full_tr_feat, test_feat = build_seasonal_archetype_features(raw_train, raw_test, cutoff_full)

    y_full_res = np.log1p(full_tr_feat["tuketim"].values) - np.log1p(full_tr_feat["seasonal_baseline"].values)

    # Train Full LightGBM
    full_lgb = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )
    full_lgb.fit(full_tr_feat[features_model], y_full_res)
    pred_test_lgb = full_lgb.predict(test_feat[features_model])

    # Train Full CatBoost
    full_cb = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        thread_count=-1,
        random_seed=42,
        verbose=False,
    )
    full_cb.fit(full_tr_feat[features_model], y_full_res, verbose=False)
    pred_test_cb = full_cb.predict(test_feat[features_model])

    # Ensemble Residual
    test_res_pred = 0.50 * pred_test_lgb + 0.50 * pred_test_cb
    test_final_pred = np.maximum(0.0, np.expm1(np.log1p(test_feat["seasonal_baseline"].values) + test_res_pred))

    # Physical ceiling check
    ceiling = 36.0 * (test_feat["guc"].values + 1.0)
    ceiling_hits = (test_final_pred > ceiling).sum()
    logger.info(f"Physical ceiling hits on test: {ceiling_hits:,d} / {len(test_final_pred):,d} ({(ceiling_hits/len(test_final_pred))*100:.2f}%)")
    test_final_pred = np.clip(test_final_pred, 0.0, ceiling)

    # Save V14 Standalone Submission
    sub_v14 = pd.DataFrame({"id": test_feat["id"], "tuketim": test_final_pred})
    sub_v14.to_csv(OUTPUT_SUB_PATH, index=False)
    sha_v14 = get_sha256(OUTPUT_SUB_PATH)

    # Save 75% V14 + 25% V8R Risk-Hedged Blend
    logger.info(f"\nCreating Risk-Hedged Blend: 0.75 * V14 + 0.25 * V8R...")
    v8r_sub = pd.read_csv(BASE_SUB_PATH)
    blend_preds = 0.75 * test_final_pred + 0.25 * v8r_sub["tuketim"].values
    sub_blend = pd.DataFrame({"id": test_feat["id"], "tuketim": blend_preds})
    sub_blend.to_csv(OUTPUT_BLEND_PATH, index=False)
    sha_blend = get_sha256(OUTPUT_BLEND_PATH)

    logger.info("=" * 75)
    logger.info("✓ V14 PIPELINE SUCCESSFULLY COMPLETED!")
    logger.info(f"✓ V14 Standalone Output : {OUTPUT_SUB_PATH} (SHA: {sha_v14})")
    logger.info(f"✓ V14 Risk-Hedged Blend : {OUTPUT_BLEND_PATH} (SHA: {sha_blend})")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_v14_pipeline()
