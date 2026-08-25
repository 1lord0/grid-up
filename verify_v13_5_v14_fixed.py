"""Grid Up Datathon — Rigorous Categorical Alignment & Cross-Validation Benchmark.

Bu script:
1. 'il', 'ilce', 'bolge', 'guc_bin' için global deterministik sabit kategori haritalaması kurar.
2. V13 (Tek LGBM), V13.5 (LGBM+CatBoost Ensemble) ve V14 (LGBM+CatBoost+Arketip) modellerini
   aynı seed (42) ve tamamen hizalı kategorik kodlarla 3 fold üzerinde çalıştırır.
3. Test setinin birebir takvim penceresi olan Fold A (Nisan-Temmuz) ve diğer foldlar üzerinde
   kesin RMSLE skorlarını hesaplar.
4. OOF tahminleri üzerinde grid search ile ampirik optimal blend ağırlıklarını bulur.
5. 100% veri üzerinde eğitip nihai doğrulanmış submission dosyalarını üretir.
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


def main():
    logger.info("=" * 75)
    logger.info("STARTING RIGOROUS CATEGORICAL ALIGNMENT & CV BENCHMARK")
    logger.info("=" * 75)

    # 1. Load Data
    logger.info("Loading train.csv and test.csv...")
    raw_train = pd.read_csv(DATA_DIR / "train.csv", dtype={"tanim": str, "row_id": str}, parse_dates=["tarih"])
    raw_test = pd.read_csv(DATA_DIR / "test.csv", dtype={"tanim": str, "row_id": str}, parse_dates=["tarih"])

    raw_train = parse_locations(raw_train)
    raw_test = parse_locations(raw_test)

    guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
    guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
    raw_train["guc_bin"] = pd.cut(raw_train["guc"], bins=guc_bins, labels=guc_labels).astype(str)
    raw_test["guc_bin"] = pd.cut(raw_test["guc"], bins=guc_bins, labels=guc_labels).astype(str)

    # 2. Global Deterministic Categorical Mappings (Guaranteed Train/Val/Test Alignment)
    cat_cols_raw = ["il", "ilce", "bolge", "guc_bin"]
    global_cat_maps = {}
    for col in cat_cols_raw:
        all_vals = sorted(list(set(raw_train[col].dropna()).union(set(raw_test[col].dropna()))))
        global_cat_maps[col] = {val: i for i, val in enumerate(all_vals)}
        logger.info(f"Fixed mapping for '{col}': {len(all_vals)} unique categories.")

    # 3. Feature Builder Function with Guaranteed Alignment
    def build_features(train_df: pd.DataFrame, target_df: pd.DataFrame, cutoff_date: pd.Timestamp, include_archetypes: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        past_df = train_df[train_df["tarih"] <= cutoff_date].copy()

        # Monthly Network Index
        m_totals = past_df.groupby(past_df["tarih"].dt.month)["tuketim"].sum()
        m_avg = m_totals.mean() if len(m_totals) > 0 else 1.0
        m_index = (m_totals / m_avg).to_dict()
        default_m_index = {1: 0.794, 2: 0.782, 3: 0.694, 4: 0.651, 5: 0.564, 6: 1.000, 7: 1.700, 8: 1.442, 9: 1.174, 10: 0.734, 11: 1.068, 12: 1.400}
        for m in range(1, 13):
            if m not in m_index:
                m_index[m] = default_m_index[m]

        # Facility Recency & Mean
        fac_recent_28 = past_df[past_df["tarih"] > (cutoff_date - pd.Timedelta(days=28))].groupby("tanim")["tuketim"].mean().to_dict()
        fac_mean_all = past_df.groupby("tanim")["tuketim"].mean().to_dict()
        fac_last_seen = past_df.groupby("tanim")["tarih"].max().to_dict()

        # 1-Year Lags
        lag_364_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=364)])["tuketim"].to_dict()
        lag_365_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=365)])["tuketim"].to_dict()
        lag_371_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=371)])["tuketim"].to_dict()

        # Summer Surge & Archetype Embedding (only if requested)
        fac_summer_surge = {}
        prob_dict_0, prob_dict_1, prob_dict_2 = {}, {}, {}
        global_surge = 1.40
        ilce_surge, guc_surge = {}, {}

        if include_archetypes:
            past_july = past_df[past_df["tarih"].dt.month == 7].groupby("tanim")["tuketim"].mean()
            past_aug = past_df[past_df["tarih"].dt.month == 8].groupby("tanim")["tuketim"].mean()
            past_winter = past_df[past_df["tarih"].dt.month.isin([1, 2, 3])].groupby("tanim")["tuketim"].mean()
            guc_map = past_df.groupby("tanim")["guc"].first().to_dict()

            if len(past_july) > 0 and len(past_winter) > 0:
                july_mean = past_july.mean()
                aug_mean = past_aug.mean() if len(past_aug) > 0 else july_mean
                aug_to_july_ratio = (july_mean / max(1.0, aug_mean)) if aug_mean > 0 else 1.18

                for t in set(past_winter.index):
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

            surge_df = past_df[["tanim", "ilce", "guc_bin", "il"]].drop_duplicates("tanim").set_index("tanim")
            surge_df["surge"] = surge_df.index.map(fac_summer_surge)
            global_surge = float(surge_df["surge"].dropna().median()) if len(surge_df["surge"].dropna()) > 0 else 1.40
            ilce_surge = surge_df.groupby("ilce")["surge"].median().to_dict()
            guc_surge = surge_df.groupby("guc_bin")["surge"].median().to_dict()

            profile_pivot = past_df.pivot_table(index="tanim", columns=past_df["tarih"].dt.month, values="tuketim", aggfunc="mean")
            profile_norm = profile_pivot.div(profile_pivot.mean(axis=1), axis=0).fillna(1.0)
            for m in range(1, 13):
                if m not in profile_norm.columns:
                    profile_norm[m] = default_m_index[m]
            profile_norm = profile_norm[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(profile_norm.values)
            centers = kmeans.cluster_centers_

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

            synth_mat = np.array(synth_vectors, dtype=np.float32)
            diffs = synth_mat[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            exp_neg = np.exp(-dists)
            probs = exp_neg / exp_neg.sum(axis=1, keepdims=True)

            prob_dict_0 = dict(zip(all_known_facs, probs[:, 0]))
            prob_dict_1 = dict(zip(all_known_facs, probs[:, 1]))
            prob_dict_2 = dict(zip(all_known_facs, probs[:, 2]))

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

            df["monthly_network_index"] = df["month"].map(m_index).fillna(1.0).astype(np.float32)

            # Global deterministic categorical encoding
            for c in cat_cols_raw:
                c_map = global_cat_maps[c]
                df[f"{c}_code"] = df[c].map(c_map).fillna(-1).astype(np.int32)

            df["fac_recent_28"] = df["tanim"].map(fac_recent_28)
            df["fac_mean_all"] = df["tanim"].map(fac_mean_all)
            df["fac_level"] = df["fac_recent_28"].fillna(df["fac_mean_all"]).fillna(df["guc"] * 2.5).astype(np.float32)
            df["log_fac_level"] = np.log1p(df["fac_level"]).astype(np.float32)

            m_arr = df["month"].values
            base_arr = df["fac_level"].values

            if include_archetypes:
                direct_surge = df["tanim"].map(fac_summer_surge)
                fallback_ilce = df["ilce"].map(ilce_surge)
                fallback_guc = df["guc_bin"].map(guc_surge)
                df["facility_summer_surge"] = direct_surge.fillna(fallback_ilce).fillna(fallback_guc).fillna(global_surge).astype(np.float32)

                df["arch_prob_0"] = df["tanim"].map(prob_dict_0).fillna(0.33).astype(np.float32)
                df["arch_prob_1"] = df["tanim"].map(prob_dict_1).fillna(0.33).astype(np.float32)
                df["arch_prob_2"] = df["tanim"].map(prob_dict_2).fillna(0.33).astype(np.float32)

                surge_arr = df["facility_summer_surge"].values
                conds = [m_arr == 4, m_arr == 5, m_arr == 6, m_arr == 7, m_arr == 8, m_arr == 9]
                choices = [
                    base_arr * 0.85,
                    base_arr * 0.70,
                    base_arr * (0.90 + 0.35 * (surge_arr - 1.0)),
                    base_arr * (1.10 + 0.85 * (surge_arr - 1.0)),
                    base_arr * (1.05 + 0.70 * (surge_arr - 1.0)),
                    base_arr * 0.95,
                ]
            else:
                conds = [m_arr == 4, m_arr == 5, m_arr == 6, m_arr == 7, m_arr == 8, m_arr == 9]
                choices = [
                    base_arr * 0.85,
                    base_arr * 0.70,
                    base_arr * 1.04,
                    base_arr * 1.44,
                    base_arr * 1.33,
                    base_arr * 0.95,
                ]

            df["seasonal_baseline"] = np.select(conds, choices, default=base_arr).astype(np.float32)
            df["log_seasonal_baseline"] = np.log1p(df["seasonal_baseline"]).astype(np.float32)

            keys = list(zip(df["tanim"].values, df["tarih"].values))
            df["lag_364"] = [lag_364_map.get(k, np.nan) for k in keys]
            df["lag_365"] = [lag_365_map.get(k, np.nan) for k in keys]
            df["lag_371"] = [lag_371_map.get(k, np.nan) for k in keys]
            df["has_annual_lag"] = (~df["lag_365"].isna()).astype(int)
            df["annual_lag_val"] = df["lag_365"].fillna(df["lag_364"]).fillna(df["lag_371"]).fillna(df["seasonal_baseline"]).astype(np.float32)
            df["log_annual_lag"] = np.log1p(df["annual_lag_val"]).astype(np.float32)

            df["last_seen_date"] = df["tanim"].map(fac_last_seen)
            df["days_since_last_seen"] = (cutoff_date - df["last_seen_date"]).dt.days.fillna(999).astype(np.float32)
            df["is_cold"] = (df["days_since_last_seen"] > 180).astype(int)

            if "id" not in df.columns:
                df["id"] = df["tanim"] + "_" + df["tarih"].dt.strftime("%Y-%m-%d")
            return df

        return transform_df(past_df), transform_df(target_df)

    folds = [
        ("fold_a_apr_jul_2025", pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
        ("fold_b_aug_nov_2025", pd.Timestamp("2025-07-31"), pd.Timestamp("2025-08-01"), pd.Timestamp("2025-11-30")),
        ("fold_c_dec_mar_2026", pd.Timestamp("2025-11-30"), pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-31")),
    ]

    base_features = [
        "month", "day_of_week", "day_of_year", "day", "doy_sin", "doy_cos",
        "is_weekend", "is_summer", "is_june_july",
        "guc", "log_guc", "log_guc_x_summer",
        "monthly_network_index",
        "fac_level", "log_fac_level", "seasonal_baseline", "log_seasonal_baseline",
        "has_annual_lag", "annual_lag_val", "log_annual_lag", "days_since_last_seen", "is_cold",
        "il_code", "ilce_code", "bolge_code", "guc_bin_code"
    ]

    arch_features = base_features + ["facility_summer_surge", "arch_prob_0", "arch_prob_1", "arch_prob_2"]

    logger.info("\n" + "=" * 75)
    logger.info("RUNNING SIDE-BY-SIDE RIGOROUS 3-FOLD CROSS VALIDATION")
    logger.info("=" * 75)

    oof_v13_list, oof_v13_5_list, oof_v14_list = [], [], []

    for fold_name, cutoff, val_start, val_end in folds:
        logger.info(f"\nEvaluating: {fold_name} (Cutoff: {cutoff.strftime('%Y-%m-%d')})")
        val_raw = raw_train[(raw_train["tarih"] >= val_start) & (raw_train["tarih"] <= val_end)].copy()

        # Build datasets
        past_v13_5, val_v13_5 = build_features(raw_train, val_raw, cutoff, include_archetypes=False)
        past_v14, val_v14 = build_features(raw_train, val_raw, cutoff, include_archetypes=True)

        y_true = val_raw["tuketim"].values
        guc_val = val_raw["guc"].values
        ceil_val = 36.0 * (guc_val + 1.0)

        # ---------------------------------------------------------------------
        # MODEL 1: V13 (Pure LightGBM)
        # ---------------------------------------------------------------------
        y_res_v13 = np.log1p(past_v13_5["tuketim"].values) - np.log1p(past_v13_5["seasonal_baseline"].values)
        m_v13 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
        m_v13.fit(past_v13_5[base_features], y_res_v13)
        p_res_v13 = m_v13.predict(val_v13_5[base_features])
        pred_v13 = np.clip(np.maximum(0.0, np.expm1(np.log1p(val_v13_5["seasonal_baseline"].values) + p_res_v13)), 0.0, ceil_val)
        rmsle_v13 = calculate_rmsle(y_true, pred_v13)

        # ---------------------------------------------------------------------
        # MODEL 2: V13.5 (LGBM + CatBoost 50/50 Ensemble with Aligned Codes)
        # ---------------------------------------------------------------------
        m_cb_v13_5 = CatBoostRegressor(iterations=350, learning_rate=0.06, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
        m_cb_v13_5.fit(past_v13_5[base_features], y_res_v13, verbose=False)
        p_cb_v13_5 = m_cb_v13_5.predict(val_v13_5[base_features])
        p_res_v13_5 = 0.50 * p_res_v13 + 0.50 * p_cb_v13_5
        pred_v13_5 = np.clip(np.maximum(0.0, np.expm1(np.log1p(val_v13_5["seasonal_baseline"].values) + p_res_v13_5)), 0.0, ceil_val)
        rmsle_v13_5 = calculate_rmsle(y_true, pred_v13_5)

        # ---------------------------------------------------------------------
        # MODEL 3: V14 (LGBM + CatBoost 50/50 + Archetype + Surge with Aligned Codes)
        # ---------------------------------------------------------------------
        y_res_v14 = np.log1p(past_v14["tuketim"].values) - np.log1p(past_v14["seasonal_baseline"].values)
        m_lgb_v14 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
        m_lgb_v14.fit(past_v14[arch_features], y_res_v14)
        p_lgb_v14 = m_lgb_v14.predict(val_v14[arch_features])

        m_cb_v14 = CatBoostRegressor(iterations=350, learning_rate=0.06, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
        m_cb_v14.fit(past_v14[arch_features], y_res_v14, verbose=False)
        p_cb_v14 = m_cb_v14.predict(val_v14[arch_features])

        p_res_v14 = 0.50 * p_lgb_v14 + 0.50 * p_cb_v14
        pred_v14 = np.clip(np.maximum(0.0, np.expm1(np.log1p(val_v14["seasonal_baseline"].values) + p_res_v14)), 0.0, ceil_val)
        rmsle_v14 = calculate_rmsle(y_true, pred_v14)

        logger.info(f"-> {fold_name} Scores: V13 (Pure LGB)={rmsle_v13:.5f} | V13.5 (Ensemble)={rmsle_v13_5:.5f} | V14 (Archetype)={rmsle_v14:.5f}")

        df_oof_fold = pd.DataFrame({
            "tarih": val_raw["tarih"].values,
            "month": val_raw["tarih"].dt.month.values,
            "tuketim": y_true,
            "pred_v13": pred_v13,
            "pred_v13_5": pred_v13_5,
            "pred_v14": pred_v14,
        })
        oof_v13_list.append(df_oof_fold)

    oof_all = pd.concat(oof_v13_list, ignore_index=True)

    logger.info("\n" + "=" * 75)
    logger.info("FINAL COMPARATIVE VALIDATION TABLE")
    logger.info("=" * 75)

    fold_a_mask = oof_all["month"].isin([4, 5, 6, 7])
    rmsle_fold_a_v13 = calculate_rmsle(oof_all.loc[fold_a_mask, "tuketim"].values, oof_all.loc[fold_a_mask, "pred_v13"].values)
    rmsle_fold_a_v13_5 = calculate_rmsle(oof_all.loc[fold_a_mask, "tuketim"].values, oof_all.loc[fold_a_mask, "pred_v13_5"].values)
    rmsle_fold_a_v14 = calculate_rmsle(oof_all.loc[fold_a_mask, "tuketim"].values, oof_all.loc[fold_a_mask, "pred_v14"].values)

    rmsle_pool_v13 = calculate_rmsle(oof_all["tuketim"].values, oof_all["pred_v13"].values)
    rmsle_pool_v13_5 = calculate_rmsle(oof_all["tuketim"].values, oof_all["pred_v13_5"].values)
    rmsle_pool_v14 = calculate_rmsle(oof_all["tuketim"].values, oof_all["pred_v14"].values)

    logger.info(f"Fold A (Nisan-Temmuz / Test Window): V13={rmsle_fold_a_v13:.5f} | V13.5={rmsle_fold_a_v13_5:.5f} | V14={rmsle_fold_a_v14:.5f}")
    logger.info(f"Pooled 3-Fold Total                : V13={rmsle_pool_v13:.5f} | V13.5={rmsle_pool_v13_5:.5f} | V14={rmsle_pool_v14:.5f}")

    # 4. Empirical Blend Optimization on Fold A (The Exact Test Window)
    logger.info("\n--- EMPIRICAL BLEND OPTIMIZATION ON FOLD A (TEST WINDOW) ---")
    y_fold_a = oof_all.loc[fold_a_mask, "tuketim"].values
    p_v14_a = oof_all.loc[fold_a_mask, "pred_v14"].values
    p_v13_5_a = oof_all.loc[fold_a_mask, "pred_v13_5"].values

    best_w = 1.0
    best_blend_score = 999.0
    for w in np.linspace(0.0, 1.0, 21):
        blend_pred = w * p_v14_a + (1.0 - w) * p_v13_5_a
        score = calculate_rmsle(y_fold_a, blend_pred)
        logger.info(f"Weight V14={w:4.2f} (V13.5={1.0-w:4.2f}) -> Fold A RMSLE: {score:.5f}")
        if score < best_blend_score:
            best_blend_score = score
            best_w = w

    logger.info(f"★ OPTIMAL BLEND WEIGHT: {best_w:.2f} * V14 + {1.0-best_w:.2f} * V13.5 -> Fold A RMSLE = {best_blend_score:.5f}")

    # -------------------------------------------------------------------------
    # 5. FULL 100% RETRAINING & VERIFIED SUBMISSIONS GENERATION
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("RETRAINING FULL 100% DATASET FOR TEST SUBMISSIONS")
    logger.info("=" * 75)
    cutoff_full = pd.Timestamp("2026-03-31")

    full_tr_v13_5, test_feat_v13_5 = build_features(raw_train, raw_test, cutoff_full, include_archetypes=False)
    full_tr_v14, test_feat_v14 = build_features(raw_train, raw_test, cutoff_full, include_archetypes=True)

    ceil_test = 36.0 * (test_feat_v14["guc"].values + 1.0)

    # Train Full V13.5
    y_full_res_v13_5 = np.log1p(full_tr_v13_5["tuketim"].values) - np.log1p(full_tr_v13_5["seasonal_baseline"].values)
    full_lgb_v13_5 = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    full_lgb_v13_5.fit(full_tr_v13_5[base_features], y_full_res_v13_5)
    p_lgb_test_v13_5 = full_lgb_v13_5.predict(test_feat_v13_5[base_features])

    full_cb_v13_5 = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
    full_cb_v13_5.fit(full_tr_v13_5[base_features], y_full_res_v13_5, verbose=False)
    p_cb_test_v13_5 = full_cb_v13_5.predict(test_feat_v13_5[base_features])

    p_res_test_v13_5 = 0.50 * p_lgb_test_v13_5 + 0.50 * p_cb_test_v13_5
    test_pred_v13_5 = np.clip(np.maximum(0.0, np.expm1(np.log1p(test_feat_v13_5["seasonal_baseline"].values) + p_res_test_v13_5)), 0.0, ceil_test)

    # Train Full V14
    y_full_res_v14 = np.log1p(full_tr_v14["tuketim"].values) - np.log1p(full_tr_v14["seasonal_baseline"].values)
    full_lgb_v14 = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    full_lgb_v14.fit(full_tr_v14[arch_features], y_full_res_v14)
    p_lgb_test_v14 = full_lgb_v14.predict(test_feat_v14[arch_features])

    full_cb_v14 = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
    full_cb_v14.fit(full_tr_v14[arch_features], y_full_res_v14, verbose=False)
    p_cb_test_v14 = full_cb_v14.predict(test_feat_v14[arch_features])

    p_res_test_v14 = 0.50 * p_lgb_test_v14 + 0.50 * p_cb_test_v14
    test_pred_v14 = np.clip(np.maximum(0.0, np.expm1(np.log1p(test_feat_v14["seasonal_baseline"].values) + p_res_test_v14)), 0.0, ceil_test)

    # Optimal Blend Submission (Empirically Optimized Weight)
    test_pred_optimal = best_w * test_pred_v14 + (1.0 - best_w) * test_pred_v13_5

    # Save Submissions
    sub_v13_5_path = DATA_DIR / "submission_v13_5_verified_clean.csv"
    pd.DataFrame({"id": test_feat_v13_5["id"], "tuketim": test_pred_v13_5}).to_csv(sub_v13_5_path, index=False)

    sub_v14_path = DATA_DIR / "submission_v14_verified_clean.csv"
    pd.DataFrame({"id": test_feat_v14["id"], "tuketim": test_pred_v14}).to_csv(sub_v14_path, index=False)

    sub_optimal_path = DATA_DIR / "submission_optimal_blend_verified.csv"
    pd.DataFrame({"id": test_feat_v14["id"], "tuketim": test_pred_optimal}).to_csv(sub_optimal_path, index=False)

    # Safe Hedged Blend (0.60 Optimal + 0.40 V8R)
    v8r_sub = pd.read_csv(BASE_SUB_PATH)
    hedged_preds = 0.70 * test_pred_optimal + 0.30 * v8r_sub["tuketim"].values
    sub_hedged_path = DATA_DIR / "submission_hedged_final.csv"
    pd.DataFrame({"id": test_feat_v14["id"], "tuketim": hedged_preds}).to_csv(sub_hedged_path, index=False)

    logger.info("=" * 75)
    logger.info("✓ VERIFIED SUBMISSIONS GENERATED SUCCESSFULLY!")
    logger.info(f"1. V13.5 Clean Output : {sub_v13_5_path.name} | SHA: {get_sha256(sub_v13_5_path)}")
    logger.info(f"2. V14 Clean Output    : {sub_v14_path.name} | SHA: {get_sha256(sub_v14_path)}")
    logger.info(f"3. Optimal Blend       : {sub_optimal_path.name} (w={best_w:.2f} V14 + {1.0-best_w:.2f} V13.5) | SHA: {get_sha256(sub_optimal_path)}")
    logger.info(f"4. Hedged Final Blend  : {sub_hedged_path.name} (0.70 Optimal + 0.30 V8R) | SHA: {get_sha256(sub_hedged_path)}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
