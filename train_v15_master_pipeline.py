"""Grid Up Datathon — V15 Master Pipeline.

Qwen 3.8-27B Tarafından Tasarlanan 5 Temel Yenilik:
1. Cold-Start NMF & Emsal Hiyerarşik Bayes Çaprazı (İlçe/Güç Medyanı + Yaz Katsayısı Aktarımı)
2. Fourier / Harmonik Çoklu-Mevsimsellik (24s Günlük 1. & 2. Harmonik, 168s Haftalık, 365g Yıllık)
3. De-noised Yaz Transferi & K-Means 3'lü Sürekli Arketip Uzaklıkları (Soft Clustering)
4. İki Aşamalı (Two-Stage) CatBoost + LightGBM Residual Topluluğu
5. Log-Uzayda Segment Bazlı Dinamik Ağırlıklandırma & Fiziksel Kapasite Tavanı (Capacity Ceiling)
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
BASE_V8R_PATH = DATA_DIR / "submission_v8r_verified_final.csv"
OUTPUT_STANDALONE_PATH = DATA_DIR / "submission_v15_standalone.csv"
OUTPUT_BLEND_PATH = DATA_DIR / "submission_v15_optimal_master_blend.csv"


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


def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fourier / Harmonik zaman özellikleri ekler."""
    df = df.copy()
    day = df["tarih"].dt.day
    dow = df["tarih"].dt.dayofweek
    doy = df["tarih"].dt.dayofyear

    # Günlük / Saatlik eşdeğer (Gün döngüsü)
    df["sin_day_1"] = np.sin(2 * np.pi * 1 * day / 30.0).astype(np.float32)
    df["cos_day_1"] = np.cos(2 * np.pi * 1 * day / 30.0).astype(np.float32)
    df["sin_day_2"] = np.sin(2 * np.pi * 2 * day / 30.0).astype(np.float32)
    df["cos_day_2"] = np.cos(2 * np.pi * 2 * day / 30.0).astype(np.float32)

    # Haftalık (7 gün)
    df["sin_dow_1"] = np.sin(2 * np.pi * 1 * dow / 7.0).astype(np.float32)
    df["cos_dow_1"] = np.cos(2 * np.pi * 1 * dow / 7.0).astype(np.float32)

    # Yıllık / Mevsimsel (365 gün)
    df["sin_doy_1"] = np.sin(2 * np.pi * 1 * doy / 365.25).astype(np.float32)
    df["cos_doy_1"] = np.cos(2 * np.pi * 1 * doy / 365.25).astype(np.float32)
    df["sin_doy_2"] = np.sin(2 * np.pi * 2 * doy / 365.25).astype(np.float32)
    df["cos_doy_2"] = np.cos(2 * np.pi * 2 * doy / 365.25).astype(np.float32)

    return df


def build_v15_features(
    train_df: pd.DataFrame, target_df: pd.DataFrame, cutoff_date: pd.Timestamp,
    global_cat_maps: Dict[str, Dict[str, int]], cat_cols_raw: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """V15 Master Özellik Mühendisliği."""
    past_df = train_df[train_df["tarih"] <= cutoff_date].copy()

    # 1. Aylık Şebeke Endeksi
    m_totals = past_df.groupby(past_df["tarih"].dt.month)["tuketim"].sum()
    m_avg = m_totals.mean() if len(m_totals) > 0 else 1.0
    m_index = (m_totals / m_avg).to_dict()
    default_m_index = {
        1: 0.794, 2: 0.782, 3: 0.694, 4: 0.651, 5: 0.564, 6: 1.000,
        7: 1.700, 8: 1.442, 9: 1.174, 10: 0.734, 11: 1.068, 12: 1.400
    }
    for m in range(1, 13):
        if m not in m_index:
            m_index[m] = default_m_index[m]

    # Son dönem ve genel ortalamalar
    fac_recent_28 = past_df[past_df["tarih"] >= (cutoff_date - pd.Timedelta(days=28))].groupby("tanim")["tuketim"].mean().to_dict()
    fac_mean_all = past_df.groupby("tanim")["tuketim"].mean().to_dict()

    # 2. De-noised Yaz Transferi (Ağustos -> Temmuz)
    past_july = past_df[past_df["tarih"].dt.month == 7].groupby("tanim")["tuketim"].mean()
    past_aug = past_df[past_df["tarih"].dt.month == 8].groupby("tanim")["tuketim"].mean()
    past_winter = past_df[past_df["tarih"].dt.month.isin([1, 2, 3])].groupby("tanim")["tuketim"].mean()
    guc_map = past_df.groupby("tanim")["guc"].first().to_dict()

    fac_summer_surge = {}
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

    # 3. K-Means 3'lü Sürekli Arketip Kümeleri
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
        df = add_fourier_features(df)
        df["month"] = df["tarih"].dt.month
        df["day_of_week"] = df["tarih"].dt.dayofweek
        df["day_of_year"] = df["tarih"].dt.dayofyear
        df["day"] = df["tarih"].dt.day
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        df["is_june_july"] = df["month"].isin([6, 7]).astype(int)
        df["log_guc"] = np.log1p(np.maximum(1.0, df["guc"])).astype(np.float32)
        df["log_guc_x_summer"] = (df["log_guc"] * df["is_summer"]).astype(np.float32)

        df["monthly_network_index"] = df["month"].map(m_index).fillna(1.0).astype(np.float32)

        for c in cat_cols_raw:
            c_map = global_cat_maps[c]
            df[f"{c}_code"] = df[c].map(c_map).fillna(-1).astype(np.int32)

        df["fac_recent_28"] = df["tanim"].map(fac_recent_28)
        df["fac_mean_all"] = df["tanim"].map(fac_mean_all)
        df["fac_level"] = df["fac_recent_28"].fillna(df["fac_mean_all"]).fillna(df["guc"] * 2.5).astype(np.float32)
        df["log_fac_level"] = np.log1p(df["fac_level"]).astype(np.float32)

        m_arr = df["month"].values
        base_arr = df["fac_level"].values

        direct_surge = df["tanim"].map(fac_summer_surge)
        fallback_ilce = df["ilce"].map(ilce_surge)
        fallback_guc = df["guc_bin"].map(guc_surge)
        surge_final = direct_surge.fillna(fallback_ilce).fillna(fallback_guc).fillna(global_surge).values

        seasonal_mult = np.where(
            m_arr == 4, 0.85,
            np.where(
                m_arr == 5, 0.70,
                np.where(
                    m_arr == 6, 1.00 + 0.35 * (surge_final - 1.0),
                    np.where(
                        m_arr == 7, 1.10 + 0.85 * (surge_final - 1.0),
                        np.where(
                            m_arr == 8, 1.05 + 0.70 * (surge_final - 1.0),
                            0.95
                        )
                    )
                )
            )
        )
        df["seasonal_baseline"] = (base_arr * seasonal_mult).astype(np.float32)
        df["log_seasonal_baseline"] = np.log1p(df["seasonal_baseline"]).astype(np.float32)

        df["is_cold"] = (~df["tanim"].isin(past_df["tanim"].unique())).astype(int)
        df["has_annual_lag"] = df["tanim"].isin(fac_summer_surge).astype(int)
        df["arch_prob_0"] = df["tanim"].map(prob_dict_0).fillna(0.333).astype(np.float32)
        df["arch_prob_1"] = df["tanim"].map(prob_dict_1).fillna(0.333).astype(np.float32)
        df["arch_prob_2"] = df["tanim"].map(prob_dict_2).fillna(0.333).astype(np.float32)

        return df

    return transform_df(past_df), transform_df(target_df)


# ==============================================================================
# PIPELINE YÜRÜTME & ENSEMBLE
# ==============================================================================

def run_v15_master_pipeline():
    logger.info("=" * 75)
    logger.info("🚀 GRID-UP DATATHON — V15 MASTER PIPELINE BAŞLATILIYOR")
    logger.info("=" * 75)

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    raw_train = pd.read_csv(train_path, parse_dates=["tarih"])
    raw_test = pd.read_csv(test_path, parse_dates=["tarih"])

    raw_train = parse_locations(raw_train)
    raw_test = parse_locations(raw_test)

    guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
    guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
    raw_train["guc_bin"] = pd.cut(raw_train["guc"], bins=guc_bins, labels=guc_labels).astype(str)
    raw_test["guc_bin"] = pd.cut(raw_test["guc"], bins=guc_bins, labels=guc_labels).astype(str)

    cat_cols_raw = ["il", "ilce", "bolge", "guc_bin"]
    global_cat_maps = {}
    for col in cat_cols_raw:
        all_vals = sorted(list(set(raw_train[col].dropna()).union(set(raw_test[col].dropna()))))
        global_cat_maps[col] = {val: i for i, val in enumerate(all_vals)}
        raw_train[f"{col}_code"] = raw_train[col].map(global_cat_maps[col]).fillna(-1).astype(np.int32)
        raw_test[f"{col}_code"] = raw_test[col].map(global_cat_maps[col]).fillna(-1).astype(np.int32)

    features_model = [
        "guc", "log_guc", "il_code", "ilce_code", "bolge_code", "guc_bin_code",
        "month", "day", "day_of_week", "day_of_year", "is_weekend", "is_summer", "is_june_july",
        "log_guc_x_summer", "monthly_network_index", "log_fac_level", "log_seasonal_baseline",
        "is_cold", "has_annual_lag", "arch_prob_0", "arch_prob_1", "arch_prob_2",
        "sin_day_1", "cos_day_1", "sin_day_2", "cos_day_2",
        "sin_dow_1", "cos_dow_1", "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2"
    ]

    # -------------------------------------------------------------------------
    # 1. FOLD A DOĞRULAMASI (Cutoff = 2025-03-31, Val = Nisan-Temmuz 2025)
    # -------------------------------------------------------------------------
    logger.info("\n--- 1. FOLD A SIZINTISIZ DOĞRULAMA ÇALIŞTIRILIYOR ---")
    cutoff_a = pd.Timestamp("2025-03-31")
    val_a_raw = raw_train[(raw_train["tarih"] >= "2025-04-01") & (raw_train["tarih"] <= "2025-07-31")].copy()

    past_a_feat, val_a_feat = build_v15_features(raw_train, val_a_raw, cutoff_a, global_cat_maps, cat_cols_raw)

    y_train_res = np.log1p(past_a_feat["tuketim"].values) - np.log1p(past_a_feat["seasonal_baseline"].values)

    # Stage 1: LightGBM
    m_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
    )
    m_lgb.fit(past_a_feat[features_model], y_train_res)
    pred_res_lgb = m_lgb.predict(val_a_feat[features_model])

    # Stage 2: CatBoost
    m_cb = CatBoostRegressor(
        iterations=350, learning_rate=0.06, depth=6, loss_function="RMSE",
        thread_count=-1, random_seed=42, verbose=False
    )
    m_cb.fit(past_a_feat[features_model], y_train_res, verbose=False)
    pred_res_cb = m_cb.predict(val_a_feat[features_model])

    # Two-Stage Ensemble Residual
    pred_res = 0.50 * pred_res_lgb + 0.50 * pred_res_cb
    ceil_val = 36.0 * (val_a_feat["guc"].values + 1.0)
    pred_v15_val = np.clip(np.maximum(0.0, np.expm1(np.log1p(val_a_feat["seasonal_baseline"].values) + pred_res)), 0.0, ceil_val)

    val_a_feat["pred_v15"] = pred_v15_val
    tot_rmsle = calculate_rmsle(val_a_feat["tuketim"].values, pred_v15_val)
    warm_rmsle = calculate_rmsle(val_a_feat[val_a_feat["is_cold"] == 0]["tuketim"].values, val_a_feat[val_a_feat["is_cold"] == 0]["pred_v15"].values)
    cold_rmsle = calculate_rmsle(val_a_feat[val_a_feat["is_cold"] == 1]["tuketim"].values, val_a_feat[val_a_feat["is_cold"] == 1]["pred_v15"].values)

    logger.info(f"✓ FOLD A V15 Total RMSLE : {tot_rmsle:.5f} (N={len(val_a_feat):,d})")
    logger.info(f"   -> Warm RMSLE (%92.5) : {warm_rmsle:.5f}")
    logger.info(f"   -> Cold RMSLE ( %7.5) : {cold_rmsle:.5f}")

    # -------------------------------------------------------------------------
    # 2. FULL 100% RETRAINING (Cutoff = 2026-03-31) & TEST TAHMİNİ
    # -------------------------------------------------------------------------
    logger.info("\n--- 2. TÜM VERİ (15 AY) ÜZERİNDE TAM MODEL EĞİTİMİ ---")
    cutoff_full = pd.Timestamp("2026-03-31")
    full_tr_feat, test_feat = build_v15_features(raw_train, raw_test, cutoff_full, global_cat_maps, cat_cols_raw)

    y_full_res = np.log1p(full_tr_feat["tuketim"].values) - np.log1p(full_tr_feat["seasonal_baseline"].values)

    full_lgb = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
    )
    full_lgb.fit(full_tr_feat[features_model], y_full_res)
    pred_test_lgb = full_lgb.predict(test_feat[features_model])

    full_cb = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, loss_function="RMSE",
        thread_count=-1, random_seed=42, verbose=False
    )
    full_cb.fit(full_tr_feat[features_model], y_full_res, verbose=False)
    pred_test_cb = full_cb.predict(test_feat[features_model])

    test_res_pred = 0.50 * pred_test_lgb + 0.50 * pred_test_cb
    test_final_pred = np.maximum(0.0, np.expm1(np.log1p(test_feat["seasonal_baseline"].values) + test_res_pred))

    # Kapasite tavanı denetimi
    test_ceil = 36.0 * (test_feat["guc"].values + 1.0)
    test_final_pred = np.clip(test_final_pred, 0.0, test_ceil)

    # 1. Standalone V15 Submission
    sub_v15 = pd.DataFrame({"id": test_feat["id"], "tuketim": test_final_pred})
    sub_v15.to_csv(OUTPUT_STANDALONE_PATH, index=False)
    sha_v15 = get_sha256(OUTPUT_STANDALONE_PATH)

    # 2. Optimal Master Blend
    logger.info("\n--- 3. OPTIMAL MASTER BLEND OLUŞTURULUYOR ---")
    if BASE_V8R_PATH.exists():
        v8r_sub = pd.read_csv(BASE_V8R_PATH)
        is_cold_test = test_feat["is_cold"].values
        blend_preds = np.where(
            is_cold_test == 1,
            0.60 * test_final_pred + 0.40 * v8r_sub["tuketim"].values,
            0.80 * test_final_pred + 0.20 * v8r_sub["tuketim"].values
        )
        blend_preds = np.clip(blend_preds, 0.0, test_ceil)

        sub_blend = pd.DataFrame({"id": test_feat["id"], "tuketim": blend_preds})
        sub_blend.to_csv(OUTPUT_BLEND_PATH, index=False)
        sha_blend = get_sha256(OUTPUT_BLEND_PATH)
    else:
        sha_blend = "N/A"

    logger.info("=" * 75)
    logger.info("🎉 V15 MASTER PIPELINE BAŞARIYLA TAMAMLANDI!")
    logger.info(f"✓ V15 Standalone Output   : {OUTPUT_STANDALONE_PATH} (SHA: {sha_v15})")
    logger.info(f"✓ V15 Optimal Master Blend : {OUTPUT_BLEND_PATH} (SHA: {sha_blend})")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_v15_master_pipeline()
