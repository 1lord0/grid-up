"""Grid Up Datathon — V16 Surgical Dual-Track Pipeline.

Qwen 3.8-27B Dual-Track Architecture:
1. Warm Track: V15 Two-Stage Champion Engine (Fourier 24s/168s/365g + De-noised Summer Transfer + K-Means 3'lü Arketip + LGBM+CatBoost Residual Ensemble -> 0.825 RMSLE)
2. Cold Track: Cerrahi Hiyerarşik Ampirik Bayes + Kapasite Normalizasyonlu Medyan Çaprazı (15 Statik K-Means Kümesi + Beta Shrinkage 0.88-0.95)
3. Log-Uzayda Segment Bazlı Nelder-Mead Karışımı & Fiziksel Kapasite Tavanı
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
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
BASE_V8R_PATH = DATA_DIR / "submission_v8r_verified_final.csv"
OUTPUT_STANDALONE_PATH = DATA_DIR / "submission_v16_standalone.csv"
OUTPUT_BLEND_PATH = DATA_DIR / "submission_v16_optimal_blend.csv"


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


def build_v16_features(
    train_df: pd.DataFrame, target_df: pd.DataFrame, cutoff_date: pd.Timestamp,
    global_cat_maps: Dict[str, Dict[str, int]], cat_cols_raw: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """V16 Master Özellik Mühendisliği ve Bayes Cold Veri Yapıları."""
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

    # 3. K-Means 3'lü Sürekli Arketip Kümeleri (Warm Modeli İçin)
    profile_pivot = past_df.pivot_table(index="tanim", columns=past_df["tarih"].dt.month, values="tuketim", aggfunc="mean")
    profile_norm = profile_pivot.div(profile_pivot.mean(axis=1), axis=0).fillna(1.0)
    for m in range(1, 13):
        if m not in profile_norm.columns:
            profile_norm[m] = default_m_index[m]
    profile_norm = profile_norm[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

    kmeans_warm = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_warm.fit(profile_norm.values)
    centers_warm = kmeans_warm.cluster_centers_

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
    diffs = synth_mat[:, np.newaxis, :] - centers_warm[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    exp_neg = np.exp(-dists)
    probs = exp_neg / exp_neg.sum(axis=1, keepdims=True)

    prob_dict_0 = dict(zip(all_known_facs, probs[:, 0]))
    prob_dict_1 = dict(zip(all_known_facs, probs[:, 1]))
    prob_dict_2 = dict(zip(all_known_facs, probs[:, 2]))

    # 4. COLD TRACK: Hiyerarşik Ampirik Bayes ve Kapasite Kullanım Medyanları
    past_df["utilization"] = past_df["tuketim"] / (past_df["guc"] + 1.0)
    past_df["month"] = past_df["tarih"].dt.month
    past_df["dow"] = past_df["tarih"].dt.dayofweek
    past_df["is_weekend"] = (past_df["dow"] >= 5).astype(int)

    # 15 Statik K-Means Kümesi
    warm_fac_meta = past_df.drop_duplicates(subset="tanim").set_index("tanim")
    static_features = np.column_stack([
        np.log1p(warm_fac_meta["guc"].values),
        warm_fac_meta["il_code"].values,
        warm_fac_meta["ilce_code"].values,
        warm_fac_meta["bolge_code"].values,
    ])
    kmeans_cold = KMeans(n_clusters=15, random_state=42, n_init=10)
    warm_fac_meta["cold_cluster"] = kmeans_cold.fit_predict(static_features)
    cold_cluster_map_warm = warm_fac_meta["cold_cluster"].to_dict()

    past_df["cold_cluster"] = past_df["tanim"].map(cold_cluster_map_warm).fillna(0).astype(int)

    # Medyan Hiyerarşisi
    med_l1 = past_df.groupby(["cold_cluster", "month", "is_weekend"])["utilization"].median().to_dict()
    med_l2 = past_df.groupby(["ilce_code", "month"])["utilization"].median().to_dict()
    med_l3 = past_df.groupby(["guc_bin_code", "month"])["utilization"].median().to_dict()
    med_l4 = past_df.groupby("month")["utilization"].median().to_dict()
    global_med = float(past_df["utilization"].median())

    # Target DF için Cold Kümelerini Ata
    target_fac_meta = target_df.drop_duplicates(subset="tanim").set_index("tanim")
    target_static = np.column_stack([
        np.log1p(target_fac_meta["guc"].values),
        target_fac_meta["il_code"].values,
        target_fac_meta["ilce_code"].values,
        target_fac_meta["bolge_code"].values,
    ])
    target_fac_meta["cold_cluster"] = kmeans_cold.predict(target_static)
    cold_cluster_map_target = target_fac_meta["cold_cluster"].to_dict()

    bayes_bundle = {
        "kmeans_cold": kmeans_cold,
        "med_l1": med_l1,
        "med_l2": med_l2,
        "med_l3": med_l3,
        "med_l4": med_l4,
        "global_med": global_med,
        "cluster_map": {**cold_cluster_map_warm, **cold_cluster_map_target}
    }

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

        # Cold Cluster ID
        df["cold_cluster"] = df["tanim"].map(bayes_bundle["cluster_map"]).fillna(0).astype(int)

        return df

    return transform_df(past_df), transform_df(target_df), bayes_bundle


def predict_cold_bayes(
    df: pd.DataFrame, bayes_bundle: Dict[str, Any], beta: float = 0.90
) -> np.ndarray:
    """Hiyerarşik Ampirik Bayes kullanarak Cold tesisler için tüketim tahmini üretir."""
    med_l1 = bayes_bundle["med_l1"]
    med_l2 = bayes_bundle["med_l2"]
    med_l3 = bayes_bundle["med_l3"]
    med_l4 = bayes_bundle["med_l4"]
    global_med = bayes_bundle["global_med"]

    c_clusters = df["cold_cluster"].values
    months = df["month"].values
    is_weekends = df["is_weekend"].values
    ilce_codes = df["ilce_code"].values
    guc_bin_codes = df["guc_bin_code"].values
    gucs = df["guc"].values

    pred_utilizations = []
    for i in range(len(df)):
        c_c = c_clusters[i]
        m = months[i]
        iw = is_weekends[i]
        ilce = ilce_codes[i]
        g_bin = guc_bin_codes[i]

        # Seviye 1
        if (c_c, m, iw) in med_l1:
            u = med_l1[(c_c, m, iw)]
        # Seviye 2
        elif (ilce, m) in med_l2:
            u = med_l2[(ilce, m)]
        # Seviye 3
        elif (g_bin, m) in med_l3:
            u = med_l3[(g_bin, m)]
        # Seviye 4
        elif m in med_l4:
            u = med_l4[m]
        else:
            u = global_med

        pred_utilizations.append(u)

    pred_utilizations = np.array(pred_utilizations, dtype=np.float32)
    # Tüketim = Kullanım Oranı * (Güç + 1.0) * Beta
    pred_cold = pred_utilizations * (gucs + 1.0) * beta
    return np.maximum(0.0, pred_cold)


# ==============================================================================
# PIPELINE YÜRÜTME & ENSEMBLE
# ==============================================================================

def run_v16_pipeline():
    logger.info("=" * 75)
    logger.info("🚀 GRID-UP DATATHON — V16 SURGICAL DUAL-TRACK PIPELINE BAŞLATILIYOR")
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

    features_warm_model = [
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
    logger.info("\n--- 1. FOLD A DUAL-TRACK DOĞRULAMASI ÇALIŞTIRILIYOR ---")
    cutoff_a = pd.Timestamp("2025-03-31")
    val_a_raw = raw_train[(raw_train["tarih"] >= "2025-04-01") & (raw_train["tarih"] <= "2025-07-31")].copy()

    past_a_feat, val_a_feat, bayes_bundle_a = build_v16_features(raw_train, val_a_raw, cutoff_a, global_cat_maps, cat_cols_raw)

    # 1.A: WARM TRACK MODELİ (Sadece geçmişi olan veriler üzerinde eğit)
    y_train_res = np.log1p(past_a_feat["tuketim"].values) - np.log1p(past_a_feat["seasonal_baseline"].values)

    m_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
    )
    m_lgb.fit(past_a_feat[features_warm_model], y_train_res)
    pred_res_lgb = m_lgb.predict(val_a_feat[features_warm_model])

    m_cb = CatBoostRegressor(
        iterations=350, learning_rate=0.06, depth=6, loss_function="RMSE",
        thread_count=-1, random_seed=42, verbose=False
    )
    m_cb.fit(past_a_feat[features_warm_model], y_train_res, verbose=False)
    pred_res_cb = m_cb.predict(val_a_feat[features_warm_model])

    pred_res_warm = 0.50 * pred_res_lgb + 0.50 * pred_res_cb
    pred_warm_val = np.maximum(0.0, np.expm1(np.log1p(val_a_feat["seasonal_baseline"].values) + pred_res_warm))

    # 1.B: COLD TRACK MODELİ (Hiyerarşik Bayes + Beta Grid Search)
    val_cold_mask = val_a_feat["is_cold"] == 1
    val_cold_df = val_a_feat[val_cold_mask].copy()

    # Optimal Beta Araştırması (Fold A üzerinde)
    best_beta = 0.90
    best_cold_rmsle = 999.0
    for b in np.arange(0.70, 1.20, 0.02):
        cold_preds_trial = predict_cold_bayes(val_cold_df, bayes_bundle_a, beta=b)
        r = calculate_rmsle(val_cold_df["tuketim"].values, cold_preds_trial)
        if r < best_cold_rmsle:
            best_cold_rmsle = r
            best_beta = float(b)

    logger.info(f"✓ Optimal Cold Beta Shrinkage Faktörü: {best_beta:.2f} (Cold RMSLE: {best_cold_rmsle:.5f})")

    # 1.C: DUAL-TRACK BİRLEŞTİRME
    pred_cold_val = predict_cold_bayes(val_a_feat, bayes_bundle_a, beta=best_beta)
    pred_v16_val = np.where(val_cold_mask, pred_cold_val, pred_warm_val)

    # Kapasite tavanı denetimi
    ceil_val = 36.0 * (val_a_feat["guc"].values + 1.0)
    pred_v16_val = np.clip(pred_v16_val, 0.0, ceil_val)

    val_a_feat["pred_v16"] = pred_v16_val
    tot_rmsle = calculate_rmsle(val_a_feat["tuketim"].values, pred_v16_val)
    warm_rmsle = calculate_rmsle(val_a_feat[~val_cold_mask]["tuketim"].values, val_a_feat[~val_cold_mask]["pred_v16"].values)
    cold_rmsle = calculate_rmsle(val_a_feat[val_cold_mask]["tuketim"].values, val_a_feat[val_cold_mask]["pred_v16"].values)

    logger.info(f"★ FOLD A V16 Total RMSLE : {tot_rmsle:.5f} (N={len(val_a_feat):,d})")
    logger.info(f"   -> Warm RMSLE (%92.5) : {warm_rmsle:.5f}")
    logger.info(f"   -> Cold RMSLE ( %7.5) : {cold_rmsle:.5f}")

    # -------------------------------------------------------------------------
    # 2. FULL 100% RETRAINING (Cutoff = 2026-03-31) & TEST TAHMİNİ
    # -------------------------------------------------------------------------
    logger.info("\n--- 2. TÜM VERİ (15 AY) ÜZERİNDE TAM DUAL-TRACK MODEL EĞİTİMİ ---")
    cutoff_full = pd.Timestamp("2026-03-31")
    full_tr_feat, test_feat, bayes_bundle_full = build_v16_features(raw_train, raw_test, cutoff_full, global_cat_maps, cat_cols_raw)

    y_full_res = np.log1p(full_tr_feat["tuketim"].values) - np.log1p(full_tr_feat["seasonal_baseline"].values)

    full_lgb = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
    )
    full_lgb.fit(full_tr_feat[features_warm_model], y_full_res)
    pred_test_lgb = full_lgb.predict(test_feat[features_warm_model])

    full_cb = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, loss_function="RMSE",
        thread_count=-1, random_seed=42, verbose=False
    )
    full_cb.fit(full_tr_feat[features_warm_model], y_full_res, verbose=False)
    pred_test_cb = full_cb.predict(test_feat[features_warm_model])

    test_res_pred = 0.50 * pred_test_lgb + 0.50 * pred_test_cb
    test_warm_pred = np.maximum(0.0, np.expm1(np.log1p(test_feat["seasonal_baseline"].values) + test_res_pred))

    # Cold Test Tahmini (Bayes)
    test_cold_pred = predict_cold_bayes(test_feat, bayes_bundle_full, beta=best_beta)

    # Dual-Track Final Birleştirme
    test_is_cold = test_feat["is_cold"] == 1
    test_final_pred = np.where(test_is_cold, test_cold_pred, test_warm_pred)

    test_ceil = 36.0 * (test_feat["guc"].values + 1.0)
    test_final_pred = np.clip(test_final_pred, 0.0, test_ceil)

    # 1. Standalone V16 Submission
    sub_v16 = pd.DataFrame({"id": test_feat["id"], "tuketim": test_final_pred})
    sub_v16.to_csv(OUTPUT_STANDALONE_PATH, index=False)
    sha_v16 = get_sha256(OUTPUT_STANDALONE_PATH)

    # 2. Optimal Master Blend (V16 + V8R)
    logger.info("\n--- 3. OPTIMAL MASTER BLEND OLUŞTURULUYOR ---")
    if BASE_V8R_PATH.exists():
        v8r_sub = pd.read_csv(BASE_V8R_PATH)
        blend_preds = np.where(
            test_is_cold,
            0.65 * test_final_pred + 0.35 * v8r_sub["tuketim"].values,
            0.85 * test_final_pred + 0.15 * v8r_sub["tuketim"].values
        )
        blend_preds = np.clip(blend_preds, 0.0, test_ceil)

        sub_blend = pd.DataFrame({"id": test_feat["id"], "tuketim": blend_preds})
        sub_blend.to_csv(OUTPUT_BLEND_PATH, index=False)
        sha_blend = get_sha256(OUTPUT_BLEND_PATH)
    else:
        sha_blend = "N/A"

    logger.info("=" * 75)
    logger.info("🎉 V16 DUAL-TRACK PIPELINE BAŞARIYLA TAMAMLANDI!")
    logger.info(f"✓ V16 Standalone Output : {OUTPUT_STANDALONE_PATH} (SHA: {sha_v16})")
    logger.info(f"✓ V16 Optimal Blend     : {OUTPUT_BLEND_PATH} (SHA: {sha_blend})")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_v16_pipeline()