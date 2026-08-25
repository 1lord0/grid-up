"""Grid Up Datathon — V17 Breakthrough Architecture.

Qwen 3.8-27B Tarafından Tasarlanan Cold Yıkım ve Warm İyileştirme Planı:
1. Cold Track: Zero-History Dedicated GBDT (LightGBM + CatBoost)
   - Hedef: log1p(tuketim / (guc + 1.0)) (Kapasite Kullanım Oranı)
   - Özellikler: OOF Target Encoding (İlçe x Güç Grubu x Ay), İlçe Güç Oranı (guc / ilce_mean_guc),
     Bölge Güç Oranı, Güç x Yaz Etkileşimleri, Harmonik Takvim Özellikleri.
2. Warm Track: Two-Stage Champion Engine + Trend Momentumu + Gelişmiş Fourier Etkileşimleri
3. Sızıntısız Fold A Doğrulaması ve 100% Veri Eğitimi
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
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
OUTPUT_V17_PATH = DATA_DIR / "submission_v17_breakthrough.csv"


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
    """Fourier ve harmonik zaman özellikleri."""
    df = df.copy()
    day = df["tarih"].dt.day
    dow = df["tarih"].dt.dayofweek
    doy = df["tarih"].dt.dayofyear
    m = df["tarih"].dt.month

    # Gün döngüsü
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

    # Ay döngüsü
    df["sin_month"] = np.sin(2 * np.pi * m / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * m / 12.0).astype(np.float32)

    return df


def build_cold_features(past_train: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Cold tesisler için Zero-History statik + coğrafi + OOF target encoding özellikleri üretir."""
    tr = past_train.copy()
    te = target_df.copy()

    tr = add_fourier_features(tr)
    te = add_fourier_features(te)

    tr["month"] = tr["tarih"].dt.month
    te["month"] = te["tarih"].dt.month
    tr["is_weekend"] = (tr["tarih"].dt.dayofweek >= 5).astype(int)
    te["is_weekend"] = (te["tarih"].dt.dayofweek >= 5).astype(int)
    tr["is_summer"] = tr["month"].isin([6, 7, 8]).astype(int)
    te["is_summer"] = te["month"].isin([6, 7, 8]).astype(int)

    # Güç özellikleri
    tr["log_guc"] = np.log1p(tr["guc"].values).astype(np.float32)
    te["log_guc"] = np.log1p(te["guc"].values).astype(np.float32)
    tr["guc_sq"] = np.sqrt(tr["guc"].values).astype(np.float32)
    te["guc_sq"] = np.sqrt(te["guc"].values).astype(np.float32)
    tr["log_guc_x_summer"] = (tr["log_guc"] * tr["is_summer"]).astype(np.float32)
    te["log_guc_x_summer"] = (te["log_guc"] * te["is_summer"]).astype(np.float32)

    # İlçe ve Bölge ortalama güç oranları (Sadece geçmiş train üzerinden!)
    ilce_mean_guc = tr.groupby("ilce_code")["guc"].mean().to_dict()
    bolge_mean_guc = tr.groupby("bolge_code")["guc"].mean().to_dict()

    tr["guc_ratio_ilce"] = (tr["guc"] / (tr["ilce_code"].map(ilce_mean_guc).fillna(630.0) + 1.0)).astype(np.float32)
    te["guc_ratio_ilce"] = (te["guc"] / (te["ilce_code"].map(ilce_mean_guc).fillna(630.0) + 1.0)).astype(np.float32)
    tr["guc_ratio_bolge"] = (tr["guc"] / (tr["bolge_code"].map(bolge_mean_guc).fillna(630.0) + 1.0)).astype(np.float32)
    te["guc_ratio_bolge"] = (te["guc"] / (te["bolge_code"].map(bolge_mean_guc).fillna(630.0) + 1.0)).astype(np.float32)

    # Hedef Değişken: Log-Kapasite Kullanım Oranı
    tr["utilization"] = tr["tuketim"] / (tr["guc"] + 1.0)
    tr["log_utilization"] = np.log1p(tr["utilization"].values).astype(np.float32)

    # OOF Target Encoding (İlçe x Güç Grubu x Ay)
    tr["oof_target_enc"] = 0.0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(tr):
        fold_tr = tr.iloc[tr_idx]
        group_means = fold_tr.groupby(["ilce_code", "guc_bin_code", "month"])["log_utilization"].mean().to_dict()
        fallback_means = fold_tr.groupby(["guc_bin_code", "month"])["log_utilization"].mean().to_dict()
        global_mean = float(fold_tr["log_utilization"].mean())

        fold_val_enc = []
        for _, r in tr.iloc[val_idx].iterrows():
            key = (r["ilce_code"], r["guc_bin_code"], r["month"])
            fb_key = (r["guc_bin_code"], r["month"])
            val = group_means.get(key, fallback_means.get(fb_key, global_mean))
            fold_val_enc.append(val)
        tr.iloc[val_idx, tr.columns.get_loc("oof_target_enc")] = fold_val_enc

    # Test setine tüm train üzerinden target encoding aktarımı
    full_group_means = tr.groupby(["ilce_code", "guc_bin_code", "month"])["log_utilization"].mean().to_dict()
    full_fallback_means = tr.groupby(["guc_bin_code", "month"])["log_utilization"].mean().to_dict()
    full_global_mean = float(tr["log_utilization"].mean())

    test_enc = []
    for _, r in te.iterrows():
        key = (r["ilce_code"], r["guc_bin_code"], r["month"])
        fb_key = (r["guc_bin_code"], r["month"])
        val = full_group_means.get(key, full_fallback_means.get(fb_key, full_global_mean))
        test_enc.append(val)
    te["oof_target_enc"] = test_enc

    features_cold = [
        "guc", "log_guc", "guc_sq", "il_code", "ilce_code", "bolge_code", "guc_bin_code",
        "month", "is_weekend", "is_summer", "log_guc_x_summer",
        "guc_ratio_ilce", "guc_ratio_bolge", "oof_target_enc",
        "sin_day_1", "cos_day_1", "sin_day_2", "cos_day_2",
        "sin_dow_1", "cos_dow_1", "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2",
        "sin_month", "cos_month"
    ]

    return tr, te, features_cold


def train_dedicated_cold_model(tr_cold_feat: pd.DataFrame, te_cold_feat: pd.DataFrame, features_cold: List[str]) -> np.ndarray:
    """Cold tesisler için özel eğitilmiş CatBoost + LightGBM model topluluğu."""
    # Eğitim: Tüm geçmiş veri üzerinde log_utilization hedeflenerek eğitilir
    y_train_util = tr_cold_feat["log_utilization"].values

    # 1. LightGBM Cold Modeli
    lgb_cold = lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
    )
    lgb_cold.fit(tr_cold_feat[features_cold], y_train_util)
    pred_log_util_lgb = lgb_cold.predict(te_cold_feat[features_cold])

    # 2. CatBoost Cold Modeli
    cb_cold = CatBoostRegressor(
        iterations=450, learning_rate=0.05, depth=6, loss_function="RMSE",
        thread_count=-1, random_seed=42, verbose=False
    )
    cb_cold.fit(tr_cold_feat[features_cold], y_train_util, verbose=False)
    pred_log_util_cb = cb_cold.predict(te_cold_feat[features_cold])

    pred_log_util = 0.50 * pred_log_util_lgb + 0.50 * pred_log_util_cb
    pred_util = np.expm1(np.maximum(0.0, pred_log_util))

    # Tüketim Tahmini: Kullanım Oranı * (Güç + 1.0)
    pred_cold_consumption = pred_util * (te_cold_feat["guc"].values + 1.0)
    return pred_cold_consumption


# ==============================================================================
# PIPELINE YÜRÜTME
# ==============================================================================

def run_v17_pipeline():
    logger.info("=" * 75)
    logger.info("🚀 GRID-UP DATATHON — V17 BREAKTHROUGH PIPELINE BAŞLATILIYOR")
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

    # -------------------------------------------------------------------------
    # 1. FOLD A DOĞRULAMASI (Cutoff = 2025-03-31, Val = Nisan-Temmuz 2025)
    # -------------------------------------------------------------------------
    logger.info("\n--- 1. FOLD A V17 BREAKTHROUGH DOĞRULAMASI ---")
    cutoff_a = pd.Timestamp("2025-03-31")
    past_a = raw_train[raw_train["tarih"] <= cutoff_a].copy()
    val_a = raw_train[(raw_train["tarih"] >= "2025-04-01") & (raw_train["tarih"] <= "2025-07-31")].copy()

    # 1.A: WARM MODELİ (V16 İki Aşamalı Şampiyon Model)
    from train_v16_surgical_cold_start import build_v16_features
    past_a_warm_feat, val_a_warm_feat, _ = build_v16_features(raw_train, val_a, cutoff_a, global_cat_maps, cat_cols_raw)

    features_warm_model = [
        "guc", "log_guc", "il_code", "ilce_code", "bolge_code", "guc_bin_code",
        "month", "day", "day_of_week", "day_of_year", "is_weekend", "is_summer", "is_june_july",
        "log_guc_x_summer", "monthly_network_index", "log_fac_level", "log_seasonal_baseline",
        "is_cold", "has_annual_lag", "arch_prob_0", "arch_prob_1", "arch_prob_2",
        "sin_day_1", "cos_day_1", "sin_day_2", "cos_day_2",
        "sin_dow_1", "cos_dow_1", "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2"
    ]
    y_train_res = np.log1p(past_a_warm_feat["tuketim"].values) - np.log1p(past_a_warm_feat["seasonal_baseline"].values)

    m_lgb_w = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    m_lgb_w.fit(past_a_warm_feat[features_warm_model], y_train_res)
    pred_res_lgb = m_lgb_w.predict(val_a_warm_feat[features_warm_model])

    m_cb_w = CatBoostRegressor(iterations=350, learning_rate=0.06, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
    m_cb_w.fit(past_a_warm_feat[features_warm_model], y_train_res, verbose=False)
    pred_res_cb = m_cb_w.predict(val_a_warm_feat[features_warm_model])

    pred_warm_val = np.maximum(0.0, np.expm1(np.log1p(val_a_warm_feat["seasonal_baseline"].values) + 0.50 * pred_res_lgb + 0.50 * pred_res_cb))

    # 1.B: COLD MODELİ (V17 Dedicated Zero-History GBDT)
    tr_cold_feat_a, val_cold_feat_a, features_cold = build_cold_features(past_a, val_a)
    pred_cold_val_raw = train_dedicated_cold_model(tr_cold_feat_a, val_cold_feat_a, features_cold)

    # 1.C: BİRLEŞTİRME VE KALİBRASYON
    is_cold_val = val_a_warm_feat["is_cold"] == 1
    pred_v17_val = np.where(is_cold_val, pred_cold_val_raw, pred_warm_val)

    # Kapasite tavanı denetimi
    ceil_val = 36.0 * (val_a["guc"].values + 1.0)
    pred_v17_val = np.clip(pred_v17_val, 0.0, ceil_val)

    val_a["pred_v17"] = pred_v17_val
    tot_rmsle = calculate_rmsle(val_a["tuketim"].values, pred_v17_val)
    warm_rmsle = calculate_rmsle(val_a[~is_cold_val]["tuketim"].values, val_a[~is_cold_val]["pred_v17"].values)
    cold_rmsle = calculate_rmsle(val_a[is_cold_val]["tuketim"].values, val_a[is_cold_val]["pred_v17"].values)

    # 22.16% Cold Ağırlıklı Test Simülasyonu
    simulated_test_lb = np.sqrt(0.7784 * (warm_rmsle**2) + 0.2216 * (cold_rmsle**2))

    logger.info(f"★ FOLD A V17 Total RMSLE : {tot_rmsle:.5f} (N={len(val_a):,d})")
    logger.info(f"   -> Warm RMSLE (%92.5) : {warm_rmsle:.5f}")
    logger.info(f"   -> Cold RMSLE ( %7.5) : {cold_rmsle:.5f}")
    logger.info(f"🔥 TEST SETİ AĞIRLIKLI LB BEKLENTİSİ: {simulated_test_lb:.5f}")

    # -------------------------------------------------------------------------
    # 2. FULL 100% RETRAINING (Cutoff = 2026-03-31) & TEST TAHMİNİ
    # -------------------------------------------------------------------------
    logger.info("\n--- 2. TÜM VERİ (15 AY) ÜZERİNDE TAM V17 MODEL EĞİTİMİ ---")
    cutoff_full = pd.Timestamp("2026-03-31")
    full_past = raw_train.copy()

    # Warm Full Model
    full_warm_feat, test_warm_feat, _ = build_v16_features(raw_train, raw_test, cutoff_full, global_cat_maps, cat_cols_raw)
    y_full_res = np.log1p(full_warm_feat["tuketim"].values) - np.log1p(full_warm_feat["seasonal_baseline"].values)

    full_lgb_w = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    full_lgb_w.fit(full_warm_feat[features_warm_model], y_full_res)
    pred_test_warm_lgb = full_lgb_w.predict(test_warm_feat[features_warm_model])

    full_cb_w = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
    full_cb_w.fit(full_warm_feat[features_warm_model], y_full_res, verbose=False)
    pred_test_warm_cb = full_cb_w.predict(test_warm_feat[features_warm_model])

    test_warm_pred = np.maximum(0.0, np.expm1(np.log1p(test_warm_feat["seasonal_baseline"].values) + 0.50 * pred_test_warm_lgb + 0.50 * pred_test_warm_cb))

    # Cold Full Model (Zero-History GBDT)
    tr_cold_feat_full, test_cold_feat_full, _ = build_cold_features(full_past, raw_test)
    test_cold_pred = train_dedicated_cold_model(tr_cold_feat_full, test_cold_feat_full, features_cold)

    # Final Birleştirme
    test_is_cold = test_warm_feat["is_cold"] == 1
    test_final_pred = np.where(test_is_cold, test_cold_pred, test_warm_pred)

    test_ceil = 36.0 * (raw_test["guc"].values + 1.0)
    test_final_pred = np.clip(test_final_pred, 0.0, test_ceil)

    sub_v17 = pd.DataFrame({"id": raw_test["id"], "tuketim": test_final_pred})
    sub_v17.to_csv(OUTPUT_V17_PATH, index=False)
    sha_v17 = get_sha256(OUTPUT_V17_PATH)

    logger.info("=" * 75)
    logger.info("🎉 V17 BREAKTHROUGH PIPELINE BAŞARIYLA TAMAMLANDI!")
    logger.info(f"✓ V17 Output File : {OUTPUT_V17_PATH} (SHA: {sha_v17})")
    logger.info(f"✓ Test Mean Tüketim: {test_final_pred.mean():.2f}")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_v17_pipeline()
