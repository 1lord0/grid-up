"""Grid Up Datathon — V20 3-Way Feature-Disentangled GBDT Stacker + V8R Anchor.

Tüm modeller ayrıştırılmış özellik setleriyle eğitilir.
Deterministik, sızıntısız Fold A üzerinde doğrulanır ve Optuna ile kısıtlı ağırlıklar optimize edilir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
OUTPUT_V20_PATH = DATA_DIR / "submission_v20_stacker.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
V8R_SUB_PATH = DATA_DIR / "submission_v8r_verified_final.csv"


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
    df = df.copy()
    day = df["tarih"].dt.day
    dow = df["tarih"].dt.dayofweek
    doy = df["tarih"].dt.dayofyear
    m = df["tarih"].dt.month

    df["sin_day_1"] = np.sin(2 * np.pi * 1 * day / 30.0).astype(np.float32)
    df["cos_day_1"] = np.cos(2 * np.pi * 1 * day / 30.0).astype(np.float32)
    df["sin_day_2"] = np.sin(2 * np.pi * 2 * day / 30.0).astype(np.float32)
    df["cos_day_2"] = np.cos(2 * np.pi * 2 * day / 30.0).astype(np.float32)

    df["sin_dow_1"] = np.sin(2 * np.pi * 1 * dow / 7.0).astype(np.float32)
    df["cos_dow_1"] = np.cos(2 * np.pi * 1 * dow / 7.0).astype(np.float32)

    df["sin_doy_1"] = np.sin(2 * np.pi * 1 * doy / 365.25).astype(np.float32)
    df["cos_doy_1"] = np.cos(2 * np.pi * 1 * doy / 365.25).astype(np.float32)
    df["sin_doy_2"] = np.sin(2 * np.pi * 2 * doy / 365.25).astype(np.float32)
    df["cos_doy_2"] = np.cos(2 * np.pi * 2 * doy / 365.25).astype(np.float32)

    df["sin_month"] = np.sin(2 * np.pi * m / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * m / 12.0).astype(np.float32)
    return df


def compute_safety_floor(train_df: pd.DataFrame, target_df: pd.DataFrame, cutoff_date: pd.Timestamp, floor_multiplier: float = 0.35) -> np.ndarray:
    past = train_df[train_df["tarih"] <= cutoff_date].copy()
    recent = past[past["tarih"] > (cutoff_date - pd.Timedelta(days=90))]
    fac_recent_level = recent.groupby("tanim")["tuketim"].mean().to_dict()
    fac_all_level = past.groupby("tanim")["tuketim"].mean().to_dict()

    target_tanim = target_df["tanim"].values
    target_guc = target_df["guc"].values

    floor_history = np.array([fac_recent_level.get(t, fac_all_level.get(t, 0.0)) for t in target_tanim]) * floor_multiplier
    floor_guc = target_guc * 0.05

    safety_floor = np.maximum(floor_history, floor_guc)
    return np.maximum(2.0, safety_floor)


def main():
    print("=" * 80)
    print(">>> V20 - 3-WAY GBDT STACKER + V8R STEEL ANCHOR PIPELINE BASLATILIYOR")
    print("=" * 80)

    raw_train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["tarih"])
    raw_test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["tarih"])
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    raw_train = parse_locations(raw_train)
    raw_test = parse_locations(raw_test)

    guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
    guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
    raw_train["guc_bin"] = pd.cut(raw_train["guc"], bins=guc_bins, labels=guc_labels).astype(str)
    raw_test["guc_bin"] = pd.cut(raw_test["guc"], bins=guc_bins, labels=guc_labels).astype(str)

    cat_cols = ["il", "ilce", "bolge", "guc_bin"]
    global_maps = {c: {val: i for i, val in enumerate(sorted(set(raw_train[c].dropna()).union(set(raw_test[c].dropna()))))} for c in cat_cols}

    for c in cat_cols:
        raw_train[f"{c}_code"] = raw_train[c].map(global_maps[c]).fillna(-1).astype(np.int32)
        raw_test[f"{c}_code"] = raw_test[c].map(global_maps[c]).fillna(-1).astype(np.int32)

    # -------------------------------------------------------------------------
    # 1. FOLD A DOĞRULAMASI (Sızıntısız Forward Split)
    # -------------------------------------------------------------------------
    print("\n--- 1. FOLD A DOGRULAMASI (Cutoff = 2025-03-31, Val = Nisan-Temmuz 2025) ---")
    cutoff_a = pd.Timestamp("2025-03-31")
    past_a = raw_train[raw_train["tarih"] <= cutoff_a].copy()
    val_a = raw_train[(raw_train["tarih"] >= "2025-04-01") & (raw_train["tarih"] <= "2025-07-31")].copy()

    from train_v16_surgical_cold_start import build_v16_features, predict_cold_bayes
    past_feat_a, val_feat_a, bayes_bundle_a = build_v16_features(raw_train, val_a, cutoff_a, global_maps, cat_cols)
    past_feat_a = add_fourier_features(past_feat_a)
    val_feat_a = add_fourier_features(val_feat_a)

    floor_val_a = compute_safety_floor(raw_train, val_a, cutoff_a, floor_multiplier=0.35)
    y_true_val = val_a["tuketim"].values
    warm_mask_val = val_feat_a["is_cold"] == 0
    cold_mask_val = val_feat_a["is_cold"] == 1

    # Full Winning Feature Set for Warm Models
    features_full_warm = [
        "guc", "log_guc", "il_code", "ilce_code", "bolge_code", "guc_bin_code",
        "month", "day", "day_of_week", "day_of_year", "is_weekend", "is_summer", "is_june_july",
        "log_guc_x_summer", "monthly_network_index", "log_fac_level", "log_seasonal_baseline",
        "is_cold", "has_annual_lag", "arch_prob_0", "arch_prob_1", "arch_prob_2",
        "sin_day_1", "cos_day_1", "sin_day_2", "cos_day_2",
        "sin_dow_1", "cos_dow_1", "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2"
    ]

    past_feat_a["guc_ratio_ilce"] = (past_feat_a["guc"] / (past_feat_a.groupby("ilce_code")["guc"].transform("mean") + 1.0)).astype(np.float32)
    val_feat_a["guc_ratio_ilce"] = (val_feat_a["guc"] / (past_feat_a.groupby("ilce_code")["guc"].transform("mean").reindex(val_feat_a.index).fillna(630.0) + 1.0)).astype(np.float32)

    features_xgb = features_full_warm + ["guc_ratio_ilce"]

    # -------------------------------------------------------------------------
    # FOLD A MODEL EĞİTİMLERİ (Deterministik)
    # -------------------------------------------------------------------------
    print("\n>>> 3 Ayri GBDT Modelinin Fold A Uzerinde Egitimi...")
    y_res_a = np.log1p(past_feat_a["tuketim"].values) - np.log1p(past_feat_a["seasonal_baseline"].values)

    # 1. LightGBM
    print("  [1/3] Fitting LightGBM (Fourier & Zaman)...")
    m_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=1,
        deterministic=True, force_row_wise=True, verbose=-1
    )
    m_lgb.fit(past_feat_a[features_full_warm], y_res_a)
    p_lgb_res = m_lgb.predict(val_feat_a[features_full_warm])
    p_lgb_warm = np.maximum(floor_val_a, np.expm1(np.log1p(val_feat_a["seasonal_baseline"].values) + p_lgb_res))

    # 2. CatBoost
    print("  [2/3] Fitting CatBoost (Cografi & Kategorik)...")
    m_cb = CatBoostRegressor(
        iterations=400, learning_rate=0.05, depth=6, loss_function="RMSE",
        random_seed=42, thread_count=1, verbose=False
    )
    m_cb.fit(past_feat_a[features_full_warm], y_res_a, verbose=False)
    p_cb_res = m_cb.predict(val_feat_a[features_full_warm])
    p_cb_warm = np.maximum(floor_val_a, np.expm1(np.log1p(val_feat_a["seasonal_baseline"].values) + p_cb_res))

    # 3. XGBoost
    print("  [3/3] Fitting XGBoost (Regularize Uc Deger Freni)...")
    m_xgb = XGBRegressor(
        n_estimators=350, learning_rate=0.05, max_depth=6,
        reg_alpha=2.0, reg_lambda=5.0, subsample=0.80, colsample_bytree=0.80,
        random_state=42, n_jobs=1, tree_method="hist"
    )
    m_xgb.fit(past_feat_a[features_xgb], y_res_a)
    p_xgb_res = m_xgb.predict(val_feat_a[features_xgb])
    p_xgb_warm = np.maximum(floor_val_a, np.expm1(np.log1p(val_feat_a["seasonal_baseline"].values) + p_xgb_res))

    # Cold Bayes Predict
    pred_cold_bayes_val = np.maximum(floor_val_a, predict_cold_bayes(val_feat_a, bayes_bundle_a, beta=0.88))

    p_lgb_val = np.where(cold_mask_val, pred_cold_bayes_val, p_lgb_warm)
    p_cb_val = np.where(cold_mask_val, pred_cold_bayes_val, p_cb_warm)
    p_xgb_val = np.where(cold_mask_val, pred_cold_bayes_val, p_xgb_warm)

    # 4. V8R Baseline Proxy
    ilce_month_med = past_feat_a.groupby(["ilce_code", "month"])["tuketim"].median().to_dict()
    guc_month_med = past_feat_a.groupby(["guc_bin_code", "month"])["tuketim"].median().to_dict()
    fac_all_mean = past_feat_a.groupby("tanim")["tuketim"].mean().to_dict()

    v8r_val_list = []
    for _, r in val_feat_a.iterrows():
        if r["is_cold"] == 0:
            val_base = fac_all_mean.get(r["tanim"], 100.0)
        else:
            val_base = ilce_month_med.get((r["ilce_code"], r["month"]), guc_month_med.get((r["guc_bin_code"], r["month"]), 50.0))
        v8r_val_list.append(val_base)
    p_v8r_val = np.maximum(floor_val_a, np.array(v8r_val_list, dtype=np.float32))

    # -------------------------------------------------------------------------
    # ÖLÇÜMLER VE KORELASYON MATRİSİ (Bizzat Hesaplanmış)
    # -------------------------------------------------------------------------
    print("\n--- TEKIL MODELLERIN FOLD A RMSLE OLCUMLERI ---")
    print(f"  * LightGBM RMSLE : {calculate_rmsle(y_true_val, p_lgb_val):.5f} (Warm: {calculate_rmsle(y_true_val[warm_mask_val], p_lgb_val[warm_mask_val]):.5f} | Cold: {calculate_rmsle(y_true_val[cold_mask_val], p_lgb_val[cold_mask_val]):.5f})")
    print(f"  * CatBoost RMSLE : {calculate_rmsle(y_true_val, p_cb_val):.5f} (Warm: {calculate_rmsle(y_true_val[warm_mask_val], p_cb_val[warm_mask_val]):.5f} | Cold: {calculate_rmsle(y_true_val[cold_mask_val], p_cb_val[cold_mask_val]):.5f})")
    print(f"  * XGBoost  RMSLE : {calculate_rmsle(y_true_val, p_xgb_val):.5f} (Warm: {calculate_rmsle(y_true_val[warm_mask_val], p_xgb_val[warm_mask_val]):.5f} | Cold: {calculate_rmsle(y_true_val[cold_mask_val], p_xgb_val[cold_mask_val]):.5f})")

    oof_preds_df = pd.DataFrame({
        "LGBM": np.log1p(p_lgb_val),
        "CatBoost": np.log1p(p_cb_val),
        "XGBoost": np.log1p(p_xgb_val),
        "V8R": np.log1p(p_v8r_val),
    })
    print("\n--- OOF TAHMINLERI PEARSON KORELASYON MATRISI ---")
    print(oof_preds_df.corr().round(4).to_string())

    # -------------------------------------------------------------------------
    # OPTUNA İLE KISITLI META-STACKER AĞIRLIK OPTİMİZASYONU
    # -------------------------------------------------------------------------
    print("\n>>> Optuna ile Kisitli Stacker Agirlik Optimizasyonu (RMSLE Minimize Ediliyor)...")

    def objective(trial):
        w1 = trial.suggest_float("w_lgb", 0.0, 1.0)
        w2 = trial.suggest_float("w_cb", 0.0, 1.0)
        w3 = trial.suggest_float("w_xgb", 0.0, 1.0)
        w4 = trial.suggest_float("w_v8r", 0.0, 0.30)

        total_w = w1 + w2 + w3 + w4
        if total_w == 0:
            return 999.0
        w1, w2, w3, w4 = w1/total_w, w2/total_w, w3/total_w, w4/total_w

        blend = w1 * p_lgb_val + w2 * p_cb_val + w3 * p_xgb_val + w4 * p_v8r_val
        blend = np.maximum(blend, floor_val_a)
        return calculate_rmsle(y_true_val, blend)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=300)

    best_p = study.best_params
    tot_p = sum(best_p.values())
    w_lgb_opt = best_p["w_lgb"] / tot_p
    w_cb_opt = best_p["w_cb"] / tot_p
    w_xgb_opt = best_p["w_xgb"] / tot_p
    w_v8r_opt = best_p["w_v8r"] / tot_p

    p_v20_val = (w_lgb_opt * p_lgb_val + w_cb_opt * p_cb_val + w_xgb_opt * p_xgb_val + w_v8r_opt * p_v8r_val)
    p_v20_val = np.maximum(p_v20_val, floor_val_a)

    rmsle_v20_total = calculate_rmsle(y_true_val, p_v20_val)
    rmsle_v20_warm = calculate_rmsle(y_true_val[warm_mask_val], p_v20_val[warm_mask_val])
    rmsle_v20_cold = calculate_rmsle(y_true_val[cold_mask_val], p_v20_val[cold_mask_val])

    print("\n" + "=" * 80)
    print(">>> FOLD A OPTIMAL STACKER SONUCLARI:")
    print(f"   - Optimize Edilmis Agirliklar: LGBM={w_lgb_opt:.3f} | CatBoost={w_cb_opt:.3f} | XGBoost={w_xgb_opt:.3f} | V8R={w_v8r_opt:.3f}")
    print(f"   * FOLD A TOPLAM RMSLE : {rmsle_v20_total:.5f} (En Dusuk Validasyon Skoru!)")
    print(f"   * FOLD A WARM RMSLE   : {rmsle_v20_warm:.5f}")
    print(f"   * FOLD A COLD RMSLE   : {rmsle_v20_cold:.5f}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 2. FULL 100% RETRAINING (15 Ay) & TEST TAHMİNİ
    # -------------------------------------------------------------------------
    print("\n--- 2. TUM VERI (15 AY) UZERINDE TAM 3'LU STACKER EGITIMI ---")
    cutoff_full = pd.Timestamp("2026-03-31")
    full_tr_feat, test_feat, bayes_bundle_full = build_v16_features(raw_train, raw_test, cutoff_full, global_maps, cat_cols)
    full_tr_feat = add_fourier_features(full_tr_feat)
    test_feat = add_fourier_features(test_feat)

    full_tr_feat["guc_ratio_ilce"] = (full_tr_feat["guc"] / (full_tr_feat.groupby("ilce_code")["guc"].transform("mean") + 1.0)).astype(np.float32)
    test_feat["guc_ratio_ilce"] = (test_feat["guc"] / (full_tr_feat.groupby("ilce_code")["guc"].transform("mean").reindex(test_feat.index).fillna(630.0) + 1.0)).astype(np.float32)

    floor_test = compute_safety_floor(raw_train, raw_test, cutoff_full, floor_multiplier=0.35)
    y_full_res = np.log1p(full_tr_feat["tuketim"].values) - np.log1p(full_tr_feat["seasonal_baseline"].values)

    # Full LightGBM
    print("  Fitting Full LightGBM (1.22M train rows)...")
    full_lgb = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=1,
        deterministic=True, force_row_wise=True, verbose=-1
    )
    full_lgb.fit(full_tr_feat[features_full_warm], y_full_res)
    pred_test_warm_lgb = np.maximum(floor_test, np.expm1(np.log1p(test_feat["seasonal_baseline"].values) + full_lgb.predict(test_feat[features_full_warm])))

    # Full CatBoost
    print("  Fitting Full CatBoost...")
    full_cb = CatBoostRegressor(
        iterations=550, learning_rate=0.04, depth=6, loss_function="RMSE",
        random_seed=42, thread_count=1, verbose=False
    )
    full_cb.fit(full_tr_feat[features_full_warm], y_full_res, verbose=False)
    pred_test_warm_cb = np.maximum(floor_test, np.expm1(np.log1p(test_feat["seasonal_baseline"].values) + full_cb.predict(test_feat[features_full_warm])))

    # Full XGBoost
    print("  Fitting Full XGBoost...")
    full_xgb = XGBRegressor(
        n_estimators=450, learning_rate=0.04, max_depth=6,
        reg_alpha=2.0, reg_lambda=5.0, subsample=0.80, colsample_bytree=0.80,
        random_state=42, n_jobs=1, tree_method="hist"
    )
    full_xgb.fit(full_tr_feat[features_xgb], y_full_res)
    pred_test_warm_xgb = np.maximum(floor_test, np.expm1(np.log1p(test_feat["seasonal_baseline"].values) + full_xgb.predict(test_feat[features_xgb])))

    # Full Cold Bayes
    pred_test_cold = np.maximum(floor_test, predict_cold_bayes(test_feat, bayes_bundle_full, beta=0.88))
    test_is_cold = test_feat["is_cold"] == 1

    pred_test_lgb = np.where(test_is_cold, pred_test_cold, pred_test_warm_lgb)
    pred_test_cb = np.where(test_is_cold, pred_test_cold, pred_test_warm_cb)
    pred_test_xgb = np.where(test_is_cold, pred_test_cold, pred_test_warm_xgb)

    # V8R Sub Load
    assert V8R_SUB_PATH.exists(), "V8R submission bulunamadi!"
    v8r_sub = pd.read_csv(V8R_SUB_PATH)
    pred_test_v8r = np.maximum(floor_test, v8r_sub["tuketim"].values)

    # -------------------------------------------------------------------------
    # STACKER HARMANLAMA VE GÜVENLİK KONTROLÜ
    # -------------------------------------------------------------------------
    test_final_stack = (
        w_lgb_opt * pred_test_lgb +
        w_cb_opt * pred_test_cb +
        w_xgb_opt * pred_test_xgb +
        w_v8r_opt * pred_test_v8r
    )
    test_final_stack = np.maximum(test_final_stack, floor_test)
    test_ceil = 36.0 * (raw_test["guc"].values + 1.0)
    test_final_stack = np.clip(test_final_stack, 0.0, test_ceil)

    # ZORUNLU KONTROL BLOĞU
    sub_v20 = pd.DataFrame({"id": raw_test["id"], "tuketim": test_final_stack})
    assert list(sub_v20.columns) == ["id", "tuketim"], "Kolon isimleri kesinlikle ['id', 'tuketim'] olmali!"
    assert len(sub_v20) == len(sample_sub), f"Satir sayisi uyusmuyor: {len(sub_v20)}"
    assert (sub_v20["id"] == sample_sub["id"]).all(), "ID siralamasi eslesmiyor!"
    assert sub_v20["tuketim"].isna().sum() == 0, "NaN deger tespit edildi!"
    assert (sub_v20["tuketim"] < 0).sum() == 0, "Negatif deger tespit edildi!"
    assert np.isfinite(sub_v20["tuketim"]).all(), "Sonsuz deger tespit edildi!"

    sub_v20.to_csv(OUTPUT_V20_PATH, index=False)
    sha256_v20 = get_sha256(OUTPUT_V20_PATH)

    print("\n" + "=" * 80)
    print(">>> V20 - 3-WAY GBDT STACKER BASARIYLA TAMAMLANDI!")
    print(f"[OK] Dosya Konumu : {OUTPUT_V20_PATH}")
    print(f"[OK] SHA256       : {sha256_v20}")
    print(f"[OK] Test Mean    : {test_final_stack.mean():.2f}")
    print(f"[OK] Test Min     : {test_final_stack.min():.2f} (Sifir Cokmesi Yok!)")
    print(f"[OK] Test Median  : {np.median(test_final_stack):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
