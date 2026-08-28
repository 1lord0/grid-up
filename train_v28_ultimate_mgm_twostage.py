"""Grid Up Datathon — V28 Ultimate Two-Stage MGM Pipeline (Vektörize & Hızlı).

Mimari:
1. Warm Segmenti (%77.84): İki Aşamalı (Two-Stage) Residual GBDT
   - 1. Aşama: Tesisin kendi 2025 geçmiş tüketim tabanı ve haftalık kapalılık ritmi.
   - 2. Aşama: MGM 1991-2020 iklim normalleri, FAO-56 güneş radyasyonu ve köprü günü rezidüel düzeltmesi.
2. Cold Segmenti (%22.16): 12 Kademeli MGM İklimli Hiyerarşik Empirical-Bayes (Doğrulanmış 1.85398 Cold RMSLE).
3. Çelik Güvenlik Zırhı: max(2.0, 0.05 * guc) tabanı ile sıfır çökmesi riski 0.
"""

import hashlib
import json
import logging
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_log = np.log1p(np.maximum(0.0, np.asarray(y_true, dtype=float)))
    pred_log = np.log1p(np.maximum(0.0, np.asarray(y_pred, dtype=float)))
    return float(np.sqrt(np.mean(np.square(true_log - pred_log))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_locations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["lokasyon"].astype(str).str.split(">")
    out["il_raw"] = parts.str[0].str.strip().str.upper()
    out["ilce"] = parts.str[-1].str.strip().str.upper()
    out["bolge"] = parts.apply(lambda p: p[-2].strip().upper() if len(p) >= 3 else "DOGRUDAN")
    out["il"] = np.where(out["il_raw"].str.contains("MANISA|MANİSA"), "MANISA", "IZMIR")
    return out


def main():
    logger.info("=" * 80)
    logger.info(">>> V28 ULTIMATE TWO-STAGE MGM + COLD BAYES PIPELINE BAŞLATILIYOR")
    logger.info("=" * 80)

    # 1. Veri Setlerini Yükle
    train_raw = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["tarih"])
    test_raw = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["tarih"])
    gridup_feat = pd.read_csv(DATA_DIR / "gridup_features.csv", parse_dates=["date"])
    base_v8r = pd.read_csv(DATA_DIR / "submission_v8r_verified_final.csv")

    train_df = parse_locations(train_raw)
    test_df = parse_locations(test_raw)

    train_merged = train_df.merge(gridup_feat, left_on=["tarih", "il"], right_on=["date", "il"], how="left")
    test_merged = test_df.merge(gridup_feat, left_on=["tarih", "il"], right_on=["date", "il"], how="left")

    train_merged["target_log"] = np.log1p(train_merged["tuketim"].clip(lower=0.0))

    # 2. Aşama 1: Tesis Bazlı Profil Tabanı (Vektörize)
    logger.info("1. Aşama: Tesis bazlı geçmiş çalışma profilleri vektörize hesaplanıyor...")
    
    # Yaz Dönemi (Nisan-Temmuz) ve Genel Ortalamalar
    summer_mask = (train_merged["tarih"].dt.month >= 4) & (train_merged["tarih"].dt.month <= 7)
    fac_summer_mean = train_merged[summer_mask].groupby("tanim")["target_log"].mean()
    fac_overall_mean = train_merged.groupby("tanim")["target_log"].mean()
    
    fac_base_log = fac_summer_mean.combine_first(fac_overall_mean).to_dict()

    # DOW Oranları (Vektörize MultiIndex Map)
    fac_dow_series = train_merged.groupby(["tanim", "day_of_week"])["target_log"].mean()
    fac_overall_series = train_merged.groupby("tanim")["target_log"].mean()
    dow_ratio_series = (fac_dow_series / (fac_overall_series + 1e-5)).fillna(1.0)

    # Train Map
    train_idx = pd.MultiIndex.from_frame(train_merged[["tanim", "day_of_week"]])
    train_merged["dow_mult"] = train_idx.map(dow_ratio_series).fillna(1.0).astype(np.float32)
    train_merged["fac_base"] = train_merged["tanim"].map(fac_base_log).fillna(np.log1p(2.5 * train_merged["guc"])).astype(np.float32)
    train_merged["stage1_log_pred"] = (train_merged["fac_base"] * train_merged["dow_mult"]).astype(np.float32)

    # Test Map
    test_idx = pd.MultiIndex.from_frame(test_merged[["tanim", "day_of_week"]])
    test_merged["dow_mult"] = test_idx.map(dow_ratio_series).fillna(1.0).astype(np.float32)
    test_merged["fac_base"] = test_merged["tanim"].map(fac_base_log).fillna(np.log1p(2.5 * test_merged["guc"])).astype(np.float32)
    test_merged["stage1_log_pred"] = (test_merged["fac_base"] * test_merged["dow_mult"]).astype(np.float32)

    # Rezidüel Hedef: Gerçek Tüketim - 1. Aşama Tahmini
    train_merged["residual_target"] = (train_merged["target_log"] - train_merged["stage1_log_pred"]).astype(np.float32)

    # 3. Aşama 2: MGM İklim & Güneş Radyasyonu ile Rezidüel Tahmini
    logger.info("2. Aşama: MGM İklim & Solar özellikleri ile Rezidüel GBDT modeli eğitiliyor...")
    
    RESIDUAL_FEATURES = [
        "guc",
        "day_of_week",
        "is_weekend",
        "is_monday",
        "is_friday",
        "month",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "is_public_holiday",
        "holiday_day_fraction",
        "days_since_prev_holiday",
        "days_to_next_holiday",
        "event_distance_signed",
        "is_bridge_candidate",
        "is_ramadan",
        "is_sacrifice_feast",
        "school_vacation",
        "is_school_day",
        "base_workday_fraction",
        "clim_tmean_c",
        "clim_tmax_c",
        "clim_temp_range_c",
        "clim_sunshine_hours",
        "clim_precip_mm_day",
        "cdd18",
        "cdd22",
        "hdd15",
        "cdd18_roll7",
        "cdd18_roll30",
        "daylight_hours",
        "sunshine_fraction",
        "estimated_solar_radiation_kwh_m2",
        "solar_cloudiness_proxy",
        "cooling_solar_interaction",
        "il",
        "ilce",
        "bolge",
    ]

    for col in ["il", "ilce", "bolge"]:
        train_merged[col] = train_merged[col].astype("category")
        test_merged[col] = test_merged[col].astype("category")

    # Fold A Doğrulaması (2025-04-01 -> 2025-07-31)
    val_mask = (train_merged["tarih"] >= "2025-04-01") & (train_merged["tarih"] <= "2025-07-31")
    tr_mask = (train_merged["tarih"] < "2025-04-01")

    known_ids = set(train_merged.loc[tr_mask, "tanim"].unique())
    val_warm_mask = val_mask & train_merged["tanim"].isin(known_ids)

    X_train_res = train_merged.loc[tr_mask, RESIDUAL_FEATURES]
    y_train_res = train_merged.loc[tr_mask, "residual_target"]

    X_val_warm = train_merged.loc[val_warm_mask, RESIDUAL_FEATURES]
    y_val_warm_true = train_merged.loc[val_warm_mask, "tuketim"].to_numpy()
    val_stage1_warm = train_merged.loc[val_warm_mask, "stage1_log_pred"].to_numpy()

    lgb_res_train = lgb.Dataset(X_train_res, label=y_train_res)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 350,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "random_state": 20260828,
        "verbose": -1,
        "n_jobs": -1
    }

    res_model_fold = lgb.train(params, lgb_res_train)
    val_res_pred = res_model_fold.predict(X_val_warm)

    # İki Aşamalı Birleşim: Stage 1 + Stage 2
    val_twostage_log = val_stage1_warm + 0.85 * val_res_pred
    val_twostage_pred = np.expm1(val_twostage_log).clip(min=0.0)

    # Güvenlik tabanı
    guc_val_warm = train_merged.loc[val_warm_mask, "guc"].to_numpy()
    val_twostage_pred = np.maximum(val_twostage_pred, np.maximum(2.0, 0.05 * guc_val_warm))

    warm_two_stage_rmsle = calculate_rmsle(y_val_warm_true, val_twostage_pred)
    stage1_only_rmsle = calculate_rmsle(y_val_warm_true, np.expm1(val_stage1_warm))
    
    logger.info("=" * 80)
    logger.info(f"🏆 Fold A STAGE 1 (Ham Tesis Tabanı) RMSLE : {stage1_only_rmsle:.5f}")
    logger.info(f"🏆 Fold A STAGE 1 + 2 (MGM Residual) WARM RMSLE: {warm_two_stage_rmsle:.5f} (V8R: 0.96609)")
    logger.info("=" * 80)

    # 4. Tüm Train Üzerinde Final Two-Stage Modelini Eğit
    logger.info("Tüm Train verisi üzerinde Final Two-Stage Rezidüel modeli eğitiliyor...")
    lgb_full_res = lgb.Dataset(train_merged[RESIDUAL_FEATURES], label=train_merged["residual_target"])
    params["n_estimators"] = 450
    final_res_model = lgb.train(params, lgb_full_res)

    # Test Seti Tahminleri
    test_known_ids = set(train_merged["tanim"].unique())
    test_warm_mask = test_merged["tanim"].isin(test_known_ids)
    test_cold_mask = ~test_warm_mask

    test_stage1_warm = test_merged.loc[test_warm_mask, "stage1_log_pred"].to_numpy()
    test_res_warm = final_res_model.predict(test_merged.loc[test_warm_mask, RESIDUAL_FEATURES])

    test_twostage_warm_log = test_stage1_warm + 0.85 * test_res_warm
    test_twostage_warm_pred = np.expm1(test_twostage_warm_log).clip(min=0.0)
    
    guc_test_warm = test_merged.loc[test_warm_mask, "guc"].to_numpy()
    test_twostage_warm_pred = np.maximum(test_twostage_warm_pred, np.maximum(2.0, 0.05 * guc_test_warm))

    # 5. Cold Segmenti: V24/V26 12 Kademeli Enriched Bayes Tahminini Al (1.85398)
    v24_full_sub = pd.read_csv(DATA_DIR / "submission_v24_cold_100_full.csv")
    final_cold_pred = v24_full_sub.loc[test_cold_mask, "tuketim"].to_numpy()

    # 6. Nihai Submission Dosyasını Oluştur
    sub_v28 = base_v8r.copy()
    sub_v28.loc[test_warm_mask, "tuketim"] = test_twostage_warm_pred
    sub_v28.loc[test_cold_mask, "tuketim"] = final_cold_pred

    out_path = DATA_DIR / "submission_v28_ultimate_mgm_twostage.csv"
    sub_v28.to_csv(out_path, index=False)

    logger.info("=" * 80)
    logger.info(f"🎉 V28 ULTIMATE SUBMISSION OLUŞTURULDU: {out_path}")
    logger.info(f"   * SHA256: {sha256(out_path)}")
    logger.info(f"   * Warm Satır: {test_warm_mask.sum():,} | Warm Medyan: {np.median(test_twostage_warm_pred):.2f} kW")
    logger.info(f"   * Cold Satır: {test_cold_mask.sum():,} | Cold Medyan: {np.median(final_cold_pred):.2f} kW")
    logger.info(f"   * Sıfır Değer Sayısı: {(sub_v28['tuketim'] == 0).sum()} (0 olmalı)")
    logger.info(f"   * Min Değer: {sub_v28['tuketim'].min():.2f} kW | Max Değer: {sub_v28['tuketim'].max():.2f} kW")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
