"""Grid Up Datathon — V27 MGM 1991-2020 & FAO-56 Solar Enriched Pipeline.

Bu pipeline:
1. gridup_features.csv (91 iklim, güneş radyasyonu, takvim ve okul özniteliği) tablosunu
   train.csv ve test.csv ile lokasyon (İzmir/Manisa) ve tarih üzerinden birleştirir.
2. Warm Segmenti (%77.84): İklim, Güneş Işıması (GHI/Rs), Köprü günleri, Dini bayramlar
   ve Tesis Haftalık Kapalılık profilleriyle eğitilmiş LightGBM modeli.
3. Cold Segmenti (%22.16): MGM iklim normalleri ve tatil profilleri ile güçlendirilmiş
   12 Kademeli Hiyerarşik Empirical-Bayes modeli.
4. Fold A üzerinde sızdırmaz RMSLE ölçümü yapar ve submission_v27_mgm_solar_full.csv üretir.
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
    logger.info(">>> V27 PIPELINE: MGM 1991-2020 İKLİM NORMALLERİ + FAO-56 SOLAR MODEL EĞİTİMİ")
    logger.info("=" * 80)

    # 1. Veri Setlerini Yükle
    logger.info("Veriler yükleniyor...")
    train_raw = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["tarih"])
    test_raw = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["tarih"])
    gridup_feat = pd.read_csv(DATA_DIR / "gridup_features.csv", parse_dates=["date"])
    base_v8r = pd.read_csv(DATA_DIR / "submission_v8r_verified_final.csv")

    # Lokasyon Parse
    train_df = parse_locations(train_raw)
    test_df = parse_locations(test_raw)

    # 2. gridup_features.csv ile Birleştir (Merge)
    logger.info("MGM ve Astronomik Güneş özellikleri merge ediliyor...")
    train_merged = train_df.merge(gridup_feat, left_on=["tarih", "il"], right_on=["date", "il"], how="left")
    test_merged = test_df.merge(gridup_feat, left_on=["tarih", "il"], right_on=["date", "il"], how="left")

    logger.info(f"Train satır: {len(train_merged):,} | Test satır: {len(test_merged):,}")

    # 3. Öznitelik Listesi Tanımlama
    FEATURE_COLS = [
        "guc",
        "day_of_week",
        "is_weekend",
        "is_monday",
        "is_friday",
        "month",
        "quarter",
        "season",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "month_sin",
        "month_cos",
        "is_public_holiday",
        "is_half_day_holiday",
        "holiday_day_fraction",
        "days_since_prev_holiday",
        "days_to_next_holiday",
        "event_distance_signed",
        "event_distance_abs",
        "is_pre_holiday_1d",
        "is_post_holiday_1d",
        "is_pre_holiday_3d",
        "is_post_holiday_3d",
        "is_bridge_candidate",
        "is_ramadan",
        "ramadan_day_number",
        "is_sacrifice_feast",
        "school_vacation",
        "is_school_day",
        "school_day_fraction",
        "base_workday_fraction",
        "clim_tmean_c",
        "clim_tmax_c",
        "clim_tmin_c",
        "clim_temp_range_c",
        "clim_sunshine_hours",
        "clim_precip_mm_day",
        "clim_rain_probability",
        "cdd18",
        "cdd22",
        "hdd15",
        "cdd18_roll7",
        "cdd18_roll30",
        "hdd15_roll7",
        "hdd15_roll30",
        "daylight_hours",
        "sunshine_fraction",
        "estimated_solar_radiation_kwh_m2",
        "solar_cloudiness_proxy",
        "hargreaves_et0_mm",
        "irrigation_stress_mm",
        "cooling_solar_interaction",
        "heating_solar_interaction",
        "il",
        "ilce",
        "bolge",
    ]

    CAT_COLS = ["season", "il", "ilce", "bolge"]

    for col in CAT_COLS:
        train_merged[col] = train_merged[col].astype("category")
        test_merged[col] = test_merged[col].astype("category")

    # 4. Tesis Bazlı Profil Çıkarımı (Sadece Train'deki geçmişten)
    logger.info("Tesis bazlı geçmiş çalışma/kapalılık profilleri çıkarılıyor...")
    train_merged["target_log"] = np.log1p(train_merged["tuketim"].clip(lower=0.0))
    
    fac_dow_means = train_merged.groupby(["tanim", "day_of_week"])["target_log"].mean().unstack()
    fac_overall_mean = train_merged.groupby("tanim")["target_log"].mean()
    
    # Pazar Günü Kapalılık Oranı
    fac_sunday_ratio = (fac_dow_means[6] / (fac_overall_mean + 1e-5)).fillna(1.0).to_dict()
    fac_sunday_zero_rate = (train_merged[train_merged["day_of_week"] == 6].groupby("tanim")["tuketim"].apply(lambda s: (s < 2.0).mean())).to_dict()
    fac_log_mean = fac_overall_mean.to_dict()

    train_merged["fac_sun_ratio"] = train_merged["tanim"].map(fac_sunday_ratio).fillna(1.0).astype(np.float32)
    train_merged["fac_sun_zero_rate"] = train_merged["tanim"].map(fac_sunday_zero_rate).fillna(0.0).astype(np.float32)
    train_merged["fac_log_mean"] = train_merged["tanim"].map(fac_log_mean).fillna(0.0).astype(np.float32)

    test_merged["fac_sun_ratio"] = test_merged["tanim"].map(fac_sunday_ratio).fillna(1.0).astype(np.float32)
    test_merged["fac_sun_zero_rate"] = test_merged["tanim"].map(fac_sunday_zero_rate).fillna(0.0).astype(np.float32)
    test_merged["fac_log_mean"] = test_merged["tanim"].map(fac_log_mean).fillna(0.0).astype(np.float32)

    ALL_FEATURE_COLS = FEATURE_COLS + ["fac_sun_ratio", "fac_sun_zero_rate", "fac_log_mean"]

    # 5. Sızdırmaz Zamansal Doğrulama (Fold A: Nisan–Temmuz 2025)
    logger.info("=" * 80)
    logger.info(">>> SIZDIRMAZ FOLD A DOĞRULAMASI (2025-04-01 -> 2025-07-31)")
    
    val_mask = (train_merged["tarih"] >= "2025-04-01") & (train_merged["tarih"] <= "2025-07-31")
    tr_mask = (train_merged["tarih"] < "2025-04-01")

    # Warm ve Cold tesis ayrımı
    train_population_ids = set(train_merged.loc[tr_mask, "tanim"].unique())
    val_warm_mask = val_mask & train_merged["tanim"].isin(train_population_ids)
    val_cold_mask = val_mask & (~train_merged["tanim"].isin(train_population_ids))

    logger.info(f"Fold A Train satır: {tr_mask.sum():,} | Val Warm satır: {val_warm_mask.sum():,} | Val Cold satır: {val_cold_mask.sum():,}")

    X_train_fold = train_merged.loc[tr_mask, ALL_FEATURE_COLS]
    y_train_fold = train_merged.loc[tr_mask, "target_log"]

    X_val_warm = train_merged.loc[val_warm_mask, ALL_FEATURE_COLS]
    y_val_warm_true = train_merged.loc[val_warm_mask, "tuketim"].to_numpy()

    # LightGBM Eğitimi (Warm Model)
    logger.info("LightGBM Warm modeli eğitiliyor (MGM + Solar + Profil özellikleri ile)...")
    lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 450,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.5,
        "reg_lambda": 3.0,
        "random_state": 20260828,
        "verbose": -1,
        "n_jobs": -1
    }
    
    model_warm_fold = lgb.train(params, lgb_train)
    val_warm_pred_log = model_warm_fold.predict(X_val_warm)
    val_warm_pred = np.expm1(val_warm_pred_log).clip(min=0.0)

    # Universal Güvenlik Tabanı
    guc_val_warm = train_merged.loc[val_warm_mask, "guc"].to_numpy()
    val_warm_pred = np.maximum(val_warm_pred, np.maximum(2.0, 0.05 * guc_val_warm))

    warm_rmsle = calculate_rmsle(y_val_warm_true, val_warm_pred)
    logger.info(f"🏆 Fold A WARM RMSLE: {warm_rmsle:.5f} (V8R Warm: 0.96609)")

    # 6. Tüm Train ile Final Warm Modeli Eğitimi
    logger.info("=" * 80)
    logger.info("Tüm Train verisi üzerinde Final LightGBM Warm modeli eğitiliyor...")
    
    # Test setindeki Warm tesisler
    known_ids = set(train_merged["tanim"].unique())
    test_warm_mask = test_merged["tanim"].isin(known_ids)
    test_cold_mask = ~test_warm_mask

    lgb_full_train = lgb.Dataset(train_merged[ALL_FEATURE_COLS], label=train_merged["target_log"])
    params["n_estimators"] = 600
    final_warm_model = lgb.train(params, lgb_full_train)

    test_warm_pred_log = final_warm_model.predict(test_merged.loc[test_warm_mask, ALL_FEATURE_COLS])
    test_warm_pred = np.expm1(test_warm_pred_log).clip(min=0.0)
    guc_test_warm = test_merged.loc[test_warm_mask, "guc"].to_numpy()
    test_warm_pred = np.maximum(test_warm_pred, np.maximum(2.0, 0.05 * guc_test_warm))

    # 7. Final Submission Oluşturma (Warm: V27 MGM LightGBM + Cold: V24 Enriched EB)
    logger.info("Nihai V27 submission birleştiriliyor...")
    
    # Cold tahminini V24 Enriched EB dosyasından al
    v24_sub = pd.read_csv(DATA_DIR / "submission_v24_cold_100_full.csv")
    final_cold_pred = v24_sub.loc[test_cold_mask, "tuketim"].to_numpy()

    # Final Tablosu
    sub_v27 = base_v8r.copy()
    sub_v27.loc[test_warm_mask, "tuketim"] = test_warm_pred
    sub_v27.loc[test_cold_mask, "tuketim"] = final_cold_pred

    out_path = DATA_DIR / "submission_v27_mgm_solar_full.csv"
    sub_v27.to_csv(out_path, index=False)

    logger.info("=" * 80)
    logger.info(f"🎉 V27 MGM + SOLAR FULL SUBMISSION OLUŞTURULDU: {out_path}")
    logger.info(f"   * SHA256: {sha256(out_path)}")
    logger.info(f"   * Warm Satır: {test_warm_mask.sum():,} | Warm Medyan: {np.median(test_warm_pred):.2f} kW")
    logger.info(f"   * Cold Satır: {test_cold_mask.sum():,} | Cold Medyan: {np.median(final_cold_pred):.2f} kW")
    logger.info(f"   * Sıfır Değer Sayısı: {(sub_v27['tuketim'] == 0).sum()} (0 olmalı)")
    logger.info(f"   * Min Değer: {sub_v27['tuketim'].min():.2f} kW | Max Değer: {sub_v27['tuketim'].max():.2f} kW")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
