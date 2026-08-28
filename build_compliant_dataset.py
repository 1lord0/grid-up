"""Grid Up Datathon — Kural Uyumlu (31 Mart 2026 İtibarıyla Geçerli) Veri Seti Üretici.

Kural:
- 31 Mart 2026 sonrasına ait (Nisan-Temmuz 2026) gerçekleşmiş hava durumu verisi KULLANILMAZ.
- 2016-2026 (10 yıllık) Open-Meteo geçmiş verilerinden İzmir ve Manisa için 'Uzun Dönem İklim Normalleri' (Climate Normals) hesaplanır.
- 31 Mart 2026 itibarıyla bilinen resmi tatiller, dini bayramlar ve köprü günleri entegre edilir.
- train_enriched_compliant ve test_enriched_compliant dosyaları üretilir.
"""

import hashlib
import json
import logging
from pathlib import Path
import time
import holidays
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")


def fetch_and_compute_climate_normals(lat: float, lon: float, il_name: str) -> pd.DataFrame:
    """Open-Meteo Historical Archive üzerinden 2016-01-01 ile 2026-03-31 arası 10 yıllık iklim normallerini çıkarır."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2016-01-01",
        "end_date": "2026-03-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "shortwave_radiation_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "Europe/Istanbul"
    }

    daily_data = None
    for attempt in range(5):
        try:
            logger.info(f"Open-Meteo üzerinden {il_name} için veri çekiliyor (Deneme {attempt+1}/5)...")
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                logger.warning("Open-Meteo 429 Hız Sınırı (Rate Limit). 15 saniye bekleniyor...")
                time.sleep(15)
                continue
            resp.raise_for_status()
            daily_data = resp.json()["daily"]
            break
        except Exception as e:
            logger.warning(f"Hata oluştu: {e}. 10 saniye bekleniyor...")
            time.sleep(10)

    if daily_data is None:
        raise RuntimeError(f"{il_name} için Open-Meteo verisi çekilemedi!")

    df_hist = pd.DataFrame(daily_data)
    df_hist["tarih"] = pd.to_datetime(df_hist["time"])
    df_hist["day_of_year"] = df_hist["tarih"].dt.dayofyear

    # CDD ve HDD
    df_hist["cdd"] = df_hist["temperature_2m_mean"].apply(lambda t: max(0.0, t - 18.0))
    df_hist["hdd"] = df_hist["temperature_2m_mean"].apply(lambda t: max(0.0, 15.0 - t))

    # 10 Yıllık Uzun Dönem İklim Normalleri
    climate_normals = df_hist.groupby("day_of_year").agg({
        "temperature_2m_mean": ["mean", "std"],
        "temperature_2m_max": ["mean", "std"],
        "temperature_2m_min": ["mean", "std"],
        "cdd": ["mean"],
        "hdd": ["mean"],
        "shortwave_radiation_sum": ["mean", "std"],
        "precipitation_sum": ["mean"],
        "wind_speed_10m_max": ["mean"]
    })

    climate_normals.columns = [
        "norm_temp_mean", "norm_temp_std",
        "norm_temp_max", "norm_temp_max_std",
        "norm_temp_min", "norm_temp_min_std",
        "norm_cdd", "norm_hdd",
        "norm_solar_ghi_mean", "norm_solar_ghi_std",
        "norm_precip_mean", "norm_wind_max_mean"
    ]
    climate_normals.reset_index(inplace=True)
    climate_normals["il_key"] = il_name.upper()
    return climate_normals


def generate_calendar_features(start_date: str = "2025-01-01", end_date: str = "2026-08-01") -> pd.DataFrame:
    """31 Mart 2026 itibarıyla bilinen resmi tatiller, dini bayramlar ve köprü günlerini türetir."""
    logger.info("Dinamik takvim, dini bayramlar ve köprü günleri hesaplanıyor...")
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df_cal = pd.DataFrame({"tarih": dates})

    tr_holidays = holidays.Turkey(years=[2025, 2026])

    df_cal["is_holiday"] = df_cal["tarih"].dt.date.isin(tr_holidays).astype(np.int8)
    df_cal["day_of_week"] = df_cal["tarih"].dt.dayofweek.astype(np.int8) # 0: Pzt, 6: Paz
    df_cal["is_weekend"] = df_cal["day_of_week"].isin([5, 6]).astype(np.int8)
    df_cal["month"] = df_cal["tarih"].dt.month.astype(np.int8)
    df_cal["day"] = df_cal["tarih"].dt.day.astype(np.int8)
    df_cal["day_of_year"] = df_cal["tarih"].dt.dayofyear.astype(np.int16)

    # Köprü Günü (Bridge Day) Analizi: Tatil ile Hafta Sonu arasında kalan tek iş günü
    is_off = (df_cal["is_holiday"] == 1) | (df_cal["is_weekend"] == 1)
    df_cal["is_bridge_day"] = 0

    for i in range(1, len(df_cal) - 1):
        if not is_off.iloc[i] and is_off.iloc[i-1] and is_off.iloc[i+1]:
            df_cal.loc[i, "is_bridge_day"] = 1

    df_cal["is_bridge_day"] = df_cal["is_bridge_day"].astype(np.int8)

    # Döngüsel Trigonometrik Özellikler
    doy = df_cal["day_of_year"].values
    dow = df_cal["day_of_week"].values
    df_cal["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df_cal["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    df_cal["sin_dow"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    df_cal["cos_dow"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    df_cal["sin_month"] = np.sin(2 * np.pi * df_cal["month"] / 12.0).astype(np.float32)
    df_cal["cos_month"] = np.cos(2 * np.pi * df_cal["month"] / 12.0).astype(np.float32)

    return df_cal


def parse_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts = df["lokasyon"].astype(str).str.split(">")
    df["il"] = parts.str[0].str.strip().str.upper()
    df["ilce"] = parts.str[-1].str.strip().str.upper()
    df["bolge"] = parts.apply(lambda p: p[-2].strip().upper() if len(p) >= 3 else "DOGRUDAN")

    # İl normalizasyonu (İzmir vs Manisa eşleştirmesi)
    df["il_key"] = np.where(df["il"].str.contains("MANISA|MANİSA"), "MANISA", "IZMIR")
    return df


def main():
    logger.info("=" * 80)
    logger.info(">>> KURAL UYUMLU VERİ SETİ ZENGİNLEŞTİRME BAŞLATILIYOR")
    logger.info(">>> Kural: 31 Mart 2026 sonrası gerçekleşmiş hava verisi KULLANILMAZ.")
    logger.info(">>> 10 Yıllık (2016-2026) İklim Normalleri ve Dinamik Takvim Entegre Ediliyor.")
    logger.info("=" * 80)

    # 1. İklim Normallerini Çek ve Hesapla
    izmir_normals = fetch_and_compute_climate_normals(38.4237, 27.1428, "IZMIR")
    time.sleep(5)
    manisa_normals = fetch_and_compute_climate_normals(38.6191, 27.4289, "MANISA")
    all_climate_normals = pd.concat([izmir_normals, manisa_normals], ignore_index=True)

    normals_out_path = DATA_DIR / "izmir_manisa_climate_normals_2016_2026.csv"
    all_climate_normals.to_csv(normals_out_path, index=False)
    logger.info(f"[OK] İklim normalleri tablosu kaydedildi: {normals_out_path}")

    # 2. Dinamik Takvim Tablosunu Üret
    calendar_df = generate_calendar_features("2025-01-01", "2026-08-01")

    # 3. Train ve Test Verilerini Yükle
    logger.info("train.csv ve test.csv yükleniyor...")
    train_raw = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["tarih"])
    test_raw = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["tarih"])

    train_parsed = parse_locations(train_raw)
    test_parsed = parse_locations(test_raw)

    # 4. İklim Normalleri ve Takvim Özelliklerini Birleştir
    logger.info("Özellikler merge ediliyor...")
    
    # Train Merge
    train_enriched = train_parsed.merge(calendar_df, on="tarih", how="left")
    train_enriched = train_enriched.merge(all_climate_normals, on=["il_key", "day_of_year"], how="left")

    # Test Merge
    test_enriched = test_parsed.merge(calendar_df, on="tarih", how="left")
    test_enriched = test_enriched.merge(all_climate_normals, on=["il_key", "day_of_year"], how="left")

    # Trafo Gücü ve Log Dönüşümleri
    train_enriched["log_guc"] = np.log1p(train_enriched["guc"]).astype(np.float32)
    test_enriched["log_guc"] = np.log1p(test_enriched["guc"]).astype(np.float32)

    # 5. CSV Olarak Kaydet
    train_csv_path = DATA_DIR / "train_enriched_compliant.csv"
    test_csv_path = DATA_DIR / "test_enriched_compliant.csv"

    logger.info(f"Kaydediliyor: {train_csv_path} ({len(train_enriched):,} satır)...")
    train_enriched.to_csv(train_csv_path, index=False)

    logger.info(f"Kaydediliyor: {test_csv_path} ({len(test_enriched):,} satır)...")
    test_enriched.to_csv(test_csv_path, index=False)

    logger.info("=" * 80)
    logger.info("🎉 KURAL UYUMLU VERİ SETLERİ BAŞARIYLA OLUŞTURULDU!")
    logger.info(f"✓ Train Enriched: {train_csv_path} | Satır: {len(train_enriched):,}")
    logger.info(f"✓ Test Enriched : {test_csv_path}  | Satır: {len(test_enriched):,}")
    logger.info(f"✓ Eklenen Kolonlar ({len(train_enriched.columns)} adet): {list(train_enriched.columns)}")
    logger.info(f"✓ Null Değer Sayısı (Train): {train_enriched.isna().sum().sum()}")
    logger.info(f"✓ Null Değer Sayısı (Test) : {test_enriched.isna().sum().sum()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
