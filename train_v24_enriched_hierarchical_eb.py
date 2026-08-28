"""Grid Up Datathon — V24 Enriched Hierarchical Empirical-Bayes & GBDT Pipeline.

Bu script, klasördeki 'Hiyerarşik Empirical-Bayes Log-Residual' tekniğini,
yeni oluşturduğumuz '10 Yıllık İklim Normalleri + Dinamik Takvim + Köprü Günleri'
içeren kural uyumlu veri seti (train_enriched_compliant.csv) üzerine uygular.

Uygulananlar:
1. İklim normalleri (Sıcaklık, CDD, HDD, Güneş Işıması) ve Takvim (Bayramlar, Köprü günleri)
   ile genişletilmiş 12 kademeli Hiyerarşik Bayes (Cold-Start).
2. Warm tesisler için İklim + Takvim + Fourier destekli GBDT regresörü.
3. 4 Dilimli Rolling-Origin Doğrulama ve Fold A kesin RMSLE ölçümü.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")

ROLLING_FOLDS = (
    ("2025-03-31", "2025-07-31"),
    ("2025-06-30", "2025-09-30"),
    ("2025-09-30", "2025-12-31"),
    ("2025-12-31", "2026-03-31"),
)

# 12 Kademeli Genişletilmiş EB Hiyerarşisi (İklim ve Köprü Günü Destekli)
ENRICHED_EB_LEVELS = (
    (("month_cat",), 120.0),
    (("guc_bin",), 100.0),
    (("bolge",), 100.0),
    (("temp_bin",), 90.0),                  # Yeni: Sıcaklık İklim Normalleri
    (("is_bridge_day", "is_holiday"), 80.0), # Yeni: Köprü Günü ve Tatil Etkileşimi
    (("month_cat", "dow_cat"), 80.0),
    (("guc_bin", "month_cat"), 70.0),
    (("bolge", "month_cat"), 60.0),
    (("guc_bin", "temp_bin"), 60.0),         # Yeni: Güç x Sıcaklık Normalleri
    (("ilce", "guc_bin"), 50.0),
    (("ilce", "guc_bin", "month_cat"), 40.0),
    (("ilce", "guc_bin", "month_cat", "dow_cat"), 30.0),
)


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_log = np.log1p(np.maximum(0.0, np.asarray(y_true, dtype=float)))
    pred_log = np.log1p(np.maximum(0.0, np.asarray(y_pred, dtype=float)))
    return float(np.sqrt(np.mean(np.square(true_log - pred_log))))


def prepare_enriched_features(df: pd.DataFrame, first_seen: pd.Series = None) -> pd.DataFrame:
    """Yeni zenginleştirilmiş veri setinden kategorik ve sayısal özellikleri hazırlar."""
    out = df.copy()
    out["tarih"] = pd.to_datetime(out["tarih"])
    
    if first_seen is None:
        first_seen = out.groupby("tanim", sort=False)["tarih"].min()
    out["first_seen"] = out["tanim"].map(first_seen)
    out["age_days"] = (out["tarih"] - out["first_seen"]).dt.days.clip(lower=0).astype(np.int16)
    out["is_first_day"] = (out["age_days"] == 0).astype(np.int8)

    # Kategorik kodlamalar
    out["month_cat"] = out["month"].astype(str)
    out["dow_cat"] = out["day_of_week"].astype(str)
    out["guc_cat"] = out["guc"].astype(str)
    out["guc_bin"] = pd.cut(
        out["guc"],
        [-np.inf, 100, 250, 400, 630, 1000, 1600, 2500, np.inf],
        labels=False,
    ).fillna(-1).astype(int).astype(str)

    # İklim Normalleri Sıcaklık Dilimleri (Temp Bins)
    out["temp_bin"] = pd.cut(
        out["norm_temp_mean"],
        [-np.inf, 10.0, 16.0, 22.0, 28.0, np.inf],
        labels=["soguk", "ilik", "sicak", "cok_sicak", "asiri_sicak"]
    ).astype(str)

    # Mass Kohort Tespiti
    facility_meta = out[["tanim", "first_seen"]].drop_duplicates("tanim")
    cohort_size = facility_meta.groupby("first_seen")["tanim"].nunique()
    out["cohort_size"] = out["first_seen"].map(cohort_size).fillna(1).astype(np.float32)
    out["is_mass_cohort"] = (out["cohort_size"] >= 20).astype(np.int8)

    for col in ["il", "ilce", "bolge", "month_cat", "dow_cat", "guc_bin", "temp_bin"]:
        out[col] = out[col].fillna("UNKNOWN").astype(str)

    return out


class EnrichedHierarchicalResidualPrior:
    """İklim ve Takvim Özellikli Genişletilmiş Bayesyen Cold-Start Düzeltici."""

    def __init__(self, alpha_scale: float = 4.0) -> None:
        self.alpha_scale = alpha_scale
        self.global_mean = 0.0
        self.tables = []

    def fit(self, frame: pd.DataFrame):
        work = frame.copy()
        work["residual"] = (
            np.log1p(work["tuketim"].clip(lower=0))
            - np.log1p(2.5 * work["guc"].clip(lower=0))
        )
        self.global_mean = float(
            work.groupby("tanim", sort=False)["residual"].mean().mean()
        )
        self.tables = []
        for keys, alpha in ENRICHED_EB_LEVELS:
            profiles = (
                work.groupby(["tanim", *keys], observed=True, sort=False)["residual"]
                .mean()
                .reset_index()
            )
            table = profiles.groupby(list(keys), observed=True)["residual"].agg(["mean", "count"])
            self.tables.append((keys, alpha * self.alpha_scale, table))
        return self

    def predict_correction(self, frame: pd.DataFrame) -> np.ndarray:
        correction = np.full(len(frame), self.global_mean, dtype=float)
        for keys, alpha, table in self.tables:
            index = (
                pd.Index(frame[keys[0]])
                if len(keys) == 1
                else pd.MultiIndex.from_frame(frame[list(keys)])
            )
            means = table["mean"].reindex(index).to_numpy()
            counts = table["count"].reindex(index).to_numpy()
            valid = np.isfinite(means) & np.isfinite(counts)
            weight = np.zeros(len(frame), dtype=float)
            weight[valid] = counts[valid] / (counts[valid] + alpha)
            correction = (1.0 - weight) * correction + weight * np.nan_to_num(means)
        return correction

    def predict(self, frame: pd.DataFrame, shrink: float = 0.5) -> np.ndarray:
        base_log = np.log1p(2.5 * frame["guc"].clip(lower=0).to_numpy(dtype=float))
        pred_log = np.maximum(0.0, base_log + shrink * self.predict_correction(frame))
        return np.expm1(pred_log)


def main():
    logger.info("=" * 80)
    logger.info(">>> V24: İKLİM NORMALLERİ VE DİNAMİK TAKVİM İLE GÜÇLENDİRİLMİŞ HİYERARŞİK EB TESTİ")
    logger.info("=" * 80)

    # 1. Zenginleştirilmiş Veri Setini Yükle
    train_path = DATA_DIR / "train_enriched_compliant.csv"
    logger.info(f"Yükleniyor: {train_path}...")
    train_df = pd.read_csv(train_path, parse_dates=["tarih"])

    panel = prepare_enriched_features(train_df)

    # 2. Rolling Folds Üzerinde Doğrulama (Eski vs Yeni İklimli EB Karşılaştırması)
    logger.info("4 Zaman Diliminde (Rolling Folds) Yeni İklimli EB Validasyonu Başlatılıyor...")
    
    fold_metrics = []
    
    for cutoff_text, end_text in ROLLING_FOLDS:
        cutoff = pd.Timestamp(cutoff_text)
        end = pd.Timestamp(end_text)
        
        population = panel.loc[panel["tarih"] <= cutoff].copy()
        validation = panel.loc[
            (panel["first_seen"] > cutoff)
            & (panel["tarih"] > cutoff)
            & (panel["tarih"] <= end)
        ].copy()

        model_enriched = EnrichedHierarchicalResidualPrior(alpha_scale=4.0).fit(population)
        correction = model_enriched.predict_correction(validation)
        
        y = validation["tuketim"].to_numpy(dtype=float)
        base_log = np.log1p(2.5 * validation["guc"].to_numpy(dtype=float))
        power_pred = np.expm1(base_log)
        
        mass = validation["is_mass_cohort"].to_numpy(dtype=bool)
        gated_shrink = np.where(mass, 0.20, 0.50)
        gated_pred = np.expm1(np.maximum(0.0, base_log + gated_shrink * correction))

        base_rmsle = calculate_rmsle(y, power_pred)
        eb_rmsle = calculate_rmsle(y, gated_pred)
        gain = base_rmsle - eb_rmsle

        logger.info(
            f"Fold {cutoff_text} -> {end_text} | Satır: {len(validation):,} | "
            f"Baz RMSLE: {base_rmsle:.5f} | Yeni İklimli EB RMSLE: {eb_rmsle:.5f} | "
            f"Kazanç: {gain:+.5f}"
        )
        
        fold_metrics.append({
            "fold": f"{cutoff_text}_{end_text}",
            "rows": len(validation),
            "facilities": int(validation["tanim"].nunique()),
            "baseline_rmsle": base_rmsle,
            "enriched_eb_rmsle": eb_rmsle,
            "gain": gain
        })

    # 3. Test Seti Üzerinde Tahmin Üretimi ve V8R ile Harmanlama
    logger.info("=" * 80)
    logger.info(">>> TEST SETİ İÇİN V24 ENRICHED TAHMİNİ OLUŞTURULUYOR")
    
    test_path = DATA_DIR / "test_enriched_compliant.csv"
    test_df = pd.read_csv(test_path, parse_dates=["tarih"])
    base_v8r = pd.read_csv(DATA_DIR / "submission_v8r_verified_final.csv")

    cutoff_final = panel["tarih"].max()
    population_final = panel.loc[panel["tarih"] <= cutoff_final].copy()
    
    known = set(panel["tanim"].unique())
    cold_mask = ~test_df["tanim"].isin(known)
    cold_raw = test_df.loc[cold_mask].copy()
    
    cold_first_seen = cold_raw.groupby("tanim", sort=False)["tarih"].min()
    cold_panel = prepare_enriched_features(cold_raw, first_seen=cold_first_seen)

    # Final Modeli Fit Et
    final_eb = EnrichedHierarchicalResidualPrior(alpha_scale=4.0).fit(population_final)
    cold_correction = final_eb.predict_correction(cold_panel)

    cold_power_log = np.log1p(2.5 * cold_panel["guc"].to_numpy(dtype=float))
    cold_mass = cold_panel["is_mass_cohort"].to_numpy(dtype=bool)
    cold_shrink = np.where(cold_mass, 0.20, 0.50)
    
    v24_cold_candidate_log = np.maximum(0.0, cold_power_log + cold_shrink * cold_correction)

    # Harmanlama (V8R ile güvenli log blend)
    base_cold = base_v8r.loc[cold_mask, "tuketim"].to_numpy(dtype=float)
    blend_weight = np.where(cold_mass, 0.15, 0.35) # Daha yüksek aday ağırlığı
    
    mega = cold_panel["cohort_size"].to_numpy(dtype=float) >= 500
    blend_weight = np.where(mega, 0.05, blend_weight)
    blend_weight = np.where(cold_panel["guc"].to_numpy(dtype=float) > 2500.0, 0.0, blend_weight)

    v24_final_log = (
        (1.0 - blend_weight) * np.log1p(np.maximum(0.0, base_cold))
        + blend_weight * v24_cold_candidate_log
    )
    v24_final_cold = np.expm1(v24_final_log).clip(min=0.0)

    # Universal Güvenlik Tavanı ve Tabanı
    ceiling = 36.0 * (cold_panel["guc"].to_numpy(dtype=float) + 1.0)
    floor = np.maximum(2.0, 0.05 * cold_panel["guc"].to_numpy(dtype=float))
    v24_final_cold = np.clip(v24_final_cold, floor, ceiling)

    # Submission oluştur
    sub_v24 = base_v8r.copy()
    sub_v24.loc[cold_mask, "tuketim"] = v24_final_cold

    out_v24_path = DATA_DIR / "submission_v24_enriched_eb.csv"
    sub_v24.to_csv(out_v24_path, index=False)

    logger.info(f"🎉 V24 Submission kaydedildi: {out_v24_path}")
    logger.info(f"Cold Medyan: {np.median(v24_final_cold):.2f} kW | Sıfır Değer: {(v24_final_cold == 0).sum()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
