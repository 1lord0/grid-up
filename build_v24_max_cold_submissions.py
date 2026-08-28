"""Grid Up Datathon — V24 Max Cold-Start Submissions.

Bu script, Cold tesislerde (%22.16) İklim Normalleri + Dinamik Takvim destekli
Hiyerarşik Empirical-Bayes gücünü maksimize eden iki yüksek performanslı submission üretir:
1. submission_v24_cold_100_full.csv (Cold: %100 V24 EB | Warm: %100 V8R)
2. submission_v24_cold_85_blend.csv (Cold: %85 V24 EB + %15 V8R | Warm: %100 V8R)
"""

import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")

ENRICHED_EB_LEVELS = (
    (("month_cat",), 120.0),
    (("guc_bin",), 100.0),
    (("bolge",), 100.0),
    (("temp_bin",), 90.0),                  # Sıcaklık İklim Normalleri
    (("is_bridge_day", "is_holiday"), 80.0), # Köprü Günü ve Tatil Etkileşimi
    (("month_cat", "dow_cat"), 80.0),
    (("guc_bin", "month_cat"), 70.0),
    (("bolge", "month_cat"), 60.0),
    (("guc_bin", "temp_bin"), 60.0),         # Güç x Sıcaklık Normalleri
    (("ilce", "guc_bin"), 50.0),
    (("ilce", "guc_bin", "month_cat"), 40.0),
    (("ilce", "guc_bin", "month_cat", "dow_cat"), 30.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_enriched_features(df: pd.DataFrame, first_seen: pd.Series = None) -> pd.DataFrame:
    out = df.copy()
    out["tarih"] = pd.to_datetime(out["tarih"])
    if first_seen is None:
        first_seen = out.groupby("tanim", sort=False)["tarih"].min()
    out["first_seen"] = out["tanim"].map(first_seen)
    out["month_cat"] = out["month"].astype(str)
    out["dow_cat"] = out["day_of_week"].astype(str)
    out["guc_bin"] = pd.cut(
        out["guc"],
        [-np.inf, 100, 250, 400, 630, 1000, 1600, 2500, np.inf],
        labels=False,
    ).fillna(-1).astype(int).astype(str)
    out["temp_bin"] = pd.cut(
        out["norm_temp_mean"],
        [-np.inf, 10.0, 16.0, 22.0, 28.0, np.inf],
        labels=["soguk", "ilik", "sicak", "cok_sicak", "asiri_sicak"]
    ).astype(str)
    facility_meta = out[["tanim", "first_seen"]].drop_duplicates("tanim")
    cohort_size = facility_meta.groupby("first_seen")["tanim"].nunique()
    out["cohort_size"] = out["first_seen"].map(cohort_size).fillna(1).astype(np.float32)
    out["is_mass_cohort"] = (out["cohort_size"] >= 20).astype(np.int8)
    for col in ["il", "ilce", "bolge", "month_cat", "dow_cat", "guc_bin", "temp_bin"]:
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


class EnrichedHierarchicalResidualPrior:
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


def main():
    logger.info("=" * 80)
    logger.info(">>> V24 MAX COLD-START SUBMISSION ÜRETİMİ")
    logger.info("=" * 80)

    train_path = DATA_DIR / "train_enriched_compliant.csv"
    test_path = DATA_DIR / "test_enriched_compliant.csv"
    v8r_path = DATA_DIR / "submission_v8r_verified_final.csv"

    logger.info("Veri setleri yükleniyor...")
    train_df = pd.read_csv(train_path, parse_dates=["tarih"])
    test_df = pd.read_csv(test_path, parse_dates=["tarih"])
    base_v8r = pd.read_csv(v8r_path)

    panel = prepare_enriched_features(train_df)
    known = set(panel["tanim"].unique())
    cold_mask = ~test_df["tanim"].isin(known)
    cold_raw = test_df.loc[cold_mask].copy()

    cold_first_seen = cold_raw.groupby("tanim", sort=False)["tarih"].min()
    cold_panel = prepare_enriched_features(cold_raw, first_seen=cold_first_seen)

    logger.info(f"Popülasyon: {len(panel):,} | Cold Test Satırı: {len(cold_panel):,} ({cold_panel['tanim'].nunique()} tesis)")

    # Modeli Eğit ve Düzeltmeyi Tahmin Et
    model = EnrichedHierarchicalResidualPrior(alpha_scale=4.0).fit(panel)
    cold_correction = model.predict_correction(cold_panel)

    cold_power_log = np.log1p(2.5 * cold_panel["guc"].to_numpy(dtype=float))
    cold_mass = cold_panel["is_mass_cohort"].to_numpy(dtype=bool)
    cold_shrink = np.where(cold_mass, 0.20, 0.50)

    # Saf %100 V24 Aday Tahmini
    v24_candidate_log = np.maximum(0.0, cold_power_log + cold_shrink * cold_correction)
    base_cold = base_v8r.loc[cold_mask, "tuketim"].to_numpy(dtype=float)

    ceiling = 36.0 * (cold_panel["guc"].to_numpy(dtype=float) + 1.0)
    floor = np.maximum(2.0, 0.05 * cold_panel["guc"].to_numpy(dtype=float))

    # -------------------------------------------------------------
    # 1. DOSYA: %100 PURE COLD V24 (Tam Bağımsız Zirve Model)
    # -------------------------------------------------------------
    cold_100_pred = np.expm1(v24_candidate_log).clip(min=0.0)
    cold_100_pred = np.clip(cold_100_pred, floor, ceiling)

    sub_100 = base_v8r.copy()
    sub_100.loc[cold_mask, "tuketim"] = cold_100_pred
    
    path_100 = DATA_DIR / "submission_v24_cold_100_full.csv"
    sub_100.to_csv(path_100, index=False)
    logger.info(f"[TAMAMLANDI] 1. Dosya (%100 Cold V24): {path_100}")
    logger.info(f"   * SHA256: {sha256(path_100)}")
    logger.info(f"   * Cold Medyan: {np.median(cold_100_pred):.2f} kW | Min: {cold_100_pred.min():.2f} | Max: {cold_100_pred.max():.2f}")

    # -------------------------------------------------------------
    # 2. DOSYA: %85 HIGH-WEIGHT BLEND (Ultra-Güçlü + %15 V8R Tabanı)
    # -------------------------------------------------------------
    w_85 = np.where(cold_mass, 0.40, 0.85)
    blend_85_log = (1.0 - w_85) * np.log1p(np.maximum(0.0, base_cold)) + w_85 * v24_candidate_log
    cold_85_pred = np.expm1(blend_85_log).clip(min=0.0)
    cold_85_pred = np.clip(cold_85_pred, floor, ceiling)

    sub_85 = base_v8r.copy()
    sub_85.loc[cold_mask, "tuketim"] = cold_85_pred

    path_85 = DATA_DIR / "submission_v24_cold_85_blend.csv"
    sub_85.to_csv(path_85, index=False)
    logger.info(f"[TAMAMLANDI] 2. Dosya (%85 Cold V24 Blend): {path_85}")
    logger.info(f"   * SHA256: {sha256(path_85)}")
    logger.info(f"   * Cold Medyan: {np.median(cold_85_pred):.2f} kW | Min: {cold_85_pred.min():.2f} | Max: {cold_85_pred.max():.2f}")

    logger.info("=" * 80)
    logger.info("🎉 HER İKİ MAX-COLD SUBMISSION DOSYASI DA KUSURSUZ ŞEKİLDE OLUŞTURULDU!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
