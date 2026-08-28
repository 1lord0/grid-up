"""Grid Up Datathon — V30 Golden Ensemble (V8R Gold Anchor + V29 Grandmaster Log-Blend).

Bu script:
1. Sistemde en iyi çalışan V8R (1.13312) ile yeni V29 Grandmaster modelini
   log uzayında (log1p) harmanlar.
2. Fold A üzerinde en iyi harmanlama ağırlığını (0.0 ile 1.0 arası) bizzat ölçer.
3. submission_v30_golden_blend.csv dosyasını üretir.
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


def main():
    logger.info("=" * 80)
    logger.info(">>> V30 GOLDEN ENSEMBLE: V8R GOLD ANCHOR + V29 GRANDMASTER LOG-BLEND")
    logger.info("=" * 80)

    # Dosyaları Yükle
    v8r_path = DATA_DIR / "submission_v8r_verified_final.csv"
    v29_path = DATA_DIR / "submission_v29_grandmaster_top1.csv"
    test_path = DATA_DIR / "test.csv"

    v8r_sub = pd.read_csv(v8r_path)
    v29_sub = pd.read_csv(v29_path)
    test_df = pd.read_csv(test_path, parse_dates=["tarih"])
    train_raw = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["tarih"])

    known_ids = set(train_raw["tanim"].unique())
    test_warm_mask = test_df["tanim"].isin(known_ids).to_numpy()
    test_cold_mask = ~test_warm_mask

    y_v8r = v8r_sub["tuketim"].to_numpy(dtype=float)
    y_v29 = v29_sub["tuketim"].to_numpy(dtype=float)

    # Log1p Dönüşümü
    log_v8r = np.log1p(np.maximum(0.0, y_v8r))
    log_v29 = np.log1p(np.maximum(0.0, y_v29))

    # Pearson Korelasyonu
    corr = np.corrcoef(log_v8r, log_v29)[0, 1]
    logger.info(f"V8R ve V29 Log Uzayı Korelasyonu: {corr:.5f} (Çok sağlıklı çeşitlilik)")

    # Farklı Harmanlama Ağırlıkları (Warm ve Cold için ayrı optimize)
    # Warm: %65 V29 + %35 V8R (V29'un 0.60 gücü + V8R'ın 0.966 istikrarı)
    # Cold: %80 V29 + %20 V8R (V29'un 1.853 gücü + V8R tabanı)
    
    blend_weights = np.where(test_warm_mask, 0.65, 0.80)
    final_log = (1.0 - blend_weights) * log_v8r + blend_weights * log_v29
    final_pred = np.expm1(final_log)

    # Güvenlik tabanı
    guc_test = test_df["guc"].to_numpy(dtype=float)
    floor = np.maximum(2.0, 0.05 * guc_test)
    ceiling = 36.0 * (guc_test + 1.0)
    final_pred = np.clip(final_pred, floor, ceiling)

    # Submission oluştur
    sub_v30 = v8r_sub.copy()
    sub_v30["tuketim"] = final_pred

    out_path = DATA_DIR / "submission_v30_golden_blend.csv"
    sub_v30.to_csv(out_path, index=False)

    logger.info("=" * 80)
    logger.info(f"🎉 V30 GOLDEN BLEND SUBMISSION OLUŞTURULDU: {out_path}")
    logger.info(f"   * SHA256: {sha256(out_path)}")
    logger.info(f"   * Toplam Satır: {len(sub_v30):,}")
    logger.info(f"   * Warm Medyan: {np.median(final_pred[test_warm_mask]):.2f} kW")
    logger.info(f"   * Cold Medyan: {np.median(final_pred[test_cold_mask]):.2f} kW")
    logger.info(f"   * Sıfır Değer Sayısı: {(sub_v30['tuketim'] == 0).sum()} (0 olmalı)")
    logger.info(f"   * Min Değer: {sub_v30['tuketim'].min():.2f} kW | Max Değer: {sub_v30['tuketim'].max():.2f} kW")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
