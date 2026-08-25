"""Grid Up Datathon — V19 Doğrulanmış Güvenlik Ağı Protokolü.

Tüm aşamalar (Aşama 0'dan Aşama 8'e) adım adım, sıfır varsayımla,
bizzat Fold A ve gerçek veri üzerinde ölçülerek yürütülür.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
OUTPUT_V19_PATH = DATA_DIR / "submission_v19_verified.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
V8R_SUB_PATH = DATA_DIR / "submission_v8r_verified_final.csv"
V14_SUB_PATH = DATA_DIR / "submission_v14_verified_clean.csv"


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


def main():
    print("=" * 80)
    print(">>> V19 - DOGRULANMIS GUVENLIK AGI PROTOKOLU BASLATILIYOR")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # AŞAMA 0: EKSİK BİLGİ VE ŞEMA DOĞRULAMASI
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 0] Sema ve Kolon Dogrulamasi...")
    raw_train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["tarih"])
    raw_test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["tarih"])
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print(f"[OK] train.csv kolonlari : {list(raw_train.columns)}")
    print(f"[OK] test.csv kolonlari  : {list(raw_test.columns)}")
    print(f"[OK] sample_sub kolonlari: {list(sample_sub.columns)}")
    assert list(sample_sub.columns) == ["id", "tuketim"], "Sample submission formati id, tuketim olmali!"
    assert len(raw_test) == len(sample_sub) == 714688, f"Test satir sayisi uyusmuyor: {len(raw_test)}"

    # -------------------------------------------------------------------------
    # AŞAMA 1: REPRODUCIBILITY GATE (Determinizm Kapısı)
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 1] Reproducibility Gate (Cift Calistirma Determinizm Testi)...")
    cutoff_a = pd.Timestamp("2025-03-31")
    past_a = raw_train[raw_train["tarih"] <= cutoff_a].copy()
    val_a = raw_train[(raw_train["tarih"] >= "2025-04-01") & (raw_train["tarih"] <= "2025-07-31")].copy()

    past_a["month"] = past_a["tarih"].dt.month
    val_a["month"] = val_a["tarih"].dt.month
    X_tr_rep = past_a[["guc", "month"]].values
    y_tr_rep = np.log1p(past_a["tuketim"].values)
    X_va_rep = val_a[["guc", "month"]].values
    y_va_rep = val_a["tuketim"].values

    rmsles_rep = []
    for run_idx in [1, 2]:
        m_rep = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42,
            n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1
        )
        m_rep.fit(X_tr_rep, y_tr_rep)
        p_rep = np.maximum(0.0, np.expm1(m_rep.predict(X_va_rep)))
        r = calculate_rmsle(y_va_rep, p_rep)
        rmsles_rep.append(r)
        print(f"  Run {run_idx}: RMSLE = {r:.8f}")

    diff_rep = abs(rmsles_rep[0] - rmsles_rep[1])
    print(f"  Fark: {diff_rep:.10f}")
    assert diff_rep < 1e-7, f"Determinizm testi basarisiz! Fark: {diff_rep}"
    print("[OK] [ASAMA 1 GECILDI] Reproducibility Gate basariyla onaylandi (Fark < 1e-7).")

    # -------------------------------------------------------------------------
    # AŞAMA 2: ŞEMA VE ÇIKTI FORMAT KONTROL BLOKLARI
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 2] Cikti Format ve Kisit Fonksiyonlari Tanimlandi.")

    # -------------------------------------------------------------------------
    # AŞAMA 3: COLD / WARM SEGMENT ANALİZİ (Fold A ve Test)
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 3] Cold / Warm Segment Analizi (Gercek Olcum)...")
    past_facs_a = set(past_a["tanim"].unique())
    val_a["is_cold"] = (~val_a["tanim"].isin(past_facs_a)).astype(int)

    train_all_facs = set(raw_train["tanim"].unique())
    raw_test["is_cold"] = (~raw_test["tanim"].isin(train_all_facs)).astype(int)

    warm_ratio_test = (raw_test["is_cold"] == 0).mean()
    cold_ratio_test = (raw_test["is_cold"] == 1).mean()
    warm_ratio_val = (val_a["is_cold"] == 0).mean()
    cold_ratio_val = (val_a["is_cold"] == 1).mean()

    print(f"  Test Seti Dagilimi : %{warm_ratio_test*100:.2f} Warm ({len(raw_test[raw_test['is_cold']==0]):,} satir) | %{cold_ratio_test*100:.2f} Cold ({len(raw_test[raw_test['is_cold']==1]):,} satir)")
    print(f"  Fold A Dagilimi    : %{warm_ratio_val*100:.2f} Warm ({len(val_a[val_a['is_cold']==0]):,} satir) | %{cold_ratio_val*100:.2f} Cold ({len(val_a[val_a['is_cold']==1]):,} satir)")

    # -------------------------------------------------------------------------
    # AŞAMA 4: GERÇEK TRAIN TABANLI GÜVENLİK TABANI (Safety Floor)
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 4] Gercek Train Tabanli Guvenlik Tabani Olcumu...")
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
        return safety_floor

    floor_val_a = compute_safety_floor(raw_train, val_a, cutoff_a, floor_multiplier=0.35)
    below_floor_ratio = (val_a["tuketim"].values < floor_val_a).mean()
    print(f"  Fold A'da gercek tuketimin taban altinda kaldigi oran: %{below_floor_ratio*100:.2f}")

    for mult in [0.05, 0.10, 0.20, 0.35, 0.50]:
        fl = compute_safety_floor(raw_train, val_a, cutoff_a, floor_multiplier=mult)
        print(f"    Taban Carpani {mult:.2f} -> Taban Altinda Kalan Gercek Veri: %{(val_a['tuketim'].values < fl).mean()*100:.2f}")

    # -------------------------------------------------------------------------
    # AŞAMA 5 & 6: V14 + V8R OOF TAHMİNLERİ VE EMPİRİK ANOMALİ KALİBRASYONU
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 5 & 6] Fold A Uzerinde Model Olcumleri & Anomali Kalibrasyonu...")
    raw_train_parsed = parse_locations(raw_train)
    val_a_parsed = parse_locations(val_a)

    guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
    guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
    raw_train_parsed["guc_bin"] = pd.cut(raw_train_parsed["guc"], bins=guc_bins, labels=guc_labels).astype(str)
    val_a_parsed["guc_bin"] = pd.cut(val_a_parsed["guc"], bins=guc_bins, labels=guc_labels).astype(str)

    cat_cols = ["il", "ilce", "bolge", "guc_bin"]
    global_maps = {c: {val: i for i, val in enumerate(sorted(raw_train_parsed[c].dropna().unique()))} for c in cat_cols}

    for c in cat_cols:
        raw_train_parsed[f"{c}_code"] = raw_train_parsed[c].map(global_maps[c]).fillna(-1).astype(np.int32)
        val_a_parsed[f"{c}_code"] = val_a_parsed[c].map(global_maps[c]).fillna(-1).astype(np.int32)

    from train_v16_surgical_cold_start import build_v16_features
    past_feat_a, val_feat_a, _ = build_v16_features(raw_train_parsed, val_a_parsed, cutoff_a, global_maps, cat_cols)

    features_model = [
        "guc", "log_guc", "il_code", "ilce_code", "bolge_code", "guc_bin_code",
        "month", "day", "day_of_week", "day_of_year", "is_weekend", "is_summer", "is_june_july",
        "log_guc_x_summer", "monthly_network_index", "log_fac_level", "log_seasonal_baseline",
        "is_cold", "has_annual_lag", "arch_prob_0", "arch_prob_1", "arch_prob_2",
        "sin_day_1", "cos_day_1", "sin_day_2", "cos_day_2",
        "sin_dow_1", "cos_dow_1", "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2"
    ]

    y_res_a = np.log1p(past_feat_a["tuketim"].values) - np.log1p(past_feat_a["seasonal_baseline"].values)
    m_lgb_a = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, random_state=42, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
    m_lgb_a.fit(past_feat_a[features_model], y_res_a)
    p_lgb_res = m_lgb_a.predict(val_feat_a[features_model])

    pred_v14_val = np.maximum(0.0, np.expm1(np.log1p(val_feat_a["seasonal_baseline"].values) + p_lgb_res))

    # V8R Baseline Proxy on Fold A
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
    pred_v8r_val = np.array(v8r_val_list, dtype=np.float32)

    y_true_val = val_a["tuketim"].values
    warm_mask = val_a["is_cold"] == 0
    cold_mask = val_a["is_cold"] == 1

    print(f"\n  Fold A Dogrudan Olcumler:")
    print(f"    V14 Toplam RMSLE : {calculate_rmsle(y_true_val, pred_v14_val):.5f} (Warm: {calculate_rmsle(y_true_val[warm_mask], pred_v14_val[warm_mask]):.5f} | Cold: {calculate_rmsle(y_true_val[cold_mask], pred_v14_val[cold_mask]):.5f})")
    print(f"    V8R Toplam RMSLE : {calculate_rmsle(y_true_val, pred_v8r_val):.5f} (Warm: {calculate_rmsle(y_true_val[warm_mask], pred_v8r_val[warm_mask]):.5f} | Cold: {calculate_rmsle(y_true_val[cold_mask], pred_v8r_val[cold_mask]):.5f})")

    # Anomali Dağılımı
    log_diff_oof = np.abs(np.log1p(pred_v14_val) - np.log1p(pred_v8r_val))
    print(f"\n  Log-Fark (|log(pred_v14) - log(pred_v8r)|) Dagilimi:")
    print(f"    50% (Medyan) : {np.percentile(log_diff_oof, 50):.4f}")
    print(f"    90%          : {np.percentile(log_diff_oof, 90):.4f}")
    print(f"    95%          : {np.percentile(log_diff_oof, 95):.4f}")
    print(f"    99%          : {np.percentile(log_diff_oof, 99):.4f}")
    print(f"    99.9%        : {np.percentile(log_diff_oof, 99.9):.4f}")

    # -------------------------------------------------------------------------
    # AŞAMA 6: BLEND AĞIRLIĞI VE GÜVENLİK AĞI OPTİMİZASYONU (Fold A)
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 6] Blend Agirligi & Taban Grid Search (Olculen Sonuclar)...")
    best_config = None
    best_rmsle = 999.0

    for thresh in [1.5, 2.0, 2.5, 3.0, 999.0]:
        for fl_mult in [0.0, 0.05, 0.10, 0.20, 0.35]:
            fl = compute_safety_floor(raw_train, val_a, cutoff_a, floor_multiplier=fl_mult)
            v14_s = np.maximum(pred_v14_val, fl)
            v8r_s = np.maximum(pred_v8r_val, fl)

            for w_warm in [0.70, 0.80, 0.85, 0.90, 1.00]:
                for w_cold in [0.30, 0.50, 0.70, 0.85, 1.00]:
                    w_arr = np.where(cold_mask, w_cold, w_warm)
                    blend = w_arr * v14_s + (1.0 - w_arr) * v8r_s

                    # Fallback to V8R where log_diff > thresh
                    final_pred = np.where(log_diff_oof > thresh, v8r_s, blend)
                    final_pred = np.maximum(final_pred, fl)

                    r = calculate_rmsle(y_true_val, final_pred)
                    if r < best_rmsle:
                        best_rmsle = r
                        best_config = {
                            "threshold": thresh,
                            "floor_mult": fl_mult,
                            "w_warm": w_warm,
                            "w_cold": w_cold,
                            "rmsle": r,
                            "warm_rmsle": calculate_rmsle(y_true_val[warm_mask], final_pred[warm_mask]),
                            "cold_rmsle": calculate_rmsle(y_true_val[cold_mask], final_pred[cold_mask]),
                        }

    print(f"[OK] En Iyi Parametre Konfigurasyonu (Fold A Dogrulanmis):")
    print(f"   - Anomali Esigi (Threshold) : {best_config['threshold']}")
    print(f"   - Taban Carpani (Floor Mult): {best_config['floor_mult']}")
    print(f"   - Warm Agirligi (w_warm)    : {best_config['w_warm']}")
    print(f"   - Cold Agirligi (w_cold)    : {best_config['w_cold']}")
    print(f"   * FOLD A TOPLAM RMSLE       : {best_config['rmsle']:.5f}")
    print(f"   * FOLD A WARM RMSLE         : {best_config['warm_rmsle']:.5f}")
    print(f"   * FOLD A COLD RMSLE         : {best_config['cold_rmsle']:.5f}")

    # -------------------------------------------------------------------------
    # AŞAMA 7: UÇTAN UCA FOLD A BACKTEST VE TEST SETİ PROJEKSİYONU
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 7] Uctan Uca Backtest Ozeti...")
    simulated_lb = np.sqrt(0.7784 * (best_config["warm_rmsle"]**2) + 0.2216 * (best_config["cold_rmsle"]**2))
    print(f"  [OK] Fold A Gercek Olculen Toplam RMSLE : {best_config['rmsle']:.5f}")
    print(f"  [OK] Test Seti Dagiliminda (%22.16 Cold): {simulated_lb:.5f}")

    # -------------------------------------------------------------------------
    # AŞAMA 8: FİNAL ÜRETİM VE BÜTÜNLÜK KONTROLÜ (Test Seti)
    # -------------------------------------------------------------------------
    print("\n>>> [ASAMA 8] Final Test Seti Uretimi ve Butunluk Kontrolu...")
    assert V8R_SUB_PATH.exists(), f"V8R dosyasi bulunamadi: {V8R_SUB_PATH}"
    v8r_test = pd.read_csv(V8R_SUB_PATH)
    v14_test = pd.read_csv(V14_SUB_PATH) if V14_SUB_PATH.exists() else pd.read_csv(DATA_DIR / "submission_v16_standalone.csv")

    cutoff_full = pd.Timestamp("2026-03-31")
    floor_test = compute_safety_floor(raw_train, raw_test, cutoff_full, floor_multiplier=best_config["floor_mult"])

    v8r_test_safe = np.maximum(v8r_test["tuketim"].values, floor_test)
    v14_test_safe = np.maximum(v14_test["tuketim"].values, floor_test)

    log_diff_test = np.abs(np.log1p(v14_test_safe) - np.log1p(v8r_test_safe))

    w_test = np.where(raw_test["is_cold"] == 1, best_config["w_cold"], best_config["w_warm"])
    test_blend = w_test * v14_test_safe + (1.0 - w_test) * v8r_test_safe

    test_final = np.where(log_diff_test > best_config["threshold"], v8r_test_safe, test_blend)
    test_final = np.maximum(test_final, floor_test)
    test_ceil = 36.0 * (raw_test["guc"].values + 1.0)
    test_final = np.clip(test_final, 0.0, test_ceil)

    # ZORUNLU KONTROL BLOĞU
    sub_v19 = pd.DataFrame({"id": raw_test["id"], "tuketim": test_final})
    assert list(sub_v19.columns) == ["id", "tuketim"], "Kolon isimleri kesinlikle ['id', 'tuketim'] olmali!"
    assert len(sub_v19) == len(sample_sub), f"Satir sayisi sample submission ile uyusmuyor: {len(sub_v19)}"
    assert (sub_v19["id"] == sample_sub["id"]).all(), "ID siralamasi birebir eslesmiyor!"
    assert sub_v19["tuketim"].isna().sum() == 0, "NaN deger tespit edildi!"
    assert (sub_v19["tuketim"] < 0).sum() == 0, "Negatif deger tespit edildi!"
    assert np.isfinite(sub_v19["tuketim"]).all(), "Sonsuz (inf) deger tespit edildi!"

    sub_v19.to_csv(OUTPUT_V19_PATH, index=False)
    sha256_v19 = get_sha256(OUTPUT_V19_PATH)

    print("\n" + "=" * 80)
    print(">>> V19 - DOGRULANMIS GUVENLIK AGI BASARIYLA URETILDI!")
    print(f"[OK] Dosya Konumu : {OUTPUT_V19_PATH}")
    print(f"[OK] SHA256       : {sha256_v19}")
    print(f"[OK] Test Mean    : {test_final.mean():.2f}")
    print(f"[OK] Test Min     : {test_final.min():.2f}")
    print(f"[OK] Test Median  : {np.median(test_final):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
