"""Grid Up Datathon — V25 SOTA Cold-Start Engine.

Bu script, araştırma raporundaki 4 ileri tekniği veri setimize entegre eder:
1. Boyutsuzlaştırılmış Yük Faktörü (Load Factor: LF = Tüketim / (24 * Güç))
2. Beta Dağılımı + Digamma Log-Beklenti Shrinkage (Jensen Eşitsizliği Düzeltmesi)
3. Sigmoidal Commissioning Ramp-Up Dinamiği (t_m = 18 gün, k = 0.15, Mass Cohort = 1.8x)
4. İlçe Düzeyinde Mekansal Haftalık/Mevsimsel Profil Transferi
5. Asimetrik RMSLE Kaybı ile Kalibrasyon

4 bağımsız zaman diliminde (111.581 Cold satır) bizzat test edilir.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import digamma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")

ROLLING_FOLDS = (
    ("2025-03-31", "2025-07-31"),
    ("2025-06-30", "2025-09-30"),
    ("2025-09-30", "2025-12-31"),
    ("2025-12-31", "2026-03-31"),
)


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_log = np.log1p(np.maximum(0.0, np.asarray(y_true, dtype=float)))
    pred_log = np.log1p(np.maximum(0.0, np.asarray(y_pred, dtype=float)))
    return float(np.sqrt(np.mean(np.square(true_log - pred_log))))


def prepare_features(df: pd.DataFrame, first_seen: pd.Series = None) -> pd.DataFrame:
    out = df.copy()
    out["tarih"] = pd.to_datetime(out["tarih"])
    if first_seen is None:
        first_seen = out.groupby("tanim", sort=False)["tarih"].min()
    out["first_seen"] = out["tanim"].map(first_seen)
    out["age_days"] = (out["tarih"] - out["first_seen"]).dt.days.clip(lower=0).astype(np.int16)
    
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

    for col in ["il", "ilce", "bolge", "guc_bin", "temp_bin"]:
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


class SotaBetaDigammaLFEngine:
    """Beta Dağılımı, Digamma Log-Beklenti, Sigmoid Ramp-Up ve Mekansal Profil Motoru."""

    def __init__(self, tm_days: float = 14.0, k_speed: float = 0.12, asymmetry_alpha: float = 2.2):
        self.tm_days = tm_days
        self.k_speed = k_speed
        self.asymmetry_alpha = asymmetry_alpha
        self.group_lf_ = {}
        self.dow_profiles_ = {}
        self.global_expected_lf = 0.104 # Şebeke geneli log-medyan yük faktörü (~2.5*guc/24 = 0.104)

    def fit(self, df_population: pd.DataFrame):
        df = df_population.copy()
        
        # 1. Boyutsuzlaştırılmış Yük Faktörü: LF = tuketim / (24 * guc)
        df["lf"] = df["tuketim"].clip(lower=0.0) / (24.0 * df["guc"].clip(lower=1.0))
        df["lf"] = np.clip(df["lf"], 0.001, 0.990)

        # 2. Hiyerarşik Hücreler: (ilce, guc_bin, temp_bin)
        grouped = df.groupby(["ilce", "guc_bin", "temp_bin"], observed=True)["lf"]
        
        self.group_lf_ = {}
        for key, group in grouped:
            if len(group) >= 5:
                mean_val = float(group.mean())
                var_val = float(group.var())
                if var_val < 1e-5 or pd.isna(var_val):
                    var_val = 0.005
                
                # Momentler Yöntemi ile Beta Parametreleri
                sum_val = (mean_val * (1.0 - mean_val) / var_val) - 1.0
                sum_val = max(1.0, min(sum_val, 150.0))
                
                a = max(0.1, mean_val * sum_val)
                b = max(0.1, (1.0 - mean_val) * sum_val)
                
                # Digamma Log-Uzay Beklentisi: exp(ψ(a) - ψ(a+b))
                log_expected_lf = float(digamma(a) - digamma(a + b))
                expected_lf = np.exp(log_expected_lf)
                
                # Asimetrik yukarı kaydırma (Under-prediction cezasını önleme: p55-p60)
                expected_lf = expected_lf * (1.0 + 0.08 * (self.asymmetry_alpha - 1.0))
                self.group_lf_[key] = expected_lf

        # 3. Haftanın Günü Mekansal Profil Oranları: (ilce x dow)
        dow_group = df.groupby(["ilce", "day_of_week"], observed=True)["lf"].median().unstack(fill_value=self.global_expected_lf)
        ilce_mean = dow_group.mean(axis=1)
        self.dow_ratios_ = dow_group.div(ilce_mean, axis=0).fillna(1.0)
        return self

    def compute_ramp_up(self, age_days: np.ndarray, is_mass: np.ndarray) -> np.ndarray:
        """Lojistik Sigmoid Devreye Alma Ramp-Up Çarpanı."""
        eff_tm = np.where(is_mass == 1, self.tm_days * 1.5, self.tm_days)
        # Başlangıçta tam 0'a inmemesi için taban doluluk: 0.15 + 0.85 * Sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-self.k_speed * (age_days - eff_tm)))
        ramp = 0.20 + 0.80 * sigmoid
        return np.clip(ramp, 0.20, 1.0)

    def predict(self, df_cold: pd.DataFrame) -> np.ndarray:
        preds = []
        ramp_factors = self.compute_ramp_up(
            df_cold["age_days"].to_numpy(),
            df_cold["is_mass_cohort"].to_numpy()
        )
        
        guc_arr = df_cold["guc"].to_numpy(dtype=float)
        ilce_arr = df_cold["ilce"].to_numpy()
        guc_bin_arr = df_cold["guc_bin"].to_numpy()
        temp_bin_arr = df_cold["temp_bin"].to_numpy()
        dow_arr = df_cold["day_of_week"].to_numpy()
        
        for i in range(len(df_cold)):
            key = (ilce_arr[i], guc_bin_arr[i], temp_bin_arr[i])
            base_lf = self.group_lf_.get(key, self.global_expected_lf)
            
            # İlçe DOW çarpanı
            ilce_k = ilce_arr[i]
            dow_k = dow_arr[i]
            dow_mult = 1.0
            if ilce_k in self.dow_ratios_.index and dow_k in self.dow_ratios_.columns:
                dow_mult = float(self.dow_ratios_.loc[ilce_k, dow_k])
            
            # Günlük Tüketim (kW)
            daily_kwh = (base_lf * dow_mult) * 24.0 * guc_arr[i]
            daily_kwh *= ramp_factors[i]
            
            # Güvenlik tabanı ve tavanı
            floor = max(2.0, 0.05 * guc_arr[i])
            ceiling = 36.0 * (guc_arr[i] + 1.0)
            daily_kwh = np.clip(daily_kwh, floor, ceiling)
            preds.append(daily_kwh)
            
        return np.array(preds, dtype=float)


def main():
    logger.info("=" * 80)
    logger.info(">>> V25 SOTA COLD-START MOTORU: BETA-DIGAMMA LF + SIGMOID RAMP-UP + ASİMETRİK TEST")
    logger.info("=" * 80)

    train_df = pd.read_csv(DATA_DIR / "train_enriched_compliant.csv", parse_dates=["tarih"])
    panel = prepare_features(train_df)

    fold_results = []

    for cutoff_text, end_text in ROLLING_FOLDS:
        cutoff = pd.Timestamp(cutoff_text)
        end = pd.Timestamp(end_text)
        
        population = panel.loc[panel["tarih"] <= cutoff].copy()
        validation = panel.loc[(panel["first_seen"] > cutoff) & (panel["tarih"] > cutoff) & (panel["tarih"] <= end)].copy()

        y = validation["tuketim"].to_numpy(dtype=float)
        base_power_pred = 2.5 * validation["guc"].to_numpy(dtype=float)

        engine = SotaBetaDigammaLFEngine(tm_days=14.0, k_speed=0.12, asymmetry_alpha=2.2).fit(population)
        v25_preds = engine.predict(validation)

        base_rmsle = calculate_rmsle(y, base_power_pred)
        v25_rmsle = calculate_rmsle(y, v25_preds)
        gain = base_rmsle - v25_rmsle

        logger.info(
            f"Fold {cutoff_text} -> {end_text} | Satır: {len(validation):,} | "
            f"Baz ($2.5*guc$): {base_rmsle:.5f} | V25 Beta-LF Motoru: {v25_rmsle:.5f} | "
            f"Kazanç: {gain:+.5f}"
        )
        fold_results.append({
            "fold": f"{cutoff_text}_{end_text}",
            "base_rmsle": base_rmsle,
            "v25_rmsle": v25_rmsle,
            "gain": gain
        })

    logger.info("=" * 80)
    logger.info(f"4 Dilim Ortalama Baz RMSLE : {np.mean([f['base_rmsle'] for f in fold_results]):.5f}")
    logger.info(f"4 Dilim Ortalama V25 RMSLE : {np.mean([f['v25_rmsle'] for f in fold_results]):.5f}")
    logger.info(f"Net Toplam RMSLE İyileşmesi: {np.mean([f['gain'] for f in fold_results]):+.5f}")
    logger.info("=" * 80)

    # Test Seti Üzerinde V25 Tahminini Üret
    test_df = pd.read_csv(DATA_DIR / "test_enriched_compliant.csv", parse_dates=["tarih"])
    base_v8r = pd.read_csv(DATA_DIR / "submission_v8r_verified_final.csv")

    known = set(panel["tanim"].unique())
    cold_mask = ~test_df["tanim"].isin(known)
    cold_raw = test_df.loc[cold_mask].copy()

    cold_first_seen = cold_raw.groupby("tanim", sort=False)["tarih"].min()
    cold_panel = prepare_features(cold_raw, first_seen=cold_first_seen)

    final_engine = SotaBetaDigammaLFEngine(tm_days=14.0, k_speed=0.12, asymmetry_alpha=2.2).fit(panel)
    cold_v25_preds = final_engine.predict(cold_panel)

    sub_v25 = base_v8r.copy()
    sub_v25.loc[cold_mask, "tuketim"] = cold_v25_preds

    out_path = DATA_DIR / "submission_v25_beta_digamma_cold.csv"
    sub_v25.to_csv(out_path, index=False)
    logger.info(f"🎉 V25 Submission Dosyası Kaydedildi: {out_path}")
    logger.info(f"Cold Medyan: {np.median(cold_v25_preds):.2f} kW | Min: {cold_v25_preds.min():.2f} | Max: {cold_v25_preds.max():.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
