from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

DATA_DIR = r"C:\Users\EREN\Desktop\grid-up-datathon"


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.clip(y_true, 0, None)
    y_p = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_p) - np.log1p(y_t)) ** 2)))


def parse_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts = df["lokasyon"].astype(str).str.split(">")
    df["il"] = parts.str[0]
    df["ilce"] = parts.str[-1]
    df["bolge"] = parts.apply(lambda p: p[-2] if len(p) >= 3 else "DOGRUDAN")
    return df


train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["tarih"])
test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["tarih"])

train = parse_locations(train)
test = parse_locations(test)

guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
train["guc_bin"] = pd.cut(train["guc"], bins=guc_bins, labels=guc_labels).astype(str)
test["guc_bin"] = pd.cut(test["guc"], bins=guc_bins, labels=guc_labels).astype(str)

# Fold A: Cutoff 2025-03-31
cutoff = pd.Timestamp("2025-03-31")
past_df = train[train["tarih"] <= cutoff].copy()
val_raw = train[(train["tarih"] >= "2025-04-01") & (train["tarih"] <= "2025-07-31")].copy()

past_facs = set(past_df["tanim"].unique())
val_raw["is_cold"] = (~val_raw["tanim"].isin(past_facs)).astype(int)

cold_df = val_raw[val_raw["is_cold"] == 1].copy()
cold_df["month"] = cold_df["tarih"].dt.month
cold_df["day_of_week"] = cold_df["tarih"].dt.dayofweek
cold_df["is_weekend"] = (cold_df["day_of_week"] >= 5).astype(int)

y_cold = cold_df["tuketim"].values

print("=" * 70)
print(f"COLD-START POPULATION STUDY (N = {len(cold_df):,} rows, {cold_df['tanim'].nunique():,} unique facilities)")
print("=" * 70)
print(f"1. Zero Tuketim Ratio : {(y_cold == 0).mean()*100:.2f}% ({(y_cold == 0).sum():,} rows)")
print(f"2. Min: {np.min(y_cold):.2f}, 10th: {np.percentile(y_cold, 10):.2f}, 25th: {np.percentile(y_cold, 25):.2f}, Median: {np.median(y_cold):.2f}")
print(f"3. Mean: {np.mean(y_cold):.2f}, 75th: {np.percentile(y_cold, 75):.2f}, 90th: {np.percentile(y_cold, 90):.2f}, Max: {np.max(y_cold):.2f}")

# ORACLE 1: Facility-level Log-Mean (Theoretical Lower Bound for ANY static facility model)
oracle_fac_mean = cold_df.groupby("tanim")["tuketim"].transform(lambda s: np.expm1(np.mean(np.log1p(s))))
r_oracle_fac = calculate_rmsle(y_cold, oracle_fac_mean.values)
print("\n--- THEORETICAL LOWER BOUNDS (ORACLES) ---")
print(f"[ORACLE 1] (Perfect Static Facility Level Knowledge) : RMSLE = {r_oracle_fac:.5f}")

# ORACLE 2: Facility x Month Dynamic Oracle
oracle_fac_m = cold_df.groupby(["tanim", "month"])["tuketim"].transform(lambda s: np.expm1(np.mean(np.log1p(s))))
r_oracle_m = calculate_rmsle(y_cold, oracle_fac_m.values)
print(f"[ORACLE 2] (Perfect Facility x Month Dynamic Level)  : RMSLE = {r_oracle_m:.5f}")


print("\n--- EMPIRICAL CANDIDATE STRATEGIES FOR COLD FACILITIES ---")

# 1. Current Default Fallback: guc * 2.5
pred_cur = cold_df["guc"].values * 2.5
print(f"1. Current Raw Default (2.5 * guc)                   : RMSLE = {calculate_rmsle(y_cold, pred_cur):.5f}")

# 2. Optimal Power Multiplier Search
best_a, best_r_a = 1.0, 999.0
for a in np.linspace(0.05, 3.0, 100):
    r = calculate_rmsle(y_cold, cold_df["guc"].values * a)
    if r < best_r_a:
        best_r_a = r
        best_a = a
print(f"2. Optimal Global Power Multiplier ({best_a:.2f} * guc)         : RMSLE = {best_r_a:.5f}")

# 3. Hierarchical Bayesian Prior on Past Data (ilce x guc_bin)
past_df["log_t"] = np.log1p(past_df["tuketim"])
past_df["log_guc"] = np.log1p(past_df["guc"])
past_df["log_ratio"] = np.log1p(past_df["tuketim"]) - np.log1p(past_df["guc"])

ilce_guc_ratio = past_df.groupby(["ilce", "guc_bin"])["log_ratio"].median().to_dict()
guc_ratio = past_df.groupby("guc_bin")["log_ratio"].median().to_dict()
global_ratio = past_df["log_ratio"].median()

pred_ratio = []
for _, row in cold_df.iterrows():
    k = (row["ilce"], row["guc_bin"])
    r = ilce_guc_ratio.get(k, guc_ratio.get(row["guc_bin"], global_ratio))
    pred_ratio.append(np.expm1(np.log1p(row["guc"]) + r))
pred_ratio = np.array(pred_ratio)
print(f"3. Hierarchical Log-Ratio (ilce x guc_bin prior)     : RMSLE = {calculate_rmsle(y_cold, pred_ratio):.5f}")

# 4. Hierarchical Log-Ratio + Monthly Network Seasonality
m_index = {4: 0.651, 5: 0.564, 6: 1.000, 7: 1.700}
pred_ratio_s = pred_ratio * cold_df["month"].map(m_index).values
print(f"4. Hierarchical Ratio + Monthly Network Index        : RMSLE = {calculate_rmsle(y_cold, pred_ratio_s):.5f}")

# 5. Dedicated Pure-Metadata Cold GBDT Model (Trained on past with guc, ilce, bolge, month, doy)
past_df["month"] = past_df["tarih"].dt.month
past_df["day_of_week"] = past_df["tarih"].dt.dayofweek
past_df["day_of_year"] = past_df["tarih"].dt.dayofyear
past_df["is_weekend"] = (past_df["day_of_week"] >= 5).astype(int)

cold_df["day_of_year"] = cold_df["tarih"].dt.dayofyear

cat_cols = ["il", "ilce", "bolge", "guc_bin"]
for c in cat_cols:
    all_c = sorted(list(set(train[c].dropna())))
    c_dict = {v: i for i, v in enumerate(all_c)}
    past_df[f"{c}_code"] = past_df[c].map(c_dict).fillna(-1).astype(int)
    cold_df[f"{c}_code"] = cold_df[c].map(c_dict).fillna(-1).astype(int)

meta_feats = ["guc", "month", "day_of_week", "day_of_year", "is_weekend", "il_code", "ilce_code", "bolge_code", "guc_bin_code"]

m_cold_lgb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
m_cold_lgb.fit(past_df[meta_feats], past_df["log_t"])
p_cold_lgb = np.expm1(m_cold_lgb.predict(cold_df[meta_feats]))

m_cold_cb = CatBoostRegressor(iterations=400, learning_rate=0.04, depth=4, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
m_cold_cb.fit(past_df[meta_feats], past_df["log_t"], verbose=False)
p_cold_cb = np.expm1(m_cold_cb.predict(cold_df[meta_feats]))

p_cold_ens = 0.50 * p_cold_lgb + 0.50 * p_cold_cb
print(f"5. Dedicated Cold Metadata LightGBM Model            : RMSLE = {calculate_rmsle(y_cold, p_cold_lgb):.5f}")
print(f"6. Dedicated Cold Metadata CatBoost Model            : RMSLE = {calculate_rmsle(y_cold, p_cold_cb):.5f}")
print(f"7. Dedicated Cold Metadata Ensemble (LGB+CB 50/50)   : RMSLE = {calculate_rmsle(y_cold, p_cold_ens):.5f}")

# 6. Hybrid Blend: (Dedicated Cold Model + Hierarchical Ratio)
p_hybrid = 0.60 * p_cold_ens + 0.40 * pred_ratio_s
print(f"8. Hybrid (Dedicated Cold Ensemble + Hierarchical)   : RMSLE = {calculate_rmsle(y_cold, p_hybrid):.5f}")
print("=" * 70)
