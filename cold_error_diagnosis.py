from __future__ import annotations

import numpy as np
import pandas as pd
from verify_v13_5_v14_fixed import calculate_rmsle, parse_locations

train = pd.read_csv(r"C:\Users\EREN\Desktop\grid-up-datathon\train.csv", parse_dates=["tarih"])
train = parse_locations(train)

cutoff = pd.Timestamp("2025-03-31")
past_facs = set(train[train["tarih"] <= cutoff]["tanim"].unique())
val_raw = train[(train["tarih"] >= "2025-04-01") & (train["tarih"] <= "2025-07-31")].copy()

val_raw["is_cold"] = (~val_raw["tanim"].isin(past_facs)).astype(int)
cold = val_raw[val_raw["is_cold"] == 1].copy()

cold["load_factor"] = cold["tuketim"] / np.maximum(1.0, cold["guc"])
cold["log_t"] = np.log1p(cold["tuketim"])
cold["log_guc"] = np.log1p(cold["guc"])

# Test actual model prediction vs simple baseline
cold["pred"] = np.clip(cold["guc"] * 2.5, 0.1, 36.0 * (cold["guc"] + 1))
cold["log_pred"] = np.log1p(cold["pred"])
cold["log_err"] = cold["log_pred"] - cold["log_t"]
cold["sq_log_err"] = cold["log_err"] ** 2

tot_mse = cold["sq_log_err"].sum()
tot_rmsle = np.sqrt(cold["sq_log_err"].mean())

print("=" * 70)
print(f"COLD SEGMENT ERROR DECOMPOSITION (N = {len(cold):,} rows, {cold['tanim'].nunique():,} facilities)")
print(f"Total Cold RMSLE: {tot_rmsle:.4f}")
print("=" * 70)

over_mask = cold["log_err"] > 1.0
under_mask = cold["log_err"] < -1.0
accurate_mask = cold["log_err"].abs() <= 1.0

over_sq = cold.loc[over_mask, "sq_log_err"].sum()
under_sq = cold.loc[under_mask, "sq_log_err"].sum()
acc_sq = cold.loc[accurate_mask, "sq_log_err"].sum()

print("\n1. ERROR DIRECTION AND ASYMMETRY:")
print(f"  A) Over-prediction (Model >> Actual by >2.7x) : {over_mask.sum():,} rows ({over_mask.mean()*100:.1f}%) | Generates {over_sq/tot_mse*100:.1f}% of total squared error!")
print(f"  B) Under-prediction (Model << Actual by >2.7x): {under_mask.sum():,} rows ({under_mask.mean()*100:.1f}%) | Generates {under_sq/tot_mse*100:.1f}% of total squared error!")
print(f"  C) Accurate (Within 2.7x factor)              : {accurate_mask.sum():,} rows ({accurate_mask.mean()*100:.1f}%) | Generates {acc_sq/tot_mse*100:.1f}% of total squared error!")

zero_mask = cold["tuketim"] == 0
micro_mask = (cold["tuketim"] > 0) & (cold["tuketim"] < 10)
zero_sq = cold.loc[zero_mask, "sq_log_err"].sum()
micro_sq = cold.loc[micro_mask, "sq_log_err"].sum()

print("\n2. IDLE / NEAR-ZERO METERS (THE SILENT KILLER):")
print(f"  - Exact Zero Rows (0 kWh)    : {zero_mask.sum():,} ({zero_mask.mean()*100:.1f}%) | Generates {zero_sq/tot_mse*100:.1f}% of all error!")
print(f"  - Micro Rows (< 10 kWh)      : {micro_mask.sum():,} ({micro_mask.mean()*100:.1f}%) | Generates {micro_sq/tot_mse*100:.1f}% of all error!")
print(f"  -> Combined Idle (<10 kWh)   : {(zero_mask | micro_mask).sum():,} ({(zero_mask | micro_mask).mean()*100:.1f}%) | Generates {(zero_sq + micro_sq)/tot_mse*100:.1f}% of all error!")

fac_lf = cold.groupby("tanim")["load_factor"].median()
print("\n3. HUGE OPERATIONAL DIVERSITY ACROSS FACILITIES (Load Factor = Tuketim / Guc):")
print(f"  - Bottom 10% Facilities Median Load Factor : {fac_lf.quantile(0.10):.4f} kWh/kW (Virtually dormant)")
print(f"  - 25th Percentile Facility Load Factor     : {fac_lf.quantile(0.25):.4f} kWh/kW (Low duty cycle)")
print(f"  - 50th Percentile (Median) Load Factor     : {fac_lf.quantile(0.50):.4f} kWh/kW (Standard load)")
print(f"  - 75th Percentile Facility Load Factor     : {fac_lf.quantile(0.75):.4f} kWh/kW (Heavy continuous load)")
print(f"  - Top 10% Facilities Median Load Factor    : {fac_lf.quantile(0.90):.4f} kWh/kW (Continuous high surge)")
print(f"  - Ratio of Top 10% to Bottom 10% Load      : {fac_lf.quantile(0.90) / max(0.001, fac_lf.quantile(0.10)):.1f}x difference!")
print("=" * 70)
