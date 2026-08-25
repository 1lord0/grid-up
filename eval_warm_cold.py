from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans

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


print("Loading train.csv and test.csv...")
raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["tarih"])
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["tarih"])

raw_train = parse_locations(raw_train)
raw_test = parse_locations(raw_test)

guc_bins = [-np.inf, 100, 400, 1000, 2500, np.inf]
guc_labels = ["Micro", "Small", "Medium", "Large", "VeryLarge"]
raw_train["guc_bin"] = pd.cut(raw_train["guc"], bins=guc_bins, labels=guc_labels).astype(str)
raw_test["guc_bin"] = pd.cut(raw_test["guc"], bins=guc_bins, labels=guc_labels).astype(str)

cat_cols_raw = ["il", "ilce", "bolge", "guc_bin"]
global_cat_maps = {}
for col in cat_cols_raw:
    all_vals = sorted(list(set(raw_train[col].dropna()).union(set(raw_test[col].dropna()))))
    global_cat_maps[col] = {val: i for i, val in enumerate(all_vals)}


def build_features(train_df, target_df, cutoff_date, include_archetypes=True):
    past_df = train_df[train_df["tarih"] <= cutoff_date].copy()

    m_totals = past_df.groupby(past_df["tarih"].dt.month)["tuketim"].sum()
    m_avg = m_totals.mean() if len(m_totals) > 0 else 1.0
    m_index = (m_totals / m_avg).to_dict()
    default_m_index = {1: 0.794, 2: 0.782, 3: 0.694, 4: 0.651, 5: 0.564, 6: 1.000, 7: 1.700, 8: 1.442, 9: 1.174, 10: 0.734, 11: 1.068, 12: 1.400}
    for m in range(1, 13):
        if m not in m_index:
            m_index[m] = default_m_index[m]

    fac_recent_28 = past_df[past_df["tarih"] > (cutoff_date - pd.Timedelta(days=28))].groupby("tanim")["tuketim"].mean().to_dict()
    fac_mean_all = past_df.groupby("tanim")["tuketim"].mean().to_dict()
    fac_last_seen = past_df.groupby("tanim")["tarih"].max().to_dict()

    lag_364_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=364)])["tuketim"].to_dict()
    lag_365_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=365)])["tuketim"].to_dict()
    lag_371_map = past_df.set_index(["tanim", past_df["tarih"] + pd.Timedelta(days=371)])["tuketim"].to_dict()

    fac_summer_surge = {}
    prob_dict_0, prob_dict_1, prob_dict_2 = {}, {}, {}
    global_surge = 1.40
    ilce_surge, guc_surge = {}, {}

    if include_archetypes:
        past_july = past_df[past_df["tarih"].dt.month == 7].groupby("tanim")["tuketim"].mean()
        past_aug = past_df[past_df["tarih"].dt.month == 8].groupby("tanim")["tuketim"].mean()
        past_winter = past_df[past_df["tarih"].dt.month.isin([1, 2, 3])].groupby("tanim")["tuketim"].mean()
        guc_map = past_df.groupby("tanim")["guc"].first().to_dict()

        if len(past_july) > 0 and len(past_winter) > 0:
            july_mean = past_july.mean()
            aug_mean = past_aug.mean() if len(past_aug) > 0 else july_mean
            aug_to_july_ratio = (july_mean / max(1.0, aug_mean)) if aug_mean > 0 else 1.18

            for t in set(past_winter.index):
                w_val = past_winter.get(t, np.nan)
                j_val = past_july.get(t, np.nan)
                a_val = past_aug.get(t, np.nan)
                g_val = guc_map.get(t, 630.0)
                prior_c = max(5.0, g_val * 0.10)

                if not np.isnan(j_val) and not np.isnan(a_val):
                    denoised_summer = 0.55 * j_val + 0.45 * (a_val * aug_to_july_ratio)
                elif not np.isnan(j_val):
                    denoised_summer = j_val
                elif not np.isnan(a_val):
                    denoised_summer = a_val * aug_to_july_ratio
                else:
                    denoised_summer = np.nan

                if not np.isnan(w_val) and not np.isnan(denoised_summer):
                    surge = (denoised_summer + prior_c) / (w_val + prior_c)
                    fac_summer_surge[t] = float(np.clip(surge, 0.1, 10.0))

        surge_df = past_df[["tanim", "ilce", "guc_bin", "il"]].drop_duplicates("tanim").set_index("tanim")
        surge_df["surge"] = surge_df.index.map(fac_summer_surge)
        global_surge = float(surge_df["surge"].dropna().median()) if len(surge_df["surge"].dropna()) > 0 else 1.40
        ilce_surge = surge_df.groupby("ilce")["surge"].median().to_dict()
        guc_surge = surge_df.groupby("guc_bin")["surge"].median().to_dict()

        profile_pivot = past_df.pivot_table(index="tanim", columns=past_df["tarih"].dt.month, values="tuketim", aggfunc="mean")
        profile_norm = profile_pivot.div(profile_pivot.mean(axis=1), axis=0).fillna(1.0)
        for m in range(1, 13):
            if m not in profile_norm.columns:
                profile_norm[m] = default_m_index[m]
        profile_norm = profile_norm[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(profile_norm.values)
        centers = kmeans.cluster_centers_

        all_known_facs = list(set(train_df["tanim"].unique()).union(set(target_df["tanim"].unique())))
        synth_vectors = []
        for t in all_known_facs:
            if t in profile_norm.index:
                v = profile_norm.loc[t].values
            else:
                s_val = fac_summer_surge.get(t, global_surge)
                v = np.array([1.0, 1.0, 1.0, 0.85, 0.70, 1.0 + 0.35 * (s_val - 1.0), 1.1 + 0.85 * (s_val - 1.0), 1.05 + 0.70 * (s_val - 1.0), 0.95, 0.85, 1.0, 1.1])
                v = v / max(0.1, v.mean())
            synth_vectors.append(v)

        synth_mat = np.array(synth_vectors, dtype=np.float32)
        diffs = synth_mat[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        exp_neg = np.exp(-dists)
        probs = exp_neg / exp_neg.sum(axis=1, keepdims=True)

        prob_dict_0 = dict(zip(all_known_facs, probs[:, 0]))
        prob_dict_1 = dict(zip(all_known_facs, probs[:, 1]))
        prob_dict_2 = dict(zip(all_known_facs, probs[:, 2]))

    def transform_df(df_target):
        df = df_target.copy()
        df["month"] = df["tarih"].dt.month
        df["day_of_week"] = df["tarih"].dt.dayofweek
        df["day_of_year"] = df["tarih"].dt.dayofyear
        df["day"] = df["tarih"].dt.day
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25).astype(np.float32)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        df["is_june_july"] = df["month"].isin([6, 7]).astype(int)
        df["log_guc"] = np.log1p(np.maximum(1.0, df["guc"])).astype(np.float32)
        df["log_guc_x_summer"] = (df["log_guc"] * df["is_summer"]).astype(np.float32)

        df["monthly_network_index"] = df["month"].map(m_index).fillna(1.0).astype(np.float32)

        for c in cat_cols_raw:
            c_map = global_cat_maps[c]
            df[f"{c}_code"] = df[c].map(c_map).fillna(-1).astype(np.int32)

        df["fac_recent_28"] = df["tanim"].map(fac_recent_28)
        df["fac_mean_all"] = df["tanim"].map(fac_mean_all)
        df["fac_level"] = df["fac_recent_28"].fillna(df["fac_mean_all"]).fillna(df["guc"] * 2.5).astype(np.float32)
        df["log_fac_level"] = np.log1p(df["fac_level"]).astype(np.float32)

        m_arr = df["month"].values
        base_arr = df["fac_level"].values

        if include_archetypes:
            direct_surge = df["tanim"].map(fac_summer_surge)
            fallback_ilce = df["ilce"].map(ilce_surge)
            fallback_guc = df["guc_bin"].map(guc_surge)
            df["facility_summer_surge"] = direct_surge.fillna(fallback_ilce).fillna(fallback_guc).fillna(global_surge).astype(np.float32)

            df["arch_prob_0"] = df["tanim"].map(prob_dict_0).fillna(0.33).astype(np.float32)
            df["arch_prob_1"] = df["tanim"].map(prob_dict_1).fillna(0.33).astype(np.float32)
            df["arch_prob_2"] = df["tanim"].map(prob_dict_2).fillna(0.33).astype(np.float32)

            surge_arr = df["facility_summer_surge"].values
            conds = [m_arr == 4, m_arr == 5, m_arr == 6, m_arr == 7, m_arr == 8, m_arr == 9]
            choices = [
                base_arr * 0.85,
                base_arr * 0.70,
                base_arr * (0.90 + 0.35 * (surge_arr - 1.0)),
                base_arr * (1.10 + 0.85 * (surge_arr - 1.0)),
                base_arr * (1.05 + 0.70 * (surge_arr - 1.0)),
                base_arr * 0.95,
            ]
        else:
            conds = [m_arr == 4, m_arr == 5, m_arr == 6, m_arr == 7, m_arr == 8, m_arr == 9]
            choices = [
                base_arr * 0.85,
                base_arr * 0.70,
                base_arr * 1.04,
                base_arr * 1.44,
                base_arr * 1.33,
                base_arr * 0.95,
            ]

        df["seasonal_baseline"] = np.select(conds, choices, default=base_arr).astype(np.float32)
        df["log_seasonal_baseline"] = np.log1p(df["seasonal_baseline"]).astype(np.float32)

        keys = list(zip(df["tanim"].values, df["tarih"].values))
        df["lag_364"] = [lag_364_map.get(k, np.nan) for k in keys]
        df["lag_365"] = [lag_365_map.get(k, np.nan) for k in keys]
        df["lag_371"] = [lag_371_map.get(k, np.nan) for k in keys]
        df["has_annual_lag"] = (~df["lag_365"].isna()).astype(int)
        df["annual_lag_val"] = df["lag_365"].fillna(df["lag_364"]).fillna(df["lag_371"]).fillna(df["seasonal_baseline"]).astype(np.float32)
        df["log_annual_lag"] = np.log1p(df["annual_lag_val"]).astype(np.float32)

        df["last_seen_date"] = df["tanim"].map(fac_last_seen)
        df["days_since_last_seen"] = (cutoff_date - df["last_seen_date"]).dt.days.fillna(999).astype(np.float32)
        df["is_cold"] = (df["days_since_last_seen"] > 180).astype(int)

        if "id" not in df.columns:
            df["id"] = df["tanim"] + "_" + df["tarih"].dt.strftime("%Y-%m-%d")
        return df

    return transform_df(past_df), transform_df(target_df)


base_features = [
    "month", "day_of_week", "day_of_year", "day", "doy_sin", "doy_cos",
    "is_weekend", "is_summer", "is_june_july",
    "guc", "log_guc", "log_guc_x_summer",
    "monthly_network_index",
    "fac_level", "log_fac_level", "seasonal_baseline", "log_seasonal_baseline",
    "has_annual_lag", "annual_lag_val", "log_annual_lag", "days_since_last_seen", "is_cold",
    "il_code", "ilce_code", "bolge_code", "guc_bin_code"
]
arch_features = base_features + ["facility_summer_surge", "arch_prob_0", "arch_prob_1", "arch_prob_2"]

cutoff = pd.Timestamp("2025-03-31")
val_raw = raw_train[(raw_train["tarih"] >= "2025-04-01") & (raw_train["tarih"] <= "2025-07-31")].copy()

past_v13_5, val_v13_5 = build_features(raw_train, val_raw, cutoff, include_archetypes=False)
past_v14, val_v14 = build_features(raw_train, val_raw, cutoff, include_archetypes=True)

y_true = val_raw["tuketim"].values
guc_val = val_raw["guc"].values
ceil_val = 36.0 * (guc_val + 1.0)

# 1. V13 & V13.5
y_res_v13 = np.log1p(past_v13_5["tuketim"].values) - np.log1p(past_v13_5["seasonal_baseline"].values)
m_lgb_v13_5 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
m_lgb_v13_5.fit(past_v13_5[base_features], y_res_v13)
p_lgb_v13_5 = m_lgb_v13_5.predict(val_v13_5[base_features])

m_cb_v13_5 = CatBoostRegressor(iterations=350, learning_rate=0.06, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
m_cb_v13_5.fit(past_v13_5[base_features], y_res_v13, verbose=False)
p_cb_v13_5 = m_cb_v13_5.predict(val_v13_5[base_features])

p_v13 = np.clip(np.maximum(0.0, np.expm1(np.log1p(val_v13_5["seasonal_baseline"].values) + p_lgb_v13_5)), 0.0, ceil_val)
p_v13_5 = np.clip(np.maximum(0.0, np.expm1(np.log1p(val_v13_5["seasonal_baseline"].values) + 0.5 * p_lgb_v13_5 + 0.5 * p_cb_v13_5)), 0.0, ceil_val)

# 2. V14
y_res_v14 = np.log1p(past_v14["tuketim"].values) - np.log1p(past_v14["seasonal_baseline"].values)
m_lgb_v14 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
m_lgb_v14.fit(past_v14[arch_features], y_res_v14)
p_lgb_v14 = m_lgb_v14.predict(val_v14[arch_features])

m_cb_v14 = CatBoostRegressor(iterations=350, learning_rate=0.06, depth=6, loss_function="RMSE", thread_count=-1, random_seed=42, verbose=False)
m_cb_v14.fit(past_v14[arch_features], y_res_v14, verbose=False)
p_cb_v14 = m_cb_v14.predict(val_v14[arch_features])

p_v14 = np.clip(np.maximum(0.0, np.expm1(np.log1p(val_v14["seasonal_baseline"].values) + 0.5 * p_lgb_v14 + 0.5 * p_cb_v14)), 0.0, ceil_val)

val_raw["pred_v13"] = p_v13
val_raw["pred_v13_5"] = p_v13_5
val_raw["pred_v14"] = p_v14
val_raw["pred_blend_70_30"] = 0.70 * p_v14 + 0.30 * p_v13_5

past_facs = set(raw_train[raw_train["tarih"] <= cutoff]["tanim"].unique())
val_raw["is_cold_facility"] = (~val_raw["tanim"].isin(past_facs)).astype(int)

warm_mask = val_raw["is_cold_facility"] == 0
cold_mask = val_raw["is_cold_facility"] == 1

print("\n=== FOLD A SEGMENT BREAKDOWN (Cutoff 2025-03-31) ===")
print("Total Rows:", len(val_raw))
print(f"Warm Rows: {warm_mask.sum():,} ({warm_mask.mean()*100:.2f}%) | Unique Warm Facilities: {val_raw[warm_mask]['tanim'].nunique()}")
print(f"Cold Rows: {cold_mask.sum():,} ({cold_mask.mean()*100:.2f}%) | Unique Cold Facilities: {val_raw[cold_mask]['tanim'].nunique()}")

print("\n" + "=" * 68)
print(f"{'Model':<20} | {'Total RMSLE':<12} | {'Warm RMSLE':<12} | {'Cold RMSLE':<12}")
print("=" * 68)

models = [
    ("V13 (Pure LGB)", "pred_v13"),
    ("V13.5 (Ensemble)", "pred_v13_5"),
    ("V14 (Archetype)", "pred_v14"),
    ("Blend (0.7V14+0.3V13.5)", "pred_blend_70_30"),
]

for name, col in models:
    tot_r = calculate_rmsle(val_raw["tuketim"].values, val_raw[col].values)
    warm_r = calculate_rmsle(val_raw[warm_mask]["tuketim"].values, val_raw[warm_mask][col].values)
    cold_r = calculate_rmsle(val_raw[cold_mask]["tuketim"].values, val_raw[cold_mask][col].values)
    print(f"{name:<20} | {tot_r:<12.5f} | {warm_r:<12.5f} | {cold_r:<12.5f}")
print("=" * 68)

