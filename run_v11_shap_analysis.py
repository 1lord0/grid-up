"""Yeni V11 Sızıntısız Veri Seti Üzerinde Çoklu Fold ve Segment SHAP Analizi.

Kurallar:
- log1p(tuketim) hedefiyle eğitilir.
- Random validation kullanılmaz; 3 ayrı ileri-zaman foldunda hesaplanır.
- Segment bazında raporlanır: overall, annual, warm, cold, Nisan, Mayıs, Haziran, Temmuz, zero-heavy.
- 3 foldun en az 2'sinde etkisiz olan değişkenler budama adayı olarak listelenir.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"C:\Users\EREN\Desktop\grid-up-datathon")
OUTPUT_DIR = DATA_DIR / "features_v11_shap"
SHAP_OUT_DIR = OUTPUT_DIR / "shap_analysis"


def run_shap_analysis():
    logger.info("=" * 70)
    logger.info("STARTING V11 LEAKAGE-FREE SHAP ANALYSIS")
    logger.info("=" * 70)

    SHAP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train_features_v11.csv.gz"

    logger.info(f"Loading {train_path}...")
    df = pd.read_csv(train_path, compression="gzip", encoding="utf-8")
    logger.info(f"Loaded: {len(df):,d} rows, {len(df.columns)} columns.")

    meta_cols = ["row_id", "tanim", "tarih", "cutoff_date", "fold_id", "segment", "tuketim"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logger.info(f"Feature count: {len(feature_cols)}")

    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or df[c].dtype.name == "category"
    ]
    logger.info(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    for c in cat_cols:
        df[c] = df[c].astype("category")

    folds = [
        ("fold_a_apr_jul_2025", "fold_b_aug_nov_2025"),
        ("fold_b_aug_nov_2025", "fold_c_dec_mar_2026"),
        ("fold_a_apr_jul_2025", "fold_c_dec_mar_2026"),
    ]

    fold_shap_results = []
    segmented_shap_records = []

    for fold_idx, (tr_fold, va_fold) in enumerate(folds, start=1):
        logger.info(f"\n>>> Running SHAP Evaluation on Validation Fold: {va_fold} (Trained on {tr_fold}) <<<")
        tr_mask = df["fold_id"] == tr_fold
        va_mask = df["fold_id"] == va_fold

        X_tr = df.loc[tr_mask, feature_cols]
        y_tr = np.log1p(df.loc[tr_mask, "tuketim"].values)

        X_va = df.loc[va_mask, feature_cols]
        y_va = np.log1p(df.loc[va_mask, "tuketim"].values)
        va_df = df.loc[va_mask].reset_index(drop=True)

        logger.info(f"Train N={len(X_tr):,d} | Val N={len(X_va):,d}")

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold_idx,
            n_jobs=-1,
            importance_type="gain",
        )
        model.fit(X_tr, y_tr)

        # Sample for SHAP computation if val is large
        sample_size = min(10000, len(X_va))
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_va), size=sample_size, replace=False)
        X_sample = X_va.iloc[sample_idx]
        va_sample_df = va_df.iloc[sample_idx].reset_index(drop=True)

        logger.info(f"Computing TreeSHAP values on {sample_size:,d} validation samples...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Global Mean Absolute SHAP for this fold
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        fold_shap_df = pd.DataFrame({
            "feature": feature_cols,
            f"shap_val_fold_{fold_idx}": mean_abs_shap,
        })
        fold_shap_results.append(fold_shap_df)

        # Segment-specific SHAP
        segments = {
            "overall": np.ones(len(va_sample_df), dtype=bool),
            "annual": (va_sample_df["segment"] == "annual").values,
            "warm": (va_sample_df["segment"] == "warm").values,
            "cold": (va_sample_df["segment"] == "cold").values,
            "april": (pd.to_datetime(va_sample_df["tarih"]).dt.month == 4).values,
            "may": (pd.to_datetime(va_sample_df["tarih"]).dt.month == 5).values,
            "june": (pd.to_datetime(va_sample_df["tarih"]).dt.month == 6).values,
            "july": (pd.to_datetime(va_sample_df["tarih"]).dt.month == 7).values,
            "zero_heavy": (va_sample_df["zero_category"] == "Heavy_Zero (>25%)").values,
        }

        for seg_name, mask in segments.items():
            if mask.sum() > 10:
                seg_shap = np.abs(shap_values[mask]).mean(axis=0)
                for f_name, val in zip(feature_cols, seg_shap):
                    segmented_shap_records.append({
                        "fold": va_fold,
                        "segment": seg_name,
                        "feature": f_name,
                        "mean_abs_shap": float(val),
                    })

    # Combine fold results
    combined_shap = fold_shap_results[0]
    for res in fold_shap_results[1:]:
        combined_shap = pd.merge(combined_shap, res, on="feature")

    val_cols = [c for c in combined_shap.columns if c.startswith("shap_val_fold_")]
    combined_shap["mean_abs_shap"] = combined_shap[val_cols].mean(axis=1)
    combined_shap["active_fold_count"] = (combined_shap[val_cols] > 1e-5).sum(axis=1)
    combined_shap["total_shap"] = combined_shap["mean_abs_shap"].sum()
    combined_shap["shap_pct"] = (combined_shap["mean_abs_shap"] / combined_shap["total_shap"]) * 100
    combined_shap = combined_shap.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    combined_shap["cum_pct"] = combined_shap["shap_pct"].cumsum()

    # Save summary report
    rep_path = SHAP_OUT_DIR / "v11_shap_importance_report.csv"
    combined_shap.to_csv(rep_path, index=False)
    logger.info(f"✓ Saved global V11 SHAP report to {rep_path}")

    # Segmented report
    seg_df = pd.DataFrame(segmented_shap_records)
    pivot_seg = seg_df.pivot_table(
        index="feature", columns="segment", values="mean_abs_shap", aggfunc="mean"
    ).reset_index()
    seg_rep_path = SHAP_OUT_DIR / "v11_segmented_shap_report.csv"
    pivot_seg.to_csv(seg_rep_path, index=False)
    logger.info(f"✓ Saved segmented V11 SHAP report to {seg_rep_path}")

    # Print Top Features
    logger.info("\n" + "=" * 70)
    logger.info("TOP 25 FEATURES BY V11 LEAKAGE-FREE SHAP IMPORTANCE:")
    logger.info("=" * 70)
    top25 = combined_shap.head(25)[["feature", "mean_abs_shap", "shap_pct", "cum_pct", "active_fold_count"]]
    logger.info("\n" + top25.to_string(index=False))

    # Dead Candidate Features across folds (<= 1 active fold out of 3)
    dead_candidates = combined_shap[combined_shap["active_fold_count"] <= 1]
    logger.info(f"\nCandidates with <=1 active fold (Pruning Candidates): {len(dead_candidates)}")
    if len(dead_candidates) > 0:
        logger.info("\n" + dead_candidates[["feature", "mean_abs_shap", "active_fold_count"]].to_string(index=False))


if __name__ == "__main__":
    run_shap_analysis()
