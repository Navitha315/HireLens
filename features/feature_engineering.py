"""
HireLens – Feature Engineering
Merges synthetic + adzuna data, selects features, normalises, outputs model-ready matrix.

Run:
    python features/feature_engineering.py

Outputs:
    data/features_train.csv   → synthetic data, labelled (for model training)
    data/features_score.csv   → adzuna data, unlabelled (for model scoring)
    data/features_combined.csv → both merged (for dashboard)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

SYNTHETIC_PATH = "data/synthetic_jobs.csv"
ADZUNA_PATH    = "data/adzuna_jobs.csv"
OUT_TRAIN      = "data/features_train.csv"
OUT_SCORE      = "data/features_score.csv"
OUT_COMBINED   = "data/features_combined.csv"

# ─────────────────────────────────────────────
# FEATURES USED BY THE MODEL
# These are the only columns that go into Isolation Forest.
# Chosen because they exist meaningfully in BOTH datasets.
# ─────────────────────────────────────────────

MODEL_FEATURES = [
    "listing_age_days",         # how long the job has been up
    "listing_longevity_ratio",  # listing_age / 30 day industry avg
    "salary_missing",           # 1 if no salary posted
    "repost_count",             # how many times role reposted
    "role_recycling_score",     # 0–1, how recycled the role is
    "hiring_velocity",          # open roles per month
    "open_close_ratio",         # open / closed roles ratio
]

# closure_ratio and engagement_latency excluded from model features
# because they are 0 across all Adzuna rows — would bias scoring unfairly.
# They are kept in the combined file for dashboard display only.

# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────

def load_data():
    print("Loading datasets...")

    # 🔥 FIX ADDED HERE (robust CSV loading)
    syn = pd.read_csv(SYNTHETIC_PATH,
                      encoding="latin1",
                      engine="python",
                      on_bad_lines="skip")

    adz = pd.read_csv(ADZUNA_PATH,
                      encoding="utf-8-sig")

    print(f"  Synthetic : {syn.shape[0]} rows")
    print(f"  Adzuna    : {adz.shape[0]} rows")


    print("\nAdzuna Columns:")
    print(adz.columns.tolist())


    syn["source"]   = "synthetic"
    adz["source"]   = "adzuna"

    return syn, adz

# ─────────────────────────────────────────────
# STEP 2 — CLEAN
# ─────────────────────────────────────────────

def clean(df, label):
    print(f"\nCleaning {label}...")
    original_len = len(df)

    # Cap extreme outliers — listing age can't exceed 2 years
    df["listing_age_days"] = df["listing_age_days"].clip(0, 730)

    # Cap open_close_ratio — outliers go to 99th percentile
    cap = df["open_close_ratio"].quantile(0.99)
    df["open_close_ratio"] = df["open_close_ratio"].clip(0, cap)

    # Cap hiring_velocity similarly
    cap_hv = df["hiring_velocity"].quantile(0.99)
    df["hiring_velocity"] = df["hiring_velocity"].clip(0, cap_hv)

    # Recompute listing_longevity_ratio after clipping
    df["listing_longevity_ratio"] = (df["listing_age_days"] / 30).round(4)

    # Fill any remaining nulls in model features with 0
    df[MODEL_FEATURES] = df[MODEL_FEATURES].fillna(0)

    print(f"  Rows before: {original_len} → after: {len(df)}")
    return df

# ─────────────────────────────────────────────
# STEP 3 — FEATURE SUMMARY (sanity check)
# ─────────────────────────────────────────────

def feature_summary(syn, adz):
    print("\n── Feature Means: Synthetic Ghost vs Authentic ──")
    ghost = syn[syn["is_ghost"] == 1][MODEL_FEATURES].mean()
    auth  = syn[syn["is_ghost"] == 0][MODEL_FEATURES].mean()
    summary = pd.DataFrame({"Ghost": ghost, "Authentic": auth})
    summary["Separation"] = (summary["Ghost"] - summary["Authentic"]).abs().round(4)
    print(summary.round(4).to_string())

    print("\n── Adzuna Feature Means (no labels) ──")
    print(adz[MODEL_FEATURES].mean().round(4).to_string())

# ─────────────────────────────────────────────
# STEP 4 — NORMALISE
# ─────────────────────────────────────────────

def normalise(syn, adz):
    """
    Fit scaler on synthetic training data only.
    Apply same scaler to Adzuna — prevents data leakage.
    """
    print("\nNormalising features (MinMaxScaler fit on synthetic)...")

    scaler = MinMaxScaler()
    scaler.fit(syn[MODEL_FEATURES])

    syn_scaled = syn.copy()
    adz_scaled = adz.copy()

    syn_scaled[MODEL_FEATURES] = scaler.transform(syn[MODEL_FEATURES])
    adz_scaled[MODEL_FEATURES] = scaler.transform(adz[MODEL_FEATURES])

    print("  Done.")
    return syn_scaled, adz_scaled, scaler

# ─────────────────────────────────────────────
# STEP 5 — SAVE
# ─────────────────────────────────────────────

def save(syn_scaled, adz_scaled, syn_raw, adz_raw):
    os.makedirs("data", exist_ok=True)

    # Ensure label column is restored after scaling
    syn_scaled["is_ghost"] = syn_raw["is_ghost"].values

    # Training set — scaled model features + label
    train_out = syn_scaled[MODEL_FEATURES + ["is_ghost", "job_id", "title",
                                              "company", "industry", "location",
                                              "posted_date", "source",
                                              "listing_age_days", "salary_missing",
                                              "closure_ratio", "engagement_latency",
                                              "status", "seniority"]].copy()
    train_out.to_csv(OUT_TRAIN, index=False)
    print(f"\n✅ Saved training features → {OUT_TRAIN}  ({train_out.shape})")

    # Scoring set — scaled model features, no label
    score_out = adz_scaled[MODEL_FEATURES + ["job_id", "title", "company",
                                              "industry", "location", "posted_date",
                                              "source", "listing_age_days",
                                              "salary_missing", "status",
                                              "seniority", "raw_description_snippet"]].copy()
    score_out.to_csv(OUT_SCORE, index=False)
    print(f"✅ Saved scoring features  → {OUT_SCORE}  ({score_out.shape})")

    # Combined (raw, unscaled) — for dashboard display
    keep_cols = ["job_id","title","seniority","company","industry","location",
                 "posted_date","listing_age_days","salary_missing","repost_count",
                 "role_recycling_score","hiring_velocity","open_close_ratio",
                 "listing_longevity_ratio","closure_ratio","engagement_latency",
                 "status","is_ghost","source"] + MODEL_FEATURES

    # only keep cols that exist
    syn_keep = syn_raw[[c for c in keep_cols if c in syn_raw.columns]].copy()
    adz_keep = adz_raw[[c for c in keep_cols if c in adz_raw.columns]].copy()

    combined = pd.concat([syn_keep, adz_keep], ignore_index=True)
    combined.to_csv(OUT_COMBINED, index=False)
    print(f"✅ Saved combined (raw)    → {OUT_COMBINED}  ({combined.shape})")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    syn_raw, adz_raw = load_data()

    syn_clean = clean(syn_raw.copy(), "synthetic")
    adz_clean = clean(adz_raw.copy(), "adzuna")

    feature_summary(syn_clean, adz_clean)

    syn_scaled, adz_scaled, scaler = normalise(syn_clean, adz_clean)

    save(syn_scaled, adz_scaled, syn_clean, adz_clean)

    print("\n── All done. Next step: python models/isolation_forest.py ──")
