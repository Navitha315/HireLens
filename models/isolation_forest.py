"""
HireLens – Isolation Forest Model
Trains on synthetic labelled data, scores Adzuna live data.
Outputs BHAS (Behavioural Hiring Authenticity Score) 0-100 for every job.

Run:
    python models/isolation_forest.py

Outputs:
    data/scored_jobs.csv        → all jobs with BHAS score + risk tier
    models/model_metrics.txt    → ROC-AUC, precision, recall on synthetic
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


import os

print("Current Working Directory:", os.getcwd())



# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

TRAIN_PATH    = "data/features_train.csv"
SCORE_PATH    = "data/features_score.csv"
COMBINED_PATH = "data/features_combined.csv"
OUT_SCORED    = "data/scored_jobs.csv"
OUT_METRICS   = "models/model_metrics.txt"

MODEL_FEATURES = [
    "listing_age_days",
    "listing_longevity_ratio",
    "salary_missing",
    "repost_count",
    "role_recycling_score",
    "hiring_velocity",
    "open_close_ratio",
]

# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────

def load():
    print("Loading feature files...")


    train = pd.read_csv(TRAIN_PATH,
                        encoding="latin1",
                        engine="python",
                        on_bad_lines="skip")

    score = pd.read_csv(SCORE_PATH,
                        encoding="latin1",
                        engine="python",
                        on_bad_lines="skip")

    combined_raw = pd.read_csv(COMBINED_PATH,
                            encoding="latin1",
                            engine="python",
                            on_bad_lines="skip")



    print(f"  Train (synthetic) : {train.shape[0]} rows")
    print(f"  Score (adzuna)    : {score.shape[0]} rows")
    return train, score, combined_raw

# ─────────────────────────────────────────────
# STEP 2 — TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────

def train_model(train_df):
    """
    Isolation Forest trained ONLY on authentic jobs.
    Logic: teach the model what normal hiring looks like.
    Ghost jobs will be anomalies — they deviate from normal.

    contamination = expected fraction of anomalies in the wild.
    We set 0.38 to match our synthetic ghost ratio.
    """
    print("\nTraining Isolation Forest...")

    # Train only on authentic jobs — this is key
    authentic = train_df[train_df["is_ghost"] == 0][MODEL_FEATURES]
    print(f"  Training on {len(authentic)} authentic job records")

    model = IsolationForest(
        n_estimators=200,       # more trees = more stable scores
        contamination=0.38,     # expected ghost ratio
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(authentic)
    print("  Training complete.")
    return model

# ─────────────────────────────────────────────
# STEP 3 — EVALUATE ON SYNTHETIC (labelled)
# ─────────────────────────────────────────────

def evaluate(model, train_df):
    """
    Evaluate on full synthetic dataset (has labels).
    Isolation Forest returns:
        -1 = anomaly (ghost)
         1 = normal  (authentic)
    We flip to:
         1 = ghost
         0 = authentic
    """
    print("\nEvaluating on synthetic labelled data...")

    X     = train_df[MODEL_FEATURES]
    y_true = train_df["is_ghost"]

    raw_scores = model.decision_function(X)   # lower = more anomalous
    predictions = model.predict(X)            # -1 or 1

    # Convert predictions: -1 → 1 (ghost), 1 → 0 (authentic)
    y_pred = (predictions == -1).astype(int)

    # Convert raw scores to probability-like ghost score (0–1)
    # decision_function: lower = more anomalous, so we invert and normalise
    ghost_score_raw = -raw_scores
    ghost_prob = (ghost_score_raw - ghost_score_raw.min()) / \
                 (ghost_score_raw.max() - ghost_score_raw.min())

    auc   = roc_auc_score(y_true, ghost_prob)
    report = classification_report(y_true, y_pred, target_names=["Authentic","Ghost"])
    cm    = confusion_matrix(y_true, y_pred)

    print(f"\n  ROC-AUC Score : {auc:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix:\n{cm}")

    metrics = {
        "roc_auc":           round(auc, 4),
        "classification_report": report,
        "confusion_matrix":  cm.tolist(),
        "n_train":           len(train_df),
        "n_ghost_true":      int(y_true.sum()),
        "n_ghost_predicted": int(y_pred.sum()),
    }
    return metrics, ghost_prob

# ─────────────────────────────────────────────
# STEP 4 — SCORE ADZUNA DATA
# ─────────────────────────────────────────────

def score_adzuna(model, score_df):
    print("\nScoring Adzuna live jobs...")

    X = score_df[MODEL_FEATURES]
    raw_scores = model.decision_function(X)

    ghost_score_raw = -raw_scores
    ghost_prob = (ghost_score_raw - ghost_score_raw.min()) / \
                 (ghost_score_raw.max() - ghost_score_raw.min())

    print(f"  Scored {len(score_df)} live job postings")
    return ghost_prob

# ─────────────────────────────────────────────
# STEP 5 — BHAS SCORE + RISK TIER
# ─────────────────────────────────────────────

def compute_bhas(ghost_prob):
    """
    BHAS = Behavioural Hiring Authenticity Score
    Higher = MORE authentic (more trustworthy)
    ghost_prob is 0–1 where 1 = very ghost-like

    BHAS = (1 - ghost_prob) * 100
    """
    bhas = ((1 - ghost_prob) * 100).round(1)
    return bhas

def assign_risk_tier(bhas):
    """
    Risk tiers based on BHAS score.
    """
    def tier(score):
        if score >= 75:   return "Low Risk"
        elif score >= 50: return "Moderate Risk"
        elif score >= 30: return "High Risk"
        else:             return "Ghost"
    return [tier(s) for s in bhas]

# ─────────────────────────────────────────────
# STEP 6 — ASSEMBLE FINAL SCORED DATAFRAME
# ─────────────────────────────────────────────

def assemble_output(train_df, score_df, combined_raw,
                    syn_ghost_prob, adz_ghost_prob):

    # ── Synthetic scored ──
    syn_out = train_df.copy()
    syn_out["ghost_probability"] = syn_ghost_prob.round(4)
    syn_out["bhas_score"]        = compute_bhas(syn_ghost_prob)
    syn_out["risk_tier"]         = assign_risk_tier(syn_out["bhas_score"])
    syn_out["source"]            = "synthetic"

    # ── Adzuna scored ──
    adz_out = score_df.copy()
    adz_out["ghost_probability"] = adz_ghost_prob.round(4)
    adz_out["bhas_score"]        = compute_bhas(adz_ghost_prob)
    adz_out["risk_tier"]         = assign_risk_tier(adz_out["bhas_score"])
    adz_out["is_ghost"]          = -1   # unknown
    adz_out["source"]            = "adzuna"

    # ── Align columns ──
    keep = ["job_id","title","seniority","company","industry","location",
            "posted_date","source","is_ghost",
            "ghost_probability","bhas_score","risk_tier"] + MODEL_FEATURES

    syn_keep = syn_out[[c for c in keep if c in syn_out.columns]]
    adz_keep = adz_out[[c for c in keep if c in adz_out.columns]]

    final = pd.concat([syn_keep, adz_keep], ignore_index=True)
    return final

# ─────────────────────────────────────────────
# STEP 7 — SAVE
# ─────────────────────────────────────────────

def save_outputs(final_df, metrics):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Scored jobs
    final_df.to_csv(OUT_SCORED, index=False)
    print(f"\n✅ Saved scored jobs → {OUT_SCORED}  ({final_df.shape})")

    # Metrics text file
    with open(OUT_METRICS, "w") as f:
        f.write("HireLens – Isolation Forest Model Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"ROC-AUC Score        : {metrics['roc_auc']}\n")
        f.write(f"Training samples     : {metrics['n_train']}\n")
        f.write(f"True ghost count     : {metrics['n_ghost_true']}\n")
        f.write(f"Predicted ghost count: {metrics['n_ghost_predicted']}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics["classification_report"])
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    print(f"✅ Saved model metrics → {OUT_METRICS}")

# ─────────────────────────────────────────────
# STEP 8 — PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(final_df):
    print("\n── BHAS Score Distribution ──")

    syn = final_df[final_df["source"] == "synthetic"]
    adz = final_df[final_df["source"] == "adzuna"]

    print("\nSynthetic (labelled):")
    print(syn.groupby("risk_tier")["bhas_score"].agg(["count","mean"]).round(2).to_string())

    print("\nAdzuna (live jobs):")
    print(adz.groupby("risk_tier")["bhas_score"].agg(["count","mean"]).round(2).to_string())

    print("\n── Top 10 Highest Ghost Risk (Adzuna) ──")
    top_ghost = adz.nsmallest(10, "bhas_score")[
        ["title","company","location","bhas_score","risk_tier",
         "listing_age_days","repost_count","salary_missing"]
    ]
    print(top_ghost.to_string(index=False))

    print("\n── Top 10 Most Authentic (Adzuna) ──")
    top_auth = adz.nlargest(10, "bhas_score")[
        ["title","company","location","bhas_score","risk_tier",
         "listing_age_days","repost_count","salary_missing"]
    ]
    print(top_auth.to_string(index=False))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_df, score_df, combined_raw = load()

    model = train_model(train_df)

    metrics, syn_ghost_prob = evaluate(model, train_df)

    adz_ghost_prob = score_adzuna(model, score_df)

    final_df = assemble_output(
        train_df, score_df, combined_raw,
        syn_ghost_prob, adz_ghost_prob
    )

    save_outputs(final_df, metrics)

    print_summary(final_df)

    print("\n── All done. Next step: python dashboard/app.py ──")
