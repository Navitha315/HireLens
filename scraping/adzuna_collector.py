"""
HireLens – Adzuna API Collector
Fetches live job postings and maps them to the HireLens feature schema.

Setup:
  1. Register free at https://developer.adzuna.com/
  2. Get your APP_ID and APP_KEY
  3. Set environment variables:
       export ADZUNA_APP_ID=your_app_id
       export ADZUNA_APP_KEY=your_app_key
     OR pass them directly when calling AdzunaCollector(app_id=..., app_key=...)
"""

import os
import uuid
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

from dotenv import load_dotenv
from pathlib import Path

# Force load .env from project root (1 level above scraping/)
ROOT_DIR = Path(__file__).resolve().parents[1]
#load_dotenv(ROOT_DIR / ".env")

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path, override=True)

print("ENV FILE PATH:", ROOT_DIR / ".env")
print("APP ID:", os.getenv("ADZUNA_APP_ID"))
#print("APP KEY:", os.getenv("ADZUNA_APP_KEY"))

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
DEFAULT_COUNTRY = "in"          # us | gb | au | ca | de | fr | in | nl | nz | za
DEFAULT_RESULTS = 50            # per page (max 50)
DEFAULT_PAGES   = 4             # pages to fetch → up to 200 jobs per query
INDUSTRY_AVG_CLOSURE_DAYS = 30  # baseline for listing longevity ratio

OUTPUT_PATH = "data/adzuna_jobs.csv"


# ─────────────────────────────────────────────
# COLLECTOR
# ─────────────────────────────────────────────

class AdzunaCollector:
    def __init__(self, app_id=None, app_key=None, country=DEFAULT_COUNTRY):

        self.app_id  = app_id  or os.getenv("ADZUNA_APP_ID")
        self.app_key = app_key or os.getenv("ADZUNA_APP_KEY")

        if not self.app_id or not self.app_key:
            raise ValueError(
                "Adzuna credentials not found. "
                "Ensure ADZUNA_APP_ID and ADZUNA_APP_KEY are set in .env."
            )


        self.country = country
        self.base    = f"{ADZUNA_BASE_URL}/{self.country}/search"

        if "YOUR_APP" in self.app_id:
            print("⚠️  WARNING: Adzuna credentials not set.")
            print("   Set ADZUNA_APP_ID and ADZUNA_APP_KEY env vars, or pass them to AdzunaCollector().")

    # ── raw API call ──────────────────────────
    def _fetch_page(self, keyword, page=1, results_per_page=DEFAULT_RESULTS):
        params = {
            "app_id":           self.app_id,
            "app_key":          self.app_key,
            "results_per_page": results_per_page,
            "what":             keyword,
            "content-type":     "application/json",
        }
        url = f"{self.base}/{page}"
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Page {page} failed: {e}")
            return None

    # ── parse one raw result into HireLens schema ──
    def _parse_job(self, raw):
        job_id   = str(uuid.uuid4())[:8].upper()
        title    = raw.get("title", "Unknown")
        company  = raw.get("company", {}).get("display_name", "Unknown")
        location = raw.get("location", {}).get("display_name", "Unknown")
        category = raw.get("category", {}).get("label", "Unknown")

        # Posted date → listing age
        created_str = raw.get("created", "")
        try:
            posted_dt = datetime.strptime(created_str[:10], "%Y-%m-%d")
            listing_age_days = (datetime.now() - posted_dt).days
        except Exception:
            listing_age_days = 0
            posted_dt = datetime.now()

        # Salary
        sal_min = raw.get("salary_min")
        sal_max = raw.get("salary_max")
        salary  = int((sal_min + sal_max) / 2) if sal_min and sal_max else None

        # Derived features
        # NOTE: Adzuna doesn't provide company-level hiring history in free tier.
        # We estimate using heuristics / placeholder values for live demo.
        # Full feature enrichment happens in the feature_engineering module.
        listing_longevity_ratio = round(listing_age_days / INDUSTRY_AVG_CLOSURE_DAYS, 4)

        return {
            "job_id":                 job_id,
            "title":                  title,
            "seniority":              self._infer_seniority(title),
            "company":                company,
            "industry":               category,
            "company_size":           "unknown",         # not available in free tier
            "location":               location,
            "posted_date":            posted_dt.strftime("%Y-%m-%d"),
            "listing_age_days":       listing_age_days,
            "salary":                 salary,
            "salary_missing":         int(salary is None),
            "repost_count":           0,                 # enriched later via dedup check
            "repost_interval_days":   0,                 # enriched later
            "company_open_roles":     0,                 # enriched later
            "company_closed_roles":   0,                 # enriched later
            "closure_ratio":          0.0,               # enriched later
            "hiring_velocity":        0.0,               # enriched later
            "role_recycling_score":   0.0,               # enriched later
            "engagement_latency":     0,                 # enriched later
            "status":                 "open",
            "listing_longevity_ratio": listing_longevity_ratio,
            "open_close_ratio":       0.0,               # enriched later
            "is_ghost":               -1,                # unknown (live data, no label)
            "source":                 "adzuna_api",
            "raw_description_snippet": raw.get("description", "")[:300],
        }

    def _infer_seniority(self, title):
        t = title.lower()
        if any(x in t for x in ["intern","graduate","entry"]):    return "Intern"
        if any(x in t for x in ["junior","jr","associate"]):      return "Junior"
        if any(x in t for x in ["senior","sr","principal","staff"]): return "Senior"
        if any(x in t for x in ["lead","head","manager"]):         return "Lead"
        if any(x in t for x in ["director","vp","chief"]):         return "Director"
        return "Mid-level"

    # ── enrich company-level features from batch ──
    def _enrich_company_features(self, df):
        """
        Since Adzuna free tier doesn't expose historical company data,
        we derive company-level signals from within our fetched batch.
        """
        print("  Enriching company-level features from batch...")

        company_stats = df.groupby("company").agg(
            company_open_roles=("job_id", "count"),
            avg_listing_age=("listing_age_days", "mean"),
        ).reset_index()

        df = df.merge(company_stats, on="company", how="left", suffixes=("","_new"))
        df["company_open_roles"] = df["company_open_roles_new"].fillna(1)
        df.drop(columns=["company_open_roles_new"], errors="ignore", inplace=True)

        # Repost detection: same company + same title = likely repost
        title_counts = df.groupby(["company","title"]).size().reset_index(name="repost_count_est")
        df = df.merge(title_counts, on=["company","title"], how="left")
        df["repost_count"] = (df["repost_count_est"] - 1).clip(lower=0)
        df.drop(columns=["repost_count_est"], errors="ignore", inplace=True)

        # Role recycling score proxy: if repost_count > 0
        df["role_recycling_score"] = (df["repost_count"] / df["repost_count"].max().clip(1)).round(4)

        # Hiring velocity: open roles / 12 months
        df["hiring_velocity"] = (df["company_open_roles"] / 12).round(2)

        # Listing longevity ratio (already set per job)
        # open_close_ratio: without closure data, approximate as listing_age / 30
        df["open_close_ratio"] = df["listing_longevity_ratio"]

        return df

    # ── main fetch function ──
    def fetch(self, keywords, pages=DEFAULT_PAGES, results_per_page=DEFAULT_RESULTS):
        """
        Fetch jobs for one or more keywords.

        Args:
            keywords (str | list): e.g. "Data Scientist" or ["Data Scientist", "ML Engineer"]
            pages (int): number of pages per keyword
            results_per_page (int): up to 50

        Returns:
            pd.DataFrame
        """
        if isinstance(keywords, str):
            keywords = [keywords]

        all_records = []
        for kw in keywords:
            print(f"\nFetching: '{kw}'")
            for page in range(1, pages + 1):
                print(f"  Page {page}/{pages}...", end=" ")
                data = self._fetch_page(kw, page, results_per_page)
                if data is None:
                    break
                results = data.get("results", [])
                print(f"{len(results)} jobs")
                for raw in results:
                    all_records.append(self._parse_job(raw))
                time.sleep(0.3)  # polite rate limiting

        if not all_records:
            print("\n⚠️  No records fetched. Check credentials or keyword.")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = self._enrich_company_features(df)
        df.drop_duplicates(subset=["company","title","location"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"\n✅ Fetched {len(df)} unique job postings from Adzuna")
        return df


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    # ── Replace with your credentials ──────────────────────────
    # Get free API key: https://developer.adzuna.com/signup

    collector = AdzunaCollector()

    # Fetch multiple job categories for richer dataset
    SEARCH_KEYWORDS = [
        "Data Scientist",
        "Software Engineer",
        "Product Manager",
        "DevOps Engineer",
        "Data Analyst",
    ]

    df = collector.fetch(keywords=SEARCH_KEYWORDS, pages=4)

    if not df.empty:

        if os.path.exists(OUTPUT_PATH):
            old = pd.read_csv(OUTPUT_PATH)
            df = pd.concat([old, df]).drop_duplicates("job_id")
            
        df.to_csv(OUTPUT_PATH, index=False)

        #df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n── Saved → {OUTPUT_PATH}  |  Shape: {df.shape}")
        print("\n── Preview ──")
        print(df[["title","company","location","listing_age_days",
                   "salary_missing","repost_count","role_recycling_score"]].head(10).to_string(index=False))
        print("\n── Feature Stats ──")
        print(df[["listing_age_days","salary_missing","repost_count",
                   "role_recycling_score","listing_longevity_ratio"]].describe().round(2).to_string())
    else:
        print("\nNo data saved. Using synthetic fallback instead.")
