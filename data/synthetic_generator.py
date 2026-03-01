"""
HireLens – Synthetic Job Data Generator
No external dependencies beyond pandas + numpy.
"""

import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

TOTAL_JOBS  = 2500
GHOST_RATIO = 0.38
OUTPUT_PATH = "data/synthetic_jobs.csv"

INDUSTRIES = ["Technology","Finance","Healthcare","Retail","Manufacturing","Consulting","Media","Education","Logistics"]

JOB_TITLES = [
    "Software Engineer","Data Scientist","Product Manager","DevOps Engineer",
    "ML Engineer","Backend Developer","Frontend Developer","Data Analyst",
    "QA Engineer","Cloud Architect","Security Engineer","UX Designer",
    "Business Analyst","Scrum Master","Full Stack Developer",
    "Marketing Manager","HR Generalist","Financial Analyst",
    "Sales Executive","Operations Manager","Data Engineer",
    "Site Reliability Engineer","Technical Lead","Platform Engineer"
]

LOCATIONS = [
    "New York, NY","San Francisco, CA","Austin, TX","Chicago, IL",
    "Seattle, WA","Boston, MA","Los Angeles, CA","Denver, CO",
    "Remote","London, UK","Toronto, CA","Berlin, DE",
    "Atlanta, GA","Miami, FL","Dallas, TX","Phoenix, AZ"
]

SENIORITY = ["Intern","Junior","Mid-level","Senior","Lead","Principal","Director"]

PREFIXES = ["Apex","Nova","Vertex","Horizon","Quantum","Nexus","Stellar","Pinnacle","Meridian","Zenith",
            "Catalyst","Synapse","Vortex","Luminary","Cascade","Prism","Axiom","Orion","Titan","Forge",
            "Arch","Ionic","Vector","Cipher","Blaze","Drift","Echo","Flux","Grid","Helix",
            "Iris","Jade","Kore","Lynx","Monolith","Neon","Opus","Pulse","Quark","Ridge"]
SUFFIXES = ["Technologies","Solutions","Systems","Group","Ventures","Labs","Analytics","Dynamics",
            "Innovations","Digital","Consulting","Networks","Platforms","Partners","Intelligence",
            "Capital","Works","Studio","Collective","Global"]

def make_companies(n=120):
    seen, companies = set(), []
    while len(companies) < n:
        name = f"{random.choice(PREFIXES)} {random.choice(SUFFIXES)}"
        if name in seen: continue
        seen.add(name)
        profile = random.choices(["authentic","ghost","mixed"], weights=[0.50,0.30,0.20])[0]
        companies.append({"name":name,"industry":random.choice(INDUSTRIES),
                          "profile":profile,"size":random.choice(["startup","mid","enterprise"])})
    return companies

COMPANIES = make_companies()

def rand_date(start=365, end=0):
    return datetime.now() - timedelta(days=random.randint(end, start))

def age(d): return (datetime.now() - d).days

def salary(seniority):
    r = {"Intern":(35000,60000),"Junior":(55000,85000),"Mid-level":(80000,120000),
         "Senior":(115000,160000),"Lead":(140000,185000),"Principal":(160000,210000),"Director":(175000,250000)}
    lo,hi = r.get(seniority,(60000,120000))
    return random.randint(lo,hi)

def ghost_job(company, jid):
    t = random.choice(JOB_TITLES); s = random.choice(SENIORITY)
    posted = rand_date(365,60); a = age(posted)
    rc = random.randint(3,12); ri = random.randint(7,21)
    op = random.randint(15,60); cl = random.randint(0,4)
    sal = salary(s) if random.random()>0.55 else None
    return {"job_id":jid,"title":t,"seniority":s,"company":company["name"],"industry":company["industry"],
            "company_size":company["size"],"location":random.choice(LOCATIONS),
            "posted_date":posted.strftime("%Y-%m-%d"),"listing_age_days":a,
            "salary":sal,"salary_missing":int(sal is None),
            "repost_count":rc,"repost_interval_days":ri,
            "company_open_roles":op,"company_closed_roles":cl,
            "closure_ratio":round(cl/max(op,1),4),"hiring_velocity":round(op/12,2),
            "role_recycling_score":round(random.uniform(0.65,1.0),4),
            "engagement_latency":random.randint(10,45),
            "status":random.choices(["open","closed"],weights=[0.92,0.08])[0],"is_ghost":1}

def authentic_job(company, jid):
    t = random.choice(JOB_TITLES); s = random.choice(SENIORITY)
    posted = rand_date(45,1); a = age(posted)
    rc = random.randint(0,1); ri = random.randint(30,90) if rc>0 else 0
    op = random.randint(2,20); cl = random.randint(5,25)
    sal = salary(s) if random.random()>0.15 else None
    return {"job_id":jid,"title":t,"seniority":s,"company":company["name"],"industry":company["industry"],
            "company_size":company["size"],"location":random.choice(LOCATIONS),
            "posted_date":posted.strftime("%Y-%m-%d"),"listing_age_days":a,
            "salary":sal,"salary_missing":int(sal is None),
            "repost_count":rc,"repost_interval_days":ri,
            "company_open_roles":op,"company_closed_roles":cl,
            "closure_ratio":round(cl/max(op,1),4),"hiring_velocity":round(op/12,2),
            "role_recycling_score":round(random.uniform(0.0,0.35),4),
            "engagement_latency":random.randint(1,10),
            "status":random.choices(["open","closed"],weights=[0.45,0.55])[0],"is_ghost":0}

def mixed_job(company, jid):
    if random.random()<0.5:
        j = authentic_job(company,jid)
        j["listing_age_days"] += random.randint(10,30)
        j["repost_count"] += random.randint(1,2)
        j["role_recycling_score"] = min(j["role_recycling_score"]+random.uniform(0.1,0.25),1.0)
        j["is_ghost"] = 0
    else:
        j = ghost_job(company,jid)
        j["closure_ratio"] = min(j["closure_ratio"]+random.uniform(0.05,0.15),1.0)
    return j

def generate(n=TOTAL_JOBS, gr=GHOST_RATIO):
    records = []
    ng = int(n*gr); na = n-ng
    print(f"Generating {n} postings  |  Ghost: {ng}  Authentic: {na}")
    gc = [c for c in COMPANIES if c["profile"]=="ghost"]
    ac = [c for c in COMPANIES if c["profile"]=="authentic"]
    mc = [c for c in COMPANIES if c["profile"]=="mixed"]
    for _ in range(ng):
        c = random.choice(gc+mc)
        records.append(ghost_job(c, str(uuid.uuid4())[:8].upper()))
    for _ in range(na):
        c = random.choice(ac+mc)
        records.append(mixed_job(c, str(uuid.uuid4())[:8].upper()) if c["profile"]=="mixed" else authentic_job(c, str(uuid.uuid4())[:8].upper()))
    df = pd.DataFrame(records).sample(frac=1,random_state=42).reset_index(drop=True)
    df["closure_ratio"] = df["closure_ratio"].clip(0,1)
    df["role_recycling_score"] = df["role_recycling_score"].clip(0,1)
    df["listing_longevity_ratio"] = (df["listing_age_days"]/30).round(4)
    df["open_close_ratio"] = (df["company_open_roles"]/df["company_closed_roles"].replace(0,0.5)).round(4)
    return df

if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved → {OUTPUT_PATH}  |  Shape: {df.shape}")
    print("\n── Label Distribution ──")
    print(df["is_ghost"].value_counts().rename(index={0:"Authentic",1:"Ghost"}).to_string())
    print("\n── Sample ──")
    print(df[["title","company","listing_age_days","repost_count","closure_ratio","role_recycling_score","is_ghost"]].head(10).to_string(index=False))
    print("\n── Stats ──")
    print(df[["listing_age_days","repost_count","repost_interval_days","closure_ratio","role_recycling_score","engagement_latency","listing_longevity_ratio","open_close_ratio"]].describe().round(2).to_string())
