"""
HireLens – Streamlit Dashboard
Behavioural Hiring Authenticity Intelligence Platform

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="HireLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>

/* ===================== GLOBAL ===================== */

html, body, [class*="css"] {

font-family: Inter, sans-serif;
font-size:18px;
color:white !important;

}

.stApp {

background:#0b0f1a;

}


/* ===================== SIDEBAR ===================== */

section[data-testid="stSidebar"] {

background:#111827;

}

section[data-testid="stSidebar"] * {

color:white !important;

}

/* Sidebar navigation */

div[role="radiogroup"] label {

font-size:21px !important;
font-weight:600 !important;

}

/* Sidebar selectbox WHITE BOX TEXT */

section[data-testid="stSidebar"] div[data-baseweb="select"] * {

color:#111827 !important;

}

/* Sidebar badges */

section[data-testid="stSidebar"] code {

background:#e5e7eb !important;
color:#111827 !important;
font-weight:700 !important;

}

/* Sidebar refresh button */

section[data-testid="stSidebar"] button {

background:#1f2937 !important;
color:white !important;
font-weight:600 !important;

}

section[data-testid="stSidebar"] button:disabled {

background:#e5e7eb !important;
color:#111827 !important;

}


/* ===================== HEADINGS ===================== */

h1,h2,h3,h4,h5,h6 {

color:white !important;

}

.stMarkdown p {

color:white !important;

}


/* ===================== SELECTBOX ===================== */

div[data-baseweb="select"] input {

color:#111827 !important;

}

div[data-baseweb="select"] span {

color:#111827 !important;

}

div[role="listbox"] div {

color:#111827 !important;

}


/* ===================== SLIDER ===================== */

label[data-testid="stWidgetLabel"] {

color:white !important;
font-weight:600 !important;

}

div[data-baseweb="slider"] span {

color:#111827 !important;
font-weight:700 !important;

}

div[data-baseweb="slider"] div[role="slider"] {

background:#ef4444 !important;

}


/* ===================== METRIC CARDS ===================== */

.metric-value {

font-size:42px !important;
font-weight:800 !important;
color:white !important;

}

.metric-label {

font-size:18px !important;
color:#d1d5db !important;

}


/* ===================== DATAFRAME ===================== */

.stDataFrame {

color:white !important;

}

/* white table cells */

[data-testid="stDataFrame"] table {

color:#111827 !important;

}


/* ===================== CODE BLOCKS (MOST IMPORTANT FIX) ===================== */

/* st.code() block */

pre {

background:#e5e7eb !important;
color:#111827 !important;
padding:14px !important;
border-radius:10px !important;

}

/* text inside */

pre code {

color:#111827 !important;
background:#e5e7eb !important;

}

/* STEP 1 WHITE TEXT FIX */

pre span {

color:#111827 !important;

}


/* markdown ``` block */

.stCodeBlock pre {

background:#e5e7eb !important;
color:#111827 !important;

}

.stCodeBlock code {

color:#111827 !important;

}


/* ===================== BULLETS ===================== */

.stMarkdown li {

color:white !important;

}

.stMarkdown span {

color:white !important;

}

.stMarkdown div {

color:white !important;

}


/* ===================== EXPANDER ===================== */

details summary {

color:white !important;
font-size:18px !important;

}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    scored   = pd.read_csv("data/scored_jobs.csv")
    combined = pd.read_csv("data/features_combined.csv")

    # Merge bhas_score and risk_tier into combined (for raw feature display)
    score_cols = scored[["job_id","bhas_score","risk_tier","ghost_probability"]].drop_duplicates("job_id")
    combined   = combined.merge(score_cols, on="job_id", how="left")

    # Raw listing age (unscaled) — use from combined
    adzuna = combined[combined["source"] == "adzuna"].copy()
    synth  = combined[combined["source"] == "synthetic"].copy()

    return scored, combined, adzuna, synth

scored, combined, adzuna, synth = load_data()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

TIER_COLORS = {
    "Ghost":         "#ff4b4b",
    "High Risk":     "#ff8c00",
    "Moderate Risk": "#ffd700",
    "Low Risk":      "#00cc88",
}

def tier_badge(tier):
    color = TIER_COLORS.get(tier, "#888")
    return f'<span class="tier-badge" style="background:{color}22;color:{color};border:1px solid {color}">{tier}</span>'

def bhas_color(score):
    if score < 30:   return "#ff4b4b"
    if score < 50:   return "#ff8c00"
    if score < 75:   return "#ffd700"
    return "#00cc88"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/job.png", width=60)
    st.title("🔍 HireLens")
    st.caption("Behavioural Hiring Intelligence Engine")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Job Explorer", "🏢 Company Analysis", "📈 Model Insights"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**Data Sources**")
    st.markdown(f"🧪 Synthetic: `{len(synth):,}` jobs")
    st.markdown(f"🌐 Adzuna API: `{len(adzuna):,}` jobs")
    st.markdown(f"📦 Total: `{len(combined):,}` jobs")

    st.divider()

    if st.button("🔄 Refresh Live Data"):

        import subprocess
        import os
        import sys

        # Get project root safely
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        BASE = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

        with st.spinner("Refreshing pipeline..."):

            # Use same Python interpreter running Streamlit
            PYTHON_EXE = sys.executable

            subprocess.run(
                [PYTHON_EXE, "scraping/adzuna_collector.py"],
                cwd=BASE
            )

            subprocess.run(
                [PYTHON_EXE, "features/feature_engineering.py"],
                cwd=BASE
            )

            subprocess.run(
                [PYTHON_EXE, "models/isolation_forest.py"],
                cwd=BASE
            )

            st.cache_data.clear()

        st.success("Live data refreshed!")
        st.rerun()







    st.divider()
    source_filter = st.selectbox("Show data from", ["Both", "Adzuna (Live)", "Synthetic"])

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────

if source_filter == "Adzuna (Live)":
    view_df = adzuna.copy()
elif source_filter == "Synthetic":
    view_df = synth.copy()
else:
    view_df = combined.copy()

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────

if page == "📊 Overview":
    st.markdown("# 🔍 HireLens — Hiring Authenticity Intelligence")
    st.markdown("Behavioural analytics engine that quantifies ghost job risk using recruitment lifecycle dynamics.")
    st.divider()

    # ── KPI Cards ──
    total     = len(view_df)
    ghost_pct = len(view_df[view_df["risk_tier"] == "Ghost"]) / total * 100 if total else 0
    high_pct  = len(view_df[view_df["risk_tier"] == "High Risk"]) / total * 100 if total else 0
    low_pct   = len(view_df[view_df["risk_tier"] == "Low Risk"]) / total * 100 if total else 0
    avg_bhas  = view_df["bhas_score"].mean() if "bhas_score" in view_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00cc88">{total:,}</div>
            <div class="metric-label">Total Jobs Analysed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value bhas-ghost">{ghost_pct:.1f}%</div>
            <div class="metric-label">Flagged as Ghost</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value bhas-high">{high_pct:.1f}%</div>
            <div class="metric-label">High Risk</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value bhas-low">{avg_bhas:.1f}</div>
            <div class="metric-label">Avg BHAS Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Tier distribution + BHAS histogram ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Tier Distribution")
        tier_counts = view_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        tier_order  = ["Ghost","High Risk","Moderate Risk","Low Risk"]
        tier_counts["Risk Tier"] = pd.Categorical(tier_counts["Risk Tier"], categories=tier_order, ordered=True)
        tier_counts  = tier_counts.sort_values("Risk Tier")
        colors       = [TIER_COLORS.get(t,"#888") for t in tier_counts["Risk Tier"]]

        fig = px.bar(tier_counts, x="Risk Tier", y="Count",
                     color="Risk Tier",
                     color_discrete_map=TIER_COLORS,
                     template="plotly_dark")
        fig.update_layout(showlegend=False, height=320,
                          plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                          margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### BHAS Score Distribution")
        fig2 = px.histogram(view_df, x="bhas_score", nbins=40,
                            color_discrete_sequence=["#4f8ef7"],
                            template="plotly_dark",
                            labels={"bhas_score": "BHAS Score"})
        fig2.update_layout(height=320,
                           plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                           margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Ghost risk by industry ──
    st.markdown("### Ghost Risk by Industry")
    ind_risk = view_df.groupby("industry").agg(
        avg_bhas=("bhas_score","mean"),
        count=("job_id","count"),
        ghost_count=("risk_tier", lambda x: (x=="Ghost").sum())
    ).reset_index()
    ind_risk["ghost_pct"] = (ind_risk["ghost_count"] / ind_risk["count"] * 100).round(1)
    ind_risk = ind_risk[ind_risk["count"] >= 5].sort_values("ghost_pct", ascending=True)

    fig3 = px.bar(ind_risk, x="ghost_pct", y="industry", orientation="h",
                  color="ghost_pct",
                  color_continuous_scale=["#00cc88","#ffd700","#ff4b4b"],
                  template="plotly_dark",
                  labels={"ghost_pct":"Ghost %","industry":"Industry"})
    fig3.update_layout(height=350, coloraxis_showscale=False,
                       plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                       margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Candidate effort waste estimator ──
    st.divider()
    st.markdown("### 🕐 Candidate Effort Waste Estimator")
    st.caption("Estimates total human hours wasted applying to ghost jobs")

    cw1, cw2 = st.columns(2)
    with cw1:
        avg_app_time = st.slider("Avg application time (hours)", 1, 5, 2)
    with cw2:
        avg_applicants = st.slider("Avg applicants per job", 50, 500, 150)


    ghost_jobs    = len(view_df[view_df["risk_tier"] == "Ghost"])
    wasted_hours  = ghost_jobs * avg_applicants * avg_app_time
    wasted_days   = wasted_hours / 8
    wasted_value  = wasted_hours * 25  # $25/hr avg

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ff4b4b">{wasted_hours:,}</div>
            <div class="metric-label">Wasted Human Hours</div>
        </div>""", unsafe_allow_html=True)
    with wc2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ff8c00">{wasted_days:,.0f}</div>
            <div class="metric-label">Equivalent Working Days</div>
        </div>""", unsafe_allow_html=True)
    with wc3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#ffd700">${wasted_value:,}</div>
            <div class="metric-label">Estimated Economic Loss</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: JOB EXPLORER
# ─────────────────────────────────────────────

elif page == "🔍 Job Explorer":

    st.markdown("# 🔍 Job Explorer")
    st.caption("Explore individual job behavioural signals and authenticity score")
    st.divider()

    # ── Filters ──
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_title = st.selectbox(
            "Select Job Title",
            sorted(view_df["title"].dropna().unique())
        )

    with col2:
        tier_filter = st.multiselect(
            "Risk Tier",
            ["Ghost", "High Risk", "Moderate Risk", "Low Risk"],
            default=["Ghost", "High Risk", "Moderate Risk", "Low Risk"]
        )

    with col3:
        industry_filter = st.multiselect(
            "Industry",
            sorted(view_df["industry"].dropna().unique()),
            default=list(view_df["industry"].dropna().unique())
        )

    # ── Apply Filters ──
    filtered = view_df.copy()

    filtered = filtered[filtered["title"] == selected_title]

    if tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]

    if industry_filter:
        filtered = filtered[filtered["industry"].isin(industry_filter)]

    if len(filtered) == 0:
        st.warning("No jobs match selected filters.")
        st.stop()

    job = filtered.iloc[0]
    score = job.get("bhas_score", 0)

    # ─────────────────────────
    # BHAS GAUGE (TOP SECTION)
    # ─────────────────────────

    st.subheader("BHAS Score")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Behavioural Hiring Authenticity Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': bhas_color(score)},
            'steps': [
                {'range': [0, 30], 'color': "#3b0d0d"},
                {'range': [30, 50], 'color': "#3a2c0d"},
                {'range': [50, 75], 'color': "#3a3a0d"},
                {'range': [75, 100], 'color': "#0d3a22"},
            ]
        }
    ))

    fig_gauge.update_layout(
        height=350,
        paper_bgcolor="#0b0f1a",
        font=dict(color="white")
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ─────────────────────────
    # JOB DETAILS + RADAR
    # ─────────────────────────

    colA, colB = st.columns([1.2, 1.8])

    with colA:
        st.markdown("### 📋 Job Details")

        st.markdown(f"""
        <div style="line-height:1.9;font-size:18px; color:white;">
        <b>Company:</b> {job.get('company','')}<br>
        <b>Location:</b> {job.get('location','')}<br>
        <b>Industry:</b> {job.get('industry','')}<br>
        <b>Seniority:</b> {job.get('seniority','')}<br>
        <b>Posted:</b> {job.get('posted_date','')}<br>
        <b>Source:</b> {job.get('source','')}<br>
        <b>Risk Tier:</b> 
            <span style="color:{TIER_COLORS.get(job.get('risk_tier',''), '#ffffff')};
                        font-weight:700;">
                {job.get('risk_tier','')}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with colB:

        st.markdown("### 🧠Behavioural Signal Radar")

        radar_features = [
            "listing_age_days",
            "repost_count",
            "salary_missing",
            "role_recycling_score",
            "hiring_velocity",
            "open_close_ratio"
        ]

        radar_features = [f for f in radar_features if f in job.index]

        values = []

        for f in radar_features:
            raw = job.get(f, 0)
            max_val = view_df[f].quantile(0.95) if f in view_df.columns else 1
            normalized = min(float(raw) / max(float(max_val), 0.001), 1.0)
            values.append(normalized)

        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_features + [radar_features[0]],
            fill="toself",
            fillcolor="rgba(255, 215, 0, 0.2)" if score < 75 else "rgba(0, 204, 136, 0.2)",
            line_color=bhas_color(score),
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_dark",
            paper_bgcolor="#1e2130",
            height=350,
            font=dict(color="white")
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ─────────────────────────
    # TABLE VIEW (OPTIONAL)
    # ─────────────────────────

    st.markdown("### Job Snapshot")

    snapshot_cols = [
        "title", "company", "location", "industry",
        "bhas_score", "risk_tier",
        "listing_age_days", "repost_count", "salary_missing"
    ]

    snapshot_cols = [c for c in snapshot_cols if c in filtered.columns]

    #st.dataframe(filtered[snapshot_cols], use_container_width=True)
    snapshot_df = filtered[snapshot_cols].copy()

    def highlight_risk(val):
        colors = {
            "Ghost": "#ff4b4b33",
            "High Risk": "#ff8c0033",
            "Moderate Risk": "#ffd70033",
            "Low Risk": "#00cc8833",
        }
        return f"background-color: {colors.get(val,'')}"

    styled = snapshot_df.style.applymap(
        highlight_risk,
        subset=["risk_tier"]
    )

    st.dataframe(styled, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: COMPANY ANALYSIS
# ─────────────────────────────────────────────

elif page == "🏢 Company Analysis":
    st.markdown("# 🏢 Company Analysis")
    st.caption("Hiring behaviour patterns aggregated at company level")
    st.divider()

    company_stats = view_df.groupby("company").agg(
        total_jobs      = ("job_id","count"),
        avg_bhas        = ("bhas_score","mean"),
        ghost_jobs      = ("risk_tier", lambda x: (x=="Ghost").sum()),
        avg_listing_age = ("listing_age_days","mean"),
        avg_repost      = ("repost_count","mean"),
        industries      = ("industry", lambda x: x.mode()[0] if len(x)>0 else ""),
    ).reset_index()

    company_stats["ghost_rate"] = (company_stats["ghost_jobs"] /
                                    company_stats["total_jobs"] * 100).round(1)
    company_stats["avg_bhas"]   = company_stats["avg_bhas"].round(1)
    company_stats = company_stats[company_stats["total_jobs"] >= 2].sort_values("ghost_rate", ascending=False)

    # ── Top suspicious companies ──

    st.markdown("## 🚨 High Ghost Risk Companies")
    st.caption("Companies exhibiting strong ghost hiring behavioural signals")
    top_suspicious = company_stats.head(15)

    fig = px.bar(top_suspicious, x="company", y="ghost_rate",
                 color="ghost_rate",
                 color_continuous_scale=["#ffd700","#ff4b4b"],
                 template="plotly_dark",
                 labels={"ghost_rate":"Ghost Rate (%)","company":"Company"})
    fig.update_layout(height=360, coloraxis_showscale=False,
                      plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                      margin=dict(l=10,r=10,t=10,b=10),
                      xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: total jobs vs avg BHAS ──
    st.markdown("### Company Hiring Volume vs Authenticity")
    fig2 = px.scatter(
        company_stats,
        x="total_jobs", y="avg_bhas",
        size="total_jobs", color="ghost_rate",
        color_continuous_scale=["#00cc88","#ffd700","#ff4b4b"],
        hover_name="company",
        hover_data={"avg_repost":":.2f","avg_listing_age":":.0f"},
        template="plotly_dark",
        labels={"total_jobs":"Total Postings","avg_bhas":"Avg BHAS Score","ghost_rate":"Ghost Rate %"}
    )
    fig2.update_layout(height=350,
                       plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                       margin=dict(l=10,r=10,t=10,b=10),legend=dict(font=dict(color="white",size=14)))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Company drilldown ──
    st.divider()
    st.markdown("### Company Drilldown")
    selected_co = st.selectbox("Select company", company_stats["company"].tolist())

    if selected_co:
        co_jobs = view_df[view_df["company"] == selected_co]
        co_stat = company_stats[company_stats["company"] == selected_co].iloc[0]



        cols = st.columns(4)

        metrics_data = [
            ("Total Postings", int(co_stat["total_jobs"])),
            ("Avg BHAS Score", f"{co_stat['avg_bhas']:.1f}"),
            ("Ghost Rate", f"{co_stat['ghost_rate']:.1f}%"),
            ("Avg Listing Age", f"{co_stat['avg_listing_age']:.0f} days")
        ]

        for i, (label, value) in enumerate(metrics_data):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)



        # All jobs from this company
        st.markdown(f"**All {len(co_jobs)} postings from {selected_co}:**")
        co_display = co_jobs[["title","location","bhas_score","risk_tier",
                               "listing_age_days","repost_count","salary_missing"]].copy()
        co_display = co_display.sort_values("bhas_score")
        st.dataframe(co_display, use_container_width=True, height=300)


# ─────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ─────────────────────────────────────────────

elif page == "📈 Model Insights":
    st.markdown("# 📈 Model Insights")
    st.caption("Isolation Forest performance on synthetic labelled data")
    st.divider()

    # ── Metrics ──
    try:
        with open("models/model_metrics.txt") as f:
            metrics_text = f.read()
        with st.expander("📄 Full Model Metrics Report", expanded=True):
            st.code(metrics_text)
    except FileNotFoundError:
        st.warning("Run models/isolation_forest.py first to generate metrics.")
    
    # SHAP explainability (if available)
    import shap
    from sklearn.ensemble import IsolationForest
    import matplotlib.pyplot as plt

    st.divider()
    st.markdown("## 🧠 SHAP Explainability")

    try:
        train = pd.read_csv("data/features_train.csv")
        MODEL_FEATURES = [
            "listing_age_days","listing_longevity_ratio","salary_missing",
            "repost_count","role_recycling_score","hiring_velocity","open_close_ratio"
        ]
        MODEL_FEATURES = [f for f in MODEL_FEATURES if f in train.columns]

        model = IsolationForest(random_state=42)
        model.fit(train[train["is_ghost"]==0][MODEL_FEATURES])

        #explainer = shap.TreeExplainer(model)
        explainer = shap.Explainer(model.predict, train[MODEL_FEATURES])
        sample = train[MODEL_FEATURES].sample(300)
        shap_values = explainer(sample)

        st.markdown("### Feature Impact Summary")
        fig_shap, ax = plt.subplots()
        #shap.summary_plot(shap_values, train[MODEL_FEATURES].sample(300),show=False)
        shap.summary_plot(shap_values.values, sample, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)

    except Exception as e:
        st.warning("SHAP explainability unavailable.")




    # ── Feature importance proxy: mean value ghost vs authentic ──
    st.markdown("### Feature Signal Strength (Ghost vs Authentic)")
    st.caption("Shows how much each feature differs between ghost and authentic jobs in synthetic data")
    MODEL_FEATURES = [
        "listing_age_days","listing_longevity_ratio","salary_missing",
        "repost_count","role_recycling_score","hiring_velocity","open_close_ratio"
    ]
    MODEL_FEATURES = [f for f in MODEL_FEATURES if f in synth.columns]


    syn_ghost = synth[synth["is_ghost"] == 1][MODEL_FEATURES].mean() if "is_ghost" in synth.columns else None
    syn_auth  = synth[synth["is_ghost"] == 0][MODEL_FEATURES].mean() if "is_ghost" in synth.columns else None



    if syn_ghost is not None and syn_auth is not None:
        feat_df = pd.DataFrame({
            "Feature":   MODEL_FEATURES,
            "Ghost Mean":    [synth[synth["is_ghost"]==1][f].mean() for f in MODEL_FEATURES],
            "Authentic Mean":[synth[synth["is_ghost"]==0][f].mean() for f in MODEL_FEATURES],
        })
        feat_df["Separation"] = (feat_df["Ghost Mean"] - feat_df["Authentic Mean"]).abs()
        feat_df = feat_df.sort_values("Separation", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Ghost", y=feat_df["Feature"], x=feat_df["Ghost Mean"],
                             orientation="h", marker_color="#ff4b4b"))
        fig.add_trace(go.Bar(name="Authentic", y=feat_df["Feature"], x=feat_df["Authentic Mean"],
                             orientation="h", marker_color="#00cc88"))
        fig.update_layout(
            barmode="group", height=380, template="plotly_dark",
            plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── BHAS score comparison: ghost vs authentic on synthetic ──
    st.markdown("### BHAS Score Distribution: Ghost vs Authentic (Synthetic)")
    if "is_ghost" in synth.columns and "bhas_score" in synth.columns:
        fig2 = px.histogram(
            synth, x="bhas_score", color="is_ghost",
            nbins=40, barmode="overlay", opacity=0.7,
            color_discrete_map={0:"#00cc88", 1:"#ff4b4b"},
            labels={"bhas_score":"BHAS Score","is_ghost":"Label"},
            template="plotly_dark",
        )
        fig2.update_layout(height=350,
                           plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                           margin=dict(l=10,r=10,t=10,b=10))
        newnames = {"0":"Authentic","1":"Ghost"}
        fig2.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))
        st.plotly_chart(fig2, use_container_width=True)

    # ── How BHAS is calculated ──
    st.divider()
    st.markdown("### How BHAS is Calculated")
    st.markdown("""
    **BHAS = Behavioural Hiring Authenticity Score (0–100)**

    Higher score = more authentic. Lower score = more ghost-like.

    ```
    STEPS:
    Step 1: Train Isolation Forest on authentic job patterns only
    Step 2: Score each job → anomaly score (lower = more anomalous)
    Step 3: Invert and normalise to 0–1 ghost probability
    Step 4: BHAS = (1 - ghost_probability) × 100

    ```

    **Input features:**
    - `listing_age_days` — how long the job has been open
    - `listing_longevity_ratio` — age vs 30-day industry average
    - `salary_missing` — whether salary is disclosed
    - `repost_count` — number of times role was reposted
    - `role_recycling_score` — similarity of role to previous postings
    - `hiring_velocity` — open roles per month at company
    - `open_close_ratio` — open vs closed positions ratio

    **Risk Tiers:**
    - 🔴 Ghost (0–30): Strong ghost behavioural pattern
    - 🟠 High Risk (30–50): Multiple suspicious signals
    - 🟡 Moderate Risk (50–75): Some anomalous behaviour
    - 🟢 Low Risk (75–100): Healthy hiring behaviour
    """)
