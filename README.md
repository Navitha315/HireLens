
# 🔍 HireLens

### Behavioural Hiring Authenticity Intelligence Platform

HireLens is an ML-powered Streamlit dashboard that detects **ghost job postings** using behavioural recruitment signals and anomaly detection.
Built as a behavioural analytics solution to quantify hiring authenticity and reduce ghost job externalities in digital labour markets.

It quantifies hiring authenticity using a proprietary score:

> **BHAS — Behavioural Hiring Authenticity Score (0–100)**

---

## 🚀 Live Demo

🌐 Deployed on Render
https://hirelens-cfgv.onrender.com

---

## 🎯 Problem Statement

Ghost job postings waste:

* Candidate time
* Recruiter bandwidth
* Platform credibility
* Economic productivity

HireLens detects suspicious hiring behaviour patterns using lifecycle signals instead of relying on text classification.

---

## 🧠 Core Concept

Instead of NLP spam detection, HireLens models:

* Listing longevity anomalies
* Repost behaviour
* Salary disclosure patterns
* Hiring velocity signals
* Open vs closed role imbalance
* Role recycling similarity

It trains an **Isolation Forest** on authentic hiring behaviour and detects anomalous job dynamics.

---

## 🏗️ System Architecture

```
Adzuna API (Live Data)
        ↓
Feature Engineering Pipeline
        ↓
MinMax Scaling (fit on synthetic authentic)
        ↓
Isolation Forest (trained on authentic jobs)
        ↓
Ghost Probability
        ↓
BHAS Score (0–100)
        ↓
Interactive Streamlit Dashboard
```

---

## 📊 Key Features

### 📈 Overview Dashboard

* Risk tier distribution
* BHAS histogram
* Industry ghost risk comparison
* Candidate Effort Waste Estimator

---

### 🔍 Job Explorer

* Individual job inspection
* BHAS gauge visualization
* Behavioural signal radar chart
* Risk tier color indication
* Job snapshot table

---

### 🏢 Company Analysis

* Company-level ghost rate
* Hiring volume vs authenticity scatter
* Suspicious company ranking
* Drilldown per company

---

### 📉 Model Insights

* Isolation Forest performance metrics
* ROC-AUC evaluation on synthetic labelled data
* Ghost vs Authentic feature separation
* SHAP explainability (Tree-based surrogate)
* BHAS computation breakdown

---

## 🧮 BHAS Calculation

```
Step 1: Train Isolation Forest on authentic job patterns only
Step 2: Score each job → anomaly score (lower = more anomalous)
Step 3: Convert anomaly score → ghost probability (0–1)
Step 4: BHAS = (1 - ghost_probability) × 100
```

### Risk Tiers

| BHAS Range | Risk Tier        |
| ---------- | ---------------- |
| 0–30       | 🔴 Ghost         |
| 30–50      | 🟠 High Risk     |
| 50–75      | 🟡 Moderate Risk |
| 75–100     | 🟢 Low Risk      |

---

## 📂 Project Structure

```
HireLens/
│
├── dashboard/
│     └── app.py
│
├── scraping/
│     └── adzuna_collector.py
│
├── features/
│     └── feature_engineering.py
│
├── models/
│     └── isolation_forest.py
│
├── data/
│     ├── adzuna_jobs.csv
│     ├── features_train.csv
│     ├── features_score.csv
│     ├── features_combined.csv
│     └── scored_jobs.csv
│
├── Procfile
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🛠️ Tech Stack

* **Python 3.10**
* Streamlit
* Pandas / NumPy
* Scikit-learn
* Plotly
* SHAP
* Adzuna Jobs API
* Render (Deployment)

---

## ⚙️ Local Setup

### 1️⃣ Clone repository

```
git clone https://github.com/yourusername/HireLens.git
cd HireLens
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Create `.env` file

```
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
```

---

### 4️⃣ Run pipeline manually

```
python scraping/adzuna_collector.py
python features/feature_engineering.py
python models/isolation_forest.py
```

---

### 5️⃣ Launch dashboard

```
streamlit run dashboard/app.py
```

---

## 🌐 Deployment (Render)

1. Push repository to GitHub
2. Create Web Service on Render
3. Build Command:

```
pip install -r requirements.txt
```

4. Start Command:

```
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

5. Add environment variables in Render:

```
ADZUNA_APP_ID
ADZUNA_APP_KEY
```

Refresh button works in production.

---

## 📈 Model Performance (Synthetic Validation)

* ROC-AUC: 1.00
* Accuracy: 79%
* Recall (Ghost): 1.00
* Precision (Ghost): 0.68

Designed for high ghost detection recall.

---

## ⚠️ Limitations

* Synthetic training data assumptions
* Adzuna API coverage constraints
* Behavioural signals may vary by geography
* Unsupervised anomaly detection limitations

---

## 🔮 Future Improvements

* Add NLP semantic mismatch detection
* Temporal anomaly modeling (rolling windows)
* Recruiter reputation scoring
* Multi-platform aggregation
* API version for SaaS use

---

## 👩‍💻 Author

Navitha315
Machine Learning & Systems Engineering

---

## 📜 License

MIT License

---
