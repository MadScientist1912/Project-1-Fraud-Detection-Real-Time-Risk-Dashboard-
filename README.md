# Project-1-Fraud-Detection-Real-Time-Risk-Dashboard-
Goal: Detect fraudulent financial transactions using ML and visualize insights in a real-time dashboard. Companies: Barclays, PayPal, BlackRock, Capital One
# Fraud Detection & Real‑Time Risk Dashboard

**Operating point (cost‑optimal)**: threshold **t = 0.0145** (minimizes `Cost = 200×FN + 3×FP`).  
**Metrics at operating point** (20k sample, fraud rate ≈ 0.17%):
- **AUC:** 0.9751
- **Precision:** 0.805
- **Recall:** 0.971
- **F1:** 0.880
- **Expected Cost:** 224

---

## 1) Overview
This project builds an **end‑to‑end fraud risk monitoring system**:
- Data ingestion (Kaggle Credit Card Fraud)
- Modeling (Logistic Regression baseline, **XGBoost**)
- **Business‑cost–aware threshold tuning** (false negatives are far more expensive than false positives)
- **Streamlit dashboard** with KPIs, **ROC/PR curves**, confusion heatmap, **cost curve**, and **alert simulation**
- Exported artifacts for deployment (`xgb_pipeline.joblib`, `metrics.json`, `transactions_sample.csv`)

Why it matters: we **optimize business loss**, not just AUC. The decision threshold is chosen by minimizing an explicit cost function.

---

## 2) Data
Kaggle **Credit Card Fraud Detection** dataset (`creditcard.csv`), with heavy class imbalance:
- Columns: `Time`, `V1..V28` (PCA components), `Amount`, `Class` (1 = fraud)
- Preprocessing: `Class → is_fraud`, `Amount → amount`, drop `Time` for the static model

> Note: This repo includes **no raw data**; download it from Kaggle and upload via the notebook or the dashboard.

---

## 3) Methodology
1. **EDA & Imbalance** — confirm fraud rate ≈ 0.17%; evaluate with **precision–recall** (not accuracy).
2. **Models**
   - Logistic Regression (class_weight='balanced')
   - XGBoost (with `scale_pos_weight` for imbalance)
3. **Threshold Selection**
   - Minimize `Cost(t) = 200×FN(t) + 3×FP(t)` on a hold‑out set.
   - **Chosen operating threshold:** `t = 0.0145` (cost‑optimal).
4. **Simulation**
   - Toy “streaming” scoring over a sample to report alerts, missed frauds, and latency proxy.

---

## 4) Results (at t = 0.0145)
- **AUC:** 0.9751  
- **Precision:** 0.805  
- **Recall:** 0.971  
- **F1:** 0.880  
- **Expected Cost:** 224  

> Trade‑offs: Raising `t` increases precision (fewer false alarms) but reduces recall (more missed frauds) and may **increase** expected cost. Lowering `t` does the opposite.

---

## 5) Dataset Link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

## 6) Getting Started

### A) Reproduce in Google Colab (recommended)
1. Open `notebook/Fraud_Detection_CreditCard_Colab.ipynb` in Colab.
2. Upload `creditcard.csv.zip` (from Kaggle) when prompted.
3. Run all cells. Artifacts will be saved under `artifacts/`.

### B) Run the Streamlit Dashboard Locally
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run dashboard/app.py
```
Then open the local URL shown by Streamlit (or expose via ngrok if needed).

**Dashboard features**
- Threshold & cost controls (with cost‑optimal default `t = 0.0145`)
- KPIs: Precision, Recall, F1, Cost, Alert Rate, Avg Alert Score
- **ROC** and **Precision–Recall** curves
- Confusion matrix heatmap
- **Cost vs Threshold** curve
- **Alert simulation** with recent‑alerts feed and CSV download

---

## 7) How Threshold Is Chosen (Policy)
1. Compute `Cost(t) = 200×FN(t) + 3×FP(t)` over a hold‑out set.
2. Pick `t` that **minimizes** expected cost (here: **0.0145**).
3. Apply operational guardrails if needed:
   - Keep **Precision ≥ 0.60**
   - Keep **Recall ≥ 0.90**
   - Respect alert‑rate capacity (raise `t` if analysts are overloaded)
4. **Recalibrate weekly** (or on drift): recompute cost curve on fresh data.

---

## 8) Screenshots
Please see the Fraud Risk Dashboard PDF

---

## 9) Tech Stack
- Python, pandas, numpy, scikit‑learn, xgboost, matplotlib
- Streamlit (UI), pyngrok (optional for Colab)
- Joblib (artifact persistence)

---

## 10) License
For educational/portfolio purposes. Data usage subject to Kaggle terms.
