# 📊 Fraud-Spike Detector — Evaluation Methodology & Dual-Track Benchmark Results

This document details the complete evaluation methodology, locked synthetic holdout results (`EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004`), and the additive real-world public benchmark validation (`EXP-REALWORLD-CCF-001`).

---

## 📌 Dual-Track Evaluation Design

To provide both deterministic streaming auditability and learned real-world fraud classification, the system is evaluated across two independent, non-overlapping tracks:

1. **Track A — Frozen Synthetic Streaming Track (`EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004`):**
   - **Model:** `StatisticalDeviationScorer` ($\tau = 5.00\sigma, P = 1, C = 5$)
   - **Purpose:** Validates real-time streaming state machine (NORMAL ➔ CANDIDATE ➔ ALERT ➔ COOLDOWN), minute-aligned sequential windowing, online expected values ($E[X]$), robust MAD scale ($S[X]$), 100% SQLite auditability, and failure recovery.
   - **Dataset:** 5-event locked synthetic holdout stream (`data/holdout/`).

2. **Track B — Real-World Public Benchmark Track (`EXP-REALWORLD-CCF-001`):**
   - **Model:** Primary Supervised `XGBoost` Classifier (Platt-Calibrated) with Isolation Forest comparator
   - **Purpose:** Validates learned multi-dimensional fraud discrimination, Platt-scaled probability calibration on CALIBRATION split, and feature group ablation on a large, highly imbalanced dataset.
   - **Dataset:** ULB / Kaggle Credit Card Fraud Benchmark (284,807 European cardholder transactions, 492 fraud transactions across 48 hours).
   - **Isolation Split:** Strict 3-way temporal split: TRAIN (70%) ➔ CALIBRATION (15%) ➔ LOCKED TEST (15%, 42,721 transactions, 52 fraud transactions).

---

## 🎯 Track A: Matching Rules & Latency Horizons

Event matching strictly follows Sections 21–22 of the Master Build Plan and `config/evaluation.yaml`:
* **Valid Event Match:** An emitted `Alert` at time $t_a$ matches Ground Truth event $E_i$ starting at $t_{\text{start}}$ iff:
  $$t_{\text{start}} \le t_a \le t_{\text{start}} + H_{\text{type}}$$
  where $H_{\text{type}}$ is the configured anomaly-specific evaluation horizon from `config/evaluation.yaml`:
  * `velocity_burst`: **60s**
  * `volume_spike`: **120s**
  * `amount_shift`: **180s**
  * `behavioral_anomaly`: **180s**
  * `attribute_shift`: **180s**
  * `sustained_spike`: **300s**
  * `compound_anomaly`: **300s**
  * `evasive_patterns`: **300s**
* **Detection Latency:** $\text{Latency} = \max(0\text{s}, t_a - t_{\text{start}})$.
* **Strict One-to-One Matching:** An emitted alert matches at most one ground truth event. The first valid alert within $[t_{\text{start}}, t_{\text{start}} + H_{\text{type}}]$ matches the event ($\text{TP}=1$). Any additional alert not consumed by a one-to-one match is an unmatched alert and scores False Positive ($\text{FP}=1$).

---

## 💰 Portfolio Cost Model

Financial impact is evaluated using the Master Plan Section 34 cost parameters:
* **False Positive Cost ($C_{\text{FP}}$):** ₹50.00 per false alarm (manual risk analyst review operational cost).
* **False Negative Exposure ($C_{\text{FN}}$):** Dynamic financial exposure calculated as $\sum \text{Amount}$ of missed fraud transactions (for Track B) or unmitigated event impact (for Track A).
* **Total Portfolio Cost:**
  $$\text{Total Cost} = (\text{FP} \times ₹50.00) + \text{FN Exposure}$$

---

## 📈 Track A: Measured Locked Holdout Results (`EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004`)

| Metric | Measured Value | Unit |
|---|---|---|
| **True Positives (TP)** | **4** | count |
| **False Positives (FP)** | **1** | count |
| **False Negatives (FN)** | **1** | count |
| **Precision** | **0.8000** (80.0%, 4/5 TP/alerts, 95% CI: `[0.2000, 0.8000]`) | rate |
| **Recall** | **0.8000** (80.0%, 4/5 TP/events, 95% CI: `[0.2000, 0.8000]`) | rate |
| **F1 Score** | **0.8000** | score |
| **Median Latency** | **64.57** | seconds |
| **P95 Latency** | **64.57** | seconds |
| **False Positive Cost** | **₹50.00** | ₹ (INR) |
| **False Negative Exposure** | **₹800.00** | ₹ (INR) |
| **Total Portfolio Cost** | **₹850.00** | ₹ (INR) |

---

## 🛡️ Track A: Detector-Aware Evasion Confirmation

Tests representative evasive attack trajectories physically embedded in `data/holdout/`:

1. **Threshold-Hugging Evasion (`EVT-HOLDOUT-002`):**
   * *Mechanism:* Configured target magnitude was $4.80\sigma$; observed score trajectory crossed decision boundary, peaking at $7.0\sigma$.
   * *Outcome:* Breached static threshold $\tau = 5.00\sigma$ peak -> Alert emitted ($\text{TP}=1$).
2. **Persistence Evasion (`EVT-HOLDOUT-003`):**
   * *Mechanism:* Single-window burst ($M = 5.60\sigma$) followed by sub-threshold drop.
   * *Outcome:* Breached threshold $\tau = 5.00\sigma$ with $P=1$ persistence -> Alert emitted ($\text{TP}=1$).
3. **Staircase Ramp (`EVT-HOLDOUT-004`):**
   * *Mechanism:* Stepwise volume progression ramping to $M = 6.50\sigma$.
   * *Outcome:* Consecutive steps breach threshold -> Alert emitted ($\text{TP}=1$).
4. **Oscillating Sub-Threshold (`EVT-HOLDOUT-005`):**
   * *Mechanism:* Sub-threshold harmonic oscillation staying strictly below decision threshold ($M = 1.64\sigma < 5.00\sigma$).
   * *Outcome:* Max score stays below threshold -> State machine remains `NORMAL` ($\text{FN}=1$).

---

## 🌐 Track B: Real-World Public Benchmark Results (`EXP-REALWORLD-CCF-001`)

Evaluated on 42,721 held-out test transactions (52 fraud transactions) under strict 3-way temporal isolation:

### Primary Model Benchmark Summary (XGBoost, Platt-Calibrated)

| Benchmark Metric | Primary XGBoost Value | 95% Bootstrap CI (N=1000) |
|---|---|---|
| **Held-Out Test Set Size** | **42,721 transactions** | N/A |
| **Held-Out Test Fraud Transactions** | **52 fraud transactions** (0.1217% rate) | N/A |
| **True Positives (TP)** | **39** | count |
| **False Positives (FP)** | **8** | count |
| **False Negatives (FN)** | **13** | count |
| **Precision** | **0.8298** (83.0%) | `[0.7142, 0.9298]` |
| **Recall** | **0.7500** (75.0%) | `[0.6274, 0.8667]` |
| **F1 Score** | **0.7879** (78.8%) | `[0.6857, 0.8687]` |
| **AUC-ROC** | **0.9825** | `[0.9692, 0.9931]` |
| **AUC-PR** | **0.7703** | N/A |
| **Calibration Error (ECE)** | **0.0001** (0.01% ECE) | Platt-scaled on CALIB split |
| **False Positive Cost** | **₹400.00** (8 × ₹50) | ₹ (INR) |
| **False Negative Exposure** | **₹2,372.40** ($\sum \text{Amount}$ of 13 FNs) | ₹ (INR) |
| **Total Portfolio Impact** | **₹2,772.40** | ₹ (INR) |

---

### Track B: Principled Feature & Model Group Ablation (6 Variants)

Evaluated on the 42,721 held-out test transactions:

| Variant ID | Features | Description | Precision | Recall | F1 Score | $\Delta\text{F1}$ | AUC-PR |
|---|---|---|---|---|---|---|---|
| **`FULL_ENSEMBLE`** | 30 | IF + XGBoost on all 30 features | 0.8605 | 0.7115 | **0.7789** | +0.0000 | 0.7489 |
| **`XGB_ONLY`** | 30 | XGBoost alone on all features (Headline) | 0.8298 | 0.7500 | **0.7879** | +0.0089 | 0.7703 |
| **`IF_ONLY`** | 30 | Isolation Forest alone (unsupervised) | 0.0542 | 0.5000 | **0.0977** | -0.6812 | 0.0429 |
| **`PCA_ONLY`** | 28 | V1–V28 PCA features only (no Time/Amount)| 0.9250 | 0.7115 | **0.8043** | +0.0254 | 0.7519 |
| **`PCA_PLUS_AMOUNT`** | 29 | V1–V28 PCA features + Amount | 0.8667 | 0.7500 | **0.8041** | +0.0252 | 0.7427 |
| **`AMOUNT_TIME_ONLY`** | 2 | Time & Amount features only | 0.0029 | 0.0385 | **0.0053** | -0.7736 | 0.0022 |

> **Scientific Insight:** Performance collapses when PCA features are removed ($F_1 = 0.0053$), indicating that $V_1$–$V_{28}$ contain the dominant predictive information in this benchmark. Unsupervised Isolation Forest alone achieves $F_1 = 0.0977$, demonstrating why supervised XGBoost with Platt-scaling calibration fitted strictly on CALIBRATION is the superior real-world detector.
