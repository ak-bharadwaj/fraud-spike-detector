# 🛡️ Fraud-Spike Detector — Merchant Risk Intelligence

**Audit-First Streaming Anomaly Detection Engine & Dual-Track Risk Intelligence System**

> **Status:** FROZEN RELEASE (`v1.1.0`)  
> **Primary Streaming Model:** `StatisticalDeviationScorer` ($\tau = 5.00\sigma, P = 1, C = 5$) — Frozen Synthetic Holdout  
> **Real-World ML Validation Track:** Primary `XGBoost` Classifier (Platt-Calibrated on 284,807 transactions) — `EXP-REALWORLD-CCF-001`  
> **Defense-Only Scope:** Produces interpretable risk scores, confidence ratings, and alerts for human risk operations teams — never auto-blocks payment transactions.

---

## 📌 Problem Statement & Hackathon Track

Buildathon Track 02 (**AI Risk Manager**) requires building a working detector for financial loss with measured precision and recall on a held-out test set.

Payment gateways process millions of transactions per minute across thousands of distinct merchants. During fraud attacks or merchant compromises, fraud rings rapidly launch card-testing bursts, volume spikes, and velocity floods.

### Why Ordinary Thresholding is Insufficient:
1. **Static Global Limits Fail:** A high-volume merchant naturally processes 500 txs/min, while a small boutique merchant processes 2 txs/min. A static global threshold floods small merchants with false alarms while missing massive relative spikes on high-volume merchants.
2. **Alert Fatigue & Flapping:** Fluctuating scores hovering near decision boundaries produce repetitive, noisy alerts without state machine persistence and cooldown logic.
3. **Black-Box Model Risk:** Deep learning and LLM classifiers lack deterministic auditability, making it impossible for risk analysts to explain why an alert was triggered during regulatory audits.

---

## 💡 Dual-Track Evaluation Architecture

Fraud-Spike Detector implements a **Dual-Track Evaluation Design** to satisfy both deterministic streaming auditability and learned real-world fraud classification:

### Track A — Frozen Synthetic Streaming Track (`EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004`)
- **Engine:** `StatisticalDeviationScorer` ($\tau = 5.00\sigma, P = 1, C = 5$)
- **Purpose:** Validates real-time streaming state machine (NORMAL ➔ CANDIDATE ➔ ALERT ➔ COOLDOWN), minute-aligned windowing, online expected values ($E[X]$), robust scale ($S[X] = \max(\text{MAD}[X], \epsilon)$), 100% SQLite auditability, and failure recovery.
- **Dataset:** 5-event locked synthetic holdout (`data/holdout/`).

### Track B — Real-World Public Benchmark Track (`EXP-REALWORLD-CCF-001`)
- **Engine:** Primary Supervised `XGBoost` Classifier (Platt-Calibrated) with Isolation Forest comparator
- **Purpose:** Validates learned multi-dimensional fraud discrimination, Platt-scaled probability calibration on CALIBRATION split, and feature group ablation on a large, highly imbalanced dataset.
- **Dataset:** ULB / Kaggle Credit Card Fraud Benchmark (284,807 European cardholder transactions, 492 fraud transactions across 48 hours).
- **Isolation Split:** Strict 3-way temporal split: TRAIN (70%) ➔ CALIBRATION (15%) ➔ LOCKED TEST (15%, 42,721 transactions, 52 fraud transactions). Zero data leakage.

---

## 🏗️ System Architecture & Workflow

```
                        DUAL-TRACK EVALUATION PIPELINE
                        
    ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
    │     Track A: Streaming Synthetic        │   │    Track B: Real-World Public Benchmark │
    │    (StatisticalDeviationScorer v1.1)    │   │     (XGBoost Platt-Calibrated Model)    │
    └────────────────────┬────────────────────┘   └────────────────────┬────────────────────┘
                         │                                             │
                         ▼                                             ▼
            TimeOrderedEventBus & Clock                   3-Way Temporal Split
                         │                               (Train 70% | Calib 15% | Test 15%)
                         ▼                                             │
            Feature Engine (1m Windows)                                ▼
                         │                               Base Model Fitting (TRAIN)
                         ▼                                             │
           Baseline Engine & Scale (MAD)                               ▼
                         │                               Platt Calibration (CALIBRATION)
                         ▼                                             │
             Standardized Magnitude M                                  ▼
                         │                               Single Pass Pass-Through
                         ▼                               (LOCKED TEST - 42,721 txs)
            AlertStateMachine (P=1, C=5)                               │
                         │                                             │
                         ▼                                             ▼
            SQLite Audit Database                        Committed Provenance Manifests
```

---

## 📊 Dual-Track Benchmark Results

### Track A: Frozen Synthetic Holdout (`EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004`)

| Benchmark Metric | Canonical Value | Unit |
|---|---|---|
| **True Positives (TP)** | **4** | count |
| **False Positives (FP)** | **1** | count |
| **False Negatives (FN)** | **1** | count |
| **Precision** | **0.8000** (80.0%, 4/5 TP/alerts, 95% CI: `[0.2000, 0.8000]`) | rate |
| **Recall** | **0.8000** (80.0%, 4/5 TP/events, 95% CI: `[0.2000, 0.8000]`) | rate |
| **F1 Score** | **0.8000** (80.0%) | score |
| **Median Latency** | **64.57** | seconds |
| **P95 Latency** | **64.57** | seconds |
| **Total Financial Impact** | **₹850.00** | ₹ (INR) |

---

### Track B: Real-World Public Benchmark (`EXP-REALWORLD-CCF-001`)

Evaluated on 42,721 held-out test transactions (52 fraud transactions) under strict 3-way temporal isolation:

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

## 🔬 Principled Real-World Feature & Model Ablation

Ablation study on the held-out test set demonstrates the individual contributions of dataset features and model components across 6 variants:

| Variant ID | Features | Description | Precision | Recall | F1 Score | $\Delta\text{F1}$ | AUC-PR |
|---|---|---|---|---|---|---|---|
| **`FULL_ENSEMBLE`** | 30 | IF + XGBoost on all 30 features | 0.8605 | 0.7115 | **0.7789** | +0.0000 | 0.7489 |
| **`XGB_ONLY`** | 30 | XGBoost alone on all features (Headline) | 0.8298 | 0.7500 | **0.7879** | +0.0089 | 0.7703 |
| **`IF_ONLY`** | 30 | Isolation Forest alone (unsupervised) | 0.0542 | 0.5000 | **0.0977** | -0.6812 | 0.0429 |
| **`PCA_ONLY`** | 28 | V1–V28 PCA features only (no Time/Amount)| 0.9250 | 0.7115 | **0.8043** | +0.0254 | 0.7519 |
| **`PCA_PLUS_AMOUNT`** | 29 | V1–V28 PCA features + Amount | 0.8667 | 0.7500 | **0.8041** | +0.0252 | 0.7427 |
| **`AMOUNT_TIME_ONLY`** | 2 | Time & Amount features only | 0.0029 | 0.0385 | **0.0053** | -0.7736 | 0.0022 |

> **Key Discovery:** Performance collapses when PCA features are removed ($F_1 = 0.0053$), indicating that $V_1$–$V_{28}$ contain the dominant predictive fraud information in this benchmark.

---

## 🔬 Scientific Honesty & Boundaries

* **Track Separation:** Track A (synthetic streaming) and Track B (real-world ML) answer different technical questions and are presented separately.
* **PCA Dimension Boundary:** The ULB credit card dataset contains anonymized PCA features (`V1`–`V28`). We evaluate raw feature matrices honestly rather than inventing artificial merchant/device identities.
* **Strict Temporal Splitting:** Calibration (Platt scaling) is fitted strictly on the 15% CALIBRATION split, preventing test-set data leakage.
* **Defense-Only Scope:** System produces risk signals and alerts for human risk operations review — never auto-blocks financial transactions.

---

## 🚀 How to Run & Demo Instructions

### 1. Launch Interactive Operations Console:
```bash
python scripts/run_ui.py
```
Open `http://localhost:8000` to inspect live minute-aligned windows, robust baselines, SQLite audit logs, and risk state transitions.

### 2. Execute Real-World ML Benchmark Training & Manifest Generation:
```bash
python scripts/train_realworld.py
```
Executes the 3-way temporal split pipeline, fits models, runs 5-way ablation, and outputs manifests to `artifacts/realworld/`.

### 3. Execute Complete Pytest Test Suite:
```bash
python -m pytest tests/ -v
```
Runs all 311 unit, integration, and real-world pipeline tests.

---

## 📁 Repository Structure

```
fraud-spike-detector/
├── config/                  # Locked configuration & freeze records
│   ├── detector.yaml
│   ├── generator.yaml
│   └── freeze_record.json
├── src/                     # Source application packages
│   ├── contracts/           # Pydantic data schemas
│   ├── generator/           # Synthetic stream generator
│   ├── stream/              # VirtualClock & TimeOrderedEventBus
│   ├── features/            # Feature aggregation engine
│   ├── baseline/            # Baseline engine & Evidence state
│   ├── scoring/             # StatisticalDeviationScorer & ML Scorers (IF, XGB, Ensemble)
│   ├── realworld/           # Kaggle Credit Card dataset adapter (3-way temporal split)
│   ├── state/               # Alert state machine
│   ├── detector/            # StreamingDetectorPipeline
│   ├── audit/               # SQLite audit persistence
│   └── web/                 # Web server & interactive single-page UI
├── tests/                   # Complete Pytest test suite (311 tests)
├── scripts/                 # Execution scripts
│   ├── run_ui.py            # Standalone web UI launcher
│   └── train_realworld.py   # E2E real-world ML benchmark training pipeline
├── data/                    # Development & locked holdout streams
├── artifacts/               # Committed canonical research artifacts
│   ├── final/               # Track A: Frozen synthetic holdout artifacts
│   └── realworld/           # Track B: Real-world benchmark report & manifests
├── docs/                    # Technical documentation
│   ├── PITCH.md             # 5-minute technical pitch script (§41)
│   ├── FAILURE_STORY.md     # Documented historical failure & fix story (§40)
│   ├── EVALUATION.md        # Complete evaluation methodology & results
│   └── LIMITATIONS.md       # Scientific honesty & known boundaries
└── README.md
```

---

## 📜 License

MIT License.
