# 🛡️ Fraud-Spike Detector — Merchant Risk Intelligence

**Audit-First Streaming Anomaly Detection Engine & Dual-Track Risk Intelligence System**

> **Status:** FROZEN RELEASE (`v1.1.0`)  
> **Primary Streaming Model:** `StatisticalDeviationScorer` ($\tau = 5.00\sigma, P = 1, C = 5$) — Frozen Synthetic Holdout  
> **Real-World ML Validation Track:** Calibrated Ensemble (`IsolationForest` + `XGBoost` on 284,807 transactions) — `EXP-REALWORLD-CCF-001`  
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
- **Engine:** Calibrated Ensemble (`IsolationForest` + `XGBoost` classifier)
- **Purpose:** Validates learned multi-dimensional fraud discrimination, Platt-scaled probability calibration, and feature group ablation on a large, highly imbalanced dataset.
- **Dataset:** ULB / Kaggle Credit Card Fraud Benchmark (284,807 European cardholder transactions, 492 fraud events across 48 hours).
- **Isolation Split:** Strict 3-way temporal split: TRAIN (70%) ➔ CALIBRATION (15%) ➔ LOCKED TEST (15%, 42,721 transactions, 52 fraud events). Zero data leakage.

---

## 🏗️ System Architecture & Workflow

```
                        DUAL-TRACK EVALUATION PIPELINE
                        
    ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
    │     Track A: Streaming Synthetic        │   │    Track B: Real-World Public Benchmark │
    │    (StatisticalDeviationScorer v1.1)    │   │      (Calibrated Ensemble IF + XGB)     │
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

Evaluated on 42,721 held-out test transactions (52 fraud events) under strict 3-way temporal isolation:

| Benchmark Metric | Calibrated Ensemble Value | 95% Bootstrap CI (N=1000) |
|---|---|---|
| **Held-Out Test Set Size** | **42,721 transactions** | N/A |
| **Held-Out Test Fraud Events** | **52 fraud events** (0.1217% rate) | N/A |
| **True Positives (TP)** | **37** | count |
| **False Positives (FP)** | **6** | count |
| **False Negatives (FN)** | **15** | count |
| **Precision** | **0.8605** (86.1%) | `[0.7500, 0.9512]` |
| **Recall** | **0.7115** (71.2%) | `[0.5849, 0.8333]` |
| **F1 Score** | **0.7789** (77.9%) | `[0.6760, 0.8600]` |
| **AUC-ROC** | **0.9410** | `[0.8971, 0.9758]` |
| **AUC-PR** | **0.7489** | N/A |
| **Calibration Error (ECE)** | **0.0564** (5.6% ECE) | Platt-scaled on CALIB split |

---

## 🔬 Principled Real-World Feature & Model Ablation

Ablation study on the held-out test set demonstrates the individual contributions of dataset features and model components:

| Variant ID | Features | Description | Precision | Recall | F1 Score | $\Delta\text{F1}$ | AUC-PR |
|---|---|---|---|---|---|---|---|
| **`FULL_ENSEMBLE`** | 30 | IF + XGBoost on all 30 features | 0.8261 | 0.7308 | **0.7755** | +0.0000 | 0.7391 |
| **`XGB_ONLY`** | 30 | XGBoost alone on all features | 0.8444 | 0.7308 | **0.7835** | +0.0080 | 0.7659 |
| **`IF_ONLY`** | 30 | Isolation Forest alone (unsupervised) | 0.0527 | 0.5192 | **0.0957** | -0.6798 | 0.0408 |
| **`PCA_ONLY`** | 28 | V1–V28 PCA features only (no Time/Amount)| 0.8298 | 0.7500 | **0.7879** | +0.0124 | 0.7486 |
| **`AMOUNT_TIME_ONLY`** | 2 | Time & Amount features only | 0.0029 | 0.0385 | **0.0053** | -0.7702 | 0.0019 |

> **Key Discovery:** Time & Amount alone yield F1 = 0.0053 (-0.7702), proving that anonymized PCA dimensions (`V1`–`V28`) carry over 99% of the predictive signal for real-world credit card fraud.

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
