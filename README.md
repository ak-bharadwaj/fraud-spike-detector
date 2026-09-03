# 🛡️ Fraud-Spike Detector — Merchant Risk Intelligence

**Audit-First Streaming Anomaly Detection Engine & Risk Operations Console for Merchant Financial Streams**

> **Status:** FROZEN RELEASE (`v1.1.0`)  
> **Primary Model:** `StatisticalDeviationScorer` ($\tau = 5.00\sigma, P = 1, C = 5$)  
> **Defense-Only Scope:** Produces interpretable risk scores, confidence ratings, and alerts for human risk operations teams — never auto-blocks payment transactions.

---

## 📌 Problem Statement

Payment gateways process millions of transactions per minute across thousands of distinct merchants. During fraud attacks or merchant compromises, fraud rings rapidly launch card-testing bursts, volume spikes, and velocity floods.

### Why Ordinary Thresholding is Insufficient:
1. **Static Global Limits Fail:** A high-volume merchant naturally processes 500 txs/min, while a small boutique merchant processes 2 txs/min. A static global threshold of 100 txs/min floods small merchants with false alarms while missing massive relative spikes on high-volume merchants.
2. **Alert Fatigue & Flapping:** Fluctuating scores hovering near decision boundaries produce repetitive, noisy alerts without state machine persistence and cooldown logic.
3. **Black-Box Model Risk:** Deep learning and LLM classifiers lack deterministic auditability, making it impossible for risk analysts to explain why an alert was triggered during regulatory audits.

---

## 💡 The Solution

Fraud-Spike Detector is an **explainable, statistical streaming anomaly detector** that learns per-merchant robust baselines in real time, evaluates multi-signal deviation magnitudes ($M$), enforces evidence sufficiency rules, and manages state transitions via an alert state machine.

### Key Highlights:
* **Per-Merchant Baseline Engine:** Tracks online running expected values $E[X]$ and robust scale $S[X] = \max(\text{MAD}[X], \epsilon)$ over minute-aligned sliding windows.
* **Statistical Deviation Scoring:** Computes standardized magnitude scores $M = \frac{|X - E[X]|}{S[X]}$ across volume, velocity, amount, and behavioral signals.
* **Composite Confidence Rating:** Integrates evidence state (`SUFFICIENT`, `DEGRADED`, `INSUFFICIENT`), feature availability, and signal agreement.
* **State Machine Persistence & Cooldown:** Enforces persistence ($P=1$) to confirm alerts and cooldown ($C=5$ consecutive normal windows) to eliminate alert fatigue.
* **100% SQLite Auditability:** Every feature snapshot, baseline expected value, deviation score, confidence level, and state transition is logged to SQLite.
* **Interactive Web Operations Console:** Live visualization console for judges and risk ops teams (`python scripts/run_ui.py`).

---

## 🏗️ System Architecture & Workflow

```
Synthetic Stream Generator (Transactions)
   │
   ▼
TimeOrderedEventBus + VirtualClock (Monotonic Time Authority)
   │
   ▼
StreamingDetectorPipeline
   │
   ├── 1. Feature Engine (Minute-aligned sliding windows [HH:MM:00, HH:MM:00 + 1m))
   │
   ├── 2. Baseline Engine + Evidence State (Online expected values E[X] & MAD scale S[X])
   │
   ├── 3. StatisticalDeviationScorer v1.1.0 (Standardized magnitude M = |X - E[X]| / S[X])
   │
   ├── 4. Composite Confidence Engine (SUFFICIENT=1.0, DEGRADED=0.5, INSUFFICIENT=0.0)
   │
   ├── 5. AlertStateMachine (NORMAL ➔ CANDIDATE ➔ ALERT ➔ COOLDOWN ➔ NORMAL)
   │
   └── 6. SQLite Audit Database (Relational persistence for audit records & alerts)
```

---

## 📊 Measured Locked Holdout Benchmark Results

Evaluated on the locked holdout dataset (`data/holdout/`) under canonical experiment `EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004` (Execution Commit SHA `2f860e8`):

| Benchmark Metric | Canonical Value | Unit |
|---|---|---|
| **True Positives (TP)** | **4** | count |
| **False Positives (FP)** | **1** | count |
| **False Negatives (FN)** | **1** | count |
| **Precision** | **0.8000** (80.0%, 4/5 TP/alerts, 95% CI: `[0.2000, 0.8000]`) | rate |
| **Recall** | **0.8000** (80.0%, 4/5 TP/events, 95% CI: `[0.2000, 0.8000]`) | rate |
| **F1 Score** | **0.8000** (80.0%) | score |
| **Median Detection Latency** | **64.57** | seconds |
| **P95 Detection Latency** | **64.57** | seconds |
| **False Positive Operational Cost** | **₹50.00** | ₹ (INR) |
| **False Negative Fraud Exposure** | **₹800.00** | ₹ (INR) |
| **Total Portfolio Financial Impact** | **₹850.00** | ₹ (INR) |

---

## 🔬 Research & Evaluation Highlights

1. **5-Way Signal Ablation Comparison:**
   Evaluates feature group contributions (`FULL`, `-VOLUME`, `-VELOCITY`, `-AMOUNT`, `-BEHAVIORAL`) via scorer-level signal masking without baseline history starvation.
2. **Detector-Aware Evasion Confirmation:**
   Confirms execution on physical holdout evasion scenarios: threshold-hugging, persistence evasion, staircase ramp, and oscillating sub-threshold.
3. **Descriptive Holdout Calibration:**
   Generates Reliability Diagrams and Expected Calibration Error ($\text{ECE} = 0.7094$, total samples $= 119$).
4. **Non-Parametric Bootstrap Uncertainty:**
   Derives 95% confidence intervals over $N = 1000$ resamples: Precision point `0.8000`, 95% CI `[0.2000, 0.8000]` (raw 4 TP / 5 alerts); Recall point `0.8000`, 95% CI `[0.2000, 0.8000]` (raw 4 TP / 5 events).

---

## 🔬 Scientific Honesty & Known Limitations

* **Synthetic Data Boundary:** Results reflect quantitative performance on reproducible synthetic streams. Synthetic streams do not capture all multi-modal complexities of real-world production fraud.
* **Unrepresented Anomaly Classes:** Certain synthetic generator anomaly classes (e.g. geo attribute shift) are not present in the 5-event locked holdout dataset and are designated as `NO_EVENTS_IN_DATASET`.
* **Descriptive Calibration:** Calibration diagrams characterize the frozen detector without feeding back into hyperparameter retuning.
* **Defense-Only Scope:** Produces risk signals and alerts for human risk operations review. Never auto-blocks or freezes financial transactions.

---

## 🚀 How to Run & Demo Instructions

### 1. Launch the Web Operations Console (Interactive UI & Demo Mode):
```bash
python scripts/run_ui.py
```
Open `http://localhost:8000` in your browser. Click **▶ Start Live Demo** or **⏭ Step Window** to observe the live processing flow!

### 2. Execute the Full Pytest Benchmark Suite:
```bash
python -m pytest tests/ -v
```
All 273+ unit, integration, and architectural boundary tests execute deterministically and pass with zero failures.

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
│   ├── scoring/             # StatisticalDeviationScorer strategy
│   ├── state/               # Alert state machine
│   ├── detector/            # StreamingDetectorPipeline
│   ├── audit/               # SQLite audit persistence
│   ├── evaluation/          # Benchmark metrics, ablation, evasion, drift, calibration
│   └── web/                 # Web server & interactive single-page UI
├── tests/                   # Complete Pytest test suite (273+ tests)
├── scripts/                 # Execution & UI launcher scripts
│   └── run_ui.py            # Standalone web UI launcher
├── data/                    # Development & locked holdout streams
│   ├── development/
│   └── holdout/
├── artifacts/               # Committed canonical research artifacts
├── docs/                    # Detailed technical documentation
│   ├── PITCH.md             # 5-minute technical pitch script (§41)
│   ├── FAILURE_STORY.md     # Documented historical failure & fix story (§40)
│   ├── DEMO.md              # 2-3 minute demonstration guide
│   ├── ARCHITECTURE.md      # System architecture specification
│   ├── EVALUATION.md        # Evaluation methodology & holdout results
│   └── LIMITATIONS.md       # Scientific honesty & known boundaries
└── README.md
```

---

## 📜 License

MIT License.
