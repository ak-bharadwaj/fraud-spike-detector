# 🎬 Fraud-Spike Detector — Judge Demonstration Script (2–3 Minutes)

This document provides a structured, step-by-step demonstration walkthrough for evaluating the Fraud-Spike Detector risk operations console and reproducible synthetic benchmark suite.

---

## 🚀 Quick Launch Instructions

1. **Start the Web Operations Console:**
   ```bash
   python scripts/run_ui.py
   ```
2. **Access URL in Browser:**  
   Open `http://localhost:8000` (opens automatically).

---

## ⏱️ Step-by-Step Demo Flow (2–3 Minutes)

### Step 1: System Overview & Defense-Only Boundary (30 Seconds)
* **Goal:** Establish system purpose, frozen detector identity, and defense-only contract.
* **Actions:**
  1. Point to the top header badges: `v1.1.0`, `FROZEN RELEASE`, and `DEFENSE-ONLY`.
  2. Explain:
     > *"Fraud-Spike Detector is an audit-first merchant risk intelligence system. It monitors streaming transaction traffic in near-real-time to detect anomalous volume spikes, velocity bursts, amount shifts, and evasive patterns. It operates strictly under a defense-only boundary—producing interpretable risk scores, confidence estimates, and alerts for human risk operations teams. It never automatically blocks or rejects financial payment flows."*

---

### Step 2: Live Detection Pipeline Walkthrough (60 Seconds)
* **Goal:** Demonstrate end-to-end processing pipeline, statistical scoring, confidence, and alert state transitions.
* **Actions:**
  1. Navigate to **Tab 1: ⚡ Live Detection Console**.
  2. Click **▶ Start Live Demo** (or use **⏭ Step Window**).
  3. Observe the **Detection Workflow Pipeline** visualization at the top highlighting each active processing stage:
     `Transactions` ➔ `Feature Engine` ➔ `Baseline + Evidence` ➔ `Statistical Scorer` ➔ `Confidence` ➔ `State Machine` ➔ `Alert / Audit Trail`.
  4. **Watch Window 0–4 (Normal Baseline):**
     * Point out merchant `M1` operating normally (~15 txs/min).
     * Show Risk Score ($M < 5.00\sigma$), Evidence State (`SUFFICIENT`), and State (`NORMAL`).
  5. **Watch Window 5 (Sudden Volume Spike Anomaly):**
     * Highlight transaction volume spiking to ~75 txs/min.
     * Point out Standardized Deviation Score jumping past the static threshold ($M = 6.50\sigma \ge 5.00\sigma$).
     * Show State Machine transition: `NORMAL` ➔ `ALERT`.
     * Point out the emitted **Alert Card** and read the **Plain-English Explanation Panel ("What Happened?")**:
       > *"CRITICAL: Merchant M1 volume (75 txs/min) breached static threshold 5.00σ with 1-window persistence. Alert emitted to audit trail!"*
  6. **Watch Window 6–10 (Cooldown & Recovery):**
     * Point out State Machine transitioning to `COOLDOWN`.
     * Explain: *"During cooldown, redundant alerts are suppressed for 5 consecutive windows to prevent alert fatigue."*
     * In Window 11, show state returning to `NORMAL` once 5 consecutive normal windows complete.

---

### Step 3: Evaluation & Evidence Inspection (45 Seconds)
* **Goal:** Present measured benchmark performance on the locked holdout dataset.
* **Actions:**
  1. Navigate to **Tab 2: 📊 Evaluation & Evidence**.
  2. Review headline Holdout KPIs:
     * **Precision:** `0.8000` (TP: 4, FP: 1)
     * **Recall:** `0.8000` (FN: 1, Total Events: 5)
     * **F1 Score:** `0.8000`
     * **Median Latency:** `64.57s` (P95 Latency: `114.57s`)
     * **Total Cost:** `₹850.00`
  3. Show the **Per-Anomaly Performance Table**:
     * Point out `VALIDATED IN HOLDOUT` vs `NO_EVENTS_IN_DATASET` badges (scientific honesty).
  4. Highlight the **5-Way Signal Ablation Comparison** (`FULL`, `-VOLUME`, `-VELOCITY`, `-AMOUNT`, `-BEHAVIORAL`).
  5. Show **Detector-Aware Evasion Confirmation** (Threshold-hugging, Persistence evasion, Staircase ramp, Oscillating sub-threshold).
  6. Point out **Descriptive Calibration** (Reliability Diagram visualizer) and **Bootstrap 95% Confidence Intervals** ($N = 1000$ resamples).

---

### Step 4: Reproducibility & SQLite Audit Verification (30 Seconds)
* **Goal:** Prove auditability and immutable provenance.
* **Actions:**
  1. Navigate to **Tab 3: 🔬 Reproducibility & Audit**.
  2. Show immutable metadata: `detector_version: 1.1.0`, `config_hash`, `development_dataset_hash`, `holdout_dataset_hash`, `seed: 42`, and canonical `artifact_sha256`.
  3. Scroll down to the **SQLite Audit Log Browser** table:
     * Point out live audit records, feature snapshots, evidence quality ratings, and emitted alert IDs saved to SQLite.

---

## 🛠️ Command-Line Verification & Testing

To independently verify the complete pytest benchmark suite from the terminal:
```bash
python -m pytest tests/ -v
```
All 273+ unit and architectural boundary tests execute deterministically and pass with zero failures.
