# 🏗️ Fraud-Spike Detector — System Architecture Specification

This document provides a comprehensive end-to-end technical specification of the Fraud-Spike Detector system architecture, event pipeline, data contracts, and operational invariants.

---

## 📌 System Overview

Fraud-Spike Detector is an audit-first, near-real-time streaming transaction anomaly detection system. It monitors transaction streams, aggregates statistical features over minute-aligned sliding windows, evaluates merchant-specific baseline deviations, computes multi-factor confidence ratings, and manages alert emissions via a state machine.

```
+------------------+     (transactions)     +-----------------------------------+
|  Synthetic Stream| ---------------------> | TimeOrderedEventBus + VirtualClock|
|    Generator     |                        +-----------------------------------+
+------------------+                                          |
                                                              v
+------------------+                        +-----------------------------------+
|  SQLite Audit    | <--------------------- |    StreamingDetectorPipeline      |
|  Database Store  |                        +-----------------------------------+
+------------------+                                          |
        ^                                                     v
        |                                   +-----------------------------------+
        |---------------------------------- |          Feature Engine           |
        |                                   +-----------------------------------+
        |                                                     |
        |                                                     v
        |                                   +-----------------------------------+
        |---------------------------------- |    Baseline Engine + Evidence     |
        |                                   +-----------------------------------+
        |                                                     |
        |                                                     v
        |                                   +-----------------------------------+
        |---------------------------------- |    StatisticalDeviationScorer    |
        |                                   +-----------------------------------+
        |                                                     |
        |                                                     v
        |                                   +-----------------------------------+
        |---------------------------------- |     AlertStateMachine (State)     |
        +---------------------------------- +-----------------------------------+
```

---

## 🧱 Core Architectural Components

### 1. Data Contracts (`src/contracts/contracts.py`)
Built on strictly validated Pydantic schemas enforcing structural type safety and explicit nullability constraints:
* **`Transaction`**: Immutable record carrying `transaction_id`, timezone-aware `timestamp`, `merchant_id`, `customer_id`, `amount`, `payment_method`, `country`, and `device_id`.
* **`FeatureSnapshot`**: Minute-aggregated window features including `volume`, `velocity`, `amount_statistics` (`mean`, `std`, `min`, `max`, `median`, `mad`), `unique_customers`, `unique_devices`, and `data_quality`.
* **`BaselineSnapshot`**: Expected baseline values derived from historical windows (`expected_values`, `robust_scale`, `history_count`, `current_window_count`, `evidence_state`).
* **`RiskScore`**: Standardized deviation magnitude score $M$, composite `confidence` float, `triggered_signals` list, and `data_quality`.
* **`Alert`**: Emitted risk notification with deterministic SHA-256 `alert_id`, `merchant_id`, `timestamp`, `risk_score`, `confidence`, `reason`, `triggered_signals`, and `detector_version`.
* **`AuditRecord`**: Complete auditable record written to SQLite containing feature values, baseline expectations, score, confidence, signals, state machine status, and data quality.
* **`FrozenDetectorConfig`**: Immutable configuration contract locking `scorer="StatisticalDeviationScorer"`, `static_threshold=5.0`, `persistence=1`, `cooldown_windows=5`, and `detector_version="1.1.0"`.

---

### 2. Time Engine & Event Bus (`src/stream/`)
* **`VirtualClock`**: Monotonic simulation clock authority ensuring 100% deterministic time advancement. Rejects backward time jumps.
* **`TimeOrderedEventBus`**: Event dispatch bus ordering incoming stream events by timestamp, breaking ties deterministically, and advancing `VirtualClock` during stream draining.

---

### 3. Feature Aggregation Engine (`src/features/feature_engine.py`)
* Aligns transactions into half-open 1-minute sliding windows `[HH:MM:00, HH:MM:00 + 1m)`.
* Computes robust non-parametric and parametric statistics:
  $$\text{Median} = \text{median}(X), \quad \text{MAD} = \text{median}(|X_i - \text{Median}|)$$
* Evaluates data quality degradation flags (`GOOD`, `DEGRADED`, `EMPTY`).

---

### 4. Baseline Engine & Evidence State (`src/baseline/baseline_engine.py`)
* Maintains historical feature windows independently per merchant.
* Computes running baseline expected values $E[X]$ and robust scale $S[X] = \max(\text{MAD}[X], \epsilon)$.
* Evaluates Evidence State based on accumulated historical window count:
  * **`INSUFFICIENT`**: $N < \text{min\_history\_count}$ (scores suppressed, returns $M = \text{None}$, confidence $= 0.0$).
  * **`DEGRADED`**: Missing required features or degraded attributes (scores calculated, confidence $= 0.5$).
  * **`SUFFICIENT`**: $N \ge \text{min\_history\_count}$ with clean features (scores calculated, confidence $= 1.0$).

---

### 5. Anomaly Scorer (`src/scoring/`)
Uses the Strategy Pattern (`AnomalyScorer` abstract base class):
* **`StatisticalDeviationScorer`** (Frozen v1.1.0 Scorer): Calculates standardized deviation magnitudes across feature groups:
  $$M_k = \frac{|X_k - E[X_k]|}{S[X_k]}$$
  $$M_{\text{composite}} = \max_{k \in \text{active\_signals}} (w_k \cdot M_k)$$
  Compares composite magnitude $M_{\text{composite}}$ against static threshold $\tau = 5.00$.

---

### 6. Alert State Machine (`src/state/alert_state_machine.py`)
Manages state transitions per merchant:
* **Lifecycle:** `NORMAL` $\rightarrow$ `CANDIDATE` (or `SUSPICIOUS`, §10 persistence accumulation state) $\rightarrow$ `ALERT` $\rightarrow$ `COOLDOWN` $\rightarrow$ `NORMAL`.
* **Persistence Gating ($P=1$):** Requires score $M \ge \tau$ for $P$ consecutive qualifying windows.
* **Cooldown Suppression ($C=5$):** After emitting an Alert, state transitions to `COOLDOWN` for 5 consecutive normal windows. Anomalous scores during cooldown reset the cooldown counter to prevent alert flooding.

---

### 7. Audit Store (`src/audit/database.py`)
* SQLite persistence layer writing relational records for every evaluation window.
* Ensures 100% auditability without mutating stream evaluation or introducing non-determinism.
