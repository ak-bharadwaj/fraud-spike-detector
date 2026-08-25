# Fraud-Spike Detector

**AI Risk Manager — Master Build & Research Plan**

> **Status:** FROZEN BUILD CONTRACT  
> **Primary Objective:** Build, evaluate, and defend a near-real-time merchant-level anomaly detector using a reproducible synthetic transaction stream.

---

## 📌 Executive Summary

Fraud-Spike Detector is a defense-only system that monitors a synthetic transaction stream to detect anomalous merchant behavior in near-real-time. It identifies volume spikes, velocity bursts, amount-distribution shifts, behavioral/device anomalies, attribute shifts, compound anomalies, and detector-aware evasive patterns.

### Key Principles
- **Explainable & Interpretable:** Uses explicit statistical models (`StaticThresholdScorer`, `StatisticalDeviationScorer`, `HybridEWMAScorer`) instead of black-box LLMs or deep neural networks for auditability.
- **Audit-First Design:** Emits detailed audit trails to SQLite for every event, score, and state transition.
- **Defense-Only:** Produces risk scores, explanations, and alerts for human review. Never auto-blocks, rejects, or freezes financial transactions.
- **Strict Reproducibility:** Uses `VirtualClock` and deterministic RNG seeds per merchant to guarantee repeatable stream playback and evaluation.

---

## 🏗️ System Architecture

```
Generator
   ↓ (transactions only)
TimeOrderedEventBus + VirtualClock
   ↓
Feature Engine
   ↓
Baseline Engine + Evidence State
   ↓
Anomaly Scorer
   ├── StaticThresholdScorer
   ├── StatisticalDeviationScorer
   └── HybridEWMAScorer
   ↓
AlertStateMachine
   ├── Alert → SQLite
   └── Error → AuditRecord only
   ↓
Evaluator
   ├── Precision / Recall
   ├── Latency
   ├── Calibration
   ├── Bootstrap CI
   ├── Ablation
   ├── Drift
   ├── Evasion
   └── Portfolio Cost
```

---

## 📁 Repository Structure

```
fraud-spike-detector/
├── config/                  # Configuration files (YAML)
│   ├── detector.yaml
│   ├── generator.yaml
│   └── evaluation.yaml
├── src/                     # Core application source code
│   ├── contracts/           # Pydantic data schemas
│   ├── generator/           # Synthetic stream generator
│   ├── stream/              # TimeOrderedEventBus & VirtualClock
│   ├── features/            # Feature aggregation engine
│   ├── baseline/            # Baseline computation & EvidenceState
│   ├── scoring/             # Anomaly scorers strategy pattern
│   ├── state/               # Alert state machine
│   ├── detector/            # Detector orchestration pipeline
│   ├── audit/               # SQLite audit persistence
│   └── evaluation/          # Benchmark & evaluation metrics
├── tests/                   # Pytest suite & boundary enforcement tests
├── scripts/                 # Execution & experiment scripts
├── data/                    # Dataset storage (development & holdout)
│   ├── development/
│   └── holdout/
├── artifacts/               # Benchmark outputs, charts, and metrics
├── requirements.txt         # Pinned python dependencies
└── README.md
```

---

## 🚀 Technology Stack

- **Runtime:** Python 3.11+
- **Data Contracts:** Pydantic
- **Streaming Logic:** `TimeOrderedEventBus` + `VirtualClock`
- **Persistence:** SQLite
- **Numerical Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Testing & Quality:** Pytest
- **Config:** YAML + Pydantic validation

---

## 📅 10-Day Build Roadmap

| Day | Phase | Deliverable |
|---|---|---|
| **Day 1** | Foundation & Contracts | Repository setup, Pydantic contracts, `VirtualClock`, RNG seeding, holdout guard |
| **Day 2** | Synthetic Stream | Merchants M1–M3, normal traffic, sudden volume spikes, ground-truth events |
| **Day 3** | Vertical Slice | End-to-end processing pipeline (Transactions → Features → Baseline → Scorer → Alert → SQLite) |
| **Day 4** | Evaluation Pipeline | Matching engine, precision/recall/latency metrics, sweep scripts |
| **Day 5** | Synthetic Benchmark | Full merchant suite M1–M9, evasion patterns, compound anomalies |
| **Day 6** | Research Experiments | Feature ablation, baseline drift, threshold-hugging evasion, degraded data tests |
| **Day 7** | System Freeze | Model selection (`HybridEWMAScorer`), hyperparameter lock, tag frozen release |
| **Day 8** | Holdout Evaluation | Unlock holdout dataset, evaluate metrics, bootstrap CI, cost model |
| **Day 9** | Evidence & Artifacts | Generate charts, reliability diagrams, cost comparisons, SQLite audit rehearsal |
| **Day 10**| Buffer & Pitch | Pitch recording, documentation finalization, final validation |

---

## 📜 License

MIT License.
