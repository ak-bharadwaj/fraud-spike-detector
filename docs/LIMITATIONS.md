# 🔬 Fraud-Spike Detector — Scientific Honesty & Known Limitations

This document explicitly outlines the scientific boundaries, dataset assumptions, non-negotiable architectural guarantees, and known limitations of the Fraud-Spike Detector system.

---

## 📌 Non-Negotiable Boundaries & Guarantees

1. **Defense-Only Scope:**
   Fraud-Spike Detector is designed strictly for risk operational intelligence, alerting, and explainable audit logging. It produces risk scores $M$ and confidence ratings for human risk analyst review. It **NEVER** auto-blocks, rejects, or freezes payment transactions.

2. **Zero Ground-Truth & Holdout Contamination:**
   Detector logic in `src/detector/`, `src/features/`, `src/baseline/`, `src/scoring/`, `src/state/`, and `src/audit/` contains ZERO imports or dependencies on ground truth event generation or holdout evaluation datasets.

3. **Deterministic Simulation & Time Handling:**
   All execution time advancement relies on `VirtualClock`. System processing contains zero wall-clock dependencies (`datetime.now()`) or non-deterministic random entropy (`uuid.uuid4()`).

---

## ⚠️ Known Scientific Limitations

1. **Synthetic Stream vs. Real-World Production Traffic:**
   * Benchmark metrics are evaluated on synthetic transaction streams generated under controlled Poisson and Gaussian distribution models.
   * Real-world financial production traffic exhibits unmodeled multi-modal behavioral shifts, complex fraud networks, and adversarial adaptation beyond clean synthetic models.

2. **Unrepresented Anomaly Classes in Locked Holdout:**
   * The locked holdout dataset contains 5 ground truth events (`EVT-HOLDOUT-001` through `EVT-HOLDOUT-005`).
   * Certain anomaly archetypes present in synthetic benchmark generators (such as attribute geo-shift or complex multi-attribute compound anomalies) are not represented in the canonical 5-event holdout dataset and are reported as `NO_EVENTS_IN_DATASET`.

3. **Descriptive Nature of Post-Holdout Analyses:**
   * Calibration reliability diagrams and portfolio cost comparisons are descriptive characterizations of holdout performance.
   * Model freeze was executed on Day 7; post-holdout findings characterize the locked release without feeding back into detector retuning.

4. **Non-Discriminative Development Ablation:**
   * On the development characterization dataset, single-signal masking (`-VOLUME`, `-VELOCITY`, `-AMOUNT`, `-BEHAVIORAL`) yields zero change in F1 score ($\Delta \text{F1} = 0.0000$).
   * This occurs because synthetic spike anomalies elevate volume, velocity, and amount statistics simultaneously, providing redundant signal paths.

---

## 🔮 Recommendations for Production Defense Systems

To transition from this reproducible benchmark prototype to an enterprise production defense pipeline:
1. **Supervised & Semi-Supervised Model Ensembles:** Combine statistical deviation scoring with supervised gradient-boosted trees (e.g. XGBoost/LightGBM) trained on historical chargeback data.
2. **Graph-Based Entity Resolution:** Integrate graph neural network (GNN) embeddings to track cross-merchant card-testing rings and shared device infrastructure.
3. **Adaptive Baseline Windowing:** Dynamic seasonal baseline windowing (e.g., matching day-of-week and hour-of-day temporal cycles).
