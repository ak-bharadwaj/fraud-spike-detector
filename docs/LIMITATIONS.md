# 🔬 Fraud-Spike Detector — Scientific Honesty & Known Limitations

This document explicitly outlines the scientific boundaries, dataset assumptions, non-negotiable architectural guarantees, and known limitations of the Fraud-Spike Detector system.

---

## 📌 Non-Negotiable Boundaries & Guarantees

1. **Defense-Only Scope:**
   Fraud-Spike Detector is designed strictly for risk operational intelligence, alerting, and explainable audit logging. It produces risk scores and confidence ratings for human risk analyst review. It **NEVER** auto-blocks, rejects, or freezes payment transactions.

2. **Zero Contamination & Data Leakage:**
   - **Track A (Synthetic Streaming):** Detector logic in `src/` contains ZERO imports or dependencies on ground truth event generation or holdout evaluation datasets (`data/holdout/`).
   - **Track B (Real-World Benchmark):** Dataset splitting uses a strict 3-way temporal split (TRAIN 70% ➔ CALIBRATION 15% ➔ LOCKED TEST 15%). Calibration models (Platt scaling) are fitted strictly on the CALIBRATION split, preventing test set data leakage.

3. **Deterministic Simulation & Time Handling:**
   All synthetic execution time advancement relies on `VirtualClock`. System processing contains zero wall-clock dependencies (`datetime.now()`) or non-deterministic random entropy (`uuid.uuid4()`).

---

## ⚠️ Known Scientific & Dataset Limitations

1. **Synthetic Stream vs. Real-World Public Benchmark Track Separation:**
   - Track A evaluates deterministic streaming state-machine behavior on 5 locked synthetic events.
   - Track B evaluates learned multi-dimensional classification on 42,721 held-out real credit card transactions (52 fraud transactions).
   - The two tracks answer distinct technical questions and are presented separately.

2. **ULB / Kaggle Dataset Dimension Anonymization:**
   - The Kaggle Credit Card dataset features 28 PCA-transformed dimensions (`V1`–`V28`) plus `Time` and `Amount`.
   - It lacks merchant IDs, IP addresses, or device fingerprints. Therefore, real-world evaluation measures transaction-level fraud discrimination rather than streaming merchant window aggregation.

3. **Feature Dominance in Real-World Credit Card Fraud:**
   - As proven by our ablation study, `Time` and `Amount` alone yield F1 = 0.0053 (-0.7736), confirming that anonymized PCA dimensions (`V1`–`V28`) contain the dominant predictive fraud information in this benchmark.

4. **Descriptive Nature of Post-Holdout Analyses:**
   - Post-holdout calibration and cost analyses characterize frozen model outputs. Holdout metrics reflect a single evaluation pass without post-hoc hyperparameter retuning.

---

## 🔮 Recommendations for Production Defense Systems

To transition from this reproducible benchmark prototype to an enterprise production defense pipeline:
1. **Hybrid Streaming Ensembles:** Combine real-time statistical deviation scoring ($\tau=5.0\sigma$) for instant velocity burst detection with calibrated XGBoost models for deep transactional fraud scoring.
2. **Graph-Based Entity Resolution:** Integrate graph neural network (GNN) embeddings to track cross-merchant card-testing rings and shared device infrastructure.
3. **Adaptive Baseline Windowing:** Dynamic seasonal baseline windowing (matching day-of-week and hour-of-day temporal cycles).
