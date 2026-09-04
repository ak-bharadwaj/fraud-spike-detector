# 🎙️ 5-Minute Technical Pitch Script (§41 Compliance)

**Project:** Real-Time Payment Fraud-Spike Detector for Razorpay  
**Presenter:** Engineering Lead  
**Target Duration:** Exactly 5:00 (300 seconds)  
**Execution Repository Commit:** Canonical Release HEAD  

---

## ⏱️ Pitch Timeline & Cue Sheet (Exact SSOT 7-Section Structure)

```
+---------------------+-----------------------------------------------------------+---------------------------------+
| Time Block          | Section & Core Technical Narrative                        | Live Demo / UI Visual Cue       |
+---------------------+-----------------------------------------------------------+---------------------------------+
| 0:00 - 0:40 (40s)   | 1. Problem & Risk-Management Framing                      | Tab 1: Live Stream Monitor      |
| 0:40 - 1:20 (40s)   | 2. System Architecture & Dual-Track Pipeline              | Tab 1: Merchant Metrics & Chart |
| 1:20 - 2:20 (60s)   | 3. Live Detection & SQLite Audit Trail                    | Tab 1: Score & State Transition |
| 2:20 - 3:10 (50s)   | 4. Benchmark Precision, Recall, Latency & Calibration     | Tab 2: Evaluation & Evidence    |
| 3:10 - 3:50 (40s)   | 5. 6-Way Ablation, Drift, Evasion & Bootstrap Uncertainty | Tab 2: Evaluation & Evidence    |
| 3:50 - 4:30 (40s)   | 6. Portfolio FP/FN Financial Cost Analysis                | Tab 2: Evaluation & Evidence    |
| 4:30 - 5:00 (30s)   | 7. Failure Story & Explicit Engineering Boundaries        | Tab 3: Replay & Audit Trail     |
+---------------------+-----------------------------------------------------------+---------------------------------+
```

---

## 📜 Complete Timed Presentation Script

### 1. Problem & Risk-Management Framing (0:00 - 0:40 | 40 Seconds)
> *"Good morning. When fraud attacks hit payment gateways like Razorpay, they erupt in hyper-dense bursts across volume, velocity, amount distribution, and device pools. Traditional batch models react too slowly, exposing merchants to massive chargeback loss (₹800+ per missed event). Conversely, naive static thresholding triggers catastrophic alert fatigue with ₹50 operational review costs per false positive.
> 
> Our solution is Fraud-Spike Detector—an event-driven, audit-first risk intelligence engine operating strictly under a defense-only boundary: producing interpretable risk scores and automated alerts for human risk operations teams without ever auto-blocking payment flows."*

---

### 2. System Architecture & Dual-Track Pipeline (0:40 - 1:20 | 40 Seconds)
> *"Architecturally, our system is built around a dual-track evaluation pipeline. In Track A, incoming payment streams are ingested through a deterministic, time-ordered event bus. For every 1-minute sliding window, our Feature Engine computes rolling multi-dimensional statistics—volume, velocity, robust median/MAD amounts, customer cardinality, and device entropy. Baseline updates execute strictly AFTER historical score calculation ($t_{\text{past}} < t_{\text{current}}$), eliminating baseline contamination during active attacks.
> 
> In Track B, we complement real-time streaming with a real-world public benchmark evaluation: training a primary XGBoost classifier (Platt-calibrated strictly on CALIBRATION split) on 284,807 transactions with strict 3-way temporal splitting."*

---

### 3. Live Detection & SQLite Audit Trail (1:20 - 2:20 | 60 Seconds)
> *"Our frozen streaming detector, `StatisticalDeviationScorer`, evaluates composite Z-score deviation magnitude $M_{\text{composite}} = \max_k M_k$ against static decision threshold $\tau = 5.00\sigma$ ($P=1, C=5$). EWMA exponential smoothing ($\alpha=0.3$, alpha=0.3) was evaluated strictly in development sweeps as part of our research trade-off portfolio.
> 
> Qualifying scores enter an Alert State Machine requiring persistence ($P=1$ window) to confirm alerts and cooldown ($C=5$ windows) to eliminate alert fatigue. In **Tab 1: Live Stream Monitor**, watch as Merchant `M1` transaction volume surges at Window 5—the Risk Score breaches threshold $\tau=5.0$, transitioning state from `NORMAL` $\to$ `CANDIDATE` $\to$ `ALERT` with 100% SQLite audit log persistence."*

---

### 4. Benchmark Precision, Recall, Latency & Calibration (2:20 - 3:10 | 50 Seconds)
> *"In **Tab 2: Evaluation & Evidence**, we present benchmark evaluation results across both tracks:
> 
> - **Track A (Synthetic Holdout):** Precision = 80.0% (4 TP / 5 Alerts), Recall = 80.0% (4 TP / 5 Events), F1 = 0.8000, Median Latency = 64.57s.
> - **Track B (Real-World Kaggle Benchmark):** Evaluated on 42,721 held-out test transactions (52 fraud transactions) under strict 3-way temporal split. Our primary Platt-calibrated XGBoost classifier achieves **Precision = 82.98%** (39 TP / 8 FP), **Recall = 75.00%** (39 TP / 13 FN), **F1 = 0.7879**, **AUC-ROC = 0.9825**, **AUC-PR = 0.7703**, and Platt-scaled Calibration ECE = 0.01%."*

---

### 5. 6-Way Ablation, Drift, Evasion & Bootstrap Uncertainty (3:10 - 3:50 | 40 Seconds)
> *"We rigorously stress-tested our detector across both tracks:
> 
> 1. **Principled Feature Ablation:** On real credit card data across 6 variants, Time & Amount alone yield F1 = 0.0053 (-0.7736), confirming that anonymized PCA dimensions (`V1`–`V28`) contain the dominant predictive fraud information. Unsupervised Isolation Forest alone yields F1 = 0.0977 (-0.6812), proving the necessity of supervised XGBoost.
> 2. **Detector-Aware Evasion:** Verified representative evasion patterns (threshold-hugging, persistence evasion, staircase ramp).
> 3. **Bootstrap Uncertainty:** 1,000 resamples yield exact 95% CIs of `[0.7142, 0.9298]` for real-world precision, `[0.6274, 0.8667]` for recall, and `[0.6857, 0.8687]` for F1."*

---

### 6. Portfolio FP/FN Financial Cost Analysis (3:50 - 4:30 | 40 Seconds)
> *"Using our financial cost model (₹50 per false positive review, ₹800 base exposure per missed event), our frozen detector achieved a total portfolio cost of **₹850.00** on the synthetic holdout stream.
> 
> On the real-world dataset, achieving 82.98% precision with 8 False Positives across 42,721 transactions limits manual analyst review overhead to ₹400.00, with FN exposure calculated dynamically as the actual sum of missed amounts (₹2,372.40 across 13 missed frauds), yielding a total portfolio cost of **₹2,772.40**."*

---

### 7. Failure Story & Explicit Engineering Boundaries (4:30 - 5:00 | 30 Seconds)
> *"Finally, in **Tab 3: Replay & Audit Trail**, we present complete transparency and document our real historical failure and explicit boundaries:
> 
> 1. **Historical Fix Story:** In `run_001`, a pseudo-probability division bug caused improper calibration, which was resolved in `run_004`.
> 2. **Small-N Holdout:** Honest reporting of wide 95% CIs `[0.2000, 0.8000]` given N=5 holdout events.
> 3. **Zero-Event Anomaly Coverage:** Unrepresented anomaly classes in the synthetic holdout are reported as `NO_EVENTS_IN_DATASET`.
> 4. **Evasion Boundary:** Low-amplitude harmonic oscillation (`EVT-HOLDOUT-005`, $M=1.64\sigma$) remains below decision threshold $\tau=5.00$—an explicit model boundary.
> 
> Thank you."*

---

## 🎯 Key Questions & Answers for Judges

1. **Q: Why was XGBoost chosen as the primary model when PCA_ONLY scored slightly higher F1?**  
   *A:* **Primary Model Pre-Designation:** XGBoost on all features was pre-designated as the primary supervised model prior to locked-test evaluation. Feature group ablations (such as PCA_ONLY) are descriptive characterization experiments evaluated under frozen hyperparameters and do not redefine the primary model post-hoc based on test set performance.

2. **Q: Why is Track A's median detection latency exactly 64.57s across all detected events?**  
   *A:* **Deterministic Window Alignment:** Ground truth events in the locked holdout start at 10-minute boundaries (`:10:00`, `:20:00`, etc.). The feature engine's 1-minute window sequence aligns to the dataset's first transaction timestamp (`:00:04.567`). The first window fully capturing an anomaly completes at `:04.567` past the subsequent minute. The difference between `:11:04.567` and `:10:00.000` is exactly $60\text{s} + 4.567\text{s} = 64.567\text{s}$.

3. **Q: Why are your 95% Confidence Intervals so wide (`[0.2000, 0.8000]`)?**  
   *A:* Because the Track A locked holdout dataset contains $N=5$ ground truth events. Non-parametric bootstrap resampling over 1,000 iterations over small sample sizes correctly reflects sampling variance rather than disguising it. In Track B ($N=42,721$ test transactions, 52 fraud transactions), bootstrap CIs shrink to tight `[0.6857, 0.8687]` bounds.

4. **Q: How do you interpret the Track B ECE of 0.0001 (0.01%)?**  
   *A:* This is the 10-bin Expected Calibration Error of Platt-scaled XGBoost probabilities on the locked test set. Because the dataset has extreme class imbalance (42,675 of 42,721 test transactions fall in $[0, 0.1)$), the near-zero ECE reflects high calibration accuracy in the dominant background bucket alongside calibrated probabilities in the minority fraud buckets.

5. **Q: How do you prevent data leakage in your real-world ML benchmark?**  
   *A:* We use a strict 3-way temporal split (TRAIN 70% ➔ CALIBRATION 15% ➔ LOCKED TEST 15%). Base models are trained on TRAIN, Platt scaling is fitted on CALIBRATION, and evaluation is performed on the un-seen LOCKED TEST set.

6. **Q: How does the system prevent baseline contamination during long-running fraud?**  
   *A:* Baseline updates execute strictly after score evaluation ($t_{\text{past}} < t_{\text{current}}$), and robust median/MAD scaling prevents sudden high-magnitude spikes from immediately inflating historical baselines.
