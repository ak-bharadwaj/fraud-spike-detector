# 🎙️ 5-Minute Technical Pitch Script (§41 Compliance)

**Project:** Real-Time Payment Fraud-Spike Detector for Razorpay  
**Presenter:** Engineering Lead  
**Target Duration:** Exactly 5:00 (300 seconds)  
**Execution Repository Commit:** Canonical Release HEAD  

---

## ⏱️ Pitch Timeline & Cue Sheet

```
+-------------------+---------------------------------------------------------+---------------------------------+
| Time Block        | Section & Core Technical Narrative                      | Live Demo / UI Visual Cue       |
+-------------------+---------------------------------------------------------+---------------------------------+
| 0:00 - 0:45 (45s) | 1. Problem & Financial Stakes                           | Tab 1: Live Stream Monitor      |
| 0:45 - 1:45 (60s) | 2. Architecture & Streaming Engine                      | Tab 1: Merchant Metrics & Chart |
| 1:45 - 2:45 (60s) | 3. Multi-Signal Scorer & Evidence Degradation Firewall | Tab 1: Score & State Transition |
| 2:45 - 3:45 (60s) | 4. Locked Holdout Evaluation & Evasion/Drift            | Tab 2: KPI & Evasion Table      |
| 3:45 - 5:00 (75s) | 5. Live Operations Console & Scientific Disclosure      | Tab 3: Replay & Audit Trail     |
+-------------------+---------------------------------------------------------+---------------------------------+
```

---

## 📜 Complete Timed Presentation Script

### 1. Problem & Financial Stakes (0:00 – 0:45 | 45 Seconds)
> *"Good morning. When fraud attacks hit payment gateways like Razorpay, they don't arrive as isolated bad transactions—they erupt in hyper-dense bursts across volume, velocity, amount distribution, and device pools. Traditional batch models react too slowly, exposing merchants to massive chargeback loss (₹800+ per missed event). Conversely, naive thresholding triggers catastrophic alert fatigue with ₹50 operational review costs per false positive.*
> 
> *Our solution is an event-driven, real-time Fraud-Spike Detector that operates strictly in streaming time with zero future data leakage, providing dynamic EWMA risk scoring, adaptive baseline tracking, and automated alert state machine persistence."*

---

### 2. Architecture & Streaming Engine (0:45 – 1:45 | 60 Seconds)
> *"Architecturally, our system is built around a deterministic, time-ordered event bus. Incoming payment transactions are ingested chronologically with virtual clock monotonicity ($t_{\text{current}} \ge t_{\text{past}}$).
> 
> For every 1-minute streaming window, our Feature Engine computes rolling multi-dimensional statistics—volume, velocity, robust median/MAD amounts, customer cardinality, and device entropy. Crucially, baseline updates occur strictly AFTER historical score calculation ($t_{\text{past}} < t_{\text{current}}$), eliminating baseline contamination during an active attack."*

---

### 3. Multi-Signal Scorer & Evidence Degradation Firewall (1:45 – 2:45 | 60 Seconds)
> *"Our scoring pipeline combines robust statistical Z-scores across four feature groups into a unified risk score via EWMA exponential smoothing ($\alpha=0.3$).
> 
> To protect against corrupted or degraded stream inputs, we implement an Evidence Quality State Machine. If data quality drops—such as missing fields or network dropouts—the system dynamically downweights confidence rather than emitting spurious alerts. Qualifying scores enter an Alert State Machine requiring persistence ($P=1$ window) before firing an alert, followed by a mandatory cooldown ($C=5$ windows) to eliminate alert flooding."*

---

### 4. Locked Holdout Evaluation & Evasion/Drift Confirmation (2:45 – 3:45 | 60 Seconds)
> *"We validated our frozen detector ($\tau=5.0, P=1, C=5$) against a locked holdout dataset (`data/holdout/`) locked prior to final evaluation.
> 
> Results:
> - **Precision:** 80.0% (4 TP / 5 Alerts, 95% Non-Parametric Bootstrap CI: `[0.2000, 0.8000]`)
> - **Recall:** 80.0% (4 TP / 5 Events, 95% Non-Parametric Bootstrap CI: `[0.2000, 0.8000]`)
> - **Detection Latency:** Median 64.57 seconds.
> - **Total Financial Impact:** ₹850.00 (₹50 FP review cost + ₹800 FN exposure).
> 
> We also verified detector-aware evasion scenarios—confirming 3 of 4 physical evasion patterns (threshold-hugging, persistence evasion, staircase ramp) while identifying specific model limits under low-amplitude harmonic oscillation."*

---

### 5. Live Operations Console & Scientific Disclosure (3:45 – 5:00 | 75 Seconds)
> *"Now let's switch to our live Web Operations Console.
> 
> In **Tab 1 (Live Stream Monitor)**, you can see real-time 1-minute window playback for Merchant `M1`. Watch as transaction volume surges at Window 5—the Risk Score breaches threshold $\tau=5.0$, transitioning state from `NORMAL` $\to$ `CANDIDATE` $\to$ `ALERT`, emitting Alert `ALT-HOLDOUT-M1-005` in 64.57 seconds.
> 
> In **Tab 2 (Evaluation & Evidence)**, we openly disclose our empirical limitations:
> 1. **Small-N Holdout Sample Size:** N=5 total ground truth events, yielding wide 95% Bootstrap CIs `[0.2000, 0.8000]`.
> 2. **Zero-Event Anomaly Coverage:** 6 anomaly classes are explicitly reported as `NO_EVENTS_IN_DATASET` rather than fabricating 100% metrics.
> 3. **Evasion Miss:** Harmonic sub-threshold oscillation remains below detection threshold, which we document as an honest engineering boundary.
> 
> Finally, **Tab 3 (Replay & Audit Trail)** provides complete SQLite audit trail transparency for regulatory compliance.
> 
> Thank you."*

---

## 🎯 Key Questions & Answers for Judges

1. **Q: Why are your 95% Confidence Intervals so wide (`[0.2000, 0.8000]`)?**  
   *A:* Because the locked holdout dataset contains N=5 ground truth events. Non-parametric bootstrap resampling over 1,000 iterations over small sample sizes correctly reflects sampling variance rather than disguising it with parametric assumptions.

2. **Q: Why do 6 anomaly classes report `NO_EVENTS_IN_DATASET`?**  
   *A:* Scientific honesty. The locked holdout dataset contains volume spikes and evasion patterns. Reporting `NO_EVENTS_IN_DATASET` prevents inventing fabricated 1.00 metrics for unrepresented anomaly classes.

3. **Q: How does the system prevent baseline contamination during long-running fraud?**  
   *A:* Baseline updates execute strictly after score evaluation ($t_{\text{past}} < t_{\text{current}}$), and EWMA smoothing prevents sudden high-magnitude spikes from immediately inflating historical baselines.
