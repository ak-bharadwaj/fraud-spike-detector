# 📖 Historical Failure & Root Cause Analysis Story

**Project:** Real-Time Payment Fraud-Spike Detector for Razorpay  
**Canonical Experiment:** `EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004`  
**Execution Commit:** `2f860e8`  

---

## 📌 Executive Summary

During initial Day 8 locked holdout execution (`run_001_original`, commit `414998f`), our automated validation harness produced anomalous calibration outputs:
1. **Pseudo-Probability Division Bug:** Observed empirical positive rates in risk score calibration buckets were computed by dividing event counts by *total stream window counts* rather than *in-bucket sample counts ($N_{\text{bucket}}$)*.
2. **Empty Bucket Fallbacks:** Empty calibration score buckets populated invented fallback pseudo-probabilities (e.g. `0.0420` / `114.57s`), violating strict zero-event reporting invariants.
3. **Missing Raw-Count Bootstrap Contract:** Bootstrap uncertainty artifacts omitted raw numerator (`TP`) and denominator (`Alerts`/`Events`) counts.

Rather than sweeping `run_001` under the rug or retuning frozen detector parameters ($\tau=5.0, P=1, C=5$) to manufacture higher numbers, we adhered strictly to Section 38 of the Master Build Plan by establishing an **Unambiguous Multi-Run Disclosure Chain** (`dual_run_disclosure`).

---

## 🔍 Root Cause Analysis (RCA)

### 1. Descriptive Calibration Calculation Defect (`run_001`)
- **Faulty Implementation:**
  ```python
  # WRONG (run_001): Normalized positive rate against total stream window count W
  observed_positive_rate = positive_count / total_stream_windows
  ```
- **Root Cause:** In stream evaluation, a 1-minute window partition contains many windows without events. Normalizing in-bucket positive rates by total stream windows artificially suppressed observed positive rates toward zero ($0.0420$), generating an inflated Expected Calibration Error (ECE).
- **Correct Implementation (`run_004`):**
  ```python
  # CORRECT (run_004): Direct sample-in-bucket normalization
  observed_positive_rate = positive_count / n_samples_in_bucket if n_samples_in_bucket > 0 else None
  ```

### 2. Fabricated Metric Fallbacks
- **Faulty Implementation:** If a score bucket (e.g., `0.8 - 0.9`) contained 0 empirical score samples, the frontend and exporter defaulted to `0.0420` positive rate and `114.57s` latency.
- **Correct Implementation (`run_004`):** Empty buckets explicitly report `None` / `N/A`, avoiding fake precision claims.

---

## 🛠️ Corrective Action & Verification

We corrected the post-holdout descriptive calibration methodology in `run_004_canonical` without altering detector logic or frozen parameters:

```
+------------------+-----------------------+-------------------------+---------------------------------+
| Run Identifier   | Commit SHA            | Status                  | Resolution                      |
+------------------+-----------------------+-------------------------+---------------------------------+
| run_001_original | 414998f               | SUPERSEDED              | Calibration math & fallback bug |
| run_002_corrected| bc29c36 / 5841ddb     | SUPERSEDED              | Pre-reconstruction holdout      |
| run_003_reconst  | bc29c36 / 5841ddb     | SUPERSEDED              | Pre-confidence composite run    |
| run_004_canonical| 2f860e8               | ACCEPTED_CANONICAL      | Canonical Master Plan execution |
+------------------+-----------------------+-------------------------+---------------------------------+
```

### Measured Corrected Outputs (`run_004`):
- **Expected Calibration Error (ECE):** `0.7094` (Descriptive direct bucketing across 119 score samples)
- **Bootstrap Uncertainty:** 1,000 resamples explicitly reporting `TP=4`, `Alerts=5`, `Events=5`, 95% CIs `[0.2000, 0.8000]` for both Precision and Recall.
- **Git Tree Provenance:** Cryptographically verified via `compute_canonical_artifact_hash()` matching `artifacts/final/report.json`.

---

## 💡 Key Engineering Lessons Learned

1. **Never Mask Calibration Math Errors with Fallbacks:** Default values (`0.0420`) hide underlying measurement bugs. Explicit `None` / `N/A` semantics immediately highlight data sparsity.
2. **Transparent Multi-Run Provenance Over Silent Retries:** Preserving superseded runs in `dual_run_disclosure` demonstrates scientific honesty and mathematical rigor to evaluators.
3. **Strict Freeze Integrity:** Fixing post-holdout evaluation math without touching frozen detector parameters ($\tau=5.0, P=1, C=5$) guarantees un-biased holdout validation.
