#!/usr/bin/env python
"""Real-World Public Benchmark Evaluation Pipeline (`EXP-REALWORLD-CCF-001`).

End-to-end training and evaluation pipeline for the ULB / Kaggle Credit Card Fraud benchmark:
1. Loads 284,807 transactions (492 fraud events across ~48 hours).
2. Applies strict 3-way temporal split: TRAIN (70%) -> CALIBRATION (15%) -> LOCKED TEST (15%).
3. Fits Isolation Forest (unsupervised) and XGBoost (supervised) on TRAIN split.
4. Fits Platt scaling calibration & IF score normalization strictly on CALIBRATION split.
5. Evaluated ONCE on LOCKED TEST split (zero hyperparameter tuning).
6. Runs 6 principled feature/model group ablations:
   - FULL_ENSEMBLE: IF + XGBoost on all 30 features
   - XGB_ONLY: XGBoost alone on all features
   - IF_ONLY: Isolation Forest alone on all features
   - PCA_ONLY: All 28 PCA dimensions (V1-V28), omitting Amount and Time
   - AMOUNT_TIME_ONLY: Amount and Time only, omitting PCA dimensions
7. Computes calibration curve & Expected Calibration Error (ECE) on LOCKED TEST split.
8. Computes non-parametric bootstrap 95% confidence intervals (N=1000 resamples).
9. Writes complete auditable manifests (dataset, splits, models, ablation, calibration, report) to artifacts/realworld/.

Usage:
    python scripts/train_realworld.py [--csv-path PATH] [--output-dir DIR]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.realworld.data_adapter import (
    KaggleCreditCardAdapter,
    ML_FEATURE_GROUPS,
    ALL_FEATURE_COLS,
)
from src.scoring.ml_scorer import (
    IsolationForestScorer,
    XGBoostFraudScorer,
    EnsembleFraudScorer,
    evaluate_binary_classifier,
    compute_calibration_curve,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    if label:
        print(f"\n  --- {label} ---")
    print(f"  TP: {metrics['tp']:>6d}  |  FP: {metrics['fp']:>6d}  |  FN: {metrics['fn']:>6d}  |  TN: {metrics['tn']:>6d}")
    print(f"  Precision: {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.1f}%)")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    if metrics.get("auc_pr") is not None:
        print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"  FP Cost (assumed Rs.50/FP):   Rs.{metrics['fp_cost']:>10.2f}")
    print(f"  FN Exposure (assumed Rs.800): Rs.{metrics['fn_exposure']:>10.2f}")
    print(f"  Total Portfolio Impact:       Rs.{metrics['total_cost']:>10.2f}")


def run_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute empirical non-parametric bootstrap 95% confidence intervals."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    rng = np.random.RandomState(seed)
    n = len(y_true)

    precision_samples = []
    recall_samples = []
    f1_samples = []
    auc_samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]

        if yt.sum() == 0 or yt.sum() == n:
            continue

        precision_samples.append(precision_score(yt, yp, zero_division=0.0))
        recall_samples.append(recall_score(yt, yp, zero_division=0.0))
        f1_samples.append(f1_score(yt, yp, zero_division=0.0))
        try:
            auc_samples.append(roc_auc_score(yt, ypr))
        except ValueError:
            pass

    def ci(arr):
        a = np.array(arr)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "ci_lower": float(np.percentile(a, 2.5)),
            "ci_upper": float(np.percentile(a, 97.5)),
            "n_samples": len(arr),
        }

    return {
        "precision": ci(precision_samples),
        "recall": ci(recall_samples),
        "f1": ci(f1_samples),
        "auc_roc": ci(auc_samples) if auc_samples else None,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> dict:
    """Find probability threshold that maximizes F1 score on CALIBRATION split."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    thresholds = np.arange(0.01, 1.0, 0.01)
    best_score = -1.0
    best_threshold = 0.5
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(np.int32)
        if y_pred.sum() == 0:
            continue

        prec = precision_score(y_true, y_pred, zero_division=0.0)
        rec = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)

        score = {"f1": f1, "precision": prec, "recall": rec}[metric]
        if score > best_score:
            best_score = score
            best_threshold = float(t)

        results.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })

    return {
        "optimal_threshold": best_threshold,
        "optimal_score": float(best_score),
        "metric": metric,
        "all_thresholds": results,
    }


def run_principled_ablation(
    adapter: KaggleCreditCardAdapter,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
) -> list:
    """Run 6 principled feature/model group ablations on the locked TEST set (Refinement #3 & #4).

    Ablation variants:
    1. FULL_ENSEMBLE: IF + XGBoost ensemble on all features
    2. XGB_ONLY: XGBoost alone on all features
    3. IF_ONLY: Isolation Forest alone on all features
    4. PCA_ONLY: V1-V28 features only (omitting Time & Amount)
    5. AMOUNT_TIME_ONLY: Time & Amount features only (omitting V1-V28 PCA)
    """
    ablation_results = []

    # Helper to fit and evaluate a variant
    def evaluate_variant(variant_id: str, description: str, feature_subset: str, use_xgb: bool, use_if: bool):
        X_tr, y_tr = adapter.get_feature_matrix(train_df, feature_subset)
        X_ca, y_ca = adapter.get_feature_matrix(calib_df, feature_subset)
        X_te, y_te = adapter.get_feature_matrix(test_df, feature_subset)
        feature_names = adapter.get_feature_names(feature_subset)

        xgb_s = None
        if_s = None

        if use_xgb:
            xgb_s = XGBoostFraudScorer(n_estimators=200, max_depth=5, seed=seed)
            xgb_s.fit(X_tr, y_tr, X_calib=X_ca, y_calib=y_ca, feature_names=feature_names)

        if use_if:
            X_tr_norm = X_tr[y_tr == 0]
            if_s = IsolationForestScorer(n_estimators=200, contamination=0.002, seed=seed)
            if_s.fit(X_tr_norm, feature_names=feature_names)
            if_s.calibrate_scores(X_ca)

        if use_xgb and use_if:
            ens = EnsembleFraudScorer(if_scorer=if_s, xgb_scorer=xgb_s, w_if=0.3, w_xgb=0.7)
            # Find optimal threshold on CALIBRATION split
            ca_probas = ens.predict_ensemble_scores(X_ca)
            opt_t = find_optimal_threshold(y_ca, ca_probas, "f1")["optimal_threshold"]
            te_probas = ens.predict_ensemble_scores(X_te)
            te_preds = (te_probas >= opt_t).astype(np.int32)
        elif use_xgb:
            ca_probas = xgb_s.predict_proba(X_ca)
            opt_t = find_optimal_threshold(y_ca, ca_probas, "f1")["optimal_threshold"]
            te_probas = xgb_s.predict_proba(X_te)
            te_preds = xgb_s.predict_labels(X_te, threshold=opt_t)
        else:
            ca_scores = if_s.predict_anomaly_scores(X_ca)
            opt_t = find_optimal_threshold(y_ca, ca_scores, "f1")["optimal_threshold"]
            te_probas = if_s.predict_anomaly_scores(X_te)
            te_preds = if_s.predict_labels(X_te, threshold=opt_t)

        metrics = evaluate_binary_classifier(y_te, te_preds, te_probas)
        return {
            "variant_id": variant_id,
            "description": description,
            "feature_subset": feature_subset,
            "n_features": X_tr.shape[1],
            "optimal_threshold_calib": float(opt_t),
            "metrics": metrics,
        }

    print("    Evaluating variant 1/5: FULL_ENSEMBLE (IF + XGB on all 30 features)...")
    res_full = evaluate_variant("FULL_ENSEMBLE", "Full ensemble (IF + XGBoost on all features)", "all", True, True)
    full_f1 = res_full["metrics"]["f1_score"]

    res_full["delta_f1"] = 0.0
    res_full["delta_precision"] = 0.0
    res_full["delta_recall"] = 0.0
    ablation_results.append(res_full)

    print("    Evaluating variant 2/5: XGB_ONLY (XGBoost alone on all features)...")
    res_xgb = evaluate_variant("XGB_ONLY", "XGBoost alone (all features)", "all", True, False)
    res_xgb["delta_f1"] = res_xgb["metrics"]["f1_score"] - full_f1
    res_xgb["delta_precision"] = res_xgb["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_xgb["delta_recall"] = res_xgb["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_xgb)

    print("    Evaluating variant 3/5: IF_ONLY (Isolation Forest alone on all features)...")
    res_if = evaluate_variant("IF_ONLY", "Isolation Forest alone (all features)", "all", False, True)
    res_if["delta_f1"] = res_if["metrics"]["f1_score"] - full_f1
    res_if["delta_precision"] = res_if["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_if["delta_recall"] = res_if["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_if)

    print("    Evaluating variant 4/5: PCA_ONLY (V1-V28 PCA dimensions only)...")
    res_pca = evaluate_variant("PCA_ONLY", "PCA features V1-V28 only (omitting Time & Amount)", "pca", True, True)
    res_pca["delta_f1"] = res_pca["metrics"]["f1_score"] - full_f1
    res_pca["delta_precision"] = res_pca["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_pca["delta_recall"] = res_pca["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_pca)

    print("    Evaluating variant 5/5: AMOUNT_TIME_ONLY (Time & Amount features only)...")
    res_amt = evaluate_variant("AMOUNT_TIME_ONLY", "Time & Amount features only (omitting PCA dimensions)", "amount_time", True, True)
    res_amt["delta_f1"] = res_amt["metrics"]["f1_score"] - full_f1
    res_amt["delta_precision"] = res_amt["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_amt["delta_recall"] = res_amt["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_amt)

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="Real-World ML Benchmark Evaluation Pipeline (EXP-REALWORLD-CCF-001)")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to creditcard.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/realworld", help="Output directory")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(PROJECT_ROOT) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ===== STAGE 1: Load Dataset & Build Manifest =====
    print_header("STAGE 1: Loading Dataset & Generating Dataset Manifest")
    adapter = KaggleCreditCardAdapter(csv_path=args.csv_path, seed=args.seed)
    dataset_manifest = adapter.get_dataset_manifest()

    print(f"  Dataset: {dataset_manifest['dataset_name']}")
    print(f"  Total transactions: {dataset_manifest['total_transactions']:>10,d}")
    print(f"  Fraud count:        {dataset_manifest['total_fraud']:>10,d}")
    print(f"  Legitimate count:   {dataset_manifest['total_legitimate']:>10,d}")
    print(f"  Fraud rate:         {dataset_manifest['fraud_rate']:>10.4%}")
    print(f"  Time span:          {dataset_manifest['time_span_hours']:>10.1f} hours")
    print(f"  Dataset SHA-256:    {dataset_manifest['dataset_sha256'][:16]}...")

    with open(output_dir / "dataset_manifest.json", "w") as f:
        json.dump(dataset_manifest, f, indent=2)

    # ===== STAGE 2: 3-Way Temporal Split =====
    print_header("STAGE 2: Enforcing Strict 3-Way Temporal Isolation Split")
    train_df, calib_df, test_df = adapter.temporal_three_way_split(train_ratio=0.70, calib_ratio=0.15)

    X_train, y_train = adapter.get_feature_matrix(train_df)
    X_calib, y_calib = adapter.get_feature_matrix(calib_df)
    X_test, y_test = adapter.get_feature_matrix(test_df)

    print(f"  TRAIN (70%):       {len(train_df):>8,d} txs ({int(y_train.sum()):>4d} fraud, {int((y_train==0).sum()):>8,d} legit)")
    print(f"  CALIBRATION (15%): {len(calib_df):>8,d} txs ({int(y_calib.sum()):>4d} fraud, {int((y_calib==0).sum()):>8,d} legit)")
    print(f"  LOCKED TEST (15%): {len(test_df):>8,d} txs ({int(y_test.sum()):>4d} fraud, {int((y_test==0).sum()):>8,d} legit)")
    print(f"  Temporal order strictly preserved. Zero test contamination.")

    # Save split manifests
    with open(output_dir / "train_manifest.json", "w") as f:
        json.dump(dataset_manifest["splits"]["train"], f, indent=2)
    with open(output_dir / "calibration_manifest.json", "w") as f:
        json.dump(dataset_manifest["splits"]["calibration"], f, indent=2)
    with open(output_dir / "test_manifest.json", "w") as f:
        json.dump(dataset_manifest["splits"]["test"], f, indent=2)

    # ===== STAGE 3: Train & Calibrate Models =====
    print_header("STAGE 3: Training & Calibrating ML Models (TRAIN + CALIBRATION)")
    feature_names = adapter.get_feature_names()

    # 1. Isolation Forest (unsupervised)
    print("  Fitting Isolation Forest on normal TRAIN transactions...")
    X_train_normal = X_train[y_train == 0]
    if_scorer = IsolationForestScorer(n_estimators=300, contamination=0.002, seed=args.seed)
    if_scorer.fit(X_train_normal, feature_names=feature_names)
    print("  Calibrating Isolation Forest scores on CALIBRATION split...")
    if_scorer.calibrate_scores(X_calib)

    # 2. XGBoost (supervised)
    print("  Fitting XGBoost Classifier on TRAIN split...")
    xgb_scorer = XGBoostFraudScorer(n_estimators=300, max_depth=6, learning_rate=0.1, seed=args.seed)
    xgb_scorer.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib, feature_names=feature_names)

    # 3. Ensemble
    print("  Constructing Ensemble Scorer (w_if=0.3, w_xgb=0.7)...")
    ensemble = EnsembleFraudScorer(if_scorer=if_scorer, xgb_scorer=xgb_scorer, w_if=0.3, w_xgb=0.7)

    # Save model binaries with embedded hashes
    model_hashes = ensemble.save(models_dir)

    # ===== STAGE 4: Optimal Threshold Selection on CALIBRATION Split =====
    print_header("STAGE 4: Selecting Decision Threshold on CALIBRATION Split")
    calib_probas = ensemble.predict_ensemble_scores(X_calib)
    opt_search = find_optimal_threshold(y_calib, calib_probas, metric="f1")
    optimal_threshold = opt_search["optimal_threshold"]
    print(f"  Selected decision threshold on CALIBRATION split: {optimal_threshold:.2f} (F1 = {opt_search['optimal_score']:.4f})")

    # ===== STAGE 5: Single Pass Evaluation on LOCKED TEST Split =====
    print_header("STAGE 5: Evaluating Frozen Models on LOCKED TEST Split")

    test_probas = ensemble.predict_ensemble_scores(X_test)
    test_preds = (test_probas >= optimal_threshold).astype(np.int32)
    test_metrics = evaluate_binary_classifier(y_test, test_preds, test_probas)
    print_metrics(test_metrics, f"Ensemble Results on Locked Test Set (threshold={optimal_threshold:.2f})")

    # XGBoost alone on test set
    xgb_test_probas = xgb_scorer.predict_proba(X_test)
    xgb_opt_t = find_optimal_threshold(y_calib, xgb_scorer.predict_proba(X_calib), "f1")["optimal_threshold"]
    xgb_test_preds = (xgb_test_probas >= xgb_opt_t).astype(np.int32)
    xgb_test_metrics = evaluate_binary_classifier(y_test, xgb_test_preds, xgb_test_probas)
    print_metrics(xgb_test_metrics, f"XGBoost Alone Results on Locked Test Set (threshold={xgb_opt_t:.2f})")

    # ===== STAGE 6: Principled Feature & Model Group Ablation =====
    print_header("STAGE 6: Principled Feature & Model Group Ablation Study")
    ablation_results = run_principled_ablation(adapter, train_df, calib_df, test_df, seed=args.seed)

    print(f"\n  {'Variant':<20} {'Features':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Delta-F1':>10} {'AUC-PR':>8}")
    print(f"  {'-'*76}")
    for res in ablation_results:
        m = res["metrics"]
        auc_pr_str = f"{m.get('auc_pr', 0.0):.4f}" if m.get("auc_pr") else "N/A"
        print(f"  {res['variant_id']:<20} {res['n_features']:>8d} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f} {res['delta_f1']:>+10.4f} {auc_pr_str:>8}")

    # ===== STAGE 7: Calibration & ECE on LOCKED TEST Split =====
    print_header("STAGE 7: Calibration Curve & ECE on LOCKED TEST Split")
    calibration = compute_calibration_curve(y_test, test_probas, n_bins=10)
    print(f"  Expected Calibration Error (ECE): {calibration['ece']:.4f}")
    print(f"  Total test samples scored: {calibration['total_samples']:,d}")
    print(f"\n  {'Bin Range':>12} {'Count':>8} {'Predicted':>12} {'Observed':>12}")
    print(f"  {'-'*48}")
    for b in calibration["bins"]:
        if b["n_samples"] > 0:
            print(f"  [{b['bin_start']:.1f},{b['bin_end']:.1f}){b['n_samples']:>8d} {b['mean_predicted']:>12.4f} {b['mean_observed']:>12.4f}")
        else:
            print(f"  [{b['bin_start']:.1f},{b['bin_end']:.1f}){'0':>8} {'N/A':>12} {'N/A':>12}")

    # ===== STAGE 8: Bootstrap Confidence Intervals (N=1000) =====
    print_header("STAGE 8: Bootstrap Confidence Intervals (N=1000 Resamples)")
    bootstrap = run_bootstrap_ci(y_test, test_preds, test_probas, n_bootstrap=args.n_bootstrap, seed=args.seed)

    for metric_name in ["precision", "recall", "f1", "auc_roc"]:
        ci_data = bootstrap.get(metric_name)
        if ci_data:
            print(f"  {metric_name.upper():>12}: {ci_data['mean']:.4f} [95% CI: {ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]")

    # ===== STAGE 9: Build Canonical EXP-REALWORLD-CCF-001 Report =====
    print_header("STAGE 9: Generating Real-World Benchmark Report & Manifests")
    elapsed = time.time() - start_time

    report = {
        "experiment_id": "EXP-REALWORLD-CCF-001",
        "track": "realworld",
        "synthetic_baseline_unchanged": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed),
        "dataset_manifest": dataset_manifest,
        "models": {
            "isolation_forest": {
                "type": "IsolationForestScorer",
                "n_estimators": 300,
                "contamination": 0.002,
                "calibration": "min_max_percentile_calibrated_on_calib_split",
            },
            "xgboost": {
                "type": "XGBoostFraudScorer",
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "calibration": "platt_scaling_fitted_on_calib_split",
                "training_stats": xgb_scorer._training_stats,
                "metrics_test": xgb_test_metrics,
                "optimal_threshold_calib": float(xgb_opt_t),
            },
            "ensemble": {
                "type": "EnsembleFraudScorer",
                "w_if": 0.3,
                "w_xgb": 0.7,
                "optimal_threshold_calib": float(optimal_threshold),
                "metrics_test": test_metrics,
            },
        },
        "ablation": ablation_results,
        "calibration": calibration,
        "bootstrap_ci": bootstrap,
        "provenance": {
            "seed": args.seed,
            "model_hashes": model_hashes,
            "dataset_sha256": dataset_manifest["dataset_sha256"],
        },
    }

    report_json = json.dumps(report, indent=2, sort_keys=True, default=str)
    report_hash = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
    report["provenance"]["artifact_sha256"] = report_hash

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {report_path}")

    with open(output_dir / "ablation.json", "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)

    with open(output_dir / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2, default=str)

    with open(output_dir / "bootstrap.json", "w") as f:
        json.dump(bootstrap, f, indent=2, default=str)

    # ===== FINAL SUMMARY =====
    print_header("FINAL REAL-WORLD BENCHMARK SUMMARY (EXP-REALWORLD-CCF-001)")
    print(f"  Track:                 Real-World Public Benchmark Validation")
    print(f"  Dataset:               284,807 real-world transactions ({dataset_manifest['total_fraud']} fraud)")
    print(f"  LOCKED TEST Set:       {len(test_df):,d} transactions ({int(y_test.sum())} fraud)")
    print(f"")
    print(f"  Primary Scorer:        Calibrated Ensemble (IF + XGBoost)")
    print(f"    Precision:           {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.1f}%) [95% CI: {bootstrap['precision']['ci_lower']:.4f}, {bootstrap['precision']['ci_upper']:.4f}]")
    print(f"    Recall:              {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.1f}%) [95% CI: {bootstrap['recall']['ci_lower']:.4f}, {bootstrap['recall']['ci_upper']:.4f}]")
    print(f"    F1 Score:            {test_metrics['f1_score']:.4f} ({test_metrics['f1_score']*100:.1f}%) [95% CI: {bootstrap['f1']['ci_lower']:.4f}, {bootstrap['f1']['ci_upper']:.4f}]")
    print(f"    AUC-ROC:             {test_metrics.get('auc_roc', 'N/A')}")
    print(f"    AUC-PR:              {test_metrics.get('auc_pr', 'N/A')}")
    print(f"")
    print(f"  Calibration ECE:       {calibration['ece']:.4f} (Platt-scaled on CALIBRATION split)")
    print(f"  Artifact Directory:    {output_dir}")
    print(f"  Synthetic Track:       EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004 UNTOUCHED")

    return report


if __name__ == "__main__":
    main()
