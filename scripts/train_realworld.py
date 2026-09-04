#!/usr/bin/env python
"""Real-World Public Benchmark Evaluation Pipeline (`EXP-REALWORLD-CCF-001`).

End-to-end training and evaluation pipeline for the ULB / Kaggle Credit Card Fraud benchmark:
1. Loads 284,807 transactions (492 fraud transactions across ~48 hours).
2. Applies strict 3-way temporal split: TRAIN (70%) -> CALIBRATION (15%) -> LOCKED TEST (15%).
3. Fits Isolation Forest (unsupervised) and XGBoost (supervised) on TRAIN split.
4. Fits Platt scaling calibration & IF score normalization strictly on CALIBRATION split.
5. Evaluated ONCE on LOCKED TEST split (zero hyperparameter tuning).
6. Runs 6 principled feature/model group ablations:
   - FULL_ENSEMBLE: IF + XGBoost on all 30 features
   - XGB_ONLY: XGBoost alone on all features
   - IF_ONLY: Isolation Forest alone on all features
   - PCA_ONLY: All 28 PCA dimensions (V1-V28), omitting Amount and Time
   - PCA_PLUS_AMOUNT: All 28 PCA dimensions + Amount
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
    print(f"  FN Exposure (sum Amount):     Rs.{metrics['fn_exposure']:>10.2f}")
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
    """Run 6 principled feature/model group ablations on the locked TEST set.

    Ablation variants:
    1. FULL_ENSEMBLE: IF + XGBoost ensemble on all features
    2. XGB_ONLY: XGBoost alone on all features
    3. IF_ONLY: Isolation Forest alone on all features
    4. PCA_ONLY: V1-V28 features only (omitting Time & Amount)
    5. PCA_PLUS_AMOUNT: V1-V28 + Amount
    6. AMOUNT_TIME_ONLY: Time & Amount features only (omitting V1-V28 PCA)
    """
    ablation_results = []

    def evaluate_variant(variant_id: str, description: str, feature_subset: str, use_xgb: bool, use_if: bool):
        X_tr, y_tr = adapter.get_feature_matrix(train_df, feature_subset)
        X_ca, y_ca = adapter.get_feature_matrix(calib_df, feature_subset)
        X_te, y_te = adapter.get_feature_matrix(test_df, feature_subset)
        test_amounts = test_df["Amount"].values
        feature_names = adapter.get_feature_names(feature_subset)

        xgb_s = None
        if_s = None

        if use_xgb:
            xgb_s = XGBoostFraudScorer(n_estimators=300, max_depth=6, learning_rate=0.1, seed=seed)
            xgb_s.fit(X_tr, y_tr, X_calib=X_ca, y_calib=y_ca, feature_names=feature_names)

        if use_if:
            X_tr_norm = X_tr[y_tr == 0]
            if_s = IsolationForestScorer(n_estimators=300, contamination=0.002, seed=seed)
            if_s.fit(X_tr_norm, feature_names=feature_names)
            if_s.calibrate_scores(X_ca)

        if use_xgb and use_if:
            ens = EnsembleFraudScorer(if_scorer=if_s, xgb_scorer=xgb_s, w_if=0.3, w_xgb=0.7)
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

        metrics = evaluate_binary_classifier(y_te, te_preds, te_probas, amounts=test_amounts)
        return {
            "variant_id": variant_id,
            "description": description,
            "feature_subset": feature_subset,
            "n_features": X_tr.shape[1],
            "optimal_threshold_calib": float(opt_t),
            "metrics": metrics,
        }

    print("    Evaluating variant 1/6: FULL_ENSEMBLE (IF + XGB on all 30 features)...")
    res_full = evaluate_variant("FULL_ENSEMBLE", "Full ensemble (IF + XGBoost on all features)", "all", True, True)
    full_f1 = res_full["metrics"]["f1_score"]

    res_full["delta_f1"] = 0.0
    res_full["delta_precision"] = 0.0
    res_full["delta_recall"] = 0.0
    ablation_results.append(res_full)

    print("    Evaluating variant 2/6: XGB_ONLY (XGBoost alone on all features)...")
    res_xgb = evaluate_variant("XGB_ONLY", "XGBoost alone (all features)", "all", True, False)
    res_xgb["delta_f1"] = res_xgb["metrics"]["f1_score"] - full_f1
    res_xgb["delta_precision"] = res_xgb["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_xgb["delta_recall"] = res_xgb["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_xgb)

    print("    Evaluating variant 3/6: IF_ONLY (Isolation Forest alone on all features)...")
    res_if = evaluate_variant("IF_ONLY", "Isolation Forest alone (all features)", "all", False, True)
    res_if["delta_f1"] = res_if["metrics"]["f1_score"] - full_f1
    res_if["delta_precision"] = res_if["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_if["delta_recall"] = res_if["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_if)

    print("    Evaluating variant 4/6: PCA_ONLY (V1-V28 PCA dimensions only)...")
    res_pca = evaluate_variant("PCA_ONLY", "PCA features V1-V28 only (omitting Time & Amount)", "pca", True, True)
    res_pca["delta_f1"] = res_pca["metrics"]["f1_score"] - full_f1
    res_pca["delta_precision"] = res_pca["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_pca["delta_recall"] = res_pca["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_pca)

    print("    Evaluating variant 5/6: PCA_PLUS_AMOUNT (V1-V28 PCA dimensions + Amount)...")
    res_pca_amt = evaluate_variant("PCA_PLUS_AMOUNT", "PCA features V1-V28 plus Amount", "pca_plus_amount", True, True)
    res_pca_amt["delta_f1"] = res_pca_amt["metrics"]["f1_score"] - full_f1
    res_pca_amt["delta_precision"] = res_pca_amt["metrics"]["precision"] - res_full["metrics"]["precision"]
    res_pca_amt["delta_recall"] = res_pca_amt["metrics"]["recall"] - res_full["metrics"]["recall"]
    ablation_results.append(res_pca_amt)

    print("    Evaluating variant 6/6: AMOUNT_TIME_ONLY (Time & Amount features only)...")
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
    test_amounts = test_df["Amount"].values

    print(f"  TRAIN (70%):       {len(train_df):>8,d} txs ({int(y_train.sum()):>4d} fraud, {int((y_train==0).sum()):>8,d} legit)")
    print(f"  CALIBRATION (15%): {len(calib_df):>8,d} txs ({int(y_calib.sum()):>4d} fraud, {int((y_calib==0).sum()):>8,d} legit)")
    print(f"  LOCKED TEST (15%): {len(test_df):>8,d} txs ({int(y_test.sum()):>4d} fraud, {int((y_test==0).sum()):>8,d} legit)")
    print(f"  Temporal order strictly preserved. Zero test contamination.")

    # Compute canonical configuration hash and common provenance header (§39 compliance)
    config_payload = {
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
        "if": {"n_estimators": 300, "contamination": 0.002},
        "ensemble": {"w_if": 0.3, "w_xgb": 0.7},
        "feature_groups": ML_FEATURE_GROUPS,
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode("utf-8")).hexdigest()

    common_provenance = {
        "experiment_id": "EXP-REALWORLD-CCF-001",
        "dataset_hash": dataset_manifest["dataset_sha256"],
        "config_hash": config_hash,
        "detector_version": "realworld-xgb-v1",
        "seed": args.seed,
    }

    # Write provenance-compliant split manifests (§39 compliance)
    for split_key, file_name in [("train", "train_manifest.json"), ("calibration", "calibration_manifest.json"), ("test", "test_manifest.json")]:
        split_data = {
            **common_provenance,
            "artifact_type": "split_manifest",
            "split": split_key.upper() if split_key != "test" else "LOCKED_TEST",
            **dataset_manifest["splits"][split_key],
        }
        with open(output_dir / file_name, "w") as f:
            json.dump(split_data, f, indent=2)

    # ===== STAGE 3: Train & Calibrate Models =====
    print_header("STAGE 3: Training & Calibrating ML Models (TRAIN + CALIBRATION)")
    feature_names = adapter.get_feature_names()

    # 1. Isolation Forest (unsupervised comparator)
    print("  Fitting Isolation Forest on normal TRAIN transactions...")
    X_train_normal = X_train[y_train == 0]
    if_scorer = IsolationForestScorer(n_estimators=300, contamination=0.002, seed=args.seed)
    if_scorer.fit(X_train_normal, feature_names=feature_names)
    print("  Calibrating Isolation Forest scores on CALIBRATION split...")
    if_scorer.calibrate_scores(X_calib)

    # 2. XGBoost (primary supervised model)
    print("  Fitting XGBoost Classifier on TRAIN split...")
    xgb_scorer = XGBoostFraudScorer(n_estimators=300, max_depth=6, learning_rate=0.1, seed=args.seed)
    xgb_scorer.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib, feature_names=feature_names)

    # 3. Ensemble (optional comparative model)
    print("  Constructing Ensemble Scorer (w_if=0.3, w_xgb=0.7)...")
    ensemble = EnsembleFraudScorer(if_scorer=if_scorer, xgb_scorer=xgb_scorer, w_if=0.3, w_xgb=0.7)

    # ===== STAGE 4: Optimal Threshold Selection on CALIBRATION Split =====
    print_header("STAGE 4: Selecting Decision Threshold on CALIBRATION Split")

    # Headline model: XGBoost threshold selection
    xgb_calib_probas = xgb_scorer.predict_proba(X_calib)
    xgb_opt_search = find_optimal_threshold(y_calib, xgb_calib_probas, metric="f1")
    xgb_opt_threshold = xgb_opt_search["optimal_threshold"]
    xgb_scorer.optimal_threshold = xgb_opt_threshold
    print(f"  [Primary] XGBoost optimal threshold on CALIBRATION split: {xgb_opt_threshold:.2f} (F1 = {xgb_opt_search['optimal_score']:.4f})")

    # Isolation Forest threshold selection
    if_calib_scores = if_scorer.predict_anomaly_scores(X_calib)
    if_opt_search = find_optimal_threshold(y_calib, if_calib_scores, metric="f1")
    if_opt_threshold = if_opt_search["optimal_threshold"]
    if_scorer.optimal_threshold = if_opt_threshold

    # Ensemble threshold selection
    ens_calib_probas = ensemble.predict_ensemble_scores(X_calib)
    ens_opt_search = find_optimal_threshold(y_calib, ens_calib_probas, metric="f1")
    ens_opt_threshold = ens_opt_search["optimal_threshold"]
    ensemble.optimal_threshold = ens_opt_threshold

    # Save model binaries with locked operating thresholds
    model_hashes = ensemble.save(models_dir)

    # ===== STAGE 5: Single Pass Evaluation on LOCKED TEST Split =====
    print_header("STAGE 5: Evaluating Frozen Models on LOCKED TEST Split")

    # XGBoost (Primary Headline Model)
    xgb_test_probas = xgb_scorer.predict_proba(X_test)
    xgb_test_preds = xgb_scorer.predict_labels(X_test, threshold=xgb_opt_threshold)
    xgb_test_metrics = evaluate_binary_classifier(y_test, xgb_test_preds, xgb_test_probas, amounts=test_amounts)
    print_metrics(xgb_test_metrics, f"Primary XGBoost Results on Locked Test Set (threshold={xgb_opt_threshold:.2f})")

    # Isolation Forest (Unsupervised Comparator)
    if_test_scores = if_scorer.predict_anomaly_scores(X_test)
    if_test_preds = if_scorer.predict_labels(X_test, threshold=if_opt_threshold)
    if_test_metrics = evaluate_binary_classifier(y_test, if_test_preds, if_test_scores, amounts=test_amounts)

    # Ensemble (Comparative Model)
    ens_test_probas = ensemble.predict_ensemble_scores(X_test)
    ens_test_preds = ensemble.predict_labels(X_test, threshold=ens_opt_threshold)
    ens_test_metrics = evaluate_binary_classifier(y_test, ens_test_preds, ens_test_probas, amounts=test_amounts)
    print_metrics(ens_test_metrics, f"Ensemble Results on Locked Test Set (threshold={ens_opt_threshold:.2f})")

    # ===== STAGE 6: Principled Feature & Model Group Ablation =====
    print_header("STAGE 6: Principled Feature & Model Group Ablation Study")
    ablation_results = run_principled_ablation(adapter, train_df, calib_df, test_df, seed=args.seed)

    print(f"\n  {'Variant':<20} {'Features':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Delta-F1':>10} {'AUC-PR':>8}")
    print(f"  {'-'*76}")
    for res in ablation_results:
        m = res["metrics"]
        auc_pr_str = f"{m.get('auc_pr', 0.0):.4f}" if m.get("auc_pr") else "N/A"
        print(f"  {res['variant_id']:<20} {res['n_features']:>8d} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f} {res['delta_f1']:>+10.4f} {auc_pr_str:>8}")

    # ===== STAGE 7: Calibration & ECE on LOCKED TEST Split (Primary Model) =====
    print_header("STAGE 7: Calibration Curve & ECE on LOCKED TEST Split (Primary XGBoost)")
    calibration = compute_calibration_curve(y_test, xgb_test_probas, n_bins=10)
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
    print_header("STAGE 8: Bootstrap Confidence Intervals (Primary XGBoost, N=1000 Resamples)")
    bootstrap = run_bootstrap_ci(y_test, xgb_test_preds, xgb_test_probas, n_bootstrap=args.n_bootstrap, seed=args.seed)

    for metric_name in ["precision", "recall", "f1", "auc_roc"]:
        ci_data = bootstrap.get(metric_name)
        if ci_data:
            print(f"  {metric_name.upper():>12}: {ci_data['mean']:.4f} [95% CI: {ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]")

    # ===== STAGE 9: Build Canonical EXP-REALWORLD-CCF-001 Report =====
    print_header("STAGE 9: Generating Real-World Benchmark Report & Manifests")

    report = {
        "experiment_id": "EXP-REALWORLD-CCF-001",
        "track": "realworld",
        "synthetic_baseline_unchanged": True,
        "dataset_manifest": dataset_manifest,
        "models": {
            "primary_xgboost": {
                "type": "XGBoostFraudScorer",
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "calibration": "platt_scaling_fitted_on_calib_split",
                "training_stats": xgb_scorer._training_stats,
                "metrics_test": xgb_test_metrics,
                "optimal_threshold_calib": float(xgb_opt_threshold),
            },
            "isolation_forest": {
                "type": "IsolationForestScorer",
                "n_estimators": 300,
                "contamination": 0.002,
                "calibration": "min_max_percentile_calibrated_on_calib_split",
                "metrics_test": if_test_metrics,
                "optimal_threshold_calib": float(if_opt_threshold),
            },
            "ensemble": {
                "type": "EnsembleFraudScorer",
                "w_if": 0.3,
                "w_xgb": 0.7,
                "optimal_threshold_calib": float(ens_opt_threshold),
                "metrics_test": ens_test_metrics,
            },
        },
        "ablation": ablation_results,
        "calibration": calibration,
        "bootstrap_ci": bootstrap,
        "provenance": {
            **common_provenance,
            "model_hashes": model_hashes,
            "locked_threshold": float(xgb_opt_threshold),
            "calibration_method": "Platt scaling (sigmoid on CALIBRATION split)",
        },
    }

    # Compute deterministic SHA-256 hash over canonical evaluation payload (zero wall-clock dependence)
    report_json_pre = json.dumps(report, indent=2, sort_keys=True, default=str)
    report_hash = hashlib.sha256(report_json_pre.encode("utf-8")).hexdigest()
    report["provenance"]["artifact_sha256"] = report_hash

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {report_path}")

    # Write §39 provenance-compliant standalone artifacts
    ablation_artifact = {
        **common_provenance,
        "artifact_type": "ablation",
        "data_split": "LOCKED_TEST",
        "ablation": ablation_results,
    }
    with open(output_dir / "ablation.json", "w") as f:
        json.dump(ablation_artifact, f, indent=2, default=str)

    calibration_artifact = {
        **common_provenance,
        "artifact_type": "calibration",
        "data_split": "LOCKED_TEST",
        "calibration": calibration,
    }
    with open(output_dir / "calibration.json", "w") as f:
        json.dump(calibration_artifact, f, indent=2, default=str)

    bootstrap_artifact = {
        **common_provenance,
        "artifact_type": "bootstrap_ci",
        "data_split": "LOCKED_TEST",
        "bootstrap_ci": bootstrap,
    }
    with open(output_dir / "bootstrap.json", "w") as f:
        json.dump(bootstrap_artifact, f, indent=2, default=str)

    # ===== FINAL SUMMARY =====
    print_header("FINAL REAL-WORLD BENCHMARK SUMMARY (EXP-REALWORLD-CCF-001)")
    print(f"  Track:                 Real-World Public Benchmark Validation")
    print(f"  Dataset:               284,807 real-world transactions ({dataset_manifest['total_fraud']} fraud)")
    print(f"  LOCKED TEST Set:       {len(test_df):,d} transactions ({int(y_test.sum())} fraud)")
    print(f"")
    print(f"  Primary Model:         XGBoost (Platt-Calibrated)")
    print(f"    Precision:           {xgb_test_metrics['precision']:.4f} ({xgb_test_metrics['precision']*100:.1f}%) [95% CI: {bootstrap['precision']['ci_lower']:.4f}, {bootstrap['precision']['ci_upper']:.4f}]")
    print(f"    Recall:              {xgb_test_metrics['recall']:.4f} ({xgb_test_metrics['recall']*100:.1f}%) [95% CI: {bootstrap['recall']['ci_lower']:.4f}, {bootstrap['recall']['ci_upper']:.4f}]")
    print(f"    F1 Score:            {xgb_test_metrics['f1_score']:.4f} ({xgb_test_metrics['f1_score']*100:.1f}%) [95% CI: {bootstrap['f1']['ci_lower']:.4f}, {bootstrap['f1']['ci_upper']:.4f}]")
    print(f"    AUC-ROC:             {xgb_test_metrics.get('auc_roc', 'N/A')}")
    print(f"    AUC-PR:              {xgb_test_metrics.get('auc_pr', 'N/A')}")
    print(f"    FN Exposure (sum):   Rs.{xgb_test_metrics['fn_exposure']:.2f}")
    print(f"    Total Cost:          Rs.{xgb_test_metrics['total_cost']:.2f}")
    print(f"")
    print(f"  Calibration ECE:       {calibration['ece']:.4f} (Platt-scaled on CALIBRATION split)")
    print(f"  Artifact Directory:    {output_dir}")
    print(f"  Synthetic Track:       EXP-DAY9-HOLDOUT-CORRECTED-CONFIDENCE-004 UNTOUCHED")

    return report


if __name__ == "__main__":
    main()
