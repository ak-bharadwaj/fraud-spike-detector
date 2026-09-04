"""ML-based fraud scoring strategies implementing AnomalyScorer ABC (Refined Architecture).

Provides three production-grade ML scoring strategies for the Real-World Benchmark Track:

1. IsolationForestScorer — Unsupervised anomaly detection with CALIBRATION-fitted score scaling
2. XGBoostFraudScorer — Supervised fraud classification with CALIBRATION-fitted Platt scaling
3. EnsembleFraudScorer — Weighted combination of calibrated IF anomaly + XGBoost probabilities

Experimental Isolation Invariants (Refinement #5 & #6):
- TRAIN split (70%): Used strictly to fit base models (IF and XGBoost).
- CALIBRATION split (15%): Used strictly to fit probability calibration (Platt scaling / sigmoid)
  and score normalization. Zero contamination of locked TEST split.
- LOCKED TEST split (15%): Evaluated in a single final pass. Zero hyperparameter tuning.
- Ensemble score combination: `P_ensemble = w_if * P_if_calibrated + w_xgb * P_xgb_calibrated`.
  Both inputs are normalized calibrated probabilities in [0, 1].
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.scoring.base import AnomalyScorer


class IsolationForestScorer(AnomalyScorer):
    """Unsupervised anomaly detection via Isolation Forest with CALIBRATION-fitted scaling.

    - Fits forest on normal TRAIN transactions.
    - Fits min-max / percentile normalization strictly on CALIBRATION split.
    - Outputs calibrated anomaly probabilities in [0, 1].
    """

    def __init__(
        self,
        n_estimators: int = 300,
        contamination: float = 0.002,
        static_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.static_threshold = float(static_threshold)
        self.seed = seed
        self.optimal_threshold: float = float(static_threshold)
        self._model: Optional[IsolationForest] = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._min_score: float = 0.0
        self._max_score: float = 1.0

    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "IsolationForestScorer":
        """Fit Isolation Forest on normal training data."""
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X_train)
        self._feature_names = list(feature_names) if feature_names else [f"f{i}" for i in range(X_train.shape[1])]
        self._is_fitted = True
        return self

    def calibrate_scores(self, X_calib: np.ndarray) -> "IsolationForestScorer":
        """Fit min-max score normalization strictly on CALIBRATION split."""
        if not self._is_fitted:
            raise RuntimeError("Must fit model before calibrating scores.")
        if X_calib is None or len(X_calib) == 0:
            raise RuntimeError("X_calib is required to calibrate Isolation Forest scores.")
        raw_scores = -self._model.decision_function(X_calib)
        self._min_score = float(np.percentile(raw_scores, 1))
        self._max_score = float(np.percentile(raw_scores, 99))
        if self._max_score <= self._min_score:
            self._max_score = self._min_score + 1.0
        return self

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated anomaly probabilities in [0, 1]."""
        if not self._is_fitted:
            raise RuntimeError("IsolationForestScorer not fitted. Call fit() first.")
        raw_scores = -self._model.decision_function(X)
        clipped = np.clip(raw_scores, self._min_score, self._max_score)
        calibrated = (clipped - self._min_score) / (self._max_score - self._min_score)
        return calibrated

    def predict_labels(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Return binary anomaly predictions."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        scores = self.predict_anomaly_scores(X)
        return (scores >= thresh).astype(np.int32)

    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
        signal_mask: Optional[Sequence[str]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> RiskScore:
        """Explicitly decoupled from Track A streaming semantics."""
        raise NotImplementedError(
            "IsolationForestScorer is a Track B real-world ML model operating on transaction "
            "feature matrices. It does not process aggregated FeatureSnapshot / BaselineSnapshot "
            "streaming events (Track A)."
        )

    def save(self, path: Union[str, Path]) -> str:
        """Serialize fitted model to disk and return SHA-256 hash."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "feature_names": self._feature_names,
            "min_score": self._min_score,
            "max_score": self._max_score,
            "optimal_threshold": self.optimal_threshold,
            "params": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "static_threshold": self.static_threshold,
                "seed": self.seed,
            },
        }, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IsolationForestScorer":
        """Deserialize fitted model from disk."""
        import joblib
        data = joblib.load(path)
        scorer = cls(**data["params"])
        scorer._model = data["model"]
        scorer._feature_names = data["feature_names"]
        scorer._min_score = data.get("min_score", 0.0)
        scorer._max_score = data.get("max_score", 1.0)
        scorer.optimal_threshold = float(data.get("optimal_threshold", scorer.static_threshold))
        scorer._is_fitted = True
        return scorer


class XGBoostFraudScorer(AnomalyScorer):
    """Supervised fraud classification using XGBoost with Platt-scaled calibration.

    - Fits XGBoost classifier on TRAIN split.
    - Fits Platt scaling (`CalibratedClassifierCV`) strictly on CALIBRATION split.
    - Zero data leakage into locked TEST split. No silent training-set fallbacks.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        static_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.static_threshold = float(static_threshold)
        self.optimal_threshold: float = float(static_threshold)
        self.seed = seed
        self._model = None
        self._calibrated_model = None
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._training_stats: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: Optional[np.ndarray] = None,
        y_calib: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostFraudScorer":
        """Train XGBoost on TRAIN split and calibrate strictly on CALIBRATION split."""
        import xgboost as xgb

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=self.seed,
            n_jobs=-1,
            tree_method="hist",
        )

        self._model.fit(X_train, y_train)
        self._feature_names = list(feature_names) if feature_names else [f"f{i}" for i in range(X_train.shape[1])]

        # Fit Platt-scaling calibrator strictly on CALIBRATION split (NO silent fallback)
        if X_calib is None or y_calib is None or len(y_calib) == 0:
            raise RuntimeError(
                "X_calib and y_calib are strictly required to fit Platt scaling calibration "
                "on the CALIBRATION split without training set leakage."
            )

        try:
            try:
                from sklearn.frozen import FrozenEstimator
                frozen = FrozenEstimator(self._model)
                self._calibrated_model = CalibratedClassifierCV(frozen, method="sigmoid")
                self._calibrated_model.fit(X_calib, y_calib)
            except (ImportError, ValueError, AttributeError):
                self._calibrated_model = CalibratedClassifierCV(self._model, cv="prefit", method="sigmoid")
                self._calibrated_model.fit(X_calib, y_calib)
        except Exception as err:
            raise RuntimeError(f"Failed to fit Platt scaling calibration on CALIBRATION split: {err}") from err

        self._is_fitted = True
        self._training_stats = {
            "n_train": len(y_train),
            "n_fraud_train": int(n_pos),
            "n_legit_train": int(n_neg),
            "scale_pos_weight": float(scale_pos_weight),
            "n_calib": len(y_calib),
            "n_fraud_calib": int(y_calib.sum()),
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return Platt-calibrated fraud probabilities."""
        if not self._is_fitted:
            raise RuntimeError("XGBoostFraudScorer not fitted. Call fit() first.")
        model = self._calibrated_model if self._calibrated_model else self._model
        return model.predict_proba(X)[:, 1]

    def predict_labels(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Return binary predictions using specified probability threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probas = self.predict_proba(X)
        return (probas >= thresh).astype(np.int32)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances from underlying XGBoost trees."""
        if not self._is_fitted or self._model is None:
            return {}
        importances = self._model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self._feature_names, importances)
        }

    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
        signal_mask: Optional[Sequence[str]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> RiskScore:
        """Explicitly decoupled from Track A streaming semantics."""
        raise NotImplementedError(
            "XGBoostFraudScorer is a Track B real-world ML model operating on transaction "
            "feature matrices. It does not process aggregated FeatureSnapshot / BaselineSnapshot "
            "streaming events (Track A)."
        )

    def save(self, path: Union[str, Path]) -> str:
        """Serialize fitted model to disk and return SHA-256 hash."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "calibrated_model": self._calibrated_model,
            "feature_names": self._feature_names,
            "training_stats": self._training_stats,
            "optimal_threshold": self.optimal_threshold,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "static_threshold": self.static_threshold,
                "seed": self.seed,
            },
        }, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBoostFraudScorer":
        """Deserialize fitted model from disk."""
        import joblib
        data = joblib.load(path)
        scorer = cls(**data["params"])
        scorer._model = data["model"]
        scorer._calibrated_model = data["calibrated_model"]
        scorer._feature_names = data["feature_names"]
        scorer._training_stats = data["training_stats"]
        scorer.optimal_threshold = float(data.get("optimal_threshold", scorer.static_threshold))
        scorer._is_fitted = True
        return scorer


class EnsembleFraudScorer(AnomalyScorer):
    """Weighted ensemble combining calibrated IF anomaly scores and XGBoost probabilities.

    Formula:
        P_ensemble = w_if * P_if_calibrated + w_xgb * P_xgb_calibrated
    """

    def __init__(
        self,
        if_scorer: Optional[IsolationForestScorer] = None,
        xgb_scorer: Optional[XGBoostFraudScorer] = None,
        w_if: float = 0.3,
        w_xgb: float = 0.7,
        static_threshold: float = 0.5,
    ):
        self.if_scorer = if_scorer
        self.xgb_scorer = xgb_scorer
        self.w_if = float(w_if)
        self.w_xgb = float(w_xgb)
        self.static_threshold = float(static_threshold)
        self.optimal_threshold: float = float(static_threshold)

    def predict_ensemble_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted ensemble probability scores on feature matrix."""
        scores = np.zeros(X.shape[0])

        if self.if_scorer and self.if_scorer._is_fitted:
            if_scores = self.if_scorer.predict_anomaly_scores(X)
            scores += self.w_if * if_scores

        if self.xgb_scorer and self.xgb_scorer._is_fitted:
            xgb_probas = self.xgb_scorer.predict_proba(X)
            scores += self.w_xgb * xgb_probas

        return np.clip(scores, 0.0, 1.0)

    def predict_labels(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary fraud predictions from ensemble scores."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        scores = self.predict_ensemble_scores(X)
        return (scores >= thresh).astype(np.int32)

    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
        signal_mask: Optional[Sequence[str]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> RiskScore:
        """Explicitly decoupled from Track A streaming semantics."""
        raise NotImplementedError(
            "EnsembleFraudScorer is a Track B real-world ML model operating on transaction "
            "feature matrices. It does not process aggregated FeatureSnapshot / BaselineSnapshot "
            "streaming events (Track A)."
        )

    def save(self, directory: Union[str, Path]) -> Dict[str, str]:
        """Save component models and write ensemble configuration manifest."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        hashes = {}
        if self.if_scorer:
            hashes["isolation_forest_sha256"] = self.if_scorer.save(d / "isolation_forest.joblib")
        if self.xgb_scorer:
            hashes["xgboost_sha256"] = self.xgb_scorer.save(d / "xgboost_fraud.joblib")

        config = {
            "w_if": self.w_if,
            "w_xgb": self.w_xgb,
            "static_threshold": self.static_threshold,
            "optimal_threshold": self.optimal_threshold,
            "model_hashes": hashes,
        }
        with open(d / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)
        return hashes

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "EnsembleFraudScorer":
        """Load ensemble from directory."""
        d = Path(directory)
        with open(d / "ensemble_config.json") as f:
            config = json.load(f)

        if_scorer = IsolationForestScorer.load(d / "isolation_forest.joblib") if (d / "isolation_forest.joblib").exists() else None
        xgb_scorer = XGBoostFraudScorer.load(d / "xgboost_fraud.joblib") if (d / "xgboost_fraud.joblib").exists() else None

        ensemble = cls(
            if_scorer=if_scorer,
            xgb_scorer=xgb_scorer,
            w_if=config["w_if"],
            w_xgb=config["w_xgb"],
            static_threshold=config["static_threshold"],
        )
        ensemble.optimal_threshold = float(config.get("optimal_threshold", ensemble.static_threshold))
        return ensemble


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    amounts: Optional[np.ndarray] = None,
    fp_unit_cost: float = 50.0,
    fn_exposure_factor: float = 800.0,
) -> Dict[str, Any]:
    """Compute binary classification metrics and financial impact.

    If `amounts` is provided, FN exposure is calculated dynamically as sum(Amount)
    of missed fraud transactions (y_true == 1 and y_pred == 0).
    Otherwise, fn_exposure_factor is used as a fallback.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    prec = precision_score(y_true, y_pred, zero_division=0.0)
    rec = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    fp_cost = float(fp * fp_unit_cost)
    if amounts is not None and len(amounts) == len(y_true):
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_exposure = float(np.sum(amounts[fn_mask]))
    else:
        fn_exposure = float(fn * fn_exposure_factor)

    result = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "fp_cost": fp_cost,
        "fn_exposure": fn_exposure,
        "total_cost": float(fp_cost + fn_exposure),
        "total_predictions": int(len(y_pred)),
        "total_positive_predictions": int(y_pred.sum()),
        "total_actual_positive": int(y_true.sum()),
    }

    if y_proba is not None:
        try:
            result["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            result["auc_roc"] = None
        try:
            result["auc_pr"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            result["auc_pr"] = None

    return result


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute calibration curve and Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_proba >= bins[i]) & (y_proba <= bins[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_stats.append({
                "bin_start": float(bins[i]),
                "bin_end": float(bins[i + 1]),
                "n_samples": 0,
                "mean_predicted": None,
                "mean_observed": None,
            })
            continue

        mean_predicted = float(y_proba[mask].mean())
        mean_observed = float(y_true[mask].mean())
        ece += (n_in_bin / total_samples) * abs(mean_predicted - mean_observed)

        bin_stats.append({
            "bin_start": float(bins[i]),
            "bin_end": float(bins[i + 1]),
            "n_samples": int(n_in_bin),
            "mean_predicted": mean_predicted,
            "mean_observed": mean_observed,
        })

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "total_samples": total_samples,
        "bins": bin_stats,
    }
