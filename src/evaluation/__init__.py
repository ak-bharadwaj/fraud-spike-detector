"""Evaluation package exports."""

from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.calibration import DetectorCalibrator, CalibrationDataset
from src.evaluation.ablation import AblationRunner, load_characterization_data
from src.evaluation.drift import DriftRunner, DriftManifest, load_drift_data
from src.evaluation.holdout import (
    HoldoutManifest,
    HoldoutProtection,
    HoldoutEvaluator,
    FrozenDetectorConfig,
    HoldoutAccessError,
    ChecksumMismatchError,
    compute_holdout_dataset_hash,
    load_locked_holdout_data,
)

__all__ = [
    "AnomalyEvaluator",
    "DetectorCalibrator",
    "CalibrationDataset",
    "AblationRunner",
    "load_characterization_data",
    "DriftRunner",
    "DriftManifest",
    "load_drift_data",
    "HoldoutManifest",
    "HoldoutProtection",
    "HoldoutEvaluator",
    "FrozenDetectorConfig",
    "HoldoutAccessError",
    "ChecksumMismatchError",
    "compute_holdout_dataset_hash",
    "load_locked_holdout_data",
]
