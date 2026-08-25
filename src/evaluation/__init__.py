"""Evaluation package exports."""

from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.calibration import DetectorCalibrator, CalibrationDataset
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
    "HoldoutManifest",
    "HoldoutProtection",
    "HoldoutEvaluator",
    "FrozenDetectorConfig",
    "HoldoutAccessError",
    "ChecksumMismatchError",
    "compute_holdout_dataset_hash",
    "load_locked_holdout_data",
]
