"""Evaluation package exports."""

from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.calibration import DetectorCalibrator, CalibrationDataset
from src.evaluation.ablation import AblationRunner, load_characterization_data
from src.evaluation.plots import generate_precision_latency_tradeoff_plot
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
from src.evaluation.holdout_execution import (
    execute_single_pass_holdout,
    compute_per_anomaly_holdout_metrics,
    compute_descriptive_holdout_calibration,
    compute_bootstrap_uncertainty,
    execute_portfolio_comparison,
    save_day8_research_artifacts,
)

__all__ = [
    "AnomalyEvaluator",
    "DetectorCalibrator",
    "CalibrationDataset",
    "AblationRunner",
    "load_characterization_data",
    "generate_precision_latency_tradeoff_plot",
    "HoldoutManifest",
    "HoldoutProtection",
    "HoldoutEvaluator",
    "FrozenDetectorConfig",
    "HoldoutAccessError",
    "ChecksumMismatchError",
    "compute_holdout_dataset_hash",
    "load_locked_holdout_data",
    "execute_single_pass_holdout",
    "compute_per_anomaly_holdout_metrics",
    "compute_descriptive_holdout_calibration",
    "compute_bootstrap_uncertainty",
    "execute_portfolio_comparison",
    "save_day8_research_artifacts",
]
