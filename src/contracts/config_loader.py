"""YAML configuration loading with Pydantic validation and runtime freeze binding."""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import yaml
import json

from src.contracts.config_schemas import DetectorConfig, GeneratorConfig, EvaluationConfig
from src.contracts.contracts import FrozenDetectorConfig


def load_detector_config(config_path: Union[str, Path]) -> DetectorConfig:
    """Load and validate raw YAML detector configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return DetectorConfig.model_validate(data)


def load_generator_config(config_path: Union[str, Path]) -> GeneratorConfig:
    """Load and validate synthetic generator YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return GeneratorConfig.model_validate(data)


def load_evaluation_config(config_path: Union[str, Path]) -> EvaluationConfig:
    """Load and validate evaluation metrics YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvaluationConfig.model_validate(data)


def sync_detector_yaml_from_freeze_record(
    freeze_record_path: Union[str, Path] = "config/freeze_record.json",
    yaml_path: Union[str, Path] = "config/detector.yaml",
) -> None:
    """Synchronize config/detector.yaml to match the frozen parameters exactly."""
    with open(freeze_record_path, "r", encoding="utf-8") as f:
        f_data = json.load(f)
    params = f_data["all_selected_parameters"]

    y_data = {
        "version": params.get("detector_version", "1.0.0"),
        "scorer": {
            "type": params["scorer"],
            "alpha": params.get("alpha"),
            "persistence": params["persistence"],
            "static_threshold": params["static_threshold"],
        },
        "evidence": {
            "min_history_count": params.get("min_history_count", params.get("min_window_count", 1)),
            "min_window_count": params.get("min_window_count", 1),
        },
        "state_machine": {
            "cooldown_windows": params["cooldown_windows"],
        },
        "signal_weights": params.get(
            "signal_weights",
            {"volume": 1.0, "velocity": 1.0, "amount": 1.0, "behavioral": 1.0},
        ),
    }

    p = Path(yaml_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(y_data, f, sort_keys=False)


def load_runtime_frozen_config(
    freeze_record_path: Union[str, Path] = "config/freeze_record.json",
    yaml_path: Optional[Union[str, Path]] = "config/detector.yaml",
) -> FrozenDetectorConfig:
    """Load authoritative runtime detector configuration bound to canonical freeze record."""
    with open(freeze_record_path, "r", encoding="utf-8") as f:
        f_data = json.load(f)
    params = f_data["all_selected_parameters"]

    # If yaml_path is provided, verify bitwise/semantic synchronization
    if yaml_path:
        y_cfg = load_detector_config(yaml_path)
        if y_cfg.version != params.get("detector_version", "1.0.0"):
            raise ValueError(f"Detector YAML version '{y_cfg.version}' does not match freeze record version '{params.get('detector_version')}'")
        if y_cfg.scorer.type != params["scorer"]:
            raise ValueError(f"Detector YAML scorer '{y_cfg.scorer.type}' does not match freeze record scorer '{params['scorer']}'")
        if y_cfg.scorer.static_threshold != params["static_threshold"]:
            raise ValueError(f"Detector YAML threshold '{y_cfg.scorer.static_threshold}' does not match freeze record threshold '{params['static_threshold']}'")
        if y_cfg.scorer.persistence != params["persistence"]:
            raise ValueError(f"Detector YAML persistence '{y_cfg.scorer.persistence}' does not match freeze record persistence '{params['persistence']}'")
        if y_cfg.evidence.min_history_count != params.get("min_history_count", params.get("min_window_count", 1)):
            raise ValueError(f"Detector YAML min_history_count '{y_cfg.evidence.min_history_count}' does not match freeze record '{params.get('min_history_count')}'")
        if y_cfg.evidence.min_window_count != params.get("min_window_count", 1):
            raise ValueError(f"Detector YAML min_window_count '{y_cfg.evidence.min_window_count}' does not match freeze record '{params.get('min_window_count')}'")
        if y_cfg.state_machine.cooldown_windows != params["cooldown_windows"]:
            raise ValueError(f"Detector YAML cooldown_windows '{y_cfg.state_machine.cooldown_windows}' does not match freeze record '{params['cooldown_windows']}'")

    return FrozenDetectorConfig(
        scorer=params["scorer"],
        ewma_alpha=params.get("alpha"),
        static_threshold=float(params["static_threshold"]),
        persistence=int(params["persistence"]),
        cooldown_windows=int(params["cooldown_windows"]),
        min_history_count=int(params.get("min_history_count", params.get("min_window_count", 1))),
        min_window_count=int(params.get("min_window_count", 1)),
        signal_weights=params.get(
            "signal_weights",
            {"volume": 1.0, "velocity": 1.0, "amount": 1.0, "behavioral": 1.0},
        ),
        detector_version=params.get("detector_version", "1.0.0"),
    )
