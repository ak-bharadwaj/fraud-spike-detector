"""YAML configuration loading with Pydantic validation."""

from pathlib import Path
from typing import Union
import yaml

from src.contracts.config_schemas import DetectorConfig, GeneratorConfig, EvaluationConfig


def load_detector_config(config_path: Union[str, Path]) -> DetectorConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return DetectorConfig.model_validate(data)


def load_generator_config(config_path: Union[str, Path]) -> GeneratorConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return GeneratorConfig.model_validate(data)


def load_evaluation_config(config_path: Union[str, Path]) -> EvaluationConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvaluationConfig.model_validate(data)
