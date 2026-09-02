"""Tests for YAML configuration loading and schema validation."""

from pathlib import Path
from src.contracts.config_loader import (
    load_detector_config,
    load_generator_config,
    load_evaluation_config,
)


def test_load_detector_config():
    config_path = Path(__file__).parent.parent / "config" / "detector.yaml"
    cfg = load_detector_config(config_path)
    assert cfg.version == "1.1.0"
    assert cfg.scorer.type in ["StaticThresholdScorer", "StatisticalDeviationScorer", "HybridEWMAScorer"]
    assert cfg.evidence.min_window_count >= 1


def test_load_generator_config():
    config_path = Path(__file__).parent.parent / "config" / "generator.yaml"
    cfg = load_generator_config(config_path)
    assert cfg.seed == 42
    assert len(cfg.merchants) == 9
    assert cfg.merchants[0].id == "M1"


def test_load_evaluation_config():
    config_path = Path(__file__).parent.parent / "config" / "evaluation.yaml"
    cfg = load_evaluation_config(config_path)
    assert cfg.horizons["volume_spike"] == 120
    assert cfg.cost_model.fp_review_cost == 50.0
