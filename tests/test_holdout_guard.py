"""Tests for Holdout protection guard."""

import pytest
from src.evaluation.holdout import (
    HoldoutManifest,
    HoldoutProtection,
    HoldoutAccessError,
    ChecksumMismatchError,
)


def test_holdout_access_denied_in_dev_mode():
    manifest = HoldoutManifest(
        dataset_hash="abc123hash",
        generator_version="1.0.0",
        seed=42,
        schema_version="1.0.0",
        created_at="2026-08-25T00:00:00Z",
    )

    with pytest.raises(HoldoutAccessError):
        HoldoutProtection.verify_access(
            manifest=manifest,
            actual_dataset_hash="abc123hash",
            explicit_evaluation_mode=False,
        )


def test_holdout_checksum_mismatch():
    manifest = HoldoutManifest(
        dataset_hash="expected_hash",
        generator_version="1.0.0",
        seed=42,
        schema_version="1.0.0",
        created_at="2026-08-25T00:00:00Z",
    )

    with pytest.raises(ChecksumMismatchError):
        HoldoutProtection.verify_access(
            manifest=manifest,
            actual_dataset_hash="wrong_hash",
            explicit_evaluation_mode=True,
        )


def test_holdout_successful_access():
    manifest = HoldoutManifest(
        dataset_hash="valid_hash_123",
        generator_version="1.0.0",
        seed=42,
        schema_version="1.0.0",
        created_at="2026-08-25T00:00:00Z",
    )

    result = HoldoutProtection.verify_access(
        manifest=manifest,
        actual_dataset_hash="valid_hash_123",
        explicit_evaluation_mode=True,
    )
    assert result is True
