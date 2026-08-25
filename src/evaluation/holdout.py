"""Holdout protection skeleton.

Rule 26: Holdout has a manifest containing dataset hash, generator version, seed,
schema version, and generation metadata. Normal development execution cannot access holdout.
Explicit evaluation mode is required. Checksum mismatch aborts evaluation.
"""

from typing import Optional
from pydantic import BaseModel


class HoldoutManifest(BaseModel):
    dataset_hash: str
    generator_version: str
    seed: int
    schema_version: str
    created_at: str


class HoldoutAccessError(PermissionError):
    """Raised when holdout is accessed without explicit evaluation mode enabled."""

    pass


class ChecksumMismatchError(ValueError):
    """Raised when holdout dataset hash fails validation against the manifest."""

    pass


class HoldoutProtection:
    """Protection guard for holdout dataset access."""

    @staticmethod
    def verify_access(
        manifest: HoldoutManifest,
        actual_dataset_hash: str,
        explicit_evaluation_mode: bool = False,
    ) -> bool:
        """Verify explicit evaluation mode and dataset hash checksum.

        Raises HoldoutAccessError if explicit_evaluation_mode is False.
        Raises ChecksumMismatchError if actual_dataset_hash != manifest.dataset_hash.
        """
        if not explicit_evaluation_mode:
            raise HoldoutAccessError(
                "Holdout access denied: Normal development mode cannot access holdout data. "
                "Explicit evaluation mode flag is required."
            )

        if actual_dataset_hash != manifest.dataset_hash:
            raise ChecksumMismatchError(
                f"Holdout dataset checksum mismatch! Expected: {manifest.dataset_hash}, "
                f"Actual: {actual_dataset_hash}. Evaluation aborted."
            )

        return True
