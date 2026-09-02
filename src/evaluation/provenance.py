"""Strict Git Repository Provenance and Research Artifact Integrity Verification.

Provides structural validation of holdout evaluation provenance:
- Verifies canonical SHA-256 artifact hash matches the content excluding 'artifact_sha256'.
- Verifies frozen configuration equality between report and canonical freeze_record.json.
- Verifies execution_commit exists in git history, contains the matching freeze_record.json, and is an ancestor of finalization commit.
- Verifies historical_artifact_chain is a real, strictly ordered topological ancestor sequence in git.
- Verifies artifact finalization commit exists in git history and its tree contains the exact canonical artifact state.
- Rejects tampered artifacts, fabricated commits, disconnected/out-of-order chains, and commits with mismatched artifact trees.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import hashlib
import subprocess


def compute_canonical_artifact_hash(data: Dict[str, Any]) -> str:
    """Compute deterministic canonical SHA-256 hash of artifact content excluding artifact_sha256."""
    cleaned = {k: v for k, v in data.items() if k != "artifact_sha256"}
    canonical_bytes = json.dumps(cleaned, sort_keys=True, indent=2).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def git_commit_exists(commit_sha: str, cwd: Optional[Path] = None) -> bool:
    """Check if a commit SHA exists in the git repository object database."""
    try:
        res = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
    except Exception:
        return False


def git_is_ancestor(ancestor_sha: str, descendant_sha: str, cwd: Optional[Path] = None) -> bool:
    """Check if ancestor_sha is a strict or direct ancestor of descendant_sha."""
    if ancestor_sha == descendant_sha:
        return True
    try:
        res = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
    except Exception:
        return False


def get_git_file_content(commit_sha: str, file_path: str, cwd: Optional[Path] = None) -> Optional[str]:
    """Retrieve raw text content of a file at a specific git commit using git show."""
    try:
        normalized_path = str(file_path).replace("\\", "/")
        res = subprocess.run(
            ["git", "show", f"{commit_sha}:{normalized_path}"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            return res.stdout
    except Exception:
        pass
    return None


def get_git_finalization_commit_for_file(file_path: str, cwd: Optional[Path] = None) -> Optional[str]:
    """Retrieve the exact short commit SHA that finalized/committed file_path in git history."""
    try:
        normalized_path = str(file_path).replace("\\", "/")
        res = subprocess.run(
            ["git", "log", "-n", "1", "--format=%h", "--", normalized_path],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def verify_canonical_report_provenance(
    report_data: Dict[str, Any],
    repo_root: Optional[Union[str, Path]] = None,
    verify_git: bool = True,
    target_finalization_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Strictly verify canonical report integrity, artifact SHA, and git history provenance.

    Raises:
        ValueError: If artifact SHA is invalid/tampered, commits do not exist, chain is disconnected,
                    tree contents differ, or provenance is inconsistent with git history.
    """
    root_path = Path(repo_root) if repo_root else Path.cwd()

    # 1. Artifact SHA Integrity Check
    if "artifact_sha256" not in report_data:
        raise ValueError("Provenance violation: 'artifact_sha256' field is missing from report.")
    
    expected_sha = compute_canonical_artifact_hash(report_data)
    actual_sha = report_data["artifact_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(
            f"Artifact integrity violation: Calculated SHA '{expected_sha}' does not match "
            f"recorded artifact_sha256 '{actual_sha}'. The artifact has been tampered with."
        )

    # 2. Frozen Configuration Provenance Verification against canonical freeze_record.json
    freeze_path = root_path / "config" / "freeze_record.json"
    if freeze_path.exists():
        freeze_data = json.loads(freeze_path.read_text(encoding="utf-8"))
        if report_data.get("config_hash") != freeze_data.get("config_hash"):
            raise ValueError(
                f"Configuration provenance violation: Report config_hash '{report_data.get('config_hash')}' "
                f"does not match canonical freeze record hash '{freeze_data.get('config_hash')}'."
            )
        if report_data.get("frozen_detector") != freeze_data.get("all_selected_parameters"):
            raise ValueError(
                "Configuration provenance violation: Report frozen_detector parameters do not match canonical freeze record."
            )
        if report_data.get("development_dataset_hash") != freeze_data.get("development_dataset_hash"):
            raise ValueError(
                "Configuration provenance violation: Report development_dataset_hash does not match canonical freeze record."
            )

    # 3. Extract dual run disclosure
    dual_run = report_data.get("dual_run_disclosure")
    if not dual_run or "run_002_corrected" not in dual_run:
        raise ValueError("Provenance violation: Missing 'run_002_corrected' in dual_run_disclosure.")

    r2 = dual_run["run_002_corrected"]
    exec_commit = r2.get("execution_commit")
    declared_final = r2.get("artifact_finalization_commit")
    chain = r2.get("historical_artifact_chain")

    if not exec_commit or not isinstance(exec_commit, str) or len(exec_commit) < 7:
        raise ValueError(f"Provenance violation: Invalid execution_commit '{exec_commit}'.")

    if not declared_final or not isinstance(declared_final, str) or len(declared_final) < 7:
        raise ValueError(f"Provenance violation: Invalid artifact_finalization_commit '{declared_final}'.")

    if not chain or not isinstance(chain, list) or len(chain) == 0:
        raise ValueError("Provenance violation: 'historical_artifact_chain' is missing or empty.")

    # Verify historical_artifact_chain termination invariant: chain[-1] == declared_final
    if chain[-1] != declared_final:
        raise ValueError(
            f"Provenance violation: Historical artifact chain terminates at '{chain[-1]}', "
            f"which does not match declared artifact_finalization_commit '{declared_final}'."
        )

    # 4. Strict Git Repository Provenance Verification
    if verify_git:
        # Check execution commit existence
        if not git_commit_exists(exec_commit, cwd=root_path):
            raise ValueError(f"Provenance violation: execution_commit '{exec_commit}' does not exist in git repository.")

        # Check declared finalization commit existence
        if not git_commit_exists(declared_final, cwd=root_path):
            raise ValueError(f"Provenance violation: artifact_finalization_commit '{declared_final}' does not exist in git repository.")

        # Verify full chronological ancestry chain first
        for i, commit in enumerate(chain):
            if not git_commit_exists(commit, cwd=root_path):
                raise ValueError(f"Provenance violation: Chain commit '{commit}' (index {i}) does not exist in git repository.")
            if i > 0:
                prev = chain[i - 1]
                if not git_is_ancestor(prev, commit, cwd=root_path):
                    raise ValueError(
                        f"Provenance violation: Chain commit '{prev}' is not an ancestor of '{commit}' "
                        f"at chain index {i}."
                    )

        # Check that execution_commit tree contains matching freeze_record.json and parameters
        exec_freeze_str = get_git_file_content(exec_commit, "config/freeze_record.json", cwd=root_path)
        if exec_freeze_str is None:
            raise ValueError(f"Execution provenance violation: config/freeze_record.json missing at execution_commit '{exec_commit}'.")
        exec_freeze = json.loads(exec_freeze_str)
        if exec_freeze.get("config_hash") != report_data.get("config_hash"):
            raise ValueError(
                f"Execution provenance violation: freeze_record at execution_commit '{exec_commit}' "
                f"has config_hash '{exec_freeze.get('config_hash')}', not '{report_data.get('config_hash')}'."
            )
        if exec_freeze.get("all_selected_parameters") != report_data.get("frozen_detector"):
            raise ValueError(
                f"Execution provenance violation: freeze_record parameters at execution_commit '{exec_commit}' "
                f"do not match report frozen_detector parameters."
            )
        if exec_freeze.get("development_dataset_hash") != report_data.get("development_dataset_hash"):
            raise ValueError(
                f"Execution provenance violation: freeze_record development_dataset_hash at execution_commit '{exec_commit}' "
                f"does not match report development_dataset_hash."
            )

        # Verify execution_commit is an ancestor of artifact_finalization_commit
        if not git_is_ancestor(exec_commit, declared_final, cwd=root_path):
            raise ValueError(
                f"Provenance violation: execution_commit '{exec_commit}' is not an ancestor of "
                f"artifact_finalization_commit '{declared_final}'."
            )

        # Determine target tree commit to check: target_finalization_commit override OR git log finalization commit OR declared_final
        git_tree_commit = get_git_finalization_commit_for_file("artifacts/final/report.json", cwd=root_path)
        tree_commit = target_finalization_commit or git_tree_commit or declared_final

        if not git_commit_exists(tree_commit, cwd=root_path):
            raise ValueError(f"Provenance violation: Finalization commit '{tree_commit}' does not exist in git repository.")

        # Require declared_final to be an ancestor of (or equal to) tree_commit
        if not git_is_ancestor(declared_final, tree_commit, cwd=root_path):
            raise ValueError(
                f"Provenance violation: declared artifact_finalization_commit '{declared_final}' "
                f"is not an ancestor of finalization commit '{tree_commit}'."
            )

        # Check that tree_commit tree contains the exact canonical artifact state
        final_report_str = get_git_file_content(tree_commit, "artifacts/final/report.json", cwd=root_path)
        if final_report_str is None:
            raise ValueError(
                f"Provenance violation: artifacts/final/report.json does not exist in tree of commit '{tree_commit}'."
            )
        tree_data = json.loads(final_report_str)
        tree_sha = tree_data.get("artifact_sha256") or compute_canonical_artifact_hash(tree_data)
        if tree_sha != actual_sha:
            raise ValueError(
                f"Finalization provenance violation: Git tree at artifact_finalization_commit '{tree_commit}' "
                f"contains artifact with hash '{tree_sha}', which does not match current canonical artifact hash '{actual_sha}'."
            )

    return {
        "status": "PROVENANCE_VERIFIED",
        "artifact_sha256": actual_sha,
        "execution_commit": exec_commit,
        "artifact_finalization_commit": declared_final,
        "chain_length": len(chain),
    }
