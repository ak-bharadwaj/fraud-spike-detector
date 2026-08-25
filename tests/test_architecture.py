"""Architecture boundary and wall-clock enforcement tests.

Rules tested:
1. Physical module separation (Rule 6): Detection components must not import generator ground_truth code.
2. Virtual Clock enforcement (Rule 10): No wall-clock calls (datetime.now, datetime.utcnow, time.time) outside clock.py.
"""

import ast
from pathlib import Path


def get_python_files(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*.py") if p.is_file()]


def test_no_ground_truth_imports_in_detection():
    """Verify src/detector and other detection packages do not import generator ground_truth code."""
    src_path = Path(__file__).parent.parent / "src"
    detector_dirs = [
        src_path / "detector",
        src_path / "features",
        src_path / "baseline",
        src_path / "scoring",
        src_path / "state",
        src_path / "audit",
    ]

    for d in detector_dirs:
        if not d.exists():
            continue
        for file_path in get_python_files(d):
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "ground_truth" not in alias.name, (
                            f"Violation in {file_path}: Detector imports ground_truth ({alias.name})"
                        )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert "ground_truth" not in module, (
                        f"Violation in {file_path}: Detector imports from ground_truth ({module})"
                    )
                    for alias in node.names:
                        assert alias.name != "ground_truth" and "GroundTruth" not in alias.name, (
                            f"Violation in {file_path}: Detector imports ground_truth element ({alias.name})"
                        )


def test_no_wall_clock_outside_clock_module():
    """Verify datetime.now, datetime.utcnow, time.time do not occur in src/ outside clock.py."""
    src_path = Path(__file__).parent.parent / "src"
    allowed_clock_file = (src_path / "stream" / "clock.py").resolve()

    for file_path in get_python_files(src_path):
        if file_path.resolve() == allowed_clock_file:
            continue

        text = file_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr = node.attr
                if attr in ("now", "utcnow"):
                    # Check if called on datetime module/class
                    assert False, f"Violation in {file_path}: Wall-clock call 'datetime.{attr}' found!"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "time":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "time":
                        assert False, f"Violation in {file_path}: Wall-clock call 'time.time()' found!"
