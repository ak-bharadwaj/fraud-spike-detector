"""Architecture boundary and wall-clock enforcement tests.

Rules tested:
1. Physical module separation (Rule 6): Detection components (src/detector, src/features,
   src/baseline, src/scoring, src/state, src/audit) must not import generator ground_truth code.
2. Virtual Clock enforcement (Rule 10): No wall-clock calls (datetime.now, datetime.utcnow, time.time)
   outside clock.py across any imported or aliased forms.
"""

import ast
from pathlib import Path
import pytest


def get_python_files(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*.py") if p.is_file()]


def get_full_attr_name(node: ast.AST) -> str:
    """Helper to extract full dot-separated name from AST Attribute / Name nodes."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        value_name = get_full_attr_name(node.value)
        return f"{value_name}.{node.attr}" if value_name else node.attr
    return ""


def check_wall_clock_in_ast(code: str, filename: str = "<string>") -> list[str]:
    """Analyze AST of Python code for forbidden wall-clock function calls.

    Precisely detects:
    - datetime.now(), datetime.utcnow(), datetime.datetime.now(), datetime.datetime.utcnow()
    - dt.now(), dt.utcnow() (aliased imports)
    - now(), utcnow() imported directly from datetime
    - time.time(), time.ctime(), time.localtime(), time.gmtime()
    - time() imported directly from time module
    """
    violations = []
    tree = ast.parse(code, filename=filename)

    # Track imported symbols
    datetime_modules = set()  # e.g. 'datetime'
    datetime_classes = set()  # e.g. 'datetime'
    time_modules = set()      # e.g. 'time'
    wall_clock_funcs = set()  # e.g. 'now', 'utcnow', 'time'

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name == "datetime":
                    datetime_modules.add(name)
                elif alias.name == "time":
                    time_modules.add(name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                local_name = alias.asname or alias.name
                if mod == "datetime":
                    if alias.name == "datetime":
                        datetime_classes.add(local_name)
                    elif alias.name in ("now", "utcnow"):
                        wall_clock_funcs.add(local_name)
                elif mod == "time":
                    if alias.name in ("time", "ctime", "localtime", "gmtime"):
                        wall_clock_funcs.add(local_name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                attr = func.attr
                full_name = get_full_attr_name(func)
                parts = full_name.split(".")
                prefixes = parts[:-1]

                if attr in ("now", "utcnow"):
                    if any(
                        p in datetime_modules or p in datetime_classes or p == "datetime"
                        for p in prefixes
                    ):
                        violations.append(f"Wall-clock call '{full_name}()' at line {node.lineno}")
                elif attr in ("time", "ctime", "localtime", "gmtime"):
                    if any(p in time_modules or p == "time" for p in prefixes):
                        violations.append(f"Wall-clock call '{full_name}()' at line {node.lineno}")

            elif isinstance(func, ast.Name):
                if func.id in wall_clock_funcs:
                    violations.append(f"Wall-clock call '{func.id}()' at line {node.lineno}")

    return violations


def test_no_ground_truth_imports_in_detection():
    """Verify src/detector and all detector sub-packages do not import generator ground_truth code."""
    src_path = Path(__file__).parent.parent / "src"
    detector_dirs = [
        src_path / "detector",
        src_path / "features",
        src_path / "baseline",
        src_path / "scoring",
        src_path / "state",
        src_path / "audit",
        src_path / "stream",
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
        violations = check_wall_clock_in_ast(text, filename=str(file_path))
        assert not violations, f"Wall-clock violation(s) in {file_path}: {violations}"


def test_wall_clock_ast_checker_detects_aliased_calls():
    """Unit test for the AST wall-clock checker itself to verify precise detection without false positives."""
    snippet_1 = "import datetime; t = datetime.datetime.now()"
    snippet_2 = "from datetime import datetime; t = datetime.now()"
    snippet_3 = "from datetime import datetime as dt; t = dt.utcnow()"
    snippet_4 = "import time; t = time.time()"
    snippet_5 = "from time import time; t = time()"

    assert len(check_wall_clock_in_ast(snippet_1)) > 0
    assert len(check_wall_clock_in_ast(snippet_2)) > 0
    assert len(check_wall_clock_in_ast(snippet_3)) > 0
    assert len(check_wall_clock_in_ast(snippet_4)) > 0
    assert len(check_wall_clock_in_ast(snippet_5)) > 0

    # Verify clean clock call is allowed
    clean_snippet = "from src.stream.clock import VirtualClock; clock = VirtualClock(); t = clock.current_time()"
    assert len(check_wall_clock_in_ast(clean_snippet)) == 0

    # Verify unrelated object method calls are not falsely flagged
    unrelated_snippet = "class CustomObject:\n  def now(self): pass\nobj = CustomObject(); obj.now()"
    assert len(check_wall_clock_in_ast(unrelated_snippet)) == 0
