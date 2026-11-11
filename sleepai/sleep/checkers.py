"""Checkers grade dream attempts. Deterministic checkers are preferred; a teacher grade
is used only when none exists, and those examples are marked weaker."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

from ..models.base import answers_match


def exact(predicted: str, expected: str) -> bool:
    return answers_match(predicted, expected)


def numeric(predicted: str, expected: float, tol: float = 1e-6) -> bool:
    nums = re.findall(r"-?\d+(?:\.\d+)?", predicted.replace(",", ""))
    return bool(nums) and abs(float(nums[-1]) - float(expected)) <= tol * max(1.0, abs(float(expected)))


def python_tests(solution_code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Run model-written code against fixed tests in a separate process with a timeout,
    an empty temp directory, and no inherited environment. For stronger isolation on your
    machine, run this inside a Docker container with networking disabled."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "attempt.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(solution_code + "\n\n" + test_code + "\n")
        env = {"PATH": "/usr/bin:/bin"}
        if os.name == "nt":            # Windows: Python needs SYSTEMROOT to start
            env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"), "PATH": ""}
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True,
                               text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]
