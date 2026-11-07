"""Use the same read-only quality checks locally and in CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for command in (
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "pyright", "--pythonpath", sys.executable],
        [sys.executable, "scripts/check_examples.py"],
    ):
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
