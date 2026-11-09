"""Build once, then verify the same wheel in independent Python environments.

Run explicitly: this installs dependencies into the supplied temporary tool venv
and new venvs under --work-dir. It never changes the source interpreters' packages.
No publishing, GPU experiment, or external model/data download is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="append", required=True, help="source interpreter")
    parser.add_argument("--tools-python", required=True, help="existing temporary build-tool venv")
    parser.add_argument("--index-url", required=True, help="package index used for this check")
    parser.add_argument("--work-dir", type=Path, required=True, help="new evidence directory")
    args = parser.parse_args()
    root = args.work_dir.absolute()
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    env.pop("PYTHONPATH", None)

    def run(parts, name, cwd=root):
        print(f"{name}: {' '.join(map(str, parts))}", flush=True)
        with (root / f"{name}.log").open("w") as log:
            subprocess.run(
                parts, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    # A Conda interpreter may supply venv's Python, but packages go only into venvs.
    run(
        [args.tools_python, "-c", "import sys; assert sys.prefix != sys.base_prefix, 'Use a venv'"],
        "tools-environment",
    )
    run(
        [
            args.tools_python,
            "-m",
            "pip",
            "install",
            "--index-url",
            args.index_url,
            "hatchling>=1.27",
            "build",
            "twine",
        ],
        "build-dependencies",
    )
    dist = root / "dist"
    run([args.tools_python, "-m", "build", "--no-isolation", "--outdir", str(dist)], "build", REPO)
    run(
        [args.tools_python, "-m", "twine", "check", "--strict", *map(str, dist.iterdir())],
        "metadata",
    )
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    versions = []
    for source_python in args.python:
        version = subprocess.check_output(
            [
                source_python,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            text=True,
            env=env,
        ).strip()
        if version not in {"3.10", "3.11"} or version in versions:
            raise ValueError(f"Expected distinct Python 3.10/3.11 interpreters, got {version}")
        versions.append(version)
        venv = root / f"python-{version}"
        run([source_python, "-m", "venv", str(venv)], f"venv-{version}")
        python = str(venv / "bin/python")
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--index-url",
                args.index_url,
                f"{wheel}[dev,serve,reproduce]",
            ],
            f"install-{version}",
        )
        run([python, "-m", "pip", "check"], f"dependencies-{version}")
        run([python, "-m", "pip", "freeze"], f"packages-{version}")
        run(
            [
                python,
                str(REPO / "scripts/verify_installed.py"),
                "--work-dir",
                str(root / f"flow-{version}"),
            ],
            f"flow-{version}",
        )
        # Keep the package name: fixtures import ``tests.data`` / ``tests.modeling``.
        suite = root / f"suite-{version}"
        tests = suite / "tests"
        shutil.copytree(REPO / "tests", tests)
        shutil.copytree(REPO / "recipes", suite / "recipes")
        run([python, "-m", "pytest", "tests", "-q"], f"tests-{version}", suite)
    assert set(versions) == {"3.10", "3.11"}, "Both supported Python versions are required"
    proof = {
        "status": "passed",
        "python_versions": versions,
        "wheel": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
    }
    (root / "validation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"PASS: {root / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
