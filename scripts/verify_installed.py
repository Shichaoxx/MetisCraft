"""Exercise a wheel outside its checkout, without importing repository test helpers.

Run with the interpreter that installed the wheel. All generated files are retained
under --work-dir for inspection. A temporary HTTP process is always stopped on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    # Cover this process as well as children, before any ML imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    import metiscraft

    location = Path(metiscraft.__file__).resolve()
    if not location.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()):
        raise SystemExit(f"Expected an installed wheel in site-packages, got {location}")
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    env.pop("PYTHONPATH", None)

    def command(*parts, cwd=root, check=True):
        result = subprocess.run(
            [sys.executable, *parts],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
        )
        with (root / "commands.log").open("a") as stream:
            stream.write(f"\n$ python {' '.join(map(str, parts))}\n{result.stdout}{result.stderr}")
        if check and result.returncode:
            raise RuntimeError(f"{parts}: {result.stdout}\n{result.stderr}")
        return result

    def cli(*parts, **kwargs):
        return command("-m", "metiscraft", *parts, **kwargs)

    print(f"Installed package: {location}", flush=True)
    cli("init", "example")
    example = root / "example"
    assert cli("init", "example", check=False).returncode != 0
    print("Preparing local random model and synthetic inputs", flush=True)
    command("prepare.py", cwd=example)
    cli("data", "import", "dataset.yaml", "--data-root", "data", cwd=example)
    cli(
        "data",
        "verify",
        "quickstart",
        "--version",
        "synthetic-v1",
        "--data-root",
        "data",
        cwd=example,
    )
    print("Training, calibrating and evaluating on CPU", flush=True)
    cli("train", "recipe.yaml", cwd=example)
    runs = list((example / "runs" / "quickstart").iterdir())
    assert len(runs) == 1
    calibration = json.loads((runs[0] / "calibration" / "calibration.json").read_text())
    assert calibration["fitted_on"] == "calib"
    assert "quickstart|choice" in calibration["temperatures"]
    artifact = str(runs[0] / "artifact")
    env["METISCRAFT_DATA_ROOT"] = str(example / "data")
    cli("eval", artifact, "quickstart", "--version", "synthetic-v1", "--split", "test", cwd=example)

    request = {
        "state": "A card was stolen.",
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Charges", "tech": "Bugs"},
            },
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "severity": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["low", "medium", "high"],
            },
        },
    }
    from metiscraft.data.jev import request_to_records

    records = [
        r.model_dump(mode="json", exclude_none=True)
        for r in request_to_records(
            request,
            task_id="quickstart",
            request_id="sample",
        )
    ]
    records.append(
        {
            "id": "sample:topics",
            "task_id": "quickstart",
            "input": {"query": "Which topics?"},
            "question": {"type": "multi_label"},
            "candidates": [{"id": "cards"}, {"id": "payments"}],
        }
    )
    (example / "all-types.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (example / "http.jsonl").write_text(json.dumps(request) + "\n")
    batch = [
        json.loads(line)
        for line in cli(
            "predict",
            artifact,
            "all-types.jsonl",
            cwd=example,
        ).stdout.splitlines()
    ]
    assert {r["kind"] for r in batch} == {"choice", "noul", "score", "multi_label"}
    assert all(r["base_verification"] == "verified" for r in batch)
    choice = next(r for r in batch if r["kind"] == "choice")
    assert choice["calibration"]["group"] == "quickstart|choice"
    assert choice["calibration"]["temperature"] == calibration["temperatures"]["quickstart|choice"]
    offline_http = json.loads(cli("predict", artifact, "http.jsonl", cwd=example).stdout)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    print(f"Testing temporary HTTP service on 127.0.0.1:{port}; auto-stopped on exit", flush=True)
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "metiscraft", "serve", artifact, "--port", str(port)],
            cwd=example,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        health = json.load(r)
                    break
                except (OSError, urllib.error.URLError):
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Service failed to start; inspect server.log") from None
                    time.sleep(0.2)
            assert health["base_verification"] == "verified"
            assert health["calibrated"] is True
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/systemone",
                data=json.dumps(request).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                served = json.load(r)
            assert served == offline_http
            # Retain the exact public tutorial response and health from this installed wheel.
            tutorial = json.loads((example / "request.json").read_text())
            tutorial_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/systemone",
                data=json.dumps(tutorial).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(tutorial_request, timeout=60) as response:
                tutorial_response = json.load(response)
            (root / "tutorial-response.json").write_text(
                json.dumps(tutorial_response, indent=2, allow_nan=False) + "\n"
            )
            (root / "tutorial-health.json").write_text(
                json.dumps(health, indent=2, allow_nan=False) + "\n"
            )
            by_name = {r["id"].split(":", 1)[1]: r for r in batch}
            for name, answer in served["answers"].items():
                actual = (
                    [answer["noul"]]
                    if answer["type"] == "noul"
                    else list(answer["probabilities"].values())
                )
                assert all(
                    abs(a - b) < 1e-6
                    for a, b in zip(
                        actual,
                        by_name[name]["probabilities"],
                        strict=True,
                    )
                )
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    import torch

    from metiscraft.serve.predictor import Predictor

    relocated = root / "relocated-models"
    shutil.move(str(example / "models"), relocated)
    predictor = Predictor.load(
        artifact, base_path=relocated / "tiny-qwen3" / "v1", device=torch.device("cpu")
    )
    assert predictor.meta["base_verification"] == "verified"
    proof = {
        "version": metiscraft.__version__,
        "python": sys.version,
        "installed_from": str(location),
        "artifact": artifact,
        "checks": [
            "init_no_overwrite",
            "local_import",
            "dataset_integrity",
            "cpu_train",
            "calibration",
            "evaluation",
            "four_record_types",
            "three_http_types",
            "http_cli_record_agreement",
            "base_relocation",
        ],
        "status": "passed",
        "model_quality_claim": False,
    }
    (root / "validation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"PASS: {root / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
