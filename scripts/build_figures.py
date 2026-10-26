#!/usr/bin/env python3
"""Build editable TikZ figures as vector PDF and white-background PNG."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


FIGURES = (
    "metis-method-en",
    "metis-method-zh",
    "metis-architecture-en",
    "metis-architecture-zh",
    "metis-tensor-training",
    "metis-tensor-inference",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=FIGURES, action="append",
                        help="Build only this figure; repeat to select several")
    parser.add_argument("--engine", default="xelatex", help="XeLaTeX or Tectonic executable")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-directory", type=Path,
                        help="Defaults to docs/architecture/figures")
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("Use at least 150 DPI for readable figure exports")
    for executable in (args.engine, "pdftoppm"):
        if shutil.which(executable) is None:
            parser.error(f"Required executable not found: {executable}")
    sources = Path(__file__).resolve().parents[1] / "docs/architecture/figures"
    destination = (args.output_directory or sources).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="metis-figures-") as temporary:
        build = Path(temporary)
        for name in args.figure or FIGURES:
            if Path(args.engine).name == "tectonic":
                command = [args.engine, "-X", "compile", "--keep-logs",
                           f"--outdir={build}", f"{name}.tex"]
            else:
                command = [args.engine, "-halt-on-error", "-interaction=nonstopmode",
                           f"-output-directory={build}", f"{name}.tex"]
            result = subprocess.run(
                command,
                cwd=sources, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if result.returncode:
                print(result.stdout)
                return result.returncode
            log_path = build / f"{name}.log"
            diagnostics = result.stdout
            if log_path.is_file():
                diagnostics += log_path.read_text(encoding="utf-8", errors="replace")
            if "Missing character:" in diagnostics:
                print(f"Missing glyphs in {name}:\n{diagnostics}")
                return 1
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(args.dpi), "-singlefile",
                 str(build / f"{name}.pdf"), str(build / name)], check=True,
            )
            for suffix in ("pdf", "png"):
                shutil.copyfile(build / f"{name}.{suffix}", destination / f"{name}.{suffix}")
            print(f"Built {name}: PDF + PNG ({args.dpi} DPI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
