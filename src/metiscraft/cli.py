"""The ``metiscraft`` command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from metiscraft import __version__


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    from metiscraft.config import load_recipe, resolve_paths

    recipe = load_recipe(args.recipe, args.overrides)
    recipe = recipe.model_copy(update={"paths": resolve_paths(recipe.paths)})
    print(json.dumps(recipe.to_dict(), indent=2, ensure_ascii=False))
    print(f"# sha256 {recipe.sha256()}", file=sys.stderr)
    return 0


def _data_root(args: argparse.Namespace):
    from pathlib import Path

    from metiscraft.config import Paths, require_path, resolve_paths

    explicit = Paths(data_root=Path(args.data_root)) if args.data_root else None
    return require_path(resolve_paths(explicit), "data_root")


def _cmd_data_list(args: argparse.Namespace) -> int:
    from metiscraft.data.registry import load_registry
    from metiscraft.data.store import list_versions

    root = None
    try:
        root = _data_root(args)
    except ValueError:
        pass
    print(f"{'name':24} {'tier':9} {'judgment':24} {'converter':16} prepared")
    for name, entry in load_registry().items():
        prepared = ", ".join(v.version for v in list_versions(root, name)) if root else "-"
        judgment = ",".join(entry.judgments)
        converter = entry.converter or "(v0.2)"
        print(f"{name:24} {entry.tier:9} {judgment:24} {converter:16} {prepared or '-'}")
    return 0


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    from metiscraft.data.prepare import prepare_dataset

    dv = prepare_dataset(args.name, _data_root(args), max_rows=args.max_rows)
    meta = dv.meta()
    print(f"{dv.name}@{dv.version} ready at {dv.root}")
    for role, info in meta["splits"].items():
        print(f"  {role:10} {info['rows']:>8} rows  from {info['from']}")
    return 0


def _cmd_data_verify(args: argparse.Namespace) -> int:
    from metiscraft.data.store import resolve_version

    dv = resolve_version(_data_root(args), args.name, args.version)
    count = dv.verify()
    print(f"{dv.name}@{dv.version}: {count} files verified")
    return 0


def _cmd_data_show(args: argparse.Namespace) -> int:
    from metiscraft.data.store import resolve_version

    dv = resolve_version(_data_root(args), args.name, args.version)
    for row in dv.load(args.split, limit=args.n):
        print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from metiscraft.config import load_recipe
    from metiscraft.pipeline import run_recipe

    run = run_recipe(load_recipe(args.recipe, args.overrides))
    print(f"run: {run}")
    report = run / "report.md"
    if report.is_file():
        print(f"report: {report}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from pathlib import Path

    from metiscraft.data.store import read_jsonl
    from metiscraft.serve.predictor import Predictor

    predictor = Predictor.load(args.artifact, calibration_dir=args.calibration)
    rows = list(read_jsonl(Path(args.input)))
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        requests = [r for r in rows if "questions" in r]
        records = [r for r in rows if "questions" not in r]
        for request in requests:
            out.write(json.dumps(predictor.answer_request(request), ensure_ascii=False) + "\n")
        for result in predictor.predict_records(records) if records else []:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from metiscraft.serve.predictor import Predictor
    from metiscraft.serve.server import serve

    serve(
        Predictor.load(args.artifact, calibration_dir=args.calibration),
        host=args.host,
        port=args.port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metiscraft",
        description="Train Jev-style decision models with calibrated, typed judgments.",
    )
    parser.add_argument("--version", action="version", version=f"metiscraft {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="print the package version")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser("config", help="inspect recipes")
    config_sub = p.add_subparsers(dest="config_command", required=True)
    show = config_sub.add_parser("show", help="print a recipe after overrides and path resolution")
    show.add_argument("recipe", help="recipe YAML file")
    show.add_argument("overrides", nargs="*", help="a.b=value overrides")
    show.set_defaults(func=_cmd_config_show)

    p = sub.add_parser("train", help="train, calibrate and evaluate one recipe")
    p.add_argument("recipe", help="recipe YAML file")
    p.add_argument("overrides", nargs="*", help="a.b=value overrides")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="answer records or official requests from a JSONL file")
    p.add_argument("artifact", help="artifact directory of a run")
    p.add_argument("input", help="JSONL of records or /v1/systemone requests")
    p.add_argument("--output", help="write results here instead of stdout")
    p.add_argument("--calibration", help="calibration directory (default: the run's)")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("serve", help="serve an artifact over HTTP (POST /v1/systemone)")
    p.add_argument("artifact", help="artifact directory of a run")
    p.add_argument("--calibration", help="calibration directory (default: the run's)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("data", help="registered datasets")
    data_sub = p.add_subparsers(dest="data_command", required=True)
    for cmd, func, help_text in (
        ("list", _cmd_data_list, "list registered datasets and prepared versions"),
        ("prepare", _cmd_data_prepare, "download, convert and write a dataset version"),
        ("verify", _cmd_data_verify, "check a prepared version against its SHA256SUMS"),
        ("show", _cmd_data_show, "print records from a prepared split"),
    ):
        q = data_sub.add_parser(cmd, help=help_text)
        q.add_argument("--data-root", help="data root (default: paths.data_root)")
        if cmd != "list":
            q.add_argument("name", help="registered dataset name")
        if cmd == "prepare":
            q.add_argument(
                "--max-rows", type=int, help="cap rows per upstream split (subset version)"
            )
        if cmd in ("verify", "show"):
            q.add_argument("--version", help="dataset version (default: the only one)")
        if cmd == "show":
            q.add_argument("--split", default="dev")
            q.add_argument("-n", type=int, default=2)
        q.set_defaults(func=func)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
