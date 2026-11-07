# Contributing

MetisCraft is preparing 0.1.0rc1. Start with the [core documentation](docs/en/index.md)
and [current status](docs/status.md). Discuss substantial changes to public APIs,
judgment semantics or artifact formats before implementing them.

## Development setup

Work on a feature branch from `main`. In an isolated Python environment:

```bash
python -m pip install -e '.[dev,serve,reproduce]'
ruff check .
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest
```

Tests use tiny random models and constructed data on CPU. They do not require
downloaded weights or datasets. The `reproduce` extra is needed by tests for
external dataset builders. Run the [quickstart](docs/en/quickstart.md) when changing
the installed-package workflow, and verify packaged resources when changing templates
or the registry.

## Implementation conventions

- Keep judgment semantics in `judgments/`, metrics in `evaluate/`, and defaults
  that affect results in `config.py`. Recipes should state experimental settings.
- Keep training, calibration and evaluation roles separate. Never use
  `eval_only` labels to fit a model or calibrator.
- Bind artifacts to their verified base content. Historical artifacts must
  remain explicitly unverified when their original identity is unavailable.
- Preserve consistent task routing between Record and HTTP prediction, including
  observable calibration fallback behavior.
- Add focused regression tests for changed behavior. New readouts and judgment
  types need shape tests; equivalent implementations need numerical comparisons.

Users can import their own data without changing the registry. Adding a reusable
built-in dataset requires a pinned source revision, converter, license and terms
in `src/metiscraft/data/registry.yaml`. Do not submit labels derived from Jev or
TypeSafe API outputs.

## Documentation and review

Maintain paired core pages in `docs/en/` and `docs/zh/`; English is the default.
Use `src/metiscraft/templates/quickstart/README.md` as the command source for the
exported quickstart. Keep documented commands aligned with the actual CLI, and
distinguish software checks from evidence about model quality. Keep project
progress in `docs/status.md` and machine-specific information out of public files.

Write commit messages in English, starting with an imperative verb. A pull
request should explain the behavior change, relevant checks and material
limitations. Keep experimental claims tied to recorded runs. Release preparation
is described in [publishing](docs/publishing.md).

## License

The code is licensed under [Apache-2.0](LICENSE). By submitting a contribution,
you agree to license it under Apache-2.0. Dataset and model licenses remain separate.
