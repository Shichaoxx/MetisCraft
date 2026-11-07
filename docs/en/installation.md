# Installation

MetisCraft requires Python **3.10 or later**. Python 3.11 in a fresh virtual
environment is a suitable starting point. The examples below use a POSIX shell.
CPU PyTorch is sufficient for the quickstart; real-model training needs a PyTorch
build and hardware that support the chosen model.

## From source

Install the current preview from the main branch. Check [GitHub releases](https://github.com/Shichaoxx/MetisCraft/releases) for published distributions:

```bash
git clone --branch main https://github.com/Shichaoxx/MetisCraft.git
cd MetisCraft
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[serve]'
metiscraft --version
```

Install an appropriate PyTorch build in this environment first if you need a
particular CPU or CUDA configuration. MetisCraft uses that environment's PyTorch.

## From a release-candidate wheel

If you have the built wheel, install it directly in your active environment:

```bash
python -m pip install './metiscraft-0.1.0rc1-py3-none-any.whl[serve]'
metiscraft --version
```

This is a local wheel installation, not a claim that `pip install metiscraft`
is available on PyPI. Check the [releases](https://github.com/Shichaoxx/MetisCraft/releases) for available versions.

## Optional dependencies

| Extra | Purpose |
|---|---|
| `serve` | FastAPI and Uvicorn for the HTTP service |
| `reproduce` | `datasets` and Jinja2 for external dataset builders, including Tev1 |
| `dev` | Tests, linting and development checks |

For development from the repository, use:

```bash
python -m pip install -e '.[dev,serve,reproduce]'
```

Installing the package does not download model weights or datasets. The
[quickstart](quickstart.md) creates its own small local inputs after installation.

[中文](../zh/installation.md)
