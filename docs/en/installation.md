# Installation

MetisCraft requires Python **3.10 or later**. Python 3.11 in a fresh virtual
environment is a suitable starting point. The examples below use a POSIX shell.
CPU PyTorch is sufficient for the quickstart; real-model training needs a PyTorch
build and hardware that support the chosen model.

## From a release-candidate wheel

`0.1.0rc1` is available as a [GitHub pre-release](https://github.com/Shichaoxx/MetisCraft/releases/tag/0.1.0rc1). GitHub hosts the distribution files for this version.

[Download wheel](https://github.com/Shichaoxx/MetisCraft/releases/download/0.1.0rc1/metiscraft-0.1.0rc1-py3-none-any.whl) · [Download source distribution](https://github.com/Shichaoxx/MetisCraft/releases/download/0.1.0rc1/metiscraft-0.1.0rc1.tar.gz) · [SHA256SUMS](https://github.com/Shichaoxx/MetisCraft/releases/download/0.1.0rc1/SHA256SUMS)

Install an appropriate PyTorch build in an isolated Python environment, then run these commands from the download directory. The `serve` extra provides HTTP dependencies.

```bash
sha256sum -c SHA256SUMS --ignore-missing
python -m pip install './metiscraft-0.1.0rc1-py3-none-any.whl[serve]'
metiscraft --version
```

## From source

Check out the fixed `0.1.0rc1` tag to build from the source used for this pre-release.

```bash
git clone --branch 0.1.0rc1 --depth 1 https://github.com/Shichaoxx/MetisCraft.git
cd MetisCraft
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[serve]'
metiscraft --version
```

Install an appropriate PyTorch build in this environment first if you need a
particular CPU or CUDA configuration. MetisCraft uses that environment's PyTorch.

Python 3.10 and 3.11 are the tested support range for this preview. The package's
minimum requirement does not establish support for later Python versions or every
dependency combination. See [compatibility and support](compatibility.md) for the
validated environments, interface boundaries and artifact-loading policy.

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
