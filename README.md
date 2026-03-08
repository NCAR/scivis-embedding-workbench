# bams-ai-data-exploration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NCAR/bams-ai-data-exploration/blob/main/notebooks/02-generate-embeddings/generate_dinov3_embeddings.ipynb)
[![Ruff Workflow](https://github.com/NCAR/bams-ai-data-exploration/actions/workflows/ruff.yml/badge.svg)](https://github.com/NCAR/bams-ai-data-exploration/actions/workflows/ruff.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Getting Started

These steps are intended for someone running the project locally for the first time.

### 1) Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### 2) Clone the repository

```bash
git clone https://github.com/NCAR/bams-ai-data-exploration.git
cd bams-ai-data-exploration
```

### 3) Install dependencies

```bash
uv sync
```

This will create or update a local virtual environment and install all dependencies from `pyproject.toml` and `uv.lock`.

### 4) Run notebooks

Start Jupyter and open any notebook under the `notebooks/` directory.

```bash
uv run jupyter notebook
```

### 5) Run the Python entry point (optional)

If you want to run the default Python script:

```bash
uv run python main.py
```

## Development Tips

- Keep dependencies updated with `uv lock` and `uv sync`.
- Run linting locally before opening a pull request:

```bash
uv run ruff check .
```
