# SciVis Embeddings Workbench

[![Tests](https://github.com/NCAR/scivis-embedding-workbench/actions/workflows/tests.yml/badge.svg)](https://github.com/NCAR/scivis-embedding-workbench/actions/workflows/tests.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![codecov](https://codecov.io/gh/NCAR/scivis-embedding-workbench/graph/badge.svg)](https://codecov.io/gh/NCAR/scivis-embedding-workbench)
[![CodeQL](https://github.com/NCAR/scivis-embedding-workbench/actions/workflows/codeql.yml/badge.svg)](https://github.com/NCAR/scivis-embedding-workbench/actions/workflows/codeql.yml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://ncar.github.io/scivis-embedding-workbench/)

## Getting Started

These steps are intended for someone running the project locally for the first time.

### 1) Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### 2) Clone the repository

```bash
git clone https://github.com/NCAR/scivis-embedding-workbench.git
cd scivis-embedding-workbench
```

### 3) Install dependencies

```bash
uv sync
```

This creates or updates a local virtual environment and installs dependencies from `pyproject.toml`.

### 4) Run notebooks

All notebooks use [marimo](https://marimo.io). Run commands from the repo root.

**Edit a notebook interactively:**
```bash
uv run marimo edit notebooks/01-prepare-data/create_image_database.py
uv run marimo edit notebooks/02-generate-embeddings/generate_dinov3_embeddings.py
```

**Run the dashboard app:**
```bash
uv run marimo run notebooks/03-dashboard-app/app.py
```

### 5) Run the Python entry point (optional)

```bash
uv run python main.py
```

## What To Do Next

After your environment is working, choose a starting point based on what data you have:

**Option A — Quickest start (pre-computed embeddings):**
1. Download embeddings from [GDEX (d041308)](https://gdex.ucar.edu/datasets/d041308/).
2. Launch the dashboard: `uv run marimo run notebooks/03-dashboard-app/app.py`

**Option B — Start from raw ERA5 data:**
1. Download ERA5 data from [GDEX](https://gdex.ucar.edu).
2. Prepare the image database: `uv run marimo edit notebooks/01-prepare-data/create_image_database.py`
3. Generate embeddings: `uv run marimo edit notebooks/02-generate-embeddings/generate_dinov3_embeddings.py`
4. Launch the dashboard: `uv run marimo run notebooks/03-dashboard-app/app.py`

## How to Review and Contribute

### Review local changes

```bash
git status
git diff
```

### Run tests before opening a PR

```bash
uv run pytest tests/ -v
```

### Create a contribution branch

```bash
git checkout -b <short-feature-name>
```

### Commit and push your changes

```bash
git add .
git commit -m "Describe your change"
git push -u origin <short-feature-name>
```

### Open a pull request

1. Open your branch in GitHub.
2. Create a PR into `main`.
3. Include:
   - What changed
   - Why it changed
   - How you tested it

## Running Marimo on NCAR Casper

To access a Marimo notebook running on a Casper compute node from your local browser, follow these steps.

### 1. Start Marimo on the Compute Node

From your active session on a Casper node (e.g., `casper39`), start the Marimo server. You must use the `--host 0.0.0.0` flag to allow the SSH tunnel to connect.

```bash
uv run marimo edit --host 0.0.0.0 --port 2718
```

- **Note the Node ID:** Look at your terminal prompt (e.g., `user@casper39`).
- **Note the Port:** Default is `2718`.
- **Copy the Token:** Marimo will output a URL with an `access_token`.

### 2. Create an SSH Tunnel (Local Machine)

Open a new terminal window on your laptop and run the following command. This uses a "ProxyJump" to get through the NCAR login gateway to your specific compute node.

```bash
ssh -J <USER>@casper.hpc.ucar.edu -L 2718:localhost:2718 <USER>@<NODE_ID>.hpc.ucar.edu
```

- Replace `<USER>` with your NCAR username.
- Replace `<NODE_ID>` with the specific node you are on (e.g., `casper39`).

### 3. Open in Browser

Once the tunnel is established, copy the link provided by Marimo in Step 1 and paste it into your browser. Ensure the URL starts with `localhost`:

```
http://localhost:2718?access_token=...
```

### Optional: Simplify with SSH Config

To avoid typing long commands, add this to your local `~/.ssh/config` file:

```
Host casper-gateway
    HostName casper.hpc.ucar.edu
    User <USER>

Host casper*
    HostName %h.hpc.ucar.edu
    User <USER>
    ProxyJump casper-gateway
```

Now, the tunnel command becomes much shorter:

```bash
ssh -L 2718:localhost:2718 casper39
```

### Troubleshooting

- **Address already in use:** If port `2718` is taken on your laptop, change the first number: `-L 9999:localhost:2718`. You would then visit `localhost:9999` in your browser.
- **VPN:** Ensure you are connected to the NCAR VPN if working remotely.

## Development Tips

- Keep dependencies updated with `uv lock` and `uv sync`.
- Keep notebook and script changes focused so they are easier to review.
