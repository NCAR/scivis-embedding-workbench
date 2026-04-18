# Getting Started

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/NCAR/scivis-embedding-workbench.git
cd scivis-embedding-workbench
```

## 2. Install Dependencies

```bash
uv sync
```

This creates a local virtual environment and installs all dependencies from `pyproject.toml` and `uv.lock`.

## 3. Run a Notebook

Start a Marimo notebook server and open any notebook under the `notebooks/` directory:

```bash
uv run marimo edit notebooks/01-prepare-data/create_image_database.py
```

## 4. Run the Pipeline

Work through the notebooks in order:

1. `notebooks/00_data_processing/e5_batch.py` — generate JPEG composites from ERA5 NetCDF files
2. `notebooks/01-prepare-data/create_image_database.py` — ingest images into LanceDB
3. `notebooks/02-generate-embeddings/generate_dinov3_embeddings.py` — run embedding experiments
4. `notebooks/03-dashboard-app/app.py` — explore results in the interactive dashboard

---

## Running on NCAR Casper

To access a Marimo notebook running on a Casper compute node from your local browser:

### 1. Start Marimo on the Compute Node

From your active session on a Casper node (e.g. `casper39`), start the Marimo server with `--host 0.0.0.0` to allow the SSH tunnel to connect:

```bash
uv run marimo edit --host 0.0.0.0 --port 2718
```

Note the node ID (e.g. `casper39`), the port (`2718`), and the `access_token` in the URL Marimo prints.

### 2. Create an SSH Tunnel (Local Machine)

Open a new terminal on your laptop:

```bash
ssh -J <USER>@casper.hpc.ucar.edu -L 2718:localhost:2718 <USER>@<NODE_ID>.hpc.ucar.edu
```

Replace `<USER>` with your NCAR username and `<NODE_ID>` with the compute node (e.g. `casper39`).

### 3. Open in Browser

Paste the Marimo URL into your browser, making sure it starts with `localhost`:

```
http://localhost:2718?access_token=...
```

### Simplify with SSH Config

Add this to `~/.ssh/config` to shorten the tunnel command:

```
Host casper-gateway
    HostName casper.hpc.ucar.edu
    User <USER>

Host casper*
    HostName %h.hpc.ucar.edu
    User <USER>
    ProxyJump casper-gateway
```

Then the tunnel becomes:

```bash
ssh -L 2718:localhost:2718 casper39
```

!!! tip "Port conflicts"
    If port `2718` is already in use on your laptop, change the local port: `-L 9999:localhost:2718` and visit `localhost:9999`.
