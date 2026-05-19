# casper-marimo

One-command launcher for Marimo on NCAR Casper. Submits a PBS GPU job, waits
for it to start, opens a local SSH tunnel, and launches your browser.

The launcher script and its template are tracked. Your filled-in
`casper-marimo.env` (with your project code and username) is **gitignored** —
see "Secrets" below.

## Auth model (no ssh config required)

This tool does **not** touch `~/.ssh/config`. Each invocation makes plain
`ssh <user>@casper.hpc.ucar.edu` calls, so you'll get a password + Duo push
prompt per ssh connection. The launcher is structured to keep the number of
connections as low as possible:

| Command                     | ssh connections | Duo pushes |
|-----------------------------|-----------------|------------|
| `casper-marimo` (`start`)   | 2               | 2          |
| `casper-marimo status`      | 0               | 0          |
| `casper-marimo status --remote` | 1           | 1          |
| `casper-marimo stop`        | 1               | 1          |

The two pushes during `start` are: (1) the orchestration connection that
uploads the PBS script, runs `qsub`, polls `qstat` until the node is
assigned, and reads the session info — all bundled into a single ssh+heredoc;
(2) the background `ssh -fN -L …` tunnel that forwards your local port to
the assigned compute node via the gateway.

## One-time setup

1. **Copy the env file and fill in your values:**

   ```sh
   cd scripts/casper
   cp casper-marimo.env.example casper-marimo.env
   $EDITOR casper-marimo.env
   ```

   You must set:
   - `PBS_ACCOUNT` — your NCAR project allocation code.
   - `REMOTE_PROJECT_DIR` — absolute path to this repo on Casper.
   - `GATEWAY_USER` — your NCAR username.

   The rest have sensible defaults.

2. **Add a shell alias** to `~/.zshrc`, using the absolute path to your local
   clone of this repo:

   ```sh
   # Adjust the path to wherever you cloned this repo
   REPO_PATH="$(pwd)/../.."  # if you're still in scripts/casper
   echo "alias casper-marimo=\"$REPO_PATH/scripts/casper/casper-marimo\"" >> ~/.zshrc
   source ~/.zshrc
   ```

That's it. No `~/.ssh/config` changes needed.

## Usage

```sh
casper-marimo            # start a session (prompts for resources, then 2 Duo pushes)
casper-marimo status     # local state only (no ssh)
casper-marimo status --remote   # also live-check PBS job state (1 ssh)
casper-marimo stop       # qdel the job + kill the local tunnel (1 ssh)
```

## What `start` does, end to end

```
casper-marimo
  → checks scripts/casper/casper-marimo.env
  → prompts: queue, account, gpu_type, ngpus, ncpus, mem, walltime
  → generates a one-shot marimo token
  → renders launch_marimo.pbs.tmpl with those values

  → ssh <user>@casper.hpc.ucar.edu bash -s <<REMOTE        # ← Duo push #1
       cat > ~/.marimo-launch.pbs <<PBS …
       qsub ~/.marimo-launch.pbs
       loop: qstat -f $jobid → wait for state=R + exec_host
       cat ~/.marimo-session.json
     REMOTE
  ← gets back: {node, port, token, jobid}

  → ssh -fN -L 2718:$node:2718 <user>@casper.hpc.ucar.edu   # ← Duo push #2
  → open http://localhost:2718/?access_token=…
```

If you close your terminal, the marimo session keeps running on Casper until
walltime expires or you `casper-marimo stop` from another terminal.

## Files

- `casper-marimo` — the Bash launcher (alias target). **Tracked.**
- `launch_marimo.pbs.tmpl` — PBS job script template. **Tracked.**
- `casper-marimo.env.example` — template for the local env file, placeholders only. **Tracked.**
- `casper-marimo.env` — your filled-in copy. **Gitignored** — never commit this.

## Secrets

The `casper-marimo.env` file in this directory holds your project code and
username. It is enforced as gitignored by a pytest guardrail
(`tests/test_casper_dir_not_committed.py`) that runs in CI:

- Asserts `casper-marimo.env` is never tracked.
- Asserts `casper-marimo.env.example` contains only `<PLACEHOLDER>` values
  for sensitive variables, so a copy-paste of the real `.env` over the
  example would fail CI.

If you ever need to share defaults with a teammate, update the example with
placeholders, never with real values.

## State files

The launcher keeps local state under `~/.casper-marimo/` (outside the repo):

- `session.json` — current session info (node, port, token, jobid).
- `tunnel.pid` — PID of the backgrounded SSH tunnel.
- `launch.pbs` — last rendered PBS script (debug).
- `remote.sh` — last remote shell script run via heredoc (debug).
