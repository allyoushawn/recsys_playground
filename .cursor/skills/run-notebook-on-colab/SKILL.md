---
name: run-notebook-on-colab
description: >-
  Runs a Jupyter notebook on Google Colab via SSH + papermill from Cursor.
  Uses scp for fast file sync (no git round-trip), handles remote execution,
  and auto-fixes errors. Use when the user says "run notebook on Colab",
  "run notebook with Colab", "execute notebook on Colab", or
  "test notebook on Colab".
---

# Run Notebook on Colab

Executes a local Jupyter notebook on a Colab GPU runtime via SSH + papermill,
so the agent can see errors and fix them directly without manual copy-paste.
Uses `scp` to sync files directly — no git commit/push/pull needed.

## No git inside the notebook execution path (scp workflow)

**Why some notebooks still have git:** A common Colab pattern is `git clone` / `git pull` / `git reset --hard origin/main` so a human can open the notebook in the browser and always run the latest **GitHub** copy. That is fine for interactive-only use.

**Why it conflicts with this skill:** When Cursor **`scp`s** the notebook to Drive and runs **`papermill`**, the synced `.ipynb` is the **source of truth**. An early cell that runs **`git fetch` + `git reset --hard`** (or similar) **replaces** the repo on disk—including the notebook you just copied—with whatever is on the remote branch. That **undoes `scp`**, drops unpushed local edits, and can run an **older** committed version (e.g. missing vocab-cache code).

**Rule for agent-driven Colab runs:** For notebooks executed via this skill (and **experiment-runtime**), **do not rely on git to deliver the notebook or experiment code** for that run. Sync with **`scp` only**. The notebook should either:

- **Skip** repo-sync cells when running under papermill after an `scp` sync (e.g. env flag, or “if notebook mtime / marker says synced”), or  
- **Never** `reset --hard` / wholesale checkout over the path that contains the target `.ipynb`.

Preflight should catch destructive git-on-repo-root patterns; see [compatibility.md](compatibility.md).

## Prerequisites

- `cloudflared` and `sshpass` installed locally (`brew install cloudflared sshpass`)
- `notebooks/ad_hoc/colab_ssh_bootstrap.ipynb` exists in the repo
- `scripts/connect_colab.sh` exists in the repo

## Inputs

Gather from the user message or ask if missing:

| Input | Description | Example |
|-------|-------------|---------|
| Notebook path | Relative path to the .ipynb | `notebooks/ad_hoc/experiment_amazon_review_game.ipynb` |
| Colab hostname | **User provides** after manual bootstrap (Phase 2) | `loud-turkey-abc.trycloudflare.com` |

If the user gives a short name like `experiment_amazon_review_game`, resolve it
by searching `notebooks/` for a matching `.ipynb` file.

## Workflow

### Phase 1: Prepare the notebook

1. **Read the notebook** and check for papermill compatibility issues.
   See [compatibility.md](compatibility.md) for the full checklist.
2. **Fix any issues found** and tell the user what changed.
3. No commit/push needed — files are synced via `scp` in Phase 3.

### Phase 2: Connect to Colab (fully automated via trigger script)

**One-time setup (first use only):**
```bash
cd /Users/fox/Projects/CodexProjects/recsys_playground
python3 scripts/trigger_colab_bootstrap.py --setup
```
Opens Chrome to accounts.google.com. Sign in, then close Chrome. Session is saved to `~/.colab_chrome_profile/` and persists across runs.

**Normal run (fully automated):**
```bash
cd /Users/fox/Projects/CodexProjects/recsys_playground
HOSTNAME=$(python3 scripts/trigger_colab_bootstrap.py)
```
The script: launches Chrome with the saved session → opens the bootstrap notebook on GitHub → connects a GPU runtime → runs all cells → waits for the notebook to relay the hostname via ntfy.sh → prints the hostname to stdout.

**If the trigger script fails** (Chrome session expired, GPU queue timeout, etc.), fall back to manual:
1. Open the bootstrap notebook: `https://colab.research.google.com/github/allyoushawn/recsys_playground/blob/main/notebooks/ad_hoc/colab_ssh_bootstrap.ipynb`
2. Connect GPU runtime and Run All.
3. Paste the `*.trycloudflare.com` hostname from cell output.

Once `<HOSTNAME>` is obtained (automated or manual), verify connectivity:

```bash
sshpass -p "cursorssh" ssh <SSH_OPTS> root@<HOSTNAME> \
  "echo OK && nvidia-smi --query-gpu=name --format=csv,noheader"
```

If it fails, tell the user (runtime not started, hostname expired, etc.).

### Phase 3: Execute

1. **Copy notebook to Colab via scp:**
   ```bash
   sshpass -p "cursorssh" scp <SSH_OPTS> \
     <LOCAL_NOTEBOOK_PATH> \
     root@<HOSTNAME>:<REMOTE_REPO_ROOT>/<NOTEBOOK_PATH>
   ```

2. **Run notebook via papermill:**
   ```bash
   sshpass -p "cursorssh" ssh <SSH_OPTS> root@<HOSTNAME> \
     "cd <REMOTE_REPO_ROOT> && papermill <NOTEBOOK_PATH> <OUTPUT_PATH> --log-output 2>&1"
   ```
   Where `<OUTPUT_PATH>` replaces the filename suffix with `_output.ipynb`
   (e.g. `experiment_foo.ipynb` -> `experiment_foo_output.ipynb`).

   Run this as a background command (`block_until_ms: 0`) and poll the
   terminal output file until `exit_code` appears or 10 minutes elapse.

3. **If exit code 0:** Report results (metrics, training curves, etc.)
   extracted from the papermill log output.

4. **If exit code non-zero (error loop):**
   - Read the error traceback from the output
   - Open the notebook locally and fix the failing cell
   - `scp` the fixed notebook to Colab (step 1 above)
   - Re-run papermill (step 2 above)
   - Repeat up to 3 times; if still failing, report the error and ask the user

## SSH/SCP Options

Use this everywhere as `<SSH_OPTS>`:

```
-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
-o ProxyCommand="cloudflared access ssh --hostname <HOSTNAME>"
```

## Paths

| Variable | Value |
|----------|-------|
| `<REMOTE_REPO_ROOT>` | `/content/drive/MyDrive/colab/recsys_playground/recsys_playground` |
| Dataset cache | `/content/drive/MyDrive/colab/data/` |
| SSH password | `cursorssh` |
