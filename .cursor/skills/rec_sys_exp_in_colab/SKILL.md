---
name: rec_sys_exp_in_colab
description: Given a dataset URL, produces a Colab notebook runnable on that dataset. Combines Colab bootstrap (Drive, work dir, clone repo) with experiment-runner workflow (download, preprocess, train MLP, report). Use when the user provides a dataset URL and wants a Colab notebook for experiments, recommendation experiments in Colab, or a runnable Colab notebook on a dataset.
---

# rec_sys_exp_in_colab

Produces a single Colab notebook that: (1) bootstraps Colab (Drive, work dir, clone repo), (2) installs dependencies, (3) downloads the dataset from the given URL, (4) auto-detects task type, preprocesses, trains an MLP, and outputs metrics and plots. The notebook is runnable in Colab with no extra setup beyond the dataset URL.

## When to Use

- User provides a dataset URL (direct, Kaggle, or HuggingFace) and wants a Colab notebook runnable on that dataset
- User wants a recommendation experiment or ML experiment in Colab from a dataset URL

## Inputs

| Input | Required | Default / source |
|-------|----------|------------------|
| Dataset URL (or `kaggle:owner/dataset`, `hf:dataset`) | Yes | From user |
| Notebook path | No | Ask user if not mentioned in query; else see "Where the notebook is created" |
| Repo URL | No | Current project remote (e.g. `https://github.com/.../recsys_playground.git`) |
| Repo directory | No | See "How project_name is determined" |
| Branch | No | `main` |
| Project name (for WORK_DIR and DATA_DIR) | No | Ask user if not mentioned in query; else see "How project_name is determined" |
| force_rewrite | No | False; if True, re-download even when data folder exists |

## Gathering

- **Notebook path**: If the user does not mention a notebook path in their query, ask them (e.g. "Where should I save the notebook?"). If they do mention a path, use it.
- **Project name**: If the user does not mention a project name in their query, ask them (e.g. "What project name should I use for the Colab work dir and data folder?"). If they do mention one, use it.

## Where the notebook is created

- **Default path** (when user does not specify): Under the current workspace root, in `notebooks/colab_experiments/`. Filename: `experiment_<slug>.ipynb` where `<slug>` is a short, filesystem-safe slug derived from the dataset URL. Create the directory if needed.
- **Override**: If the user provides a path, use it. Path is relative to the workspace root unless the user gives an absolute path.

## How project_name is determined

- **project_name** is used in the Colab notebook for: (1) `WORK_DIR = '/content/drive/MyDrive/colab/<project_name>'`, (2) `DATA_DIR = '/content/drive/MyDrive/colab/data/<project_name>'` (where the dataset is downloaded).
- **Default** (when user does not specify): Use the repo directory name (e.g. from git remote). For recsys_playground that is `recsys_playground`.
- **Override**: If the user says "use project name X" or "work dir X", use that for `project_name` in the bootstrap and config cells.

## Data download location and force_rewrite

- In the generated notebook, data is downloaded to **DATA_DIR** = `/content/drive/MyDrive/colab/data/<project_name>` (e.g. `/content/drive/MyDrive/colab/data/recsys_playground`).
- **Skip re-download**: If the folder DATA_DIR already exists and is non-empty, the notebook does not download again unless the user sets **force_rewrite = True**. Default is **force_rewrite = False**.
- The Config cell must define `force_rewrite = False` (or True if the user requested it). The Download cell must: (1) set DATA_DIR from PROJECT_NAME as above, (2) create DATA_DIR if needed, (3) if DATA_DIR exists and force_rewrite is False, skip download and use existing data; else run the download logic and save into DATA_DIR.

## Steps

1. **Resolve inputs** — Dataset URL (required). If notebook path or project name not in user query, ask. Resolve repo URL, repo dir, branch from current project or user.
2. **Build notebook** — Write a valid .ipynb. Use colab-notebook-bootstrap for cells 1–3; use experiment-runner for cells 4–11. Substitute gathered values. Do not duplicate the other skills' content; read and apply them.

## Notebook structure (generated .ipynb)

| Section | Cells | Source |
|---------|-------|--------|
| Bootstrap | 1–3 | Colab bootstrap templates (Drive, WORK_DIR, clone repo). **Read and apply colab-notebook-bootstrap.** |
| Env | 4 | `pip install` torch, pandas, numpy, scikit-learn, matplotlib, seaborn, kaggle, huggingface_hub, datasets, requests (from experiment-runner Dependencies; omit jupyter in Colab). |
| Config | 5 | `DATASET_URL`, `PROJECT_NAME`, `force_rewrite = False` (and optional `TASK_TYPE` override). |
| Download | 6 | DATA_DIR = `/content/drive/MyDrive/colab/data/<project_name>`; download only if DATA_DIR missing or `force_rewrite` True; else use existing. **Read and apply experiment-runner** for download logic (all sources: URL, Kaggle, HuggingFace). |
| Inspect | 7 | Load with pandas, shape/dtypes/head. |
| Task detection | 8 | Heuristic: categorical target → classification, numeric → regression, user-item-rating → ranking. |
| Preprocess | 9 | Missing values, encode categoricals, normalize, train/test split. |
| Model + train | 10 | MLP, task-appropriate loss and metrics (Accuracy/F1, MSE/R², NDCG/MRR). |
| Report | 11 | Plots (training curve, confusion/regression/ranking viz), metrics summary. |

## Dependency on other skills

- **Cells 1–3**: Read and apply [colab-notebook-bootstrap](~/.cursor/skills/colab-notebook-bootstrap/SKILL.md) for the bootstrap cell templates. Substitute `project_name`, `repo_url`, `repo_dir`, `branch_name` with gathered values.
- **Cells 4–11**: Read and apply [experiment-runner](~/.cursor/skills/experiment-runner/SKILL.md) for the download workflow, task types, metrics, preprocessing, model, and reporting. The generated notebook inlines the logic (no scripts in repo required); the agent produces that code by following the experiment-runner skill.

## Kaggle

For Kaggle datasets, the generated notebook must include a short comment or markdown that the user must upload `kaggle.json` to the Colab session (or set path in Drive) so the inline download can call the Kaggle API.
