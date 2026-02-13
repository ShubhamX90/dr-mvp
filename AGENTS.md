# AGENTS.md — Canonical Sharanga H100 Workflow for `dr-mvp`

These are the persistent instructions all coding agents must follow for this repository.

## Scope and Operating Model

- **No remote workspace usage on the cluster.** Do not assume VS Code Remote, IDE tunnels, or any coding agent running “inside” the cluster.
- All code changes must be delivered via **Git commits + pull requests**.
- Cluster execution is performed manually by humans via: **SSH → `git pull` → Slurm job submission**.

## Compute and Scheduling Policy (Sharanga)

- **Never run training/benchmarks on the login node.**
- All compute/GPU workloads must run through **Slurm**.
- Canonical GPU target:
  - `--partition=gpu_h100_4`
  - `--gres=gpu:1` by default
- Use more than 1 GPU **only when explicitly requested**.
- Always set `--time` (<= `24:00:00` on `gpu_h100_4`).
- Respect cluster policy constraints, including the partition’s **24-hour time limit**.

## Storage Policy

- Keep source code and small configs in the `$HOME` repository clone.
- Store large outputs under `./runs`.
  - On cluster, `./runs` maps to `/scratch/$USER/projects/dr-mvp/runs` (symlink).
- Do **not** write large artifacts into `$HOME`.
- Large artifacts include (non-exhaustive): large run directories, tensorboard event files, large log dumps, checkpoints, datasets, run artifacts.

## Python Environment (Single Canonical Env)

- Canonical conda env name: **`dr-mvp`**
- Python version: **3.10**
- Required setup method: **`ENV_SETUP.md`**
- Canonical install sequence:
  1. `conda create -n dr-mvp python=3.10 -y`
  2. `conda activate dr-mvp`
  3. `python -m pip install -U pip setuptools wheel`
  4. `python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
- Do **not** introduce conda-forge CUDA runtime stacks or CuPy unless explicitly requested.

## Planning Rule for Non-Trivial Changes

For any non-trivial change, first create or extend `PLANS.md` with:

1. **Goal**
2. **Files to change**
3. **Commands to run**
4. **Acceptance criteria**

Then implement **exactly** that plan.

## Pre-PR Validation (Required)

Before opening a PR, run and report:

1. `python -m compileall .`
2. `python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"`

## PR and Commit Hygiene

- Keep PRs small and reviewable.
- Prefer minimal diffs and clear commit messages.
- Never commit large artifacts (datasets, checkpoints, large run directories, tensorboard event files, large log dumps, run outputs).
- Never commit secrets (tokens, keys, passwords).
- Ensure ignores cover `runs/` and other large outputs **if needed**, but do not change `.gitignore` unless the task explicitly asks for it.
