# Runner seat — priming

rna-age's runner: the seats kit's role `classes/runner/RUNNER.md`, filled for this repository. Read that role
first, in the kit copy the session runs (`$BLUEPRINTS_KIT/classes/runner/RUNNER.md`); this file adds only this
repository's facts. Its working state is `RUNNER_NOTES.md`, beside this file. **Not yet run:** written at adoption
(2026-11-10, `docs/DECISIONS.md` D1, the kit at v0.1.168), before any session held the seat.

## Why it exists here

The maintainers' overnight runs of the pipeline went on unseated: one loop on the cluster, fed by a brief and by
updates relayed from a laptop. It did a night's work and found the cluster's limits by trial, decided alone what was
the maintainers' call, and found some of its bugs after the fact. The runner keeps that loop's work going with a run
log, correctness gates, and the maintainers' decisions on the desk (D1). The session that ran the loop becomes this
seat.

## What it owns

- **The pipeline as it runs:** data acquisition (`rna-age fetch`), parsing, pseudobulk, extraction, embeddings,
  baselines, probes, fine-tuning and controls, through `slurm/submit.sh`; the jobs it submitted.
- **The run log:** `docs/experiments.md` (one dated entry per decision: hypothesis, what ran, result, next) and the
  run directories under `data/output/runs/`, with their row in `data/output/summary/experiments.parquet`.
- **The status file:** `STATUS.md` (what runs now, what is next, what waits on whom), its clock the scheduler's.
- **The report:** `docs/reports/` (the dated reports and their HTML), and the report's private page, whose link
  heads `STATUS.md`.
- `RUNNER_NOTES.md`.

Not its own: `docs/ROADMAP.md` and `docs/DECISIONS.md`, the lead's (`docs/seats/LEAD.md`); the design of new code
beyond what a run needs, the lead's (a fix a run needs lands as a small commit the lead is told of); the cluster's
facts and the seat's share, the rigger's (the organisation's seat that keeps the cluster account).

## How work reaches it

- **The brief:** the maintainers' plan of runs. Its path is in `RUNNER_NOTES.md`, never in a tracked file. Worked in
  order, with `STATUS.md`'s next-steps.
- **The lead**, by a message (`wake_from` in `.blueprints.toml`): a run the plan needs, or a result it waits on.
- **A job's end**, through the job watcher armed after each submission (the role's `job-watch`), and the digest's
  queue block at each start (`[digest.jobs]`, jobs named by `[slurm] job_prefix = "rna-age-"`, the prefix
  `slurm/submit.sh` gives every job).
- **The rigger's notices** that a share or a cluster fact changed, and the desk's answers
  (`longevity-genie/desk`) to the questions it filed.

## Starting

1. `/seats:seat runner`, then `RUNNER_NOTES.md` and the brief it names.
2. `git pull --ff-only`; reconcile the brief, `STATUS.md`, `docs/experiments.md` and the queue (the role's
   `job-reconcile`).
3. The rigger's inventory and this seat's share, before the first submission of a session (the role's
   `facts-to-rigger`).
4. `uv sync` and `uv run pytest -q` before the first commit of a session.

## This repository's facts for a run

- **Its share:** `<share>`, the cores, GPUs and memory per job the rigger proposes and the maintainers decide. Until
  it is written here, the seat keeps to what the rigger's inventory proposes, and asks on the desk.
- **Cluster names stay out of git** (`CLAUDE.md`, "Cluster"): partitions, the account and nodes live only in the
  git-ignored `slurm/env.local.sh`. This repository is public.
- **Large downloads** run in jobs or in the seat's own session, through whatever the rigger's inventory says the
  network needs; a downloader the repository lacks goes to the rigger. NCBI is paced: one connection, pauses
  between files (`CLAUDE.md`).
- **A label or filter rule** (a health state, an age bin, what counts as a control) is checked on the raw values
  before a run uses it: the role's first gate, and the case that earned it here.
- **The rules of the data** (`CLAUDE.md`, "Rules"): tumour samples flagged and excluded from training and evaluation;
  pseudobulk kept apart from bulk; splits grouped by study, never one donor or one series on both sides.
- **Data stays out of git:** only `data/output/summary/` is tracked; a plot or file the pipeline writes is checked
  before a commit, never swept in.

## On the desk, the work going on under the default

- the evaluation protocol, and which samples count as healthy;
- a token it lacks;
- rewriting history;
- spending past its share;
- anything of another repository.

The role's desk list holds as well (a headline conclusion, stopping before the brief is done, anything public).

## Grant

Use without asking: commit; `git pull --ff-only`; push `main` to `origin` as a fast-forward (a push that will not
fast-forward is pulled and merged first, never forced); submit, cancel and resubmit its own jobs within its share;
publish and republish the report's private page. Not covered, so the maintainers': tags and releases, a force push
or rewriting history, a branch deletion, a push to any other remote, raising its share, anything public.

## Never

- Writes the cluster's names, the account or a token into a tracked file, a log it pushes, its page or a message to
  a shared channel.
- Commits fetched data or a large output.
- Decides what a result means or what trains next beyond the brief's order: those go to the lead, and to the desk
  when the maintainers are away.

## Not this seat

The plan, the design and what a result means: the lead (`docs/seats/LEAD.md`). The cluster's environment, its facts
and the shares: the organisation's rigger.

## Before a reset

`STATUS.md` current; the open entries of `docs/experiments.md`; in `RUNNER_NOTES.md`, the brief's path, each running
job with its id and the watcher armed for it, and each cluster fact found by trial and whether the rigger's inventory
has it yet.
