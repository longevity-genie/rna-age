# Decisions

The calls settled with the maintainers, one section each, `## Dn — the call (date)`, newest last: what was decided,
why, and the alternative it rejected. A call is never edited after the fact; a later one that changes it says which
it amends.

## D1 — Seats: a runner and a lead, tracked, on genie's organisation (2026-11-10)

The repository adopts the seats kit (agent-seat-blueprints, v0.1.168), with the kit's adoption wizard run by
winternewt from a laptop. The answers, as given:

- **Maintainers:** antonkulaga, the author, and winternewt. Either one's word decides.
- **Where the seats run:** on the cluster's login node, where the pipeline's jobs are submitted. The laptop the
  wizard ran on only drafted: no leak gate, no site id and no chat account were set up on it.
- **The organisation and the desk:** genie's (Anton's Zulip organisation, the desk `longevity-genie/desk`, whose
  secretary makes the seats' accounts). Rejected: no organisation, which would leave the seats' questions in their
  sessions, unseen from a phone.
- **Seats:**
  - `runner` (`docs/seats/RUNNER.md`): keeps the pipeline's jobs running, logs every run in `docs/experiments.md`
    and `data/output/runs/`, keeps `STATUS.md` and the report current, and judges nothing a result means beyond its
    correctness gates. Grant: commit, `git pull --ff-only`, push `main` as a fast-forward, its own jobs within its
    share, the report's private page. Designed from the kit's seat draft for this repository. The overnight loop
    already working the repository on the cluster becomes the runner, in place.
  - `lead` (`docs/seats/LEAD.md`), added by winternewt during the wizard ("add lead also"): owns the plan, a new
    `docs/ROADMAP.md`, and this file, designs what is built next with the maintainers and builds it. Grant: commit,
    push `main` as a fast-forward, release tags. Taken by a new session on the cluster. Rejected: the plan kept in
    `STATUS.md`, shared with the runner (two seats writing one file, which the kit's plan watcher cannot read).
- **Tracked, not an overlay:** `.claude/settings.json` pins the kit and `.blueprints.toml` is committed, so every
  clone, and every collaborator's agent, loads the kit. Winternewt's call, made although every commit of the last 30
  days is antonkulaga's (his agent's loop): the seats change those sessions too. Going to an overlay later is
  `bin/overlay.py untrack`, a pull request.
- **The team file** is on the desk (`teams/rna-age.toml`), not here: the repository is public.
- **Cluster facts stay off the trunk:** the runner's share is a placeholder until the organisation's rigger proposes
  it and the maintainers decide; partitions, the account and nodes stay in the git-ignored `slurm/env.local.sh`.

## D2 — A KronLSTM expression encoder, Geneformer-style ranks first, bulk first (2026-11-11)

Anton (#rna-age, 2026-11-11 ~01:00): "kronlstm is fast to train so no needs to make it neccesary small. Also, in
geneformers there were other ideas, like they use scRNA-Seq and then trained transfromers on gene symbols ordered by
expresison ranks, can you run investigation (not neccessary in bulk but maybe in sc,, depends on what works) if we
can use kronlstm with similar approach". The lead's proposal before it (a small genome-ordered model first, its
bar, three attempts) was answered by this message; the lead settled the rest as its defaults:

- **Two input encodings, one model:** Geneformer's rank-value encoding (each sample's genes sorted by expression
  normalised by the gene's training-set median, the top 2,048 gene ids as tokens) and a genome-ordered encoding
  (the 20k HVGs along the genome, each token a gene embedding plus its log-CPM). A bidirectional KronLSTM
  (kron-fm's `kron_core`) reads either; a mean over tokens feeds an age head.
- **Bulk first**, because the labelled splits, the MLP reference and the evaluation exist there; single cells
  after, through the same rank tokenizer, if bulk says the encoder is worth it.
- **Supervised from scratch first, masked-gene pretraining on all ARCHS4 second**, the second only if the first
  is within reach of the bar: pretraining is days of GPU, and Geneformer's gain came from it.
- **The bar:** a single model beats the single MLP on 20k HVGs on validation by more than 0.5 y (run-to-run noise);
  three attempts with no gain above it end the line as a reported negative.
- **kron-fm as an optional dependency** (`uv sync --extra kron`), pinned to a commit: kron-fm is public, and the
  kernels need CUDA, so the base install stays without it.
- Rejected: a deliberately small model (Anton: no need); single cells first (no labelled split or reference yet).
