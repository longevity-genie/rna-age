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

## D3 — D2 amended after its critic: leakage, the bar, ablations, pretraining not conditional (2026-11-11)

A fresh critic read D2 alone (the lead's step before building); its must-fixes, as the lead settled them (Anton away;
each reversible, listed for his review):

- **No leakage:** every tokenizer statistic (the per-gene medians of the rank encoding, the HVGs, the value scaling)
  is computed on training samples only; masked-gene pretraining reads only training-split studies, never a
  validation or test study, although ARCHS4 holds them.
- **The bar is paired and by study:** three seeds of each KronLSTM arm against three seeds of the 20k-HVG MLP
  trained on all health states (same samples, same splits); the arm passes only if its mean validation MAE is lower
  and the 95 % CI of the difference, bootstrapped over validation studies, excludes 0. Reported per source (GTEx,
  ARCHS4, Bgee) and on pseudobulk transfer, never only pooled. Test is read once, for the arm that passed. Replaces
  D2's fixed 0.5 y margin and "three attempts".
- **The encoder is separated from the encoding:** an MLP on the same rank encoding (each gene's rank score, 0 when
  outside the top L), and the genome-ordered arm against the same genes in a shuffled order. A result is the
  KronLSTM's only where it beats both its own ablation and the MLP.
- **The context length is swept**, 512 / 2,048 / 4,096 tokens on the rank arm, one seed each, before the three-seed
  runs; parameter counts and epochs are recorded with every run.
- **Pretraining is not conditional:** a masked-gene pretraining probe on training studies runs whatever the
  from-scratch result, since Geneformer's gain came from pretraining; a from-scratch failure is reported as
  "from scratch", never as the line's end. It is its own plan item, after the from-scratch code lands.
- Single cells stay out of this item: the bulk medians do not carry to cells (mostly zero); a cell tokenizer needs
  its own statistics (non-zero medians, as Geneformer) and is planned only on a bulk result.

## D4 — A plate single-cell series is excluded whole; meningioma is a tumour (2026-11-11)

Found by the lead while building RM10 and RM11 (Anton away; both follow the repository's rules, the threshold is a
constant the lead picked, listed for his review):

- **Single cells:** ARCHS4's per-sample single-cell probability (> 0.5 excluded) misses plate-based protocols,
  whose cells score just under it: GSE116672 (viscRNA-seq, 1,161 cells of 10 dengue patients, median 0.45) was
  11 % of the exact-age test set; GSE152197 (1,215, 32 donors) and GSE249746 (545, 3 donors) were in training. A
  series whose samples' median probability exceeds **0.4** is now excluded whole: 169 series with then-eligible
  samples, 5,308 samples (3,866 train, 132 val, 1,310 test). Rejected: 0.3 (39 more series, 866 samples, more
  borderline low-input bulk among them); a samples-per-donor rule (donor ids are mostly missing).
- **Tumours:** meningioma (471 eligible samples, 5 studies) and other benign or CNS tumour names join the tumour
  flag; the no-cancer rule already required it.
- Both take effect at one re-parse of ARCHS4 and a re-merge (RM12). Splits are hashed per study, so no remaining
  study changes side. Every result before RM12 includes these samples; `kron-compare` and `robustness` read only
  currently eligible samples, and the headline is re-scored on the new table before it is quoted again.
