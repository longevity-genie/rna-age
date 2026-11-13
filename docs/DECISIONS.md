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

## D5 — Ablations dropped; KronLSTM from scratch is behind (2026-11-11)

Anton (#rna-age, 12:41): "do not resume ablation, we have more important stuff to do than numerous ablation
experiments". The rank-MLP ablation of D3 is cancelled; D3's other ablation (the shuffled genome order) is dropped
with it. The from-scratch context sweep ended at validation MAE 13.95 y (2,048 tokens; 512: 14.41; 4,096 stopped at
epoch 3, 14.52) against ~8.7 for the 20k-HVG MLP, so the KronLSTM line is behind by ~5 y without telling encoding
from model apart. What comes next for it (one pretraining run, or stop) is asked of Anton; until he answers, no new
KronLSTM run starts.

## D6 — The reported health clock: the three-seed health-aware fine-tune; KronLSTM stopped (2026-11-12)

The lead's calls (Anton away; each reversible, listed for his review):

- **The reported health clock is the mean of the three health-aware Atlas fine-tunes on the clean table**
  (`20261112-0022-ens-atlasft-rank1-3s`): aging-linked within-study gap +2.94 y [1.24, 4.74] over 33 unseen studies,
  0 of 200 label shuffles reach it, neurodegenerative +4.68 [1.46, 7.96], test MAE 10.6. The gap is the stable
  part (+2.85 to +3.09 across seeds); accuracy moves ~0.9 y. Rejected: the health ensemble with the MLPs (test 9.5,
  gap +2.79, but the MLPs dilute the neurodegenerative signal to n.s.), kept as the accuracy-balanced variant; and
  the earlier health ensemble (its fine-tune trained on the pre-D4 table). The plain headline stays the age clock.
- **KronLSTM for rna-age is stopped** (RM8, RM9 canceled): the question to Anton (#rna-age > decisions, asked
  2026-11-11 12:42) went unanswered to its date, and its default was to stop. From scratch it trailed the plain MLP
  by ~5 y on validation; the pretraining code (412030a) stays in the tree for a later try.

## D7 — Tabular foundation models as single-cell aging clocks (2026-11-12)

Anton (#rna-age, 2026-11-12 ~01:10): "let's make some tests and see", to the lead's proposal: single-cell cohorts are
small tables (the label is the donor's age, a cohort has a few hundred donors), the regime in-context tabular models
are built for, unlike bulk where they lost (200 PCs of 20k genes, ~47k samples).

- **Data:** the seven cohorts with per-donor cell-type pseudobulk on disk (blood: 981, 625, 261 donors and Terekhova's
  166 people; brain: 1,042, 300, 152 donors). Per representation: the donor's whole pseudobulk and each cell class
  alone (≥ 20 cells in ≥ 50 donors).
- **Protocol:** 5-fold cross-validation grouped by person; inside each training fold the 500 most variable genes,
  standardised on the fold (500 fits TabICLv2's feature budget); models TabICLv2 (no licence key needed), ridge and a
  small network (128-32).
- **The bar:** TabICLv2's MAE minus ridge's on the same donors, 95 % CI from resampling donors, excludes 0 in its
  favour, in at least two cohorts. Reported per cohort and cell class.
- Rejected: TabPFN (needs a Prior Labs key, D5's context); more genes through PCA (what failed on bulk).

## D8 — Scoring the CD4 T-cell Perturb-seq with the CD4 T clock (2026-11-12)

On Anton's word ("download virtual cell data", 2026-11-12), the lead's design, written before any screen profile is
scored:

- **Clock:** ridge on the CD4 T-cell pseudobulk of all 981 donors of the largest blood cohort (`3faad104…`), genes
  shared with the screen's probe panel, the 2,000 most variable. Ridge, not TabICLv2: one model scores ~100k
  profiles, and RM16 found ridge within ~0.8 y of TabICLv2 in that cohort.
- **Readout:** relative only. Each profile's shift = its prediction minus the mean of the non-targeting guides of
  the same donor and condition; per gene and condition the mean shift, with an empirical p against the
  non-targeting guides' own shifts. Absolute ages in the screen are not read (cultured cells, Flex probes).
- **The check, before reading any hit:** per condition, the per-gene shifts correlate (Spearman, two-sided p < 0.05)
  with the paper's own predicted aging effects of the same perturbations
  (`aging_prediction_condition_comparison_regulator_coefficients.csv`, their CD4 T rows). If they do not, the hit
  list is reported as not validated. The sign of their coefficients is read from their paper before the comparison.
- Rejected: the per-cell files (~1.8 TB); absolute predicted ages.

**D8 outcome (2026-11-12, RM17 done, 5ecdf24):** the check passed in every condition: Spearman ρ 0.26 of the per-gene
shifts against the paper's coefficients (p < 1e-36, ~2,400 regulators each; ρ ≈ 0.4 on their strongest fifth), in
the same direction (their positive coefficient is a knockdown that looks older: the preprint's Fig. 5B and the authors'
code), and within each cell-count tertile (ρ 0.23-0.32). No single gene passes FDR 5 % with the empirical null; the
hit list waits for RM19's fitness correction.

## D9 — Patients-vs-controls tests on single-cell cohorts are read within processing batch (2026-11-12)

RM18's lupus gap (+4.8 y on every clock, p 0.001) vanished within the balanced processing batch: batch 1 held only
controls and batch 2 mostly patients. From now on, a single-cell disease gap counts as a finding only when it is
tested within batch (labels shuffled within batch), with the batch read from the dataset's own per-cell metadata
(`donor_obs/*.mixed.parquet`). Where the release has no batch, the result is reported as unadjusted, never as a
finding. Sex and ancestry are checked too, but they did not stand in for batch here.

**D8, RM19 outcome (2026-11-12, `rna-age perturb-hits`, 1efb19b):** after removing each knockdown's cell-loss trend
(shift on log cells over the non-targeting median; the trend turned out nearly flat), 16 genes pass FDR 5 % in two or
more conditions with one sign. Older: CCT3, HSPE1, UQCRC1, NSMCE3, GSPT1, METAP2, MTHFD1, TSG101, PRCC, EFR3A, and the
TCR genes CD3D and ZAP70 (when stimulated). Younger: SOD1, LSM8, FBXO5, KIF18B, RBSN. Of the 20 hit × condition pairs
the paper also scores, 19 agree with its sign. Most are essential genes, and cell count is a weak fitness proxy; read
them as candidates, not as findings.

**RM21 outcome (2026-11-12, runner, 3487491/ea8cdd9; reviewed by the lead):** an ESM-2 (650M) embedding of the
knocked-down gene's protein does not predict a knockdown's age effect. The bar (beat the training mean and a ridge on
shuffled embeddings, both CIs above 0) is met in 2 of 12 tests, both trivially (ρ 0.03-0.05 for our clock's Stim48hr
shifts, skill ≤ 0.005), and in none for the study's coefficients; label-shuffle |ρ| ≤ 0.04. Failed: what the protein
is says almost nothing about what its knockdown does to age here, and per-gene shifts resting on one guide cap any
predictor. RM20 should not lean on gene embeddings for its held-out genes.

**RM20 outcome (2026-11-12, runner, fdf1310/8050fc1; reviewed by the lead):** the multitask network (age + Marson
knockdown Δ, DEG-weighted) fails its bar. Δ on held-out genes: DEG recall@50 0.052, below the training mean's 0.085 and
ridge's 0.091. Age MAE: 10.64, level with the network without the Δ head, while a ridge gets 8.17. Predicted shifts
against the paper's coefficients: ρ −0.03. One lead remains, RM24: the rank arm's clock on measured profiles, ρ 0.30
vs placebo 0.23 vs RM17's ridge 0.25, on one seed.

**RM24 outcome (2026-11-12, runner, aad5a37; read by the lead):** RM20's rank term passes both bars fixed before the
run, over 5 seed pairs: rank minus placebo ρ +0.056 [0.035, 0.071], rank minus RM17's ridge +0.029 [0.009, 0.049]
(seed 2 by 0.001), at age MAE 8.66-8.78 against ridge's 8.17. Read with what the term trains on: it ranks the
training genes' knockdowns by the paper's own coefficients, so the held-out ρ is that axis generalising to new genes,
not an independent check like D8's. Lead's reading: real but small (+0.03 ρ for −0.5 y of accuracy); the multitask
line is parked, not adopted. The clock stays ridge / TabICLv2; the rank term returns only if a use needs the paper's
perturbation axis inside the clock.

**RM23 outcome (2026-11-12, runner, 0c93566/07352db/388433b; read by the lead):** canceled: no released MaxToki can
be tested fairly. The temporal 217M's age answer does not track true age (Spearman 0.02-0.10 on 150 validation
prompts; the paper's 217M answers through a regression head not released), and its knockdown effects sit inside the
shuffle null against the screen (ρ −0.03 to +0.01). Neither released 1B has time tokens (a stage-1 model, and a
training checkpoint with genes and query tokens only), so the age gate cannot be computed; no 1B was downloaded. It
returns only with the 217M head or a temporal 1B from the authors (an outside contact: Anton's call; default none).

**RM22 outcome (2026-11-12, runner, 705332b/8c9766c; reviewed by the lead):** fails its bar, 0 of 7 cohorts. Frozen
Geneformer V2-104M start-token embeddings (at most 200 cells per donor, averaged per donor) into TabICLv2, against the
2,000-gene pseudobulk TabICLv2 on the D7 folds: MAE worse in 5 cohorts (+0.43 to +2.08 y, CIs above 0) and tied in 2
(+0.09, −0.07); the CD4 T rows are worse in all 4 blood cohorts. Limits: one model and seed, no fine-tuning, a
200-cell cap against all cells for pseudobulk; Tahoe-x1 was not run (it needs bf16 and FlashAttention). Reading: on
these cohorts a frozen single-cell model's embedding carries less age than the genes it reads; the expression clocks
stay on genes, and continued pretraining of an expression encoder is not planned on this evidence.

**RM25 outcome (2026-11-13, runner, job 6439059, 0cd6a88; read by the lead):** fails its bar by far. TabICLv2 on 2,000
standardised HVGs with a nearest-study context (4,000 training samples, at most 200 per study): test MAE 15.30 y,
r 0.51, bias slope −0.69 (8,843 samples), against the bar of 9.1 (the 20k-HVG MLP's 8.1 plus 1 y); the first try
(200 PCs, random 10k context) was 15.7. Predictions shrink to the context mean. Reading: on bulk, with tens of
thousands of heterogeneous samples, an in-context regressor cannot replace a trained network; TabICLv2 stays a
single-cell clock (D7), where tables are a few hundred donors. No fine-tuning step.

**RM18 outcome (2026-11-13, lead; steps 1-2 above in the plan's history, `clock_compare_same_donors.parquet`,
`clock_compare_disease.parquet`):** done with step 3 canceled for want of data. Accuracy: on the same blood donors the
bulk clock, never trained on them, ranks age as well as the single-cell clock or better (r 0.80-0.92 vs 0.77-0.85);
it reads single-nucleus cortex 10-39 y young; gap agreement between the two 0.35-0.64. Unhealthy cohorts: no clock
shows a patient-control gap that survives processing batch (COVID none; lupus withdrawn, D9). Interventions: a search
of CELLxGENE, GEO, ArrayExpress and the literature (2026-11-13) found no public paired before/after single-cell blood
data for calorie restriction, fasting, exercise training, metformin, rapalogs, senolytics, NAD precursors, plasma
exchange, thymus regeneration, bariatric surgery or smoking cessation; those designs exist only in bulk or cytometry,
and the bulk ones were RM10's tests. What exists: long-COVID recovery series (GSE262861, 20 donors × 3 time points,
time confounded with pooling batch; GSE270479), autologous transplant series of 3 patients each, and gender-affirming
testosterone (Lakshmikanth et al. 2024, 23 donors, mostly stimulated PBMC). None is an aging intervention with enough
donors and clean pairing; they stay ideas, not plan items.

**RM26 outcome (2026-11-13, runner, 7afe44c/c624a5c/35d2685; read by the lead):** fails its bar, narrowly. The 20k-HVG
MLP (3 seeds per arm, genes recount3 measures) trained with recount3's 3,289 new adult samples against the same recipe
without them, on the unchanged test (8,843 samples, 212 studies): 7.73 vs 7.90, −0.17 y [−0.39, +0.02]; per seed the
arm is 7.99 / 7.87 / 7.80 against 7.92 / 8.32 / 7.99 (better in 2 of 3 pairings), not resolved by study. Bgee's 4 test studies gain −0.64 [−1.64, −0.20].
Against the 7.46 headline the arm alone is +0.27 [−0.14, +0.69] (it lacks the two fine-tunes). The 494 recount3
samples of new test studies score 11.2 y, far worse than the old test: those studies are harder or their labels
noisier, and they are the honest estimate for studies outside ARCHS4. Reading: 3 % more samples buys at most ~0.2 y;
the headline stays on the old table, no fine-tune retrain, and the recount3 source stays in the tree for later use
(e.g. as an external test). More data of the same kind is not where the next year of error is.

**RM27 outcome (2026-11-13, lead's code cefc4fc, runner's job 6470599; read by the lead):** fails its gap bar. The
health MLP (within-study ranking + disease head) with a Pasta-style age-order term (81k same study × tissue control
pairs ≥ 20 y apart) against the same recipe without it, 3 seeds each: aging-linked within-study gap 2.69 vs 2.55 y
(seeds 2.71 / 2.95 / 2.40 vs 2.30 / 2.44 / 2.92), below the bar of 2.9 and inside the seed spread; control test MAE
9.12 vs 9.39, so the term costs nothing and may help accuracy slightly (not tested for significance). Best seed:
200 within-study shuffles, null +0.01, p 0.005. Reading: ordering ages within a study does not sharpen the disease
gap; the health clock stays D6's. The term may return as a cheap regulariser where accuracy is the aim.

**RM28 outcome (2026-11-13, lead's code ee227b2, runner's jobs; read by the lead):** fails every bar. Our own
single-cell encoder (a masked autoencoder over 2,000 genes, 15M parameters, pretrained on 696k healthy cells of 513
donors outside the evaluation cohorts), in three arms, read out on the 7 cohorts against the best pseudobulk clock,
mean MAE difference in years (0 of 7 cohorts better in every column):

| arm | frozen + ridge | frozen + TabICLv2 | fine-tuned on donor age |
|---|---|---|---|
| masking | +1.71 | +1.41 | +1.78 |
| masking + perturbation | +1.76 | +1.61 | +1.85 |
| universal (+ age, health heads) | +1.83 | +1.25 | +1.97 |

The universal arm's bulk test MAE is 18.7 y (bar: within 1 y of the MLP's 8.1), so its health gap (+6.0 y on a
control MAE of 14.9) is not readable; its perturbation head is barely above no change (Δ r 0.045). The perturbation
objective did not help the clock (masking + perturbation is worse than masking alone). Caveat: these ran in emulated
bf16 (fixed in e25363a afterwards), which has bf16's precision, so the conclusion stands. Reading, with RM22: on
these cohorts a cell representation learned without labels, ours or Geneformer's, carries less age than summed
expression read by TabICLv2; the single-cell clock stays TabICLv2 on donor pseudobulk. RM29 is retargeted to RM30's
perturbation task.

**RM31 outcome (2026-11-13, lead's code fcdca9c/e52b2ae, runner's job 6479183, logged 3e92738; read by the lead):**
fails its bar, narrowly. Donors with ≥ 20 cells in each of 5 cell classes, D7 folds; arms against the whole-sample
clock rerun on the same donors, cohorts of 7 better beyond the paired CI (mean MAE difference, y):

| model | concat (5 classes + composition) | mean of class clocks | mean incl. whole |
|---|---|---|---|
| TabICLv2 | 3 (−0.31) | 1 (+0.53) | 1 (+0.32) |
| ridge | 1 (+0.24) | 1 (+0.71) | 2 (+0.43) |

TabICLv2 `concat` wins in the lupus cohort (6.72 vs 7.44) and both larger brain cohorts (5.99 vs 7.01; 5.13 vs 5.97),
leans its way in Terekhova (6.70 vs 7.28) and 5e57cd50, and loses in c838aec3 (6.73 vs 5.88), where a single CD4 T
clock still beats whole blood. Reading: giving the in-context model each cell class side by side helps on average
(−0.31 y) but not in enough cohorts to replace the whole-sample clock; averaging per-class clocks loses. The bigger
set-model version is not started on this result; RM32 tests the per-cell form of the same idea.
