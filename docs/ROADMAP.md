# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM34 — Positive-control datasets for system ageing: fetch and check which are usable

**State:** urgent (Anton, 2026-11-13, #rna-age > general chat: "Ok, download these data and validate which is
usable?") · **Owner:** lead (code), runner (downloads and parse jobs)

From RM33's dataset search (~30 candidates: conditions where a known system should look older, with controls in the
same study). Step 1, before any large download: read each CELLxGENE dataset's per-cell metadata over HTTP range
requests and each GEO series' sample characteristics (bulk series first looked up in our ARCHS4 table), and score it:
condition column and arms, donors per arm, share with exact ages, age overlap between arms (donors left after
matching), cells per donor × coarse type, assay/platform, whether the condition is confounded with a batch column
(D9), sex. Verdict per dataset: usable, usable with a caveat, or not, with the reason, in
`data/output/summary/positive_controls_usability.parquet`. Step 2: download only the usable ones (CELLxGENE h5ads
from the Census release or, if newer, the Discover API; GEO bulk from ARCHS4 where present), then per-donor metadata
and per-type pseudobulk with the existing `census-donor-obs` / `census-pseudobulk --by-cell-type`.
Bar: step 1 scores every candidate; a dataset counts as usable with ≥ 10 donors per arm after age matching, exact or
≤ 5-year ages, and the condition not confounded with batch.

## RM33 — System-level age gaps: which cell types or tissues make a person look older

**State:** unattended (design A by default, staged; Anton's pick of A/B/C asked in #rna-age > per-cell clock,
2026-11-13) · **Owner:** lead

Review (2026-11-13; a design agent wrote three designs, two separate reviewers judged them on our data and cluster):
A labelled cell types 6/10 validity, 8/10 feasibility; B label-free groups on our encoder 4/4 (encoder never saw brain,
lacks CD4/CD8A/KLRC2 and glial markers; CNA kept as a cross-check on A); C gene programs 3/3 (expected programs are
the disease's response programs: circular). Fixes taken into A: cases and controls scored by the same fold models;
every bar in years of acceleration (clock sensitivity differs 2.6× between CD8 T and monocytes), nulls as equivalence
tests, accuracy as non-inferiority; clocks retrained without the condition's DE genes (from another dataset) and on
disjoint gene halves; QC and ambient covariates, test-retest reliability where repeat visits exist; reliability floor
not tied to age range. Stages: 0 data on disk, ridge, shared vs specific gaps and AD in PsychAD (go/no-go: microglia
and astrocytes older than oligodendrocytes); 1 HLCA smoking within study, ILD; 2 Allen CMV after its obs column is
confirmed; 3 pseudocells and fine types only for passing systems. Datasets come from RM34.

Design (Anton, same topic: "we have to modify design to allow idea of seeing age difference per system"): his
individual + group loss with the system (cell type; tissue in bulk) as the unit. (0) One clock per system on
chronological age; the person's age is a learned non-negative weighted sum of the system ages; loss = person error +
λ × each system's error; output per person: overall age, each system's bias-corrected gap with a CI split into the shared person-level part and the system-specific deviation, its weight. Bar:
the system-sum model stays within the paired CI of the whole-sample clock in at least 5 of 7 cohorts (RM31's plain
mean lost 0.3-0.7 y). Unit (Anton: "pseudocells ... make it balanced ... save computation but still provide per system distributions"):
a fixed number of 15-cell pseudocells per donor × system (e.g. 30, with replacement for small donors), so every
donor and system weighs the same and compute is fixed; each system's age per person is the mean over its pseudocells,
the spread gives its CI and shows mixed systems. Read by ridge and TabICLv2 on summed expression (Buckley 2023's
setting), against the per-type pseudobulk of RM31; similarity metacells (SEACells-style) as the variant that keeps
sub-states apart.
Annotation bias (Anton: "is it not biased on cell type detection and attribution of cell types to subsystems?"):
coarse classes first; a second, uniform annotation of every cohort by reference mapping (e.g. CellTypist, one
reference) beside the authors' labels, a system gap reported only where the two agree (r ≥ 0.7 across donors); each
type's share and the unassigned share against age, a type that drifts with age flagged; and label-free arms (Anton: "approaches that do not depend on how authors labeled cells"):
clusters of all cohorts' cells on RM28's cell embedding as systems, named afterwards by marker genes; covarying
neighbourhood analysis (CNA, Reshef 2022) for where in cell space age sits; gene programs (cNMF) as functional
systems if the clusters hold; learned soft groups last. Then the known-cause tests
below, on its system gaps.

RM31's per-cell-type clocks (out of fold, TabICLv2) give each donor an age per cell type; their age-adjusted gaps
correlate at r 0.2-0.5 between types, so part of a gap is type-specific, real or noise. Tests where the answer is
known: (1) Alzheimer donors vs controls in the brain cohorts 37a17b78 and 5e57cd50, within processing batch (D9). Bar:
at least one glial type (microglia, astrocyte) shows a larger AD gap than OPCs and oligodendrocytes, beyond a donor
bootstrap CI, in both cohorts. (2) CMV serostatus in Terekhova 2023, if its donor metadata carries it. Bar: the CD8 T
gap is higher in CMV+ beyond the CI and the monocyte gap is not. (3) GTEx per-donor tissue gaps from the bulk clock,
descriptive (how much organ ages agree within a person). CPU only.

## RM29 — Score a perturbation model on the Virtual Cell Challenge H1 set

**State:** after RM30 (RM28's perturbation head failed its own check, 2026-11-13; retargeted to RM30's
perturbation task, which must first beat the mean-change baseline on held-out knockdowns) · **Owner:** lead (code),
runner (GPU runs)

Anton (2026-11-13): "is our model compatible with virtual cell challenge, curious to score it". Train the
perturbation model on the challenge's public H1 training perturbations, predict the held-out ones (cells sampled
around the predicted mean for the cell-level format), score with Arc's public cell-eval. Bar: beat the mean baseline
on PDS and DES on the held-out perturbations. Not started unless RM30's perturbation skill on held-out knockdowns is
above 0.

## RM30 — An in-context tabular model pretrained on expression tables: age, health, perturbation

**State:** building (job 6479218 training, 10,000 steps, due ~02:45 on 2026-11-14) · **Owner:** lead
(design, code), runner (GPU runs)

TabICLv2 (BSD-3, architecture and pretraining code public) is pretrained on synthetic tables; on donor tables it is our
best single-cell clock, on bulk it collapses to the context mean (RM25). Here the same architecture is trained on
real expression episodes. An episode is a table drawn from our corpus (a study, a cohort, a tissue, a Perturb-seq
condition): rows are samples or donors, columns a random subset of the shared gene panel, and the target one of:
age (regression), case vs control (classification), a held-out gene's expression (masked expression in tabular form),
or a knockdown's change on a gene given the perturbed gene's features (perturbation). Context rows carry labels, query
rows are predicted; episodes from the evaluation cohorts, the bulk val/test studies and held-out knockdowns never
enter training. Corpus: bulk (≈94k samples), recount3, single-cell donor pseudobulk of the pretraining pool, Census
donors, Marson CD4 pseudobulk, the Virtual Cell Challenge H1 training split. One arm: a new model initialised from the
released TabICLv2 weights, at its own size, trained on these episodes (Anton, 2026-11-13: "by from scratch I still
meant reusing tabicl2 weights, not using random, I meant that we do not fine tune old stuff"), not a fine-tune of
our earlier clocks; no random-init arm.
Bars, fixed now: (1) single-cell age on the 7 cohorts beats pretrained TabICLv2 beyond the paired CI in at least 4;
(2) bulk test MAE within 1 y of the 20k-HVG MLP (RM25's bar, 9.1); (3) within-study health gap at least D6's +2.9 y on
the same unseen studies; (4) perturbation scored as RM29. Code reviewed and on main 2026-11-13 (8232415, 81190c0):
1,000-gene panel of 16,585 measured everywhere; training groups age 352, health 64, masked gene 410, perturbation 12
donor × condition tables (2,370 of 11,929 knocked-down genes held out). First-run limits: a held-out knocked-down
gene is described only by its own baseline expression and a hashed id (no gene embedding), and recount3 and the
Virtual Cell Challenge data are not yet in the corpus. Deviation (2026-11-13): the model does not fit an 11 GB card at the specified
episode size even one episode per step, so the first run trains on smaller episodes (64-256 rows, 100-300 columns);
evaluation is unchanged. Gradient checkpointing and a rerun at full size follow only if this run shows a gain. Reported per head; "universal" needs (1) or (2) and no head
failing its reference by more than its CI.

## RM13 — A manuscript draft of the results (plan line 5)

**State:** building (default of desk #23 applied 2026-11-13; draft v0.1 kept outside git, the repository being public; where it goes: desk #23) · **Owner:** lead

Outline, methods and figures regenerated from the recorded runs (D4 clean table, D6 health clock, RM10 tests),
kept private in the repository until Anton decides where it goes.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
