# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

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

**State:** after RM28 (Anton, 2026-11-13: "train tabular model from scratch to be able to do aging, perturbations
(and whatever virtual cell asks) and health related stuff ... increase training corpus"; code is written meanwhile,
the first GPU run follows RM28's first result, which shares the GPUs) · **Owner:** lead
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

## RM31 — Several cell types per donor at once in the single-cell clock

**State:** building (Anton, 2026-11-13, #rna-age > milestones: "improve prediction if we get multiple cells per
evaluation for single cell clocks together") · **Owner:** lead (code, fcdca9c), runner (the TabICLv2 run)

Per cohort, donors with ≥ 20 cells in each of the 5 best-covered classes; D7 folds. Arms: the class pseudobulks side
by side with the composition fractions (`concat`), the mean of the per-class clocks, and that mean with the
whole-sample clock, each against the whole-sample clock rerun on the same donors. Bar: an arm beats the whole-sample
TabICLv2 beyond the paired CI (persons) in at least 4 of 7 cohorts. If one does, the bigger version (a set model over
the cells, on RM28's encoder) follows. Ridge check on the lupus cohort: `concat` ties (6.80 vs 6.80), the means lose.

## RM32 — Anton's per-cell clock: cells push the donor's age up or down, individual + group loss

**State:** building (Anton, 2026-11-13, #rna-age > plan: "make a separate topic to build and test my design"; topic
#rna-age > per-cell clock) · **Owner:** lead (code 331e85c), runner (GPU runs)

Every cell gets an age from a head on RM28's cell encoder; the donor's age is a pooled sum of its cells' ages (equal
weights, or gated attention that learns which cells matter); loss = the donor's error + λ × each cell's error
against the donor's age. Arms on the best RM28 encoder: mean λ=0 (RM28's read-out, the reference), attention λ=0, and mean
and attention with λ ∈ {0.1, 0.5} on 15-cell pseudocells (`--pseudocell 15`). The literature check (posted in the
topic, 2026-11-13) found no single-cell age model trained on per-cell and donor losses together; the per-cell loss
equals the donor loss plus a penalty on within-donor spread (Javanmard et al. 2024), so each arm also reports the
slope of predicted vs true age, the warning sign for shrinkage.
Bar: an arm beats the whole-sample TabICLv2 beyond the paired CI in at least 4 of 7 cohorts; per-cell ages are kept
for the cells-that-look-old analysis only if it passes.

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
