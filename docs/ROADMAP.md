# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM24 — RM20's rank term over seeds

**State:** building (runner) · **Owner:** runner runs, lead reads

RM20 failed as specified (Δ on held-out genes below the mean and ridge; predicted shifts ρ −0.03), but its rank arm's
clock reads measured Perturb-seq profiles at ρ 0.30 against the paper's coefficients, versus 0.23 for its placebo and
0.25 for RM17's ridge (one seed). Five seeds each, same seeds, splits and panel. Fixed before the run: it counts if (a)
rank minus placebo ρ > 0 with a paired bootstrap 95 % CI over the seed pairs, and (b) the rank arm beats 0.25 by the
same test. Age MAE reported beside (ridge 8.17). If (a) fails, the multitask line stops.

## RM22 — Frozen virtual-cell embeddings as clock features

**State:** building (runner, same word) · **Owner:** runner builds, lead reviews the merge

Tahoe-x1 or Geneformer cell embeddings, pooled per donor, into TabICL; against our pseudobulk TabICL on the same folds
(D7: 5-fold by person). Bar: lower MAE by more than the paired bootstrap's 95 % CI in at least 4 of 7 cohorts. Any
disease reading follows D9 (within batch).

## RM23 — MaxToki in-silico knockdown screen of CD4 T cells

**State:** building (runner, same word) · **Owner:** runner builds, lead reviews the merge

Bar: its predicted aging effects correlate with the Marson screen's and with our clock's RM17 shifts (Spearman, p <
0.05 per condition), compared with the same correlation for a random-gene-order placebo. Licence of the weights read
before any output leaves the repository.

## RM25 — TabICLv2 as the bulk clock

**State:** unattended (proposed to Anton 2026-11-12 in #rna-age > general chat; default: build) · **Owner:** lead
(code), runner (GPU run)

TabICLv2 in context, no weight updates, on the clean bulk table (D4), same split and features as the headline MLP
(20k HVGs, or the top PCs if the context needs it); the training set as context, subsampled per study if it does not
fit. Bar: test MAE below the headline's 7.46 by more than the paired bootstrap CI. Fine-tuning TabICL on bulk +
pseudobulk follows only if this is competitive.

## RM18 — Bulk clock vs single-cell clock, same people: accuracy, unhealthy cohorts, interventions

**State:** building · **Owner:** lead (comparison code), then runner (pseudobulk passes and GPU runs)

Anton (2026-11-12): compare the bulk and single-cell clocks on accuracy and on their response to unhealthy cohorts
and interventions. (1) Each single-cell cohort scored by both clocks on the same donors; (2) patients vs controls in
the lupus and COVID blood single-cell cohorts we hold, gap per clock and per cell type, with the label-shuffle null;
(3) interventions once RM17's survey finds single-cell before/after blood data. One side-by-side table at the end.

Step 1 landed (`rna-age clock-compare`, 2026-11-12): 7 cohorts, 3,300 donors. Blood (4 cohorts): the bulk clock, which
never trained on these people, ranks age as well as the single-cell clock or better (r 0.80-0.92 vs 0.77-0.85), MAE
equal in 3 of 4 (OneK1K: 2.9 y worse, a -7.6 y offset). Brain (3 single-nucleus cortex cohorts): the bulk clock reads
10-39 y young (a level offset; r 0.70-0.89 vs 0.76-0.91). The two clocks agree only partly on who is old for their
age (Spearman of the age-adjusted gaps 0.35-0.64; -0.13 in the cohort of narrow ages).

Step 2 landed (`rna-age clock-disease`; single-cell clocks trained on 3faad104's 981 healthy donors, all clocks
outside the cohort; label-shuffle null, 1,000 shuffles). Lupus (162 vs 99): every clock reads patients older, bulk
+4.8 y, health clock +4.6, single-cell whole sample +5.2, CD4 T +5.1, CD8 T +3.9, NK/B +3.5, monocytes +1.5 (all p
0.001, the null's floor). COVID-19 (156 vs 25 controls): no clock separates (single-cell whole sample +4.8, p 0.055;
the rest p > 0.15). Step 2b: sex is no confound (controls 97 % women, patients 91 %; women only: whole sample
+4.8, CD4 T +4.7, CD8 T +3.5, monocytes +1.4, all p ≤ 0.002). Ancestry is no confound either (the dataset's donor metadata, 4bf6298):
European American, 75 vs 74, whole sample +4.2, CD4 T +4.3 (p 0.001), monocytes +0.6 (p 0.08); Asian, 83 vs 24, whole
sample +6.9, CD4 T +4.4 (p 0.03), monocytes +3.6 (p 0.001). **Batch confounds it** (Processing_Cohort, runner's 4bf6298
`.mixed.parquet`): batch 1 is controls only, batch 2 is 103 patients vs 16 controls (15 of them spanning batches).
Stratified by batch (labels shuffled within batch) the gap shrinks to +1.3 to +3.6 y (p 0.001-0.04); in the one
balanced batch, single-batch donors only (25 vs 41), it is gone for the T cells (CD4 +0.1, CD8 +0.4, p > 0.8) and not
significant for the whole sample (+2.2, p 0.17) or the bulk clock (+2.4, p 0.06); monocytes +1.7 (p 0.013, one of six
tests). Verdict: no clock shows a lupus gap that survives batch; step 2's lupus line is withdrawn as a finding.

## RM13 — A manuscript draft of the results (plan line 5)

**State:** waits until 2026-11-13: Anton's answer, desk #23 and #rna-age > decisions; default: start the draft · **Owner:** lead

Outline, methods and figures regenerated from the recorded runs (D4 clean table, D6 health clock, RM10 tests),
kept private in the repository until Anton decides where it goes.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
