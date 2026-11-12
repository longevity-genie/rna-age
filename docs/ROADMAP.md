# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM19 — A fitness-corrected hit list from the Perturb-seq scores

**State:** unattended, after RM18 step 2 · **Owner:** lead

RM17's largest age shifts are essential genes. Model each knockdown's shift with the cells it left (and the
knockdown's overall transcriptional effect size) as covariates, test the residual against the non-targeting guides,
and report genes passing FDR 5 % in at least two conditions.

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
+4.8, CD4 T +4.7, CD8 T +3.5, monocytes +1.4, all p ≤ 0.002). Still open: processing batch and ancestry, which the
cohort's census pseudobulk lacks; the original dataset's per-donor metadata (processing cohort, ancestry) is fetched
by the runner, then the gap is tested within batch and ancestry.

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
