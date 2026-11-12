# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM17 — Perturbation screens scored by the single-cell clocks

**State:** done 2026-11-12 (scoring, D8's check passed); the hit list is RM19 · **Owner:** lead

Survey done (2026-11-12): the genome-scale CD4 T-cell Perturb-seq (Zhu, Dann … Marson, Cell 2026,
doi:10.1016/j.cell.2026.08.002; CZI Virtual Cells Platform, MIT licence), per guide × donor × condition pseudobulk.
Scored by `rna-age perturb-score` (job 6434582, 202,110 profiles, 4 donors, 3 conditions); checked by
`rna-age perturb-check`. **Result:** the clock's per-gene shifts agree with the paper's own aging coefficients in every
condition (Spearman ρ 0.26, p < 1e-36, ~2,400 regulators each; ρ ≈ 0.4 on their strongest fifth), in the same
direction (their positive coefficient = knockdown looks older, read from the preprint and code), and within each
cell-count tertile (ρ 0.23-0.32, so not a cell-fitness artefact). Single genes: none passes FDR 5 % with the
empirical null (its floor is p ≈ 0.001 with ~960 non-targeting guides); 39-48 per condition do with a normal null
fitted to the non-targeting guides. The largest movers are essential genes (chaperonins, splicing, DNA repair), so a
hit list needs the fitness confound handled first. Like the paper, mTORC1 brakes (TSC1/2, DEPDC5) knocked down read
slightly younger (not significant one by one); MTOR knocked down reads 15 y younger at 48 h stimulation.

## RM19 — A fitness-corrected hit list from the Perturb-seq scores

**State:** unattended, after RM18 step 1 · **Owner:** lead

RM17's largest age shifts are essential genes. Model each knockdown's shift with the cells it left (and the
knockdown's overall transcriptional effect size) as covariates, test the residual against the non-targeting guides,
and report genes passing FDR 5 % in at least two conditions.

## RM18 — Bulk clock vs single-cell clock, same people: accuracy, unhealthy cohorts, interventions

**State:** building · **Owner:** lead (comparison code), then runner (pseudobulk passes and GPU runs)

Anton (2026-11-12): compare the bulk and single-cell clocks on accuracy and on their response to unhealthy cohorts
and interventions. (1) Each single-cell cohort scored by both clocks on the same donors; (2) patients vs controls in
the lupus and COVID blood single-cell cohorts we hold, gap per clock and per cell type, with the label-shuffle null;
(3) interventions once RM17's survey finds single-cell before/after blood data. One side-by-side table at the end.

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
