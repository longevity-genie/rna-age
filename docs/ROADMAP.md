# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM17 — Perturbation screens scored by the single-cell clocks

**State:** unattended · **Owner:** lead (fetcher and scoring code), then runner (fetch, runs)

Survey done (2026-11-12): the genome-scale CD4 T-cell Perturb-seq (Zhu, Dann … Marson, Cell 2026,
doi:10.1016/j.cell.2026.08.002; CZI Virtual Cells Platform, MIT licence) ships per guide × donor × condition
pseudobulk counts. Anton's word (2026-11-12: "download virtual cell data") covers fetching that file only. Then score
each knockdown with the CD4 T clock (TabICLv2 / ridge, RM14-16) as an age shift against non-targeting guides,
per condition (D8: ridge on OneK1K-size CD4 T; the check against the paper's own aging coefficients is fixed
before the run). Fetcher 9a158aa's predecessor and `rna-age perturb-score` (9a158aa) landed; runs (runner): the fetch
(job 6434376), then `rna-age perturb-score` (CPU, ~60-100 GB RAM for the 44.6 GB file).

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
