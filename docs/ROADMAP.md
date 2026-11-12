# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM14 — Tabular foundation models as single-cell aging clocks (D7)

**State:** building · **Owner:** runner (runs; code landed in d998476), then lead (what they mean)

Runs, one GPU each: `uv run --with tabicl rna-age sc-tabular <cohort>` for the blood cohorts (`3faad104…`,
`c838aec3…`, `218acb0f…`, `terekhova2023`) and with `--mapping cell_type` for the brain cohorts (`37a17b78…`,
`5c97eeeb…`, `5e57cd50…`). Output: `data/output/summary/sc_tabular_<cohort>.parquet` and the compare table in the
log. The bar (D7): TabICLv2 beats ridge, donor-bootstrap CI excluding 0, in at least two cohorts.

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
