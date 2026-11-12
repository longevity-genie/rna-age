# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM16 — Single-cell tabular clock against stronger baselines (Anton: ridge is a low bar)

**State:** building · **Owner:** runner (runs), then lead (what they mean)

For each of RM14's seven cohorts (same `--mapping` as before): `rna-age sc-tabular <cohort> --models enet,gbm`
(CPU), then `uv run --with tabicl rna-age sc-tabular <cohort> --models tabicl --n-genes 2000` (GPU). TabICLv2 counts
as better only against the best of ridge, elastic net and boosted trees per cohort (`--reference`), CI excluding 0.

## RM15 — Single-cell clock, next step (after RM14)

**State:** waits until 2026-11-13: Anton's choice in #rna-age > decisions (health per cell type, a usable clock, or stop); default: health per cell type · **Owner:** lead

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
