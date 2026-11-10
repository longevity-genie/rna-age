# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM1 — Health-aware Atlas fine-tune, then the health ensemble

**State:** building · **Owner:** runner (job `rna-age-ft-u4-rank1`, started 2026-11-10 ~20:15), then lead (what it means); epoch 3 of 6 at 00:10

The ranking loss and disease head (`src/rna_age/train/health.py`) applied to the Atlas fine-tune
(`src/rna_age/train/finetune.py`). When it ends: the case − control gap on held-out studies against the health-aware
MLP (+2.35 y), and a health ensemble (health-aware MLPs + this fine-tune) scored for test MAE and the gap.

## RM4 — Composition: Terekhova cell-type pseudobulk (optional)

**State:** unattended · **Owner:** runner (runs), lead (what it means)

Blood and brain (PsychENCODE, 783 and 299 donors) both say the aging signal sits within cell types: fixed composition
r 0.82 / 0.91 vs composition alone 0.41 / 0.46; lung cohorts too small (commits 366099b, 3217d30, 748260e). Left:
cell-type pseudobulk from Terekhova 2023, to say which blood cell types carry it.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
