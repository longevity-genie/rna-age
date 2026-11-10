# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM1 — Health-aware Atlas fine-tune, then the health ensemble

**State:** building · **Owner:** runner (job `rna-age-ft-u4-rank1`, started 2026-11-10 ~20:15), then lead (what it means)

The ranking loss and disease head (`src/rna_age/train/health.py`) applied to the Atlas fine-tune
(`src/rna_age/train/finetune.py`). When it ends: the case − control gap on held-out studies against the health-aware
MLP (+2.35 y), and a health ensemble (health-aware MLPs + this fine-tune) scored for test MAE and the gap.

## RM2 — Tabular foundation models on our splits

**State:** building · **Owner:** runner (LimiX-2 job `rna-age-tfm-limix2-hvg-2k` running)

TabICLv2, GeneICL and TabFM 1.1 ran (test 15.6–16.3, far behind the MLP's 10.4); Mitra-v2 does not fit an 11 GB card.
Left: LimiX-2 (running). TabPFN-3.5 is RM6.

## RM3 — Seed-1 fine-tune into the fine-tune mean; re-check the headline

**State:** unattended · **Owner:** runner

The seed-1 all-health-states fine-tune finished (test ~1 y worse than seed 0). Score the headline with it in the
fine-tune mean and keep whichever is better on validation (never on test).

## RM4 — Composition beyond blood: Terekhova cell-type pseudobulk, then lung and brain

**State:** after RM1 · **Owner:** lead (design), runner (runs)

Blood says the aging signal sits within cell types, mostly T cells. Next: cell-type pseudobulk from Terekhova 2023,
then the same composition-vs-within test in lung (HLCA) and brain (PsychENCODE, MSSM).

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM6 — TabPFN-3.5 arm

**State:** blocked: desk #10 (default: comparison dropped) · **Owner:** runner

The `tabpfn` package needs a Prior Labs API key besides the HF gate; not bypassed.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
