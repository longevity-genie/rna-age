# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM1 — Health-aware Atlas fine-tune: seeds and the health ensemble

**State:** building · **Owner:** runner (the runs below), then lead (what they mean)

Seed 0 (run 20261110-1516-atlas-ft-u4-rank1) passes plan line 1 on one seed: aging-linked within-study gap +2.93 y
[1.17, 4.67] on 35 held-out studies at control test MAE 10.37; the within-study label-shuffle null is ~0 and below
the observed gap in all 200 shuffles (054db35). Neurodegenerative +3.84 [0.66, 7.47], the first significant one.
Pseudobulk transfer worse (17.7 vs 13.3). Runs:

1. Health ensemble (CPU): `rna-age ensemble` of the three health-aware MLPs (rank1-head s0, s1, s2) and this
   fine-tune, then `health-eval --permutations 200` and its accuracy.
2. Seeds 1 and 2 of the fine-tune, the same flags (`finetune --unfreeze 4 --epochs 6 --max-train 20000
   --rank-weight 1 --seed 1|2`), sharing GPU slots with RM8 (RM8's sweep first); then the three-seed gap.

What a positive gap means is open: the ranking loss teaches "cases older" on training studies, so a held-out gap
shows that disease signal generalises into the age axis, not yet that disease ages the body. Plan line 2 (dose,
severity, interventions) is the test of that.

## RM8 — KronLSTM expression encoder, from scratch (D2, D3)

**State:** building · **Owner:** runner (the runs below; code landed in 52b6f16), then lead (what they mean)

Bidirectional KronLSTM (kron-fm's `kron_core`, read from the `../kron-fm` clone) over rank tokens or the HVGs in
genome order, from scratch. Runs, in order (GPU, one card each):

1. Smoke: `rna-age kron --encoding rank --length 512 --epochs 1 --name kron-smoke`: the Triton kernels under this
   project's torch were never run; a failure here comes back to the lead.
2. Context sweep, seed 0, `--epochs 8` for all (a sweep, not full training): `--encoding rank --length 512 | 2048 | 4096`.
3. The best length (validation), seeds 1 and 2; `--encoding genome` and `--encoding genome-shuffled`, seeds 0-2
   (20k tokens: lower `--batch` or `--d-model` to fit the card, the same for both).
4. Ablation: `rna-age kron-rank-mlp --length <best> --seed 0|1|2`.
5. The bar: `rna-age kron-compare <arm's 3 runs> <three seeds of hvg20000-mlp-trainall>`, for each arm and for
   the rank-MLP; logged with the per-source lines.

## RM9 — Masked-gene pretraining probe for the KronLSTM encoder (D3)

**State:** after RM8 · **Owner:** runner (runs; code landed in 412030a), then lead (what they mean)

Runs once RM8's smoke has passed and its context sweep has named the best length L:

1. `rna-age kron-pretrain --length L` (training-split studies only, ages unused; resumable).
2. `rna-age kron --init <its run id> --seed 0|1|2`, and `kron-compare` against the from-scratch rank arm and the
   20k-HVG MLP.

## RM10 — Does the disease gap mean biology? Severity and before/after data (plan line 2)

**State:** unattended · **Owner:** lead (datasets, analysis), then runner (any runs)

Desk #19 asks Anton whether now; its default is now, so it is built under that. Public held-out series with disease
severity within a study, or the same people before and after a treatment or an intervention; the existing models
(health ensemble, RM1 fine-tune, the headline) scored on them; the expected direction written before each run.

## RM11 — Robustness: error by tissue and platform (plan line 3)

**State:** after RM12 · **Owner:** lead

`rna-age robustness RUN` landed (d63c73c). After the re-merge: the table for the headline and the health ensemble,
and what it means (first look: NextSeq 500 flagged, blood over-predicted by ~7 y); leave-one-tissue-out runs only
where a flagged group needs it.

## RM12 — Re-parse ARCHS4 and re-merge: tumours and plate single-cell series out (D4)

**State:** unattended · **Owner:** runner, then lead (the headline re-scored)

d63c73c (meningioma and other tumours) and 98f1c5e (series with median single-cell probability > 0.4: 169 series,
5,308 samples, GSE116672 among them) take effect at `rna-age parse archs4` then `rna-age merge` (CPU; splits hashed
per study, no other study moves). It rewrites `data/input/archs4/counts/`: run it when no rna-age job is loading
data, before the next new run. Then the headline ensemble and the health ensemble are re-scored on the new table
(existing predictions, currently eligible samples) before either number is quoted again.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
