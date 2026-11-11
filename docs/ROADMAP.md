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

**State:** unattended · **Owner:** lead

First battery done (registry 99f56bc, before results; run a407f4b), re-run on the clean table (D4) for the new
headline and health ensembles: the stage effects in the diseased tissue replicate for both (liver fibrosis x2,
Braak, Lewy); the plain headline passes the exercise control; the health ensemble fails it (one exercise bout shifts
its gap, p 0.034): it reads acute inflammation as "older". Paired tests registered (bba614a) before extraction; runs (runner, in order):

1. `rna-age external GSE65540,GSE274137,GSE192587,GSE66306,GSE137631,GSE107894` (CPU, from the local ARCHS4 file).
2. On the clean table, with `--external`: `probe --features hvg --model mlp --n-hvg 20000 --train-set all
   --seed 0|1|2` and `train-health --rank-weight 1 --disease-head --seed 0|1|2`.
3. `rna-age biology-paired <the three MLP runs> --name plain` and `<the three health runs> --name health`.

The health models' inflammation response (they fail the exercise control) is stated with any health claim.

## RM12 — Re-train on the clean table (D4)

**State:** building · **Owner:** runner (runs), then lead (re-scored headline)

The re-parse and re-merge ran (6429048, 06:58): eligible 100,312 → 94,526; train 47,493, val 12,793, test 9,097
(ARCHS4). Runs on the new table, in order: RM1 seed 1 restarted, seed 2 (released), seed 0 again; the 20k-HVG MLP
(all health states) seeds 0-2; the health-aware MLPs (rank1-head) s0-s2; then the headline and the health ensembles
rebuilt from them (old fine-tunes inside until the new seeds land, noted). The sweep keeps the old table (it only
picks a length). The lead then re-scores (accuracy, health-eval with the null, robustness, biology) before any
number is quoted again.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
