# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM1 — Health-aware Atlas fine-tune, then the health ensemble

**State:** building · **Owner:** runner (job `rna-age-ft-u4-rank1`, started 2026-11-10 ~20:15), then lead (what it means); last epoch, eta ~03:30

The ranking loss and disease head (`src/rna_age/train/health.py`) applied to the Atlas fine-tune
(`src/rna_age/train/finetune.py`). When it ends: the case − control gap on held-out studies against the health-aware
MLP (+2.35 y), and a health ensemble (health-aware MLPs + this fine-tune) scored for test MAE and the gap.

## RM8 — KronLSTM expression encoder, from scratch (D2, D3)

**State:** building · **Owner:** runner (the runs below; code landed in 52b6f16), then lead (what they mean)

Bidirectional KronLSTM (kron-fm's `kron_core`, read from the `../kron-fm` clone) over rank tokens or the HVGs in
genome order, from scratch. Runs, in order (GPU, one card each):

1. Smoke: `rna-age kron --encoding rank --length 512 --epochs 1 --name kron-smoke`: the Triton kernels under this
   project's torch were never run; a failure here comes back to the lead.
2. Context sweep, seed 0: `--encoding rank --length 512 | 2048 | 4096`.
3. The best length (validation), seeds 1 and 2; `--encoding genome` and `--encoding genome-shuffled`, seeds 0-2
   (20k tokens: lower `--batch` or `--d-model` to fit the card, the same for both).
4. Ablation: `rna-age kron-rank-mlp --length <best> --seed 0|1|2`.
5. The bar: `rna-age kron-compare <arm's 3 runs> <three seeds of hvg20000-mlp-trainall>`, for each arm and for
   the rank-MLP; logged with the per-source lines.

## RM9 — Masked-gene pretraining probe for the KronLSTM encoder (D3)

**State:** building · **Owner:** lead (pretraining code, `rna-age kron-pretrain`), then runner (runs)

Masked-gene prediction over rank tokens of training-split studies only (ARCHS4 + GTEx), then the RM8 age head on
top; runs whatever RM8's from-scratch result.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
