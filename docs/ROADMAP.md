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

## RM8 — KronLSTM expression encoder (D2, D3, D5)

**State:** waits until 2026-11-12: Anton's answer in chat (one pretraining run, or stop the line; asked 12:42) · **Owner:** lead

From scratch, the sweep ended at val 13.95 (2,048 tokens) vs ~8.7 for the 20k-HVG MLP; ablations cancelled on
Anton's word (D5). RM9's pretraining code (412030a) is ready if he chooses option 1; otherwise both items end
canceled. Default if unanswered: stop the line.

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
