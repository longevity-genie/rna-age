# Lead notes

The state the lead's last holder left. State, not rules: the rules are the priming, `LEAD.md` beside this file, and
the kit's role `classes/lead/LEAD.md`.

## 2026-11-13 22:20

- Plan: RM32 per-cell clock (3 of 4 arms worse than the plain fine-tune, +1.81/+2.22/+2.41 vs +1.78; arm (d) running),
  RM30 (job 6479218, 10k steps, due ~02:45 2026-11-14; quick evals at 2k/4k/6k, stop if flat by 5k), RM29 after RM30,
  RM13 draft (outside git), RM5 desk #9, RM7 waits until 2026-11-17.
- Closed: RM28 fail (0 of 7, every arm), RM31 fail narrowly (TabICLv2 concat 3 of 7, −0.31 y).
- Anton away from ~20:10; chat-only questions moved to the desk: MaxToki contact #37 (default no), Tahoe-x1 test #38
  (default no). Census pin stays the runner's.

## 2026-11-13 19:20

- Plan: RM27 age-order loss on bulk (job 6470599, 3 seeds, 2 done), RM28 own single-cell pretraining, arms mask /
  mask+perturb / universal (pool job 6472026, then 6472027/6472097/6472138 on 11 GB 2080 Ti cards, memory
  unmeasured), RM29 Virtual Cell Challenge scoring (after RM28), RM30 TabICLv2-initialised model on expression
  episodes (build agent writing the code in a worktree; review before merge), RM13 draft (outside git,
  data/output/manuscript), RM5 desk #9, RM7 waits until 2026-11-17.
- Closed today: RM18 (step 3 no data), RM22 fail, RM25 fail, RM26 fail narrowly (−0.17 y, CI crosses 0).
- Duties held acting since 19:10 (L139): milestones (one post per landed result in #rna-age > milestones; the kept
  `state` post stays the runner's, as Anton set via the secretary on 2026-11-12) and upstream-index (watcher over
  tabicl, baseline 2.2.0, the version we use; `--upstream` waits for a kit with L143).

## 2026-11-13 00:05

- Plan: RM25 TabICLv2 bulk clock (job 6439059 running; bar: test within 1 y of 8.1 earns fine-tuning, beat 7.46
  beyond CI = finding), RM18 step 3 interventions (needs before/after single-cell blood data), RM13 manuscript (starts
  2026-11-13 under desk #23's default), RM5 (desk #9), RM7 (waits until 2026-11-17).
- Closed 2026-11-12 (DECISIONS): RM17/D8 rho 0.26 agrees with the paper; RM19 16 replicated candidates; D9 disease gaps
  within batch only (lupus gap withdrawn); RM20 fail, RM21 fail, RM22 fail (0 of 7, frozen Geneformer), RM23 canceled,
  RM24 passes but parked (trains on the paper's coefficients).
- Open to Anton: MaxToki authors contact (default no); newer Census release (runner asks).
- The main checkout is shared with the runner: never pull/rebase there (memory shared-main-checkout).

## 2026-11-12 16:40

- Plan: RM18 bulk vs single-cell clocks (building: step 1 done, step 2 `rna-age clock-disease` running, step 3
  interventions open), RM19 fitness-corrected Perturb-seq hit list (after RM18 step 2), RM13 manuscript (waits until
  2026-11-13, desk #23), RM5 (desk #9), RM7 (waits until 2026-11-17). RM14-17 done.
- Single-cell (D7): TabICLv2 2,000 genes beats ridge/enet/GBM in 4 of 7 cohorts. RM17 (D8 outcome): the CD4 T clock's
  Perturb-seq shifts agree with the paper's aging coefficients (rho 0.26 each condition; their + = knockdown looks
  older), no single-gene hit yet.
- RM18 step 1 (`clock_compare_same_donors.parquet`): bulk clock ranks blood ages as well as single-cell, reads
  single-nucleus cortex 10-39 y young; gap agreement 0.35-0.64. Step 2 trains every single-cell clock on 3faad104
  (COVID cohort 9dbab10c has only 25 controls) and tests patients vs controls with a label shuffle.

## 2026-11-12 01:00

- Plan: RM13 manuscript draft (waits until 2026-11-13, desk #23, default: start), RM5 PRIMO (desk #9), RM7 NCBI
  (waits until 2026-11-17). RM1, RM10, RM11, RM12 done; RM8, RM9 canceled (D6: KronLSTM stopped by default).
- Reported clocks (D6): age clock = headline ensemble `20261111-0747-ens-mlp3t2-plus-atlasft2` (test 7.46, median
  5.3, r 0.88, clean table); health clock = `20261112-0022-ens-atlasft-rank1-3s` (gap +2.94 [1.24, 4.74], null
  0/200, neurodegenerative +4.68). Never compare with pre-D4 figures.
- RM10: gap follows disease stage in the affected tissue; no intervention reverses it in months (cf. Horvath 2014,
  liver, bariatric surgery); the health clock reads acute inflammation (fails the exercise control).
- Desk: #9 PRIMO (default no), #18 BulkFormer spending (default: where it earns its place), #23 manuscript.
- Infographic: scripts/infographic_bioage.py (accuracy vs Horvath 2013 reported 3.6 y median; health gap of both clocks).
- Reporting follows the desk's docs/REPORTING.md (finding first, one number ± and n, marks). #rna-age > decisions
  holds the open questions (KronLSTM next, PRIMO, BulkFormer spending, history rewrite), one per message.
- Reporting (Anton, via the secretary): #rna-age > state is the runner's kept post; #rna-age > milestones is the
  lead's, one line per landed result with its meaning (first post 12:50; figures from scripts/milestone_figures.py).
- RM8 waits on Anton: pretrain once or stop KronLSTM (default stop, 2026-11-12).
- kron-fm read from `../kron-fm`, not a dependency. Kit #411 open.
