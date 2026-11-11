# Lead notes

The state the lead's last holder left. State, not rules: the rules are the priming, `LEAD.md` beside this file, and
the kit's role `classes/lead/LEAD.md`.

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
