# Lead notes

The state the lead's last holder left. State, not rules: the rules are the priming, `LEAD.md` beside this file, and
the kit's role `classes/lead/LEAD.md`.

## 2026-11-11 09:30

- Plan: RM1 seeds on the clean table (runner), RM8 sweep (2048/4096 running; 512: val 14.4, far behind), RM9 after
  RM8, RM12 re-training (runner; MLPs and ensembles done, fine-tune seeds pending), RM5 PRIMO (desk #9), RM7 NCBI
  (waits until 2026-11-17). RM10 and RM11 done.
- Results to keep straight: clean-table headline test 7.46, r 0.88 (pre-D4 sets read 9.2: never compare across).
  RM10: the gap tracks disease stage in the affected tissue (fibrosis x2, Braak, Lewy); no intervention reverses it
  in months (9 paired tests, none passes, several opposite n.s.); the health clock reads acute inflammation (fails
  the exercise control). Claim "tracks stage", never "responds to intervention".
- Desk: #9 PRIMO (default no), #18 BulkFormer-only spending (default: only where it earns its place).
- External samples: `data/input/external/` (six series), scored by MLPs with `--external`; fine-tunes cannot.
- kron-fm read from `../kron-fm`, not a dependency. Kit #411 open.
