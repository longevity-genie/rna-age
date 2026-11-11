# Lead notes

The state the lead's last holder left. State, not rules: the rules are the priming, `LEAD.md` beside this file, and
the kit's role `classes/lead/LEAD.md`.

## 2026-11-11 03:45

- Plan order: RM1 health-aware fine-tune (runner; eval after ~03:30), RM8 KronLSTM from scratch (code landed
  52b6f16; runs handed to the runner, smoke first), RM9 masked-gene pretraining probe (lead, after RM8), RM5 PRIMO
  (blocked: desk #9), RM7 NCBI GEO (waits until 2026-11-17).
- RM9 code landed (412030a), runs after RM8's sweep; nothing of the lead's left to build until RM1 and RM8 report.
- kron-fm is not a dependency (its uv sources pin torch cu128); `kron_core` is read from `../kron-fm` or
  `$RNA_AGE_KRON_FM`. The kernels have never run under our torch 2.15 / triton 3.9: RM8's smoke job checks it.
- Open with Anton (in #rna-age): the research plan (lines 1-5, criteria); his answer on line 4 (BulkFormer-specific
  work) and line 2 (gap biology) default to the desk if unanswered. D3's choices are listed for his review.
- Kit #411 (plan watcher cannot read desk questions): tooling seat's. Desk answers still arrive via watch.desk.
- Tests: 94 pass. The project venv is not synced while jobs run from it.
