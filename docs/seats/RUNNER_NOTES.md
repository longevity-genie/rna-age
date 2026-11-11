# Runner notes

The state the runner's last holder left. State, not rules: the rules are the priming, `RUNNER.md` beside this
file, and the kit's role `classes/runner/RUNNER.md`.

_Seated 2026-11-10 21:10 CET by the session that ran the overnight loop (session 4d5dba17); kit v0.1.169._

## The brief

`/data/uq2347/briefs/rna-age-overnight.md` (outside the repository). Worked with `STATUS.md`'s "Next".
Standing words of the maintainers recorded there and in `docs/experiments.md`: parquet only, no cancer cohorts,
≤ 4 GPUs at once and never the A100 nodes, ≤ ~64 CPU cores, NCBI paced, the Synapse token read at run time only and
never written to Synapse; tabular foundation models paused (Anton, 2026-11-10 19:30); PRIMO submission only on
Anton's go (HF login, model name, email).

## Share

The rigger's queue map (`~/.config/agent-seat-blueprints/rigger/queue.toml`): rna-age has a soft CPU share of 25
cores on the account's cap; GPU jobs are outside it. GPU jobs go to the preemptible requeue partition on gpu1-10
(RTX 2080 Ti 11 GB), excluding the A100 nodes; names only in `slurm/env.local.sh`.

## Running (2026-11-11 09:40 CET)

All watched by the seat's jobs watcher (`[seat.runner.watch.jobs]`, kit v0.1.183); every job carries the account's
comment (seat / what / eta / cuda).

| job | what |
|---|---|
| 6429117 | RM1 health-aware fine-tune seed 1, RM12 table (`atlas-ft-u4-rank1-s1-t2`) |
| 6428869 | RM1 seed 2, RM12 table (`atlas-ft-u4-rank1-s2`) |
| 6429118 | RM1 seed 0 again, RM12 table (`atlas-ft-u4-rank1-s0-t2`), queued after seed 2 |
| 6429023 | RM8 sweep, 2,048 tokens, 8 epochs, batch 32 as micro 16 |
| 6429024 | RM8 sweep, 4,096 tokens, 8 epochs, batch 32 as micro 8 |

## Cluster facts found by trial

| fact | sent to the rigger | in its inventory |
|---|---|---|
| no git on compute nodes | 2026-11-10 21:15 | listed as unchecked |
| 2080 Ti 11 GB: ~400M-param tabular FMs with 4k-row prompts run out of memory | 2026-11-10 21:15 | no |
| CPU jobs pend `AssocGrpCpuLimit` at times; GPU-node CPU jobs on the requeue partition as fallback | 2026-11-10 21:15 | yes (cap), fallback not |
| a fine-tune slowed to 2.3 s/step (from 1.3) alone on gpu1 card 0 after restart | 2026-11-10 21:15 | no; unexplained |
| the repository's default job limit (`RNA_AGE_TIME`, 4 h) killed three long jobs: size `TIME=` from the measured rate | not a cluster fact | — |
| jobs starting together on one node race for card 0 (a job shows on its card only after loading data); fixed with a lock and per-card claims in `slurm/job.sbatch` (937866e) | 2026-11-11 05:40 | was on its unchecked list |
| `scontrol update JobId= Comment=` works on one's own running job; `sacct` keeps no comment | rule came from the rigger | yes |

## Open items

- `unattended`: health-aware fine-tune → its health eval vs the health-aware MLP (+2.3 to +2.6 y); resubmit
  under the same name if it reaches its limit (it resumes).
- `unattended`: composition outside blood (brain, lung) once the pseudobulk jobs end; log in `docs/experiments.md`.
- `blocked: Anton`: PRIMO submission (HF login, model name, email); tabular FMs paused.
- `unattended`: report page and `STATUS.md` republished after each result.
