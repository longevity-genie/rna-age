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

## Running (21:15 CET)

| job | what | watcher |
|---|---|---|
| 6427754 | `atlas-ft-u4-rank1`: health-aware Atlas fine-tune (ranking loss), resumed at epoch 2, 10 h limit; epoch 0 val 12.59, epoch 1 11.72 | background task beapljd68 (runs `rna-age` health eval at the end itself) |
| 6427856-58 | cell-type pseudobulk: PsychENCODE MSSM brain (1,023 donors), PsychENCODE 5c97eeeb brain, ILD lung | background task baybnzgra, which then runs `rna-age composition ... --mapping cell_type` on each |

## Cluster facts found by trial

| fact | sent to the rigger | in its inventory |
|---|---|---|
| no git on compute nodes | 2026-11-10 21:15 | listed as unchecked |
| 2080 Ti 11 GB: ~400M-param tabular FMs with 4k-row prompts run out of memory | 2026-11-10 21:15 | no |
| CPU jobs pend `AssocGrpCpuLimit` at times; GPU-node CPU jobs on the requeue partition as fallback | 2026-11-10 21:15 | yes (cap), fallback not |
| a fine-tune slowed to 2.3 s/step (from 1.3) alone on gpu1 card 0 after restart | 2026-11-10 21:15 | no; unexplained |
| the repository's default job limit (`RNA_AGE_TIME`, 4 h) killed three long jobs: size `TIME=` from the measured rate | not a cluster fact | — |

## Open items

- `unattended`: health-aware fine-tune → its health eval vs the health-aware MLP (+2.3 to +2.6 y); resubmit
  under the same name if it reaches its limit (it resumes).
- `unattended`: composition outside blood (brain, lung) once the pseudobulk jobs end; log in `docs/experiments.md`.
- `blocked: Anton`: PRIMO submission (HF login, model name, email); tabular FMs paused.
- `unattended`: report page and `STATUS.md` republished after each result.
