# Decisions

The calls settled with the maintainers, one section each, `## Dn — the call (date)`, newest last: what was decided,
why, and the alternative it rejected. A call is never edited after the fact; a later one that changes it says which
it amends.

## D1 — Seats: a runner and a lead, tracked, on genie's organisation (2026-11-10)

The repository adopts the seats kit (agent-seat-blueprints, v0.1.168), with the kit's adoption wizard run by
winternewt from a laptop. The answers, as given:

- **Maintainers:** antonkulaga, the author, and winternewt. Either one's word decides.
- **Where the seats run:** on the cluster's login node, where the pipeline's jobs are submitted. The laptop the
  wizard ran on only drafted: no leak gate, no site id and no chat account were set up on it.
- **The organisation and the desk:** genie's (Anton's Zulip organisation, the desk `longevity-genie/desk`, whose
  secretary makes the seats' accounts). Rejected: no organisation, which would leave the seats' questions in their
  sessions, unseen from a phone.
- **Seats:**
  - `runner` (`docs/seats/RUNNER.md`): keeps the pipeline's jobs running, logs every run in `docs/experiments.md`
    and `data/output/runs/`, keeps `STATUS.md` and the report current, and judges nothing a result means beyond its
    correctness gates. Grant: commit, `git pull --ff-only`, push `main` as a fast-forward, its own jobs within its
    share, the report's private page. Designed from the kit's seat draft for this repository. The overnight loop
    already working the repository on the cluster becomes the runner, in place.
  - `lead` (`docs/seats/LEAD.md`), added by winternewt during the wizard ("add lead also"): owns the plan, a new
    `docs/ROADMAP.md`, and this file, designs what is built next with the maintainers and builds it. Grant: commit,
    push `main` as a fast-forward, release tags. Taken by a new session on the cluster. Rejected: the plan kept in
    `STATUS.md`, shared with the runner (two seats writing one file, which the kit's plan watcher cannot read).
- **Tracked, not an overlay:** `.claude/settings.json` pins the kit and `.blueprints.toml` is committed, so every
  clone, and every collaborator's agent, loads the kit. Winternewt's call, made although every commit of the last 30
  days is antonkulaga's (his agent's loop): the seats change those sessions too. Going to an overlay later is
  `bin/overlay.py untrack`, a pull request.
- **The team file** is on the desk (`teams/rna-age.toml`), not here: the repository is public.
- **Cluster facts stay off the trunk:** the runner's share is a placeholder until the organisation's rigger proposes
  it and the maintainers decide; partitions, the account and nodes stay in the git-ignored `slurm/env.local.sh`.
