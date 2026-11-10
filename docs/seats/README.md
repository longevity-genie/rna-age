# Seats

The standing roles agent sessions take in this repository, run by the seats kit (agent-seat-blueprints;
`docs/DECISIONS.md` D1). The seat table is `.blueprints.toml` at the root; a session takes a seat with
`/seats:seat NAME`, one session per seat.

| Seat | Role (the kit's) | Owns | Priming | Notes |
|---|---|---|---|---|
| `lead` | `lead` | `docs/ROADMAP.md`, `docs/DECISIONS.md`, the design of the code | [LEAD.md](LEAD.md) | [LEAD_NOTES.md](LEAD_NOTES.md) |
| `runner` | `runner` | the pipeline's jobs, `STATUS.md`, `docs/experiments.md`, `docs/reports/` | [RUNNER.md](RUNNER.md) | [RUNNER_NOTES.md](RUNNER_NOTES.md) |
