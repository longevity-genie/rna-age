"""Milestone figures for #rna-age > milestones (2026-11-11): predicted vs actual age of the headline on the clean
held-out set, and the pre-registered stage tests of both clocks. Usage: uv run python scripts/milestone_figures.py OUT_DIR"""

import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rna_age.paths import OUTPUT
from rna_age.train.features import sample_table

out = sys.argv[1]
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
    }
)
# 1: predicted vs actual, headline, clean test set
H = "20261111-0747-ens-mlp3t2-plus-atlasft2"
p = pl.read_parquet(OUTPUT / "runs" / H / "predictions.parquet").join(
    sample_table().select("sample_id"), on="sample_id", how="semi"
)
p = p.filter((pl.col("split") == "test") & (pl.col("age_type") == "exact") & ~pl.col("is_pseudobulk"))
a, y = p["age_years"].to_numpy(), p["pred"].to_numpy()
mae, r = np.abs(y - a).mean(), np.corrcoef(a, y)[0, 1]
fig, ax = plt.subplots(figsize=(5.2, 4.8))
hb = ax.hexbin(a, y, gridsize=40, cmap="Blues", mincnt=1, bins="log", linewidths=0)
ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls="--")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("chronological age (years)")
ax.set_ylabel("predicted age (years)")
ax.set_title(f"Held-out studies: MAE {mae:.1f} y, r {r:.2f} (n {len(a):,})", color=INK, loc="left", fontsize=11)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.colorbar(hb, ax=ax, label="samples (log)").outline.set_visible(False)
fig.tight_layout()
fig.savefig(f"{out}/headline_pred_vs_age.png", dpi=150)
# 2: pre-registered stage tests, rho per test, both clocks
b = pl.read_parquet(OUTPUT / "summary" / "biology_tests.parquet")
names = {
    "liver_fibrosis_a": "Liver fibrosis (cohort A)",
    "liver_fibrosis_b": "Liver fibrosis (cohort B)",
    "braak_ad": "Alzheimer, Braak stage",
    "lewy_stage": "Parkinson, Lewy stage",
    "lung_severity": "Lung disease severity",
    "covid_severity": "COVID severity (blood)",
    "viral_severity": "Acute viral severity (blood)",
    "ra_activity": "RA activity (blood)",
    "heart_failure_nyha": "Heart failure class",
    "acute_exercise_control": "One exercise bout (control)",
}
order = list(names)
fig, ax = plt.subplots(figsize=(6.6, 5.0))
for k, (run, lab, col, dy) in enumerate(
    [(H, "plain clock", S1, -0.14), ("20261111-0747-ens-health3t2-plus-atlasft-rank1", "health-aware clock", S2, 0.14)]
):
    d = {r["test"]: r for r in b.filter(pl.col("run_id") == run).to_dicts()}
    ys = np.arange(len(order)) + dy
    rho = [d[t]["rho"] for t in order]
    sig = [d[t]["p"] < 0.05 for t in order]
    ax.scatter(rho, ys, s=46, color=col, edgecolor=col, linewidth=1.5, label=lab, zorder=3)
    ax.scatter(
        [x for x, s_ in zip(rho, sig) if not s_],
        [yv for yv, s_ in zip(ys, sig) if not s_],
        s=46,
        color=SURF,
        edgecolor=col,
        linewidth=1.5,
        zorder=4,
    )
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([names[t] for t in order], color=INK)
ax.invert_yaxis()
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("Spearman rho, age gap vs disease stage (age removed)")
fig.suptitle(
    "Pre-registered tests: filled = p < 0.05, open = not significant", color=INK, x=0.02, ha="left", fontsize=11
)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(f"{out}/biology_stage_tests.png", dpi=150)
print(mae, r, len(a))
