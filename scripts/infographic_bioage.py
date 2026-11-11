"""Infographic for #rna-age (2026-11-11, Anton's ask): how good the transcriptomic age gap is, what it shows about
health, and how it sits beside methylation clocks. Our numbers are read from the runs; the methylation figure is
quoted from the literature and drawn hatched. Usage: ``uv run python scripts/infographic_bioage.py OUT_DIR``."""

import sys

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rna_age.paths import OUTPUT  # noqa: E402

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"
HEALTH = "health_20261111-0747-ens-health3t2-plus-atlasft-rank1_pred_null"
PLAIN = "health_20261111-0747-ens-mlp3t2-plus-atlasft2_pred_null"
LABELS = {
    "all_aging_linked": "All aging-linked diseases",
    "autoimmune_inflammatory": "Autoimmune / inflammatory",
    "neurodegenerative": "Neurodegenerative",
    "psychiatric": "Psychiatric",
    "cardiovascular": "Cardiovascular",
}


def main(out: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})  # fmt: skip
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})

    # A: chronological accuracy, ours (measured) vs the pan-tissue methylation clock (reported)
    names = [
        "Our RNA clock\n(held-out studies, many tissues)",
        "Methylation clock, Horvath 2013\n(reported, many tissues)",
    ]
    vals = [5.3, 3.6]  # median absolute error, the measure Horvath reports; ours: the clean-table headline
    bars = a.barh(names, vals, color=[S1, SURF], edgecolor=[S1, INK2], hatch=["", "///"], height=0.5)
    for bar, v in zip(bars, vals, strict=True):
        a.text(v + 0.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f} y", va="center", color=INK)
    a.set_xlim(0, 10)
    a.invert_yaxis()
    a.set_xlabel("median error in chronological age (years)")
    a.set_title("A. Age itself: methylation is sharper", loc="left", color=INK, fontsize=11)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # B: within-study gap, patients vs their own study's controls, unseen studies
    h = pl.read_parquet(OUTPUT / "summary" / f"{HEALTH}.parquet").filter(pl.col("category").is_in(list(LABELS)))
    h = h.filter(pl.col("studies") >= 4).sort("within_study_gap", descending=True).with_columns(clock=pl.lit("health"))
    plain = pl.read_parquet(OUTPUT / "summary" / f"{PLAIN}.parquet").filter(pl.col("category") == "all_aging_linked")
    first = h.filter(pl.col("category") == "all_aging_linked")
    rest = h.filter(pl.col("category") != "all_aging_linked")
    h = pl.concat(
        [first, plain.select(h.columns[:-1]).with_columns(clock=pl.lit("plain")), rest], how="vertical_relaxed"
    )
    ys = range(h.height)
    for y, row in zip(ys, h.iter_rows(named=True), strict=True):
        sig = row["ci_low"] > 0 or row["ci_high"] < 0
        col = S2 if row["clock"] == "health" else S1
        b.plot([row["ci_low"], row["ci_high"]], [y, y], color=col, lw=2, solid_capstyle="round")
        b.scatter(row["within_study_gap"], y, s=60, color=col if sig else SURF, edgecolor=col, linewidth=2, zorder=3)
        b.text(9.5, y, f"{row['within_study_gap']:+.1f} y, {row['studies']} studies", va="center", color=INK2,
               fontsize=9)  # fmt: skip
    b.axvline(0, color=INK2, lw=1)
    b.set_yticks(list(ys))
    b.set_yticklabels([LABELS[c] + (" (plain clock)" if k == "plain" else "")
                       for c, k in zip(h["category"], h["clock"], strict=True)], color=INK)  # fmt: skip
    b.invert_yaxis()
    b.set_xlim(-9, 15)
    b.grid(axis="x", color=GRID, lw=0.8)
    b.set_axisbelow(True)
    b.set_xlabel("patients minus their own study's controls (years, 95 % CI)")
    b.set_title("B. Health: patients read older on the health-aware clock", loc="left", color=INK, fontsize=11)
    b.scatter([], [], s=60, color=S2, label="health-aware clock")
    b.scatter([], [], s=60, color=S1, label="plain clock")
    b.scatter([], [], s=60, color=SURF, edgecolor=INK2, linewidth=2, label="CI includes 0")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, fontsize=9)
    for s in ("top", "right", "left"):
        b.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out}/bioage_infographic.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
