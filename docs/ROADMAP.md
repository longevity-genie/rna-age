# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM27 — An age-shift ranking objective on bulk (after RM26)

**State:** building (RM26 scored 2026-11-13) · **Owner:** lead (code), runner (GPU runs)

Pasta-style (Salignon et al. 2025): same-tissue sample pairs far apart in age, the network trained to order them, as
a second head beside age. Bar: a within-study aging-linked disease gap at least D6's +2.9 y on the same 33 unseen
studies with test MAE no more than 0.5 y worse than the plain recipe.

## RM28 — Our own single-cell pretraining: masked expression plus perturbations, then donor age

**State:** building (Anton, 2026-11-13, #rna-age > general chat: "single cell pretraining esp with masking part of
expression and with perturbations is worth trying") · **Owner:** lead (design, code), runner (GPU runs)

RM22 showed an off-the-shelf frozen embedding (Geneformer) loses to genes; this asks whether pretraining on our own
terms does better. Encoder over the 2,000 most variable genes of a cell (log-normalised), trained on (a) masked
expression: hide 15-30 % of a cell's genes, reconstruct them, on healthy adult cells from the pinned Census release
**excluding every donor of the 7 evaluation cohorts**; (b) perturbation: on the Marson CD4 Perturb-seq pseudobulk,
from the donor × condition non-targeting profile plus the knocked-down gene's token, predict the knockdown profile
(DEG-weighted loss, held-out genes). Ablation arms: masking only, masking + perturbation. Read-out: per-donor mean of
the cell embeddings, then ridge and TabICLv2, on the D7 folds, and fine-tuning the encoder on donor age.
Bars, fixed now: (1) the age clock on the pretrained encoder beats TabICLv2 on genes beyond the paired CI in at least
4 of 7 cohorts; (2) the perturbation arm adds to masking only if it raises (1) or D8's ρ on held-out knockdowns
beyond its CI. Compute: a few A100-days at most; stopped if the masked loss has not beaten a mean-gene baseline after
one epoch.

## RM13 — A manuscript draft of the results (plan line 5)

**State:** building (default of desk #23 applied 2026-11-13; draft v0.1 kept outside git, the repository being public; where it goes: desk #23) · **Owner:** lead

Outline, methods and figures regenerated from the recorded runs (D4 clean table, D6 health clock, RM10 tests),
kept private in the repository until Anton decides where it goes.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** waits until 2026-11-17: NCBI's `/geo/download/` answers with a reCAPTCHA since 2026-11-09; look again, never worked around · **Owner:** lead

`sources/ncbi_geo.py` is parked. An approved route (NCBI contact, a published FTP copy) is Anton's to ask for.
