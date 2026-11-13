# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM26 — recount3's new adult bulk samples into the table, headline retrained

**State:** building (default of the lead's proposal in #rna-age > decisions, 2026-11-13 12:00, no answer) · **Owner:**
runner (source, parse, GPU runs), lead reviews the source before merge

The 2026-11-12 inventory found 4,430 recount3 bulk samples (healthy or no disease stated, adult, exact age) not in our
table. A `recount3` source: gene sums of those projects, coverage converted to read counts by mapped read length,
parsed into our schema; de-duplicated against ARCHS4 and GTEx by SRA run and experiment id; the same annotation flags
(tumour, cell line, single cell, prenatal) and the D4 rules. Splits by the same study hash, so no current study
changes side. Bar: the headline recipe retrained on the enlarged table lowers test MAE on the **unchanged** test
samples beyond the paired bootstrap's 95 % CI (by study); recount3 studies that hash to test are scored separately as
new unseen studies, never mixed into the comparison. Source reviewed and passed 2026-11-13 (c87fe6c: 3,289 samples, 198 studies). Both arms use only the
genes recount3 measures (GENCODE v26; ~1,600 of ours are absent there), the control arm being the old table on that
gene set, so the comparison isolates the samples; plasma cell-free RNA (SRP223445) stays out of training.

## RM27 — An age-shift ranking objective on bulk (after RM26)

**State:** waits until RM26 is scored · **Owner:** lead (code), runner (GPU runs)

Pasta-style (Salignon et al. 2025): same-tissue sample pairs far apart in age, the network trained to order them, as
a second head beside age. Bar: a within-study aging-linked disease gap at least D6's +2.9 y on the same 33 unseen
studies with test MAE no more than 0.5 y worse than the plain recipe.

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
