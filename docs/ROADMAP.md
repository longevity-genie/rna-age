# Roadmap

Open work only, in order. Each item is a section `## RMn — name` with a line saying its state and its owner. A done
item leaves this file; `STATUS.md` and `docs/experiments.md` say what landed and ran.

The lead (`docs/seats/LEAD.md`) keeps it. Seeded at the lead's first seating (2026-11-10) from `STATUS.md`'s "Next".

## Items

## RM34 — Positive-control datasets for system ageing: fetch and check which are usable

**State:** urgent (Anton, 2026-11-13, #rna-age > general chat: "Ok, download these data and validate which is
usable?") · **Owner:** lead (code), runner (downloads and parse jobs)

From RM33's dataset search (~30 candidates: conditions where a known system should look older, with controls in the
same study). Step 1, before any large download: read each CELLxGENE dataset's per-cell metadata over HTTP range
requests and each GEO series' sample characteristics (bulk series first looked up in our ARCHS4 table), and score it:
condition column and arms, donors per arm, share with exact ages, age overlap between arms (donors left after
matching), cells per donor × coarse type, assay/platform, whether the condition is confounded with a batch column
(D9), sex. Verdict per dataset: usable, usable with a caveat, or not, with the reason, in
`data/output/summary/positive_controls_usability.parquet`. Step 2: download only the usable ones (CELLxGENE h5ads
from the Census release or, if newer, the Discover API; GEO bulk from ARCHS4 where present), then per-donor metadata
and per-type pseudobulk with the existing `census-donor-obs` / `census-pseudobulk --by-cell-type`.
Bar: step 1 scores every candidate; a dataset counts as usable with ≥ 10 donors per arm after age matching, exact or
≤ 5-year ages, and the condition not confounded with batch.
Step 1 built (`rna-age positive-controls-scan`, 2026-11-14): every condition in the metadata, not a fixed list, from
CELLxGENE (148 human datasets with primary cells, a non-cancer disease and normal donors; obs over HTTP range
requests), ARCHS4 bulk series, and scBaseCount studies, scored per dataset × condition × tissue in
`positive_controls_usability.parquet`, rolled up per normalised condition (ontology id where present) in
`positive_controls_conditions.parquet` with `seed` marking the conditions RM33's hand search had. D9 is applied
strictly to single cell: a set without a two-level processing-batch column is at best "caveat".
Step 2 (Anton, 2026-11-14: downloads in parallel with RM33): `rna-age positive-controls-fetch` (23c0779) for the usable
and caveat single-cell sets, small ones first, Allen and ALTRA once the obs job shows their CMV column, SEA-AD deferred;
after each, per-donor obs, per-type pseudobulk and the verdict rerun on the full file. Bulk seeds the scan could not
use: curated case/control labels from GEO characteristics for series in ARCHS4 without them (GSE113957, GSE111016,
GSE167186, GSE114007) and NCBI-generated counts for those missing from ARCHS4 (GSE190125, GSE135251, GSE47460).
Step 3 (Anton, 2026-11-14: "did you search for scrna-seq on awesome singlecell, websearch, open data and others?"):
age-related diseases from outside reference lists searched per disease in GEO, and GEO single-cell series as a whole,
awesome-single-cell, Synapse, Zenodo/figshare and web search; the HCA/DISCO/SCEA/PBMCpedia scan merged; labels
normalised to disease families. Rule change, unless Anton objects: usability is scored per condition, studies pooled by
meta-analysis of within-study contrasts (study a random effect), so studies with ≥ 3 donors per arm count; per-donor
age may come from a paper's supplement.
Bulk seeds built (`rna-age positive-controls-bulk-labels [--counts]`, 2026-11-14): one rule per series on GEO's
characteristics → `data/input/positive_controls/bulk_labels.parquet`, which the scan uses instead of our ARCHS4
annotation for those series. Usable: GSE190125 (Down syndrome blood, 96 pairs, NCBI counts), GSE167186 and GSE111016
(sarcopenia, 24 and 20 pairs). Not usable: GSE113957 (HGPS: 9 of 10 lines from PRF, all controls Coriell),
GSE114007 (OA: 5 pairs, controls young), GSE135251 (MASLD: no ages in GEO, NCBI counts on disk), GSE47460
(microarray only). Positive-control series are kept out of clock training: their NCBI counts are not parsed into the
`ncbi_geo` source (its generic annotation would label trisomy-21 samples healthy).

## RM33 — System-level age gaps: which cell types or tissues make a person look older

**State:** building (design A; stage 0 run, a601870; its two guards run 2026-11-14, `sys-gap --guards`; CMV run 2026-11-14: CD8 T holds, B fails reproducibly; four more controls run 2026-11-20: T2D, PD, sarcopenia inconclusive, Down syndrome awaits a GPU job) · **Owner:** lead

Review (2026-11-13; a design agent wrote three designs, two separate reviewers judged them on our data and cluster):
A labelled cell types 6/10 validity, 8/10 feasibility; B label-free groups on our encoder 4/4 (encoder never saw brain,
lacks CD4/CD8A/KLRC2 and glial markers; CNA kept as a cross-check on A); C gene programs 3/3 (expected programs are
the disease's response programs: circular). Fixes taken into A: cases and controls scored by the same fold models;
every bar in years of acceleration (clock sensitivity differs 2.6× between CD8 T and monocytes), nulls as equivalence
tests, accuracy as non-inferiority; clocks retrained without the condition's DE genes (from another dataset) and on
disjoint gene halves; QC and ambient covariates, test-retest reliability where repeat visits exist; reliability floor
not tied to age range. Stages: 0 data on disk, ridge, shared vs specific gaps and AD in PsychAD (go/no-go: microglia
and astrocytes older than oligodendrocytes); 1 HLCA smoking within study, ILD; 2 Allen CMV after its obs column is
confirmed; 3 pseudocells and fine types only for passing systems. Datasets come from RM34.
Stage 0 (2026-11-14, a601870, `rna-age sys-gap`): reliability 0.75-0.98 for most types; shared person factor 26-62 %
of a type's gap in healthy cohorts; PsychAD AD − control: astrocytes +9.3 y, microglia +7.7, oligodendrocytes +5.7,
OPC +5.4, excitatory neurons +3.0; bar passes through astrocytes only (− oligodendrocytes 3.6 y [0.7, 6.6], − OPC 3.9
[0.6, 7.3]); microglia not beyond the CI. Provisional: every type is older in AD, and the leave-DE-genes-out and
gene-half clocks are not in yet; RADC unmeasurable (18 controls with exact ages). Next: those two guards, AD
replication on cff99df2 and the microglia atlas, then CMV on Allen.
Guards (2026-11-14, `rna-age sys-gap --guards`, `sys_gap_guards.parquet`, DE in `sys_gap_de.parquet`; same folds,
case assignment and bootstrap draws as stage 0, which the run reproduces exactly). Leave-DE-genes-out, DE from RADC
5e57cd50 and cff99df2 (AD vs control per type, `gene ~ case + age + sex [+ age-censored flag]`, Stouffer-combined;
every type had independent DE, no fallback used): the bar holds with 200 genes out (astrocyte − oligodendrocyte 3.1 y
[0.2, 6.2], − OPC 3.6 [0.3, 6.9]) and, barely, with 500 (2.9 [0.1, 6.0]; 3.3 [0.0, 6.7]); every type still older in
AD (astrocyte 8.7-9.0 y, microglia 7.6-8.0, oligodendrocyte 5.8, OPC 5.3). But the independent DE is weak (max |t|
4-5 per type, near the null maximum over ~22k genes; 10-19 of the 200 removed genes were among a clock's 2,000
candidates), so this guard has little power. Gene halves (5 splits × 2): the bar holds in 6 of 10 halves; astrocyte −
OPC is beyond the CI in all 10 (median 5.6 y, 4.8-6.5), astrocyte − oligodendrocyte in 6 (median 3.8, 1.5-5.8),
microglia − oligodendrocyte in 1 (median 1.9, −0.5-4.1). Reading: astrocytes > OPC is robust, astrocytes >
oligodendrocytes is not; the AD shift common to all types (5-11 y) dominates. Stage 0 bar: not robustly passed.
Overall bar (Anton, 2026-11-14, #rna-age > general chat: "it is ok where majority of systems behave well while several
fail reproducibly as our assumptions can be biased in some cases and in some cases it can be a finding rather than
bug"): RM33 is judged across all positive controls, not on one contrast. It passes when most pre-registered
condition × system predictions hold; a prediction that fails reproducibly (in every guard and replication) is reported
as a finding about the assumption, not hidden. Next: CMV in the Allen aging cohort (ff4235bb, downloaded; condition from
its CMV column, clocks on CMV-negative donors) with ALTRA (789ad837) as replication, then the other usable controls.
CMV pre-registration (2026-11-14, written before any CMV run). Condition from the donor metadata column `subject.cmv`:
cases `Positive`, controls `Negative`, both healthy by the disease label (the label minus "cytomegalovirus infection"
is `normal`; ALTRA's RA donors, and donors with CMV `Equivocal` or missing, leave the contrast). The person is the
subject (`subject.subjectGuid`); a subject with several samples enters clock and contrast with its first visit (lowest
age at draw), the later visits are scored by the same fold model and give each type's test-retest reliability
(Pearson r of the gaps of visits 1 and 2, CMV-negative subjects), beside the split-half reliability. Stage 0 as for AD:
per-type ridge clocks on CMV-negative donors, same fold and case-assignment rules, gaps in years, CMV+ − CMV− with
age, age², sex (+ `batch_id` in Allen; ALTRA has no batch-like column). Predictions, scored in Allen (ff4235bb), then
ALTRA as replication, and per gene-half guard in Allen: (1) CD8 T and (2) CD4 T accelerate in CMV+ — hold if the 95 %
CI is above 0, fail if it is below 0 or the 90 % CI lies inside ±2 y, else inconclusive; (3) CD8 T − monocyte > 0,
same rule; (4) monocytes and (5) B do not accelerate — hold if the 90 % CI lies inside ±2 y (equivalence), fail if
the 95 % CI lies wholly above 2 or below −2, else inconclusive; NK may accelerate: exploratory, scored by rule (1) but
not counted. CMV passes when (1)-(5) hold in Allen and none fails in ALTRA (ALTRA's ~23 CMV-negative controls make
inconclusive likely there). Output `sys_gap_predictions.parquet`.
CMV results (2026-11-14, `rna-age sys-gap --cohorts cmv --guards`). Allen: 225 subjects with exact ages (9 aged
"80 and over" leave; ages 40-89), 127 CMV+ / 96 CMV−, one sample per subject in this Census release (no repeat
visits, so no test-retest; split-half 0.81-0.99), batch_id (17 levels) as covariate. CMV+ − CMV− in years [95 % CI]:
CD8 T +4.9 [1.0, 9.1], CD4 T −1.2 [−3.9, 1.6], NK −5.4 [−10.1, −1.6], monocyte −6.7 [−25.5, 3.1] (slope 0.15: the
years scale is wide; −0.16 z [−0.39, 0.08]), B −8.7 [−18.1, −2.7], other T −0.5, DC −8.0 [−24.4, 1.1]; CD8 T −
monocyte +11.6 [2.5, 29.7], CD8 T − B +13.6 [6.6, 23.0]. Verdicts: (1) CD8 T hold, (2) CD4 T inconclusive (neither
above 0 nor inside ±2 y), (3) CD8 T − monocyte hold, (4) monocyte inconclusive (90 % CI −21.2-1.3), (5) B fail (B
look *younger* in CMV+), NK (exploratory) younger, fail. Gene halves (10): CD8 T holds in 9, CD8 T − monocyte in 10,
NK younger in 10, B younger beyond the CI in 8, CD4 T never above 0: the CD8 T signal and the B / NK shifts are
reproducible. ALTRA: 23 CMV-negative controls (25 healthy controls have no CMV status) give no usable clocks (B and
monocyte slopes ≤ 0, CD8 T r 0.44, CIs tens of years wide): every prediction inconclusive, none fails, so it neither
replicates nor contradicts. CMV not passed: 2 of 5 hold, B fails reproducibly — reported as a finding about the
assumption (CMV shifts the B and NK compartments, e.g. adaptive NK and memory-B composition within the coarse class,
the opposite way to the age trend), not hidden. Doubts: Allen is 10x Flex (probe-based) and ages 40+ only; ALTRA
pools a donor's visits into one pseudobulk; ALTRA's "healthy" include ACPA+ at-risk donors.
CMV fine-type pre-registration (2026-11-20, written before any fine-type run). Hypothesis: CMV shifts subset
composition inside a coarse class (more effector-memory CD8, adaptive NK, memory B), and a coarse clock reads that
shift as age. Test: the same pipeline in Allen (ff4235bb) with the dataset's own fine `cell_type` labels (uniform
within this cohort) as systems (`--level fine`), each its own system: naive CD8 T, effector memory CD8 T, naive CD4 T,
central memory CD4 T, effector memory CD4 T, memory Treg, MAIT, gamma-delta T, CD16+ CD56-dim NK, CD56-bright NK,
naive B, memory B, classical (CD14+) monocyte, CD16+ monocyte (≥ 20 cells in most donors). Same case definition,
clocks on CMV-negative donors only, same fold and case-assignment rules, age, age², sex, `batch_id`; gene-half guard
(5 splits × 2) on the counted predictions. Scoring as in the CMV pre-registration: "accelerates" holds if the 95 % CI
is above 0; "within ±2 y" holds if the 90 % CI lies inside ±2 y and fails if the 95 % CI lies wholly beyond it.
Predictions: (P1) effector memory CD8 T accelerates in CMV+; (P2) naive B and (P3) memory B within ±2 y (the coarse
B "younger" is composition); (P4) CD16+ CD56-dim NK within ±2 y (the coarse NK "younger" is composition); (P5)
classical monocytes within ±2 y. Exploratory, not counted: every other fine type, and composition: each fine type's
share of its coarse class, CMV+ − CMV−, adjusted for age, age², sex and batch (difference in share with a person-
bootstrap CI), to show which shifts drive the coarse results. Outputs: `level` column (coarse / fine) in
`sys_gap_type`, `sys_gap_contrast`, `sys_gap_predictions`; `sys_gap_composition.parquet`.
CMV fine-type results (2026-11-20, `rna-age sys-gap --cohorts ff4235bb… --level both --guards`; the coarse Allen rows
reproduce the committed ones exactly). All 14 fine clocks measured (reliability 0.78-0.98, slopes 0.15-0.54). CMV+ −
CMV− in years [95 % CI]: EM CD8 T +5.7 [−0.0, 11.8], naive CD8 T −1.9 [−6.4, 3.7], naive CD4 T −1.0 [−3.8, 1.7], CM
CD4 T −4.1 [−7.6, −0.9], EM CD4 T +0.2 [−2.6, 3.2], memory Treg −3.2 [−7.7, 0.4], MAIT +2.7 [−1.1, 7.7], gd T +3.1
[−1.8, 8.5], CD56dim NK −5.5 [−10.2, −1.6], CD56bright NK −9.0 [−17.1, −2.7], naive B −4.3 [−12.5, 1.7], memory B
−10.4 [−18.6, −4.7], classical monocyte −6.5 [−17.7, 0.5], CD16 monocyte −2.9 [−12.2, 4.8]; EM CD8 T − naive CD8 T
+7.5 [1.3, 13.7]. Verdicts (gene halves, 10): P1 EM CD8 T inconclusive (95 % CI touches 0; halves 0 hold, median
+0.2 y), P2 naive B inconclusive (10 inconclusive, median −5.4), P3 memory B fail (8 fail), P4 CD56dim NK
inconclusive (90 % CI −9.5 to −2.1; 10 inconclusive, median −4.5), P5 classical monocyte inconclusive (10). 0 of 5
hold, 1 fails. Composition (share of the coarse class, percentage points): CD8 T EM +15.2 [10.6, 19.7], naive −12.9
[−17.3, −8.4]; other T gd +15.3 [9.3, 21.1], MAIT −12.9 [−18.7, −7.2]; naive CD4 −4.5 [−8.7, −0.2], "CD4 memory T"
+6.1 [4.7, 7.7]; but B (memory +0.2 [−4.1, 5.1]), NK (CD56dim +0.9 [−0.8, 2.6]) and monocytes (classical −0.9)
do not shift. Reading: the hypothesis fails for B and NK: their labelled subsets do not move with CMV, and every B and
NK subset reads younger by itself (memory B reproducibly), so the coarse B / NK "younger" is not composition at this
label resolution (it may be a state inside a label, e.g. NKG2C+ adaptive NK within CD56dim, which Allen does not
label, or CMV programs that run against the age trend). For CD8 the composition story holds: the EM shift is large and
EM CD8 by itself is only weakly older (not robust to gene halves), so the coarse CD8 +4.9 y is partly composition.
Four more controls: pre-registration (2026-11-20, written before any run on these cohorts; only donors per arm and
which labels exist were looked at). Scoring as in the CMV pre-registration: "accelerates" holds if the 95 % CI is
above 0, fails if it is below 0 or the 90 % CI lies inside ±2 y, else inconclusive; "exploratory" is scored the same
way and not counted. Single cell (`sys-gap`, stage-0 rules unchanged: per-type ridge clocks on the cohort's own
controls, `min_controls` 20, ≥ 20 cells per donor and type, 5 folds by person, cases scored by their fold's model,
`accel ~ case + age + age² + sex [+ batch]` inside the arms' common age range, 2,000-draw person bootstrap; gene-half
guard, 5 splits × 2, on the counted predictions):
(A) Type 2 diabetes, islets (e51bae9a): cases "type 2 diabetes mellitus", controls `normal`, prediabetes donors out
of the counted contrast. Systems (dataset labels): beta, alpha, delta, PP, ductal, acinar, stellate, vascular
(endothelial). (T1) beta accelerates in T2D; (T2) beta − acinar > 0. Exploratory: every other type; prediabetes vs
normal. The cohort has 17 control donors, below `min_controls` (20): under the stage-0 rules no type gets a clock, so
T1 and T2 are scored inconclusive ("unmeasurable: 17 controls") and the bar is not lowered.
(B) Parkinson disease, five brain regions (d3cb449b: prefrontal, primary motor and visual cortex, medial globus
pallidus, dorsal motor nucleus of the vagus; 73 PD, 24 controls, 84-92 donors per region). Regions: one unit per
donor, each type's counts summed over all of the donor's regions (no region covariate; the regions a donor lacks
differ a little between donors). Systems: the PsychAD coarse classes; the dataset has no microglia label, its
"central nervous system macrophage" counts as microglia; "central nervous system neuron", ependymal, mural and
leukocyte are "other". Batch: `Brain_bank` (donor metadata; UD holds PD donors only) as covariate. (P1) microglia
accelerate in PD (counted). Exploratory: oligodendrocyte − OPC, microglia − oligodendrocyte, microglia − OPC, every
other type. Dopaminergic neurons are not labelled and the substantia nigra is not sampled: the DA-neuron prediction
is not measurable and not registered.
Bulk (whole tissue as one system, existing bulk clocks, no retraining). Gap: prediction minus the bias line of
prediction on age fitted on the study's own controls, in predicted years (one clock in one tissue: not divided by
the line's slope, which a narrow age range makes unstable; the slope is reported); case − control by OLS `gap ~ case
+ age + age² + sex [+ batch]` on the donors inside the arms' common age range, 2,000-draw donor bootstrap with the
control line refitted in every draw (the single-cell `contrast`).
(C) Down syndrome, whole blood (GSE190125, NCBI counts; Entrez → Ensembl by NCBI's annotation as `ncbi_geo.parse`;
never in a training table). The headline ensemble `20261111-0747-ens-mlp3t2-plus-atlasft2` cannot score new samples:
its three MLP members kept no weights, and refitting them is retraining. The clock here is its other half, the
Atlas fine-tune pair `20261110-1329-ens-atlasft-u4u6` (mean of `20261110-0555-atlas-ft-u4-all` and
`20261110-0533-atlas-ft-u6-all`, saved best checkpoints, test MAE 10.1 y against the headline's 7.5), applied as
trained (GPU). Batch: GEO's processing batch (A-D) as covariate. (D1) adults (≥ 18 y, line on adult controls):
trisomy-21 blood reads older (counted). Exploratory: children (< 18 y, line on child controls).
(D) Sarcopenia, skeletal muscle (GSE111016, GSE167186; ARCHS4 counts, labels in `bulk_labels.parquet`): the headline
ensemble's own predictions (`20261111-0747-ens-mlp3t2-plus-atlasft2`, commit d025d51), reused as they are. Every
labelled sample of both series is in the val split (none in train; val served early stopping and ensemble choice
only), so neither series counts as contaminated. No batch (one platform per series). (S1) GSE111016 and (S2)
GSE167186: sarcopenic muscle reads older than age-matched healthy muscle (counted, each). Exploratory: the same with
the Atlas fine-tune pair's predictions.
The overall tally after these runs counts every pre-registered counted prediction so far: AD stage 0 (the glia bar as
one prediction), CMV coarse (5), CMV fine (5), T2D (2), PD (1), Down syndrome (1), sarcopenia (2).
Four more controls: results (2026-11-20, `rna-age sys-gap --cohorts more --guards`, `rna-age bulk-controls-score`;
rows in the sys_gap tables, bulk ones with `level` `bulk` and the clock named). (A) T2D: 17 controls (16 with ≥ 20
endothelial cells, 9 with PP cells), every type below `min_controls`: no clock, T1 and T2 inconclusive
(unmeasurable), prediabetes not measurable either. (B) PD: 70 PD / 22-24 controls per type inside the common age
range (64-101; `Brain_bank` as batch). The controls give no age clock: out-of-fold r with age −0.33 to −0.06 and
slope ≤ 0 in every type but OPC (r 0.33, slope 0.12, so its years scale is wide: +20 y [3, 161], exploratory "hold",
not read). P1 microglia: slope ≤ 0, inconclusive (unmeasured), in the plain run and in all 10 gene halves. An
exploratory check per region (one region at a time, 17-22 controls) gives no clock either (r −0.62 to +0.21), so
pooling the regions is not what kills it: 24 controls aged 64-101 (median 74) are too few and too old. (D) Sarcopenia,
headline ensemble's own val predictions (gap in predicted years, case − control [95 % CI], donors in the common age
range): GSE111016 +1.6 y [−0.7, 3.8] (20 vs 16, ages 66-78; clock r with age on controls 0.26, slope 0.23), GSE167186
+0.9 [−1.6, 3.3] (20 vs 19, 67-87; r 0.94, slope 0.32): S1 and S2 inconclusive (both point older, neither beyond the
CI nor inside ±2 y); the Atlas fine-tune pair gives the same (+1.0 [−0.8, 2.9]; +1.1 [−3.4, 5.4]). (C) Down syndrome:
not run here. Scoring GSE190125 needs a GPU (the Atlas fine-tune pair) and NCBI's annotation (not on disk yet;
`bulk-controls-predict` fetches it); code and tests are in (`rna-age bulk-controls-predict GSE190125` as a GPU job, then
`rna-age bulk-controls-score`), D1 is pending.
Overall tally of counted predictions (16 scored, 1 pending): hold 3 (AD glia bar, plain run only, 6 of 10 gene halves;
CMV CD8 T; CMV CD8 T − monocyte), fail 2 (CMV coarse B, CMV fine memory B: both reproducible, reported as findings),
inconclusive 11 (CMV CD4 T and monocyte; CMV fine P1, P2, P4, P5; T2D T1, T2; PD P1; sarcopenia S1, S2), pending 1
(Down syndrome D1). Of the 5 decided, 3 hold; but most predictions are undecided, and every new control so far was
undecided for want of power (T2D, PD: too few controls for a cohort's own clock; sarcopenia: 16-19 controls in the
common age range with CIs of ±2-3 y). RM33 does not pass on this count; the limiting factor is controls per cohort,
not a contrary result. Next: the Down syndrome GPU job (96 adult pairs, the best-powered bulk control); for small
single-cell cohorts, a clock trained on other cohorts' healthy donors of the same tissue (a design change, to be
pre-registered before use) instead of the cohort's own controls.
External clocks: design and pre-registration (2026-11-20, written before any run with them; only donors per arm,
regions and labels were looked at). For a target cohort, each coarse type's ridge clock is trained on the healthy
(`normal`) donors of *other* cohorts of the same tissue, never on a donor of the target: brain pool PsychAD 37a17b78 +
HBCC 5c97eeeb (for a target that is one of them, the other only); blood pool OneK1K 3faad104 + AIDA c838aec3 +
Terekhova 2023 (loaded as stage 0 loads it). Training donors: exact ages, adults (≥ 18 y; HBCC's controls reach down
to infancy and every target but COVID is adult), ≥ 20 cells of the type. Genes: the intersection of the pool cohorts'
and the target's genes; log-CPM per unit over that intersection; the 2,000 most variable genes chosen and
standardised on the training donors; RidgeCV on age (stage-0 alphas). One clock per type and target scores every
target unit, cases and controls alike (no folds: nothing of the target is in training). Bias line and years scale
(slope) on the target's own controls, so platform, region and batch offsets are absorbed within study; contrast as
stage 0 (`accel ~ case + age + age² + sex [+ batch]`, arms' common age range, 2,000-draw person bootstrap with the line
refitted per draw). A type is measurable only if, in the target's controls (≥ 10 with the type), the line's slope is
> 0 and the reliability (binomial thinning of the target's counts, both halves scored by the same external clock,
Spearman-Brown) is ≥ 0.5; a counted prediction touching an unmeasured type is inconclusive whatever its CI. Gene-half
guard (5 splits × 2 of the common genes, each half its own external clock) on the counted predictions. Scoring as in
the CMV pre-registration. Rows carry `clock` (`own` for every earlier row, `external:brain` / `external:blood`).
Targets and predictions (counted unless marked; replications counted separately from first tests):
(X1) PD brain d3cb449b, regions summed per donor and `Brain_bank` as batch as before: microglia accelerate in PD
(P1, now measured with an external clock) and microglia − OPC > 0 (new, counted). First tests.
(X2) AD replication in RADC 5e57cd50 (exact ages only: 18 controls; donors aged "90 and over" leave clock line and
contrast) and (X3) in cff99df2 (Brodmann area 9 only, 14 controls / 28 AD: the region the DLPFC pool was trained on;
areas 17 and 7 left out rather than summed, since donors have 1-3 regions; cff99df2 gave DE genes for the PsychAD
guard, which does not touch this contrast): astrocytes accelerate in AD and astrocyte − OPC > 0, in each cohort
(4 replication predictions; the PsychAD stage-0 result).
(X4) CMV replication in ALTRA 789ad837 (case definition, first visits and persons as in the CMV run): CD8 T
accelerates, CD8 T − monocyte > 0, B within ±2 y (3 replication predictions; B is expected to fail as in Allen: a
reproducible failure is a finding). Exploratory: CD4 T, monocyte, NK.
(X5) COVID-19 9dbab10c, blood units only (157 cases, 25 controls): exploratory, per-type deltas, nothing counted
(acute infection). (X6) Allen ff4235bb with the external blood clock: the CMV coarse predictions scored but not
counted (robustness: does CD8 T hold with a clock never trained on Allen?).
T2D islets e51bae9a stays inconclusive: no second islet cohort is on disk (not downloaded in this step).

Design (Anton, same topic: "we have to modify design to allow idea of seeing age difference per system"): his
individual + group loss with the system (cell type; tissue in bulk) as the unit. (0) One clock per system on
chronological age; the person's age is a learned non-negative weighted sum of the system ages; loss = person error +
λ × each system's error; output per person: overall age, each system's bias-corrected gap with a CI split into the shared person-level part and the system-specific deviation, its weight. Bar:
the system-sum model stays within the paired CI of the whole-sample clock in at least 5 of 7 cohorts (RM31's plain
mean lost 0.3-0.7 y). Unit (Anton: "pseudocells ... make it balanced ... save computation but still provide per system distributions"):
a fixed number of 15-cell pseudocells per donor × system (e.g. 30, with replacement for small donors), so every
donor and system weighs the same and compute is fixed; each system's age per person is the mean over its pseudocells,
the spread gives its CI and shows mixed systems. Read by ridge and TabICLv2 on summed expression (Buckley 2023's
setting), against the per-type pseudobulk of RM31; similarity metacells (SEACells-style) as the variant that keeps
sub-states apart.
Annotation bias (Anton: "is it not biased on cell type detection and attribution of cell types to subsystems?"):
coarse classes first; a second, uniform annotation of every cohort by reference mapping (e.g. CellTypist, one
reference) beside the authors' labels, a system gap reported only where the two agree (r ≥ 0.7 across donors); each
type's share and the unassigned share against age, a type that drifts with age flagged; and label-free arms (Anton: "approaches that do not depend on how authors labeled cells"):
clusters of all cohorts' cells on RM28's cell embedding as systems, named afterwards by marker genes; covarying
neighbourhood analysis (CNA, Reshef 2022) for where in cell space age sits; gene programs (cNMF) as functional
systems if the clusters hold; learned soft groups last. Then the known-cause tests
below, on its system gaps.

RM31's per-cell-type clocks (out of fold, TabICLv2) give each donor an age per cell type; their age-adjusted gaps
correlate at r 0.2-0.5 between types, so part of a gap is type-specific, real or noise. Tests where the answer is
known: (1) Alzheimer donors vs controls in the brain cohorts 37a17b78 and 5e57cd50, within processing batch (D9). Bar:
at least one glial type (microglia, astrocyte) shows a larger AD gap than OPCs and oligodendrocytes, beyond a donor
bootstrap CI, in both cohorts. (2) CMV serostatus in Terekhova 2023, if its donor metadata carries it. Bar: the CD8 T
gap is higher in CMV+ beyond the CI and the monocyte gap is not. (3) GTEx per-donor tissue gaps from the bulk clock,
descriptive (how much organ ages agree within a person). CPU only.
Next (proposed to Anton 2026-11-20 in #rna-age > per-cell clock, built unless he objects): for cohorts too small for
own-control clocks, each cell type's clock is trained on other cohorts' healthy donors of the same tissue (pancreas:
islet atlases; brain: PsychAD and HBCC controls; blood: OneK1K, AIDA, Terekhova), the small cohort's cases and controls
scored with it and contrasted within study; its predictions pre-registered before the run. Down syndrome (D1) waits
on the runner's GPU job.

## RM13 — A manuscript draft of the results (plan line 5)

**State:** building (default of desk #23 applied 2026-11-13; draft v0.1 kept outside git, the repository being public; where it goes: desk #23) · **Owner:** lead

Outline, methods and figures regenerated from the recorded runs (D4 clean table, D6 health clock, RM10 tests),
kept private in the repository until Anton decides where it goes.

## RM5 — PRIMO submission

**State:** blocked: desk #9 (default: not submitted) · **Owner:** lead

Local scoring done (16 public tasks, no gain over frozen Atlas). Submitting is public and needs Anton's account.

## RM7 — NCBI-generated GEO counts (gap-filling beyond ARCHS4)

**State:** unattended (checked 2026-11-20: `/geo/download/` now redirects to NCBI's FTP copy, `ftp.ncbi.nlm.nih.gov/geo/rnaseq_counts/`, no reCAPTCHA; RM34 fetched two series this way on 2026-11-14) · **Owner:** lead (code), runner (fetch jobs)

`sources/ncbi_geo.py` resumes on NCBI's published FTP copy (the route the 2026-11-09 note asked for), one connection with pauses: cross-check targets and age-annotated series missing from ARCHS4; positive-control series stay out of training (RM34).
