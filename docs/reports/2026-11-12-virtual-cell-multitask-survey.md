# Multitask objectives for the aging clocks: what the virtual-cell models of 2025-2026 suggest

Survey made 2026-11-12 for Anton's question ("which multitask learning should we push it through, ideally to predict
age-related perturbations"), by a research agent of the runner seat. Labels: **[page]** read on the source page,
**[snippet]** search-result summary only, **[memory]** the agent's prior knowledge, not re-checked. Nothing here was
run by us; every claim is the source's own.

## The 2026 models

- **MaxToki** (Theodoris lab / NVIDIA, 2026) [page] https://www.theodorislab.gladstone.org/post/maxtoki: 217M and 1B
  parameters, ~100M cell-state trajectories from ~3,800 people (birth to the tenth decade); tasks: predict the elapsed
  time to the next state, and generate the state after a given time, both directions. In-silico screen: a gene is
  pro-aging if its perturbed state is reached faster. Validated in human cardiomyocytes and young mice. Weights
  `theodoris-lab/MaxToki`, licence unnamed. The closest existing age-perturbation model.
- **X-Cell** (Xaira, 2026-03-17) [page] https://xaira-therapeutics.github.io/X-Cell: diffusion model, 83M to 4.9B
  parameters; 25.6M perturbed cells from 7 genome-wide CRISPRi screens (cell lines incl. Jurkat). Claims zero-shot
  transfer to primary CD4 T cells, confirmed for several hits by an independent primary T-cell screen. Weights "coming
  soon", CC BY-NC-SA 4.0.
- **Stack** (Arc, 2026-01-09) [page] https://arcinstitute.org/news/foundation-model-stack: in-context prompting with
  sets of cells; 149M cells pretraining, 55M post-training (CellxGene + Parse PBMC). No numbers or licence on the page.
- **Lingshu-Cell** (arXiv 2603.25240, 2026-03) [page]: masked discrete diffusion over ~18,000 genes; claims leading
  results on the Virtual Cell Challenge H1 task and PBMC cytokine responses. No sizes or licence in the abstract.
- **PerturbGen** (bioRxiv 2026-03-04) [page]
  https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1.full: predicts how a perturbation reshapes downstream
  states along a trajectory; ~107M cells. In vivo LPS validation in 6 donors (IL1B knockout reverses later
  inflammatory pathways, 75-77 % accuracy).
- **"Virtual Cells Need Context, Not Just Scale"** (bioRxiv 2026-02-04) [page]
  https://www.biorxiv.org/content/10.64898/2026.02.04.703804v1.full: **on our Marson CD4 Perturb-seq data**, a latent
  diffusion model beats mean baselines on Δ metrics but recalls only ~9 % of the differentially expressed genes;
  coverage of contexts (donor × condition) drives generalisation, not cell count.
- **"Tabular Foundation Models Are Competitive Cellular Perturbation Predictors"** (bioRxiv 2026-07-01) [page]
  https://www.biorxiv.org/content/10.64898/2026.06.28.735106v1: TabICL and TabPFN as plain regressors match or beat
  specialised perturbation models on pseudobulk prediction over five Perturb-seq sets, including the CD4 T-cell screen.
- 2025 baselines: State (Arc; MMD on cell sets; non-commercial), Tahoe-x1 (masked expression + drug token, up to 3B,
  Apache 2.0), scGPT, Geneformer, scFoundation, UCE, CellFM, GEARS, CPA [page/memory].

## Does it buy biology? Mostly not yet

- Five foundation models and two deep models beat neither linear nor additive baselines on perturbation prediction
  (Ahlmann-Eltze et al., Nature Methods 2025) [snippet]; the training mean beats scGPT and scFoundation (Csendes et al.,
  BMC Genomics 2025) [snippet]; zero-shot embeddings give no consistent gain (PertEval-scFM, ICML 2025) [snippet].
- Systema (Nature Biotechnology 2025): scores inflated by the systematic perturbed-vs-control shift [snippet].
- "Diversity by Design" (arXiv 2506.22641): DEG-weighted MSE instead of MSE lifted GEARS' weighted R²(Δ) on Norman19
  from −0.07 to 0.56 [page]: the one clear loss-level ablation.
- Virtual Cell Challenge 2025 wrap-up [page] https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up: almost
  nothing beat the mean on MAE; hybrid statistical-neural models and ESM-2 embeddings of the perturbed gene helped.
- Every 2026 gain is self-reported.

## Standard objectives (cell type, tissue, sex, donor/batch, masked expression)

No published ablation isolates them as training heads. Masked-expression pretraining has only indirect, negative
evidence on perturbation tasks. scCello (NeurIPS 2024) ablates cell-type/ontology objectives [snippet, numbers not
seen]. Reading: use these labels as conditioning inputs and evaluation strata; as heads only if our own ablation shows
a gain.

## Aging on a perturbation axis

- **Pasta** (Salignon et al., bioRxiv 2025-06) [snippet]: a ridge classifier on rank differences of same-tissue sample
  pairs ≥ 40 years apart (an age-shift ranking objective, applicable to bulk); screened ~3M CMAP L1000 profiles; two
  wet-lab confirmations. Code github.com/jsalignon/pasta.
- REVIVE (Aging 2025-11): a lasso clock scoring > 10,000 perturbation signatures [snippet].
- UCSF/Junevity TF Perturb-seq (CRISPRa + CRISPRi) in aged fibroblasts [snippet].
- No public human perturbation screen uses young vs old donors: the clearest data gap.

## Datasets

Marson CD4 CRISPRi (in hand; 22M cells, 4 donors, 3 conditions); Parse 10M PBMC cytokines (12 donors, 90 cytokines,
CC BY-NC 4.0); Tahoe-100M (drugs on cancer lines, CC0 per card); X-Atlas/Pisces (25.6M cells, non-commercial);
Replogle 2022 (K562/RPE1); Schmidt/Steinhart 2022 primary T-cell CRISPRa (Perturb-seq on ~70 hits only) [snippets].

## Ranked recommendation (the agent's)

Every test: three references (mean Δ over training knockdowns, ridge, the clock without the head); hold out genes and
a donor/condition context.

1. **Perturbation-Δ head** on a shared encoder, Marson CD4 pseudobulk: control profile + ESM-2 embedding of the
   knocked-down gene → Δ expression; DEG-weighted MSE. Must beat mean and ridge on held-out genes (report DEG recall,
   not only Δ correlation); then age MAE within ~0.2 y, the patient-control gap, and ρ with the screen's aging effects
   on held-out knockdowns (now ~0.26).
2. **Pasta-style age-shift ranking**: the order of clock(control + Δ) − clock(control) should match the screen's
   aging effects on training knockdowns; placebo with effects shuffled across genes.
3. **Condition/cytokine discrimination head** (Parse PBMC + Marson's stimulations); check PBS controls do not move.
4. **Context conditioning** (donor, tissue, sex, study) with whole contexts held out, instead of a batch head.
5. **Continued masked-expression pretraining**, lowest priority, only as a regulariser.

Not yet: cross-species objectives; trajectory objectives (MaxToki, PerturbGen) until donor-age-ordered pseudobulk
pairs are built from the single-cell cohorts.
