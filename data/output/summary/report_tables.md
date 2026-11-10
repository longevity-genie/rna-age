# Report tables (generated from parquet; do not edit)

## Samples by source and age type

| source | age_type | samples | eligible | studies | donors | healthy_eligible | pseudobulk |
|---|---|---|---|---|---|---|---|
| archs4 | binned | 1,203 | 795 | 44 | 754 | 152 | 0 |
| archs4 | censored | 129 | 81 | 60 | 81 | 0 | 0 |
| archs4 | exact | 149,572 | 74,293 | 1,676 | 60,159 | 8,281 | 0 |
| bgee | binned | 428 | 428 | 3 | 0 | 428 | 0 |
| bgee | exact | 825 | 825 | 37 | 0 | 825 | 0 |
| census_pseudobulk | binned | 19 | 19 | 1 | 15 | 19 | 19 |
| census_pseudobulk | censored | 306 | 306 | 2 | 306 | 55 | 306 |
| census_pseudobulk | exact | 3,803 | 3,787 | 14 | 3,766 | 2,517 | 3,803 |
| gtex | binned | 19,788 | 19,461 | 946 | 946 | 19,461 | 0 |
| terekhova2023 | exact | 317 | 317 | 1 | 166 | 317 | 317 |

## Splits (eligible, de-duplicated)

| source | age_type | test | train | val |
|---|---|---|---|---|
| archs4 | binned | 346 | 441 | 8 |
| archs4 | censored | 8 | 60 | 13 |
| archs4 | exact | 10,053 | 50,949 | 13,291 |
| bgee | binned | 0 | 24 | 22 |
| bgee | exact | 99 | 275 | 80 |
| census_pseudobulk | binned | 0 | 0 | 19 |
| census_pseudobulk | censored | 0 | 306 | 0 |
| census_pseudobulk | exact | 0 | 2,294 | 1,493 |
| gtex | binned | 2,948 | 13,026 | 3,487 |
| terekhova2023 | exact | 0 | 317 | 0 |

## Cross-source duplicates removed

| source | kept_in | samples |
|---|---|---|
| bgee | archs4 | 753 |

## Experiments (sorted by validation MAE)

| run_id | model | features | val_exact_mae | test_exact_mae | test_exact_pearson_r | test_exact_bias_slope | test_binned_interval_hit | test_binned_mae_mid | transfer_exact_mae | transfer_exact_pearson_r |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261110-0304-ens-mlp20k-trainall-5 | ensemble (mean) | 20261110-0237-hvg20000-mlp-trainall+20261110-0248-hvg20000-mlp-trainall-seed1+20261110-0255-hvg20000-mlp-trainall-seed2+20261110-0302-hvg20000-mlp-trainall-seed3+20261110-0247-hvg20000-mlp-horvath-trainall | 9.02 | 9.63 | 0.79 | -0.33 | 0.46 | 6.25 | 15.02 | 0.60 |
| 20261110-0530-ens-mlp-trainall-plus-atlasft | ensemble (mean) | 20261110-0304-ens-mlp20k-trainall-5+20261110-0134-atlas-ft-u4 | 9.10 | 9.50 | 0.81 | -0.37 | 0.38 | 7.30 | 13.62 | 0.67 |
| 20261110-0530-hvg20000-mlp-trainall-seed5 | mlp probe | logCPM HVG 20000 | 9.14 | 9.49 | 0.80 | -0.33 | 0.45 | 6.39 | 15.82 | 0.51 |
| 20261110-0302-hvg20000-mlp-trainall-seed3 | mlp probe | logCPM HVG 20000 | 9.14 | 10.05 | 0.77 | -0.33 | 0.44 | 6.58 | 14.89 | 0.66 |
| 20261110-0312-hvg20000-mlp-h1024x256-trainall | mlp probe | logCPM HVG 20000 | 9.18 | 9.72 | 0.79 | -0.32 | 0.44 | 6.69 | 14.70 | 0.57 |
| 20261110-0255-hvg20000-mlp-trainall-seed2 | mlp probe | logCPM HVG 20000 | 9.22 | 9.56 | 0.79 | -0.33 | 0.44 | 6.45 | 15.13 | 0.57 |
| 20261110-0248-hvg20000-mlp-trainall-seed1 | mlp probe | logCPM HVG 20000 | 9.22 | 9.40 | 0.80 | -0.32 | 0.46 | 6.32 | 16.15 | 0.51 |
| 20261110-0312-atlas-cls-hvg-mlp-trainall | mlp probe | Atlas CLS + HVG 20000 | 9.22 | 9.80 | 0.78 | -0.35 | 0.45 | 6.48 | 18.61 | 0.54 |
| 20261110-0247-hvg20000-mlp-horvath-trainall | mlp probe | logCPM HVG 20000 | 9.22 | 10.14 | 0.77 | -0.33 | 0.46 | 6.31 | 13.60 | 0.71 |
| 20261110-0526-hvg20000-mlp-trainall-seed4 | mlp probe | logCPM HVG 20000 | 9.25 | 9.36 | 0.80 | -0.32 | 0.44 | 6.58 | 16.03 | 0.46 |
| 20261110-0237-hvg20000-mlp-trainall | mlp probe | logCPM HVG 20000 | 9.48 | 10.28 | 0.76 | -0.37 | 0.44 | 6.79 | 16.52 | 0.46 |
| 20261110-0240-ens-mlp20k-7 | ensemble (mean) | 20261110-0210-hvg20000-mlp+20261110-0217-hvg20000-mlp-seed1+20261110-0232-hvg20000-mlp-seed2+20261110-0236-hvg20000-mlp-seed3+20261110-0239-hvg20000-mlp-seed4+20261110-0237-hvg20000-mlp-trainall+20261110-0217-hvg20000-mlp-horvath | 9.83 | 10.26 | 0.77 | -0.38 | 0.44 | 6.44 | 14.65 | 0.62 |
| 20261110-0218-ens-mlp20k-3 | ensemble (mean) | 20261110-0210-hvg20000-mlp+20261110-0217-hvg20000-mlp-seed1+20261110-0217-hvg20000-mlp-horvath | 10.00 | 10.36 | 0.77 | -0.36 | 0.44 | 6.47 | 14.05 | 0.68 |
| 20261110-0240-ens-mlp20k-5seeds | ensemble (mean) | 20261110-0210-hvg20000-mlp+20261110-0217-hvg20000-mlp-seed1+20261110-0232-hvg20000-mlp-seed2+20261110-0236-hvg20000-mlp-seed3+20261110-0239-hvg20000-mlp-seed4 | 10.00 | 10.44 | 0.77 | -0.39 | 0.44 | 6.50 | 14.59 | 0.62 |
| 20261110-0220-tpmhvg20000-mlp | mlp probe | log2 TPM HVG 20000 | 10.02 | 10.24 | 0.77 | -0.36 | 0.44 | 6.62 | 22.07 | 0.09 |
| 20261110-0225-tpmhvg20000-mlp | mlp probe | log2 TPM HVG 20000 | 10.04 | 10.75 | 0.74 | -0.39 | 0.40 | 7.37 | 31.21 | 0.21 |
| 20261110-0217-hvg20000-mlp-seed1 | mlp probe | logCPM HVG 20000 | 10.04 | 10.64 | 0.76 | -0.38 | 0.45 | 6.51 | 14.04 | 0.66 |
| 20261110-0224-hvg20000-mlp-interval | mlp probe | logCPM HVG 20000 | 10.05 | 11.23 | 0.73 | -0.41 | 0.43 | 6.87 | 17.96 | 0.30 |
| 20261110-0210-hvg20000-mlp | mlp probe | logCPM HVG 20000 | 10.13 | 10.43 | 0.77 | -0.38 | 0.41 | 6.89 | 14.81 | 0.65 |
| 20261110-0232-hvg20000-mlp-seed2 | mlp probe | logCPM HVG 20000 | 10.26 | 10.60 | 0.76 | -0.36 | 0.43 | 6.61 | 14.64 | 0.62 |
| 20261110-0236-hvg20000-mlp-seed3 | mlp probe | logCPM HVG 20000 | 10.27 | 10.53 | 0.76 | -0.41 | 0.43 | 6.81 | 15.86 | 0.47 |
| 20261110-0239-hvg20000-mlp-seed4 | mlp probe | logCPM HVG 20000 | 10.31 | 11.14 | 0.73 | -0.41 | 0.42 | 6.93 | 14.52 | 0.62 |
| 20261110-0134-atlas-ft-u4 | Atlas fine-tune (last 4 blocks + head) | Atlas panel 4096 | 10.35 | 10.85 | 0.76 | -0.40 | 0.31 | 9.19 | 13.81 | 0.66 |
| 20261110-0229-hvg20000-mlp-studyw | mlp probe | logCPM HVG 20000 | 10.39 | 11.93 | 0.70 | -0.42 | 0.42 | 6.90 | 15.68 | 0.52 |
| 20261110-0217-hvg20000-mlp-horvath | mlp probe | logCPM HVG 20000 | 10.44 | 10.78 | 0.77 | -0.34 | 0.45 | 6.48 | 13.93 | 0.69 |
| 20261110-0145-atlas-panel-mlp | mlp probe | Atlas panel log2TPM (no encoder) | 10.69 | 11.45 | 0.75 | -0.37 | 0.38 | 7.53 | 16.30 | 0.48 |
| 20261110-0209-hvg4096-mlp | mlp probe | logCPM HVG 4096 | 11.07 | 11.88 | 0.71 | -0.42 | 0.44 | 7.11 | 20.39 | 0.18 |
| 20261110-0109-ridge-hvg20000-gtexmid-horvath | ridge | logCPM HVG 20000 | 11.57 | 12.64 | 0.65 | -0.41 | 0.43 | 6.85 | 14.16 | 0.54 |
| 20261110-0146-atlas-cls-hvg-mlp | mlp probe | Atlas CLS + HVG 1200 | 11.60 | 12.46 | 0.68 | -0.48 | 0.34 | 8.23 | 19.91 | 0.14 |
| 20261110-0057-ridge-hvg10000-gtexmid-horvath | ridge | logCPM HVG 10000 | 11.93 | 12.91 | 0.65 | -0.45 | 0.40 | 7.41 | 16.08 | 0.40 |
| 20261110-0059-ridge-hvg20000-gtexmid | ridge | logCPM HVG 20000 | 11.97 | 12.81 | 0.65 | -0.43 | 0.43 | 6.92 | 13.53 | 0.57 |
| 20261110-0151-ridge-hvg20000-gtexmid-horvath-withpb | ridge | logCPM HVG 20000 | 12.03 | 12.65 | 0.65 | -0.40 | 0.43 | 6.83 |  |  |
| 20261110-0052-ridge-hvg10000-gtexmid | ridge | logCPM HVG 10000 | 12.11 | 13.06 | 0.65 | -0.47 | 0.40 | 7.46 | 15.75 | 0.44 |
| 20261110-0131-atlas-cls-mlp | mlp probe | Atlas CLS | 12.63 | 13.69 | 0.63 | -0.54 | 0.35 | 8.61 | 16.99 | 0.23 |
| 20261110-0135-elasticnet-hvg5000-gtexmid | elasticnet | logCPM HVG 5000 | 12.82 | 13.40 | 0.64 | -0.53 | 0.37 | 8.11 | 15.59 | 0.53 |
| 20261110-0051-ridge-hvg5000-gtexmid | ridge | logCPM HVG 5000 | 12.85 | 13.72 | 0.61 | -0.51 | 0.38 | 8.10 | 16.66 | 0.51 |
| 20261110-0135-atlas-panel-ridge | ridge probe | Atlas panel log2TPM (no encoder) | 12.87 | 13.70 | 0.63 | -0.51 | 0.37 | 8.23 | 18.81 | 0.39 |
| 20261110-0134-atlas-ft-u2 | Atlas fine-tune (last 2 blocks + head) | Atlas panel 4096 | 13.22 | 13.67 | 0.66 | -0.49 | 0.34 | 9.78 | 16.54 | 0.61 |
| 20261110-0240-ens-mlp20k-7-cal | calibrated(20261110-0240-ens-mlp20k-7) | as base | 13.39 | 14.02 | 0.77 | -0.00 | 0.27 | 10.80 | 14.31 | 0.62 |
| 20261110-0133-atlas-cls-hvg-ridge | ridge probe | Atlas CLS + HVG 1200 | 13.62 | 14.61 | 0.59 | -0.57 | 0.32 | 9.20 | 17.66 | 0.28 |
| 20261110-0334-hgb-hvg5000-gtexmid | hgb | logCPM HVG 5000 | 13.70 | 13.81 | 0.65 | -0.61 | 0.36 | 8.56 | 16.50 | 0.58 |
| 20261110-0050-ridge-hvg2000-gtexmid | ridge | logCPM HVG 2000 | 14.13 | 14.82 | 0.57 | -0.60 | 0.34 | 9.12 | 15.68 | 0.49 |
| 20261110-0130-atlas-cls-ridge | ridge probe | Atlas CLS | 14.73 | 15.45 | 0.54 | -0.61 | 0.31 | 9.71 | 19.36 | 0.09 |

## tAge control (relative clock; r with age delta within stratum)

| source | split | model | subset | n | r_delta | strata |
|---|---|---|---|---|---|---|
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl | all | 8,643 | 0.33 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl | not_in_training | 8,561 | 0.33 | 184 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff.pkl | all | 8,643 | 0.34 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff.pkl | not_in_training | 8,561 | 0.33 | 184 |
| archs4 | test | 20261110-0109-ridge-hvg20000-gtexmid-horvath | all | 8,643 | 0.56 | 185 |
| archs4 | test | 20261110-0109-ridge-hvg20000-gtexmid-horvath | not_in_training | 8,561 | 0.56 | 184 |
| archs4 | test | 20261110-0134-atlas-ft-u4 | all | 8,643 | 0.61 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | all | 8,643 | 0.33 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | all | 8,643 | 0.34 | 185 |
| archs4 | test | 20261110-0134-atlas-ft-u4 | not_in_training | 8,561 | 0.60 | 184 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | not_in_training | 8,561 | 0.33 | 184 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | not_in_training | 8,561 | 0.33 | 184 |
| archs4 | test | 20261110-0210-hvg20000-mlp | all | 8,643 | 0.66 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | all | 8,643 | 0.33 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | all | 8,643 | 0.34 | 185 |
| archs4 | test | 20261110-0210-hvg20000-mlp | not_in_training | 8,561 | 0.66 | 184 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | not_in_training | 8,561 | 0.33 | 184 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | not_in_training | 8,561 | 0.33 | 184 |
| archs4 | test | 20261110-0304-ens-mlp20k-trainall-5 | all | 8,643 | 0.70 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | all | 8,643 | 0.33 | 185 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | all | 8,643 | 0.34 | 185 |
| archs4 | test | 20261110-0304-ens-mlp20k-trainall-5 | not_in_training | 8,561 | 0.70 | 184 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | not_in_training | 8,561 | 0.33 | 184 |
| archs4 | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | not_in_training | 8,561 | 0.33 | 184 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl | all | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl | not_in_training | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff.pkl | all | 2,926 | 0.28 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff.pkl | not_in_training | 2,926 | 0.28 | 52 |
| gtex | test | 20261110-0059-ridge-hvg20000-gtexmid | all | 2,926 | 0.73 | 52 |
| gtex | test | 20261110-0059-ridge-hvg20000-gtexmid | not_in_training | 2,926 | 0.73 | 52 |
| gtex | test | 20261110-0109-ridge-hvg20000-gtexmid-horvath | all | 2,926 | 0.73 | 52 |
| gtex | test | 20261110-0109-ridge-hvg20000-gtexmid-horvath | not_in_training | 2,926 | 0.73 | 52 |
| gtex | test | 20261110-0134-atlas-ft-u4 | all | 2,926 | 0.50 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | all | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | all | 2,926 | 0.28 | 52 |
| gtex | test | 20261110-0134-atlas-ft-u4 | not_in_training | 2,926 | 0.50 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | not_in_training | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | not_in_training | 2,926 | 0.28 | 52 |
| gtex | test | 20261110-0210-hvg20000-mlp | all | 2,926 | 0.76 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | all | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | all | 2,926 | 0.28 | 52 |
| gtex | test | 20261110-0210-hvg20000-mlp | not_in_training | 2,926 | 0.76 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | not_in_training | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | not_in_training | 2,926 | 0.28 | 52 |
| gtex | test | 20261110-0304-ens-mlp20k-trainall-5 | all | 2,926 | 0.77 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | all | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | all | 2,926 | 0.28 | 52 |
| gtex | test | 20261110-0304-ens-mlp20k-trainall-5 | not_in_training | 2,926 | 0.77 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_scaleddiff | not_in_training | 2,926 | 0.32 | 52 |
| gtex | test | EN_Chronoage_Multispecies_Multitissue_yugenediff | not_in_training | 2,926 | 0.28 | 52 |
