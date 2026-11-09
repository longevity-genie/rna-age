# Report tables (generated from parquet; do not edit)

## Samples by source and age type

| source | age_type | samples | eligible | studies | donors | healthy_eligible | pseudobulk |
|---|---|---|---|---|---|---|---|
| archs4 | binned | 1,203 | 795 | 44 | 754 | 152 | 0 |
| archs4 | censored | 129 | 81 | 60 | 81 | 0 | 0 |
| archs4 | exact | 149,572 | 74,293 | 1,676 | 60,159 | 7,571 | 0 |
| bgee | binned | 428 | 428 | 3 | 0 | 428 | 0 |
| bgee | exact | 825 | 825 | 37 | 0 | 825 | 0 |
| census_pseudobulk | binned | 19 | 19 | 1 | 15 | 19 | 19 |
| census_pseudobulk | censored | 52 | 52 | 1 | 52 | 8 | 52 |
| census_pseudobulk | exact | 2,967 | 2,958 | 12 | 2,937 | 2,241 | 2,967 |
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
| census_pseudobulk | censored | 0 | 52 | 0 |
| census_pseudobulk | exact | 0 | 1,465 | 1,493 |
| gtex | binned | 2,948 | 13,026 | 3,487 |
| terekhova2023 | exact | 0 | 317 | 0 |

## Cross-source duplicates removed

| source | kept_in | samples |
|---|---|---|
| bgee | archs4 | 753 |

## Experiments (sorted by validation MAE)

| run_id | model | features | val_exact_mae | test_exact_mae | test_exact_pearson_r | test_exact_bias_slope | test_binned_interval_hit | test_binned_mae_mid | transfer_exact_mae | transfer_exact_pearson_r |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261110-0052-ridge-hvg10000-gtexmid | ridge | logCPM HVG 10000 | 12.11 | 13.06 | 0.65 | -0.47 | 0.40 | 7.46 | 15.75 | 0.44 |
| 20261110-0051-ridge-hvg5000-gtexmid | ridge | logCPM HVG 5000 | 12.85 | 13.72 | 0.61 | -0.51 | 0.38 | 8.10 | 16.66 | 0.51 |
| 20261110-0050-ridge-hvg2000-gtexmid | ridge | logCPM HVG 2000 | 14.13 | 14.82 | 0.57 | -0.60 | 0.34 | 9.12 | 15.68 | 0.49 |
