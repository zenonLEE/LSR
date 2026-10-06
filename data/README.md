# Private Data Contract

Experimental datasets are intentionally excluded from Git and ignored under `data/raw/` and
`data/processed/`.

The default configs expect authorized local copies at:

- `processed/biochar_472.csv`
- `processed/dac_with_xgb.csv`

The Biochar table must provide the 19 process/composition columns and four property columns listed
in `configs/biochar_linearity.json`. The main Biochar config uses the same process/composition
columns plus the three textural properties to predict `CO2_uptake`.

The DAC table must provide the feature, target, split and reference-prediction columns listed in
`configs/dac.json`. Training rows may leave the reference-prediction column empty when the selected
workflow fits its own training-only anchor.

Use a private config with an absolute `data_path` when local data should remain outside the checkout.
Do not commit literature-derived records without row-level citations, a units/codebook table and an
explicit redistribution license.
