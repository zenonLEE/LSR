# Methodology

## Biochar Protocol

The Biochar config uses a fixed random holdout, training-only standardization, local-neighbor
retrieval and multi-scale target summaries. A training-only surrogate may provide the numerical
anchor. Manually curated physical rules are frozen before query prediction.

## DAC Protocol

The DAC config uses a locked split column, training-only standardization, local-neighbor retrieval
and multi-scale summaries. A versioned or newly fitted training-only surrogate can provide the
anchor. Rule text is frozen before any separate locked-test inference.

The repository does not distribute either dataset. Exact features, split settings, neighbor scales,
output ranges and expected private paths are declared in `configs/`.

## Query-Centred Local Linearity

The local-linearity analysis excludes every analysed target from the retrieval features. For each
held-out query, it retrieves a training-only neighborhood after fitting the scaler on the training
partition.

Two quantities are intentionally separated:

1. Support-fit R2 is calculated by fitting and scoring Ridge regression on the same selected
   training support. It is a descriptive local-linearizability diagnostic, not held-out predictive
   performance.
2. Held-out R2 compares global and query-specific linear/nonlinear predictors on the query targets.
   Query targets are never used for fitting or neighbor selection.

Relative gain is calculated from full-precision held-out R2 values as
`100 * (local_linear_r2 - global_linear_r2) / global_linear_r2`. Values should be rounded only after
this calculation. Runtime CSVs and figures are written to an ignored output directory.

## Prediction Contract

A prompt contains physical context, frozen rules, neighborhood statistics, labeled training
neighbors, unlabeled query features and an optional numerical anchor. It never contains the query
target. The provider must return one finite number inside the configured range.

The anchor abstraction accepts either a training-only surrogate prediction or a local statistic
such as the neighborhood mean or median. The concrete configs describe which option is used; the
code does not claim that one reported anchor instantiation establishes all possible variants.

## Training-Only Prompt Refinement

The refinement loop operates inside the outer training partition. It predicts an inner diagnostic
subset, pairs signed residuals with local statistics, proposes broad relative correction rules,
checks rule structure and evidence IDs, and freezes accepted text. Physical rationales remain an
author-review responsibility.

Locked-test evaluation is a separate command after rule freezing. This separation makes the data
boundary executable rather than relying on comments in a one-off script.

## Leakage Boundary

No locked-test target enters retrieval fitting, diagnostic prediction, residual mining, rule
proposal or rule screening. Development choices informed by prior test inspection must still be
reported as development history rather than treated as pristine confirmatory validation.

Datasets, diagnostic predictions, locked-test predictions and research upper-bound artifacts remain
private and are not versioned in this code repository.

## Release Scope

This snapshot includes frozen-guidance prediction, DAC diagnostic refinement, metric evaluation and
Biochar local-linearity analysis. Historical baseline sweeps, paired-bootstrap inputs and repeated-
split result archives are not distributed here. The manuscript-reported scores require their
corresponding processed data, split definitions and saved predictions for exact verification.
