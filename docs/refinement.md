# Residual-Guided Refinement

## Data Flow

`lsr-co2 refine` loads the configured table, isolates the outer-training partition and creates an
inner sub-training/diagnostic split. Scaling, neighbor retrieval and residual analysis use only
these training-derived partitions. Locked-test targets are not used for diagnostic prediction,
residual analysis or rule proposal.

The workflow has five stages:

1. Fit the local retriever on the inner sub-training partition.
2. Predict diagnostic rows with the current rule-free prompt, or load an explicitly supplied local
   diagnostic table.
3. Pair signed errors (`prediction - target`) with local mean and dispersion statistics.
4. Ask the text provider for broad, relative correction rules under the configured policy.
5. Validate rule count, evidence count, evidence IDs and required sections, then freeze accepted
   rule text and metadata.

The command does not evaluate a locked test. Evaluation remains a separate `lsr-co2 run` invocation
after rules have been frozen.

## Local Outputs

Each successful run writes `diagnostic_predictions.csv`, `candidate_rules.md`, `screening.json`,
`frozen_rules.md` and `refinement.metadata.json`. Optional prompts and raw provider responses are
also local artifacts. Write these files under `artifacts/` or another ignored private directory.

Metadata records data and prompt hashes, split settings, model name, token usage, diagnostic source
and structural-screening scope. It is evidence about a local run, not a file intended for this
public code repository.

## Evidence Boundary

Structural screening is not an independent chemistry validator. The `Physics:` sentence makes a
rule reviewable, but authors must assess plausibility and guard against narrow sample-specific
conditions before using it.

Running `refine` does not itself establish predictive improvement. Any manuscript metric must come
from a separately governed evaluation with a declared validation boundary.
