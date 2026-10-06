# LSR: Local Subspace Reasoning

Research code accompanying **Local Subspace Reasoning with Adaptive Guidance for Robust Materials Property Prediction**.

LSR predicts materials properties from small, heterogeneous experimental datasets by combining query-specific local evidence, frozen scientific guidance and multi-scale neighbourhood statistics. In the reported CO2 adsorption experiments, an XGBoost prediction supplies the numerical reference that the language model refines. The language model is used through in-context learning, without task-specific fine-tuning.

The implementation includes two case-specific workflows: biomass-derived activated carbons (**Biochar**) and amine-functionalized adsorbents for direct air capture (**DAC**).

## Method

For each query, LSR:

1. Standardizes descriptors using the training partition and retrieves nearby labelled training observations.
2. Summarizes the neighbour targets at several nested neighbourhood sizes.
3. Combines the local examples and statistics with the query features, a numerical reference and frozen scientific rules.
4. Requests one CO2-uptake estimate from the language model and validates its numeric range.

Biochar uses four manually curated physical rules. DAC additionally supports residual-guided rule generation within an inner diagnostic split of the outer training partition. Generated rules are frozen before a separate prediction run. The released screening checks rule structure and evidence references; scientific plausibility still requires author review.

All configured neighbourhood scales are supplied jointly. Local target dispersion describes the available evidence; it is not a predictive confidence interval.

## Manuscript results

The following values are reported in the manuscript for the final CO2-uptake configurations:

| Case | Samples / predictors | Train / test | Neighbourhood sizes | LSR R² | XGBoost reference R² |
| --- | --- | --- | --- | --- | --- |
| Biochar | 472 / 22 | 377 / 95 | 10, 20, 30 | 0.8502 | 0.8463 |
| DAC | 147 / 14 | 117 / 30 | 7, 11, 15 | 0.8234 | 0.7300 |

The Biochar result is comparable to the supervised reference. The DAC point estimate improves by ΔR² = 0.0934. These comparisons measure the complete hybrid workflow relative to its numerical reference, rather than an LLM trained independently of XGBoost.

These are manuscript-reported results, not scores generated during installation or testing. Exact result verification requires the corresponding processed tables, versioned partitions and saved predictions. Fresh LLM calls may produce different predictions.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev,analysis,live]"
python -m lsr_co2 --help
```

Live prediction and rule generation require an installed, authenticated Codex CLI with access to the requested model and support for the provider's permission-profile and isolation options. The configured model is `gpt-5.4` with `xhigh` reasoning effort. Both can be set on the command line. Local metric evaluation, tests and linearity analysis do not make LLM calls.

## Data preparation

This is a **code-only release**. Experimental datasets, saved predictions, raw model responses and the manuscript PDF are not bundled. Place authorized processed tables at:

```text
data/processed/biochar_472.csv
data/processed/dac_with_xgb.csv
```

The exact feature columns, target, split settings, scales and output ranges are declared in [configs/](configs/). See [data/README.md](data/README.md) for the data contract.

Biochar uses an 80:20 random holdout with seed 42 and fits its XGBoost reference on the training partition. DAC uses the table's `split` column (`train` / `test`) and requires the saved `xgb_pred` reference column. The provenance of that reference must be supplied with the data. For new evaluations, fit and select the numerical reference using training data only.

To keep data elsewhere, edit `data_path` in a local configuration to an absolute path. If copying configs, also check `rules_path` and `experiment_config`: relative paths resolve from the configuration directory's parent. Preserve units and categorical encodings when preparing the tables.

## Usage

Run the following commands from the repository root. Write generated files below the ignored `artifacts/` directory.

### Validate a dataset

```bash
python -m lsr_co2 inspect-data --config configs/biochar.json
python -m lsr_co2 inspect-data --config configs/dac.json
```

### Predict CO2 uptake with frozen guidance

```bash
python -m lsr_co2 run --config configs/biochar.json --output artifacts/biochar-predictions.csv --model gpt-5.4 --resume
python -m lsr_co2 run --config configs/dac.json --output artifacts/dac-predictions.csv --model gpt-5.4 --resume
```

The CSV records the target, reference prediction, LSR prediction and status. A separate metadata file records run settings, metrics and token usage. Provider errors stop the run by default; fallback and raw-response saving are explicit options. `--resume` skips completed sample IDs in an existing output, so reuse a file only with the same data, configuration and model.

### Generate new DAC rules

```bash
python -m lsr_co2 refine --config configs/dac_refinement.json --output-dir artifacts/dac-refinement --model gpt-5.4
python -m lsr_co2 run --config configs/dac.json --rules-path artifacts/dac-refinement/frozen_rules.md --output artifacts/dac-new-rules.csv --model gpt-5.4
```

The manuscript's DAC refinement design uses 93 sub-training and 24 diagnostic observations within the 117-row outer training partition. The released command writes diagnostic predictions, screened candidate rules, frozen rules and provenance metadata. It does not evaluate the locked test or automatically replace the default rule file. Review newly generated rules before using them in the separate prediction command.

### Analyse local linearity

```bash
python -m lsr_co2 analyze-linearity --config configs/biochar_linearity.json --output-dir artifacts/biochar-linearity
```

This analysis uses 19 process/composition predictors and four targets. It separates descriptive support-fit R² from held-out prediction R². Final Biochar CO2 prediction instead uses 22 predictors, including measured textural properties.

### Evaluate saved predictions

```bash
python -m lsr_co2 evaluate artifacts/biochar-predictions.csv --prediction-column biochar_v3_pred
```

The evaluator reports R², mean absolute error and signed bias. To validate a separately supplied artifact manifest, use `python -m lsr_co2 verify --manifest artifacts/manifest.json`.

## Repository contents

```text
configs/        Dataset protocols and configuration templates
rules/          Frozen Biochar guidance and DAC correction rules
src/lsr_co2/    Retrieval, prompts, providers, refinement, analysis and CLI
tests/          Synthetic-data unit and integration tests
scripts/        Secret scanning and conceptual method-figure generation
data/README.md  Processed-data contract
docs/           Methodology, refinement and release documentation
```

This repository distributes the reusable core implementation. Historical baseline sweeps, paired-bootstrap inputs, repeated-split result archives and original-run transcripts are not part of this snapshot. Adapting LSR to another material system requires updating its dataset validation, descriptor schema, prompt rendering and scientific guidance.

## Validation and security

```bash
python -m ruff check src tests scripts
python -m pytest
python scripts/check_secrets.py --history
```

Tests construct synthetic Biochar and DAC data and can run without experimental datasets or a live model account. CI runs the same checks.

The live provider starts each request in a temporary workspace, filters the child-process environment and uses a strict permission profile. Generated artifacts are ignored by Git. See [SECURITY.md](SECURITY.md) for credential-handling and runtime requirements, [docs/methodology.md](docs/methodology.md) for protocol boundaries and [docs/refinement.md](docs/refinement.md) for rule generation.

## Citation

Until a publication or archived-release identifier is available, cite the manuscript as:

```bibtex
@unpublished{li2026lsr,
  title  = {Local Subspace Reasoning with Adaptive Guidance for Robust Materials Property Prediction},
  author = {Li, Yuanming and Zhang, Shengyue and Li, Chunfeng and Zhou, Yihang and Deng, Shuai and Li, Shuangjun},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

No software or data license has been selected for this release. Availability of the repository does not grant a reuse license for the code or the underlying experimental records.
