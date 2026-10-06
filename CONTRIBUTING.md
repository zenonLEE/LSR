# Contributing

Keep scientific-protocol changes separate from software-only refactors. Changes to splits,
features, neighbor scales, rules, prompts, baseline parameters or parsing require a new experiment
version and must not overwrite separately archived paper artifacts.

Before a pull request, run `python -m ruff check src tests scripts`, `python -m pytest`, and
`python scripts/check_secrets.py --history`. If a private run manifest is supplied, also run
`python -m lsr_co2 verify --manifest <path>`. Record exact commands, output directories, model
identifiers, seeds, fallback counts and key metrics in the appropriate private run record.
