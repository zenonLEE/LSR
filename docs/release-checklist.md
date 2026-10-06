# Public Release Checklist

## Required Before Public Release

- [ ] Confirm that every literature-derived data row may be redistributed.
- [ ] Add dataset citations and a complete data dictionary with units.
- [ ] Select software and data licenses.
- [ ] Add final title, authors, DOI/preprint URL and `CITATION.cff`.
- [ ] Confirm no private dataset, prediction, manifest or run metadata is tracked; include only approved publication figures.
- [ ] Recheck manuscript tables against the separately governed private evidence archive.
- [ ] Check that each reported score is paired with its original prediction and split artifacts.

## Automated Checks

```bash
python -m pip install -e ".[dev,analysis,live]"
ruff check src tests scripts
python scripts/check_secrets.py --history
pytest
```

- [ ] Confirm live `run` and `refine` outputs are outside the repository or below `artifacts/`.
- [ ] Confirm generated response JSONL files remain untracked.
- [ ] Review dependency and GitHub Actions updates opened by Dependabot.

Do not upload raw Codex/Claude transcripts. Historical chats can contain credentials or VPN
configuration even when research files are clean.
