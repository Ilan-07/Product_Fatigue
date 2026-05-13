# Contributing

Thanks for your interest in improving Product Fatigue Detection. This document covers local setup, the dev loop, and how to send a change.

## Local setup

```bash
git clone https://github.com/<your-username>/Product_Fatigue.git
cd Product_Fatigue
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # if present
pre-commit install
```

Frontend:

```bash
cd frontend
npm install
```

## Data

Raw datasets are not in this repo. See the README's **Data Setup** section for download links, or pull from the DVC remote if you have access:

```bash
dvc pull
```

Place files in `data/raw/`, then run the EDA notebooks to generate `data/processed/`.

## Dev loop

| Task | Command |
|------|---------|
| Lint Python | `ruff check .` |
| Format Python | `black .` |
| Run tests | `PYTHONPATH=. pytest tests/ -v` |
| Run training pipeline | `python3 src/main.py` |
| Run API | `python3 -m uvicorn src.api.main:app --reload --port 8000` |
| Run frontend | `cd frontend && npm run dev` |
| Lint frontend | `cd frontend && npm run lint` |

Or use the `Makefile` shortcuts: `make lint`, `make test`, `make train`, `make api`, `make frontend`.

## Notebooks

Notebooks are stripped of outputs by `nbstripout` (set up by `pre-commit install`). Don't commit notebook output cells — they bloat diffs and the repo.

## Sending a PR

1. Branch from `main` (`git checkout -b feat/short-description`)
2. Make focused commits — one logical change per commit
3. Run `pre-commit run --all-files` and `pytest` locally
4. Push and open a PR using the template
5. CI must pass before review

### Commit message style

```
<type>: <short summary in imperative mood>

<optional body explaining *why*, not *what*>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.

## Reporting bugs

Use the bug report issue template. Include:
- What you ran (exact command)
- What you expected
- What happened (full traceback if applicable)
- Environment (OS, Python version, `pip freeze | grep -E 'xgboost|scikit-learn|fastapi'`)

## Anti-leakage rules (ML-specific)

If you touch `src/data_loader.py`, `src/train.py`, or anything in the training path, read the **Anti-Leakage Safeguards** section of the README first. Adding a feature that is a deterministic function of the label silently destroys the model — there is no automated check that will catch this.
