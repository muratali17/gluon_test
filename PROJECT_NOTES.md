# gluon_test — Project Notes

AutoGluon (AutoML) + Streamlit project. Goal: a web app where users upload a CSV,
pick a target label, and train an AutoGluon model (with leaderboard, evaluation,
and prediction).

## Structure
- `test/` — AutoGluon prototypes (all on titanic.csv via `../datasets/titanic.csv`):
  - `gluon_train.py` — basic `TabularPredictor(label=...).fit(train_data)`, evaluate + leaderboard. Saves to auto-generated `AutogluonModels/` path.
  - `gluon_hyperparameter.py` — fit with `hyperparameters` dict (GBM/CAT/XGB + single-tree RF) and `time_limit=300`.
  - `gluon_load.py` — loads a saved predictor from a hardcoded path and evaluates it.
- `app/web_app/app.py` — Streamlit CSV **viewer** (upload + preview + dtypes). **Phase 1 (DONE)**. No ML wired in yet.
- `app/ml/` — the ML engine, a Python package (`__init__.py` makes it importable as `ml`):
  - `auto_gluon_ml.py` — `AutoGluonML` class: `train(data, label, time_limit)`, `save_model(task_name)`, `load(task_name)`, `predict(data)`.
  - `utilities.py` — `validate_task_name()`, `get_model_path()`, `TRAINED_MODELS_DIR` constant.
- `requirements.txt` — pins both `streamlit` (1.62.0) and `autogluon` (v1.6.1) plus jupyter stack.
- `roadmap.md` — phased plan (6 phases) for building the full app.

## ML engine (Phase 2 — DONE)
- `AutoGluonML` design decisions:
  - **Class** (not functions) — chosen by user for future extensions / Streamlit use.
  - `__init__()` takes nothing; task_name and time_limit are method args.
  - Trains 4 models via `HYPERPARAMETERS` (GBM/CAT/XGB + single-tree RF), 80/20 split, `random_state=42`.
  - `train()` fits WITHOUT a path → AutoGluon writes to an auto-generated `app/AutogluonModels/ag-*` temp dir.
  - `save_model(task_name)` copies that temp dir to `trained_models/<task_name>/` (overwrites if exists).
    - Gotcha: `TabularPredictor.save()` ignores path args (saves to `self.path` only), so saving = `shutil.copytree`, NOT `predictor.save()`.
  - `load(task_name)` reloads from `trained_models/<task_name>/`.
  - Task names are validated in `utilities.py` (rejects empty/whitespace, `/`, `\`, `.`, `..`).

## Key facts
- Datasets live in `datasets/` (gitignored). NOTE: was missing from disk on this machine; restored titanic.csv (891×12) from `https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv`.
- `trained_models/` (project root) is gitignored; saved predictors land there.
- Python app entry: `app/web_app/app.py`, launched via `streamlit run`.
- Comments in test scripts are in Turkish.

## Vision (confirmed with user)
- Two-part app: Streamlit UI (`app/web_app/`) for user interaction + ML engine (`app/ml/`) where AutoGluon lives.
- User uploads CSV via Streamlit → engine trains **4 models** (GBM, CAT, XGB, single-tree RF) with AutoGluon.
- Save the **full `TabularPredictor`** (all 4 models + ensemble), NOT just the best one.
- Save path: `trained_models/<task_name>/`, where task name is provided by the user (e.g. `task_finding_best_pipe`).
- Future plan: UI lets the user select a previously saved model for reuse (prediction on new data).
- We build **incrementally per roadmap phases**, not all at once.

## Status / next steps
- Phase 1 (CSV upload + viewer): DONE.
- Phase 2 (ML engine in `app/ml/`): DONE (train/save/load/predict + utilities, smoke-tested on titanic).
- Phase 3 (load/predict wrappers for reuse): effectively done via `load()`/`predict()`; may formalize.
- Phase 4 (wire training into Streamlit UI): NEXT.
- See `roadmap.md` for the full plan.
