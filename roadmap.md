# gluon_test — Roadmap

Goal: AutoML web app. Two parts:
- **Streamlit UI** (`app/web_app/`) — user uploads a CSV, configures a training run, views results.
- **ML engine** (`app/ml/`) — AutoGluon trains 4 models (GBM, CAT, XGB, RF) and saves the
  full predictor under `trained_models/<task_name>` so it can be reused later.

Decisions locked in:
- Save the **full `TabularPredictor`** (contains all 4 trained models + AutoGluon ensemble),
  not just the best model.
- Model name = user-provided task name, e.g. `task_finding_best_pipe` →
  `trained_models/task_finding_best_pipe/`.
- `trained_models/` is gitignored (models can be large).
- ML engine is a **class** (`AutoGluonML`) — chosen by user for future extensions & Streamlit use.
- `__init__()` takes no args; `task_name`/`time_limit` are method args.
- Task name validation lives in `utilities.py`.

---

## Phase 1 — CSV upload & viewer (DONE)
- [x] Streamlit app that accepts a CSV upload and shows preview + dtypes.
- Entry point: `app/web_app/app.py`, run via `streamlit run`.

## Phase 2 — ML engine: training core (DONE)
Reusable training logic in `app/ml/` (pure Python, no Streamlit imports).
- [x] `app/ml/auto_gluon_ml.py` — `AutoGluonML` class:
  - `train(data, label, time_limit=300)` — 80/20 split, fits 4 models, returns predictor.
  - `save_model(task_name)` — copies trained predictor dir to `trained_models/<task_name>/`.
  - `load(task_name)`, `predict(data)`.
- [x] `app/ml/utilities.py` — `validate_task_name()`, `get_model_path()`, `TRAINED_MODELS_DIR`.
- [x] `app/ml/__init__.py` exposing the API.
- [x] `trained_models/` at project root, added to `.gitignore`.
- [x] Smoke-tested on titanic.csv (train → save → load → predict).
- [ ] Confirm regression (numeric target) works — only classification (`Survived`) tested so far.

## Phase 3 — ML engine: load & predict wrappers (DONE)
- [x] `load(task_name)` and `predict(data)` on the class.
- [x] `ml/__init__.py` exposes `AutoGluonML` + `TRAINED_MODELS_DIR`.

## Phase 4 — Wire training into the UI (DONE)
- [x] `sys.path` fix in `app.py` so `web_app` can `import ml`.
- [x] Target label picker (dropdown of columns).
- [x] Task name text input + time-limit control.
- [x] "Train" button → spinner → trains + saves model, shows leaderboard + model path.
- [x] Tested end-to-end via Streamlit AppTest (upload titanic → train → saved).

## Phase 5 — Model management
- [ ] List saved models in `trained_models/` (from Streamlit UI).
- [ ] Show per-model metadata (task name, path, saved-at, best metric).
- [ ] Allow deleting a model.

## Phase 6 — Future: reuse saved models
- [ ] Model selector: pick a saved model from the list.
- [ ] Upload a new CSV → batch predictions on it.
- [ ] Download predictions as CSV.
