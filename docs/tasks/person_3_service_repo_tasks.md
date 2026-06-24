# Person 3 — Repo, Service & Explainability Owner — Stage 1 Task Plan

> Project: TEKNOFEST 2026 E-Commerce **Search Relevance Engine** (Stage 1, Kaggle binary relevance: `related / not_related`, metric = **Macro-F1**).
> Follow the module names from `03_STAGE1_ARCHITECTURE.md` and the prep workstreams in `04_PRE_KAGGLE_PREPARATION_PLAN.md` exactly. Do not invent a new architecture. Do not assume final Kaggle column names. **No transformer/LLM/API anywhere in the inference or explanation path.** Everything must run **offline** (`docker run --network none`) and be **reproducible** for the top-20 inspection.

---

## Role

You own the **repository, the reproducibility spine, and the serving/explainability skeleton** — everything that turns Person 2's trained artifact bundle into something that can be loaded, served, explained, benchmarked, and shipped offline. You are the integrator: you don't train models or mine negatives, but every other person's work lands in the structure you build and is judged through the gates you wire.

Concretely you own five subsystems:

1. **Repository foundation** — folder structure, packaging (`pyproject.toml`/`requirements.txt`), `Makefile`, `.gitignore`, the empty `__init__.py` tree, CI/pre-commit (workstreams A, C, Q).
2. **The frozen-interface contract doc** — `docs/interfaces.md`, the single shared file all three people read so they can build in parallel.
3. **Artifact loading + serving skeleton** — `ModelBundle` loader, FastAPI `/predict` and `/explain` skeletons (workstream P).
4. **Real-feature explainability** — local attribution from model coefficients/importances + lexical-match highlight, **never an LLM narrator** (failure F17).
5. **Offline + compliance + reproducibility gates** — Docker, no-network test, latency benchmark, dummy end-to-end dry run, Kaggle-roster check, rerun check, README + report skeleton + demo (workstream S + compliance).

You are the **single source of truth** for three things the other two people depend on:

- `docs/interfaces.md` — the frozen cross-person contracts (you assemble it; P1/P2 provide their halves).
- The **config ownership map** and the scaffolded empty config/`__init__.py` tree (so nobody co-edits the same file Day 0).
- `configs/champion.yaml` (canonical Stage 1 decisions) and `configs/service.yaml`, `configs/data.yaml`.

**Scope discipline (from the audit, Areas 10–11):** FastAPI, Docker, and explainability are **skeletons + stubs now** — interfaces and a working dummy path, not a polished Stage-2 service. Full service/UI/SHAP work is deferred to post-finalist Stage-2 prep so it does **not** eat Kaggle-pipeline time. **But two things are real gates starting now, not later:** the **Kaggle-roster compliance check** (failure F1, the cheapest disqualification to avoid) and **offline `--network none` discipline** (failure F16). Build those for real.

---

## Mission (what must exist before the 26/06 data release)

A **clean, installable repository whose dummy pipeline runs end-to-end and whose serving/offline/compliance skeletons are wired**, so that on data day the team only has to swap in the real schema:

- `pip install -e .` works; `pytest` runs; the full `configs/` and `src/**/__init__.py` tree exists so P1/P2 fill module bodies without collisions.
- `docs/interfaces.md` freezes every cross-person signature (field registry, normalization, composer, folds, BM25, FeaturePipeline, artifact bundle, label map).
- `ModelBundle.load(run_dir)` reads Person 2's artifact bundle and exposes `.predict(record)` end-to-end on a dummy bundle.
- FastAPI `/predict` and `/explain` skeletons return correct shapes from a warmed singleton (dummy model), using **Person 1's `compose_views` + Person 2's `FeaturePipeline.transform_one`** — the same code path as training (no train/serve skew).
- `/explain` produces a **real-feature** explanation (top contributing features + lexical-match highlight + attribute match/mismatch), text **and** a simple visual, with **zero external calls**.
- A Docker skeleton builds and the dummy pipeline runs under `docker run --network none`; `scripts/no_network_check.sh` exists and passes.
- `scripts/run_dummy_pipeline.py` runs the **whole** integration on dummy data (schema adapter → folds → easy+hard-stub negatives → TF-IDF/BM25 → LogReg → threshold → submission → manifest) and writes `reports/dry_run/dummy_pipeline_report.md`.
- `reports/compliance/kaggle_roster_check.md` template exists; the latency benchmark and rerun-check scripts exist.
- README explains scope + offline run; the report skeleton maps to the official scoring rubric.

You do **not** train real models, you do **not** tune anything, and you do **not** assume the label encoding (`var/yok` vs `0/1` vs `related/not_related`) — `champion.yaml::label_map` is filled on data day with Person 2.

---

## Files to create or edit

```text
# Repository foundation + packaging
pyproject.toml
requirements.txt
Makefile
.gitignore
.dockerignore
.pre-commit-config.yaml
README.md
data/README.md

# Empty package tree you scaffold Day 0 (P1/P2 fill bodies)
src/__init__.py
src/data/__init__.py  src/text/__init__.py  src/sampling/__init__.py
src/validation/__init__.py  src/retrieval/__init__.py  src/features/__init__.py
src/models/__init__.py  src/submission/__init__.py  src/utils/__init__.py
src/experiments/__init__.py
src/serving/__init__.py
src/service/__init__.py
src/service/routers/__init__.py
src/service/explain/__init__.py

# Configs you own + scaffold (empty skeletons for P1/P2 to fill)
configs/data.yaml            # you own
configs/service.yaml         # you own
configs/champion.yaml        # you own (+ label_map on data day)
configs/schema.yaml          configs/preprocessing.yaml   configs/sampling.yaml      # scaffold empty -> P1 fills
configs/validation.yaml      configs/thresholds.yaml      configs/features.yaml      # scaffold empty -> P2 fills
configs/model_linear.yaml    configs/model_lgbm.yaml      configs/model_catboost.yaml  configs/experiment.yaml  # scaffold empty -> P2 fills

# Artifact loading + serving skeleton
src/serving/artifact_loader.py            # ModelBundle (reads P2 bundle)
src/service/app.py
src/service/schemas.py
src/service/routers/predict.py
src/service/routers/explain.py
src/service/explain/attribution.py        # real-feature attribution, NO LLM

# Docker + offline
docker/Dockerfile
docker/entrypoint.sh

# Scripts
scripts/run_dummy_pipeline.py
scripts/benchmark_latency.py
scripts/no_network_check.sh
scripts/demo.py

# Tests
tests/serving/test_artifact_loader.py
tests/service/test_predict_contract.py
tests/service/test_explain_contract.py
tests/service/test_no_llm_imports.py

# Docs + reports
docs/interfaces.md                        # YOU OWN — the shared contract
docs/repo_structure.md
docs/offline_checklist.md
docs/report_skeleton.md
reports/compliance/kaggle_roster_check.md
reports/repro/rerun_check.md
reports/offline/no_network_test.md
reports/service/latency_benchmark.md
reports/demo/explain_spec.md
reports/dry_run/dummy_pipeline_report.md
```

> Package note: the import root is `src/` installed editable (`pip install -e .`), so imports read `from src.text.turkish_normalize import normalize_text`. You scaffold the empty `__init__.py` tree on Day 0; P1/P2 fill the module bodies. The dense/transformer encoder is **deferred to Stage 2** — do not add it to `requirements.txt` now.

---

## Config ownership map (you publish this Day 0 to prevent co-editing)

| Config | Owner | Filled when |
|---|---|---|
| `data.yaml`, `service.yaml`, `champion.yaml` | **P3 (you)** | now (`champion.yaml::label_map` on data day) |
| `schema.yaml`, `preprocessing.yaml`, `sampling.yaml` | P1 | now / data day |
| `validation.yaml`, `thresholds.yaml`, `features.yaml`, `model_linear.yaml`, `model_lgbm.yaml`, `model_catboost.yaml`, `experiment.yaml` | P2 | now |

You **scaffold all of them as empty/keyed stubs Day 0** so the files exist and imports/loaders don't crash; P1/P2 fill their own. Nobody edits a config they don't own.

---

## Shared frozen interfaces you must publish (`docs/interfaces.md`)

You assemble `docs/interfaces.md` from all three people. Your own half (the artifact bundle + loader + label map) is below; P1's half (field registry, normalization, composer, sampler) and P2's half (folds, BM25, FeaturePipeline) are provided by them.

```python
# src/serving/artifact_loader.py  (you own)
class ModelBundle:
    @classmethod
    def load(cls, run_dir: str) -> "ModelBundle": ...    # reads P2's artifacts/runs/<run_id>/
    def predict_proba(self, record: dict) -> float: ...  # compose_views -> transform_one -> model
    def predict(self, record: dict) -> dict: ...         # {"label": "related"|"not_related", "score": float, "threshold": float}
    @property
    def threshold(self) -> float: ...
    @property
    def feature_pipeline(self): ...                      # P2 FeaturePipeline (loaded)
    @property
    def model(self): ...
    @property
    def manifest(self) -> dict: ...
```

```text
# Artifact bundle layout (P2 WRITES, you READ) — frozen with Person 2
artifacts/runs/<run_id>/
├── manifest.json            # run_id, hashes, model, threshold, metrics, artifact paths
├── config_snapshot.yaml
├── model.*                  # joblib / native model file
├── vectorizers/             # fitted TF-IDF
├── bm25/                    # fitted BM25 index
├── thresholds.json          # tuned threshold (+ prior sweep, calibration)
├── feature_spec.json        # ordered feature names + group toggles + vectorizer/bm25 paths
├── cv_metrics.json
├── track_metrics.json
├── oof_predictions.parquet
├── error_analysis.md
└── submission.csv           # if submitted
```

```yaml
# configs/champion.yaml  (you own) — canonical Stage 1 decisions (from 04 §C4) + label_map
stage: { current: stage1_binary, train_multiclass_now: false }
negative_sampling:
  default_recipe: provisional_v1      # NOT "champion" until ablations on real data
  neg_per_pos: 2.0
  composition: { easy: 0.25, medium: 0.35, hard_lexical: 0.40, hard_semantic: 0.00 }
validation: { split: group_kfold, group_key: query_norm, five_tracks: true, transfer_stress_required: true, threshold_tuning: oof_only }
model_order: [bm25_threshold, tfidf_threshold, logistic_regression, lightgbm, catboost, dense_features_later]
threshold_tuning: oof_only
label_map:                            # FILLED ON DATA DAY with Person 2
  internal: { related: 1, not_related: 0 }
  official: null                      # e.g. {1: "var", 0: "yok"} or {1: 1, 0: 0} — set after seeing sample_submission
  id_column: null
  label_column: null
```

`ModelBundle.predict` reconstructs the FeaturePipeline via `FeaturePipeline.load(...)` and reuses Person 1's `compose_views` — so the service runs the **same feature code path as training**. The bundle layout is the contract; if it changes, you and Person 2 update `docs/interfaces.md` together.

---

## Task list

### P3-01 — Repository scaffold, `.gitignore`, package tree, `data/README.md`
- **Purpose:** Create the clean repo structure (workstream A) and the empty package tree so three people can commit in parallel without collisions or import errors.
- **Files:** the full folder tree (`configs/ data/ docs/ notebooks/ src/ tests/ reports/ artifacts/ docker/ scripts/ reference/stage1/ presentations/`), all `src/**/__init__.py`, `.gitignore`, `data/README.md`.
- **Technical details:** Create the directory layout from `04_PRE_KAGGLE_PREPARATION_PLAN.md §A2`. Scaffold **every** `__init__.py` listed in *Files to create or edit* (empty is fine) so `from src.x.y import z` resolves the moment P1/P2 add code. `.gitignore` excludes `data/raw/`, `data/interim/`, `data/processed/`, `artifacts/`, `*.parquet`, `__pycache__/`, `.venv/`, model binaries, and any `*.csv` under `data/` (never commit the real dataset). `data/README.md` documents the `raw → interim → processed → submissions` layers and the "no real data in git" rule.
- **Expected functions/classes/config keys:** n/a (structure).
- **Input/output contract:** the repo is the artifact; `git status` is clean and no data is tracked.
- **Acceptance criteria:** `tree -L 2` matches the agreed structure; `git status` shows no dataset files; every `src` subpackage imports without error.
- **Tests:** a trivial `tests/serving/test_artifact_loader.py` import check confirms the package tree resolves (expanded later).
- **Common mistakes:** missing an `__init__.py` (breaks editable imports); committing `data/raw/`; deep-nesting that diverges from the architecture doc.
- **Dependencies:** none. **Do this first on Day 0 — it unblocks P1 and P2.**
- **Definition of Done:** structure committed; `.gitignore` verified against a dummy data file; announced in team channel.

### P3-02 — `pyproject.toml` + `requirements.txt` (pinned, CPU-only, offline-installable)
- **Purpose:** Make `src/` an editable package and pin a CPU-only, offline-vendorable dependency set with **no internet-calling libraries** in the runtime path.
- **Files:** `pyproject.toml`, `requirements.txt`
- **Technical details:** `pyproject.toml` declares `src` as the package (setuptools, editable install: `pip install -e .`), Python `>=3.10`, ruff config, and pytest config. `requirements.txt` is **fully pinned** and limited to: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `catboost`, `rapidfuzz`, `pyyaml`, `pydantic`, `fastapi`, `uvicorn`, `joblib`, `pytest`, `ruff`. **No** `openai`/`google-generativeai`/`anthropic`/`requests`-as-predictor/`sentence-transformers`/`torch` (dense encoder is a **Stage-2** decision and would jeopardize offline packaging now). Note in a comment that all wheels must be installable from a local cache for the air-gapped finals.
- **Expected functions/classes/config keys:** `[project] dependencies`, `[tool.ruff]`, `[tool.pytest.ini_options]`, `[build-system]`.
- **Input/output contract:** `pip install -e .` succeeds offline from a vendored wheel cache; `import src` works.
- **Acceptance criteria:** clean-env `pip install -e .` works; `python -c "import src, fastapi, lightgbm, catboost, rapidfuzz"` works; no banned package present.
- **Tests:** `tests/service/test_no_llm_imports.py` greps the dependency set + `src/service/` for banned imports (see P3-08).
- **Common mistakes:** unpinned versions (non-reproducible); adding `torch`/`sentence-transformers` now (breaks offline goal, scope creep); a network-calling client in runtime deps.
- **Dependencies:** P3-01.
- **Definition of Done:** editable install works; deps pinned; banned-import test green.

### P3-03 — `Makefile`
- **Purpose:** One-command entry points so the whole team (and the inspector) runs the same commands.
- **Files:** `Makefile`
- **Technical details:** Targets: `install` (`pip install -e .`), `test` (`pytest -q`), `lint` (`ruff check`), `format` (`ruff format`), `dummy` (`python scripts/run_dummy_pipeline.py`), `serve` (`uvicorn src.service.app:app`), `benchmark` (`python scripts/benchmark_latency.py`), `offline-check` (`bash scripts/no_network_check.sh`), `docker-build`, `docker-run-offline` (`docker run --network none ...`), `roster-check`, `repro-check`. Each target is a thin wrapper; no hidden logic in the Makefile.
- **Expected functions/classes/config keys:** Make targets above.
- **Input/output contract:** `make <target>` runs the documented command from repo root.
- **Acceptance criteria:** `make install`, `make test`, `make dummy` all run; targets documented in README.
- **Tests:** n/a (smoke-run targets manually).
- **Common mistakes:** putting real logic in the Makefile instead of scripts; targets that assume a non-root working dir.
- **Dependencies:** P3-01, P3-02; `dummy` depends on P3-12.
- **Definition of Done:** core targets run; listed in README + `docs/repo_structure.md`.

### P3-04 — Config system skeletons + `champion.yaml` + `data.yaml` + `service.yaml`
- **Purpose:** Make every important decision live in `configs/` (no hidden constants), and publish the ownership map so nobody co-edits a config.
- **Files:** all `configs/*.yaml` (you author `champion.yaml`, `data.yaml`, `service.yaml`; scaffold the rest as empty/keyed stubs for P1/P2).
- **Technical details:** `champion.yaml` exactly as in *Shared frozen interfaces* (canonical decisions from `04 §C4` + `label_map` stub). `data.yaml` holds paths (`raw_dir`, `interim_dir`, `processed_dir`, `submissions_dir`, `artifacts_dir`) and file-name placeholders to be filled on data day. `service.yaml` holds serving config: `champion_run_dir`, `host`, `port`, `warmup: true`, `max_batch`, `explain.top_k_features: 8`, `explain.enable_visual: true`. Provide a tiny `load_yaml(path)` helper in `src/utils/io.py` ownership note (P2 owns `io.py`; agree they expose it, or you add a 3-line local loader in serving). Publish the **config ownership table** in `docs/repo_structure.md`.
- **Expected functions/classes/config keys:** keys above; `champion.yaml::{stage,negative_sampling,validation,model_order,threshold_tuning,label_map}`; `service.yaml::{champion_run_dir,host,port,warmup,explain.*}`; `data.yaml::{*_dir}`.
- **Input/output contract:** configs load as plain dicts; missing optional keys have safe defaults.
- **Acceptance criteria:** all 13 config files exist; `champion.yaml` validates as YAML and contains the model order + recipe + label_map stub; ownership map published.
- **Common mistakes:** putting a real default only in code (forbidden); marking the recipe "champion" (it is `provisional_v1`); editing P1/P2 configs.
- **Tests:** a YAML-loads-cleanly check folded into `tests/serving/test_artifact_loader.py`.
- **Dependencies:** P3-01.
- **Definition of Done:** configs present and load; ownership map shared; `label_map` clearly marked "filled on data day".

### P3-05 — `docs/interfaces.md` — the shared frozen-interface contract
- **Purpose:** One document where all cross-person signatures are frozen, so three people build against stable contracts in parallel. This is the most important coordination artifact in the repo.
- **Files:** `docs/interfaces.md`
- **Technical details:** Assemble and freeze: P1's `CANONICAL_FIELDS`, `normalize_text`/`turkish_lower_safe`, `compose_views`/`build_text_views`/`VIEW_COLUMNS`, `NegativeSampler.generate`/`PROVENANCE_COLS`; P2's `assign_folds`/`iter_folds`, `BM25Index.fit/score/topk`, `FeaturePipeline.fit_fold/transform/transform_one/save/load` + `feature_spec.json` schema; your `ModelBundle` + the artifact-bundle layout + `champion.yaml::label_map`. For each: signature, one-line contract, owner, and "frozen as of Day 0 — changes require a channel announcement". Include the data-layer contract (`raw → interim → processed`).
- **Expected functions/classes/config keys:** documentation of all of the above.
- **Input/output contract:** the doc is the contract; any signature change is reflected here first.
- **Acceptance criteria:** every interface another person imports appears with its exact signature and owner; P1 and P2 sign off that their halves are correct.
- **Common mistakes:** letting the doc drift from code; freezing interfaces verbally instead of in this file.
- **Tests:** n/a.
- **Dependencies:** P1 and P2 provide their halves on Day 0; you assemble.
- **Definition of Done:** committed Day 0; both teammates confirm their sections; referenced by every other task.

### P3-06 — Artifact rules + `ModelBundle` loader  *(FROZEN INTERFACE)*
- **Purpose:** Load Person 2's artifact bundle and expose a clean inference surface (`predict_proba`, `predict`) that the service and demo call — the only bridge between training artifacts and serving.
- **Files:** `src/serving/artifact_loader.py`
- **Technical details:** `ModelBundle.load(run_dir)` reads `manifest.json`, loads the model (`joblib`/native), reconstructs the feature pipeline via `FeaturePipeline.load(run_dir)` (reads `feature_spec.json` + `vectorizers/` + `bm25/`), and loads the threshold from `thresholds.json`. `predict_proba(record)` = `feature_pipeline.transform_one(record)` (which internally calls P1's `compose_views`) → `model.predict_proba` → positive-class probability. `predict(record)` applies `self.threshold` and maps the internal int label to the official string via `champion.yaml::label_map`, returning `{"label","score","threshold"}`. Validate the bundle on load (all required files present) and raise a precise error if any artifact is missing (so a non-reproducible bundle fails loudly — failure F2).
- **Expected functions/classes/config keys:** `class ModelBundle(.load,.predict_proba,.predict,.threshold,.feature_pipeline,.model,.manifest)`, `validate_bundle(run_dir)->list[str]`.
- **Input/output contract:** in = `run_dir` path; out = a ready bundle. `predict(record)` in = canonical-keyed dict; out = label dict. Never silently returns on a missing artifact.
- **Acceptance criteria:** loads a **dummy bundle** end-to-end and returns a well-formed prediction; `validate_bundle` lists exactly what's missing on an incomplete bundle.
- **Tests:** `tests/serving/test_artifact_loader.py` (loads dummy bundle; predict shape; missing-artifact error path; label_map applied).
- **Common mistakes:** rebuilding features differently from training (skew — must go through `FeaturePipeline`/`compose_views`); hardcoding the threshold or label encoding; swallowing a missing-file error.
- **Dependencies:** **Person 2's `FeaturePipeline.load` + bundle layout; Person 1's `compose_views`.** Use a hand-made dummy bundle until P2's is ready.
- **Definition of Done:** dummy bundle loads and predicts; tests green; signature in `docs/interfaces.md`.

### P3-07 — FastAPI `/predict` skeleton (warmed singleton)
- **Purpose:** Serve single/batch predictions from a **warmed** model loaded once at startup — no per-request rebuild, no cold start (failure F18). Skeleton now; full service is Stage-2.
- **Files:** `src/service/app.py`, `src/service/schemas.py`, `src/service/routers/predict.py`
- **Technical details:** On FastAPI startup, load `ModelBundle.load(service.yaml::champion_run_dir)` **once** into a module-level singleton (lifespan event). `schemas.py` (pydantic): `PredictRequest{query, title, category_path?, attributes?}` and `PredictResponse{label, score, threshold, latency_ms}`; map request fields to canonical keys before calling the bundle. `/predict` handles one pair; `/predict_batch` loops the warmed pipeline (no re-fit). Add `/health` returning model run_id. Until a real bundle exists, the singleton loads the dummy bundle.
- **Expected functions/classes/config keys:** `app`, `lifespan` loader, `PredictRequest`, `PredictResponse`, `POST /predict`, `POST /predict_batch`, `GET /health`. Reads `service.yaml`.
- **Input/output contract:** in = JSON pair(s); out = label + score + threshold + measured latency. Model is loaded exactly once.
- **Acceptance criteria:** `uvicorn src.service.app:app` boots, loads the singleton once, and `/predict` returns a valid response on the dummy bundle; no vectorizer is rebuilt per request.
- **Tests:** `tests/service/test_predict_contract.py` (response schema; singleton loaded once; canonical-key mapping).
- **Common mistakes:** loading the model inside the request handler (cold start / per-request rebuild); diverging field mapping from `CANONICAL_FIELDS`; using an HTML `<form>` pattern (this is JSON API).
- **Dependencies:** P3-06 (`ModelBundle`); P1/P2 interfaces (via the bundle).
- **Definition of Done:** service boots warm; `/predict` contract test green; latency field populated.

### P3-08 — FastAPI `/explain` skeleton + real-feature attribution (NO LLM)
- **Purpose:** Return the prediction **plus a real reason** (text + visual) derived from the actual model and features — explicitly **not** an LLM narrator and with **zero external calls** (failure F17, scored 10%).
- **Files:** `src/service/routers/explain.py`, `src/service/explain/attribution.py`
- **Technical details:** `attribution.py` builds the explanation from real signal only: (1) **per-feature contribution** — for linear models, `coef_ × feature_value` per feature; for tree models (LGBM/CatBoost), per-prediction feature importance (native contribution/`predict(pred_contrib=True)` or split-based importance) — top-k by `service.yaml::explain.top_k_features`; (2) **lexical-match highlight** — which query tokens matched title/category/attributes (reuse Person 1's tokenization + overlap features) and which did **not**; (3) **attribute match/mismatch flags** — e.g. `color: query=mavi vs product=siyah → mismatch`, surfacing contradiction features. Output a structured `ExplanationResponse` (text summary + a list of contributing features with sign/weight + matched/unmatched token lists + attribute flags) and a **simple visual** spec (a feature-contribution bar list renderable as inline HTML/SVG — no external chart service). Write `reports/demo/explain_spec.md` describing exactly what the jury sees and asserting no LLM/API is in the path.
- **Expected functions/classes/config keys:** `explain(record, bundle)->ExplanationResponse`, `feature_contributions(model, x, names)->list`, `lexical_highlight(record)->dict`, `attribute_flags(record)->list`; `POST /explain`; pydantic `ExplanationResponse`.
- **Input/output contract:** in = pair JSON; out = `{label, score, top_features:[{name,value,contribution,direction}], matched_tokens, unmatched_tokens, attribute_flags, visual}` — fully offline.
- **Acceptance criteria:** `/explain` returns the prediction + ≥ top-k real feature contributions + matched/unmatched tokens + attribute flags on the dummy bundle; a dependency/AST scan of the explain path finds **no** LLM/HTTP client import.
- **Tests:** `tests/service/test_explain_contract.py` (response schema; contributions present and signed; tokens highlighted) and `tests/service/test_no_llm_imports.py` (no banned import anywhere under `src/service/`).
- **Common mistakes:** calling an LLM to "write the explanation" (DQ-adjacent: violates paid-API + offline); text-only with no visual; explanations not grounded in the actual features/coefficients.
- **Dependencies:** P3-06 (`ModelBundle.model` + `feature_pipeline`); Person 2's `feature_spec.json` (names) + coef/importance access; Person 1's tokenization/overlap for highlighting.
- **Definition of Done:** `/explain` works on the dummy bundle; no-LLM scan green; `reports/demo/explain_spec.md` written.

### P3-09 — Docker skeleton + entrypoint + `.dockerignore`
- **Purpose:** A CPU-only container that runs the dummy pipeline and the service **offline** (`docker run --network none`) — the environment for the finals offline gate (failure F16). Skeleton now.
- **Files:** `docker/Dockerfile`, `docker/entrypoint.sh`, `.dockerignore`
- **Technical details:** Base on a slim CPU Python image; copy the repo, `pip install -e .` from a **vendored wheel cache** (no network at build-runtime), copy `artifacts/` (vendored weights/bundle) and `configs/`. `entrypoint.sh` supports two modes: `serve` (`uvicorn src.service.app:app`) and `dummy` (`python scripts/run_dummy_pipeline.py`). `.dockerignore` excludes `data/raw`, `.venv`, `__pycache__`, `.git`. **No runtime downloads, no model-hub pulls, no license checks** — everything vendored.
- **Expected functions/classes/config keys:** Dockerfile stages; `entrypoint.sh` mode switch.
- **Input/output contract:** `docker build` produces an image; `docker run --network none <img> dummy` runs the dummy pipeline with networking disabled.
- **Acceptance criteria:** image builds; `docker run --network none` runs the dummy pipeline to completion (proves no runtime network dependency).
- **Tests:** validated via `scripts/no_network_check.sh` (P3-10) and recorded in `reports/offline/no_network_test.md`.
- **Common mistakes:** `pip install` reaching the internet at runtime; not vendoring weights; importing a library that phones home on first use.
- **Dependencies:** P3-02 (pinned deps), P3-12 (dummy pipeline). Defer image-size polish to Stage 2.
- **Definition of Done:** offline `docker run --network none ... dummy` succeeds; result logged.

### P3-10 — Offline checklist + `no_network_check.sh`
- **Purpose:** Turn "runs offline" from a hope into a repeatable, signed-off **finals gate** (failure F16).
- **Files:** `scripts/no_network_check.sh`, `docs/offline_checklist.md`, `reports/offline/no_network_test.md`
- **Technical details:** `no_network_check.sh` runs the dummy pipeline (and optionally the service smoke) with networking disabled — preferred path `docker run --network none`; a fallback for local runs blocks egress (e.g. unset proxies / firewall note) and asserts the pipeline still completes. It greps the codebase for banned network/LLM clients in the runtime path and fails on any hit. `docs/offline_checklist.md` enumerates: vendored wheels, vendored model/bundle, no runtime downloads, no API keys read at inference, no license check, explain path offline. `reports/offline/no_network_test.md` records the date, command, and pass/fail.
- **Expected functions/classes/config keys:** the shell script + checklist items.
- **Input/output contract:** exit 0 = offline-clean; non-zero with a clear reason on any network/LLM dependency.
- **Acceptance criteria:** script passes on the dummy pipeline under `--network none`; flips to fail if a `requests`/LLM import is added to the runtime path (tested by temporarily injecting one).
- **Common mistakes:** treating the no-network test as nice-to-have; only checking the service and not the pipeline; missing a transitively network-calling dependency.
- **Tests:** `tests/service/test_no_llm_imports.py` complements the script (static side).
- **Dependencies:** P3-09, P3-12.
- **Definition of Done:** offline gate passes and is reproducible; checklist + report committed; this is a **release gate**, not optional.

### P3-11 — Latency benchmark (p50/p95, warmed)
- **Purpose:** Measure serving speed across payload sizes with a **warmed** model (Model Hızı is 10% of scoring; the Final Set runs under time constraints — failure F18).
- **Files:** `scripts/benchmark_latency.py`, `reports/service/latency_benchmark.md`
- **Technical details:** Warm the `ModelBundle` once, then time `predict`/`predict_batch` over batch sizes (e.g. 1, 8, 32, 128) for N repeats; report **p50/p95/p99** ms per request and throughput, separating cold (first call) from warm. Write a table to `reports/service/latency_benchmark.md`. Use the dummy bundle until a real one exists.
- **Expected functions/classes/config keys:** `benchmark(bundle, sizes, repeats)->dict`, percentile helpers.
- **Input/output contract:** in = bundle + sizes; out = latency table on disk.
- **Acceptance criteria:** produces p50/p95 across ≥ 3 payload sizes on the dummy bundle; warm path excludes load time.
- **Tests:** smoke run (no hard assertion beyond "table produced").
- **Common mistakes:** benchmarking with model load inside the loop (measures cold start); single payload size; reporting only the mean (need tail percentiles).
- **Dependencies:** P3-06, P3-07.
- **Definition of Done:** benchmark runs; p50/p95 table written; method documented.

### P3-12 — Dummy end-to-end dry run + report  *(integration — the pre-data proof)*
- **Purpose:** Prove the **whole pipeline wires together** on dummy data before real data exists, so day-1 isn't lost to integration surprises (failures F2, F10; workstream S).
- **Files:** `scripts/run_dummy_pipeline.py`, `reports/dry_run/dummy_pipeline_report.md`
- **Technical details:** Orchestrate the **minimal real path** end-to-end on Person 1's dummy fixtures: `SchemaAdapter.adapt` → `build_text_views` → `assign_folds` → `NegativeSampler.generate` with **easy + medium + hard-lexical stub** + FN-filter → `FeaturePipeline.fit_fold/transform` (TF-IDF + BM25) → `fit_linear` (LogReg) via `run_cv` → `tune_threshold` → `build_submission` + `validate_submission` → write a manifest bundle → `ModelBundle.load` the bundle and run one `/predict`. Catch and report any interface mismatch. Write `reports/dry_run/dummy_pipeline_report.md` listing each stage, its status, row counts, and the produced artifact paths.
- **Expected functions/classes/config keys:** `run_dummy_pipeline()` calling each owner's public interface; no private internals.
- **Input/output contract:** in = dummy fixtures + configs; out = a complete dummy bundle + a dry-run report + a successful dummy prediction.
- **Acceptance criteria:** the script runs start-to-finish on dummy data, produces a bundle that `ModelBundle` can load and predict from, and writes the report — **with zero manual steps**.
- **Tests:** the dry run itself is the integration test; `make dummy` runs it.
- **Common mistakes:** reaching into another module's internals instead of the frozen interface; letting the dry run drift from the real day-1 path; skipping the FN-filter in the dummy run.
- **Dependencies:** **P1 (adapter, composer, sampler, fixtures) + P2 (folds, BM25, FeaturePipeline, LogReg, run_cv, submission) stubs.** This is the integration point — run it *with* P1/P2 at end of Day 1.
- **Definition of Done:** dummy dry run is green and reproducible; report committed; `make dummy` works; this is the headline pre-data deliverable.

### P3-13 — Kaggle roster compliance check (F1) + reproducibility rerun check (F2)
- **Purpose:** Eliminate the two cheapest disqualifications: **Kaggle team ≠ KYS roster** (instant DQ) and a **non-reproducible champion** (top-20 notebooks must reproduce). These cost nothing to fail, so build the checks now.
- **Files:** `reports/compliance/kaggle_roster_check.md`, `reports/repro/rerun_check.md`
- **Technical details:** `kaggle_roster_check.md` is a checklist the captain signs **before the first submission**: list each KYS-registered member, each Kaggle-team member, diff them, confirm no member is double-rostered and (if applicable) the high-school advisor is present. `rerun_check.md` documents the clean-environment rerun procedure: from a fresh checkout + vendored wheels, `run_cv` on the champion config must reproduce the submitted Macro-F1 within tolerance (uses the manifest `data_hash`/`config_hash`/`seed`). Add a tiny helper to assert `manifest.config_hash` matches the active config snapshot before a submission is treated as "the champion".
- **Expected functions/classes/config keys:** checklist docs + an optional `assert_manifest_matches_config(run_dir)` helper.
- **Input/output contract:** roster check = a signed-off doc gating the first upload; rerun check = a documented, runnable procedure producing a pass/fail.
- **Acceptance criteria:** roster doc exists and is part of the "before first submission" gate; rerun procedure is written and runnable on the dummy bundle (reproduces dummy OOF Macro-F1).
- **Common mistakes:** treating roster as an afterthought (it's an instant DQ); claiming reproducibility without a clean-env rerun; submitting before the roster is signed.
- **Tests:** `assert_manifest_matches_config` unit check folded into `tests/serving/test_artifact_loader.py`.
- **Dependencies:** Person 2's manifest/hashing for the rerun check.
- **Definition of Done:** roster checklist committed and wired as a pre-submission gate; rerun procedure validated on the dummy bundle. **Do the roster check first thing on data day, before any submission.**

### P3-14 — README + report skeleton + demo script
- **Purpose:** Project front door, the official report scaffold mapped to the scoring rubric, and a rehearsable jury demo (Report 10% + Presentation 10% + Explainability live demo).
- **Files:** `README.md`, `docs/report_skeleton.md`, `scripts/demo.py`
- **Technical details:** `README.md`: project scope (Search Relevance Engine, Stage 1 binary Macro-F1), setup (`make install`), how to run the dummy pipeline, how to serve, how to run offline, and the team roles + each member's contribution (required for the jury). `docs/report_skeleton.md`: section headings mapped to the official scoring weights — **Kaggle private LB (40%)**, **Final Set on hidden data (20%)**, **Presentation (10%)**, **Model Speed (10%)**, **Explainability (10%)**, **Final Report (10%)** — with placeholders for negative-sampling method, validation protocol, threshold strategy, error analysis, offline packaging, and per-member roles. `scripts/demo.py`: a CLI that takes a query+product, calls `ModelBundle.predict` + the `/explain` attribution, and prints the label + the real-feature explanation + matched tokens — the live-demo rehearsal (offline).
- **Expected functions/classes/config keys:** `demo(query, product)->prints prediction + explanation`.
- **Input/output contract:** README/report are docs; `demo.py` runs offline against the dummy (later champion) bundle and prints a grounded explanation.
- **Acceptance criteria:** README lets a new member install + run the dummy pipeline using only the doc; report skeleton headings map 1:1 to the scoring rubric; `demo.py` prints a prediction + real explanation offline.
- **Common mistakes:** leaving the report to the end; missing required headings / per-member roles; a demo that needs the internet.
- **Tests:** `demo.py` smoke run.
- **Dependencies:** P3-06/08.
- **Definition of Done:** README complete; report skeleton mapped to the rubric; demo runs offline.

### P3-15 — CI / pre-commit + serving tests
- **Purpose:** Keep the repo green and the offline/no-LLM invariants enforced automatically as three people commit fast.
- **Files:** `.pre-commit-config.yaml`, all `tests/serving/*`, `tests/service/*`
- **Technical details:** `.pre-commit-config.yaml` runs `ruff` (lint + format) and `pytest -q` (or a fast subset) on commit. Finalize the serving/service tests: artifact loader, predict contract, explain contract, and the **no-LLM-imports** guard. Document a simple CI command (`make test`) the team runs before every PR. Keep heavy tests (CatBoost/LGBM) runnable locally even if trimmed in pre-commit.
- **Expected functions/classes/config keys:** pre-commit hooks; the four serving/service test files.
- **Input/output contract:** failing lint/tests/no-LLM-guard blocks the commit/PR.
- **Acceptance criteria:** `pre-commit run --all-files` passes on a clean tree; the no-LLM guard fails if a banned import is introduced anywhere under `src/service/`.
- **Common mistakes:** CI that's so slow people skip it; not enforcing the no-LLM/offline invariant in automation; flaky service tests from a non-warmed model.
- **Tests:** the suite itself.
- **Dependencies:** P3-06/07/08.
- **Definition of Done:** pre-commit green; serving/service tests green; no-LLM guard active.

---

## Daily execution order

**Day 0 (24/06) — scaffold + contracts + the cheapest DQ gate**
1. P3-01 repo scaffold + `.gitignore` + full `__init__.py` tree → **publish immediately (unblocks P1/P2).**
2. P3-02 `pyproject.toml` + pinned `requirements.txt`; P3-03 `Makefile`.
3. P3-04 config skeletons + `champion.yaml` + **config ownership map** → publish (so nobody co-edits configs).
4. P3-05 assemble + freeze `docs/interfaces.md` with P1/P2's halves → the coordination spine.
5. P3-13 stand up `reports/compliance/kaggle_roster_check.md` (F1) — cheapest DQ, do it now.

**Day 1 (25/06) — loader, service skeletons, offline, integration**
6. P3-06 `ModelBundle` loader against a hand-made dummy bundle.
7. P3-07 `/predict` warmed singleton; P3-08 `/explain` + real-feature attribution (no-LLM).
8. P3-09 Docker skeleton + entrypoint; P3-10 offline checklist + `no_network_check.sh`.
9. P3-11 latency benchmark; P3-15 CI/pre-commit + serving tests.
10. **P3-12 dummy end-to-end dry run — run it *with* P1/P2** at end of day; fix interface mismatches together.

**Day 2 morning (26/06, pre-data)**
11. P3-14 README + report skeleton + demo; freeze interfaces; re-run the dummy dry run + offline check; stand ready (see First-24h).

> If the team is behind, the **minimum viable path** is: P3-01 → P3-02 → P3-04 → P3-05 (interfaces) → P3-13 (roster) → P3-12 (dummy dry run) → P3-06 (loader). FastAPI `/explain` polish, Docker, and latency can trail — but **never** drop the roster check (F1), the offline discipline (F16), or the dummy dry run.

---

## Pull request checklist

- [ ] `pip install -e .` works in a clean env; `requirements.txt` is fully pinned, CPU-only, with **no LLM/network/transformer** runtime deps.
- [ ] Every `src` subpackage has an `__init__.py`; imports resolve; no dataset committed (`.gitignore` verified).
- [ ] `docs/interfaces.md` matches the code; any changed signature is reflected there and announced.
- [ ] `ModelBundle` goes through `FeaturePipeline`/`compose_views` (no train/serve skew); missing-artifact errors are loud (F2).
- [ ] Service loads the model **once** at startup (warmed singleton); no per-request vectorizer rebuild (F18).
- [ ] `/explain` is grounded in real features (coef/importance + lexical highlight + attribute flags) with text **and** visual; **no LLM/API** in the path (F17) — `test_no_llm_imports` green.
- [ ] Docker image runs the dummy pipeline under `docker run --network none`; `no_network_check.sh` passes (F16).
- [ ] `scripts/run_dummy_pipeline.py` runs the full integration on dummy data with zero manual steps and writes the dry-run report.
- [ ] Roster checklist (F1) is present and wired as a pre-submission gate; rerun procedure validated on the dummy bundle.
- [ ] `champion.yaml::label_map` is config-driven (no hardcoded `var/yok` vs `0/1`); ownership map respected (no co-edited configs).
- [ ] `pytest -q tests/serving tests/service` green; pre-commit passes.

---

## Perfect completion criteria (measurable)

1. Clean-env `pip install -e .` succeeds offline; `python -c "import src, fastapi, lightgbm, catboost"` works; **0** banned imports in deps or `src/service/`.
2. Full `configs/` (13 files) + `src/**/__init__.py` tree exist; `docs/interfaces.md` freezes **every** cross-person signature with owner sign-off.
3. `ModelBundle.load` → `predict` returns a well-formed label on a dummy bundle; `validate_bundle` lists exactly what's missing on an incomplete one.
4. `/predict` boots warm and returns the contract shape; model loaded **once** (asserted in test).
5. `/explain` returns ≥ top-k real feature contributions + matched/unmatched tokens + attribute flags + a visual, fully offline; no-LLM scan green.
6. `docker run --network none ... dummy` completes; `no_network_check.sh` passes and flips to fail when a network/LLM import is injected.
7. `scripts/run_dummy_pipeline.py` runs the whole integration end-to-end on dummy data with **0** manual steps and writes `reports/dry_run/dummy_pipeline_report.md`.
8. Latency benchmark produces p50/p95 across ≥ 3 payload sizes (warmed); roster checklist + rerun procedure committed; `pytest -q tests/serving tests/service`: **0 failures**.

---

## First 24 hours after Kaggle release (26/06)

You own the **compliance, integration, and offline-readiness** spine of data day, in lockstep with Person 1 (schema/negatives) and Person 2 (folds/baseline/submission). Your job is to make the real schema flow through the structure with zero rebuild and to guard the DQ lines.

**H0 — Roster compliance, before anything else (F1).** Diff the Kaggle team against the KYS roster; captain signs `reports/compliance/kaggle_roster_check.md`. **No submission goes up until this is signed.** Simultaneously help Person 1 freeze the official `train/test/sample_submission` into `data/raw/` and collect their SHA-256 hashes into `reports/data/raw_data_manifest.json`.

**H1–2 — Wire the label map + submission format (with P2).** Inspect `sample_submission`; fill `champion.yaml::label_map.official`, `id_column`, `label_column` with the real encoding (`var/yok` or `0/1` or `related/not_related`). Confirm the daily submission limit and whether final subs are team-selectable, and record it. Make sure Person 2's submission validator reads your `label_map`.

**H2–6 — Point the integration at the real path.** As Person 1 fills `schema.yaml` and Person 2 fills folds, update `scripts/run_dummy_pipeline.py`'s configuration to run on the **real** canonical data path (same orchestration, real inputs). Keep `data.yaml` paths current. Goal: the integration that passed on dummy data now runs on real data with no code change.

**H6–14 — Keep the offline + reproducibility gates live.** Re-run `no_network_check.sh` against the real-data dummy/first run; confirm nothing introduced a network dependency. Stand up the rerun-check procedure so that the moment Person 2 produces the first baseline bundle, you can reproduce its OOF Macro-F1 from a clean env (F2).

**H22–38 — Support the first submission = format probe (with P2).** When the first baseline bundle exists, load it through `ModelBundle`, run one `/predict` to confirm the serving path works on real artifacts, and have Person 2 build the submission through the validated `label_map`. **Confirm the roster is signed before upload.** Log the probe; record the OOF↔public-LB gap as a probe, not a verdict.

**Do not in the first 24h:** build the real FastAPI service / Docker polish / SHAP UI (that's Stage-2 prep — don't let it eat pipeline time); put any LLM/API in the inference or explanation path; allow any submission before the roster check is signed; let the offline or rerun gates lapse.
