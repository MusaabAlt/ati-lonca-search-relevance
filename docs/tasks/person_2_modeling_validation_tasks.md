# Person 2 — Modeling & Validation Owner — Stage 1 Task Plan

> Project: TEKNOFEST 2026 E-Commerce **Search Relevance Engine** (Stage 1, Kaggle binary relevance: `related / not_related`, metric = **Macro-F1**).
> Follow the module names from `03_STAGE1_ARCHITECTURE.md` and the model order from `11_MODELING_ROADMAP_STAGE1.md` exactly. **Baselines before complexity.** No transformers/LLM/API as predictors. Threshold tuned on **OOF only**. Champion chosen by **robust validation, not public LB**.

---

## Role

You own the **honest validation loop** and the **baseline-first model stack**. Your job is to make sure a model's local score actually predicts its private-LB behavior, given that **our negatives are synthetic while Kaggle's are organizer-made** — that gap is the central risk of the whole competition.

You own:

1. **Metric + validation framework** — Macro-F1, grouped leak-safe folds, OOF storage, five validation tracks, gates (`src/validation/`).
2. **Threshold + calibration** — OOF threshold search, **prior-mismatch sweep**, isotonic/Platt calibration (`src/validation/`).
3. **Sparse signal + features** — TF-IDF (word + char_wb), BM25 per field, overlap/category/attribute/numeric features, the `FeaturePipeline` (`src/retrieval/`, `src/features/`).
4. **Models** — rule/threshold baseline → Logistic Regression / LinearSVC → LightGBM → CatBoost (`src/models/`).
5. **Experiment discipline** — CV runner, manifest, experiment log, submission builder + validator (`src/utils/`, `src/submission/`).

You provide two interfaces the others depend on:

- `src/validation/splits.py::assign_folds` — Person 1 needs `fold_id` for split-before-sample.
- `src/retrieval/bm25.py::BM25Index.topk` — Person 1's hard-lexical miner candidate source.
- `src/features/build_matrix.py::FeaturePipeline` (with `.transform_one` + `feature_spec.json`) — Person 3's `/predict` and `/explain` run on it.

---

## Mission (what must exist before the 26/06 data release)

A **validation + baseline skeleton that runs on dummy data** and is correct by construction:

- Macro-F1 matches `sklearn` exactly; OOF storage works.
- `assign_folds` does **GroupKFold by `query_norm`** with **fuzzy clustering of typo variants** so near-duplicate queries can't straddle folds (failure F6).
- Threshold search runs `0.02→0.98` on OOF and reports `0.5` vs tuned; **prior sweep** + calibration hooks exist (failures F4/F13).
- TF-IDF/BM25 are wrapped so they **physically cannot fit outside a fold** (failure F8); `BM25Index.topk` is ready as Person 1's candidate source.
- Logistic Regression baseline trains via the CV runner on dummy features and emits a full manifest.
- Submission validator + builder exist; the first real submission will be a **format probe**.
- `pytest -q tests/validation tests/features tests/models tests/io` green.

You do **not** assume class balance, label encoding, or the test negative distribution.

---

## Files to create or edit

```text
# Validation framework
src/validation/__init__.py
src/validation/metrics.py
src/validation/splits.py                 # FROZEN INTERFACE (P1 imports assign_folds)
src/validation/leakage.py
src/validation/oof.py
src/validation/thresholds.py
src/validation/calibration.py
src/validation/tracks.py
src/validation/gates.py
configs/validation.yaml                  # you own
configs/thresholds.yaml                  # you own

# Retrieval / sparse scoring
src/retrieval/__init__.py
src/retrieval/tfidf.py
src/retrieval/bm25.py                     # FROZEN INTERFACE (P1 imports .topk)

# Features
src/features/__init__.py
src/features/lexical.py
src/features/overlap.py
src/features/category.py
src/features/attributes.py
src/features/numeric_units.py
src/features/build_matrix.py              # FROZEN INTERFACE (P3 imports FeaturePipeline)
configs/features.yaml                     # you own

# Models
src/models/__init__.py
src/models/rule_baseline.py
src/models/linear_baseline.py
src/models/train_lgbm.py
src/models/train_catboost.py
src/models/predict.py
configs/model_linear.yaml                 # you own
configs/model_lgbm.yaml                    # you own
configs/model_catboost.yaml                # you own

# Experiment + submission + EDA
src/utils/__init__.py
src/utils/experiment.py
src/utils/hashing.py
src/utils/seed.py
src/utils/io.py
src/experiments/run_cv.py
src/experiments/run_eda.py
src/submission/__init__.py
src/submission/build_submission.py
src/submission/validator.py
configs/experiment.yaml                    # you own

# Tests + docs
tests/validation/test_metrics.py
tests/validation/test_splits.py
tests/validation/test_leakage.py
tests/validation/test_thresholds.py
tests/features/test_fit_in_fold.py
tests/features/test_overlap.py
tests/features/test_feature_pipeline.py
tests/models/test_baseline_smoke.py
tests/io/test_submission_validator.py
docs/validation_protocol.md
docs/feature_spec.md
```

---

## Shared frozen interfaces you must publish

```python
# src/validation/splits.py
def assign_folds(df, *, group_col="query_norm", n_splits=5, seed=42,
                 fuzzy_cluster=True) -> "pd.DataFrame": ...     # adds fold_id, group_id
def iter_folds(df) -> "Iterator[tuple[np.ndarray, np.ndarray]]": ...

# src/retrieval/bm25.py
class BM25Index:
    def fit(self, corpus_fields: dict) -> "BM25Index": ...      # fit on TRAIN-FOLD corpus only
    def score(self, query_norm: str, field: str) -> float: ...
    def topk(self, query_norm: str, k: int) -> list[tuple[str, float]]: ...   # (product_id, score)

# src/features/build_matrix.py
class FeaturePipeline:
    def fit_fold(self, train_df) -> "FeaturePipeline": ...
    def transform(self, df) -> tuple["np.ndarray", list[str]]: ...
    def transform_one(self, record: dict) -> "np.ndarray": ...  # inference path (P3)
    def save(self, out_dir: str) -> None: ...                   # writes feature_spec.json + vectorizers/ + bm25/
    @classmethod
    def load(cls, in_dir: str) -> "FeaturePipeline": ...
```

`transform_one` must reuse Person 1's `compose_views` so the **service uses the same code path as training** (no train/serve skew).

---

## Task list

### P2-01 — Macro-F1 metric & binary report
- **Purpose:** A correct, self-owned Macro-F1 (Kaggle's metric) plus the diagnostic report every run must emit.
- **Files:** `src/validation/metrics.py`
- **Technical details:** `macro_f1(y_true, y_pred)` = unweighted mean of per-class F1 over `{0,1}` (handle the empty-class edge case). `binary_report(y_true, y_prob, threshold)` returns Macro-F1, per-class F1, precision/recall per class, positive recall, negative recall, confusion matrix. Validate against `sklearn.metrics.f1_score(average="macro")` to ±1e-9 on random arrays.
- **Expected functions/classes/config keys:** `macro_f1`, `per_class_f1`, `confusion`, `recall_by_class`, `binary_report(y_true, y_prob, threshold)->dict`.
- **Input/output contract:** in = 1-D int/array of labels + optional probs + threshold; out = float / dict. Deterministic.
- **Acceptance criteria:** matches sklearn macro-F1 on 1000 random cases; handles all-positive and all-negative degenerate inputs without crashing.
- **Tests:** `tests/validation/test_metrics.py` (parity vs sklearn, degenerate classes, threshold application).
- **Common mistakes:** using accuracy/AUC as the objective; weighting F1 by support (that's weighted-F1, not macro).
- **Dependencies:** none. **Build first** — every other task reports through it.
- **Definition of Done:** parity test green; `binary_report` schema documented in `docs/validation_protocol.md`.

### P2-02 — Grouped split + fuzzy typo clustering + `validation.yaml`  *(FROZEN INTERFACE)*
- **Purpose:** Leak-safe folds. Group by normalized query, but first cluster typo/near-duplicate queries so variants of the same query can't appear in both train and validation.
- **Files:** `src/validation/splits.py`, `configs/validation.yaml`
- **Technical details:** `assign_folds` first builds query clusters (`fuzzy_query_clusters`): char-3gram Jaccard union-find over `query_norm` with `threshold` (default 0.9) — variants merge into one `group_id`. Then `GroupKFold(n_splits)` over `group_id`. Output adds `fold_id` and `group_id`. `iter_folds` yields `(train_idx, val_idx)`. Seeded and deterministic.
- **Expected functions/classes/config keys:** `assign_folds`, `iter_folds`, `fuzzy_query_clusters(queries, cfg)->labels`. Config: `split.{scheme:group_kfold, group_key:query_norm, n_splits:5, seed:42, fuzzy_cluster:true, fuzzy.method:char3gram_jaccard_unionfind, fuzzy.threshold:0.9}`, `product_overlap_policy: audit`, `positive_recall_floor: 0.90`, `decision_metric: worst_of_t2_t3_t5`.
- **Input/output contract:** in = DataFrame with `query_norm`; out = same + `fold_id`, `group_id`. Same input ⇒ same folds.
- **Acceptance criteria:** no `query_norm` appears in two folds; injected typo variants (`kahve makinası`/`kahve makinesi`) land in the **same** fold; determinism across runs.
- **Tests:** `tests/validation/test_splits.py` (no query overlap, typo-variant collision test F6, determinism).
- **Common mistakes:** grouping by raw query (typos split → leak); plain KFold (product/query leakage); non-deterministic clustering.
- **Dependencies:** Person 1's `query_norm` (from `compose_views`). **Person 1 imports this for split-before-sample.**
- **Definition of Done:** overlap + typo-collision tests green; published in `docs/interfaces.md`.

### P2-03 — Cross-fold leakage assertions
- **Purpose:** Make leakage a hard failure, not a hope. Enforce the product-overlap policy and query isolation at runtime.
- **Files:** `src/validation/leakage.py`
- **Technical details:** `assert_no_query_overlap(train_df, val_df)` raises if any `query_norm` is shared. `count_product_overlap(folds)` reports product_id overlap across folds; `assert_product_overlap_policy(folds, policy)` enforces `forbid` (raise) or `audit` (log + record in report) per `validation.yaml::product_overlap_policy`. `assert_vectorizer_not_fit_on_test(...)` guard hook for P2-06.
- **Expected functions/classes/config keys:** `assert_no_query_overlap`, `count_product_overlap`, `assert_product_overlap_policy`.
- **Input/output contract:** in = fold frames / fold list; out = None (raises on violation) or a small report dict.
- **Acceptance criteria:** an injected shared query across folds raises; product-overlap count is correct on a constructed example.
- **Tests:** `tests/validation/test_leakage.py` (query overlap raises; product overlap counted; policy modes).
- **Common mistakes:** auditing product overlap but never enforcing or documenting a decision; forgetting to call the asserter inside the CV runner.
- **Dependencies:** P2-02. Decision on `forbid` vs `audit` documented in `docs/validation_protocol.md`.
- **Definition of Done:** asserters wired into `run_cv`; tests green; policy decision written down.

### P2-04 — OOF prediction storage
- **Purpose:** Persist out-of-fold predictions so threshold tuning and error analysis never touch in-fold or test data.
- **Files:** `src/validation/oof.py`
- **Technical details:** `OOFStore.save(run_dir, df)` writes `oof_predictions.parquet` with `pair_id, fold_id, y_true, y_prob`. `OOFStore.load(run_dir)` reads it. Assert every training row appears exactly once across OOF; assert no test row is present.
- **Expected functions/classes/config keys:** `class OOFStore(.save, .load)`, `assert_oof_complete(df, n_train_rows)`.
- **Input/output contract:** in = per-fold val predictions assembled into one frame; out = parquet on disk + reload parity.
- **Acceptance criteria:** OOF coverage == train rows, each exactly once; reload equals saved.
- **Tests:** part of `tests/validation/test_thresholds.py` (uses OOF fixtures).
- **Common mistakes:** including in-fold predictions; double-counting rows; storing test predictions in OOF.
- **Dependencies:** P2-02.
- **Definition of Done:** OOF round-trips; completeness asserted in the CV runner.

### P2-05 — Threshold tuning + prior-mismatch sweep + calibration + `thresholds.yaml`
- **Purpose:** Pick a Macro-F1 threshold that **transfers** despite our OOF positive-rate ≠ the private test positive-rate. Add probability calibration before thresholding for tree models. (Failures F4, F13.)
- **Files:** `src/validation/thresholds.py`, `src/validation/calibration.py`, `configs/thresholds.yaml`
- **Technical details:** `tune_threshold(y_true, y_prob, lo=0.02, hi=0.98, step=0.01, metric=macro_f1)` returns best threshold + the full F1-vs-threshold curve, plus Macro-F1 at `0.5`. `prior_sweep(y_true, y_prob, priors)` resamples OOF to a set of assumed test positive-rates and reports the Macro-F1-optimal threshold at each — then `select_robust_threshold` picks the threshold that is **stable across priors and across tracks** (tie-breakers: better worst-class F1 → lower cross-track variance → safer positive recall). `calibration.py`: `fit_calibrator(y_true, y_prob, method="isotonic")` (Platt as alt), applied to tree-model probabilities before thresholding.
- **Expected functions/classes/config keys:** `tune_threshold`, `prior_sweep`, `select_robust_threshold`, `ThresholdReport` (dataclass → `thresholds.json`), `fit_calibrator`, `apply_calibrator`. Config: `search.{lo:0.02,hi:0.98,step:0.01,metric:macro_f1}`, `prior_sweep.{assumed_positive_rates:[0.3,0.4,0.5,0.6,0.7],resample:true}`, `calibration.{enabled_for_trees:true,method:isotonic}`, `tie_breakers`, `default_diagnostic:0.5`.
- **Input/output contract:** in = OOF `y_true`,`y_prob` (+ track OOF for stability); out = `ThresholdReport` (best threshold, F1@0.5, F1@tuned, per-class F1, per-prior thresholds, cross-track stability) serialized to `thresholds.json`.
- **Acceptance criteria:** tuned threshold ≥ F1@0.5 on OOF; prior sweep produces a threshold per assumed rate; a threshold that wins one track but collapses another is rejected by `select_robust_threshold`.
- **Tests:** `tests/validation/test_thresholds.py` (tuned ≥ 0.5 baseline; OOF-only enforced; prior sweep shape; stability tie-break).
- **Common mistakes:** shipping one OOF-optimal threshold (miscalibrated to the private prior); tuning on train predictions; using `0.5` as final without comparison; thresholding uncalibrated tree scores.
- **Dependencies:** P2-01, P2-04, and P2-13 track OOF for stability.
- **Definition of Done:** `thresholds.json` produced by the CV runner; prior sweep + calibration wired; tests green.

### P2-06 — Fit-in-fold TF-IDF + BM25 + `features.yaml`  *(BM25 FROZEN INTERFACE)*
- **Purpose:** Sparse lexical signal that **cannot leak** (fit inside fold only), plus the BM25 candidate source Person 1 mines hard negatives from. (Failure F8; audit Area 5 — BM25 is a hard deliverable, not "if available".)
- **Files:** `src/retrieval/tfidf.py`, `src/retrieval/bm25.py`, `configs/features.yaml`
- **Technical details:** `FoldTfidf` wraps `TfidfVectorizer` (word `ngram_range=(1,2)`, and a separate `char_wb (3,5)`), exposing `.fit(train_texts)`, `.transform(texts)`, `.cosine(a,b)` — with a runtime guard that raises if `.transform` is called on data not seen as "train-fold context" while in validation, and a hard assert it is never `.fit` on test. `BM25Index` indexes title/category/attributes/all_text separately; `.score(query,field)`, `.topk(query,k)` (the hard-neg source). N-gram ranges and BM25 `k1,b` are **pinned in `features.yaml`**, never in code.
- **Expected functions/classes/config keys:** `class FoldTfidf`, `class BM25Index` (`.fit/.score/.topk`). Config: `tfidf_word.{ngram_range:[1,2],min_df:2,max_features:200000,sublinear_tf:true}`, `tfidf_char.{analyzer:char_wb,ngram_range:[3,5],min_df:2,max_features:300000}`, `bm25.{k1:1.5,b:0.75,fields:[title,category,attributes,all_text]}`.
- **Input/output contract:** in = train-fold texts → fit; val texts → transform; out = sparse matrices / cosine floats / topk lists. Index is fold-scoped during CV; a full-train index is built only for the final submission model.
- **Acceptance criteria:** a test that `.fit` on validation/test raises; identical seeds ⇒ identical vectors; `BM25Index.topk` returns ranked `(product_id, score)`.
- **Tests:** `tests/features/test_fit_in_fold.py` (fit-on-test raises; fold isolation; BM25 topk ordering).
- **Common mistakes:** fitting TF-IDF on `train+val` "for convenience"; pinning n-grams in code; dropping BM25 under time pressure.
- **Dependencies:** Person 1's text views (`all_text_norm`, field views). **Person 1's hard-lexical miner imports `BM25Index.topk`.**
- **Definition of Done:** fit-in-fold guard test green; BM25 topk ready; n-gram/BM25 params in `features.yaml`; published in `docs/interfaces.md`.

### P2-07 — Feature groups: overlap, category, attribute, numeric, lexical
- **Purpose:** The explicit relevance features that are the real backbone of the model (cheap, CPU-friendly, predictive for term↔product).
- **Files:** `src/features/lexical.py`, `src/features/overlap.py`, `src/features/category.py`, `src/features/attributes.py`, `src/features/numeric_units.py`
- **Technical details:** **lexical**: BM25 per field, word/char TF-IDF cosine per field, exact-phrase hit, query coverage. **overlap**: token Jaccard query↔title/category/attrs, char-ngram cosine, brand-token match. **category**: category-path depth, leaf match, ancestor overlap, category-token overlap. **attributes**: attr count, key overlap, value overlap, **missingness flags**, and **contradiction flags** (e.g., `şekerli` vs `şekersiz`, color/material/size mismatch) using Person 1's `NEGATION_MARKERS`/`COLOR_TERMS`. **numeric_units**: exact number match, dimension match (`14x14`), unit-type match, capacity/size match, model-code match. **Missingness features are mandatory**: `is_title_missing`, `is_category_missing`, `is_attrs_missing`, `attr_count`, `category_depth`, `has_number_query/product`, `has_unit_query/product`.
- **Expected functions/classes/config keys:** one `build(pair_views: dict) -> dict[str,float]` per module + a `FEATURE_NAMES` list per module. Toggled by `features.yaml::feature_groups.{lexical,overlap,category,attributes,numeric_units,interactions}`.
- **Input/output contract:** in = composed views for one pair (+ fitted vectorizers/BM25 for lexical); out = ordered float feature dict; missing field → defined default (e.g., 0 + missingness flag), never NaN-by-accident.
- **Acceptance criteria:** every feature has a stable name and a defined missing-value behavior; contradiction flag fires on `şekersiz` vs `şekerli`.
- **Tests:** `tests/features/test_overlap.py` (Jaccard correctness, contradiction flag, missingness defaults, dimension match).
- **Common mistakes:** silent NaNs from missing attributes; collapsing field identity; forgetting contradiction/missingness features (they are Stage-2 load-bearing too).
- **Dependencies:** P2-06 (lexical needs vectorizers/BM25); Person 1's dictionaries/units.
- **Definition of Done:** all groups produce named features with missing-value handling; `docs/feature_spec.md` lists every feature.

### P2-08 — `FeaturePipeline` assembler + `transform_one` + `feature_spec.json`  *(FROZEN INTERFACE)*
- **Purpose:** One object that fits inside a fold, transforms batches for training, transforms a single record for serving, and serializes itself for Person 3's `ModelBundle`.
- **Files:** `src/features/build_matrix.py`
- **Technical details:** `FeaturePipeline.fit_fold(train_df)` fits TF-IDF/BM25 on the train fold and records feature order. `.transform(df)` → `(X, feature_names)`. `.transform_one(record)` calls Person 1's `compose_views(record)` then runs the same feature builders → identical features at train and serve time. `.save(dir)` writes `feature_spec.json` (ordered feature names + group toggles + vectorizer/BM25 paths + interaction definitions) and persists `vectorizers/` and `bm25/`. `.load(dir)` reconstructs it. **Interaction features** (`bm25_title*category_match`, `attr_conflict*lexical_score`, `brand_match*char_similarity`) are computed here from base features so they're consistent everywhere.
- **Expected functions/classes/config keys:** `class FeaturePipeline(.fit_fold,.transform,.transform_one,.save,.load)`; writes `feature_spec.json`.
- **Input/output contract:** training: DataFrame → dense `X` + names; serving: one record dict → 1-D vector in the **same column order** as `feature_spec.json`.
- **Acceptance criteria:** `transform_one(record)` equals the matching row of `transform(df)` for the same pair (no skew); `save`→`load` reproduces identical features.
- **Tests:** `tests/features/test_feature_pipeline.py` (train/serve parity; save/load parity; feature order stability).
- **Common mistakes:** different code paths for batch vs single (skew); not persisting feature order (service produces misaligned vectors); fitting vectorizers in `transform`.
- **Dependencies:** P2-06/07; Person 1's `compose_views`. **Person 3's `/predict` and `/explain` import this; the artifact bundle layout is shared.**
- **Definition of Done:** parity + save/load tests green; `feature_spec.json` schema in `docs/interfaces.md`.

### P2-09 — Rule + linear baselines + `model_linear.yaml`
- **Purpose:** Tier 0 (threshold baselines) and Tier 1 (first supervised classifier) — the floor every later model must beat.
- **Files:** `src/models/rule_baseline.py`, `src/models/linear_baseline.py`, `src/models/predict.py`, `configs/model_linear.yaml`
- **Technical details:** `rule_baseline.py`: BM25/TF-IDF-cosine threshold model + simple overlap rule — produces OOF + a valid submission, used forever as debug/fallback/explanation anchor and hard-neg source. `linear_baseline.py`: Logistic Regression (default, `class_weight=balanced`, `liblinear`) and LinearSVC (with `CalibratedClassifierCV` when probabilities are needed). `predict.py`: shared `predict_proba(model, X)`.
- **Expected functions/classes/config keys:** `fit_rule_baseline`, `RuleThresholdModel`, `fit_linear(X, y, cfg)`, `predict_proba`. Config: `model: logistic_regression`, `logreg.{C:1.0,class_weight:balanced,max_iter:1000,solver:liblinear}`, `linear_svc.{C:1.0,class_weight:balanced,calibrate:true,calibration_method:sigmoid}`.
- **Input/output contract:** in = feature matrix + labels; out = fitted model + OOF probabilities via the CV runner.
- **Acceptance criteria:** LogReg trains on dummy features, produces OOF, beats (or explains vs) the rule baseline; positive recall not collapsed.
- **Tests:** `tests/models/test_baseline_smoke.py` (LogReg fits, OOF shape correct, probabilities in [0,1]).
- **Common mistakes:** skipping the Tier-0 baseline; using LinearSVC decision scores as if they were probabilities (calibrate first); starting with trees before the linear baseline is logged.
- **Dependencies:** P2-08 (features), P2-12 (CV runner), P2-05 (threshold).
- **Definition of Done:** rule + LogReg both run through `run_cv` and emit manifests; smoke test green.

### P2-10 — LightGBM / CatBoost skeletons + configs
- **Purpose:** Tier-2 models — fast strong tabular model (LGBM) and a categorical-aware challenger (CatBoost) — gated behind robust validation.
- **Files:** `src/models/train_lgbm.py`, `src/models/train_catboost.py`, `configs/model_lgbm.yaml`, `configs/model_catboost.yaml`
- **Technical details:** LGBM over the explicit feature matrix; CatBoost receives **raw `category_path` as a categorical feature** (uses its categorical edge — otherwise the "challenger" rationale is empty; audit Area 7). Both output OOF probabilities → **calibrated (P2-05) before thresholding** (failure F13). Skeletons accept a config and run via `run_cv`; hyper-tuning happens after data.
- **Expected functions/classes/config keys:** `fit_lgbm(X,y,cfg)`, `fit_catboost(df,y,cfg,cat_features)`. Config: LGBM `{num_leaves, learning_rate, n_estimators, min_child_samples, subsample, colsample_bytree, class_weight}`; CatBoost `{depth, learning_rate, iterations, l2_leaf_reg, cat_features:[category_path], auto_class_weights:Balanced}`. Both include `calibrate: true`.
- **Input/output contract:** in = features (+ categorical for CatBoost) + labels; out = OOF probs + feature importance.
- **Acceptance criteria:** both run on dummy data through the CV runner; CatBoost actually consumes the categorical column; importances emitted.
- **Tests:** smoke-level inclusion in `tests/models/test_baseline_smoke.py` (skippable if libs heavy on CI, but must run locally).
- **Common mistakes:** promoting on mean OOF alone; thresholding uncalibrated tree scores; not giving CatBoost a categorical feature; tuning before baselines are logged.
- **Dependencies:** P2-08, P2-12, P2-05, P2-13.
- **Definition of Done:** both skeletons run end-to-end on dummy data; promotion deferred to track-based gates.

### P2-11 — Five-track harness + gates
- **Purpose:** Detect sampler-artifact overfitting. A model is judged on **worst of T2/T3/T5 + positive-recall floor + threshold stability**, never mean OOF alone. (Failure F5.)
- **Files:** `src/validation/tracks.py`, `src/validation/gates.py`
- **Technical details:** `build_tracks` constructs T1 easy, T2 medium/category, T3 hard-lexical, T4 attribute-conflict, T5 transfer-stress (**negatives from a different recipe than training** — the artifact detector, labeled "proxy, not guarantee"). `evaluate_tracks(model, data, cfg)` returns per-track Macro-F1 + per-class F1 + positive/negative recall. `gates.run_gates(run_dir)` enforces the pre-submission gates: leakage, FN-audit present, determinism, transfer-stress present, per-class sanity, threshold robustness, submission format, manifest present.
- **Expected functions/classes/config keys:** `build_tracks`, `evaluate_tracks`, `decision_score(track_scores)` (= worst of T2/T3/T5 subject to recall floor), `run_gates(run_dir)->GateResult`. Config: `validation.yaml::tracks`, `decision_metric`, `positive_recall_floor`.
- **Input/output contract:** in = trained model + track datasets; out = `track_metrics.json` + a pass/fail `GateResult`.
- **Acceptance criteria:** all five tracks score; `decision_score` returns the worst of T2/T3/T5; gates fail loudly when any required artifact/threshold/leakage check is missing.
- **Tests:** track construction sanity in `tests/validation/test_splits.py`/`test_leakage.py`; a `run_gates` unit test with a deliberately-incomplete run dir.
- **Common mistakes:** selecting champion on T1/mean OOF; treating T5 as ground truth (it's a proxy); letting a submission through with a failed gate.
- **Dependencies:** P2-02..P2-10; Person 1's negative recipes for T-tracks (T5 uses a different recipe).
- **Definition of Done:** five tracks + `decision_score` + gates implemented; gate failure blocks submission in the runner.

### P2-12 — CV runner + experiment logging + hashing/seed + `experiment.yaml`
- **Purpose:** The orchestrator that makes every experiment reproducible and logged — directly answers the top-20 inspection.
- **Files:** `src/experiments/run_cv.py`, `src/utils/experiment.py`, `src/utils/hashing.py`, `src/utils/seed.py`, `src/utils/io.py`, `configs/experiment.yaml`
- **Technical details:** `run_cv(config)` executes the fixed control flow: freeze config → `assign_folds` → (consume Person 1 fold-aware negatives) → `FeaturePipeline.fit_fold`/`transform` per fold → train → assemble OOF → `tune_threshold`(+prior sweep + calibration) → `evaluate_tracks` → `run_gates` → write the artifact bundle + manifest + append `experiment_log.csv`. `experiment.py`: `Manifest` dataclass (all fields from `12_EXPERIMENT_PROTOCOL.md §7`), `write_manifest`, `append_experiment_log` (columns from §8). `hashing.py`: `hash_df`, `hash_file`, `hash_config`. `seed.py`: `set_seed(42)`. Run-id format `YYYYMMDD_HHMM_<short_model>_<short_feature>_<short_sampling>`.
- **Expected functions/classes/config keys:** `run_cv(config)->run_id`, `class Manifest`, `write_manifest`, `append_experiment_log`, `make_run_id(...)`, `hash_df/hash_file/hash_config`, `set_seed`. Config: `experiment.yaml::{run_id_format, artifact_root: artifacts/runs, log_path: reports/experiments/experiment_log.csv, random_seed: 42}`.
- **Input/output contract:** in = a frozen config; out = a populated `artifacts/runs/<run_id>/` bundle (manifest.json, config_snapshot.yaml, model.*, vectorizers/, bm25/, thresholds.json, feature_spec.json, cv_metrics.json, track_metrics.json, oof_predictions.parquet, error_analysis.md, submission.csv if submitted) + one appended CSV row.
- **Acceptance criteria:** a dummy `run_cv` produces a complete bundle and a deterministic re-run yields the same OOF Macro-F1 within tolerance; missing artifacts cause the run to fail its own gate.
- **Tests:** a smoke run in `tests/models/test_baseline_smoke.py` asserting the bundle is complete.
- **Common mistakes:** results not written to a manifest (then "the result doesn't exist"); changing several factors per run (one-variable rule); non-deterministic runs.
- **Dependencies:** all P2 modules; Person 1's processed negatives; the bundle layout is shared with Person 3.
- **Definition of Done:** one command produces a full reproducible bundle on dummy data; experiment log row written; bundle layout matches `docs/interfaces.md`.

### P2-13 — Submission builder + validator
- **Purpose:** Make the first Kaggle submission a guaranteed-valid **format probe**, and prevent label-encoding/format zeros that waste scarce submissions. (Failure F9.)
- **Files:** `src/submission/build_submission.py`, `src/submission/validator.py`
- **Technical details:** `build_submission(test_ids, labels_int, sample_submission, run_id, label_map)` maps the internal int label (`1=related`,`0=not related`) to the **official encoding** (`label_map` filled on data day — `var/yok` or `0/1` or `related/not_related`), preserves test row order, and writes `data/submissions/<run_id>_submission.csv`. `validate_submission(sub_df, sample_submission)` checks: exact column names, row count == sample, id alignment, no NaN, label values ∈ allowed set, and compares against the sample-submission hash.
- **Expected functions/classes/config keys:** `build_submission(...)->path`, `validate_submission(sub_df, sample_submission)->ValidationResult`. Reads `champion.yaml::label_map` (owned by Person 3, filled on data day).
- **Input/output contract:** in = test ids + int predictions + sample submission; out = validated CSV path or a raised error listing exactly what's wrong.
- **Acceptance criteria:** validator rejects wrong column names, wrong row count, NaNs, and out-of-set labels; builder reproduces the same file from a run_id.
- **Tests:** `tests/io/test_submission_validator.py` (each rejection path; happy path).
- **Common mistakes:** assuming `0/1` vs `var/yok` (configure it); reordering rows; submitting before validating; unlogged submissions.
- **Dependencies:** `label_map` from Person 3's `champion.yaml`; predictions from `run_cv`.
- **Definition of Done:** validator catches all four failure classes in tests; first real submission is a format probe.

### P2-14 — EDA script
- **Purpose:** Hour-6 EDA that answers how labels/test-negatives might differ and sizes the CV↔LB risk before serious modeling.
- **Files:** `src/experiments/run_eda.py`
- **Technical details:** Emit row counts, missing values, duplicates, unique queries/products, query/product frequency, category coverage + depth distribution, attribute coverage, label values, query/title length distributions, train vs test schema differences, and sample 100/1000 rows. Runs on canonical records from Person 1's adapter.
- **Expected functions/classes/config keys:** `run_eda(canonical_train, canonical_test, sample_submission)->writes reports/eda/*`.
- **Input/output contract:** in = canonical frames; out = `reports/eda/initial_eda.md`, `sample_100_rows.csv`, `sample_1000_rows.csv`, `train_test_difference_notes.md`.
- **Acceptance criteria:** runs on dummy canonical data pre-release; produces all four outputs.
- **Tests:** smoke run on fixtures (no assertion beyond "files produced").
- **Common mistakes:** skipping EDA and jumping to modeling; ignoring train/test schema diffs.
- **Dependencies:** Person 1's adapter output.
- **Definition of Done:** EDA runs on dummy data; ready to run on real data at H6.

### P2-15 — Tests + validation/feature docs
- **Purpose:** Lock correctness and document the validation protocol + feature spec for teammates and the inspection.
- **Files:** all `tests/validation/*`, `tests/features/*`, `tests/models/*`, `tests/io/*`; `docs/validation_protocol.md`, `docs/feature_spec.md`
- **Technical details:** `pytest -q tests/validation tests/features tests/models tests/io` green. `docs/validation_protocol.md` covers Macro-F1, fold scheme + fuzzy clustering, leakage policy, OOF rule, threshold + prior sweep + calibration, five tracks + decision metric, gates, run-id + manifest. `docs/feature_spec.md` lists every feature, its group, and its missing-value behavior.
- **Acceptance criteria:** green suite; both docs complete and consistent with the configs.
- **Tests:** the suite.
- **Common mistakes:** docs drifting from configs; untested leakage/threshold paths.
- **Dependencies:** P2-01..P2-14.
- **Definition of Done:** green suite + two docs; `docs/interfaces.md` reflects your frozen signatures.

---

## Daily execution order

**Day 0 (24/06) — unblock + correctness primitives**
1. P2-01 Macro-F1 (everything reports through it).
2. P2-02 `assign_folds` + fuzzy clustering → **publish to Person 1 immediately** (split-before-sample is blocked without it).
3. P2-06 `BM25Index` skeleton → **publish `.topk` to Person 1** (hard-neg source).
4. P2-08 `FeaturePipeline` interface stub (`transform_one` signature) → publish (Person 3 needs it).
5. Co-author `docs/interfaces.md` (your half) with Person 3.

**Day 1 (25/06) — build the loop**
6. P2-03 leakage asserts, P2-04 OOF store, P2-05 threshold + prior sweep + calibration.
7. P2-06 finish TF-IDF/BM25 with fit-in-fold guard; P2-07 feature groups; P2-08 full FeaturePipeline (parity + save/load).
8. P2-09 rule + LogReg; P2-12 CV runner + manifest/log (run LogReg end-to-end on dummy).
9. P2-13 submission builder + validator; P2-11 tracks + gates; P2-10 LGBM/CatBoost skeletons.
10. P2-14 EDA script; P2-15 tests + docs; dummy dry-run with Person 3.

**Day 2 morning (26/06, pre-data)**
11. Freeze interfaces; confirm `run_cv` produces a complete bundle on dummy data; stand ready.

> Minimum viable path if behind: P2-01 → P2-02 → P2-06(TF-IDF+BM25) → P2-08(features) → P2-09(LogReg) → P2-05(threshold+prior) → P2-12(run_cv+manifest) → P2-13(submission validator). Tracks/gates can start partial (transfer-stress mandatory from the **second** serious submission).

---

## Pull request checklist

- [ ] Objective is **Macro-F1** everywhere; no accuracy/AUC selection sneaking in.
- [ ] Folds: GroupKFold by `query_norm` with fuzzy typo clustering; no query in two folds (test asserts).
- [ ] Threshold tuned on **OOF only**; `0.5` compared vs tuned; prior sweep + calibration applied for trees.
- [ ] TF-IDF/BM25 **fit-in-fold**; fit-on-test raises in a test; n-gram/BM25 params live in `features.yaml`.
- [ ] `FeaturePipeline.transform_one` == matching `transform` row (no train/serve skew); save/load parity holds.
- [ ] Model order respected: rule/LogReg logged before LGBM/CatBoost; CatBoost gets a categorical feature.
- [ ] `run_cv` writes a complete bundle + manifest + appends `experiment_log.csv`; re-run reproduces OOF Macro-F1.
- [ ] Submission validator rejects wrong columns/rows/NaN/labels; label encoding is config-driven.
- [ ] Gates block submission when leakage/threshold/manifest checks fail.
- [ ] `docs/interfaces.md`, `docs/validation_protocol.md`, `docs/feature_spec.md` updated.

---

## Perfect completion criteria (measurable)

1. `macro_f1` matches sklearn to **±1e-9** on 1000 random cases.
2. Fold tests: **0** query overlaps across folds; typo variants collide in the same fold; deterministic.
3. Fit-in-fold guard: a fit-on-test attempt **raises** in tests; identical seeds ⇒ identical vectors.
4. Threshold: tuned Macro-F1 **≥** F1@0.5 on OOF; prior sweep returns a threshold per assumed positive-rate; unstable thresholds rejected.
5. `FeaturePipeline`: train/serve parity **0** mismatches; save→load **0** mismatches; stable feature order.
6. `run_cv` on dummy data produces a **complete** bundle; a clean re-run reproduces OOF Macro-F1 within tolerance.
7. Submission validator catches **4/4** failure classes (columns, row count, NaN, label set).
8. `pytest -q tests/validation tests/features tests/models tests/io`: **0 failures**.

---

## First 24 hours after Kaggle release (26/06)

You own **Hours 6–38** of the runbook (folds → features → baseline → threshold → first submission), in lockstep with Person 1 (schema/negatives) and Person 3 (raw freeze, roster, format).

**H1–2 (with P3) — confirm the metric & submission format.** Confirm Kaggle's metric is Macro-F1, the exact submission columns, label encoding, daily submission limit, and whether final subs are team-selectable. Fill `champion.yaml::label_map` with Person 3.

**H6–10 — EDA.** Run `run_eda` on real canonical data (from Person 1's adapter). Record class signal, query/product frequency, category depth, attribute coverage, and **train vs test schema differences**. This sizes the CV↔LB gap before you trust either.

**H10–14 — Folds on real queries.** Run `assign_folds` on the real `query_norm`; run the typo-collision check and `assert_no_query_overlap`; decide+document the product-overlap policy (`forbid` vs `audit`). Publish `fold_id` to Person 1 so negatives are fold-aware on real data.

**H14–18 — Fit-in-fold sparse features on the real schema.** Fit TF-IDF (word + char_wb) and BM25 per field inside folds; confirm the fit-in-fold assertions are green on real columns. Stand up `BM25Index.topk` for Person 1's hard-lexical negatives. Build overlap/category/attribute/numeric features; confirm missingness handling on real missing fields.

**H22–30 — First baseline → OOF.** Run rule baseline then Logistic Regression through `run_cv` on the real fold-aware data from Person 1. Save OOF, confirm OOF completeness, emit a full manifest. **Reproduce the baseline from a clean re-run before building further** (failure F2).

**H30–34 — Threshold (prior-robust).** Tune the threshold on OOF (`0.02→0.98`), run the **prior sweep** over assumed test positive-rates, compare vs `0.5`, and select a prior-robust threshold. Save `thresholds.json`.

**H34–38 — First submission = format probe.** Build the submission with the real `label_map`, run the validator against the real `sample_submission` (and its hash), and have Person 3 confirm the roster before upload. Log it in `reports/kaggle/lb_probe_log.md` as a **calibration probe**, record the OOF↔public-LB gap — interpret it as a probe, not the judge.

**Through the window — verify CV↔public-LB correlation.** After the first probe, compare public LB vs OOF rank. A widening gap = sampler-artifact warning, not a reason to chase LB.

**Do not in the first 24h:** start transformers/dense; train multiclass; promote a model on public LB; submit multiple unlogged files; tune threshold on train predictions; fit any vectorizer outside folds.
