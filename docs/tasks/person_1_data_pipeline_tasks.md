# Person 1 — Data Pipeline Owner — Stage 1 Task Plan

> Project: TEKNOFEST 2026 E-Commerce **Search Relevance Engine** (Stage 1, Kaggle binary relevance: `related / not_related`, metric = **Macro-F1**).
> This file is execution-ready. Follow the module names from `03_STAGE1_ARCHITECTURE.md` exactly. Do not invent a new architecture. Do not assume final Kaggle column names. No transformer/LLM/API in the inference path. Everything must run offline and be reproducible.

---

## Role

You own everything that turns **unknown official raw files** into a **clean, leak-safe, fully-provenanced training table** that Person 2 can model and Person 3 can serve.

Concretely you own four subsystems:

1. **Schema adapter** — maps unknown Kaggle columns into one canonical internal record (`src/data/`).
2. **Turkish multi-view preprocessing** — safe, contradiction-preserving text views (`src/text/`).
3. **Fold-aware negative sampling** — easy / medium / hard-lexical negatives with full provenance and false-negative protection (`src/sampling/`).
4. **Data correctness** — fixtures, unit tests, leakage guards, and the data-contract documentation.

You are the **single source of truth** for two interfaces that the other two people import directly:

- `src/text/turkish_normalize.py` + `src/text/composers.py` (used by P2 for features and by P3 for serving).
- `src/data/field_registry.py::CANONICAL_FIELDS` (the canonical column contract everyone codes against).

If these change after they are frozen, you must announce it — three people break otherwise.

---

## Mission (what must exist before the 26/06 data release)

A **schema-agnostic data pipeline skeleton that runs end-to-end on dummy data** and is ready to absorb the real schema within hours:

- The schema adapter maps **two different dummy schemas** to identical canonical records.
- Turkish normalization passes its regression tests (brand tokens, İ/ı, units, contradiction pairs).
- The negative sampler is **deterministic**, **fold-aware**, emits **full provenance**, and **never produces a self-negative or a known positive**.
- Easy + medium miners work fully on dummy data; the hard-lexical miner is wired to Person 2's BM25 candidate API behind a stable interface (stub until real data).
- Every module has tests; `pytest -q tests/data tests/text tests/sampling` is green.
- `docs/data_pipeline.md` and `docs/schema_contract.md` document the canonical contract and the data layers.

You do **not** train models, you do **not** tune anything, and you do **not** assume label encoding (`var/yok` vs `0/1` vs `related/not_related`) — that is filled on data day.

---

## Files to create or edit

```text
# Canonical contract + schema adapter
src/data/__init__.py
src/data/field_registry.py
src/data/loaders.py
src/data/schema.py
configs/schema.yaml                      # you own this config

# Turkish preprocessing (multi-view)
src/text/__init__.py
src/text/turkish_normalize.py            # FROZEN INTERFACE (P2 + P3 import)
src/text/tokenization.py
src/text/units.py
src/text/dictionaries.py
src/text/attributes.py
src/text/text_views.py
src/text/composers.py                    # FROZEN INTERFACE (P3 imports for serving)
configs/preprocessing.yaml               # you own this config

# Negative sampling
src/sampling/__init__.py
src/sampling/base.py
src/sampling/family.py
src/sampling/easy.py
src/sampling/medium.py
src/sampling/hard_lexical.py
src/sampling/filters.py
src/sampling/orchestrator.py
configs/sampling.yaml                    # you own this config (provisional_v1)

# Fixtures + tests
tests/fixtures/dummy_schema_a.csv
tests/fixtures/dummy_schema_b.json
tests/fixtures/dummy_positives.csv
tests/data/test_schema.py
tests/data/test_loaders.py
tests/text/test_turkish_normalize.py
tests/text/test_attributes.py
tests/text/test_composers.py
tests/sampling/test_family.py
tests/sampling/test_sampler.py
tests/sampling/test_filters.py

# Docs
docs/data_pipeline.md
docs/schema_contract.md
```

> Package note (agree with Person 3): the import root is `src/` installed editable, so imports read `from src.text.turkish_normalize import normalize_text`. Person 3 scaffolds the empty `__init__.py` tree on Day 0; you fill the module bodies.

---

## Shared frozen interfaces you must honor

These are the exact signatures the other two people will call. Freeze them on Day 0 in `docs/interfaces.md` (Person 3 owns that doc; you provide your half).

```python
# src/data/field_registry.py
CANONICAL_FIELDS = [
    "pair_id", "query_id", "product_id",
    "query_text", "product_title", "category_path",
    "attributes_raw", "attributes_kv", "label",
]

# src/text/turkish_normalize.py
def normalize_text(s: str, *, fold_diacritics: bool = False, protect_brands: bool = True) -> str: ...
def turkish_lower(s: str) -> str: ...
def turkish_lower_safe(s: str) -> str: ...      # brand/SKU-safe lowercasing
def fold_diacritics(s: str) -> str: ...

# src/text/composers.py
def compose_views(record: dict) -> dict: ...     # single pair -> view dict (used at inference by P3)
def build_text_views(df, cfg: dict): ...         # batch -> df with *_raw/_norm/_folded columns

# src/sampling/orchestrator.py
class NegativeSampler:
    def __init__(self, cfg: dict, seed: int = 42): ...
    def generate(self, positives_with_folds, candidate_source, family_index) -> "pd.DataFrame": ...
```

`candidate_source` is Person 2's BM25 object exposing `.topk(query_norm, k) -> list[(product_id, score)]`. Until BM25 exists, use the stub in `src/sampling/hard_lexical.py`.

---

## Task list

### P1-01 — Canonical record contract & field registry
- **Purpose:** Define the one internal schema everyone codes against, so no module ever touches a raw Kaggle column name.
- **Files:** `src/data/field_registry.py`
- **Technical details:** Declare `CANONICAL_FIELDS` (above). Declare `REQUIRED_AFTER_RELEASE = ["query_text", "product_title"]` and `OPTIONAL_FIELDS` = everything else (because category/attributes/ids may be absent or folded into title). Add `dtype map` (all text fields `str`, `label` nullable int) and a `validate_canonical(df) -> list[str]` that returns a list of contract violations (missing required column, wrong dtype, null in required field).
- **Expected functions/classes/config keys:** `CANONICAL_FIELDS`, `REQUIRED_AFTER_RELEASE`, `OPTIONAL_FIELDS`, `CANONICAL_DTYPES`, `validate_canonical(df)`, `resolve_alias(candidate_names, available_columns)`.
- **Input/output contract:** in = list of available raw columns; out = canonical name → source name (or `None`). `validate_canonical` in = DataFrame; out = list of human-readable problems (empty list = valid).
- **Acceptance criteria:** importable with zero deps beyond pandas; `validate_canonical` flags a missing required column and a null in `query_text`.
- **Tests:** `tests/data/test_schema.py::test_field_registry_required`, `::test_validate_canonical_detects_missing`.
- **Common mistakes:** hardcoding `train["search_term"]`; treating `category_path`/`attributes_raw` as required (they may be missing — only `query_text` + `product_title` are required).
- **Dependencies:** none. **This is the first task and blocks P2/P3.** Publish it Day 0 morning.
- **Definition of Done:** `CANONICAL_FIELDS` is referenced (not redefined) everywhere; committed; announced in team channel + `docs/interfaces.md`.

### P1-02 — Schema adapter + loaders + `schema.yaml`
- **Purpose:** Turn unknown raw files into canonical records via config-driven alias mapping with safe fallbacks.
- **Files:** `src/data/schema.py`, `src/data/loaders.py`, `configs/schema.yaml`
- **Technical details:** `SchemaAdapter(cfg)` resolves each canonical field by trying the candidate list in `schema.yaml::column_map` (first hit wins), with a logged fuzzy fallback (`rapidfuzz.process.extractOne` over remaining columns, threshold ≥ 90). Parse `attributes_raw` later (Person task P1-05) — adapter only carries it through. If a field is absent: behave per `fallback.on_missing_field` (`empty`/`null`/`error`). Synthesize `pair_id`/`query_id` via stable hash when missing (`hash_str(query_text + "||" + (product_id or product_title))`). `loaders.py` reads CSV/JSON-ish files and computes a SHA-256 of each raw file.
- **Expected functions/classes/config keys:** `class SchemaAdapter(.adapt(raw_df)->df, .infer_mapping(raw_df)->dict, .report()->str)`; `load_raw(path, cfg)->df`; `compute_file_hash(path)->str`. Config keys: `files`, `read.csv_sep`, `read.encoding`, `column_map.{query_text,product_title,category_path,attributes_raw,product_id,query_id,pair_id,label}`, `attributes.format`, `category.separator`, `label.encoding`, `fallback.{make_pair_id,make_query_id,on_missing_field}`.
- **Input/output contract:** in = raw DataFrame + config; out = DataFrame with exactly `CANONICAL_FIELDS` columns (extra raw columns dropped but logged). Must never raise on a missing optional field; must raise a clear error only when a required field cannot be mapped.
- **Acceptance criteria:** runs on both dummy fixtures; produces identical canonical DataFrames; `.report()` lists every resolved mapping and every fallback used.
- **Tests:** `tests/data/test_schema.py` (mapping resolution, fuzzy fallback, missing-field behavior, synthesized ids), `tests/data/test_loaders.py` (hash is stable, CSV+JSON load).
- **Common mistakes:** crashing on missing `category_path`; silent fuzzy matches (always log them); normalizing text inside the adapter (forbidden — adapter does mapping only); overwriting raw files.
- **Dependencies:** P1-01 (field registry). Person 2 reads canonical output; Person 3 reads `compute_file_hash` for the raw-data manifest.
- **Definition of Done:** `python -c "from src.data.schema import SchemaAdapter"` works; both fixtures map identically; report generated.

### P1-03 — Dummy multi-schema fixtures
- **Purpose:** Prove schema-agnosticism by making the adapter pass on two deliberately different layouts before real data exists.
- **Files:** `tests/fixtures/dummy_schema_a.csv`, `tests/fixtures/dummy_schema_b.json`, `tests/fixtures/dummy_positives.csv`
- **Technical details:** Schema A: columns `search_term, product_name, category, attributes` with `attributes` as a `key:value|key:value` string, no ids, no label (positive-only, mimicking train). Schema B (JSON-lines): keys `query, title, category_path, specs` with `specs` as a JSON object, plus `pid`. Include Turkish edge cases on purpose: `iPhone 13`, `şekersiz kola`, `suni deri çanta`, `14x14 çerçeve`, `1,5 L termos`, mixed casing `KAHVE Makinesi`. `dummy_positives.csv` is ~60 rows used by sampler tests.
- **Expected functions/classes/config keys:** n/a (data files) + a `schema_dummy_a.yaml` / `schema_dummy_b.yaml` mapping pair under `tests/fixtures/`.
- **Input/output contract:** fixtures are read-only inputs to tests.
- **Acceptance criteria:** both fixtures load and map to identical canonical columns; edge-case rows survive into canonical form unmodified.
- **Tests:** consumed by `tests/data/test_schema.py` and `tests/sampling/*`.
- **Common mistakes:** making both schemas too similar (defeats the purpose); forgetting the positive-only/no-label case for the train-like fixture.
- **Dependencies:** P1-02.
- **Definition of Done:** fixtures committed; adapter tests use them; at least one schema has missing ids and one has missing label.

### P1-04 — Turkish normalization core + `preprocessing.yaml`  *(FROZEN INTERFACE)*
- **Purpose:** Provide safe, Turkish-aware, contradiction-preserving normalization used by features (P2) and serving (P3).
- **Files:** `src/text/turkish_normalize.py`, `configs/preprocessing.yaml`
- **Technical details:** Apply NFKC, collapse whitespace, Turkish-aware lowercasing (`İ→i`, `I→ı`) but **`turkish_lower_safe` exempts ASCII-I inside Latin brand/SKU tokens** so `iPhone → iphone`, never `ıphone` (failure F15). Preserve digits, units, dimensions (`14x14`, `1.5L`/`1,5 l`), and intra-token punctuation in model codes (`x-14`, `iphone-13`). **No stemming, no lemmatization, no spell correction by default.** `fold_diacritics` is a *separate optional view*, never destructive to the normalized view. Never collapse contradiction-bearing morphology: `şekerli` ≠ `şekersiz`, `deri` ≠ `suni deri` (keep the negating suffix/qualifier).
- **Expected functions/classes/config keys:** `nfkc`, `turkish_lower`, `turkish_lower_safe`, `fold_diacritics`, `normalize_text(s, *, fold_diacritics=False, protect_brands=True)`, `is_brand_like(token)`. Config keys: `lowercase: turkish_safe`, `nfkc`, `collapse_whitespace`, `protect_brand_tokens`, `brand_token_regex`, `preserve.{digits,units,dimensions,intra_token_punct}`, `stemming: false`, `lemmatization: false`, `spell_correction: false`, `diacritic_map`.
- **Input/output contract:** in = arbitrary str; out = normalized str; pure function, deterministic, never returns `None` (empty in → empty out).
- **Acceptance criteria:** the regression suite below passes 100%.
- **Tests:** `tests/text/test_turkish_normalize.py`:
  - `test_brand_token_preserved` (`iPhone`→`iphone`, not `ıphone`)
  - `test_turkish_casing` (`İSTANBUL`→`istanbul`, `IRMAK`→`ırmak`)
  - `test_units_preserved` (`14x14`, `1,5 L` survive)
  - `test_contradiction_preserved` (`şekersiz` not stemmed to `şeker`; `suni deri` ≠ `deri`)
  - `test_idempotent` (`normalize_text(normalize_text(x)) == normalize_text(x)`)
- **Common mistakes:** using Python `str.lower()` (breaks Turkish casing); diacritic-folding the main normalized view; “cleaning” away `x`/numbers; stemming away negation.
- **Dependencies:** none. **Freeze early — P2 and P3 both import it.**
- **Definition of Done:** all five tests green; signature published in `docs/interfaces.md`; P2/P3 notified it is stable.

### P1-05 — Attribute parsing & normalization
- **Purpose:** Turn the raw attributes blob into a normalized key→value dict so attribute-match features and conflict detection are possible.
- **Files:** `src/text/attributes.py`
- **Technical details:** `parse_attributes(raw, fmt)` auto-detects format (`json`, `kv_string` like `Materyal:Porselen|Renk:Mavi`, or `wide_columns` already split). Normalize keys to a small canonical set with a synonym map in `dictionaries.py` (e.g., `materyal/material → material`, `renk/color → color`, `stil/style → style`, `boyut/ölçü/size → size`). Normalize values with `normalize_text`. Keep raw too. Detect numeric/size values and route them to `units.py`.
- **Expected functions/classes/config keys:** `parse_attributes(raw, fmt="auto")->dict`, `normalize_attributes(kv)->dict`, `attribute_keys(kv)->set`, `ATTR_KEY_SYNONYMS` (in dictionaries). Reads `configs/preprocessing.yaml` and `configs/schema.yaml::attributes`.
- **Input/output contract:** in = raw attributes value (str/dict/NaN); out = `{canonical_key: normalized_value}` (empty dict if missing, never error).
- **Acceptance criteria:** parses all three formats; missing/empty attributes → `{}`; `Materyal: Porselen` and `materyal:porselen` map to the same key/value.
- **Tests:** `tests/text/test_attributes.py` (three formats, synonym mapping, missing → `{}`, numeric value extraction).
- **Common mistakes:** assuming one fixed attribute format; dropping unknown keys (keep them, just don't canonicalize); failing on NaN.
- **Dependencies:** P1-04 (normalize), P1-06 (units). Person 2 consumes `attributes_kv` for features.
- **Definition of Done:** robust to all three formats and to missing data; tests green.

### P1-06 — Tokenization, units, dictionaries
- **Purpose:** Shared low-level text utilities for overlap features and unit/number matching.
- **Files:** `src/text/tokenization.py`, `src/text/units.py`, `src/text/dictionaries.py`
- **Technical details:** `tokenize(s)` = whitespace + intra-token-aware splitter that keeps model codes and dimensions as single tokens. `units.py`: `extract_numbers(s)` (handles `1,5` and `1.5`), `extract_dimensions(s)` (`14x14`→`(14,14)`), `extract_units(s)` (l, ml, cm, mm, gb, w, kg…), `unit_type(token)`. `dictionaries.py`: `ATTR_KEY_SYNONYMS`, `COLOR_TERMS`, `MATERIAL_TERMS`, `NEGATION_MARKERS` (`sız/siz/suz/süz`, `olmadan`, `hariç`), `UNIT_ALIASES`. Keep dictionaries small and Turkish-first; this is not a full lexicon.
- **Expected functions/classes/config keys:** `tokenize`, `char_ngrams(s, n_lo, n_hi)`, `extract_numbers`, `extract_dimensions`, `extract_units`, `unit_type`, plus the dictionary constants above.
- **Input/output contract:** pure functions returning lists/sets/tuples; deterministic; empty in → empty out.
- **Acceptance criteria:** `14x14` tokenizes as one token and `extract_dimensions` returns `(14,14)`; `1,5 L` → number `1.5`, unit `l`.
- **Tests:** folded into `tests/text/test_turkish_normalize.py` (units/dimensions) + a small `test_tokenization` case.
- **Common mistakes:** splitting `14x14` into `14`,`x`,`14`; locale-confusing `1,5` vs `1.5`.
- **Dependencies:** P1-04. Person 2 imports these in numeric/overlap features.
- **Definition of Done:** utilities stable and tested; constants documented in `docs/schema_contract.md`.

### P1-07 — Text view composer  *(FROZEN INTERFACE)*
- **Purpose:** Build field-specific multi-view text (raw/normalized/folded) while **preserving field identity** (a title match ≠ a category match).
- **Files:** `src/text/text_views.py`, `src/text/composers.py`
- **Technical details:** Produce per-field views: `query_raw/query_norm/query_folded`, `title_raw/title_norm/title_folded`, `category_raw/category_norm`, `attributes_norm` (joined normalized k:v), and `all_text_raw/all_text_norm/all_text_folded`. `compose_views(record: dict)` is the single-pair version used by Person 3 at inference (same code path as training to avoid train/serve skew). `build_text_views(df, cfg)` is the batch version. Views are *added columns*; raw is never overwritten.
- **Expected functions/classes/config keys:** `compose_views(record)->dict`, `build_text_views(df, cfg)->df`, `VIEW_COLUMNS` (list). Reads `configs/preprocessing.yaml::views`.
- **Input/output contract:** in (batch) = canonical DataFrame; out = same rows + `VIEW_COLUMNS`. in (single) = dict with canonical keys; out = dict with view keys. Missing field → empty-string views (never crash).
- **Acceptance criteria:** identical normalization between `compose_views` (single) and `build_text_views` (batch) on the same row; field identity preserved (separate columns).
- **Tests:** `tests/text/test_composers.py` (single vs batch parity; missing-field safety; field-identity columns present).
- **Common mistakes:** concatenating everything into one string and losing field identity; divergent single vs batch code paths (causes train/serve skew — the #1 silent service bug).
- **Dependencies:** P1-04/05/06. **P3 imports `compose_views` for `/predict`; P2 imports `build_text_views` for features.**
- **Definition of Done:** parity test green; `VIEW_COLUMNS` published in `docs/interfaces.md`.

### P1-08 — Product-family hashing & family index
- **Purpose:** Enable false-negative protection — a mined "negative" product that belongs to the **same product family** as a known positive for that query is almost certainly a true positive.
- **Files:** `src/sampling/family.py`
- **Technical details:** `product_family_id(record)` derives a stable family key from available signal in priority order: explicit family/parent id if present → normalized brand+model tokens from title → fallback `hash(title_norm)`. Because `product_family_id` may be underivable from the real schema, **the function must degrade gracefully and flag low-confidence families**. `FamilyIndex` maps `query_norm → set(positive product_family_ids)` and `query_norm → set(positive product_ids)` for O(1) FN checks.
- **Expected functions/classes/config keys:** `product_family_id(record)->str`, `family_confidence(record)->str` (`high|low`), `class FamilyIndex(.build(positives_df), .positives_for(query_norm)->set, .families_for(query_norm)->set)`.
- **Input/output contract:** in = canonical+views row(s); out = family id string + confidence; index built only from **training-fold positives** (caller enforces fold scoping).
- **Acceptance criteria:** deterministic; same title → same family; low confidence flagged when only the title-hash fallback is used.
- **Tests:** `tests/sampling/test_family.py` (determinism, fallback path, confidence flag).
- **Common mistakes:** assuming a `product_family_id` column exists; building the index from all data instead of the training fold (leakage).
- **Dependencies:** P1-04/07. Used by P1-11 filters and P1-09 sampler.
- **Definition of Done:** family derivation works with and without explicit family ids; tested.

### P1-09 — Negative sampler framework + provenance + `sampling.yaml`  *(FROZEN INTERFACE)*
- **Purpose:** Orchestrate fold-aware negative generation with the default Stage 1 recipe and full provenance, deterministically.
- **Files:** `src/sampling/base.py`, `src/sampling/orchestrator.py`, `configs/sampling.yaml`
- **Technical details:** `base.py` defines the `Miner` protocol (`.mine(positive_row, candidate_source, n) -> list[NegativeCandidate]`) and the frozen provenance schema `PROVENANCE_COLS`. `NegativeSampler.generate(positives_with_folds, candidate_source, family_index)`: for each **training-fold** positive, request candidates per the recipe composition, attach provenance, then hand off to the FN-filter (P1-11). **Split-before-sample is mandatory**: the sampler receives positives that already carry `fold_id` (from Person 2's `assign_folds`) and never looks across folds. Deterministic via a per-row seeded RNG (`seed` + `source_positive_id`). Recipe is `provisional_v1` (not "champion").
- **Expected functions/classes/config keys:** `class NegativeSampler`, `Miner` protocol, `PROVENANCE_COLS = ["neg_type","miner","miner_rank","miner_score","source_positive_id","query_id","product_id","product_family_id","recipe_version","seed","fold_id","false_negative_filter_flags","quarantine_tag"]`. Config keys: `recipe_version: provisional_v1`, `neg_per_pos: 2.0`, `composition.{easy:0.25,medium:0.35,hard_lexical:0.40,hard_semantic:0.00}`, `seed: 42`, `miners.*`, `false_negative_filter.*`, `fold_aware: true`, `ratio_ablation_grid`.
- **Input/output contract:** in = positives DataFrame **with `fold_id`** + `candidate_source` + `family_index`; out = negatives DataFrame with canonical pair columns **plus every `PROVENANCE_COLS` field**. Row count ≈ `neg_per_pos × n_positives`, split by composition.
- **Acceptance criteria:** deterministic (same seed ⇒ identical negatives); bucket counts match recipe within rounding; every negative carries all provenance columns; every negative's `fold_id` equals its source positive's `fold_id`.
- **Tests:** `tests/sampling/test_sampler.py` (determinism, composition counts, provenance completeness, fold integrity).
- **Common mistakes:** mining globally then splitting (leakage F11); non-deterministic sampling; dropping provenance; labeling the recipe "champion" (it is provisional until ablations on real data).
- **Dependencies:** P1-08 (family), P1-10/11/12 (miners + filter). **Depends on Person 2's `assign_folds` for `fold_id` and on Person 2's BM25 `candidate_source` (stub until then).**
- **Definition of Done:** runs on `dummy_positives.csv` with a stub candidate source; all four sampler tests green; `sampling.yaml` committed as `provisional_v1`.

### P1-10 — Easy & medium miners
- **Purpose:** Provide the two low-risk negative types that need no lexical index, so the sampler is fully testable before BM25 exists.
- **Files:** `src/sampling/easy.py`, `src/sampling/medium.py`
- **Technical details:** **Easy** = random product from a *distant* category (different top-level category path), low lexical overlap. **Medium** = product from the *same parent category* but with a leaf or attribute mismatch (`require_attr_or_leaf_mismatch: true`). Both draw only from the **training-fold candidate pool** and respect the per-row seed. Each returns `NegativeCandidate(product_id, neg_type, miner, miner_rank, miner_score)`.
- **Expected functions/classes/config keys:** `class EasyMiner(Miner)`, `class MediumMiner(Miner)`. Config under `sampling.yaml::miners.{easy,medium}`.
- **Input/output contract:** in = positive row + fold candidate pool + n; out = list of `NegativeCandidate`.
- **Acceptance criteria:** easy negatives are from a different top-level category; medium negatives share a parent category but differ on leaf/attribute; deterministic.
- **Tests:** part of `tests/sampling/test_sampler.py` (per-type category relationship checks).
- **Common mistakes:** "easy" that accidentally shares the exact category (then it's medium); ignoring the fold pool.
- **Dependencies:** P1-09. Category info comes from canonical `category_path`.
- **Definition of Done:** both miners produce correctly-typed negatives on dummy data; tested.

### P1-11 — Hard-lexical miner placeholder (BM25 hook)
- **Purpose:** Define the hard-negative interface now and wire it to Person 2's BM25 candidate API later, without blocking on it.
- **Files:** `src/sampling/hard_lexical.py`
- **Technical details:** `HardLexicalMiner` calls `candidate_source.topk(query_norm, k)` and selects from a **rank band** (`rank_band: [5, 40]`) — deliberately skipping the top 0–4 candidates, which are the highest false-negative risk. Until BM25 exists, ship `StubCandidateSource` (returns shuffled in-fold products with synthetic scores) so the sampler and its tests run. Mark every hard negative for the FN-filter.
- **Expected functions/classes/config keys:** `class HardLexicalMiner(Miner)`, `class StubCandidateSource(.topk(q,k)->list)`. Config: `sampling.yaml::miners.hard_lexical.{strategy:bm25_topk,k:50,rank_band:[5,40]}`.
- **Input/output contract:** in = positive row + `candidate_source` + n; out = list of `NegativeCandidate` from the configured rank band, each flagged `needs_fn_check=True`.
- **Acceptance criteria:** never selects rank < `rank_band[0]`; works with both the stub and (later) the real BM25 object behind the identical `.topk` interface.
- **Tests:** `tests/sampling/test_sampler.py::test_hard_band` (no candidate below the band; FN flag set).
- **Common mistakes:** taking rank-0 candidates as hard negatives (likely true positives); coupling to BM25 internals instead of the `.topk` contract.
- **Dependencies:** **Person 2's BM25 `candidate_source` (real version)**; stub until then.
- **Definition of Done:** runs with the stub; swap-in of real BM25 requires zero code change beyond passing the object.

### P1-12 — False-negative filter, quarantine & sampling leakage guards
- **Purpose:** Stop mined negatives that are actually positives, and hard-assert the sampler never leaks. This is the **dominant Macro-F1 risk** (failure F3) and must be code, not a comment.
- **Files:** `src/sampling/filters.py`
- **Technical details:** `false_negative_filter(neg_df, family_index, cfg)` drops any negative whose product is a known positive for that query (exact pair) or shares a `product_family_id` with a positive for that query (`drop_same_family: true`); flags very-high lexical similarity (`max_lexical_sim_for_keep: 0.97`) for quarantine. `apply_quarantine` tags (not silently drops) the top rank band so it can be manually audited (`reports/sampling/fn_audit.md`). `assert_sampler_integrity(neg_df, positives_df)` raises on: any self-negative, any known positive used as negative, any negative whose `fold_id` ≠ source positive `fold_id`.
- **Expected functions/classes/config keys:** `false_negative_filter(neg_df, family_index, cfg)->df`, `apply_quarantine(neg_df, cfg)->df`, `assert_sampler_integrity(neg_df, positives_df)->None`. Config: `sampling.yaml::false_negative_filter.{drop_same_family,quarantine_top_rank_band,max_lexical_sim_for_keep}`.
- **Input/output contract:** in = raw negatives + family index; out = filtered negatives with `false_negative_filter_flags` + `quarantine_tag` populated; integrity asserter raises `AssertionError` with a precise message on any violation.
- **Acceptance criteria:** a constructed self-negative is removed; a same-family negative is removed; a quarantine-band negative is tagged not dropped; integrity asserter catches an injected cross-fold negative.
- **Tests:** `tests/sampling/test_filters.py` (self-negative removed, same-family removed, quarantine tagging, integrity assertion fires).
- **Common mistakes:** only filtering exact pairs (miss family-level FNs); silently dropping quarantine candidates (you lose audit signal); treating the filter as optional.
- **Dependencies:** P1-08 (family index), P1-09 (provenance columns).
- **Definition of Done:** all filter tests green; FN-audit report template exists; integrity asserter is called inside `NegativeSampler.generate`.

### P1-13 — Data unit tests & pipeline documentation
- **Purpose:** Lock the contracts so three people can build in parallel without surprises, and document the data layers for the top-20 inspection.
- **Files:** all `tests/data/*`, `tests/text/*`, `tests/sampling/*` (final pass), `docs/data_pipeline.md`, `docs/schema_contract.md`
- **Technical details:** Ensure `pytest -q tests/data tests/text tests/sampling` is green. `docs/data_pipeline.md` documents the **raw → interim → processed** contract: raw = untouched official files (`data/raw/`); interim = canonical records + text views (`data/interim/`); processed = positives + fold-aware negatives + provenance (`data/processed/`). `docs/schema_contract.md` documents `CANONICAL_FIELDS`, view columns, attribute key set, provenance columns, and the "missing-field behavior" table.
- **Expected functions/classes/config keys:** n/a (tests + docs).
- **Input/output contract:** docs describe inputs/outputs of each layer precisely enough that P2/P3 never have to read your code.
- **Acceptance criteria:** all data-side tests green; both docs complete; a new reader can map a hypothetical new schema using only `schema.yaml` + `docs/schema_contract.md`.
- **Tests:** the suite itself.
- **Common mistakes:** documenting code instead of contracts; leaving the missing-field behavior undefined.
- **Dependencies:** P1-01..P1-12.
- **Definition of Done:** green suite + two committed docs; `docs/interfaces.md` reflects your frozen signatures.

---

## Daily execution order

**Day 0 (24/06) — contracts first, unblock everyone**
1. P1-01 field registry → publish `CANONICAL_FIELDS` immediately (P2/P3 are blocked without it).
2. P1-04 Turkish normalization core → publish signatures (P2/P3 import it).
3. P1-07 composer skeleton with `compose_views` signature stubbed → publish (P3 needs it for `/predict`).
4. Stub `NegativeSampler.generate` + `PROVENANCE_COLS` (P1-09 interface only) → publish.
5. Help Person 3 freeze `docs/interfaces.md` (your half).

**Day 1 (25/06) — implement and test**
6. P1-02 schema adapter + `schema.yaml` + P1-03 fixtures → both fixtures map identically.
7. P1-05 attributes + P1-06 units/tokenization/dictionaries.
8. P1-07 full composer (single/batch parity test).
9. P1-08 family hashing/index.
10. P1-09 sampler + P1-10 easy/medium + P1-11 hard-lexical stub + P1-12 FN-filter/guards.
11. P1-13 finalize tests + docs; run the dummy end-to-end with Person 3.

**Day 2 morning (26/06, pre-data)**
12. Freeze interfaces; dry-run once more on fixtures; stand ready to wire the real schema (see First-24h).

> If the team is behind, the **minimum viable path** is: P1-01 → P1-02 → P1-04 → P1-09 + P1-12 (easy+medium+FN-filter), skipping hard-lexical until data, but never skipping the FN-filter or the integrity asserter.

---

## Pull request checklist

- [ ] No module imports a raw Kaggle column name; everything uses `CANONICAL_FIELDS`.
- [ ] `turkish_lower` is used (never Python `str.lower`); brand-exemption test passes.
- [ ] Composer single (`compose_views`) and batch (`build_text_views`) produce identical normalization on the same row.
- [ ] Sampler is deterministic (seeded) and **split-before-sample** (consumes `fold_id`, never crosses folds).
- [ ] Every negative row has all 13 provenance columns populated.
- [ ] FN-filter removes self-negatives and same-family negatives; quarantine tags (not drops) the top band.
- [ ] `assert_sampler_integrity` is invoked and catches an injected cross-fold negative in tests.
- [ ] No assumption about label encoding (`var/yok` vs `0/1`) anywhere.
- [ ] `pytest -q tests/data tests/text tests/sampling` green; configs you own are committed.
- [ ] `docs/interfaces.md` updated for any signature you touched.

---

## Perfect completion criteria (measurable)

1. Schema adapter maps **2 distinct dummy schemas** to byte-identical canonical DataFrames.
2. Turkish regression suite: **5/5** tests green, including brand-exemption and contradiction preservation.
3. Sampler: same seed → **identical** output across 3 runs; bucket counts within ±1 of recipe; **0** self-negatives; **0** cross-fold negatives; **100%** provenance coverage.
4. FN-filter: removes **100%** of injected same-family negatives in tests; quarantine band tagged not dropped.
5. `compose_views` (single) and `build_text_views` (batch) parity: **0** mismatches on the fixture set.
6. `pytest -q tests/data tests/text tests/sampling`: **0 failures**.
7. `docs/data_pipeline.md` + `docs/schema_contract.md` complete; a teammate maps a new hypothetical schema using only config + docs.

---

## First 24 hours after Kaggle release (26/06)

You own **Hours 0–24** of the data-side runbook (`05_KAGGLE_DAY_1_AND_FIRST_48H_PLAN.md`). Coordinate with Person 3 (raw freeze) and Person 2 (folds).

**H0–2 — Raw freeze & schema discovery (with P3).**
- Save official `train/test/sample_submission` to `data/raw/`, never edit them. Run `compute_file_hash` on each; hand hashes to Person 3 for `reports/data/raw_data_manifest.json`.
- Print train/test/sample columns + dtypes. Identify candidates for query/title/category/attributes/ids/label. Note train (positive-only) vs test schema differences.

**H2–6 — Wire the real schema.**
- Fill `configs/schema.yaml::column_map` with the real column names; set `attributes.format`, `category.separator`, and `label.encoding`/`label.positive_value` from what you observe.
- Run `SchemaAdapter.adapt` on real train + test; confirm `validate_canonical` returns no violations. Update `tests/data/test_schema.py` with the real mapping. Write `docs/dataset_schema.md` (official files, train cols, test cols, sample-submission cols, canonical mapping, required/optional, missing-field behavior, label format, open questions).

**H6–10 — Real-text normalization sanity (feeds P2's first features).**
- Run normalization on real titles/queries; spot-check brand slices (`iPhone`, `Samsung A14`), unit/size slices (`14x14`, `1,5 L`), Turkish casing, and contradiction terms. Fix `brand_token_regex`/`diacritic_map` if real data reveals gaps. Confirm `attributes.py` parses the real attribute format → log attribute key coverage.

**H10–14 — Build family index + candidate pools.**
- Build `FamilyIndex` from **training-fold positives only** (use Person 2's `assign_folds` output). Report `product_family_id` derivability on real data (how often only the title-hash fallback fires) → this sizes your FN risk and goes in `reports/sampling/first_sampling_report.md`.

**H14–22 — First fold-aware negatives (v0 = easy + medium first).**
- Generate v0 negatives with **easy + medium** (no hard-lexical until Person 2's BM25 index is ready), run the FN-filter and `assert_sampler_integrity`. As soon as BM25 lands, add hard-lexical via `.topk`. Emit `artifacts/sampling/first_negative_dataset.parquet` with full provenance + `reports/sampling/first_sampling_report.md`.
- **Hand the processed positives+negatives table to Person 2** so the baseline can train.

**H22–24 — Verify & document.**
- Confirm: 0 self-negatives, 0 known-positive-as-negative, 0 cross-fold negatives on real data; bucket counts match recipe. Note any schema surprise in `reports/assumptions/open_questions.md`.

**Do not in the first 24h:** add semantic hard negatives; tune the recipe (run ratio ablations later); assume `product_family_id` exists; skip the FN-filter even under time pressure.
