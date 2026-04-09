# Architecture

## 1. Goal
Thin-file 사용자 대상 오프라인 금융상품 Top-K 추천 시스템.

- 입력: `(user_id, as_of_date)` 스냅샷 기반 사용자 데이터
- 출력: 추천 상품 리스트 + 점수
- 방식: Candidate generation + Feature-based Learning-to-Rank (LightGBM)
- 설명: Grounded explanation object 기반 설명 생성 + 검증(Verifier)

## 2. High-Level Flow
```text
[Raw Data: 11, 09(lag), 12]
  -> [Snapshot Builder]
  -> [User Feature Engineering]
  -> [Product Normalization]
  -> [Candidate Generation]
  -> [User-Item Pair Features]
  -> [Baseline Scoring]
  -> [LGBMRanker (optional fit)]
  -> [Top-K Recommendation]
  -> [Reason Extractor]
  -> [Explanation Object]
  -> [Renderer (Template or OpenAI LLM)]
  -> [Verifier]
```

## 3. Data Scope & Time Alignment
- Table 11 (`11.통신카드CB 결합정보`): quarterly anchor
- Table 09 (`09.개인 CB정보`): yearly lagged feature (previous year December)
- Table 12 (`12.금융상품정보`): product catalog

Time alignment rule:
- anchor `YYYYMM` from table 11 -> join with lagged `((YYYY-1)12)` in table 09

## 4. Package Structure
```text
src/thin_filer/
  pipeline.py                  # backward-compatible facade
  pipeline_config.py           # RecommenderConfig
  pipeline_helpers.py          # constants + utility functions
  recommender.py               # ThinFilerRecommender core pipeline

  explainer.py                 # GroundedExplainer orchestrator
  explainer_common.py          # labels/schema/common helpers
  explainer_reasoning.py       # reason extraction/fact retrieval/object build
  explainer_render_verify.py   # render + verifier logic
  llm_renderer.py              # OpenAI API renderer
```

## 5. Recommender Core (`ThinFilerRecommender`)
Defined in `src/thin_filer/recommender.py`.

- Split mode support:
  - `recommender_family="deposit"`: deposit-only recommender
  - `recommender_family="fund"`: public-fund-only recommender
  - `recommender_family="all"`: unified recommender (default)

### 5.1 Snapshot & Features
1. `_load_table11()` / `_load_table09()`
2. `build_user_snapshots()`
3. `_engineer_user_features()`
   - component features (credit depth, stability, liquidity pressure, digital, sophistication)
   - preference features (`risk_tol`, `liquidity_need`, `horizon_pref`, `amount_bin`)

### 5.2 Products & Candidates
1. `_load_and_normalize_products()`
   - deposit/fund normalization to unified schema
2. `generate_candidates()`
   - risk tolerance + investability rule
   - liquidity/amount filtering
   - candidate min/max enforcement

### 5.3 Pair, Labels, Ranking
1. `_add_pair_features()`
   - match features: `risk/liquidity/horizon/complexity/amount/family/digital`
2. `_compute_baseline_score()`
3. `_build_labels()` (weak label)
4. `fit()` with `LGBMRanker`
5. `recommend()` / `batch_recommend()` / `evaluate()`

## 6. Explainer Architecture (`GroundedExplainer`)
Defined in `src/thin_filer/explainer.py` (orchestrator).

### 6.1 Reasoning Layer
`src/thin_filer/explainer_reasoning.py`
- `extract_reasons()`
  - SHAP (if available) -> fallback to feature importance/value heuristic
- `retrieve_product_facts()`
- `build_explanation_object()`

### 6.2 Rendering Layer
`src/thin_filer/explainer_render_verify.py`
- `render_explanation()` (deterministic template)

`src/thin_filer/llm_renderer.py`
- `OpenAILLMRenderer`
- Input: explanation object only
- Output: constrained formatted explanation text

### 6.3 Verification Layer
`src/thin_filer/explainer_render_verify.py`
- `reason_alignment()`
- `check_fact_consistency()`
- `hallucination_rate()`
- `contains_forbidden_claims()`
- `verify()`

## 7. LLM Integration Policy
When using `OpenAILLMRenderer`:
- LLM is used only for verbalization
- Source of truth is explanation object
- No external facts allowed
- If verifier fails and fallback is enabled:
  - automatically replace with deterministic template output
  - mark render source as `template_fallback`

## 8. Public API Compatibility
`src/thin_filer/pipeline.py` keeps compatibility exports:
- `RecommenderConfig`
- `ThinFilerRecommender`
- `to_json`
- helper symbols (`_ndcg_at_k`, etc.) used by existing scripts

This allows old imports like:
```python
from thin_filer.pipeline import RecommenderConfig, ThinFilerRecommender
```

## 9. Script Entry Points
```text
scripts/recommender/
  run_recommender.py
  explain_recommender.py
  improve_recommender_with_utility.py

scripts/evaluation/
  evaluate.py
  evaluate_explainer.py

scripts/analysis/
  ...

scripts/visualization/
  ...
```

## 10. Runtime Notes
- `PYTHONPATH=src` required when running scripts directly
- Optional dependencies:
  - `lightgbm`, `shap`
  - `openai` (LLM renderer only)
- If OpenAI API key is missing, LLM renderer mode cannot be used

## 11. Recommended Execution
### 11.1 Ranker
```bash
PYTHONPATH=src python3 scripts/recommender/run_recommender.py --fit --sample-users 200
```

Deposit-only:
```bash
PYTHONPATH=src python3 scripts/recommender/run_recommender.py --fit --sample-users 200 --family deposit
```

Fund-only:
```bash
PYTHONPATH=src python3 scripts/recommender/run_recommender.py --fit --sample-users 200 --family fund
```

### 11.2 Grounded Explainer (template)
```bash
PYTHONPATH=src python3 scripts/recommender/explain_recommender.py --fit --sample-users 200 --top-k 5
```

### 11.3 Grounded Explainer (OpenAI LLM)
```bash
export OPENAI_API_KEY=your_api_key
PYTHONPATH=src python3 scripts/recommender/explain_recommender.py \
  --fit --sample-users 200 --top-k 5 \
  --use-llm-renderer --llm-model gpt-5-mini
```
