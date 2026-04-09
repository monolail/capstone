# 아키텍처 개요

## 1. 목표
Thin-file 사용자를 위한 오프라인 금융상품 Top-K 추천 시스템입니다.

- 입력: `(user_id, as_of_date)` 스냅샷 기반 사용자 데이터
- 출력: 추천 상품 리스트 + 점수
- 방식: 후보군 생성 + Feature 기반 Learning-to-Rank (LightGBM)
- 설명: Grounded explanation object 기반 설명 생성 + Verifier 검증

## 2. 전체 흐름
```text
[원천 데이터: 11, 09(lag), 12]
  -> [스냅샷 빌더]
  -> [사용자 피처 엔지니어링]
  -> [상품 정규화]
  -> [후보군 생성]
  -> [사용자-상품 Pair 피처 생성]
  -> [Baseline 점수 계산]
  -> [LGBMRanker 학습(옵션)]
  -> [Top-K 추천]
  -> [Reason 추출]
  -> [Explanation Object 생성]
  -> [Renderer (Template 또는 OpenAI LLM)]
  -> [Verifier]
```

## 3. 데이터 범위 및 시점 정렬
- Table 11 (`11.통신카드CB 결합정보`): 분기 기준 anchor
- Table 09 (`09.개인 CB정보`): 전년도 12월 lag feature
- Table 12 (`12.금융상품정보`): 상품 카탈로그

시점 정렬 규칙:
- table 11의 `YYYYMM` anchor -> table 09의 `((YYYY-1)12)`와 조인

## 4. 패키지 구조
```text
src/thin_filer/
  pipeline.py                  # 하위호환 facade
  pipeline_config.py           # RecommenderConfig
  pipeline_helpers.py          # 상수/유틸 함수
  recommender.py               # ThinFilerRecommender 핵심 구현

  explainer.py                 # GroundedExplainer 오케스트레이터
  explainer_common.py          # 공통 스키마/라벨/문장 유틸
  explainer_reasoning.py       # reason 추출/fact 조회/object 생성
  explainer_render_verify.py   # 렌더링 + 검증
  llm_renderer.py              # OpenAI API 렌더러
```

## 5. 추천기 코어 (`ThinFilerRecommender`)
정의 위치: `src/thin_filer/recommender.py`

- 분리 모드 지원:
  - `recommender_family="deposit"`: 수신상품 전용 추천기
  - `recommender_family="fund"`: 공모펀드 전용 추천기
  - `recommender_family="all"`: 통합 추천기 (기본값)

### 5.1 스냅샷/사용자 피처
1. `_load_table11()` / `_load_table09()`
2. `build_user_snapshots()`
3. `_engineer_user_features()`
   - component features (신용깊이, 안정성, 유동성 압박, 디지털, 금융복잡도)
   - preference features (`risk_tol`, `liquidity_need`, `horizon_pref`, `amount_bin`)

### 5.2 상품/후보군
1. `_load_and_normalize_products()`
   - 수신/펀드 스키마 통합 정규화
2. `generate_candidates()`
   - 위험성향 + 투자가능 여부 규칙
   - 유동성/가입금액 필터
   - candidate min/max 보장

### 5.3 Pair/라벨/랭킹
1. `_add_pair_features()`
   - match features: `risk/liquidity/horizon/complexity/amount/family/digital`
2. `_compute_baseline_score()`
3. `_build_labels()` (weak label)
4. `fit()` (`LGBMRanker`)
5. `recommend()` / `batch_recommend()` / `evaluate()`

## 6. 설명기 구조 (`GroundedExplainer`)
정의 위치: `src/thin_filer/explainer.py` (오케스트레이션 담당)

### 6.1 Reasoning 계층
`src/thin_filer/explainer_reasoning.py`
- `extract_reasons()`
  - SHAP 사용 가능 시 우선, 불가 시 feature importance/value 기반 fallback
- `retrieve_product_facts()`
- `build_explanation_object()`

### 6.2 Rendering 계층
`src/thin_filer/explainer_render_verify.py`
- `render_explanation()` (결정론적 템플릿 렌더링)

`src/thin_filer/llm_renderer.py`
- `OpenAILLMRenderer`
- 입력: explanation object only
- 출력: 제약된 포맷의 설명 텍스트

### 6.3 Verification 계층
`src/thin_filer/explainer_render_verify.py`
- `reason_alignment()`
- `check_fact_consistency()`
- `hallucination_rate()`
- `contains_forbidden_claims()`
- `verify()`

## 7. LLM 통합 정책
`OpenAILLMRenderer` 사용 시 정책:
- LLM은 “문장화(Verbalization)”에만 사용
- source of truth는 explanation object
- 외부 사실/추정 추가 금지
- verifier 실패 + fallback 활성화 시:
  - 템플릿 렌더링으로 자동 대체
  - `render_source = template_fallback`으로 기록

## 8. 공개 API/하위호환
`src/thin_filer/pipeline.py`는 하위호환 export 유지:
- `RecommenderConfig`
- `ThinFilerRecommender`
- `to_json`
- 기존 스크립트에서 쓰던 helper (`_ndcg_at_k` 등)

기존 import 예시는 계속 동작합니다:
```python
from thin_filer.pipeline import RecommenderConfig, ThinFilerRecommender
```

## 9. 스크립트 엔트리 포인트
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

## 10. 런타임 메모
- 스크립트 직접 실행 시 `PYTHONPATH=src` 필요
- 선택 의존성:
  - `lightgbm`, `shap`
  - `openai` (LLM renderer 사용 시)
- `OPENAI_API_KEY`가 없으면 LLM renderer 실행 불가

## 11. 권장 실행 예시
### 11.1 추천기 학습/추천
```bash
PYTHONPATH=src python3 scripts/recommender/run_recommender.py --fit --sample-users 200
```

수신상품 전용:
```bash
PYTHONPATH=src python3 scripts/recommender/run_recommender.py --fit --sample-users 200 --family deposit
```

공모펀드 전용:
```bash
PYTHONPATH=src python3 scripts/recommender/run_recommender.py --fit --sample-users 200 --family fund
```

### 11.2 설명기 (템플릿)
```bash
PYTHONPATH=src python3 scripts/recommender/explain_recommender.py --fit --sample-users 200 --top-k 5
```

### 11.3 설명기 (OpenAI LLM)
```bash
export OPENAI_API_KEY=your_api_key
PYTHONPATH=src python3 scripts/recommender/explain_recommender.py \
  --fit --sample-users 200 --top-k 5 \
  --use-llm-renderer --llm-model gpt-5-mini
```
