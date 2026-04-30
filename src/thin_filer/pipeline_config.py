from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class RecommenderConfig:
    data_root: Path = Path("data")
    # 원본 합성데이터 ZIP 경로 (추가)
    raw_data_root: Path = Path(r"C:\Users\이호준\OneDrive\바탕 화면\3학년 1학기\데이터캡스톤 디자인\117.금융 합성데이터\3.개방데이터\1.데이터\1. 합성데이터")
    
    table11_dir: str = "table11"
    table09_dir: str = "table09"
    table12_dir: str = "table12"

    user_key_11: str = "CUST_ID"
    user_key_09: str = "ID"
    # "all" | "deposit" | "fund"
    recommender_family: str = "all"

    # TPS v2.0 가중치 설정 (사용자 정의)
    tps_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trust": 0.7,
            "activity": 0.15,
            "potential": 0.15
        }
    )
    
    # Trust Score 내부 가중치 (최적화 결과 반영)
    trust_overdue_weight: float = 10.0
    trust_inst_weight: float = 1.0

    # Activity Score 내부 가중치 (최적화 결과 반영)
    activity_amt_weight: float = 0.2
    activity_cnt_weight: float = 0.4
    activity_digi_weight: float = 0.4

    # Potential Score 내부 가중치 (최적화 결과 반영)
    potential_income_weight: float = 0.3
    potential_cb_weight: float = 0.4
    potential_tel_weight: float = 0.1
    potential_youth_weight: float = 0.2

    candidate_min: int = 50
    candidate_max: int = 100
    top_k: int = 5
    max_train_users: int = 5000
    random_state: int = 42

    risk_threshold: float = 1.25
    investment_asset_threshold: float = 2_000_000.0

    baseline_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "risk_match": 0.35,
            "liquidity_match": 0.25,
            "horizon_match": 0.20,
            "complexity_match": 0.10,
            "digital_match": 0.10,
        }
    )

    ranker_params: Dict[str, object] = field(
        default_factory=lambda: {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5],
            "lambdarank_truncation_level": 5,
            "label_gain": [0, 1, 3, 15],
            "n_estimators": 500,
            "learning_rate": 0.02,
            "num_leaves": 128,
            "random_state": 42,
            "verbose": -1,
        }
    )
