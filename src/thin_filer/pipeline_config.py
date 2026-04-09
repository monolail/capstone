from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class RecommenderConfig:
    data_root: Path = Path("data")
    table11_dir: str = "11.통신카드CB 결합정보"
    table09_dir: str = "09.개인 CB정보"
    table12_dir: str = "12.금융상품정보"

    user_key_11: str = "CUST_ID"
    user_key_09: str = "ID"
    # "all" | "deposit" | "fund"
    recommender_family: str = "all"

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
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "random_state": 42,
            "verbose": -1,
        }
    )
