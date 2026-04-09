#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from thin_filer.pipeline_config import RecommenderConfig
from thin_filer.recommender import ThinFilerRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate thin-file recommender (baseline vs ranker)")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--sample-users", type=int, default=1000)
    p.add_argument("--max-train-users", type=int, default=800)
    p.add_argument("--max-eval-users", type=int, default=300)
    p.add_argument("--fit", action="store_true", help="Fit LightGBMRanker before evaluation")
    p.add_argument("--family", choices=["all", "deposit", "fund"], default="all")
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    p.add_argument("--as-of-dates", nargs="*", default=None)
    return p.parse_args()


def _split_users(snapshots, user_col: str, train_ratio: float = 0.8):
    users = snapshots[user_col].drop_duplicates().sample(frac=1.0, random_state=42)
    cutoff = max(1, int(len(users) * train_ratio))
    train_users = set(users.iloc[:cutoff])
    train_df = snapshots[snapshots[user_col].isin(train_users)].copy()
    eval_df = snapshots[~snapshots[user_col].isin(train_users)].copy()
    if eval_df.empty:
        eval_df = train_df.copy()
    return train_df, eval_df


def main() -> None:
    args = parse_args()
    cfg = RecommenderConfig(data_root=args.data_root, recommender_family=args.family)
    rec = ThinFilerRecommender(cfg)

    snapshots = rec.build_user_snapshots(
        as_of_dates=args.as_of_dates,
        sample_users=args.sample_users,
    )
    rec.load_products()

    train_df, eval_df = _split_users(snapshots, user_col=cfg.user_key_11)

    if args.fit:
        rec.fit(snapshots=train_df, max_users=args.max_train_users)

    report = {
        "snapshot_quality": rec.snapshot_quality_report(snapshots),
        "split": {
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
            "train_users": int(train_df[cfg.user_key_11].nunique()),
            "eval_users": int(eval_df[cfg.user_key_11].nunique()),
        },
        "evaluation": rec.evaluate(eval_df, ks=args.ks, max_users=args.max_eval_users),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
