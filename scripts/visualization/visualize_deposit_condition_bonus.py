#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_korean_font() -> str:
    candidates = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize bonus-rate by deposit preferential conditions")
    p.add_argument("--raw-deposit-csv", type=Path, default=Path("data/12.금융상품정보/은행수신상품.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/product12/deposit_conditions"))
    p.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("data/processed/product12_deposit_condition_bonus_summary.csv"),
    )
    return p.parse_args()


def condition_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("우대금리조건_") and c.endswith("_여부")]
    return [c for c in cols if "기타" not in c]


def clean_binary(s: pd.Series) -> pd.Series:
    out = s.replace({"Y": 1, "N": 0, "y": 1, "n": 0, True: 1, False: 0})
    out = pd.to_numeric(out, errors="coerce").fillna(0)
    return (out > 0).astype(int)


def short_name(col: str) -> str:
    return col.replace("우대금리조건_", "").replace("_여부", "")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    font_name = set_korean_font()

    raw = pd.read_csv(args.raw_deposit_csv, low_memory=False)
    yn_cols = condition_cols(raw)

    rows = []
    dist_rows = []

    for yn in yn_cols:
        rate_col = yn.replace("_여부", "_우대금리")
        if rate_col not in raw.columns:
            continue

        active = clean_binary(raw[yn])
        rate = pd.to_numeric(raw[rate_col], errors="coerce")

        # product-level dedup by 상품코드
        tmp = pd.DataFrame(
            {
                "product_id": raw["상품코드"].astype(str),
                "active": active,
                "rate": rate,
            }
        )
        prod = tmp.groupby("product_id", as_index=False).agg(active=("active", "max"), rate=("rate", "max"))
        act = prod[prod["active"] == 1].copy()
        if act.empty:
            continue

        rate_nonnull = act["rate"].dropna()
        positive = (rate_nonnull > 0).sum()
        positive_ratio = float(positive / len(act)) if len(act) else 0.0

        rows.append(
            {
                "condition": short_name(yn),
                "active_product_count": int(len(act)),
                "mean_bonus_rate": float(rate_nonnull.mean()) if len(rate_nonnull) else 0.0,
                "median_bonus_rate": float(rate_nonnull.median()) if len(rate_nonnull) else 0.0,
                "p75_bonus_rate": float(rate_nonnull.quantile(0.75)) if len(rate_nonnull) else 0.0,
                "p90_bonus_rate": float(rate_nonnull.quantile(0.90)) if len(rate_nonnull) else 0.0,
                "max_bonus_rate": float(rate_nonnull.max()) if len(rate_nonnull) else 0.0,
                "positive_bonus_ratio": positive_ratio,
            }
        )

        for v in rate_nonnull.values:
            dist_rows.append({"condition": short_name(yn), "bonus_rate": float(v)})

    summary = pd.DataFrame(rows).sort_values(["mean_bonus_rate", "active_product_count"], ascending=[False, False])
    summary.to_csv(args.out_summary_csv, index=False, encoding="utf-8-sig")

    # Figure 1: 평균 우대금리 (상위 20)
    top = summary.head(20)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(data=top, y="condition", x="mean_bonus_rate", ax=ax, color="#2E86AB")
    ax.set_title("조건별 평균 제공 우대금리 (상위 20)")
    ax.set_xlabel("평균 우대금리")
    ax.set_ylabel("조건")
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["mean_bonus_rate"], i, f" n={int(r['active_product_count'])}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "05_condition_mean_bonus_top20.png", dpi=160)
    plt.close(fig)

    # Figure 2: 활성 상품수 vs 평균 우대금리 (버블=양수제공비율)
    fig, ax = plt.subplots(figsize=(10, 6))
    s = (summary["positive_bonus_ratio"].clip(0, 1) * 1500 + 80)
    sc = ax.scatter(
        summary["active_product_count"],
        summary["mean_bonus_rate"],
        s=s,
        c=summary["median_bonus_rate"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4,
    )
    for _, r in summary.head(15).iterrows():
        ax.text(r["active_product_count"], r["mean_bonus_rate"], r["condition"], fontsize=8)
    ax.set_title("조건별 제공금리 프로파일 (버블=양수 제공비율)")
    ax.set_xlabel("조건 활성 상품 수")
    ax.set_ylabel("평균 우대금리")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("중앙값 우대금리")
    fig.tight_layout()
    fig.savefig(args.out_dir / "06_condition_bonus_profile_bubble.png", dpi=160)
    plt.close(fig)

    # Figure 3: 분포 비교 (상위 12 조건 박스플롯)
    dist = pd.DataFrame(dist_rows)
    top12 = summary.head(12)["condition"].tolist()
    d12 = dist[dist["condition"].isin(top12)].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=d12, x="condition", y="bonus_rate", order=top12, ax=ax, showfliers=False, color="#fdae6b")
    ax.set_title("상위 조건 우대금리 분포 비교 (Boxplot)")
    ax.set_xlabel("조건")
    ax.set_ylabel("우대금리")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(args.out_dir / "07_condition_bonus_distribution_boxplot.png", dpi=160)
    plt.close(fig)

    # update markdown index
    md = args.out_dir / "README_bonus.md"
    md.write_text(
        "\n".join(
            [
                "# 우대조건별 제공금리 시각화",
                "",
                f"- 적용 폰트: {font_name}",
                f"- 요약 CSV: `{args.out_summary_csv}`",
                "",
                "- [05_condition_mean_bonus_top20.png](05_condition_mean_bonus_top20.png)",
                "- [06_condition_bonus_profile_bubble.png](06_condition_bonus_profile_bubble.png)",
                "- [07_condition_bonus_distribution_boxplot.png](07_condition_bonus_distribution_boxplot.png)",
            ]
        ),
        encoding="utf-8",
    )

    print(f"font: {font_name}")
    print(f"saved summary: {args.out_summary_csv}")
    print(f"saved figures: {args.out_dir}")


if __name__ == "__main__":
    main()
