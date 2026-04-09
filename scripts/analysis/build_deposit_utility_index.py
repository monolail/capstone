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
    p = argparse.ArgumentParser(description="Build deposit product utility index (item-level)")
    p.add_argument("--raw-deposit-csv", type=Path, default=Path("data/12.금융상품정보/은행수신상품.csv"))
    p.add_argument("--out-csv", type=Path, default=Path("data/processed/product12_deposit_utility_index.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/product12/deposit_utility"))
    return p.parse_args()


def normalize01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def clean_binary(s: pd.Series) -> pd.Series:
    out = s.replace({"Y": 1, "N": 0, "y": 1, "n": 0, True: 1, False: 0})
    out = pd.to_numeric(out, errors="coerce").fillna(0)
    return (out > 0).astype(int)


def cond_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("우대금리조건_") and c.endswith("_여부")]
    return [c for c in cols if "기타" not in c]


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    raw = pd.read_csv(args.raw_deposit_csv, low_memory=False)

    yn_cols = cond_cols(raw)
    cond_count = pd.Series(0, index=raw.index, dtype=float)
    for c in yn_cols:
        cond_count = cond_count + clean_binary(raw[c]).astype(float)

    df = pd.DataFrame(
        {
            "product_id": raw.get("상품코드", raw.index.astype(str)).astype(str),
            "product_name": raw.get("상품명", "").astype(str),
            "group_code": raw.get("상품그룹코드", "").astype(str),
            "group_name": raw.get("상품그룹명", "").astype(str),
            "deposit_mode": raw.get("예금입출금방식", "").astype(str),
            "maturity_type": raw.get("만기여부", "").astype(str),
            "base_rate": pd.to_numeric(raw.get("기본금리", 0), errors="coerce").fillna(0.0),
            "max_bonus_rate": pd.to_numeric(raw.get("최대우대금리", 0), errors="coerce").fillna(0.0),
            "condition_count": cond_count,
        }
    )

    # outlier-safe bonus scaling
    bonus_cap = float(df["max_bonus_rate"].quantile(0.99))
    bonus_capped = df["max_bonus_rate"].clip(upper=bonus_cap)

    liquidity_score = np.where(df["maturity_type"].str.contains("만기 없음", na=False), 1.0, 0.55)

    # Item-level utility (deposit product only)
    # U_rate: basic return attractiveness
    # U_bonus: potential upside from preferential bonus
    # U_feasibility: penalty for too many conditions
    # U_liquidity: convenience from maturity type
    u_rate = normalize01(df["base_rate"])
    u_bonus = normalize01(bonus_capped)
    u_feasibility = 1.0 - normalize01(df["condition_count"])  # fewer conditions -> higher utility
    u_liquidity = pd.Series(liquidity_score, index=df.index)

    utility = 0.45 * u_rate + 0.25 * u_bonus + 0.20 * u_feasibility + 0.10 * u_liquidity

    cluster_key = (
        "G" + df["group_code"].replace("", "NA")
        + "|" + df["deposit_mode"].replace("", "NA")
        + "|" + df["maturity_type"].replace("", "NA")
    )

    out = df.copy()
    out["U_rate"] = u_rate
    out["U_bonus"] = u_bonus
    out["U_feasibility"] = u_feasibility
    out["U_liquidity"] = u_liquidity
    out["deposit_utility"] = utility
    out["cluster_key"] = cluster_key

    out = out.sort_values("deposit_utility", ascending=False).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # Figures
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(out["deposit_utility"], bins=35, color="#2E86AB", ax=ax)
    ax.set_title("수신상품 Utility 분포 (상품 관점)")
    ax.set_xlabel("deposit_utility")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.out_dir / "01_utility_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=out, x="base_rate", y="max_bonus_rate", hue="deposit_utility", palette="viridis", ax=ax, alpha=0.8)
    ax.set_title("기본금리 vs 최대우대금리 (색=Utility)")
    ax.set_xlabel("base_rate")
    ax.set_ylabel("max_bonus_rate")
    fig.tight_layout()
    fig.savefig(args.out_dir / "02_base_vs_bonus_utility_scatter.png", dpi=160)
    plt.close(fig)

    top_clusters = out["cluster_key"].value_counts().head(12).index
    cdf = out[out["cluster_key"].isin(top_clusters)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=cdf, x="cluster_key", y="deposit_utility", ax=ax, color="#74c476")
    ax.set_title("상위 클러스터별 Utility 분포")
    ax.set_xlabel("cluster_key")
    ax.set_ylabel("deposit_utility")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(args.out_dir / "03_cluster_utility_boxplot.png", dpi=160)
    plt.close(fig)

    top10 = out.head(10)[["product_id", "product_name", "base_rate", "max_bonus_rate", "condition_count", "deposit_utility", "cluster_key"]]

    cluster_summary = (
        out.groupby("cluster_key", as_index=False)
        .agg(
            product_count=("product_id", "count"),
            mean_utility=("deposit_utility", "mean"),
            mean_base_rate=("base_rate", "mean"),
            mean_bonus=("max_bonus_rate", "mean"),
            mean_condition_count=("condition_count", "mean"),
        )
        .sort_values(["product_count", "mean_utility"], ascending=[False, False])
    )
    cluster_summary.to_csv(args.out_dir / "cluster_utility_summary.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 수신상품 Utility 인덱스 보고서 (상품 관점)")
    lines.append("")
    lines.append(f"- 적용 폰트: {font_name}")
    lines.append(f"- 총 상품 수: {len(out)}")
    lines.append(f"- 평균 utility: {out['deposit_utility'].mean():.4f}")
    lines.append(f"- 중앙 utility: {out['deposit_utility'].median():.4f}")
    lines.append("")
    lines.append("## 정의")
    lines.append("- deposit_utility = 0.45*U_rate + 0.25*U_bonus + 0.20*U_feasibility + 0.10*U_liquidity")
    lines.append("- U_rate: 기본금리 정규화")
    lines.append("- U_bonus: 최대우대금리(상위 1% cap) 정규화")
    lines.append("- U_feasibility: (1 - 조건개수 정규화)")
    lines.append("- U_liquidity: 만기 없음=1.0, 만기 있음=0.55")
    lines.append("")
    lines.append("## Top 10 상품")
    for _, r in top10.iterrows():
        lines.append(
            f"- {r['product_id']} | {r['product_name']} | utility={r['deposit_utility']:.4f} | base={r['base_rate']:.3f} | bonus={r['max_bonus_rate']:.3f} | cond={int(r['condition_count'])}"
        )
    lines.append("")
    lines.append("## 산출물")
    lines.append(f"- csv: `{args.out_csv}`")
    lines.append(f"- csv: `{args.out_dir / 'cluster_utility_summary.csv'}`")
    lines.append(f"- figure: `{args.out_dir / '01_utility_distribution.png'}`")
    lines.append(f"- figure: `{args.out_dir / '02_base_vs_bonus_utility_scatter.png'}`")
    lines.append(f"- figure: `{args.out_dir / '03_cluster_utility_boxplot.png'}`")

    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {args.out_csv}")
    print(f"saved: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
