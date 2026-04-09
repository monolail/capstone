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
    p = argparse.ArgumentParser(description="Interpretive analytics for deposit utility")
    p.add_argument("--utility-csv", type=Path, default=Path("data/processed/product12_deposit_utility_index.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/product12/deposit_utility_interpretive"))
    return p.parse_args()


def _dedup_products(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "product_id", "product_name", "cluster_key", "base_rate", "max_bonus_rate", "condition_count",
        "U_rate", "U_bonus", "U_feasibility", "U_liquidity", "deposit_utility",
    ]
    use = [c for c in cols if c in df.columns]
    d = df[use].copy()
    d = d.sort_values(["product_id", "deposit_utility"], ascending=[True, False])
    d = d.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)
    return d


def _cluster_alias(cluster_key: str) -> str:
    s = str(cluster_key)
    if "자유입출식" in s and "만기 없음" in s:
        return "입출금-수시형"
    if "적립식" in s and "만기 있음" in s:
        return "적립-만기형"
    if "거치식" in s and "만기 있음" in s:
        return "거치-만기형"
    if "적립식" in s and "만기 없음" in s:
        return "적립-유동형"
    return "기타"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    raw = pd.read_csv(args.utility_csv)
    df = _dedup_products(raw)
    df["cluster_alias"] = df["cluster_key"].map(_cluster_alias)

    # 1) Utility distribution + quantiles
    q10, q25, q50, q75, q90 = df["deposit_utility"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["deposit_utility"], bins=35, color="#2E86AB", ax=ax)
    for v, label, c in [(q25, "Q25", "#95a5a6"), (q50, "Q50", "#2c3e50"), (q75, "Q75", "#95a5a6")]:
        ax.axvline(v, color=c, linestyle="--", linewidth=1)
        ax.text(v, ax.get_ylim()[1] * 0.92, label, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title("수신상품 Utility 분포 (중앙 사분위 강조)")
    ax.set_xlabel("deposit_utility")
    ax.set_ylabel("count")
    ax.text(
        0.98,
        0.98,
        f"n={len(df):,}\nQ10={q10:.3f}\nQ50={q50:.3f}\nQ90={q90:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8f9fa", "edgecolor": "#bdc3c7", "alpha": 0.95},
    )
    fig.tight_layout(); fig.savefig(args.out_dir / "01_utility_distribution_iqr.png", dpi=180); plt.close(fig)

    # 2) Decile component profile (interpretability)
    dec = pd.qcut(df["deposit_utility"], q=10, labels=False, duplicates="drop") + 1
    ddec = df.assign(decile=dec)
    comp_cols = ["U_rate", "U_bonus", "U_feasibility", "U_liquidity"]
    prof = ddec.groupby("decile", as_index=False)[comp_cols + ["deposit_utility"]].mean()

    fig, ax = plt.subplots(figsize=(11, 6))
    for c, color in [("U_rate", "#1f77b4"), ("U_bonus", "#17becf"), ("U_feasibility", "#2ca02c"), ("U_liquidity", "#ff7f0e")]:
        ax.plot(prof["decile"], prof[c], marker="o", label=c, color=color)
    ax.plot(prof["decile"], prof["deposit_utility"], marker="s", linewidth=2.5, label="deposit_utility", color="#2c3e50")
    ax.set_title("Decile별 Utility 구성요소 프로파일")
    ax.set_xlabel("utility decile (낮음→높음)")
    ax.set_ylabel("mean score")
    ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(args.out_dir / "02_decile_component_profile.png", dpi=180); plt.close(fig)

    # 3) Practical trade-off map: rate vs feasibility
    # Remove upper outliers using p99 for clearer central structure.
    p99_rate = df["U_rate"].quantile(0.99)
    p99_feas = df["U_feasibility"].quantile(0.99)
    p99_bonus = df["max_bonus_rate"].quantile(0.99)
    trade = df[
        (df["U_rate"] <= p99_rate)
        & (df["U_feasibility"] <= p99_feas)
        & (df["max_bonus_rate"] <= p99_bonus)
    ].copy()

    # U_feasibility is discretized by construction, so add tiny jitter for readability.
    rng = np.random.default_rng(42)
    jittered = trade.copy()
    jittered["U_feasibility_j"] = (jittered["U_feasibility"] + rng.normal(0, 0.006, size=len(jittered))).clip(0, 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=jittered,
        x="U_rate",
        y="U_feasibility_j",
        hue="deposit_utility",
        size="max_bonus_rate",
        palette="viridis",
        alpha=0.58,
        sizes=(20, 180),
        linewidth=0.2,
        edgecolor="white",
        ax=ax,
    )
    ax.axvline(trade["U_rate"].median(), linestyle="--", color="#7f8c8d", linewidth=1)
    ax.axhline(trade["U_feasibility"].median(), linestyle="--", color="#7f8c8d", linewidth=1)
    ax.set_title("기본금리 정규화(U_rate) vs 조건용이성(U_feasibility)")
    ax.set_xlabel("U_rate")
    ax.set_ylabel("U_feasibility (jittered)")
    ax.text(
        0.02,
        0.02,
        f"p99 filter 적용: n={len(trade):,}/{len(df):,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(args.out_dir / "03_tradeoff_rate_vs_feasibility.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4) Cluster leaderboard with alias
    csum = (
        df.groupby(["cluster_key", "cluster_alias"], as_index=False)
        .agg(
            product_count=("product_id", "count"),
            mean_utility=("deposit_utility", "mean"),
            p75_utility=("deposit_utility", lambda x: x.quantile(0.75)),
            mean_u_rate=("U_rate", "mean"),
            mean_u_feas=("U_feasibility", "mean"),
        )
        .sort_values(["product_count", "mean_utility"], ascending=[False, False])
    )
    ctop = csum.head(12).copy()
    ctop["label"] = ctop["cluster_alias"] + " | " + ctop["cluster_key"]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=ctop, y="label", x="mean_utility", color="#5DADE2", ax=ax)
    for i, r in ctop.reset_index(drop=True).iterrows():
        ax.text(r["mean_utility"] + 0.003, i, f"n={int(r['product_count'])}, p75={r['p75_utility']:.3f}", va="center", fontsize=8)
    ax.set_title("클러스터 Utility 리더보드 (규모 상위 12)")
    ax.set_xlabel("mean_utility")
    ax.set_ylabel("cluster")
    fig.tight_layout(); fig.savefig(args.out_dir / "04_cluster_leaderboard.png", dpi=180); plt.close(fig)

    # 5) Top/Bottom products explanation table-like chart
    top8 = df.nlargest(8, "deposit_utility")[ ["product_name","deposit_utility","U_rate","U_bonus","U_feasibility","U_liquidity"] ]
    bot8 = df.nsmallest(8, "deposit_utility")[ ["product_name","deposit_utility","U_rate","U_bonus","U_feasibility","U_liquidity"] ]
    tb = pd.concat([top8.assign(group="Top8"), bot8.assign(group="Bottom8")], ignore_index=True)
    tb_plot = tb.copy()
    tb_plot["row"] = tb_plot["group"] + " | " + tb_plot["product_name"].str.slice(0, 24)
    tb_plot = tb_plot.set_index("row")[["deposit_utility","U_rate","U_bonus","U_feasibility","U_liquidity"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(tb_plot, cmap="RdYlGn", linewidths=0.2, ax=ax)
    ax.set_title("상/하위 상품 비교 프로파일 (Top8 vs Bottom8)")
    ax.set_xlabel("metric")
    ax.set_ylabel("products")
    fig.tight_layout(); fig.savefig(args.out_dir / "05_top_bottom_profile_heatmap.png", dpi=180); plt.close(fig)

    # Outputs
    csum.to_csv(args.out_dir / "cluster_diagnostics.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# 수신상품 Utility 해석 리포트 (강화 버전)")
    lines.append("")
    lines.append(f"- 폰트: {font_name}")
    lines.append(f"- 분석 상품 수(중복 product_id 제거): {len(df):,}")
    lines.append(f"- utility 평균: {df['deposit_utility'].mean():.4f}")
    lines.append(f"- utility 중앙값: {q50:.4f}")
    lines.append(f"- utility 상위10%(Q90): {q90:.4f}")
    lines.append("")
    lines.append("## 핵심 해석")
    lines.append("- Decile 프로파일에서 상위 구간은 U_rate와 U_bonus가 올라가도, U_feasibility가 크게 무너지면 utility 상승이 제한됩니다.")
    lines.append("- Trade-off 맵에서 우상단(기본금리 정규화+조건용이성 동시 우수) 상품군이 실무 추천 우선군입니다.")
    lines.append("- 클러스터 리더보드로 '규모는 크지만 평균 utility가 낮은 군집'을 분리해 라벨 품질 점검 대상으로 지정할 수 있습니다.")
    lines.append("")
    lines.append("## 산출물")
    lines.append(f"- 01_utility_distribution_iqr.png")
    lines.append(f"- 02_decile_component_profile.png")
    lines.append(f"- 03_tradeoff_rate_vs_feasibility.png")
    lines.append(f"- 04_cluster_leaderboard.png")
    lines.append(f"- 05_top_bottom_profile_heatmap.png")
    lines.append(f"- cluster_diagnostics.csv")

    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved interpretive report: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
