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
    p = argparse.ArgumentParser(description="Visualize applied utility (deposit/fund)")
    p.add_argument("--deposit-csv", type=Path, default=Path("data/processed/product12_deposit_utility_index.csv"))
    p.add_argument("--fund-csv", type=Path, default=Path("data/processed/product12_fund_utility_index.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/product12/utility_figures"))
    return p.parse_args()


def _add_value_labels_h(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    for p in ax.patches:
        w = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(w + 0.005, y, fmt.format(w), va="center", fontsize=9)


def _utility_distribution_plot(df: pd.DataFrame, utility_col: str, title: str, out_path: Path, color: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(df[utility_col], bins=35, color=color, ax=axes[0])
    q10, q50, q90 = df[utility_col].quantile([0.1, 0.5, 0.9]).tolist()
    for v, c, name in [(q10, "#7f8c8d", "Q10"), (q50, "#2c3e50", "Q50"), (q90, "#7f8c8d", "Q90")]:
        axes[0].axvline(v, color=c, linestyle="--", linewidth=1)
        axes[0].text(v, axes[0].get_ylim()[1] * 0.92, name, rotation=90, va="top", ha="right", fontsize=8)
    axes[0].set_title(f"{title} 분포")
    axes[0].set_xlabel(utility_col)
    axes[0].set_ylabel("count")

    dec = pd.qcut(df[utility_col], q=10, labels=False, duplicates="drop") + 1
    tmp = pd.DataFrame({"decile": dec, utility_col: df[utility_col]})
    sns.boxplot(data=tmp, x="decile", y=utility_col, color="#dfe6e9", ax=axes[1])
    axes[1].set_title(f"{title} Decile Box")
    axes[1].set_xlabel("decile (낮음→높음)")
    axes[1].set_ylabel(utility_col)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _cluster_scatter(df: pd.DataFrame, utility_col: str, out_path: Path, title: str, label_offset: float) -> None:
    csum = (
        df.groupby("cluster_key", as_index=False)
        .agg(product_count=("product_id", "count"), mean_utility=(utility_col, "mean"), utility_std=(utility_col, "std"))
        .fillna(0)
        .sort_values("product_count", ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=csum,
        x="product_count",
        y="mean_utility",
        size="product_count",
        hue="utility_std",
        palette="viridis",
        sizes=(60, 900),
        ax=ax,
    )
    top5 = csum.nlargest(5, "product_count")
    for _, r in top5.iterrows():
        ax.text(r["product_count"] + label_offset, r["mean_utility"], r["cluster_key"], fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("cluster product_count")
    ax.set_ylabel("mean utility")
    ax.text(
        0.02,
        0.98,
        "색상=클러스터 내 utility 변동성(std)\n라벨=규모 상위 5개만 표시",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8f9fa", "edgecolor": "#bdc3c7", "alpha": 0.95},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_deposit(deposit: pd.DataFrame, out_dir: Path) -> None:
    comp_cols = ["U_rate", "U_bonus", "U_feasibility", "U_liquidity"]
    comp_mean = deposit[comp_cols].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(x=comp_mean.values, y=comp_mean.index, color="#2E86AB", ax=ax)
    _add_value_labels_h(ax)
    ax.set_title("수신상품 Utility 컴포넌트 평균")
    ax.set_xlabel("mean score")
    ax.set_ylabel("component")
    fig.tight_layout()
    fig.savefig(out_dir / "deposit_01_component_mean.png", dpi=180)
    plt.close(fig)

    _utility_distribution_plot(
        deposit,
        "deposit_utility",
        "수신상품 Utility",
        out_dir / "deposit_02_utility_distribution_decile.png",
        "#2E86AB",
    )

    _cluster_scatter(
        deposit,
        "deposit_utility",
        out_dir / "deposit_03_cluster_bubble_improved.png",
        "수신상품 클러스터 규모-품질 맵 (Top20)",
        label_offset=2,
    )

    top = deposit.head(25).copy()
    top = top[["product_name", "U_rate", "U_bonus", "U_feasibility", "U_liquidity", "deposit_utility"]].set_index("product_name")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(top, cmap="YlGnBu", linewidths=0.2, ax=ax)
    ax.set_title("수신상품 Top25 Utility 프로파일")
    ax.set_xlabel("component")
    ax.set_ylabel("product_name")
    fig.tight_layout()
    fig.savefig(out_dir / "deposit_04_top25_profile_heatmap.png", dpi=180)
    plt.close(fig)


def plot_fund(fund: pd.DataFrame, out_dir: Path) -> None:
    comp_cols = ["U_return", "U_risk_eff", "U_cost_eff", "U_liquidity", "U_simplicity"]
    comp_mean = fund[comp_cols].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(x=comp_mean.values, y=comp_mean.index, color="#00897B", ax=ax)
    _add_value_labels_h(ax)
    ax.set_title("펀드 Utility 컴포넌트 평균")
    ax.set_xlabel("mean score")
    ax.set_ylabel("component")
    fig.tight_layout()
    fig.savefig(out_dir / "fund_01_component_mean.png", dpi=180)
    plt.close(fig)

    _utility_distribution_plot(
        fund,
        "fund_utility",
        "펀드 Utility",
        out_dir / "fund_02_utility_distribution_decile.png",
        "#00897B",
    )

    _cluster_scatter(
        fund,
        "fund_utility",
        out_dir / "fund_03_cluster_bubble_improved.png",
        "펀드 클러스터 규모-품질 맵 (Top20)",
        label_offset=8,
    )

    top = fund.head(25).copy()
    top = top[["product_name", "U_return", "U_risk_eff", "U_cost_eff", "U_liquidity", "U_simplicity", "fund_utility"]].set_index("product_name")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(top, cmap="YlOrBr", linewidths=0.2, ax=ax)
    ax.set_title("펀드 Top25 Utility 프로파일")
    ax.set_xlabel("component")
    ax.set_ylabel("product_name")
    fig.tight_layout()
    fig.savefig(out_dir / "fund_04_top25_profile_heatmap.png", dpi=180)
    plt.close(fig)


def plot_compare(deposit: pd.DataFrame, fund: pd.DataFrame, out_dir: Path) -> None:
    d = deposit[["deposit_utility"]].rename(columns={"deposit_utility": "utility"}).copy()
    d["family"] = "deposit"
    f = fund[["fund_utility"]].rename(columns={"fund_utility": "utility"}).copy()
    f["family"] = "fund"
    all_df = pd.concat([d, f], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.violinplot(
        data=all_df,
        x="family",
        y="utility",
        hue="family",
        inner="quartile",
        palette=["#2E86AB", "#00897B"],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("상품군별 Utility 분포")

    qdf = all_df.groupby("family", as_index=False)["utility"].quantile([0.1, 0.5, 0.9]).reset_index()
    qdf = qdf.rename(columns={"level_1": "q"})
    qmap = {0.1: "Q10", 0.5: "Q50", 0.9: "Q90"}
    qdf["q"] = qdf["q"].map(qmap)
    sns.pointplot(data=qdf, x="q", y="utility", hue="family", dodge=0.25, ax=axes[1])
    axes[1].set_title("분위수 비교 (Q10/Q50/Q90)")

    fig.tight_layout()
    fig.savefig(out_dir / "compare_01_family_violin_quantile.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    deposit = pd.read_csv(args.deposit_csv).sort_values("deposit_utility", ascending=False)
    fund = pd.read_csv(args.fund_csv).sort_values("fund_utility", ascending=False)

    plot_deposit(deposit, args.out_dir)
    plot_fund(fund, args.out_dir)
    plot_compare(deposit, fund, args.out_dir)

    lines = [
        "# Applied Utility Figures (Improved)",
        "",
        f"- 폰트: {font_name}",
        "- 개선 포인트: 분포+분위수, 클러스터 품질변동성, 가독성 라벨링",
        "",
        "## Deposit",
        "- deposit_01_component_mean.png",
        "- deposit_02_utility_distribution_decile.png",
        "- deposit_03_cluster_bubble_improved.png",
        "- deposit_04_top25_profile_heatmap.png",
        "",
        "## Fund",
        "- fund_01_component_mean.png",
        "- fund_02_utility_distribution_decile.png",
        "- fund_03_cluster_bubble_improved.png",
        "- fund_04_top25_profile_heatmap.png",
        "",
        "## Compare",
        "- compare_01_family_violin_quantile.png",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"saved improved figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
