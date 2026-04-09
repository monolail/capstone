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
    p = argparse.ArgumentParser(description="High-information visualization for product12 split catalogs")
    p.add_argument("--deposit-csv", type=Path, default=Path("data/processed/product12_deposit_catalog.csv"))
    p.add_argument("--fund-csv", type=Path, default=Path("data/processed/product12_fund_catalog.csv"))
    p.add_argument("--deposit-cluster-csv", type=Path, default=Path("data/processed/product12_deposit_clusters.csv"))
    p.add_argument("--out-deposit-dir", type=Path, default=Path("reports/product12/figures_deposit_rich"))
    p.add_argument("--out-fund-dir", type=Path, default=Path("reports/product12/figures_fund_rich"))
    return p.parse_args()


def save_heatmap_counts_and_pct(ct: pd.DataFrame, title: str, out_path: Path) -> None:
    row_pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    annot = ct.astype(int).astype(str) + "\n(" + (row_pct * 100).round(1).astype(str) + "%)"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(row_pct, annot=annot, fmt="", cmap="YlGnBu", cbar_kws={"label": "row %"}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def visualize_deposit(dep: pd.DataFrame, cluster_map: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    d = dep.merge(cluster_map[["product_id", "cluster_id", "cluster_key"]], on="product_id", how="left")
    d["cluster_id"] = d["cluster_id"].fillna("DCL_00")

    # 1) 입출금방식 x 만기여부 : count + row%
    ct = pd.crosstab(d["raw_group"], d["maturity_type"])
    f = "01_입출금방식x만기_행비율히트맵.png"
    save_heatmap_counts_and_pct(ct, "수신상품: 입출금방식 x 만기여부 (건수 + 행비율)", out_dir / f)
    files.append(f)

    # 2) 클러스터 프로파일 버블: 평균 기본금리 vs 평균 최대우대금리, 버블크기=상품수
    g = (
        d.groupby(["cluster_id", "cluster_key"], as_index=False)
        .agg(
            product_count=("product_id", "count"),
            mean_base_rate=("base_rate", "mean"),
            mean_max_rate=("max_rate", "mean"),
            avg_min_amount_bin=("min_amount_bin", "mean"),
        )
        .sort_values("product_count", ascending=False)
    )

    f = "02_클러스터프로파일_버블차트.png"
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(
        g["mean_base_rate"],
        g["mean_max_rate"],
        s=(g["product_count"] / g["product_count"].max()) * 1800 + 120,
        c=g["avg_min_amount_bin"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, r in g.iterrows():
        ax.text(r["mean_base_rate"], r["mean_max_rate"], r["cluster_id"], fontsize=9, ha="center", va="center")
    ax.set_title("수신상품 클러스터 프로파일 (버블=상품수, 색=평균 가입금액구간)")
    ax.set_xlabel("평균 기본금리")
    ax.set_ylabel("평균 최대우대금리")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("평균 min_amount_bin")
    fig.tight_layout()
    fig.savefig(out_dir / f, dpi=160)
    plt.close(fig)
    files.append(f)

    # 3) 클러스터별 기본금리 분포 (boxplot)
    f = "03_클러스터별_기본금리_박스플롯.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    order = g["cluster_id"].tolist()
    sns.boxplot(data=d, x="cluster_id", y="base_rate", order=order, ax=ax, color="#9ecae1", showfliers=False)
    ax.set_title("수신상품 클러스터별 기본금리 분포")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("base_rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / f, dpi=160)
    plt.close(fig)
    files.append(f)

    # 4) 클러스터별 우대금리 분포 (violin)
    f = "04_클러스터별_최대우대금리_바이올린.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.violinplot(data=d, x="cluster_id", y="max_rate", order=order, ax=ax, inner="quartile", cut=0, color="#74c476")
    ax.set_title("수신상품 클러스터별 최대우대금리 분포")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("max_rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / f, dpi=160)
    plt.close(fig)
    files.append(f)

    return files


def visualize_fund(fund: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    # 1) 위험수준 x 기간(horizon) 히트맵 (건수 + 행비율)
    ct = pd.crosstab(fund["risk_level"], fund["horizon"])
    f = "01_위험수준x기간_행비율히트맵.png"
    save_heatmap_counts_and_pct(ct, "공모펀드: 위험수준 x 기간 (건수 + 행비율)", out_dir / f)
    files.append(f)

    # 2) 위험수준별 성과 proxy 분포 (boxplot)
    f = "02_위험수준별_수익률proxy_박스플롯.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=fund, x="risk_level", y="max_rate", ax=ax, color="#fdae6b", showfliers=False)
    ax.set_title("공모펀드 위험수준별 1년성과(proxy=max_rate) 분포")
    ax.set_xlabel("risk_level")
    ax.set_ylabel("max_rate")
    fig.tight_layout()
    fig.savefig(out_dir / f, dpi=160)
    plt.close(fig)
    files.append(f)

    # 3) 수수료구간 x 위험수준 히트맵 (건수)
    ct2 = pd.crosstab(fund["fee_level"], fund["risk_level"])
    f = "03_수수료구간x위험수준_히트맵.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(ct2, annot=True, fmt="d", cmap="Oranges", ax=ax)
    ax.set_title("공모펀드 수수료구간 x 위험수준 교차분포")
    ax.set_xlabel("risk_level")
    ax.set_ylabel("fee_level")
    fig.tight_layout()
    fig.savefig(out_dir / f, dpi=160)
    plt.close(fig)
    files.append(f)

    # 4) 복잡도별 위험/성과 프로파일 버블
    g = (
        fund.groupby("complexity", as_index=False)
        .agg(
            product_count=("product_id", "count"),
            mean_risk=("risk_level", "mean"),
            mean_return=("max_rate", "mean"),
            mean_fee=("fee_level", "mean"),
        )
        .sort_values("product_count", ascending=False)
    )
    f = "04_복잡도프로파일_버블차트.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        g["mean_risk"],
        g["mean_return"],
        s=(g["product_count"] / g["product_count"].max()) * 2000 + 150,
        c=g["mean_fee"],
        cmap="plasma",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, r in g.iterrows():
        ax.text(r["mean_risk"], r["mean_return"], f"C{int(r['complexity'])}", ha="center", va="center", fontsize=10)
    ax.set_title("공모펀드 복잡도 프로파일 (버블=상품수, 색=평균 수수료구간)")
    ax.set_xlabel("평균 risk_level")
    ax.set_ylabel("평균 max_rate")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("평균 fee_level")
    fig.tight_layout()
    fig.savefig(out_dir / f, dpi=160)
    plt.close(fig)
    files.append(f)

    return files


def write_index(path: Path, title: str, font_name: str, files: list[str]) -> None:
    lines = [f"# {title}", "", f"- 적용 폰트: {font_name}", ""]
    lines.extend([f"- [{f}]({f})" for f in files])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    font_name = set_korean_font()

    dep = pd.read_csv(args.deposit_csv)
    fund = pd.read_csv(args.fund_csv)
    c_map = pd.read_csv(args.deposit_cluster_csv)

    dep_files = visualize_deposit(dep, c_map, args.out_deposit_dir)
    fund_files = visualize_fund(fund, args.out_fund_dir)

    write_index(args.out_deposit_dir / "README.md", "수신상품 고정보량 시각화", font_name, dep_files)
    write_index(args.out_fund_dir / "README.md", "공모펀드 고정보량 시각화", font_name, fund_files)

    print(f"font: {font_name}")
    print(f"saved deposit rich figures: {args.out_deposit_dir}")
    print(f"saved fund rich figures: {args.out_fund_dir}")


if __name__ == "__main__":
    main()
