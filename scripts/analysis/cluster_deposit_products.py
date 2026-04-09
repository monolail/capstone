#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    p = argparse.ArgumentParser(description="Cluster deposit products by group code + account type + maturity")
    p.add_argument("--raw-deposit-csv", type=Path, default=Path("data/12.금융상품정보/은행수신상품.csv"))
    p.add_argument("--out-cluster-csv", type=Path, default=Path("data/processed/product12_deposit_clusters.csv"))
    p.add_argument("--out-report", type=Path, default=Path("reports/product12/deposit_cluster_report.md"))
    p.add_argument("--out-fig-dir", type=Path, default=Path("reports/product12/figures_deposit_cluster"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    font_name = set_korean_font()

    df = pd.read_csv(args.raw_deposit_csv, low_memory=False)
    work = pd.DataFrame(
        {
            "product_id": df["상품코드"].astype(str),
            "product_name": df["상품명"].astype(str),
            "product_group_code": df.get("상품그룹코드", "").astype(str),
            "account_type": df.get("예금입출금방식", "").astype(str),
            "maturity_type": df.get("만기여부", "").astype(str),
            "base_rate": pd.to_numeric(df.get("기본금리", 0), errors="coerce").fillna(0.0),
            "max_pref_rate": pd.to_numeric(df.get("최대우대금리", 0), errors="coerce").fillna(0.0),
        }
    )

    work["cluster_key"] = (
        work["product_group_code"] + "|" + work["account_type"] + "|" + work["maturity_type"]
    )

    cluster_counts = work["cluster_key"].value_counts().reset_index()
    cluster_counts.columns = ["cluster_key", "count"]
    cluster_counts["share"] = cluster_counts["count"] / cluster_counts["count"].sum()

    cluster_counts = cluster_counts.sort_values("count", ascending=False).reset_index(drop=True)
    cluster_counts["cluster_id"] = [f"DCL_{i+1:02d}" for i in range(len(cluster_counts))]

    key_to_id = dict(zip(cluster_counts["cluster_key"], cluster_counts["cluster_id"]))
    work["cluster_id"] = work["cluster_key"].map(key_to_id)

    summary = (
        work.groupby(["cluster_id", "cluster_key"], as_index=False)
        .agg(
            product_count=("product_id", "count"),
            mean_base_rate=("base_rate", "mean"),
            mean_max_pref_rate=("max_pref_rate", "mean"),
        )
        .merge(cluster_counts[["cluster_id", "share"]], on="cluster_id", how="left")
        .sort_values("product_count", ascending=False)
    )

    out_map = work[
        [
            "product_id",
            "product_name",
            "cluster_id",
            "cluster_key",
            "product_group_code",
            "account_type",
            "maturity_type",
        ]
    ].copy()

    args.out_cluster_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_fig_dir.mkdir(parents=True, exist_ok=True)

    out_map.to_csv(args.out_cluster_csv, index=False, encoding="utf-8-sig")

    # Figure 1: cluster distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=summary, x="cluster_id", y="product_count", color="#2E86AB", ax=ax)
    ax.set_title("수신상품 클러스터 분포")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("상품 수")
    for i, v in enumerate(summary["product_count"].tolist()):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)

    top5 = summary.head(5).copy()
    explain_lines = ["[DCL 해설]", "DCL_01~05 정의:"]
    for _, r in top5.iterrows():
        explain_lines.append(f"{r['cluster_id']} = {r['cluster_key']}")
    explain_text = "\n".join(explain_lines)
    ax.text(
        1.02,
        0.98,
        explain_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F4F6F7", "edgecolor": "#95A5A6", "alpha": 0.95},
    )

    fig.subplots_adjust(right=0.72)
    fig.tight_layout()
    fig.savefig(args.out_fig_dir / "01_cluster_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 2: account_type x maturity
    ct = pd.crosstab(work["account_type"], work["maturity_type"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("입출금방식 x 만기여부 교차분포")
    ax.set_xlabel("만기여부")
    ax.set_ylabel("예금입출금방식")
    fig.tight_layout()
    fig.savefig(args.out_fig_dir / "02_account_maturity_heatmap.png", dpi=150)
    plt.close(fig)

    # report
    lines = []
    lines.append("# 수신상품 클러스터링 보고서")
    lines.append("")
    lines.append("- 기준: `상품그룹코드 + 예금입출금방식 + 만기여부`")
    lines.append(f"- 적용 폰트: {font_name}")
    lines.append(f"- 총 상품 수: {len(work)}")
    lines.append(f"- 총 클러스터 수: {summary['cluster_id'].nunique()}")
    lines.append("")
    lines.append("## 클러스터 요약")
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['cluster_id']}: {r['cluster_key']} | count={int(r['product_count'])}, share={r['share']:.3f}, mean_base_rate={r['mean_base_rate']:.3f}, mean_max_pref_rate={r['mean_max_pref_rate']:.3f}"
        )
    lines.append("")
    lines.append("## 산출물")
    lines.append(f"- 클러스터 매핑 CSV: `{args.out_cluster_csv}`")
    lines.append(f"- Figure: `{args.out_fig_dir / '01_cluster_distribution.png'}`")
    lines.append(f"- Figure: `{args.out_fig_dir / '02_account_maturity_heatmap.png'}`")
    lines.append("")
    lines.append("## 라벨 다양성 검증 연결")
    lines.append("- 추천/학습 샘플에서 cluster_id 분포를 측정하면 특정 상품군 편향 여부를 수치로 검증할 수 있음")
    lines.append("- 예: 사용자별 top-k 내 고유 cluster_id 수, 전체 추천의 cluster entropy")

    args.out_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved cluster csv: {args.out_cluster_csv}")
    print(f"saved cluster report: {args.out_report}")
    print(f"saved figures dir: {args.out_fig_dir}")


if __name__ == "__main__":
    main()
