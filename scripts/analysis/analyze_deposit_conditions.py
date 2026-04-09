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
    p = argparse.ArgumentParser(description="Deposit condition analysis with corrected whitelist")
    p.add_argument("--raw-deposit-csv", type=Path, default=Path("data/12.금융상품정보/은행수신상품.csv"))
    p.add_argument("--cluster-csv", type=Path, default=Path("data/processed/product12_deposit_clusters.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/product12/deposit_conditions"))
    p.add_argument("--out-profile-csv", type=Path, default=Path("data/processed/product12_deposit_cluster_condition_profile.csv"))
    return p.parse_args()


def condition_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("우대금리조건_") and c.endswith("_여부")]
    # whitelist: remove opaque "기타" fields
    return [c for c in cols if "기타" not in c]


def clean_binary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].replace({"Y": 1, "N": 0, "y": 1, "n": 0, True: 1, False: 0})
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
    out = (out > 0).astype(int)
    return out


def short_name(col: str) -> str:
    return col.replace("우대금리조건_", "").replace("_여부", "")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_profile_csv.parent.mkdir(parents=True, exist_ok=True)

    font_name = set_korean_font()

    raw = pd.read_csv(args.raw_deposit_csv, low_memory=False)
    clusters = pd.read_csv(args.cluster_csv)

    cond = condition_cols(raw)
    X = clean_binary(raw, cond)
    X.columns = [short_name(c) for c in X.columns]

    # product-level aggregation to avoid raw-row duplication bias
    product_cond = (
        pd.concat(
            [
                raw[["상품코드", "상품명"]].rename(columns={"상품코드": "product_id", "상품명": "product_name"}),
                X,
            ],
            axis=1,
        )
        .groupby(["product_id", "product_name"], as_index=False)
        .max()
    )

    cluster_map = clusters[["product_id", "cluster_id"]].drop_duplicates("product_id")
    data = product_cond.merge(cluster_map, on="product_id", how="left")
    data["cluster_id"] = data["cluster_id"].fillna("DCL_00")
    X = data[[c for c in data.columns if c not in {"product_id", "product_name", "cluster_id"}]]

    # 1) activation frequency
    freq = X.sum().sort_values(ascending=False)
    freq_rate = (freq / len(X)).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=freq.values, y=freq.index, ax=ax, color="#2E86AB")
    ax.set_title("우대금리 조건 활성화 빈도 (화이트리스트)")
    ax.set_xlabel("활성 상품 수")
    ax.set_ylabel("조건")
    fig.tight_layout()
    fig.savefig(args.out_dir / "01_condition_activation_whitelist.png", dpi=160)
    plt.close(fig)

    # 2) number of active conditions per product
    cnt = X.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(cnt, bins=range(0, int(cnt.max()) + 2), discrete=True, ax=ax, color="#74c476")
    ax.set_title("상품당 활성 조건 수 분포 (화이트리스트)")
    ax.set_xlabel("활성 조건 수")
    ax.set_ylabel("상품 수")
    fig.tight_layout()
    fig.savefig(args.out_dir / "02_condition_count_hist_whitelist.png", dpi=160)
    plt.close(fig)

    # 3) co-occurrence heatmap for top conditions
    top = freq.head(15).index.tolist()
    Xt = X[top]
    co_np = Xt.T.to_numpy().dot(Xt.to_numpy())
    np.fill_diagonal(co_np, 0)
    co = pd.DataFrame(co_np, index=top, columns=top)
    # Jaccard-like normalize for readability
    denom = np.maximum.outer(Xt.sum(axis=0).values, np.ones(len(top)))
    co_rate = co / denom

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(co_rate, cmap="YlOrRd", ax=ax)
    ax.set_title("조건 동시출현 강도 히트맵 (Top15, 정규화)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "03_condition_cooccurrence_heatmap.png", dpi=160)
    plt.close(fig)

    # 4) cluster x condition profile (activation rate)
    prof = data.groupby("cluster_id", as_index=False)[top].mean()
    prof_long = prof.set_index("cluster_id")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(prof_long, cmap="Blues", ax=ax)
    ax.set_title("클러스터별 조건 활성화율 프로파일 (Top15)")
    ax.set_xlabel("조건")
    ax.set_ylabel("cluster_id")
    fig.tight_layout()
    fig.savefig(args.out_dir / "04_cluster_condition_profile_heatmap.png", dpi=160)
    plt.close(fig)

    # save profile csv (all whitelist conditions)
    full_prof = data.groupby("cluster_id", as_index=False)[list(X.columns)].mean()
    full_prof.to_csv(args.out_profile_csv, index=False, encoding="utf-8-sig")

    # report
    report = []
    report.append("# 수신상품 우대조건 보정 분석 보고서")
    report.append("")
    report.append("- 분석 기준: 화이트리스트 조건(기타1~5 제외)")
    report.append(f"- 적용 폰트: {font_name}")
    report.append(f"- 총 상품 수: {len(data)}")
    report.append(f"- 조건 수(화이트리스트): {len(X.columns)}")
    report.append("")
    report.append("## 핵심 인사이트")
    report.append(f"- 상품당 활성 조건 수: 평균 {cnt.mean():.2f}, 중앙값 {cnt.median():.0f}, 최대 {cnt.max():.0f}")
    report.append(f"- 조건 0개 상품 비중: {(cnt.eq(0).mean()*100):.1f}%")
    report.append(f"- 조건 10개 이상 상품 비중: {(cnt.ge(10).mean()*100):.1f}%")
    report.append("")
    report.append("## 상위 조건 (활성 비율)")
    for k, v in freq_rate.head(12).items():
        report.append(f"- {k}: {v*100:.1f}%")
    report.append("")
    report.append("## 산출물")
    report.append(f"- 클러스터 조건 프로파일 CSV: `{args.out_profile_csv}`")
    report.append(f"- Figure: `{args.out_dir / '01_condition_activation_whitelist.png'}`")
    report.append(f"- Figure: `{args.out_dir / '02_condition_count_hist_whitelist.png'}`")
    report.append(f"- Figure: `{args.out_dir / '03_condition_cooccurrence_heatmap.png'}`")
    report.append(f"- Figure: `{args.out_dir / '04_cluster_condition_profile_heatmap.png'}`")

    (args.out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"saved report: {args.out_dir / 'report.md'}")
    print(f"saved profile csv: {args.out_profile_csv}")
    print(f"saved figures dir: {args.out_dir}")


if __name__ == "__main__":
    main()
