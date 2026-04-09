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
    p = argparse.ArgumentParser(description="Build fund utility index (item-level)")
    p.add_argument("--fund-csv", type=Path, default=Path("data/processed/product12_fund_catalog.csv"))
    p.add_argument("--out-csv", type=Path, default=Path("data/processed/product12_fund_utility_index.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("reports/product12/fund_utility"))
    return p.parse_args()


def normalize01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def robust_return_norm(ret: pd.Series) -> pd.Series:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    lo = float(r.quantile(0.01))
    hi = float(r.quantile(0.99))
    r = r.clip(lower=lo, upper=hi)
    return normalize01(r)


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_name = set_korean_font()

    fund = pd.read_csv(args.fund_csv)
    fund = fund[fund["product_family"].astype(str).eq("fund")].copy()

    for c in ["risk_level", "liquidity_level", "complexity", "fee_level", "base_rate", "max_rate"]:
        if c in fund.columns:
            fund[c] = pd.to_numeric(fund[c], errors="coerce").fillna(0.0)

    # Fund-only utility components
    # U_return: recent performance attractiveness (robust clipped)
    # U_cost_eff: low fee preference
    # U_risk_eff: return adjusted by risk level
    # U_liquidity: redemption convenience proxy
    # U_simplicity: complexity penalty
    u_return = robust_return_norm(fund["base_rate"])
    u_cost_eff = 1.0 - normalize01(fund["fee_level"])

    risk_adj = fund["base_rate"] / (1.0 + fund["risk_level"])
    u_risk_eff = robust_return_norm(risk_adj)

    u_liquidity = normalize01(fund["liquidity_level"])
    u_simplicity = 1.0 - normalize01(fund["complexity"])

    fund_utility = 0.35 * u_return + 0.25 * u_risk_eff + 0.20 * u_cost_eff + 0.10 * u_liquidity + 0.10 * u_simplicity

    out = fund.copy()
    out["U_return"] = u_return
    out["U_risk_eff"] = u_risk_eff
    out["U_cost_eff"] = u_cost_eff
    out["U_liquidity"] = u_liquidity
    out["U_simplicity"] = u_simplicity
    out["fund_utility"] = fund_utility
    out["cluster_key"] = (
        out["raw_group"].astype(str).fillna("NA")
        + "|risk" + out["risk_level"].astype(int).astype(str)
        + "|fee" + out["fee_level"].astype(int).astype(str)
    )

    out = out.sort_values("fund_utility", ascending=False).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    cluster_summary = (
        out.groupby("cluster_key", as_index=False)
        .agg(
            product_count=("product_id", "count"),
            mean_utility=("fund_utility", "mean"),
            mean_return=("base_rate", "mean"),
            mean_risk=("risk_level", "mean"),
            mean_fee=("fee_level", "mean"),
        )
        .sort_values(["product_count", "mean_utility"], ascending=[False, False])
    )
    cluster_summary.to_csv(args.out_dir / "cluster_utility_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(out["fund_utility"], bins=40, color="#00897B", ax=ax)
    ax.set_title("펀드 Utility 분포 (상품 관점)")
    ax.set_xlabel("fund_utility")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.out_dir / "01_utility_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=out, x="base_rate", y="risk_level", hue="fund_utility", palette="viridis", alpha=0.7, ax=ax)
    ax.set_title("수익률 vs 위험등급 (색=Utility)")
    ax.set_xlabel("base_rate(1Y proxy)")
    ax.set_ylabel("risk_level")
    fig.tight_layout()
    fig.savefig(args.out_dir / "02_return_vs_risk_scatter.png", dpi=160)
    plt.close(fig)

    top_clusters = out["cluster_key"].value_counts().head(12).index
    cdf = out[out["cluster_key"].isin(top_clusters)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=cdf, x="cluster_key", y="fund_utility", color="#4DB6AC", ax=ax)
    ax.set_title("상위 펀드 클러스터별 Utility 분포")
    ax.set_xlabel("cluster_key")
    ax.set_ylabel("fund_utility")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(args.out_dir / "03_cluster_utility_boxplot.png", dpi=160)
    plt.close(fig)

    top10 = out.head(10)[["product_id", "product_name", "fund_utility", "base_rate", "risk_level", "fee_level", "complexity", "cluster_key"]]

    lines = []
    lines.append("# 펀드 Utility 인덱스 보고서 (상품 관점)")
    lines.append("")
    lines.append(f"- 적용 폰트: {font_name}")
    lines.append(f"- 총 펀드 수: {len(out)}")
    lines.append(f"- 평균 utility: {out['fund_utility'].mean():.4f}")
    lines.append(f"- 중앙 utility: {out['fund_utility'].median():.4f}")
    lines.append("")
    lines.append("## 정의")
    lines.append("- fund_utility = 0.35*U_return + 0.25*U_risk_eff + 0.20*U_cost_eff + 0.10*U_liquidity + 0.10*U_simplicity")
    lines.append("- U_return: 수익률 정규화(상하위 1% clipping)")
    lines.append("- U_risk_eff: 위험조정 수익률 정규화")
    lines.append("- U_cost_eff: (1 - 보수수준 정규화)")
    lines.append("- U_liquidity: 환매 용이성 proxy")
    lines.append("- U_simplicity: 복잡도 낮을수록 가점")
    lines.append("")
    lines.append("## 주의")
    lines.append("- 본 지표는 수신상품 utility와 별개로 해석해야 하며, 스케일 직접 비교를 권장하지 않습니다.")
    lines.append("")
    lines.append("## Top 10 펀드")
    for _, r in top10.iterrows():
        lines.append(
            f"- {r['product_id']} | {r['product_name']} | utility={r['fund_utility']:.4f} | ret={r['base_rate']:.3f} | risk={int(r['risk_level'])} | fee={int(r['fee_level'])} | cpx={int(r['complexity'])}"
        )
    lines.append("")
    lines.append("## 산출물")
    lines.append(f"- csv: `{args.out_csv}`")
    lines.append(f"- csv: `{args.out_dir / 'cluster_utility_summary.csv'}`")
    lines.append(f"- figure: `{args.out_dir / '01_utility_distribution.png'}`")
    lines.append(f"- figure: `{args.out_dir / '02_return_vs_risk_scatter.png'}`")
    lines.append(f"- figure: `{args.out_dir / '03_cluster_utility_boxplot.png'}`")

    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {args.out_csv}")
    print(f"saved: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
