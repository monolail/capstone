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
    candidates = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "AppleGothic",
        "Malgun Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for c in candidates:
        if c in available:
            chosen = c
            break
    if chosen is None:
        chosen = "DejaVu Sans"
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize split product12 catalogs (deposit/fund)")
    p.add_argument("--deposit-csv", type=Path, default=Path("data/processed/product12_deposit_catalog.csv"))
    p.add_argument("--fund-csv", type=Path, default=Path("data/processed/product12_fund_catalog.csv"))
    p.add_argument("--out-deposit-dir", type=Path, default=Path("reports/product12/figures_deposit"))
    p.add_argument("--out-fund-dir", type=Path, default=Path("reports/product12/figures_fund"))
    return p.parse_args()


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path, topn: int | None = None) -> None:
    s = series.value_counts(dropna=False)
    if topn is not None:
        s = s.head(topn)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=s.index.astype(str), y=s.values, ax=ax, color="#2E86AB")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(s.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_hist(series: pd.Series, title: str, xlabel: str, out_path: Path, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(pd.to_numeric(series, errors="coerce").dropna(), bins=bins, ax=ax, color="#7DCEA0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, row: str, col: str, title: str, out_path: Path) -> None:
    ct = pd.crosstab(df[row].astype(str), df[col].astype(str))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_index(out_dir: Path, title: str, files: list[str], font_name: str) -> None:
    lines = [f"# {title}", "", f"- 적용 폰트: {font_name}", ""]
    lines += [f"- [{f}]({f})" for f in files]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def visualize_deposit(df: pd.DataFrame, out_dir: Path, font_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    f = "01_예금_유형분포_raw_group.png"
    save_bar(df["raw_group"], "은행수신 상품유형 분포(상위 15)", "예금입출금방식", "건수", out_dir / f, topn=15)
    files.append(f)

    f = "02_예금_만기유형분포.png"
    save_bar(df["maturity_type"], "은행수신 만기유형 분포", "만기유형", "건수", out_dir / f)
    files.append(f)

    f = "03_예금_유동성분포.png"
    save_bar(df["liquidity_level"], "은행수신 유동성 수준 분포", "liquidity_level", "건수", out_dir / f)
    files.append(f)

    f = "04_예금_수평선호_horizon.png"
    save_bar(df["horizon"], "은행수신 기간(horizon) 분포", "horizon", "건수", out_dir / f)
    files.append(f)

    f = "05_예금_가입금액구간.png"
    save_bar(df["min_amount_bin"], "은행수신 최소가입금액 구간 분포", "min_amount_bin", "건수", out_dir / f)
    files.append(f)

    f = "06_예금_기본금리히스토그램.png"
    save_hist(df["base_rate"], "은행수신 기본금리 분포", "base_rate", out_dir / f, bins=35)
    files.append(f)

    f = "07_예금_최대금리히스토그램.png"
    save_hist(df["max_rate"], "은행수신 최대우대금리 분포", "max_rate", out_dir / f, bins=35)
    files.append(f)

    f = "08_예금_유동성x기간_heatmap.png"
    save_heatmap(df, "liquidity_level", "horizon", "은행수신 유동성 x 기간", out_dir / f)
    files.append(f)

    write_index(out_dir, "은행수신상품 시각화 결과", files, font_name)


def visualize_fund(df: pd.DataFrame, out_dir: Path, font_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    f = "01_펀드_대유형분포_top15.png"
    save_bar(df["raw_group"], "공모펀드 대유형 분포(상위 15)", "대유형", "건수", out_dir / f, topn=15)
    files.append(f)

    f = "02_펀드_위험수준분포.png"
    save_bar(df["risk_level"], "공모펀드 위험수준 분포", "risk_level", "건수", out_dir / f)
    files.append(f)

    f = "03_펀드_유동성분포.png"
    save_bar(df["liquidity_level"], "공모펀드 유동성 수준 분포", "liquidity_level", "건수", out_dir / f)
    files.append(f)

    f = "04_펀드_기간분포_horizon.png"
    save_bar(df["horizon"], "공모펀드 기간(horizon) 분포", "horizon", "건수", out_dir / f)
    files.append(f)

    f = "05_펀드_복잡도분포.png"
    save_bar(df["complexity"], "공모펀드 복잡도 분포", "complexity", "건수", out_dir / f)
    files.append(f)

    f = "06_펀드_수수료구간분포.png"
    save_bar(df["fee_level"], "공모펀드 수수료 구간 분포", "fee_level", "건수", out_dir / f)
    files.append(f)

    f = "07_펀드_수익률proxy히스토그램.png"
    save_hist(df["max_rate"], "공모펀드 1년성과(max_rate proxy) 분포", "max_rate", out_dir / f, bins=40)
    files.append(f)

    f = "08_펀드_위험x기간_heatmap.png"
    save_heatmap(df, "risk_level", "horizon", "공모펀드 위험수준 x 기간", out_dir / f)
    files.append(f)

    write_index(out_dir, "공모펀드상품 시각화 결과", files, font_name)


def main() -> None:
    args = parse_args()
    font_name = set_korean_font()

    dep = pd.read_csv(args.deposit_csv)
    fund = pd.read_csv(args.fund_csv)

    visualize_deposit(dep, args.out_deposit_dir, font_name)
    visualize_fund(fund, args.out_fund_dir, font_name)

    print(f"font: {font_name}")
    print(f"saved deposit figures: {args.out_deposit_dir}")
    print(f"saved fund figures: {args.out_fund_dir}")


if __name__ == "__main__":
    main()
