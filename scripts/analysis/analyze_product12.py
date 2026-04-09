#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_from_text(value: object, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    s = str(value)
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else default


def amount_bin(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    s = str(value)
    if "제한없음" in s:
        return 0
    n = parse_int_from_text(s, default=0)
    if n <= 0:
        return 0
    if n < 100:
        return 0
    if n < 500:
        return 1
    if n < 1000:
        return 2
    return 3


def normalize_deposit(dep: pd.DataFrame) -> pd.DataFrame:
    max_term = dep.get("계약기간개월수_최대구간", "").astype(str)
    maturity = dep.get("만기여부", "").astype(str)

    horizon = np.where(
        max_term.str.contains("6|12", regex=True, na=False),
        "short",
        np.where(max_term.str.contains("24|36", regex=True, na=False), "mid", "long"),
    )

    liquidity = np.where(maturity.str.contains("만기 없음", na=False), 3, 1)

    return pd.DataFrame(
        {
            "product_id": dep["상품코드"].astype(str),
            "product_name": dep["상품명"].astype(str),
            "source": "bank_deposit",
            "product_family": "deposit",
            "product_group_code": dep.get("상품그룹코드", "").astype(str),
            "risk_level": 0,
            "liquidity_level": liquidity,
            "horizon": horizon,
            "complexity": pd.to_numeric(dep.get("우대금리조건_개수", 0), errors="coerce").fillna(0).clip(0, 2),
            "min_amount_bin": dep.get("가입금액_최소구간", "").map(amount_bin),
            "fee_level": 0,
            "principal_variation": 0,
            "base_rate": pd.to_numeric(dep.get("기본금리", 0), errors="coerce").fillna(0.0),
            "max_rate": pd.to_numeric(dep.get("최대우대금리", 0), errors="coerce").fillna(0.0),
            "maturity_type": maturity,
            "raw_group": dep.get("예금입출금방식", "").astype(str),
        }
    )


def normalize_fund(fund: pd.DataFrame) -> pd.DataFrame:
    risk_raw = pd.to_numeric(fund.get("투자위험등급", 2), errors="coerce").fillna(2)
    fund_fee = pd.to_numeric(fund.get("판매보수", 0), errors="coerce").fillna(0)

    horizon = np.where(
        fund.get("대유형", "").astype(str).str.contains("채권|MMF", na=False),
        "short",
        np.where(fund.get("대유형", "").astype(str).str.contains("혼합", na=False), "mid", "long"),
    )

    liquidity = np.where(
        fund.get("중유형", "").astype(str).str.contains("MMF|채권", regex=True, na=False),
        2,
        1,
    )

    complexity = (
        fund.get("고난도금융상품", "N").astype(str).eq("Y").astype(int)
        + fund.get("레버리지", "N").astype(str).eq("Y").astype(int)
    ).clip(0, 2)

    return pd.DataFrame(
        {
            "product_id": fund["펀드코드"].astype(str),
            "product_name": fund["펀드명"].astype(str),
            "source": "public_fund",
            "product_family": "fund",
            "risk_level": (risk_raw - 1).clip(0, 3),
            "liquidity_level": liquidity,
            "horizon": horizon,
            "complexity": complexity,
            "min_amount_bin": 1,
            "fee_level": pd.qcut(fund_fee.rank(method="average"), q=4, labels=False, duplicates="drop").fillna(0),
            "principal_variation": 1,
            "base_rate": pd.to_numeric(fund.get("펀드성과정보_1년", 0), errors="coerce").fillna(0.0),
            "max_rate": pd.to_numeric(fund.get("펀드성과정보_1년", 0), errors="coerce").fillna(0.0),
            "maturity_type": "market_linked",
            "raw_group": fund.get("대유형", "").astype(str),
        }
    )


def describe_top(series: pd.Series, n: int = 10) -> str:
    vc = series.astype(str).value_counts(dropna=False).head(n)
    return "\n".join([f"- {idx}: {int(cnt)}" for idx, cnt in vc.items()])


def build_deposit_report(dep: pd.DataFrame, norm: pd.DataFrame, out_path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 12번 데이터 EDA (은행수신상품 전용)",
        "",
        f"- 생성시각: {now}",
        f"- 원천 크기: {dep.shape[0]} rows / {dep.shape[1]} cols",
        f"- 정규화 크기: {norm.shape[0]} rows / {norm.shape[1]} cols",
        "",
        "## 핵심 결측률",
        f"- 상품코드: {dep['상품코드'].isna().mean():.4f}",
        f"- 상품명: {dep['상품명'].isna().mean():.4f}",
        f"- 만기여부: {dep['만기여부'].isna().mean():.4f}",
        f"- 기본금리: {dep['기본금리'].isna().mean():.4f}",
        f"- 최대우대금리: {dep['최대우대금리'].isna().mean():.4f}",
        "",
        "## 주요 카테고리 분포",
        "### 예금입출금방식",
        describe_top(dep['예금입출금방식']),
        "",
        "### 만기여부",
        describe_top(dep['만기여부']),
        "",
        "## 정규화 특성 분포",
    ]
    for c in ["risk_level", "liquidity_level", "horizon", "complexity", "min_amount_bin"]:
        lines.append(f"### {c}")
        lines.append(describe_top(norm[c]))
        lines.append("")

    lines.extend([
        "## 해석",
        "- 우대금리 결측이 크므로 금리 신호는 보조로 사용",
        "- 만기/입출금방식 중심으로 유동성 성향 매핑이 핵심",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_fund_report(fund: pd.DataFrame, norm: pd.DataFrame, out_path: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 12번 데이터 EDA (공모펀드상품 전용)",
        "",
        f"- 생성시각: {now}",
        f"- 원천 크기: {fund.shape[0]} rows / {fund.shape[1]} cols",
        f"- 정규화 크기: {norm.shape[0]} rows / {norm.shape[1]} cols",
        "",
        "## 핵심 결측률",
        f"- 펀드코드: {fund['펀드코드'].isna().mean():.4f}",
        f"- 펀드명: {fund['펀드명'].isna().mean():.4f}",
        f"- 대유형: {fund['대유형'].isna().mean():.4f}",
        f"- 투자위험등급: {fund['투자위험등급'].isna().mean():.4f}",
        f"- 판매보수: {fund['판매보수'].isna().mean():.4f}",
        "",
        "## 주요 카테고리 분포",
        "### 대유형",
        describe_top(fund['대유형']),
        "",
        "### 중유형",
        describe_top(fund['중유형']),
        "",
        "### 투자위험등급",
        describe_top(pd.to_numeric(fund['투자위험등급'], errors='coerce')),
        "",
        "## 정규화 특성 분포",
    ]
    for c in ["risk_level", "liquidity_level", "horizon", "complexity", "fee_level"]:
        lines.append(f"### {c}")
        lines.append(describe_top(norm[c]))
        lines.append("")

    lines.extend([
        "## 해석",
        "- 펀드 카탈로그 비중이 매우 크므로 후보생성 시 과대표집 제어 필요",
        "- 위험등급/보수/유형 축으로 사용자 위험성향 매칭을 세분화할 수 있음",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze table 12 with split outputs (deposit/fund)")
    parser.add_argument("--data-root", type=Path, default=Path("data/12.금융상품정보"))
    parser.add_argument("--out-deposit-csv", type=Path, default=Path("data/processed/product12_deposit_catalog.csv"))
    parser.add_argument("--out-fund-csv", type=Path, default=Path("data/processed/product12_fund_catalog.csv"))
    parser.add_argument("--out-deposit-report", type=Path, default=Path("reports/product12/deposit_eda_report.md"))
    parser.add_argument("--out-fund-report", type=Path, default=Path("reports/product12/fund_eda_report.md"))
    args = parser.parse_args()

    dep = pd.read_csv(args.data_root / "은행수신상품.csv", low_memory=False)
    fund = pd.read_csv(args.data_root / "공모펀드상품.csv", low_memory=False)

    dep_norm = normalize_deposit(dep).drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    fund_norm = normalize_fund(fund).drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    args.out_deposit_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_fund_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_deposit_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_fund_report.parent.mkdir(parents=True, exist_ok=True)

    dep_norm.to_csv(args.out_deposit_csv, index=False, encoding="utf-8-sig")
    fund_norm.to_csv(args.out_fund_csv, index=False, encoding="utf-8-sig")

    build_deposit_report(dep, dep_norm, args.out_deposit_report)
    build_fund_report(fund, fund_norm, args.out_fund_report)

    print(f"saved deposit csv: {args.out_deposit_csv}")
    print(f"saved fund csv: {args.out_fund_csv}")
    print(f"saved deposit report: {args.out_deposit_report}")
    print(f"saved fund report: {args.out_fund_report}")


if __name__ == "__main__":
    main()
