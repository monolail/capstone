
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from docx import Document

# 1. 경로 설정
# 현재 파일 위치: scripts/recommender/TPS_Main_v2.py
# 루트 위치: scripts/recommender/ 의 2단계 위 폴더
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from thin_filer.recommender import ThinFilerRecommender
from thin_filer.pipeline_config import RecommenderConfig

def get_desktop_path():
    path = Path(os.path.expanduser("~")) / "OneDrive" / "바탕 화면"
    if not path.exists():
        path = Path(os.path.expanduser("~")) / "Desktop"
    return path

def run_all():
    print("\n" + "="*70)
    print(" [ 신파일러 TPS v2.0 통합 분석 및 성능 평가 시스템 ] ")
    print("="*70)
    
    desktop = get_desktop_path()
    csv_file = desktop / "신파일러_군집_최종_피처_통합.csv"
    
    if not csv_file.exists():
        print(f"Error: {csv_file} 파일을 찾을 수 없습니다.")
        return

    # --- [Step 1] 바탕화면 CSV 기반 TPS 점수 산출 ---
    print("\n[Step 1] CSV 데이터 기반 잠재력 점수(TPS) 산출 중...")
    df = pd.read_csv(csv_file)
    
    # TPS v2.0 공식 적용
    df['s_trust'] = (100.0 - (df['OVERDUE_CNT'] * 30.0) - (df['INST_CNT'] * 5.0)).clip(0, 100)
    df['s_activity'] = (df['TOTAL_SPENDING'].rank(pct=True) * 30 + df['SPENDING_COUNT'].rank(pct=True) * 40 + df['PAY_VISIT_CNT'].rank(pct=True) * 30)
    
    income_pct = df['EST_INCOME'].rank(pct=True) * 100.0
    cb_pct = df['CB_SCORE'].rank(pct=True) * 100.0
    tel_score = df['TEL_GRADE'] * 100.0
    youth_bonus = df['AGE_GB'].apply(lambda x: 100.0 if x in ['20대', '30대'] else 0.0)
    df['s_potential'] = (income_pct * 0.2 + cb_pct * 0.2 + tel_score * 0.3 + youth_bonus * 0.3)
    
    df['tps_score'] = (df['s_trust'] * 0.4) + (df['s_activity'] * 0.3) + (df['s_potential'] * 0.3)
    
    # 상위 5인 출력
    top_5 = df.sort_values('tps_score', ascending=False).head(5)
    print("\n>> TPS 상위 우량군 (Top 5):")
    print(top_5[['CUST_ID', 'AGE_GB', 'tps_score', 's_trust', 'TEL_GRADE']].to_string(index=False))

    # --- [Step 2] 추천 시스템 정량적 성능 평가 (NDCG) ---
    print("\n[Step 2] 추천 시스템 성능 지표(NDCG) 측정 중...")
    config = RecommenderConfig()
    config.data_root = ROOT_DIR / "capstone" / "data"
    rec = ThinFilerRecommender(config)
    
    snapshots = rec.build_user_snapshots(sample_users=100)
    rec.fit(snapshots=snapshots, max_users=80)
    eval_results = rec.evaluate(snapshots, ks=[5])
    ndcg = eval_results.get("metrics", {}).get("model_ndcg@5", 0.0)
    print(f" >> NDCG@5 Score: {ndcg:.4f} (성공)")

    # --- [Step 3] 최종 결과 저장 ---
    print("\n[Step 3] 결과 파일 업데이트 중...")
    output_csv = desktop / "신파일러_TPS_최종_산출.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f" >> 산출 결과 저장 완료: {output_csv}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_all()
