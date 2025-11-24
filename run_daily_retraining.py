# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import pickle
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import lightgbm as lgb
import os
import sys

# --- 경로 설정 (GitHub Actions 환경에 맞춘 상대 경로) ---
new_labeled_file = "labeled_for_training.csv"
historical_file = "historical_scraps.csv"
model_path = "scrap_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

def run_daily_retraining():
    print(f"--- 일일 재학습 프로세스를 시작합니다. ---")
    
    # 1. 신규 라벨링 데이터 로드
    if not os.path.exists(new_labeled_file):
        print(f"알림: '{new_labeled_file}' 파일이 없습니다. 작업을 종료합니다.")
        return

    try:
        new_df = pd.read_csv(new_labeled_file, encoding='utf-8-sig')
        print(f"1. 신규 데이터 로드 완료. (총 {len(new_df)}개)")
        print(f"▶ 파일 내 컬럼 목록: {new_df.columns.tolist()}")

        # [수정] '최종선택여부' 컬럼 확인 (사진상 짤려있을 수 있어 유연하게 처리)
        target_col = '최종선택여부'
        if target_col not in new_df.columns:
            # 혹시 '최종선택여' 처럼 짤려있거나 공백이 있을 경우를 대비해 확인
            possible_cols = [c for c in new_df.columns if '최종선택' in c]
            if possible_cols:
                target_col = possible_cols[0]
                print(f"알림: '{target_col}' 컬럼을 라벨(y)로 사용합니다.")
                new_df.rename(columns={target_col: '최종선택여부'}, inplace=True)
            else:
                print(f"오류: '{new_labeled_file}'에 [최종선택여부] 컬럼이 없습니다.")
                return
        
        # 라벨 데이터 결측치 처리 (빈 칸은 선택 안 함(0)으로 간주)
        new_df['최종선택여부'] = new_df['최종선택여부'].fillna(0).astype(int)

    except Exception as e:
        print(f"신규 데이터 로드 중 오류: {e}")
        return

    # 2. 데이터 통합 및 컬럼명 변경
    # [핵심 수정] 사진에 있는 '본문_요약'을 '크롤링된_본문'으로 변경하여 학습에 사용
    rename_map = {
        '제목': '크롤링된_제목', 
        '본문_요약': '크롤링된_본문',  # 사진에 맞춰 수정됨
        '본문': '크롤링된_본문'       # 혹시 '본문'이라고 된 파일이 들어올 경우 대비
    }
    new_df.rename(columns=rename_map, inplace=True)
    
    # 필수 컬럼 확인
    required_cols = ['크롤링된_제목', '크롤링된_본문', '최종선택여부']
    for col in required_cols:
        if col not in new_df.columns:
            print(f"오류: '{col}' 컬럼을 찾을 수 없습니다. (현재 컬럼: {new_df.columns.tolist()})")
            return
            
    # 필요한 컬럼만 남기기
    new_df = new_df[required_cols]

    # 기존 데이터 로드 및 병합
    if os.path.exists(historical_file):
        try:
            historical_df = pd.read_csv(historical_file, encoding='utf-8-sig')
            # 기존 데이터와 컬럼명이 다를 경우를 대비해 맞춤
            if '본문' in historical_df.columns and '크롤링된_본문' not in historical_df.columns:
                historical_df.rename(columns={'제목': '크롤링된_제목', '본문': '크롤링된_본문'}, inplace=True)
            
            combined_df = pd.concat([historical_df, new_df], ignore_index=True)
            print(f"2. 데이터 통합 완료 (기존: {len(historical_df)} + 신규: {len(new_df)})")
        except Exception as e:
            print(f"기존 데이터 로드 오류: {e}")
            combined_df = new_df
    else:
        print("기존 데이터 없음. 신규 생성.")
        combined_df = new_df

    # 중복 제거 및 저장
    combined_df.drop_duplicates(subset=['크롤링된_제목', '크롤링된_본문'], inplace=True)
    combined_df.to_csv(historical_file, index=False, encoding='utf-8-sig')
    print(f"3. 통합 데이터 저장 완료 ('{historical_file}', 총 {len(combined_df)}개)")

    # 3. 모델 재학습
    df = combined_df.copy()
    # 텍스트가 없는 행 제거
    df.dropna(subset=['크롤링된_본문', '크롤링된_제목'], inplace=True)

    okt = Okt()
    keywords = {
        '업계': ['식품', '화학', '바이오', '패키징', '플라스틱', '항암제', '배터리', '친환경', 'D램', '삼양', '초순수', '제약사', '숙취', '상쾌환', '설탕', '칼로리', '삼양그룹', '삼양사', '삼양패키징', '삼양엔씨켐', '삼양바이오팜'],
        '경영': ['경영', '경제', '환율', 'M&A', '인수', '투자', '한일경제협회', '실적', '조직문화', '무역']
    }

    def preprocess(text):
        text = re.sub(r'[^ㄱ-ㅎㆠ-ㆺ가-힣 ]','', str(text))
        return ' '.join(okt.nouns(text))

    print("4. 특성 추출 중...")
    df['processed_text'] = df.apply(lambda row: preprocess(str(row['크롤링된_제목']) + " " + str(row['크롤링된_본문'])), axis=1)
    df['processed_title'] = df['크롤링된_제목'].apply(preprocess)

    for key, words in keywords.items():
        df[f'{key}_키워드_개수'] = df['processed_text'].apply(lambda x: sum(w in x for w in words))
        df[f'{key}_키워드_제목_개수'] = df['processed_title'].apply(lambda x: sum(w in x for w in words))

    df['본문_길이'] = df['크롤링된_본문'].str.len().replace(0, 1)
    df['업계_키워드_밀도'] = df['업계_키워드_개수'] / df['본문_길이']
    df['경영_키워드_밀도'] = df['경영_키워드_개수'] / df['본문_길이']

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1500, min_df=2, ngram_range=(1, 2))
    text_features = tfidf.fit_transform(df['processed_text'])

    meta_cols = ['업계_키워드_개수', '경영_키워드_개수', '업계_키워드_제목_개수', '경영_키워드_제목_개수', '본문_길이', '업계_키워드_밀도', '경영_키워드_밀도']
    meta_features = df[meta_cols].astype(np.float32).values * 0.5

    X = hstack([text_features, meta_features]).tocsr()
    y = df['최종선택여부']

    print("5. LightGBM 학습 시작...")
    model = lgb.LGBMClassifier(random_state=42, min_child_samples=5, is_unbalance=True)
    model.fit(X, y)
    
    with open(model_path, "wb") as f: pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f: pickle.dump(tfidf, f)
        
    print(f"\n--- 재학습 완료 및 저장 성공 ---")

if __name__ == "__main__":
    run_daily_retraining()
