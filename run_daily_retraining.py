# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import pickle
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack
import lightgbm as lgb
import os
import sys

os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk-25+36'

# --- 경로 설정 ---
new_labeled_file = "C:/Users/syc217052/Documents/ai_inov/scrap/labeled_for_training.csv"
historical_file = "C:/Users/syc217052/Documents/ai_inov/scrap/historical_scraps.csv"
model_path = "C:/Users/syc217052/Documents/ai_inov/scrap/scrap_model.pkl"
vectorizer_path = "C:/Users/syc217052/Documents/ai_inov/scrap/tfidf_vectorizer.pkl"

def run_daily_retraining():
    """
    매일 사용자가 라벨링한 새로운 데이터를 기존 학습 데이터에 통합하고,
    전체 데이터를 사용하여 모델을 재학습하고 저장합니다.
    """
    print(f"--- 일일 재학습 프로세스를 시작합니다. ---")

    # 1. 신규 라벨링 데이터 로드
    try:
        new_df = pd.read_csv(new_labeled_file, encoding='utf-8-sig')
        # 최종선택여부 컬럼이 없거나 비어있으면 중단
        if '최종선택여부' not in new_df.columns or new_df['최종선택여부'].isnull().all():
            print(f"오류: '{new_labeled_file}'에 [최종선택여부] 컬럼이 없거나, 모든 값이 비어있습니다.")
            print("사용자가 직접 스크랩할 기사를 선택하고 '1'을 입력해야 합니다.")
            return
        print(f"1. 신규 라벨링 데이터 로드 완료. (총 {len(new_df)}개)")
    except FileNotFoundError:
        print(f"오류: 신규 라벨링 파일인 '{new_labeled_file}'을(를) 찾을 수 없습니다.")
        print("scrapped_news_today.csv 파일에 '최종선택여부' 컬럼을 추가하고, 파일명을 변경해주세요.")
        return
    except Exception as e:
        print(f"신규 데이터 로드 중 오류: {e}")
        return

    # 2. 데이터 통합
    # 컬럼명 통일 (스크래핑 스크립트와 학습 스크립트 간의 컬럼명 차이 해결)
    new_df.rename(columns={'제목': '크롤링된_제목', '본문': '크롤링된_본문'}, inplace=True)
    # 필요한 컬럼만 선택
    new_df = new_df[['크롤링된_제목', '크롤링된_본문', '최종선택여부']]
    
    # 최종선택여부가 NaN인 경우 0으로 채우고, 정수형으로 변환 (선택(1) 외에는 모두 미선택(0)으로 간주)
    new_df['최종선택여부'].fillna(0, inplace=True)
    new_df['최종선택여부'] = new_df['최종선택여부'].astype(int)
    print(f"   - 신규 데이터 라벨 분포:\n{new_df['최종선택여부'].value_counts().to_string()}")

    try:
        historical_df = pd.read_csv(historical_file, encoding='utf-8-sig')
        combined_df = pd.concat([historical_df, new_df], ignore_index=True)
        print(f"2. 기존 학습 데이터와 신규 데이터를 통합했습니다. (기존: {len(historical_df)}개, 신규: {len(new_df)}개 -> 총: {len(combined_df)}개)")
    except FileNotFoundError:
        print(f"   - 기존 학습 파일('{historical_file}')이 없어 새로 생성합니다.")
        combined_df = new_df
    
    # 중복 제거
    combined_df.drop_duplicates(subset=['크롤링된_제목', '크롤링된_본문'], inplace=True)
    
    # 통합된 데이터 저장
    combined_df.to_csv(historical_file, index=False, encoding='utf-8-sig')
    print(f"3. 통합된 전체 학습 데이터를 '{historical_file}'에 저장했습니다. (최종 {len(combined_df)}개)")

    # 3. 모델 재학습 (run_model_training.py 로직과 동일)
    df = combined_df
    df.dropna(subset=['크롤링된_본문', '크롤링된_제목', '최종선택여부'], inplace=True)

    okt = Okt()
    keywords = {
        '업계': ['식품', '화학', '바이오', '패키징', '플라스틱', '항암제', '배터리', '친환경', 'D램', '삼양', '초순수', '제약사', '숙취', '상쾌환', '설탕', '칼로리', '삼양그룹', '삼양사', '삼양패키징', '삼양엔씨켐', '삼양바이오팜'],
        '경영': ['경영', '경제', '환율', 'M&A', '인수', '투자', '한일경제협회', '실적', '조직문화', '무역']
    }

    def preprocess(text):
        text = re.sub(r'[^ㄱ-ㅎㆠ-ㆺ가-힣 ]','', str(text))
        tokens = okt.nouns(text)
        return ' '.join(tokens)

    print("4. 텍스트 전처리 및 특성 추출 중...")
    df['processed_title'] = df['크롤링된_제목'].apply(preprocess)
    df['processed_text'] = df.apply(lambda row: preprocess(row['크롤링된_제목'] + " " + row['크롤링된_본문']), axis=1)
    
    # 전체 키워드 개수
    df['업계_키워드_개수'] = df['processed_text'].apply(lambda x: sum(keyword in x for keyword in keywords['업계']))
    df['경영_키워드_개수'] = df['processed_text'].apply(lambda x: sum(keyword in x for keyword in keywords['경영']))
    
    # 제목 키워드 개수
    df['업계_키워드_제목_개수'] = df['processed_title'].apply(lambda x: sum(keyword in x for keyword in keywords['업계']))
    df['경영_키워드_제목_개수'] = df['processed_title'].apply(lambda x: sum(keyword in x for keyword in keywords['경영']))

    # 본문 길이 및 키워드 밀도
    df['본문_길이'] = df['크롤링된_본문'].str.len().replace(0, 1) # 0으로 나누는 것을 방지
    df['업계_키워드_밀도'] = df['업계_키워드_개수'] / df['본문_길이']
    df['경영_키워드_밀도'] = df['경영_키워드_개수'] / df['본문_길이']

    # TF-IDF (단일 및 이중 단어 조합 포함)
    tfidf_vectorizer = TfidfVectorizer(max_features=1500, min_df=2, ngram_range=(1, 2))
    text_features = tfidf_vectorizer.fit_transform(df['processed_text'])

    # 메타데이터 특성 리스트 확장
    feature_names = [
        '업계_키워드_개수', '경영_키워드_개수', 
        '업계_키워드_제목_개수', '경영_키워드_제목_개수',
        '본문_길이', 
        '업계_키워드_밀도', '경영_키워드_밀도'
    ]
    metadata_features = df[feature_names].astype(np.float32).values

    # --- 머신러닝(TF-IDF) 특성 비중 강화를 위해 키워드 기반 특성 영향력 축소 ---
    keyword_feature_weight = 0.5 
    metadata_features = metadata_features * keyword_feature_weight

    X = hstack([text_features, metadata_features]).tocsr()
    y = df['최종선택여부']
    print("5. 모든 특성 결합 완료.")

    print("6. LightGBM 모델 재학습 중...")
    final_model = lgb.LGBMClassifier(random_state=42, min_child_samples=5, is_unbalance=True)
    final_model.fit(X, y)
    
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
        
    print(f"\n--- 재학습 완료 ---")
    print(f"7. 재학습된 모델과 Vectorizer를 '{model_path}', '{vectorizer_path}' 파일로 저장했습니다.")

if __name__ == "__main__":
    run_daily_retraining()
