# -*- coding: utf-8 -*-
# 통합 실행 스크립트 v5: run_all_in_one.py
# 기능: 동적 크롤링, 중복 제거, 스크랩 가치 예측, 정적 파일 생성을 모두 수행합니다.

import pandas as pd
import numpy as np
import re
import pickle
import os
import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
import json
import sys
import io
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# --- 전역 설정 및 파일 경로 ---

# SSL 오류 방지 설정 (필요 시)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_CERTIFICATE_VERIFICATION'] = '1'

# 파일 경로 정의 (상대 경로 사용)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAP_DIR = os.path.join(BASE_DIR, 'scrap')
SCRAPPED_NEWS_TODAY_CSV = os.path.join(SCRAP_DIR, 'scrapped_news_today.csv')
MODEL_PATH = os.path.join(SCRAP_DIR, 'scrap_model.pkl')
VECTORIZER_PATH = os.path.join(SCRAP_DIR, 'tfidf_vectorizer.pkl')
JSON_OUTPUT_FILE = os.path.join(BASE_DIR, 'news_data.json')
HTML_TEMPLATE_FILE = os.path.join(SCRAP_DIR, 'templates', 'index.html')
HTML_OUTPUT_FILE = os.path.join(BASE_DIR, 'index.html')
JS_TEMPLATE_FILE = os.path.join(SCRAP_DIR, 'static', 'js', 'main.js')
JS_OUTPUT_FILE = os.path.join(BASE_DIR, 'static', 'js', 'github_pages_main.js')

# --- 공통 전처리 함수 및 객체 ---
okt = Okt()
def preprocess(text):
    """ 텍스트에서 명사만 추출하여 공백으로 구분된 문자열로 반환합니다. """
    return ' '.join(okt.nouns(re.sub(r'[^\ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(text))))

# --- 1단계: 동적 크롤링, 중복 제거 및 전처리 ---

def get_today_articles():
    """ 5대 일간지에서 오늘의 주요 기사 목록(제목, 링크, 신문사)을 수집합니다. """
    print("--- 1.1: 주요 일간지 기사 목록 수집 시작 ---")
    newspapers = {
        '한국경제': '015', '매일경제': '009', '동아일보': '020',
        '조선일보': '023', '중앙일보': '025',
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_articles = []
    for name, oid in newspapers.items():
        try:
            url = f"https://media.naver.com/press/{oid}/newspaper"
            response = requests.get(url, headers=headers, verify=False, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.select('div.sc_offc_lst._paper_article_list a')
            for link in links:
                title = link.get_text(strip=True)
                href = link.get('href', '')
                if title and href:
                    all_articles.append({
                        '신문사': name, '제목': title,
                        '링크': 'https://media.naver.com' + href if href.startswith('/') else href
                    })
            time.sleep(0.5)
        except Exception as e:
            print(f"  ✗ {name} 수집 오류: {e}")
    df = pd.DataFrame(all_articles).drop_duplicates(subset=['링크']).reset_index(drop=True)
    print(f"  - 총 {len(df)}개의 고유 기사 목록 수집 완료.")
    return df

def get_article_content(url):
    """ 네이버 뉴스 링크에서 본문을 크롤링합니다. """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_area = soup.select_one('#dic_area, #articeBody, #article_content')
        if content_area:
            for el in content_area.select('script, style, .reporter_area, .ad_area, .promotion_area, div.byline, a, span.end_photo_org'):
                el.decompose()
            return content_area.get_text(strip=True)
    except Exception as e:
        print(f"    - 링크 크롤링 실패: {url}, 오류: {e}")
    return ""

def run_step_one_crawling_and_preprocessing():
    """ 크롤링, 요약, 중복 제거, 전처리를 통합 실행합니다. """
    articles_df = get_today_articles()
    if articles_df.empty: return pd.DataFrame()

    print("--- 1.2: 기사 본문 크롤링 및 요약 생성 ---")
    crawled_data = []
    for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="  - 기사 본문 크롤링"):
        content = get_article_content(row['링크'])
        if content:
            crawled_data.append([
                row['신문사'], row['제목'], row['링크'], content, content[:400] + "..."
            ])
        time.sleep(0.5)
    
    if not crawled_data:
        print("크롤링된 기사가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(crawled_data, columns=['신문사', '제목', '링크', '본문', '본문_요약'])

    print("--- 1.3: 내용 기반 중복 기사 제거 (유사도 0.6 기준) ---")
    if not df.empty and '본문' in df.columns and not df['본문'].isnull().all():
        df['processed_text_for_dedup'] = df['본문'].apply(preprocess)
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(df['processed_text_for_dedup'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        df['본문_길이_dedup'] = df['본문'].str.len()
        df = df.sort_values(by='본문_길이_dedup', ascending=False).reset_index(drop=True)
        
        tfidf_matrix_sorted = vectorizer.fit_transform(df['processed_text_for_dedup'])
        cosine_sim_sorted = cosine_similarity(tfidf_matrix_sorted)
        
        to_drop = set()
        for i in range(len(cosine_sim_sorted)):
            if i in to_drop: continue
            for j in range(i + 1, len(cosine_sim_sorted)):
                if j in to_drop: continue
                if cosine_sim_sorted[i, j] > 0.6:
                    to_drop.add(j)
        
        if to_drop:
            print(f"   - {len(to_drop)}개의 유사 기사를 제거했습니다.")
            df.drop(index=list(to_drop), inplace=True)
        
        df.drop(columns=['processed_text_for_dedup', '본문_길이_dedup'], inplace=True)

    print(f"   - 최종 분석 대상 기사: {len(df)}개")
    print("--- 1단계: 크롤링 및 전처리 완료 ---")
    print()
    return df

# --- 2단계: 스크랩 가치 예측 ---

def run_step_two_prediction(df):
    """ 학습된 모델을 사용하여 스크랩 가치를 예측하고 최종 결과를 CSV로 저장합니다. """
    if df.empty:
        print("2단계 실패: 분석할 데이터가 없습니다.")
        return False
        
    print("--- 2.1: 모델 및 Vectorizer 로드 ---")
    try:
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f: tfidf_vectorizer = pickle.load(f)
    except FileNotFoundError:
        print(f"오류: 모델 파일({MODEL_PATH}) 또는 Vectorizer 파일({VECTORIZER_PATH})을 찾을 수 없습니다.")
        return False

    print("--- 2.2: 텍스트 전처리 및 특성 추출 ---")
    keywords = {
        '업계': ['식품', '화학', '바이오', '패키징', '플라스틱', '항암제', '배터리', '친환경', 'D램', '삼양', '초순수', '제약사', '숙취', '상쾌환', '설탕', '칼로리', '삼양그룹', '삼양사', '삼양패키징', '삼양엔씨켐', '삼양바이오팜'],
        '경영': ['경영', '경제', '환율', 'M&A', '인수', '투자', '실적', '한일경제협회', '조직문화', '무역']
    }

    df['processed_title'] = df['제목'].apply(preprocess)
    df['processed_text'] = df.apply(lambda r: preprocess(r['제목'] + " " + r['본문']), axis=1)
    
    df['업계_키워드_개수'] = df['processed_text'].apply(lambda x: sum(k in x for k in keywords['업계']))
    df['경영_키워드_개수'] = df['processed_text'].apply(lambda x: sum(k in x for k in keywords['경영']))
    df['업계_키워드_제목_개수'] = df['processed_title'].apply(lambda x: sum(k in x for k in keywords['업계']))
    df['경영_키워드_제목_개수'] = df['processed_title'].apply(lambda x: sum(k in x for k in keywords['경영']))
    df['본문_길이'] = df['본문'].str.len().replace(0, 1)
    df['업계_키워드_밀도'] = df['업계_키워드_개수'] / df['본문_길이']
    df['경영_키워드_밀도'] = df['경영_키워드_개수'] / df['본문_길이']

    text_features = tfidf_vectorizer.transform(df['processed_text'])
    metadata_features = df[['업계_키워드_개수', '경영_키워드_개수', '업계_키워드_제목_개수', '경영_키워드_제목_개수', '본문_길이', '업계_키워드_밀도', '경영_키워드_밀도']].values
    
    X_new = hstack([text_features, metadata_features * 0.5]).tocsr()

    print("--- 2.3: 스크랩 가치 점수 예측 및 결과 저장 ---")
    df['예측점수'] = model.predict_proba(X_new)[:, 1]
    
    df['카테고리'] = '기타'
    df.loc[df['업계_키워드_개수'] > 0, '카테고리'] = '업계'
    df.loc[(df['업계_키워드_개수'] == 0) & (df['경영_키워드_개수'] > 0), '카테고리'] = '경영'

    # 최종선택여부 컬럼 추가 (사용자 입력을 위해 빈 값으로 초기화)
    df['최종선택여부'] = ''

    sorted_df = df.sort_values(by='예측점수', ascending=False)
    
    # '본문' 컬럼 제외하고 저장
    output_columns = ['신문사', '제목', '본문_요약', '링크', '카테고리', '예측점수', '최종선택여부']
    final_df = sorted_df[output_columns]
    
    final_df.to_csv(SCRAPPED_NEWS_TODAY_CSV, index=False, encoding='utf-8-sig')
    print(f"  - 최종 결과가 '{SCRAPPED_NEWS_TODAY_CSV}' 파일로 저장되었습니다.")
    print("--- 2단계: 스크랩 가치 예측 완료 ---")
    print()
    return True

# --- 3단계: 정적 파일 생성 ---
def generate_static_files():
    print("--- 3단계: GitHub Pages용 정적 파일 생성 시작 ---")
    
    if not os.path.exists(SCRAPPED_NEWS_TODAY_CSV) or os.path.getsize(SCRAPPED_NEWS_TODAY_CSV) == 0:
        print(f"  오류: '{SCRAPPED_NEWS_TODAY_CSV}' 파일이 비어있거나 존재하지 않습니다.")
        return False
        
    try:
        print("  [1/3] JSON 파일 생성 중...")
        df = pd.read_csv(SCRAPPED_NEWS_TODAY_CSV, encoding='utf-8-sig')
        df.to_json(JSON_OUTPUT_FILE, orient='records', force_ascii=False, indent=4)
        print(f"    '{JSON_OUTPUT_FILE}' 생성 완료.")
    except Exception as e:
        print(f"  오류: JSON 변환 실패 - {e}")
        return False

    try:
        print("  [2/3] HTML 파일 생성 중...")
        with open(HTML_TEMPLATE_FILE, 'r', encoding='utf-8') as f: html_content = f.read()
        modified_html = html_content.replace('<script src="/static/js/main.js"></script>', '<script src="static/js/github_pages_main.js"></script>')
        with open(HTML_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(modified_html)
        print(f"    '{HTML_OUTPUT_FILE}' 생성 완료.")
    except FileNotFoundError:
        print(f"  오류: HTML 템플릿 파일 '{HTML_TEMPLATE_FILE}'을 찾을 수 없습니다.")
        return False
    except Exception as e:
        print(f"  오류: HTML 파일 생성 실패 - {e}")
        return False

    try:
        print("  [3/3] JavaScript 파일 생성 중...")
        with open(JS_TEMPLATE_FILE, 'r', encoding='utf-8') as f: js_content = f.read()
        modified_js = js_content.replace("fetch('/api/news')", "fetch('news_data.json')")
        modified_js = modified_js.replace("setInterval(fetchNews, 30000);", "/* 자동 새로고침 비활성화 */")
        with open(JS_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(modified_js)
        print(f"    '{JS_OUTPUT_FILE}' 생성 완료.")
    except FileNotFoundError:
        print(f"  오류: JS 템플릿 파일 '{JS_TEMPLATE_FILE}'을 찾을 수 없습니다.")
        return False
    except Exception as e:
        print(f"  오류: JavaScript 파일 생성 실패 - {e}")
        return False

    print("--- 3단계: 정적 파일 생성 완료 ---")
    print()
    return True

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # 1단계 실행
    processed_articles_df = run_step_one_crawling_and_preprocessing()
    
    # 2단계 실행
    if not processed_articles_df.empty:
        prediction_success = run_step_two_prediction(processed_articles_df)
        
        # 3단계 실행
        if prediction_success:
            if generate_static_files():
                print("🎉 모든 작업이 성공적으로 완료되었습니다.")
            else:
                print("!!!!! 3단계(정적 파일 생성)에서 오류가 발생했습니다. !!!!!")
        else:
            print("!!!!! 2단계(스크랩 예측)에서 오류가 발생하여 중단되었습니다. !!!!!")
    else:
        print("!!!!! 1단계(크롤링 및 전처리)에서 처리할 기사를 찾지 못했습니다. !!!!!")
