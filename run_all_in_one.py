# -*- coding: utf-8 -*-
# 통합 실행 스크립트 v3: run_all_in_one.py
# 기능: 크롤링, 스크랩 가치 예측, 정적 파일 생성을 모두 수행하며, JSON 변환 오류를 해결합니다.

import pandas as pd
import numpy as np
import re
import pickle
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import lightgbm as lgb
import os
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import sys
import io

# --- 전역 설정 및 파일 경로 ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if 'JAVA_HOME' not in os.environ: pass

NEWSPAPERS_CSV = 'newspapers.csv'
CRAWLED_ARTICLES_CSV = 'crawled_articles.csv'
SCRAPPED_NEWS_TODAY_CSV = 'scrapped_news_today.csv'
MODEL_PATH = 'scrap_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
JSON_OUTPUT_FILE = 'news_data.json'
HTML_TEMPLATE_FILE = os.path.join('templates', 'index.html')
HTML_OUTPUT_FILE = 'github_pages_index.html'
JS_TEMPLATE_FILE = os.path.join('static', 'js', 'main.js')
JS_OUTPUT_FILE = os.path.join('static', 'js', 'github_pages_main.js')

# --- 1단계: 뉴스 크롤링 ---
def run_crawling(input_csv_path=NEWSPAPERS_CSV, output_csv_path=CRAWLED_ARTICLES_CSV):
    print("--- 1단계: 뉴스 크롤링 시작 ---")
    try:
        df = pd.read_csv(input_csv_path, encoding='utf-8')
    except Exception as e:
        print(f"  오류: 입력 파일 '{input_csv_path}' 읽기 실패 - {e}")
        return False
    crawled_data = []
    for index, row in df.iterrows():
        url = row.get('링크', '')
        if not url or not isinstance(url, str) or 'n.news.naver.com' not in url:
            continue
        print(f"  [{index+1}/{len(df)}] 크롤링 중: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            crawled_title = (soup.select_one('h2.media_end_head_headline') or soup.select_one('#title_area span')).get_text(strip=True)
            crawled_content_element = soup.select_one('#dic_area') or soup.select_one('#articeBody')
            if crawled_content_element:
                for el in crawled_content_element.select('script, style, .reporter_area, .ad_area'): el.decompose()
                crawled_content = crawled_content_element.get_text(strip=True)
            else:
                crawled_content = "본문 없음"
            crawled_data.append({'신문사': row.get('신문사', ''), '링크': url, '크롤링된_제목': crawled_title, '크롤링된_본문': crawled_content})
            time.sleep(0.5)
        except Exception as e:
            print(f"  오류: {url} 처리 중 - {e}")
    if not crawled_data:
        print("  크롤링된 데이터가 없습니다.")
        return False
    pd.DataFrame(crawled_data).to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"  크롤링 완료! '{output_csv_path}'에 저장되었습니다.")
    print("--- 1단계: 뉴스 크롤링 완료 ---\\n")
    return True

# --- 2단계: 스크랩 가치 예측 ---
def run_daily_scraping(input_file=CRAWLED_ARTICLES_CSV, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH, output_file=SCRAPPED_NEWS_TODAY_CSV):
    print("--- 2단계: 스크랩 가치 예측 시작 ---")
    try:
        with open(model_path, "rb") as f: model = pickle.load(f)
        with open(vectorizer_path, "rb") as f: tfidf_vectorizer = pickle.load(f)
        df = pd.read_csv(input_file, encoding='utf-8')
        df.dropna(subset=['크롤링된_본문', '크롤링된_제목'], inplace=True)
    except Exception as e:
        print(f"  오류: 파일 또는 모델 로딩 실패 - {e}")
        return False
    
    okt = Okt()
    def preprocess(text): return ' '.join(okt.nouns(re.sub(r'[^ㄱ-ㅎㆠ-ㆺ가-힣 ]','', str(text))))
    df['processed_text'] = df.apply(lambda r: preprocess(r['크롤링된_제목'] + " " + r['크롤링된_본문']), axis=1)
    
    keywords = {'업계': ['식품', '화학', '바이오', '패키징'], '경영': ['경영', '경제', '환율', 'M&A']}
    df['업계_키워드_개수'] = df['processed_text'].apply(lambda x: sum(k in x for k in keywords['업계']))
    df['경영_키워드_개수'] = df['processed_text'].apply(lambda x: sum(k in x for k in keywords['경영']))
    df['본문_길이'] = df['크롤링된_본문'].str.len()
    
    text_features = tfidf_vectorizer.transform(df['processed_text'])
    metadata_features = df[['업계_키워드_개수', '경영_키워드_개수', '본문_길이']].values
    X_new = hstack([text_features, metadata_features])
    df['예측점수'] = model.predict_proba(X_new)[:, 1]
    
    df['카테고리'] = '기타'
    df.loc[df['업계_키워드_개수'] > 0, '카테고리'] = '업계'
    df.loc[(df['업계_키워드_개수'] == 0) & (df['경영_키워드_개수'] > 0), '카테고리'] = '경영'
    
    sorted_articles = df.sort_values(by='예측점수', ascending=False)
    output_df = sorted_articles[['신문사', '크롤링된_제목', '링크', '카테고리', '예측점수']]
    output_df.insert(2, '본문_요약', sorted_articles['크롤링된_본문'].str[:300] + "...")
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("--- 2단계: 스크랩 가치 예측 완료 ---\\n")
    return True

# --- 3단계: 정적 파일 생성 ---
def generate_static_files():
    print("--- 3단계: GitHub Pages용 정적 파일 생성 시작 ---")
    if not os.path.exists(SCRAPPED_NEWS_TODAY_CSV) or os.path.getsize(SCRAPPED_NEWS_TODAY_CSV) == 0:
        print(f"  오류: '{SCRAPPED_NEWS_TODAY_CSV}' 파일이 비어있거나 존재하지 않습니다.")
        return False
        
    # === JSON 생성 로직 수정 ===
    try:
        print("  [1/3] JSON 파일 생성 중...")
        df = pd.read_csv(SCRAPPED_NEWS_TODAY_CSV, encoding='utf-8-sig')
        
        # numpy의 NaN 값을 Python의 None으로 명시적으로 변환
        # 이렇게 하면 json.dump가 NaN 대신 null로 올바르게 변환합니다.
        df = df.replace({np.nan: None})
        
        cleaned_data = df.to_dict(orient='records')
        
        with open(JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
            
        print(f"    '{JSON_OUTPUT_FILE}' 생성 완료.")
    except Exception as e:
        print(f"  오류: JSON 변환 실패 - {e}")
        return False

    # HTML 및 JS 생성
    try:
        print("  [2/3] HTML 파일 생성 중...")
        with open(HTML_TEMPLATE_FILE, 'r', encoding='utf-8') as f: html_content = f.read()
        modified_html = html_content.replace('<script src="/static/js/main.js"></script>', '<script src="static/js/github_pages_main.js"></script>')
        with open(HTML_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(modified_html)
        print(f"    '{HTML_OUTPUT_FILE}' 생성 완료.")
        
        print("  [3/3] JavaScript 파일 생성 중...")
        with open(JS_TEMPLATE_FILE, 'r', encoding='utf-8') as f: js_content = f.read()
        modified_js = js_content.replace("fetch('/api/news')", "fetch('news_data.json')")
        modified_js = modified_js.replace("setInterval(fetchNews, 30000);", "")
        with open(JS_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(modified_js)
        print(f"    '{JS_OUTPUT_FILE}' 생성 완료.")
    except Exception as e:
        print(f"  오류: HTML 또는 JS 파일 생성 실패 - {e}")
        return False
        
    print("--- 3단계: 정적 파일 생성 완료 ---\\n")
    return True

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    print("========== 통합 스크립트 실행 시작 ==========")
    if run_crawling():
        if run_daily_scraping():
            if generate_static_files():
                print("========== 모든 작업이 성공적으로 완료되었습니다. ==========")
            else:
                print("!!!!! 3단계(정적 파일 생성)에서 오류가 발생하여 중단되었습니다. !!!!!")
        else:
            print("!!!!! 2단계(스크랩 예측)에서 오류가 발생하여 중단되었습니다. !!!!!")
    else:
        print("!!!!! 1단계(크롤링)에서 오류가 발생하여 중단되었습니다. !!!!!")
