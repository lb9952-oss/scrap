1 # -*- coding: utf-8 -*-
     2 # 통합 실행 스크립트 v8 (최종): run_all_in_one.py
     3 # 기능: 최상위 경로 구조에 맞춰 동적 크롤링, 중복 제거, 스크랩 가치 예측, 정적 파일 생성을 모두 수행합니다.
     4 
     5 import pandas as pd
     6 import numpy as np
     7 import re
     8 import pickle
     9 import os
    10 import requests
    11 from bs4 import BeautifulSoup
    12 from konlpy.tag import Okt
    13 from sklearn.feature_extraction.text import TfidfVectorizer
    14 from scipy.sparse import hstack
    15 import time
    16 import json
    17 import sys
    18 import io
    19 from tqdm import tqdm
    20 from sklearn.metrics.pairwise import cosine_similarity
    21 
    22 # --- 전역 설정 및 파일 경로 (최상위 경로 기준) ---
    23 
    24 # SSL 오류 방지 설정 (필요 시)
    25 os.environ['CURL_CA_BUNDLE'] = ''
    26 os.environ['REQUESTS_CA_BUNDLE'] = ''
    27 os.environ['HF_HUB_DISABLE_CERTIFICATE_VERIFICATION'] = '1'
    28 
    29 # 파일 경로 정의
    30 BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    31 SCRAPPED_NEWS_TODAY_CSV = os.path.join(BASE_DIR, 'scrapped_news_today.csv')
    32 MODEL_PATH = os.path.join(BASE_DIR, 'scrap_model.pkl')
    33 VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
    34 JSON_OUTPUT_FILE = os.path.join(BASE_DIR, 'news_data.json')
    35 HTML_TEMPLATE_FILE = os.path.join(BASE_DIR, 'index.html')
    36 HTML_OUTPUT_FILE = os.path.join(BASE_DIR, 'index.html')
    37 JS_TEMPLATE_FILE = os.path.join(BASE_DIR, 'static', 'js', 'main.js')
    38 JS_OUTPUT_FILE = os.path.join(BASE_DIR, 'static', 'js', 'github_pages_main.js')
    39 
    40 # --- 공통 전처리 함수 및 객체 ---
    41 okt = Okt()
    42 def preprocess(text):
    43     """ 텍스트에서 명사만 추출하여 공백으로 구분된 문자열로 반환합니다. """
    44     return ' '.join(okt.nouns(re.sub(r'[^\ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(text))))
    45 
    46 # --- 1단계: 동적 크롤링, 중복 제거 및 전처리 ---
    47 
    48 def get_today_articles():
    49     """ 5대 일간지에서 오늘의 주요 기사 목록(제목, 링크, 신문사)을 수집합니다. """
    50     print("--- 1.1: 주요 일간지 기사 목록 수집 시작 ---")
    51     newspapers = {
    52         '한국경제': '015', '매일경제': '009', '동아일보': '020',
    53         '조선일보': '023', '중앙일보': '025',
    54     }
    55     headers = {'User-Agent': 'Mozilla/5.0'}
    56     all_articles = []
    57     for name, oid in newspapers.items():
    58         try:
    59             url = f"https://media.naver.com/press/{oid}/newspaper"
    60             response = requests.get(url, headers=headers, verify=False, timeout=10)
    61             response.raise_for_status()
    62             soup = BeautifulSoup(response.text, 'html.parser')
    63             links = soup.select('div.sc_offc_lst._paper_article_list a')
    64             for link in links:
    65                 title = link.get_text(strip=True)
    66                 href = link.get('href', '')
    67                 if title and href:
    68                     all_articles.append({
    69                         '신문사': name, '제목': title,
    70                         '링크': 'https://media.naver.com' + href if href.startswith('/') else href
    71                     })
    72             time.sleep(0.5)
    73         except Exception as e:
    74             print(f"  ✗ {name} 수집 오류: {e}")
    75     df = pd.DataFrame(all_articles).drop_duplicates(subset=['링크']).reset_index(drop=True)
    76     print(f"  - 총 {len(df)}개의 고유 기사 목록 수집 완료.")
    77     return df
    78 
    79 def get_article_content(url):
    80     """ 네이버 뉴스 링크에서 본문을 크롤링합니다. """
    81     try:
    82         headers = {'User-Agent': 'Mozilla/5.0'}
    83         response = requests.get(url, headers=headers, timeout=10, verify=False)
    84         response.raise_for_status()
    85         soup = BeautifulSoup(response.text, 'html.parser')
    86         content_area = soup.select_one('#dic_area, #articeBody, #article_content')
    87         if content_area:
    88             for el in content_area.select('script, style, .reporter_area, .ad_area, .promotion_area, div.byline, a, span.end_photo_org'):
    89                 el.decompose()
    90             return content_area.get_text(strip=True)
    91     except Exception as e:
    92         print(f"    - 링크 크롤링 실패: {url}, 오류: {e}")
    93     return ""
    94 
    95 def run_step_one_crawling_and_preprocessing():
    96     """ 크롤링, 요약, 중복 제거, 전처리를 통합 실행합니다. """
    97     articles_df = get_today_articles()
    98     if articles_df.empty: return pd.DataFrame()
    99 
   100     print("--- 1.2: 기사 본문 크롤링 및 요약 생성 ---")
   101     crawled_data = []
   102     for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="  - 기사 본문 크롤링"):
   103         content = get_article_content(row['링크'])
   104         if content:
   105             crawled_data.append([
   106                 row['신문사'], row['제목'], row['링크'], content, content[:400] + "..."
   107             ])
   108         time.sleep(0.5)
   109     
   110     if not crawled_data:
   111         print("크롤링된 기사가 없습니다.")
   112         return pd.DataFrame()
   113 
   114     df = pd.DataFrame(crawled_data, columns=['신문사', '제목', '링크', '본문', '본문_요약'])
   115 
   116     print("--- 1.3: 내용 기반 중복 기사 제거 (유사도 0.6 기준) ---")
   117     if not df.empty and '본문' in df.columns and not df['본문'].isnull().all():
   118         df['processed_text_for_dedup'] = df['본문'].apply(preprocess)
   119         vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
   120         tfidf_matrix = vectorizer.fit_transform(df['processed_text_for_dedup'])
   121         cosine_sim = cosine_similarity(tfidf_matrix)
   122         
   123         df['본문_길이_dedup'] = df['본문'].str.len()
   124         df = df.sort_values(by='본문_길이_dedup', ascending=False).reset_index(drop=True)
   125         
   126         tfidf_matrix_sorted = vectorizer.fit_transform(df['processed_text_for_dedup'])
   127         cosine_sim_sorted = cosine_similarity(tfidf_matrix_sorted)
   128         
   129         to_drop = set()
   130         for i in range(len(cosine_sim_sorted)):
   131             if i in to_drop: continue
   132             for j in range(i + 1, len(cosine_sim_sorted)):
   133                 if j in to_drop: continue
   134                 if cosine_sim_sorted[i, j] > 0.6:
   135                     to_drop.add(j)
   136         
   137         if to_drop:
   138             print(f"   - {len(to_drop)}개의 유사 기사를 제거했습니다.")
   139             df.drop(index=list(to_drop), inplace=True)
   140         
   141         df.drop(columns=['processed_text_for_dedup', '본문_길이_dedup'], inplace=True)
   142 
   143     print(f"   - 최종 분석 대상 기사: {len(df)}개")
   144     print("--- 1단계: 크롤링 및 전처리 완료 ---")
   145     print()
   146     return df
   147 
   148 # --- 2단계: 스크랩 가치 예측 ---
   149 
   150 def run_step_two_prediction(df):
   151     """ 학습된 모델을 사용하여 스크랩 가치를 예측하고 최종 결과를 CSV로 저장합니다. """
   152     if df.empty:
   153         print("2단계 실패: 분석할 데이터가 없습니다.")
   154         return False
   155         
   156     print("--- 2.1: 모델 및 Vectorizer 로드 ---")
   157     try:
   158         with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
   159         with open(VECTORIZER_PATH, "rb") as f: tfidf_vectorizer = pickle.load(f)
   160     except FileNotFoundError:
   161         print(f"오류: 모델 파일({MODEL_PATH}) 또는 Vectorizer 파일({VECTORIZER_PATH})을 찾을 수 없습니다.")
   162         return False
   163 
   164     print("--- 2.2: 텍스트 전처리 및 특성 추출 ---")
   165     keywords = {
   166         '업계': ['식품', '화학', '바이오', '패키징', '플라스틱', '항암제', '배터리', '친환경', 'D램', '삼양', '초순수', '제약사', '숙취', '상쾌환', '설탕', '칼로리', '삼양그룹', '삼양사', '삼양패키징', '삼양엔씨켐', '삼양바이오팜'],
   167         '경영': ['경영', '경제', '환율', 'M&A', '인수', '투자', '실적', '한일경제협회', '조직문화', '무역']
   168     }
   169 
   170     df['processed_title'] = df['제목'].apply(preprocess)
   171     df['processed_text'] = df.apply(lambda r: preprocess(r['제목'] + " " + r['본문']), axis=1)
   172     
   173     df['업계_키워드_개수'] = df['processed_text'].apply(lambda x: sum(k in x for k in keywords['업계']))
   174     df['경영_키워드_개수'] = df['processed_text'].apply(lambda x: sum(k in x for k in keywords['경영']))
   175     df['업계_키워드_제목_개수'] = df['processed_title'].apply(lambda x: sum(k in x for k in keywords['업계']))
   176     df['경영_키워드_제목_개수'] = df['processed_title'].apply(lambda x: sum(k in x for k in keywords['경영']))
   177     df['본문_길이'] = df['본문'].str.len().replace(0, 1)
   178     df['업계_키워드_밀도'] = df['업계_키워드_개수'] / df['본문_길이']
   179     df['경영_키워드_밀도'] = df['경영_키워드_개수'] / df['본문_길이']
   180 
   181     text_features = tfidf_vectorizer.transform(df['processed_text'])
   182     metadata_features = df[['업계_키워드_개수', '경영_키워드_개수', '업계_키워드_제목_개수', '경영_키워드_제목_개수', '본문_길이', '업계_키워드_밀도', '경영_키워드_밀도']].values
   183     
   184     X_new = hstack([text_features, metadata_features * 0.5]).tocsr()
   185 
   186     print("--- 2.3: 스크랩 가치 점수 예측 및 결과 저장 ---")
   187     df['예측점수'] = model.predict_proba(X_new)[:, 1]
   188     
   189     df['카테고리'] = '기타'
   190     df.loc[df['업계_키워드_개수'] > 0, '카테고리'] = '업계'
   191     df.loc[(df['업계_키워드_개수'] == 0) & (df['경영_키워드_개수'] > 0), '카테고리'] = '경영'
   192 
   193     # 최종선택여부 컬럼 추가 (사용자 입력을 위해 빈 값으로 초기화)
   194     df['최종선택여부'] = ''
   195 
   196     sorted_df = df.sort_values(by='예측점수', ascending=False)
   197     
   198     # '본문' 컬럼 제외하고 저장
   199     output_columns = ['신문사', '제목', '본문_요약', '링크', '카테고리', '예측점수', '최종선택여부']
   200     final_df = sorted_df[output_columns]
   201     
   202     final_df.to_csv(SCRAPPED_NEWS_TODAY_CSV, index=False, encoding='utf-8-sig')
   203     print(f"  - 최종 결과가 '{SCRAPPED_NEWS_TODAY_CSV}' 파일로 저장되었습니다.")
   204     print("--- 2단계: 스크랩 가치 예측 완료 ---")
   205     print()
   206     return True
   207 
   208 # --- 3단계: 정적 파일 생성 ---
   209 def generate_static_files():
   210     print("--- 3단계: GitHub Pages용 정적 파일 생성 시작 ---")
   211     
   212     if not os.path.exists(SCRAPPED_NEWS_TODAY_CSV) or os.path.getsize(SCRAPPED_NEWS_TODAY_CSV) == 0:
   213         print(f"  오류: '{SCRAPPED_NEWS_TODAY_CSV}' 파일이 비어있거나 존재하지 않습니다.")
   214         return False
   215         
   216     try:
   217         print("  [1/3] JSON 파일 생성 중...")
   218         df = pd.read_csv(SCRAPPED_NEWS_TODAY_CSV, encoding='utf-8-sig')
   219         df.to_json(JSON_OUTPUT_FILE, orient='records', force_ascii=False, indent=4)
   220         print(f"    '{JSON_OUTPUT_FILE}' 생성 완료.")
   221     except Exception as e:
   222         print(f"  오류: JSON 변환 실패 - {e}")
   223         return False
   224 
   225     # HTML 파일은 이미 올바른 JS를 참조하므로 별도 처리하지 않음.
   226     print("  [2/3] HTML 파일 처리 건너뜀 (index.html이 이미 올바른 스크립트를 참조).")
   227 
   228     try:
   229         print("  [3/3] JavaScript 파일 생성 중...")
   230         # JS 출력 디렉토리 생성
   231         os.makedirs(os.path.dirname(JS_OUTPUT_FILE), exist_ok=True)
   232         with open(JS_TEMPLATE_FILE, 'r', encoding='utf-8') as f: js_content = f.read()
   233         modified_js = js_content.replace("fetch('/api/news')", "fetch('news_data.json')")
   234         modified_js = modified_js.replace("setInterval(fetchNews, 30000);", "/* 자동 새로고침 비활성화 */")
   235         with open(JS_OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(modified_js)
   236         print(f"    '{JS_OUTPUT_FILE}' 생성 완료.")
   237     except FileNotFoundError:
   238         print(f"  오류: JS 템플릿 파일 '{JS_TEMPLATE_FILE}'을 찾을 수 없습니다.")
   239         return False
   240     except Exception as e:
   241         print(f"  오류: JavaScript 파일 생성 실패 - {e}")
   242         return False
   243 
   244     print("--- 3단계: 정적 파일 생성 완료 ---")
   245     print()
   246     return True
   247 
   248 # --- 메인 실행 블록 ---
   249 if __name__ == "__main__":
   250     # 1단계 실행
   251     processed_articles_df = run_step_one_crawling_and_preprocessing()
   252     
   253     # 2단계 실행
   254     if not processed_articles_df.empty:
   255         prediction_success = run_step_two_prediction(processed_articles_df)
   256         
   257         # 3단계 실행
   258         if prediction_success:
   259             if generate_static_files():
   260                 print("🎉 모든 작업이 성공적으로 완료되었습니다.")
   261             else:
   262                 print("!!!!! 3단계(정적 파일 생성)에서 오류가 발생했습니다. !!!!!")
   263         else:
   264             print("!!!!! 2단계(스크랩 예측)에서 오류가 발생하여 중단되었습니다. !!!!!")
   265     else:
   266         print("!!!!! 1단계(크롤링 및 전처리)에서 처리할 기사를 찾지 못했습니다. !!!!!")
