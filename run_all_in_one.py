"""
All-in-one pipeline to:
- Collect today's major newspaper article links directly from Naver Media pages
- Crawl full article contents
- Score news with pre-trained model (TF-IDF + LightGBM)
- Produce CSV and JSON for GitHub Pages (keys: title, url, summary, press)

Refactored to remove dependency on newspapers.csv and align logic with
run_final_scraping.py while keeping outputs compatible with static/js/main.js.
"""

import os
import re
import io
import sys
import time
import json
import pickle
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Optional: lightgbm imported for model compatibility (pickle contains LGBM)
import lightgbm as lgb  # noqa: F401

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Paths (relative to repo/scrap)
CRAWLED_ARTICLES_CSV = "crawled_articles.csv"
SCRAPPED_NEWS_TODAY_CSV = "scrapped_news_today.csv"
MODEL_PATH = "scrap_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
JSON_OUTPUT_FILE = "news_data.json"
HTML_TEMPLATE_FILE = os.path.join("templates", "index.html")
HTML_OUTPUT_FILE = "github_pages_index.html"
JS_TEMPLATE_FILE = os.path.join("static", "js", "main.js")
JS_OUTPUT_FILE = os.path.join("static", "js", "github_pages_main.js")


# ----- Utilities -----
okt = Okt()


def preprocess(text: str) -> str:
    # Keep Hangul characters only; extract nouns via Okt
    return " ".join(okt.nouns(re.sub(r"[^\uAC00-\uD7A3\s]", "", str(text))))


def get_today_articles() -> pd.DataFrame:
    """Collect today's headline lists from selected newspapers on Naver Media.

    Returns a DataFrame with columns: press, title, url
    """
    print("1단계 1/2: 주요 신문사 목록 수집 중...")

    newspapers = {
        "한국경제": "015",
        "매일경제": "009",
        "동아일보": "020",
        "조선일보": "023",
        "중앙일보": "025",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0 Safari/537.36"
        )
    }

    items = []
    for press, oid in newspapers.items():
        try:
            url = f"https://media.naver.com/press/{oid}/newspaper"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            container = soup.find("div", class_="sc_offc_lst _paper_article_list")
            if not container:
                continue

            for a in container.find_all("a"):
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not title or not href:
                    continue
                if href.startswith("/"):
                    href = "https://media.naver.com" + href
                items.append({"press": press, "title": title, "url": href})
            time.sleep(0.2)
        except Exception as e:
            print(f"  - {press}: 수집 오류 - {e}")

    df = pd.DataFrame(items).drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"  -> {len(df)}건 수집 완료")
    return df


def fetch_article_content(url: str) -> str:
    """Fetch article body text from a Naver news page (best-effort)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        body = soup.select_one("#dic_area") or soup.select_one("#articeBody") or soup.select_one("#article_content")
        if not body:
            return ""
        for el in body.select("script, style, .reporter_area, .ad_area, .promotion_area, div.byline, a, span.end_photo_org"):
            el.decompose()
        return body.get_text(strip=True)
    except Exception as e:
        print(f"  - 본문 수집 실패: {url} ({e})")
        return ""


# ----- Step 1: Crawl list and contents -----
def run_crawling(output_csv_path: str = CRAWLED_ARTICLES_CSV) -> bool:
    print("--- 1단계: 뉴스 크롤링 시작 ---")

    list_df = get_today_articles()
    if list_df.empty:
        print("  - 수집된 기사 목록이 없습니다.")
        return False

    crawled = []
    for i, row in list_df.iterrows():
        url = row["url"]
        title = row["title"]
        press = row["press"]
        print(f"  [{i+1}/{len(list_df)}] 본문 수집: {title}")
        content = fetch_article_content(url)
        if not content:
            continue
        crawled.append({
            "press": press,
            "title": title,
            "url": url,
            "content": content,
        })
        time.sleep(0.2)

    if not crawled:
        print("  - 본문이 수집되지 않았습니다.")
        return False

    pd.DataFrame(crawled).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"  -> '{output_csv_path}' 저장 완료")
    print("--- 1단계 완료 ---\n")
    return True


# ----- Step 2: Predict value score -----
def run_daily_scraping(
    input_file: str = CRAWLED_ARTICLES_CSV,
    model_path: str = MODEL_PATH,
    vectorizer_path: str = VECTORIZER_PATH,
    output_file: str = SCRAPPED_NEWS_TODAY_CSV,
) -> bool:
    print("--- 2단계: 가치 점수 예측 시작 ---")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        df = pd.read_csv(input_file, encoding="utf-8")
        df.dropna(subset=["title", "content"], inplace=True)
    except Exception as e:
        print(f"  - 입력/모델 로딩 실패: {e}")
        return False

    df["processed_text"] = df.apply(lambda r: preprocess(f"{r['title']} {r['content']}"), axis=1)

    keywords = {
        "bio": ["의약", "바이오", "신약", "제약", "헬스"],
        "biz": ["경영", "경제", "수익", "M&A", "투자"],
    }
    df["bio_kw_cnt"] = df["processed_text"].apply(lambda x: sum(k in x for k in keywords["bio"]))
    df["biz_kw_cnt"] = df["processed_text"].apply(lambda x: sum(k in x for k in keywords["biz"]))
    df["content_len"] = df["content"].str.len().fillna(0).astype(int)

    text_features = tfidf_vectorizer.transform(df["processed_text"])
    meta = df[["bio_kw_cnt", "biz_kw_cnt", "content_len"]].astype(np.float32).values
    X_new = hstack([text_features, meta])
    df["pred_score"] = model.predict_proba(X_new)[:, 1]

    # Simple categorization
    df["category"] = "일반"
    df.loc[df["bio_kw_cnt"] > 0, "category"] = "의/생명"
    df.loc[(df["bio_kw_cnt"] == 0) & (df["biz_kw_cnt"] > 0), "category"] = "경영"

    # Summary (first 300 chars)
    df["summary"] = df["content"].astype(str).str.slice(0, 300) + "..."

    ordered = df.sort_values("pred_score", ascending=False)
    out_cols = [
        "press",
        "title",
        "summary",
        "url",
        "category",
        "pred_score",
    ]
    ordered[out_cols].to_csv(output_file, index=False, encoding="utf-8-sig")
    print("--- 2단계 완료 ---\n")
    return True


# ----- Step 3: Static files for GitHub Pages -----
def generate_static_files() -> bool:
    print("--- 3단계: 정적 파일 생성 시작 ---")
    if not os.path.exists(SCRAPPED_NEWS_TODAY_CSV) or os.path.getsize(SCRAPPED_NEWS_TODAY_CSV) == 0:
        print(f"  - '{SCRAPPED_NEWS_TODAY_CSV}' 가 비었거나 없습니다.")
        return False

    # 1) JSON for GitHub Pages (keys expected by static/js/main.js)
    try:
        print("  [1/3] JSON 생성")
        df = pd.read_csv(SCRAPPED_NEWS_TODAY_CSV, encoding="utf-8-sig")
        df = df.replace({np.nan: None})

        # Ensure correct keys: title, url, summary, press
        json_records = [
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "summary": row.get("summary"),
                "press": row.get("press"),
            }
            for _, row in df.iterrows()
        ]
        with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(json_records, f, ensure_ascii=False, indent=2)
        print(f"    -> '{JSON_OUTPUT_FILE}' 생성 완료")
    except Exception as e:
        print(f"  - JSON 생성 실패: {e}")
        return False

    # 2) HTML (swap script path for GitHub Pages)
    try:
        print("  [2/3] HTML 생성")
        with open(HTML_TEMPLATE_FILE, "r", encoding="utf-8") as f:
            html_content = f.read()
        modified_html = html_content.replace(
            '<script src="/static/js/main.js"></script>',
            '<script src="static/js/github_pages_main.js"></script>',
        )
        with open(HTML_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(modified_html)
        print(f"    -> '{HTML_OUTPUT_FILE}' 생성 완료")
    except Exception as e:
        print(f"  - HTML 생성 실패: {e}")
        return False

    # 3) JS (switch fetch target and remove interval)
    try:
        print("  [3/3] JS 생성")
        with open(JS_TEMPLATE_FILE, "r", encoding="utf-8") as f:
            js_content = f.read()
        modified_js = js_content.replace("fetch('/api/news')", "fetch('news_data.json')")
        modified_js = modified_js.replace("setInterval(fetchNews, 30000);", "")
        with open(JS_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(modified_js)
        print(f"    -> '{JS_OUTPUT_FILE}' 생성 완료")
    except Exception as e:
        print(f"  - JS 생성 실패: {e}")
        return False

    print("--- 3단계 완료 ---\n")
    return True


if __name__ == "__main__":
    print("========== 통합 스크립트 실행 시작 ==========")
    if run_crawling():
        if run_daily_scraping():
            if generate_static_files():
                print("========== 모든 작업이 성공적으로 완료되었습니다 ==========")
            else:
                print("!!!!! 3단계(정적 파일 생성)에서 오류가 발생했습니다 !!!!!")
        else:
            print("!!!!! 2단계(가치 점수 예측)에서 오류가 발생했습니다 !!!!!")
    else:
        print("!!!!! 1단계(뉴스 크롤링)에서 오류가 발생했습니다 !!!!!")

