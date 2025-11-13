// C:\Users\syc217052\Documents\ai_inov\scrap\static\js\main.js

// DOM 요소 가져오기
const newsListContainer = document.getElementById('news-list');
const scrapListContainer = document.getElementById('scrap-list');

// 페이지 로드 시 초기 데이터 로드 및 렌더링
document.addEventListener('DOMContentLoaded', () => {
    fetchNews();
    renderScrapList();
    // 30초마다 자동으로 뉴스 업데이트
    setInterval(fetchNews, 30000);
});

/**
 * 서버로부터 최신 뉴스 데이터를 가져와 화면에 렌더링합니다.
 */
async function fetchNews() {
    try {
        const response = await fetch('/api/news');
        if (!response.ok) {
            // 서버가 에러 응답을 보냈을 때, 상세 내용을 표시
            let errorText = `HTTP error! status: ${response.status}`;
            try {
                // 에러 내용이 JSON 형태일 경우, error 메시지를 추출
                const errorData = await response.json();
                errorText = errorData.error || JSON.stringify(errorData);
            } catch (e) {
                // 에러 내용이 JSON이 아닐 경우(예: HTML 에러 페이지), 텍스트 그대로를 사용
                errorText = await response.text();
            }
            newsListContainer.innerHTML = `<div class="alert alert-danger"><strong>데이터 로딩 실패:</strong><pre>${errorText}</pre></div>`;
            return;
        }
        const newsItems = await response.json();
        
        // 1. HTML 렌더링
        newsListContainer.innerHTML = newsItems.map((item, index) => createArticleCard(item, index)).join('');

        // 2. 렌더링된 각 요소에 데이터 객체를 직접 첨부 (문자열 변환 회피)
        newsItems.forEach((item, index) => {
            const checkbox = document.getElementById(`scrap-${index}`);
            if (checkbox) {
                const itemWithId = {...item, uniqueIdForLogic: index};
                checkbox.itemData = itemWithId; // 요소의 프로퍼티로 데이터 객체를 직접 저장
            }
        });

    } catch (error) {
        // 네트워크 오류 등 fetch 자체가 실패했을 때
        console.error('뉴스 로딩 중 오류 발생:', error);
        newsListContainer.innerHTML = `<div class="alert alert-danger"><strong>뉴스를 불러오는 데 실패했습니다:</strong><br>${error.toString()}</div>`;
    }
}

/**
 * localStorage에 저장된 스크랩 목록을 화면에 렌더링합니다.
 */
function renderScrapList() {
    const scraps = getScraps();
    if (scraps.length === 0) {
        scrapListContainer.innerHTML = '<p class="text-muted">스크랩한 기사가 없습니다.</p>';
        return;
    }
    const scrapHTML = scraps.map(scrap => {
        const displayUrl = scrap.url || '#';
        const displayTitle = scrap.title || '제목 없음';
        return `<div class="list-group-item list-group-item-action">
            <a href="${displayUrl}" target="_blank" class="text-decoration-none">${displayTitle}</a>
         </div>`;
    }).join('');
    scrapListContainer.innerHTML = `<div class="list-group">${scrapHTML}</div>`;
}

/**
 * 개별 뉴스 기사 카드를 생성합니다.
 * @param {object} item - 뉴스 기사 데이터
 * @returns {string} - HTML 카드 문자열
 */
function createArticleCard(item, index) {
    const scraps = getScraps();
    const uniqueIdForLogic = index;
    const isScrapped = scraps.some(scrap => scrap.uniqueIdForLogic === uniqueIdForLogic);
    
    const displayUrl = item.url || '#';
    const displayTitle = item.title || '제목 없음';
    const displaySummary = item.summary || '요약 정보가 없습니다.';

    return `
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">
                    <a href="${displayUrl}" target="_blank">${displayTitle}</a>
                </h5>
                <h6 class="card-subtitle mb-2 text-muted">${item.press || '언론사 정보 없음'}</h6>
                <p class="card-text">${displaySummary}</p>
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" value="" 
                           id="scrap-${uniqueIdForLogic}" 
                           ${isScrapped ? 'checked' : ''}
                           onchange="toggleScrap(this)">
                    <label class="form-check-label" for="scrap-${uniqueIdForLogic}">
                        스크랩하기
                    </label>
                </div>
            </div>
        </div>
    `;
}

/**
 * 체크박스 상태에 따라 스크랩 목록을 토글합니다.
 * @param {HTMLElement} checkbox - 클릭된 체크박스 요소
 */
function toggleScrap(checkbox) {
    const item = checkbox.itemData; // data-item 속성 대신, 요소의 프로퍼티에서 직접 데이터 객체를 가져옵니다.
    const scraps = getScraps();
    // 내용 기반이 아닌, 절대적인 고유 ID로 스크랩 목록에서 항목을 찾습니다.
    const existingIndex = scraps.findIndex(scrap => scrap.uniqueIdForLogic === item.uniqueIdForLogic);

    if (checkbox.checked && existingIndex === -1) {
        // 스크랩 추가
        scraps.push(item);
    } else if (!checkbox.checked && existingIndex > -1) {
        // 스크랩 제거
        scraps.splice(existingIndex, 1);
    }

    saveScraps(scraps);
    renderScrapList();
}

/**
 * localStorage에서 스크랩 목록을 가져옵니다.
 * @returns {Array} - 스크랩된 기사 객체 배열
 */
function getScraps() {
    return JSON.parse(localStorage.getItem('scrappedNews') || '[]');
}

/**
 * 스크랩 목록을 localStorage에 저장합니다.
 * @param {Array} scraps - 저장할 스크랩 객체 배열
 */
function saveScraps(scraps) {
    localStorage.setItem('scrappedNews', JSON.stringify(scraps));
}
