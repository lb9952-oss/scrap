// C:\Users\syc217052\Documents\ai_inov\scrap\static\js\main.js

// 1. 전역 함수로 선언하여 HTML 버튼에서 무조건 찾을 수 있게 함 (가장 중요)
window.toggleSummary = function(index) {
    const contentId = `summary-content-${index}`;
    const element = document.getElementById(contentId);
    
    // 요소가 없으면 종료
    if (!element) return;

    // Bootstrap의 'show' 클래스를 직접 토글 (CSS로 제어)
    // .collapse는 기본적으로 숨김, .collapse.show는 보임 상태입니다.
    if (element.classList.contains('show')) {
        element.classList.remove('show');
    } else {
        element.classList.add('show');
    }
};

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
            let errorText = `HTTP error! status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorText = errorData.error || JSON.stringify(errorData);
            } catch (e) {
                errorText = await response.text();
            }
            newsListContainer.innerHTML = `<div class="alert alert-danger"><strong>데이터 로딩 실패:</strong><pre>${errorText}</pre></div>`;
            return;
        }
        const newsItems = await response.json();

        // --- 스크랩 초기화 로직 ---
        if (newsItems.length > 0) {
            const lastNewsId = localStorage.getItem('lastNewsId');
            const currentNewsId = newsItems[0].링크; 

            if (lastNewsId !== currentNewsId) {
                console.log('새로운 뉴스 목록이 감지되었습니다. 스크랩 목록을 초기화합니다.');
                localStorage.removeItem('scrappedNews'); 
                localStorage.setItem('lastNewsId', currentNewsId); 
                renderScrapList(); 
            }
        }
        
        // 1. HTML 렌더링
        newsListContainer.innerHTML = newsItems.map((item, index) => createArticleCard(item, index)).join('');

        // 2. 렌더링된 각 요소에 데이터 객체를 직접 첨부
        newsItems.forEach((item, index) => {
            const checkbox = document.getElementById(`scrap-${index}`);
            if (checkbox) {
                const itemWithId = {...item, uniqueIdForLogic: index};
                checkbox.itemData = itemWithId; 
            }
        });

    } catch (error) {
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
        const displayUrl = scrap.링크 || '#';
        const displayTitle = scrap.제목 || '제목 없음';
        return `<div class="list-group-item list-group-item-action">
            <a href="${displayUrl}" target="_blank" class="text-decoration-none">${displayTitle}</a>
         </div>`;
    }).join('');
    scrapListContainer.innerHTML = `<div class="list-group">${scrapHTML}</div>`;
}

/**
 * 개별 뉴스 기사 카드를 생성합니다.
 */
function createArticleCard(item, index) {
    const scraps = getScraps();
    const uniqueIdForLogic = index;
    const isScrapped = scraps.some(scrap => scrap.uniqueIdForLogic === uniqueIdForLogic);
    
    const displayUrl = item.링크 || '#';
    const displayTitle = item.제목 || '제목 없음';
    const displaySummary = item.본문_요약 || '요약 정보가 없습니다.';
    
    // 단순한 ID 생성
    const contentId = `summary-content-${index}`;

    return `
        <div class="card mb-3">
            <div class="card-body">
                <h5 class="card-title">
                    <a href="${displayUrl}" target="_blank">${displayTitle}</a>
                </h5>
                <h6 class="card-subtitle mb-2 text-muted">${item.신문사 || '언론사 정보 없음'}</h6>
                
                <div class="mb-2">
                    <button class="btn btn-sm btn-outline-secondary" type="button" 
                            onclick="window.toggleSummary(${index})">
                        📄 본문 요약 보기
                    </button>
                </div>

                <div class="collapse" id="${contentId}">
                    <div class="card card-body bg-light border-0">
                        ${displaySummary}
                    </div>
                </div>

                <div class="form-check mt-2">
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
 * 스크랩 관련 함수들 (기존 유지)
 */
window.toggleScrap = function(checkbox) {
    const item = checkbox.itemData; 
    const scraps = getScraps();
    const existingIndex = scraps.findIndex(scrap => scrap.uniqueIdForLogic === item.uniqueIdForLogic);

    if (checkbox.checked && existingIndex === -1) {
        scraps.push(item);
    } else if (!checkbox.checked && existingIndex > -1) {
        scraps.splice(existingIndex, 1);
    }

    saveScraps(scraps);
    renderScrapList();
}

function getScraps() {
    return JSON.parse(localStorage.getItem('scrappedNews') || '[]');
}

function saveScraps(scraps) {
    localStorage.setItem('scrappedNews', JSON.stringify(scraps));
}
