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
    
    // 고유 ID 생성
    const collapseId = `collapseSummary-${index}`;

    // [수정] onclick 이벤트를 사용하여 직접 함수를 호출하도록 변경
    return `
        <div class="card mb-3">
            <div class="card-body">
                <h5 class="card-title">
                    <a href="${displayUrl}" target="_blank">${displayTitle}</a>
                </h5>
                <h6 class="card-subtitle mb-2 text-muted">${item.신문사 || '언론사 정보 없음'}</h6>
                
                <div class="mb-2">
                    <button class="btn btn-sm btn-outline-secondary" type="button" 
                            onclick="toggleSummary('${collapseId}')">
                        📄 본문 요약 보기
                    </button>
                </div>

                <div class="collapse mb-3" id="${collapseId}">
                    <div class="card card-body bg-light border-0">
                        ${displaySummary}
                    </div>
                </div>

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
 * [추가됨] 요약 내용을 직접 토글하는 함수
 * Bootstrap 객체가 있으면 애니메이션을 사용하고, 없으면 클래스만 조작합니다.
 */
function toggleSummary(id) {
    const element = document.getElementById(id);
    if (!element) return;

    // window.bootstrap이 로드되어 있다면 (일반적인 경우)
    if (window.bootstrap) {
        const bsCollapse = bootstrap.Collapse.getOrCreateInstance(element);
        bsCollapse.toggle();
    } else {
        // 만약 Bootstrap JS가 로드되지 않았다면 강제로 클래스 토글 (안전장치)
        if (element.classList.contains('show')) {
            element.classList.remove('show');
        } else {
            element.classList.add('show');
        }
    }
}

/**
 * 스크랩 토글 및 저장 관련 함수들
 */
function toggleScrap(checkbox) {
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
