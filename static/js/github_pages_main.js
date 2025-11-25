// 전역 변수로 현재 로드된 뉴스 데이터를 저장 (CSV 다운로드용)
let currentNewsData = [];

// DOM 요소 가져오기
const newsListContainer = document.getElementById('news-list');
const scrapListContainer = document.getElementById('scrap-list');

// 페이지 로드 시 초기 데이터 로드 및 렌더링
document.addEventListener('DOMContentLoaded', () => {
    fetchNews();
    renderScrapList();
    
    // [추가] CSV 다운로드 버튼 이벤트 리스너 연결 (HTML에 버튼이 있다고 가정)
    const downloadBtn = document.getElementById('download-csv-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadTrainingData);
    }
});

/**
 * 서버로부터 최신 뉴스 데이터를 가져와 화면에 렌더링합니다.
 */
async function fetchNews() {
    try {
        // github pages용 주소 (로컬 테스트시 main.js에서는 '/api/news' 사용)
        const response = await fetch('news_data.json'); 
        
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
        currentNewsData = newsItems; // [중요] CSV 생성을 위해 데이터 전역 변수에 저장

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

        // 2. 데이터 객체 첨부
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
 * [기능 2] 본문 접기/펼치기 기능이 적용된 카드 생성
 */
function createArticleCard(item, index) {
    const scraps = getScraps();
    const uniqueIdForLogic = index;
    const isScrapped = scraps.some(scrap => scrap.uniqueIdForLogic === uniqueIdForLogic);
    
    const displayUrl = item.링크 || '#';
    const displayTitle = item.제목 || '제목 없음';
    // 본문 내용이 없으면 요약이라도 보여줌
    const displayBody = item.본문 || item.본문_요약 || '내용이 없습니다.'; 

    return `
        <div class="card mb-3">
            <div class="card-body">
                <h5 class="card-title">
                    <a href="${displayUrl}" target="_blank" class="text-decoration-none text-dark">${displayTitle}</a>
                </h5>
                <h6 class="card-subtitle mb-3 text-muted" style="font-size: 0.9em;">
                    ${item.신문사 || '언론사 정보 없음'}
                </h6>

                <div id="content-area-${index}" class="mb-3" style="display: none;">
                    <p class="card-text text-secondary" style="font-size: 0.95rem; line-height: 1.6;">
                        ${displayBody}
                    </p>
                </div>

                <div class="d-flex justify-content-between align-items-center">
                    <button class="btn btn-sm btn-outline-primary" onclick="toggleContent(${index})" id="btn-toggle-${index}">
                        본문 보기 ▼
                    </button>

                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" role="switch"
                               id="scrap-${uniqueIdForLogic}" 
                               ${isScrapped ? 'checked' : ''}
                               onchange="toggleScrap(this)" style="cursor: pointer;">
                        <label class="form-check-label fw-bold" for="scrap-${uniqueIdForLogic}" style="cursor: pointer;">
                            학습 데이터로 선택
                        </label>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * [기능 2] 본문 접기/펼치기 토글 함수
 */
function toggleContent(index) {
    const contentArea = document.getElementById(`content-area-${index}`);
    const btn = document.getElementById(`btn-toggle-${index}`);
    
    if (contentArea.style.display === 'none') {
        contentArea.style.display = 'block';
        btn.innerText = '본문 접기 ▲';
        btn.classList.replace('btn-outline-primary', 'btn-outline-secondary');
    } else {
        contentArea.style.display = 'none';
        btn.innerText = '본문 보기 ▼';
        btn.classList.replace('btn-outline-secondary', 'btn-outline-primary');
    }
}

/**
 * [기능 1] 현재 뉴스 목록을 학습용 CSV 파일로 다운로드
 */
function downloadTrainingData() {
    if (currentNewsData.length === 0) {
        alert("다운로드할 뉴스 데이터가 없습니다.");
        return;
    }

    // 현재 스크랩(선택)된 항목들의 ID(uniqueIdForLogic 기준이 아닌 링크 기준 매칭 권장) 리스트 가져오기
    // 하지만 여기선 화면 순서(index)와 currentNewsData 순서가 같으므로 index로 체크박스 상태 확인
    
    // CSV 헤더 (파이썬 코드와 일치)
    let csvContent = "크롤링된_제목,크롤링된_본문,최종선택여부\n";

    currentNewsData.forEach((item, index) => {
        // 체크박스 상태 확인
        const checkbox = document.getElementById(`scrap-${index}`);
        const isSelected = checkbox && checkbox.checked ? 1 : 0;

        // CSV 포맷에 맞게 데이터 정제 (따옴표, 줄바꿈 처리)
        // 본문이 없으면 본문_요약 사용
        const title = (item.제목 || "").replace(/"/g, '""'); 
        const body = (item.본문 || item.본문_요약 || "").replace(/"/g, '""');

        // CSV 행 추가 (Excel 호환을 위해 값들을 따옴표로 감쌈)
        csvContent += `"${title}","${body}",${isSelected}\n`;
    });

    // BOM 추가 (엑셀에서 한글 깨짐 방지)
    const bom = "\uFEFF";
    const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
    
    // 다운로드 링크 생성 및 클릭
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", "labeled_for_training.csv");
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// --- 기존 로직 유지 ---

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

function renderScrapList() {
    const scraps = getScraps();
    if (scraps.length === 0) {
        scrapListContainer.innerHTML = '<p class="text-muted p-2">선택한 기사가 없습니다.</p>';
        return;
    }
    const scrapHTML = scraps.map(scrap => {
        const displayUrl = scrap.링크 || '#';
        const displayTitle = scrap.제목 || '제목 없음';
        return `<div class="list-group-item list-group-item-action py-2">
            <a href="${displayUrl}" target="_blank" class="text-decoration-none small text-truncate d-block">${displayTitle}</a>
         </div>`;
    }).join('');
    scrapListContainer.innerHTML = `<div class="list-group list-group-flush">${scrapHTML}</div>`;
}

function getScraps() {
    return JSON.parse(localStorage.getItem('scrappedNews') || '[]');
}

function saveScraps(scraps) {
    localStorage.setItem('scrappedNews', JSON.stringify(scraps));
}
