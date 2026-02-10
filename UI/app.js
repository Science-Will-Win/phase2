// ============================================
// Inference Chat - Frontend Logic (DEMO MODE)
// ============================================

// ============================================
// DEMO MODE - Mock API Responses
// ============================================
const DEMO_MODE = true;
let mockConversationCounter = 1;

const MOCK_PLAN_RESPONSE = `I'll help you create a comprehensive plan for this task.

## Goal
Analyze the given requirements and provide a structured approach

## Proposed Steps

1. **Data Collection & Preprocessing**
   - Gather all relevant data sources
   - Clean and normalize the data

2. **Exploratory Analysis**
   - Identify patterns and trends
   - Generate statistical summaries

3. **Implementation**
   - Develop the core solution
   - Integrate with existing systems

4. **Validation & Testing**
   - Verify results accuracy
   - Perform edge case testing

5. **Documentation & Deployment**
   - Document the process
   - Deploy to production

This structured plan will ensure systematic progress towards your goal.`;

const MOCK_RESPONSES = [
    "I understand your question. Based on my analysis, here's what I can tell you:\n\nThe approach you're considering is valid. I would recommend starting with a clear definition of your objectives, then systematically working through each component.\n\nWould you like me to elaborate on any specific aspect?",
    "That's an interesting point. Let me explain further:\n\nFrom a technical perspective, there are several considerations to keep in mind:\n\n1. **Performance**: Ensure optimal resource utilization\n2. **Scalability**: Design for future growth\n3. **Maintainability**: Keep the code clean and well-documented\n\nLet me know if you need more details on any of these areas.",
    "Here's what I found regarding your query:\n\nThe solution involves multiple steps, each building upon the previous one. The key is to maintain consistency throughout the process while adapting to any challenges that arise.\n\n```python\n# Example code snippet\ndef process_data(input_data):\n    result = transform(input_data)\n    return validate(result)\n```\n\nFeel free to ask follow-up questions!",
    "Great question! Here's my analysis:\n\nConsidering the context you've provided, I suggest the following approach:\n\n- First, establish a baseline understanding\n- Then, iterate on the solution\n- Finally, validate and refine\n\nThis methodology has proven effective in similar scenarios.",
];

// Mock Plan Steps for demo
const MOCK_PLAN_STEPS = [
    { name: 'CRISPR 스크린 설계 및 설계도 작성', tool: 'search_papers', description: 'CRISPR 스크린 프로토콜 설계' },
    { name: 'CRISPR 스크린 모델 세포 준비', tool: 'retrieve_gene_info', description: '모델 세포 및 유전자 정보 수집' },
    { name: 'CRISPR 배양 및 세포 주입', tool: 'code_gen', description: '세포 배양 프로토콜 코드 생성' },
    { name: 'T세포 고갈 평가 실험 설계', tool: 'search_papers', description: '평가 실험 설계 및 분석' }
];

// Mock Step Results for demo
const MOCK_STEP_RESULTS = [
    { success: true, summary: 'CRISPR 스크린 프로토콜 관련 논문 5편 검색 완료' },
    { success: true, summary: '타겟 유전자 정보 및 세포주 데이터 수집 완료' },
    { success: true, summary: '세포 배양 프로토콜 코드 생성 완료' },
    { success: true, summary: 'T세포 고갈 평가 실험 설계 완료' }
];

// Mock Plan Analysis for demo
const MOCK_PLAN_ANALYSIS = `## 계획 분석

이 계획은 CRISPR 스크린 실험을 위한 4단계 프로세스입니다.

### 주요 단계
1. **설계 단계**: 프로토콜 설계 및 논문 검색
2. **준비 단계**: 모델 세포 및 유전자 정보 수집
3. **실행 단계**: 세포 배양 및 주입
4. **평가 단계**: T세포 고갈 평가 실험

### 예상 결과
모든 단계가 성공적으로 완료되면 CRISPR 스크린 실험을 위한 완전한 프로토콜이 준비됩니다.`;

// Mock Step Codes for demo
const MOCK_STEP_CODES = [
    {
        code: `# Step 1: CRISPR 스크린 프로토콜 검색
import requests

def search_crispr_papers():
    """CRISPR 스크린 관련 논문 검색"""
    query = "CRISPR screen protocol T cell"
    # PubMed API 호출
    results = fetch_pubmed(query, limit=5)
    return results

papers = search_crispr_papers()
print(f"Found {len(papers)} papers")`,
        language: 'python'
    },
    {
        code: `# Step 2: 유전자 정보 수집
from biomart import BiomartServer

def get_gene_info(gene_symbols):
    """타겟 유전자 정보 조회"""
    server = BiomartServer("http://www.ensembl.org/biomart")
    dataset = server.datasets['hsapiens_gene_ensembl']
    return dataset.search({'filters': {'hgnc_symbol': gene_symbols}})

genes = get_gene_info(['CD8A', 'PDCD1', 'LAG3'])`,
        language: 'python'
    },
    {
        code: `# Step 3: 세포 배양 프로토콜
protocol = {
    "cell_line": "Jurkat T cells",
    "medium": "RPMI 1640 + 10% FBS",
    "seeding_density": "5e5 cells/ml",
    "transfection_method": "electroporation",
    "cas9_concentration": "1 ug/ml"
}

def prepare_cells(protocol):
    print(f"Preparing {protocol['cell_line']}")
    return protocol`,
        language: 'python'
    },
    {
        code: `# Step 4: T세포 고갈 평가
import pandas as pd

def evaluate_exhaustion(samples):
    """T세포 고갈 마커 평가"""
    markers = ['PD1', 'TIM3', 'LAG3', 'CTLA4']
    results = pd.DataFrame()
    for marker in markers:
        results[marker] = analyze_expression(samples, marker)
    return results

exhaustion_score = evaluate_exhaustion(samples)`,
        language: 'python'
    }
];

// Mock Step Outputs for demo
const MOCK_STEP_OUTPUTS = [
    {
        papers: [
            { title: "CRISPR Screen for T Cell Exhaustion", journal: "Nature", year: 2024 },
            { title: "Genome-wide CRISPR Analysis", journal: "Cell", year: 2023 }
        ],
        summary: "5편의 관련 논문 검색 완료"
    },
    {
        genes: [
            { symbol: "CD8A", name: "T-cell surface glycoprotein CD8 alpha chain" },
            { symbol: "PDCD1", name: "Programmed cell death protein 1" }
        ],
        summary: "타겟 유전자 3개 정보 수집"
    },
    {
        protocol: { cell_line: "Jurkat", method: "Electroporation" },
        summary: "세포 배양 프로토콜 생성 완료"
    },
    {
        markers: ["PD1", "TIM3", "LAG3"],
        summary: "T세포 고갈 평가 설계 완료"
    }
];

// ============================================
// State
// ============================================

// State
let currentConversationId = null;
let conversations = [];
let isStreaming = false;
let sidebarCollapsed = false;
let currentAbortController = null;
let currentMessages = [];
let editingIndex = -1;  // -1 = 수정 모드 아님
let pendingFiles = [];  // 첨부 파일 배열 { type: 'image'|'audio', name, data }
let scrollLockUntil = 0;  // 스크롤 잠금 해제 시간 (timestamp)
let currentStepQuestion = null;  // Step 질문 컨텍스트 { stepNum, tool, stepName, context, previousSteps }

// Detail Panel State
let detailPanelOpen = false;
let detailPanelWidth = 400;  // Default width
let detailPanelData = {
    goal: '',
    steps: [],
    results: [],      // tool_result 누적
    codes: {},        // Step별 코드 { stepIndex: { language, code, task } }
    summaryCode: null, // 종합 코드
    analysis: '',     // analyze_plan 결과
    currentStep: 0
};
let currentCodeStep = 'all';  // 현재 Code 탭에서 선택된 Step (기본값: all)

// DOM Elements
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const newChatBtn = document.getElementById('newChatBtn');
const conversationsList = document.getElementById('conversationsList');
const modelName = document.getElementById('modelName');
const messagesContainer = document.getElementById('messagesContainer');
const messagesWrapper = document.getElementById('messagesWrapper');
const welcomeMessage = document.getElementById('welcomeMessage');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const sendIcon = document.getElementById('sendIcon');
const stopIcon = document.getElementById('stopIcon');
const loadingOverlay = document.getElementById('loadingOverlay');
const mainContent = document.querySelector('.main-content');
const scrollToBottomBtn = document.getElementById('scrollToBottomBtn');

// Detail Panel DOM Elements
const detailPanel = document.getElementById('detailPanel');
const detailToggle = document.getElementById('detailToggle');
const detailClose = document.getElementById('detailClose');
const detailResizeHandle = document.getElementById('detailResizeHandle');
const chatArea = document.getElementById('chatArea');

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Apply saved theme
    setTheme(getTheme());
    
    // Apply saved background image, blur, and opacity
    applyBgImage();
    applyBgBlur();
    applyBgOpacity();
    
    // Load sidebar state from localStorage (transition 없이 초기 적용)
    const savedSidebarState = localStorage.getItem('sidebarCollapsed');
    if (savedSidebarState === 'true') {
        sidebarCollapsed = true;
        sidebar.style.transition = 'none';
        mainContent.style.transition = 'none';
        mainContent.style.marginLeft = '50px';
        sidebar.classList.add('collapsed');
        
        // 다음 프레임에서 transition 속성 완전 제거
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                sidebar.style.removeProperty('transition');
                mainContent.style.removeProperty('transition');
            });
        });
    }
    
    // Setup event listeners
    setupEventListeners();
    
    // Load conversations list
    await loadConversations();
    
    // Get model info
    await getModelInfo();
    
    // Auto-resize textarea
    setupTextareaAutoResize();
}

function setupEventListeners() {
    // Sidebar toggle
    sidebarToggle.addEventListener('click', () => toggleSidebar());
    
    // New chat
    newChatBtn.addEventListener('click', () => createNewChat());
    
    // Send message or Stop
    sendBtn.addEventListener('click', () => {
        if (isStreaming) {
            stopGeneration();
        } else {
            sendMessage();
        }
    });
    
    // Enter to send, Backspace to remove tag
    messageInput.addEventListener('keydown', (e) => {
        // Backspace로 태그 삭제 (입력창이 비어있을 때)
        if (e.key === 'Backspace' && messageInput.value === '' && currentStepQuestion) {
            removeStepTag();
            e.preventDefault();
            return;
        }
        
        if (e.key === 'Enter' && !e.shiftKey && !isStreaming) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // File upload
    setupFileUpload();
    
    // 휠 이벤트 - 스크롤 잠금/해제 (사용자의 의도적인 스크롤만)
    messagesContainer.addEventListener('wheel', (e) => {
        if (e.deltaY < 0) {
            // 위로 스크롤 → 무한 잠금 (사용자가 하단으로 스크롤할 때까지)
            scrollLockUntil = Infinity;
        } else if (e.deltaY > 0) {
            // 아래로 스크롤 → 하단 근처면 잠금 해제 (자동 스크롤 재활성화)
            const threshold = 150;  // 하단에서 150px 이내
            const distanceFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
            if (distanceFromBottom < threshold) {
                scrollLockUntil = 0;
            }
        }
    });
    
    // 스크롤 이벤트 - 버튼 표시/숨김만 (잠금 해제는 wheel에서만)
    messagesContainer.addEventListener('scroll', () => {
        const threshold = 200;
        const distanceFromBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight;
        
        if (distanceFromBottom > threshold) {
            scrollToBottomBtn.classList.add('visible');
        } else {
            scrollToBottomBtn.classList.remove('visible');
            // 잠금 해제는 wheel 이벤트에서만 처리 (프로그래매틱 스크롤과 구분)
        }
    });
    
    // 하단 이동 버튼 클릭
    scrollToBottomBtn.addEventListener('click', () => {
        scrollLockUntil = 0;
        scrollToBottom(true);
    });
    
    // Input area hover 시 버튼 선명하게
    const inputArea = document.getElementById('dropZone');
    inputArea.addEventListener('mouseenter', () => {
        scrollToBottomBtn.classList.add('input-hover');
    });
    inputArea.addEventListener('mouseleave', () => {
        scrollToBottomBtn.classList.remove('input-hover');
    });
}

// ============================================
// File Upload (Drag & Drop)
// ============================================

function setupFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const attachBtn = document.getElementById('attachBtn');
    const chatContainer = document.querySelector('.main-content');
    
    // Helper functions for drag events
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
        chatContainer.classList.add('drag-over');
    }
    
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        // Only remove if leaving the container entirely
        if (!chatContainer.contains(e.relatedTarget)) {
            dropZone.classList.remove('drag-over');
            chatContainer.classList.remove('drag-over');
        }
    }
    
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
        chatContainer.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    }
    
    // Drag and drop events on input area
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // Also allow drag and drop on messages container
    messagesContainer.addEventListener('dragover', handleDragOver);
    messagesContainer.addEventListener('dragleave', handleDragLeave);
    messagesContainer.addEventListener('drop', handleDrop);
    
    // Click to attach
    attachBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
        fileInput.value = '';  // Reset for same file selection
    });
    
    // Paste from clipboard (Ctrl+V)
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        
        const files = [];
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                const file = item.getAsFile();
                if (file) files.push(file);
            }
        }
        
        if (files.length > 0) {
            e.preventDefault();
            handleFiles(files);
        }
    });
}

function getMaxImages() {
    return parseInt(localStorage.getItem('maxImages') || '5');
}

function handleFiles(files) {
    const maxImages = getMaxImages();
    
    for (const file of files) {
        // Check image limit before adding
        if (file.type.startsWith('image/')) {
            const currentImageCount = pendingFiles.filter(f => f.type === 'image').length;
            if (currentImageCount >= maxImages) {
                alert(`Maximum ${maxImages} images allowed. You can change this in Settings.`);
                return;
            }
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            if (file.type.startsWith('image/')) {
                // Double-check limit in async callback
                const currentImageCount = pendingFiles.filter(f => f.type === 'image').length;
                if (currentImageCount < maxImages) {
                    pendingFiles.push({ type: 'image', name: file.name, data: e.target.result });
                    renderFilePreviews();
                }
            } else if (file.type.startsWith('audio/')) {
                pendingFiles.push({ type: 'audio', name: file.name, data: e.target.result });
                renderFilePreviews();
            }
        };
        reader.readAsDataURL(file);
    }
}

function renderFilePreviews() {
    const container = document.getElementById('filePreviewContainer');
    if (pendingFiles.length === 0) {
        container.innerHTML = '';  // :empty CSS로 자동 숨김
        return;
    }
    
    // 파일 미리보기 렌더링 (:empty가 아니므로 자동 표시)
    container.innerHTML = pendingFiles.map((f, i) => `
        <div class="file-preview ${f.type}">
            ${f.type === 'image' 
                ? `<img src="${f.data}" alt="${escapeHtml(f.name)}">` 
                : `<span class="file-icon">🎵</span><span class="file-name">${escapeHtml(f.name)}</span>`
            }
            <button class="file-remove" onclick="removeFile(${i})" title="Remove">×</button>
        </div>
    `).join('');
}

function removeFile(index) {
    pendingFiles.splice(index, 1);
    renderFilePreviews();
}

function setupTextareaAutoResize() {
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
    });
}

// ============================================
// Sidebar
// ============================================

function toggleSidebar(forceCollapse = null) {
    if (forceCollapse !== null) {
        sidebarCollapsed = forceCollapse;
    } else {
        sidebarCollapsed = !sidebarCollapsed;
    }
    
    // 메인 영역 margin-left 설정
    const sidebarWidth = sidebarCollapsed ? 50 : 280;
    mainContent.style.marginLeft = sidebarWidth + 'px';
    
    // 클래스 토글 (transform은 CSS에서 처리)
    sidebar.classList.toggle('collapsed', sidebarCollapsed);
    localStorage.setItem('sidebarCollapsed', sidebarCollapsed);
}

// ============================================
// Conversations
// ============================================

async function loadConversations() {
    if (DEMO_MODE) {
        // Demo mode: use local conversations array
        renderConversationsList();
        return;
    }
    
    try {
        const response = await fetch('/api/conversations');
        if (!response.ok) throw new Error('Failed to load conversations');
        
        conversations = await response.json();
        renderConversationsList();
    } catch (error) {
        console.error('Error loading conversations:', error);
        conversations = [];
        renderConversationsList();
    }
}

function renderConversationsList() {
    if (conversations.length === 0) {
        conversationsList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <p>No conversations yet</p>
            </div>
        `;
        return;
    }
    
    conversationsList.innerHTML = conversations.map(conv => `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}" 
             data-id="${conv.id}"
             onclick="loadConversation('${conv.id}')">
            <div class="conversation-icon">💬</div>
            <div class="conversation-info">
                <div class="conversation-title">${escapeHtml(conv.title || 'New Chat')}</div>
                <div class="conversation-date">${formatDate(conv.updated_at || conv.created_at)}</div>
            </div>
            <button class="conversation-delete" onclick="deleteConversation(event, '${conv.id}')" title="Delete">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14"/>
                </svg>
            </button>
        </div>
    `).join('');
}

async function loadConversation(id) {
    // Hide detail panel when switching conversations
    hideDetailPanel();
    
    // 다른 채팅으로 이동 시 생성 중이면 중단 (현재까지 내용은 서버에서 자동 저장됨)
    if (isStreaming && currentConversationId !== id) {
        stopGeneration();
        // 잠시 대기하여 서버가 partial response를 저장할 시간 확보
        await new Promise(r => setTimeout(r, 300));
    }
    
    // 채팅 로드 시 자동 스크롤 활성화 (기본 동작)
    scrollLockUntil = 0;
    
    if (DEMO_MODE) {
        // Demo mode: find conversation from local array
        const conv = conversations.find(c => c.id === id);
        if (conv) {
            currentConversationId = id;
            renderConversationsList();
            renderMessages(conv.messages || []);
        }
        return;
    }
    
    try {
        const response = await fetch(`/api/conversation/${id}`);
        if (!response.ok) throw new Error('Failed to load conversation');
        
        const conversation = await response.json();
        currentConversationId = id;
        
        renderConversationsList();
        renderMessages(conversation.messages || []);
    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

async function createNewChat() {
    // Hide detail panel for new chat
    hideDetailPanel();
    
    // 생성 중이면 중단 (현재까지 내용은 서버에서 자동 저장됨)
    if (isStreaming) {
        stopGeneration();
        await new Promise(r => setTimeout(r, 300));
    }
    
    // 새 채팅 시 자동 스크롤 활성화
    scrollLockUntil = 0;
    
    if (DEMO_MODE) {
        // Demo mode: create local conversation
        const newId = `demo-${mockConversationCounter++}`;
        const newConv = {
            id: newId,
            title: 'New Chat',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            messages: []
        };
        conversations.unshift(newConv);
        currentConversationId = newId;
        
        renderConversationsList();
        renderMessages([]);
        messageInput.focus();
        return;
    }
    
    try {
        const response = await fetch('/api/new', { method: 'POST' });
        if (!response.ok) throw new Error('Failed to create new chat');
        
        const data = await response.json();
        currentConversationId = data.id;
        
        await loadConversations();
        renderMessages([]);
        messageInput.focus();
    } catch (error) {
        console.error('Error creating new chat:', error);
    }
}

async function deleteConversation(event, id) {
    event.stopPropagation();
    
    if (!confirm('Delete this conversation?')) return;
    
    if (DEMO_MODE) {
        // Demo mode: remove from local array
        conversations = conversations.filter(c => c.id !== id);
        if (currentConversationId === id) {
            hideDetailPanel();
            currentConversationId = null;
            renderMessages([]);
        }
        renderConversationsList();
        return;
    }
    
    try {
        const response = await fetch(`/api/conversation/${id}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete conversation');
        
        if (currentConversationId === id) {
            hideDetailPanel();
            currentConversationId = null;
            renderMessages([]);
        }
        
        await loadConversations();
    } catch (error) {
        console.error('Error deleting conversation:', error);
    }
}

async function clearCurrentChat() {
    if (!currentConversationId) return;
    if (!confirm('Clear all messages in this conversation?')) return;
    
    // Hide detail panel when clearing chat
    hideDetailPanel();
    
    if (DEMO_MODE) {
        // Demo mode: clear messages from local array
        const conv = conversations.find(c => c.id === currentConversationId);
        if (conv) {
            conv.messages = [];
        }
        currentMessages = [];
        renderMessages([]);
        renderConversationsList();
        return;
    }
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/clear`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to clear chat');
        
        renderMessages([]);
        await loadConversations();
    } catch (error) {
        console.error('Error clearing chat:', error);
    }
}

// ============================================
// Messages
// ============================================

function renderMessages(messages) {
    currentMessages = messages;
    
    if (messages.length === 0) {
        welcomeMessage.style.display = 'block';
        messagesWrapper.innerHTML = '';
        messagesWrapper.appendChild(welcomeMessage);
        return;
    }
    
    welcomeMessage.style.display = 'none';
    messagesWrapper.innerHTML = messages.map((msg, index) => createMessageHTML(msg, index)).join('');
    
    // Render math
    renderMath();
    
    // Scroll to bottom
    scrollToBottom();
}

function createMessageHTML(message, index = -1) {
    const isUser = message.role === 'user';
    const userDisplayName = getUserDisplayName();
    const avatar = isUser ? userDisplayName.charAt(0).toUpperCase() : 'A';
    const roleName = isUser ? userDisplayName : 'Assistant';
    
    let contentHTML = '';
    
    if (!isUser) {
        // Assistant message (including empty placeholder for streaming)
        // Parse all special tokens from content
        const parsed = parseSpecialTokens(message.content || '');
        
        // Render special tokens in order
        if (parsed.think) {
            contentHTML += createCoTHTML(parsed.think);
        }
        if (parsed.tools) {
            contentHTML += createToolsHTML(parsed.tools);
        }
        if (parsed.toolCalls) {
            contentHTML += createToolCallsHTML(parsed.toolCalls);
        }
        if (parsed.toolResults) {
            contentHTML += createToolResultsHTML(parsed.toolResults);
        }
        if (parsed.toolContent) {
            contentHTML += createToolContentHTML(parsed.toolContent);
        }
        if (parsed.fim) {
            contentHTML += createFimHTML(parsed.fim);
        }
        if (parsed.args) {
            contentHTML += createArgsHTML(parsed.args);
        }
        if (parsed.callId) {
            contentHTML += createCallIdBadge(parsed.callId);
        }
        if (parsed.planComplete) {
            contentHTML += createCompletedPlanHTML(parsed.planComplete);
        }
        
        // Render main answer with image/audio placeholders
        let answerHTML = renderMarkdown(parsed.answer);
        answerHTML = answerHTML.replace(/\{\{IMG_PLACEHOLDER\}\}/g, createImgPlaceholder());
        answerHTML = answerHTML.replace(/\{\{AUDIO_PLACEHOLDER\}\}/g, createAudioPlaceholder());
        
        contentHTML += `<div class="answer-content">${answerHTML}</div>`;
    } else {
        // User message - render attached files first
        let filesHTML = '';
        if (message.files && message.files.length > 0) {
            filesHTML = '<div class="message-files">';
            for (const f of message.files) {
                if (f.type === 'image') {
                    filesHTML += `<img src="${f.data}" alt="${escapeHtml(f.name)}" class="message-image" title="${escapeHtml(f.name)}">`;
                } else if (f.type === 'audio') {
                    filesHTML += `<div class="message-audio"><span class="audio-icon">🎵</span><span class="audio-name">${escapeHtml(f.name)}</span></div>`;
                }
            }
            filesHTML += '</div>';
        }
        
        // Extract text content (remove file references like [Image: xxx])
        let textContent = message.content || '';
        textContent = textContent.replace(/\[Image: [^\]]+\]\s*/g, '').replace(/\[Audio: [^\]]+\]\s*/g, '').trim();
        
        // Render markdown and convert step tag markers
        let renderedContent = renderMarkdown(textContent);
        renderedContent = renderedContent.replace(/\{\{STEP_TAG:(\d+)\}\}/g, '<span class="chat-step-tag">Step $1</span>');
        
        contentHTML = filesHTML + `<div class="answer-content">${renderedContent}</div>`;
    }
    
    // Action buttons (only for saved messages with valid index)
    let actionsHTML = '';
    if (index >= 0) {
        if (isUser) {
            actionsHTML = `
                <div class="message-actions">
                    <button class="message-action-btn" onclick="editMessage(${index})" title="Edit and resend">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                    <button class="message-action-btn" onclick="copyMessage(${index})" title="Copy">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                    </button>
                    <button class="message-action-btn delete-btn" onclick="deleteFromMessage(${index})" title="Delete from here">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            `;
        } else {
            actionsHTML = `
                <div class="message-actions">
                    <button class="message-action-btn" onclick="regenerateFrom(${index})" title="Regenerate">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M23 4v6h-6M1 20v-6h6"/>
                            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                        </svg>
                    </button>
                    <button class="message-action-btn" onclick="copyMessage(${index})" title="Copy">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                    </button>
                    <button class="message-action-btn delete-btn" onclick="deleteFromMessage(${index})" title="Delete from here">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            `;
        }
    }
    
    return `
        <div class="message ${message.role}" data-index="${index}">
            <div class="message-header">
                <div class="message-avatar">${avatar}</div>
                <div class="message-role">${roleName}</div>
            </div>
            <div class="message-content">
                ${contentHTML}
            </div>
            ${actionsHTML}
        </div>
    `;
}

// ============================================
// Special Token Parsing
// ============================================

/**
 * Extract a complete JSON object or array from text starting at a given position.
 * Handles nested structures and strings with escaped characters.
 */
function extractJsonFromPosition(text, startPos) {
    let depth = 0;
    let inString = false;
    let escape = false;
    let start = -1;
    
    for (let i = startPos; i < text.length; i++) {
        const char = text[i];
        
        if (escape) {
            escape = false;
            continue;
        }
        
        if (char === '\\' && inString) {
            escape = true;
            continue;
        }
        
        if (char === '"' && !escape) {
            inString = !inString;
            continue;
        }
        
        if (!inString) {
            if (char === '{' || char === '[') {
                if (depth === 0) start = i;
                depth++;
            } else if (char === '}' || char === ']') {
                depth--;
                if (depth === 0) {
                    return { json: text.substring(start, i + 1), endPos: i + 1 };
                }
            }
        }
    }
    return null;
}

function parseSpecialTokens(content) {
    let result = {
        think: null,
        tools: null,
        toolCalls: null,
        toolResults: null,
        toolContent: null,
        planComplete: null,
        fim: null,
        img: [],
        audio: null,
        args: null,
        callId: null,
        answer: content
    };
    
    // [THINK]...[/THINK]
    const thinkMatch = result.answer.match(/\[THINK\]([\s\S]*?)\[\/THINK\]/);
    if (thinkMatch) {
        result.think = thinkMatch[1].trim();
        result.answer = result.answer.replace(/\[THINK\][\s\S]*?\[\/THINK\]/, '');
    }
    
    // [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]
    const toolsMatch = result.answer.match(/\[AVAILABLE_TOOLS\]([\s\S]*?)\[\/AVAILABLE_TOOLS\]/);
    if (toolsMatch) {
        result.tools = toolsMatch[1].trim();
        result.answer = result.answer.replace(/\[AVAILABLE_TOOLS\][\s\S]*?\[\/AVAILABLE_TOOLS\]/, '');
    }
    
    // [TOOL_RESULTS]...[/TOOL_RESULTS]
    const toolResultsMatch = result.answer.match(/\[TOOL_RESULTS\]([\s\S]*?)\[\/TOOL_RESULTS\]/);
    if (toolResultsMatch) {
        result.toolResults = toolResultsMatch[1].trim();
        result.answer = result.answer.replace(/\[TOOL_RESULTS\][\s\S]*?\[\/TOOL_RESULTS\]/, '');
    }
    
    // [TOOL_CALLS]tool_name[ARGS]{json} - parse as complete unit
    const toolCallPattern = /\[TOOL_CALLS\](\w+)\[ARGS\]/;
    const toolCallMatch = result.answer.match(toolCallPattern);
    if (toolCallMatch) {
        const toolName = toolCallMatch[1];
        const argsStartPos = result.answer.indexOf('[ARGS]') + 6;
        const extracted = extractJsonFromPosition(result.answer, argsStartPos);
        
        if (extracted) {
            try {
                const args = JSON.parse(extracted.json);
                result.toolCalls = { name: toolName, arguments: args };
                // Remove the entire tool call from answer
                const toolCallStartPos = result.answer.indexOf('[TOOL_CALLS]');
                const fullMatch = result.answer.substring(toolCallStartPos, extracted.endPos);
                result.answer = result.answer.replace(fullMatch, '');
            } catch (e) {
                // JSON parse failed, fall back to simple extraction
                result.toolCalls = toolCallMatch[1];
                result.answer = result.answer.replace(/\[TOOL_CALLS\][\s\S]*?(?=\[(?!\/)|$)/, '');
            }
        } else {
            // Could not extract JSON, fall back to simple extraction
            result.toolCalls = toolCallMatch[1];
            result.answer = result.answer.replace(/\[TOOL_CALLS\][\s\S]*?(?=\[(?!\/)|$)/, '');
        }
    } else {
        // Fallback: [TOOL_CALLS] without [ARGS] format
        const simpleToolCallsMatch = result.answer.match(/\[TOOL_CALLS\]([\s\S]*?)(?=\[(?!\/)|$)/);
        if (simpleToolCallsMatch) {
            result.toolCalls = simpleToolCallsMatch[1].trim();
            result.answer = result.answer.replace(/\[TOOL_CALLS\][\s\S]*?(?=\[(?!\/)|$)/, '');
        }
    }
    
    // Handle incomplete tool calls (during streaming)
    // If we see [TOOL_CALLS] but couldn't parse it completely, mark as pending
    if (!result.toolCalls && result.answer.includes('[TOOL_CALLS]')) {
        result.toolCalls = 'pending';
        const toolCallStart = result.answer.indexOf('[TOOL_CALLS]');
        result.answer = result.answer.substring(0, toolCallStart);
    }
    
    // Also handle case where [ARGS] appears but JSON is incomplete
    if (result.answer.includes('[ARGS]')) {
        result.toolCalls = result.toolCalls || 'pending';
        const argsStart = result.answer.indexOf('[ARGS]');
        const toolCallStart = result.answer.indexOf('[TOOL_CALLS]');
        result.answer = result.answer.substring(0, Math.min(
            toolCallStart >= 0 ? toolCallStart : Infinity,
            argsStart
        ));
    }
    
    // [TOOL_CONTENT] (단독 토큰)
    const toolContentMatch = result.answer.match(/\[TOOL_CONTENT\]([\s\S]*?)(?=\[(?!\/)|$)/);
    if (toolContentMatch) {
        result.toolContent = toolContentMatch[1].trim();
        result.answer = result.answer.replace(/\[TOOL_CONTENT\][\s\S]*?(?=\[(?!\/)|$)/, '');
    }
    
    // FIM: [PREFIX]...[SUFFIX]...[MIDDLE]...
    const prefixMatch = result.answer.match(/\[PREFIX\]([\s\S]*?)(?=\[SUFFIX\]|\[MIDDLE\]|$)/);
    const suffixMatch = result.answer.match(/\[SUFFIX\]([\s\S]*?)(?=\[MIDDLE\]|$)/);
    const middleMatch = result.answer.match(/\[MIDDLE\]([\s\S]*?)$/);
    
    if (prefixMatch || suffixMatch || middleMatch) {
        result.fim = {
            prefix: prefixMatch ? prefixMatch[1] : '',
            suffix: suffixMatch ? suffixMatch[1] : '',
            middle: middleMatch ? middleMatch[1] : ''
        };
        result.answer = result.answer
            .replace(/\[PREFIX\][\s\S]*?(?=\[SUFFIX\]|\[MIDDLE\]|$)/, '')
            .replace(/\[SUFFIX\][\s\S]*?(?=\[MIDDLE\]|$)/, '')
            .replace(/\[MIDDLE\][\s\S]*?$/, '');
    }
    
    // [IMG], [IMG_BREAK], [IMG_END]
    const imgMatches = result.answer.match(/\[IMG\]|\[IMG_BREAK\]|\[IMG_END\]/g);
    if (imgMatches) {
        result.img = imgMatches;
        result.answer = result.answer.replace(/\[IMG\]|\[IMG_BREAK\]|\[IMG_END\]/g, '{{IMG_PLACEHOLDER}}');
    }
    
    // [AUDIO], [BEGIN_AUDIO]
    const audioMatch = result.answer.match(/\[AUDIO\]|\[BEGIN_AUDIO\]/);
    if (audioMatch) {
        result.audio = audioMatch[0];
        result.answer = result.answer.replace(/\[AUDIO\]|\[BEGIN_AUDIO\]/g, '{{AUDIO_PLACEHOLDER}}');
    }
    
    // [ARGS]...(JSON until next token or end) - only if not already consumed by tool call
    if (!result.toolCalls || typeof result.toolCalls === 'string') {
        const argsMatch = result.answer.match(/\[ARGS\]([\s\S]*?)(?=\[(?!\/)|$)/);
        if (argsMatch) {
            result.args = argsMatch[1].trim();
            result.answer = result.answer.replace(/\[ARGS\][\s\S]*?(?=\[(?!\/)|$)/, '');
        }
    }
    
    // [CALL_ID]...(ID until next token or whitespace)
    const callIdMatch = result.answer.match(/\[CALL_ID\](\S+)/);
    if (callIdMatch) {
        result.callId = callIdMatch[1].trim();
        result.answer = result.answer.replace(/\[CALL_ID\]\S+/, '');
    }
    
    // [PLAN_COMPLETE]{json} - completed plan with results
    const planCompleteMatch = result.answer.match(/\[PLAN_COMPLETE\]([\s\S]*)$/);
    if (planCompleteMatch) {
        try {
            result.planComplete = JSON.parse(planCompleteMatch[1].trim());
        } catch (e) {
            // JSON parse failed, keep raw
            result.planComplete = { raw: planCompleteMatch[1].trim() };
        }
        result.answer = result.answer.replace(/\[PLAN_COMPLETE\][\s\S]*$/, '');
    }
    
    result.answer = result.answer.trim();
    return result;
}

// Legacy wrapper for backward compatibility
function parseCoT(content) {
    const result = parseSpecialTokens(content);
    return { cot: result.think, answer: result.answer };
}

// ============================================
// Special Token HTML Renderers
// ============================================

function createCoTHTML(cotContent) {
    const preview = cotContent.substring(0, 80).replace(/\n/g, ' ');
    const displayPreview = preview.length < cotContent.length ? preview + '...' : preview;
    
    return `
        <div class="cot-container">
            <button class="cot-toggle" onclick="toggleSpecialToken(this)">
                <span class="cot-icon">✦</span>
                <span class="cot-label">생각하는 과정 표시</span>
                <span class="cot-arrow">▼</span>
            </button>
            <div class="cot-content">${escapeHtml(cotContent)}</div>
        </div>
    `;
}

function createToolsHTML(content) {
    return `
        <div class="special-token-container tools-container">
            <button class="special-toggle" onclick="toggleSpecialToken(this)">
                <span class="special-icon">🔧</span>
                <span class="special-label">Available Tools</span>
                <span class="special-arrow">▶</span>
            </button>
            <div class="special-content"><pre>${escapeHtml(content)}</pre></div>
        </div>
    `;
}

/**
 * Create inline plan steps HTML for rendering create_plan tool calls
 */
function createInlinePlanStepsHTML(args) {
    const goal = args.goal || '';
    const steps = args.steps || [];
    
    // Goal header with plan reference button
    let goalHTML = goal ? `
        <div class="plan-goal-container">
            <div class="plan-goal">${escapeHtml(goal)}</div>
            <button class="plan-ref-btn" onclick="askAboutPlan(this)" title="Plan 전체 참고하여 질문">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </button>
        </div>
    ` : '';
    
    let stepsHTML = '';
    steps.forEach((step, index) => {
        const stepNum = index + 1;
        stepsHTML += `
            <div class="plan-step pending" data-step-id="${stepNum}" data-tool="${escapeHtml(step.tool || '')}">
                <div class="step-header">
                    <div class="step-header-main" onclick="toggleStepResult(this.parentElement)">
                        <div class="step-indicator">${stepNum}</div>
                        <div class="step-content">
                            <div class="step-name">${escapeHtml(step.name || '')}</div>
                            <div class="step-tool">${escapeHtml(step.tool || '')}</div>
                        </div>
                    </div>
                    <div class="step-actions">
                        <button class="step-action-btn" onclick="event.stopPropagation(); retryStep(${stepNum})" title="재시도">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6"/>
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); editStepResult(${stepNum})" title="결과 수정">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); askAboutStep(${stepNum})" title="질문하기">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </button>
                    </div>
                    <div class="step-toggle" onclick="toggleStepResult(this.closest('.step-header'))" style="visibility: hidden;">▼</div>
                </div>
                <div class="step-result" style="display: none;"></div>
            </div>
        `;
    });
    
    return `<div class="plan-steps-box">${goalHTML}<div class="plan-steps">${stepsHTML}</div></div>`;
}

/**
 * Create HTML for a completed plan (loaded from saved [PLAN_COMPLETE] message)
 */
function createCompletedPlanHTML(planData) {
    const goal = planData.goal || '';
    const steps = planData.steps || [];
    const results = planData.results || [];
    
    // Create a map of step index to result
    const resultMap = {};
    results.forEach(r => {
        resultMap[r.step] = r;
    });
    
    // Goal header with plan reference button (button inside goal bar)
    let goalHTML = goal ? `
        <div class="plan-goal plan-goal-row">
            <span class="plan-goal-text">${escapeHtml(goal)}</span>
            <button class="plan-ref-btn" onclick="askAboutPlan(this)" title="Plan 전체 참고하여 질문">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </button>
        </div>
    ` : '';
    
    let stepsHTML = '';
    steps.forEach((step, index) => {
        const stepNum = index + 1;
        const result = resultMap[stepNum];
        const statusClass = result ? (result.success ? 'completed' : 'error') : 'pending';
        const indicator = result ? (result.success ? '✓' : '!') : stepNum;
        
        let resultHTML = '';
        if (result) {
            // Use formatStepResult for full display (thought, action, result)
            const formattedResult = formatStepResult({
                success: result.success,
                thought: result.thought,
                action: result.action,
                result: result.result
            });
            resultHTML = `<div class="step-result" style="display: block;">${formattedResult}</div>`;
        }
        
        stepsHTML += `
            <div class="plan-step ${statusClass}" data-step-id="${stepNum}" data-tool="${escapeHtml(step.tool || '')}">
                <div class="step-header">
                    <div class="step-header-main" onclick="toggleStepResult(this.parentElement)">
                        <div class="step-indicator">${indicator}</div>
                        <div class="step-content">
                            <div class="step-name">${escapeHtml(step.name || '')}</div>
                            <div class="step-tool">${escapeHtml(step.tool || '')}</div>
                        </div>
                    </div>
                    <div class="step-actions">
                        <button class="step-action-btn" onclick="event.stopPropagation(); retryStep(${stepNum})" title="재시도">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6"/>
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); editStepResult(${stepNum})" title="결과 수정">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); askAboutStep(${stepNum})" title="질문하기">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </button>
                    </div>
                    <div class="step-toggle" onclick="toggleStepResult(this.closest('.step-header'))" style="visibility: ${result ? 'visible' : 'hidden'};">${result ? '▲' : '▼'}</div>
                </div>
                ${resultHTML}
            </div>
        `;
    });
    
    return `<div class="plan-steps-box completed-plan">${goalHTML}<div class="plan-steps">${stepsHTML}</div></div>`;
}

function createToolCallsHTML(content) {
    // Skip if content is 'pending' (incomplete tool call that failed to parse)
    if (content === 'pending') {
        return '';
    }
    
    // Check if content is a parsed tool call object (from new parsing)
    if (typeof content === 'object' && content.name === 'create_plan') {
        return createInlinePlanStepsHTML(content.arguments);
    }
    
    // Fallback to existing behavior for raw string content or other tools
    const displayContent = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    return `
        <div class="special-token-container tool-calls-container">
            <div class="special-badge tool-calls-badge">
                <span class="special-icon">⚡</span>
                <span class="special-label">Tool Call</span>
            </div>
            <div class="special-content"><pre>${escapeHtml(displayContent)}</pre></div>
        </div>
    `;
}

function createToolResultsHTML(content) {
    return `
        <div class="special-token-container tool-results-container">
            <button class="special-toggle" onclick="toggleSpecialToken(this)">
                <span class="special-icon">📋</span>
                <span class="special-label">Tool Results</span>
                <span class="special-arrow">▶</span>
            </button>
            <div class="special-content"><pre>${escapeHtml(content)}</pre></div>
        </div>
    `;
}

function createToolContentHTML(content) {
    return `
        <div class="special-token-container tool-content-container">
            <div class="special-badge">
                <span class="special-icon">📦</span>
                <span class="special-label">Tool Content</span>
            </div>
            <div class="special-content"><pre>${escapeHtml(content)}</pre></div>
        </div>
    `;
}

function createFimHTML(fim) {
    let html = '<div class="fim-container">';
    
    if (fim.prefix) {
        html += `<div class="fim-section fim-prefix">
            <span class="fim-label">PREFIX</span>
            <pre>${escapeHtml(fim.prefix)}</pre>
        </div>`;
    }
    
    if (fim.middle) {
        html += `<div class="fim-section fim-middle">
            <span class="fim-label">MIDDLE (Generated)</span>
            <pre>${escapeHtml(fim.middle)}</pre>
        </div>`;
    }
    
    if (fim.suffix) {
        html += `<div class="fim-section fim-suffix">
            <span class="fim-label">SUFFIX</span>
            <pre>${escapeHtml(fim.suffix)}</pre>
        </div>`;
    }
    
    html += '</div>';
    return html;
}

function createImgPlaceholder() {
    return `<span class="img-placeholder" title="Image token">🖼️</span>`;
}

function createAudioPlaceholder() {
    return `<span class="audio-placeholder" title="Audio token">🔊</span>`;
}

function createArgsHTML(content) {
    let formatted = content;
    try {
        const parsed = JSON.parse(content);
        formatted = JSON.stringify(parsed, null, 2);
    } catch (e) {
        // Not valid JSON, use as-is
    }
    return `
        <div class="args-container">
            <span class="args-label">Args:</span>
            <pre class="args-json">${escapeHtml(formatted)}</pre>
        </div>
    `;
}

function createCallIdBadge(id) {
    return `<span class="call-id-badge" title="Call ID">#${escapeHtml(id)}</span>`;
}

function toggleSpecialToken(button) {
    button.classList.toggle('expanded');
}

// Keep old function name for backward compatibility
function toggleCoT(button) {
    toggleSpecialToken(button);
}

// ============================================
// Send Message
// ============================================

async function sendMessage(customContent = null) {
    const content = customContent || messageInput.value.trim();
    const files = [...pendingFiles];  // Copy pending files
    
    if ((!content && files.length === 0) || isStreaming) return;
    
    // Check for step question tag (pill-based)
    const stepTag = document.querySelector('#inputTags .input-tag');
    if (stepTag && currentStepQuestion && content) {
        await sendStepQuestionFromMain(content);
        // Clear tag after sending
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
        return;
    }
    
    // Clear step question context if not a step question
    currentStepQuestion = null;
    document.getElementById('inputTags').innerHTML = '';
    messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
    
    // 메시지 전송 시 자동 스크롤 활성화
    scrollLockUntil = 0;
    
    // Create new conversation if needed
    if (!currentConversationId) {
        await createNewChat();
    }
    
    // Clear input and files
    if (!customContent) {
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
    pendingFiles = [];
    renderFilePreviews();
    
    // Build display content for user message
    let displayContent = content;
    if (files.length > 0) {
        const fileLabels = files.map(f => f.type === 'image' ? `[Image: ${f.name}]` : `[Audio: ${f.name}]`).join(' ');
        displayContent = fileLabels + (content ? '\n' + content : '');
    }
    
    // Add user message to UI
    appendMessage({ role: 'user', content: displayContent, files });
    
    // Create placeholder for assistant response
    const assistantMessage = appendMessage({ role: 'assistant', content: '' });
    const contentDiv = assistantMessage.querySelector('.message-content');
    
    // Start streaming
    isStreaming = true;
    currentAbortController = new AbortController();
    setStreamingUI(true);
    
    // Add streaming indicator
    contentDiv.innerHTML = '<span class="streaming-indicator"><span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span></span>';
    
    try {
        // DEMO MODE: Simulate streaming response
        if (DEMO_MODE) {
            const isPlanRequest = content.toLowerCase().includes('plan');
            
            if (isPlanRequest) {
                // Plan 요청: 텍스트 스트리밍 없이 바로 Plan box 생성
                const planData = {
                    goal: 'CRISPR 스크린 실험을 위한 단계별 계획',
                    steps: MOCK_PLAN_STEPS.map((s, i) => ({
                        id: i + 1,
                        name: s.name,
                        tool: s.tool,
                        description: s.description
                    }))
                };
                
                const mockToolCall = { arguments: planData };
                const planBox = createPlanStepsBox(mockToolCall);
                contentDiv.appendChild(planBox);
                
                // Detail Panel 열기 (planData 전달)
                openDetailPanel(planData);
                
                // streaming indicator 제거
                const indicator = contentDiv.querySelector('.streaming-indicator');
                if (indicator) indicator.remove();
                
                // Step 완료 애니메이션 (비동기 - await 없이 실행)
                animatePlanSteps(planData.steps);
                
                // 대화 저장
                const conv = conversations.find(c => c.id === currentConversationId);
                if (conv) {
                    if (!conv.messages) conv.messages = [];
                    conv.messages.push({ role: 'user', content: displayContent });
                    conv.messages.push({ role: 'assistant', content: '[Plan Created]' });
                    if (conv.title === 'New Chat') {
                        conv.title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
                    }
                    conv.updated_at = new Date().toISOString();
                }
                currentMessages.push({ role: 'user', content: displayContent });
                currentMessages.push({ role: 'assistant', content: '[Plan Created]' });
                
                isStreaming = false;
                setStreamingUI(false);
                renderConversationsList();
                scrollToBottom();
                return;
            }
            
            // 일반 응답: 텍스트 스트리밍
            const mockResponse = MOCK_RESPONSES[Math.floor(Math.random() * MOCK_RESPONSES.length)];
            
            // Simulate token-by-token streaming
            let fullContent = '';
            const words = mockResponse.split(' ');
            
            for (let i = 0; i < words.length; i++) {
                if (!isStreaming) break;  // Check if stopped
                
                fullContent += (i > 0 ? ' ' : '') + words[i];
                updateAssistantMessage(contentDiv, fullContent);
                scrollToBottom();
                await new Promise(r => setTimeout(r, 30 + Math.random() * 20));
            }
            
            // Update conversation in local array
            const conv = conversations.find(c => c.id === currentConversationId);
            if (conv) {
                if (!conv.messages) conv.messages = [];
                conv.messages.push({ role: 'user', content: displayContent });
                conv.messages.push({ role: 'assistant', content: fullContent });
                // Update title from first user message
                if (conv.title === 'New Chat' && content.length > 0) {
                    conv.title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
                }
                conv.updated_at = new Date().toISOString();
            }
            
            // Update currentMessages
            currentMessages.push({ role: 'user', content: displayContent });
            currentMessages.push({ role: 'assistant', content: fullContent });
            
            // Remove streaming indicator
            const indicator = contentDiv.querySelector('.streaming-indicator');
            if (indicator) indicator.remove();
            
            // Finish streaming
            isStreaming = false;
            setStreamingUI(false);
            renderConversationsList();
            return;
        }
        
        // Build request body with files
        const requestBody = {
            conversation_id: currentConversationId,
            message: content
        };
        if (files.length > 0) {
            requestBody.files = files;
        }
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
            signal: currentAbortController.signal
        });
        
        if (!response.ok) throw new Error('Failed to send message');
        
        // Update sidebar title immediately (server already updated it when saving user message)
        loadConversations();
        
        // Read SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';
        let buffer = '';  // Buffer for incomplete SSE lines
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();  // Keep incomplete line for next chunk
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.token) {
                            fullContent += data.token;
                            updateAssistantMessage(contentDiv, fullContent);
                        }
                        
                        // Handle tool call events
                        if (data.tool_call) {
                            if (data.tool_call.name === 'create_plan') {
                                const toolBox = createPlanStepsBox(data.tool_call);
                                contentDiv.appendChild(toolBox);
                                scrollToBottom();
                                
                                // Open Detail Panel with plan data
                                const args = data.tool_call.arguments || {};
                                openDetailPanel({
                                    goal: args.goal || '',
                                    steps: (args.steps || []).map((s, i) => ({
                                        id: i + 1,
                                        name: s.name || '',
                                        tool: s.tool || '',
                                        description: s.description || ''
                                    }))
                                });
                            } else {
                                // Check if this tool is part of the current plan
                                const planBox = document.getElementById('current-plan-box');
                                if (planBox) {
                                    // Find step by tool name (first pending one with matching tool)
                                    const toolName = data.tool_call.name;
                                    const stepEl = planBox.querySelector(`.plan-step.pending[data-tool="${toolName}"]`) ||
                                                   planBox.querySelector(`[data-tool="${toolName}"]`);
                                    if (stepEl) {
                                        // Update step status to "running"
                                        stepEl.classList.remove('pending');
                                        stepEl.classList.add('running');
                                        // Force visual update
                                        void stepEl.offsetHeight;
                                        scrollToBottom();
                                    }
                                }
                                // If not part of plan, don't create separate box (cleaner UI)
                            }
                        }
                        
                        // Handle tool result events
                        if (data.tool_result) {
                            updateToolResultBox(data.tool_result);
                            scrollToBottom();
                            
                            // Update Detail Panel with tool result
                            if (detailPanelOpen && data.tool_result.step !== undefined) {
                                const stepIndex = data.tool_result.step - 1; // Convert to 0-based
                                addToolResultToDetailPanel(stepIndex, data.tool_result);
                            }
                        }
                        
                        // Handle step start events (LLM starting to generate tool call for this step)
                        if (data.step_start) {
                            const planBox = document.getElementById('current-plan-box');
                            if (planBox) {
                                const stepEl = planBox.querySelector(`[data-step-id="${data.step_start.step}"]`);
                                if (stepEl && !stepEl.classList.contains('completed')) {
                                    stepEl.classList.remove('pending');
                                    stepEl.classList.add('running');
                                    scrollToBottom();
                                }
                            }
                        }
                        
                        if (data.done) {
                            // If plan execution completed, update plan box with results
                            if (data.plan_complete) {
                                // Streaming indicator 제거
                                const streamingIndicator = contentDiv.querySelector('.streaming-indicator');
                                if (streamingIndicator) {
                                    streamingIndicator.remove();
                                }
                                
                                const planBox = document.getElementById('current-plan-box');
                                if (planBox) {
                                    updatePlanBoxWithResults(planBox, data.plan_complete);
                                }
                                
                                // Trigger Detail Panel completion updates
                                if (detailPanelOpen) {
                                    onPlanComplete();
                                }
                                
                                // Update currentMessages
                                currentMessages.push({ role: 'user', content: displayContent });
                                currentMessages.push({ role: 'assistant', content: fullContent });
                                
                                // Update action buttons on BOTH messages
                                const messageElements = messagesWrapper.querySelectorAll('.message');
                                const userMsgIndex = currentMessages.length - 2;
                                const assistantMsgIndex = currentMessages.length - 1;
                                
                                if (messageElements.length >= 2) {
                                    const userMsgEl = messageElements[messageElements.length - 2];
                                    const assistantMsgEl = messageElements[messageElements.length - 1];
                                    
                                    // Re-render user message
                                    if (userMsgEl) {
                                        userMsgEl.outerHTML = createMessageHTML(currentMessages[userMsgIndex], userMsgIndex);
                                    }
                                    
                                    // Update assistant message action buttons WITHOUT re-rendering (to avoid duplicate plan box)
                                    if (assistantMsgEl) {
                                        // Set data-index attribute
                                        assistantMsgEl.setAttribute('data-index', assistantMsgIndex);
                                        
                                        // Find or create actions div
                                        let actionsDiv = assistantMsgEl.querySelector('.message-actions');
                                        if (!actionsDiv) {
                                            actionsDiv = document.createElement('div');
                                            actionsDiv.className = 'message-actions';
                                            assistantMsgEl.appendChild(actionsDiv);
                                        }
                                        
                                        // Set action buttons HTML for assistant message
                                        actionsDiv.innerHTML = `
                                            <button class="message-action-btn" onclick="regenerateFrom(${assistantMsgIndex})" title="Regenerate">
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                                    <path d="M23 4v6h-6M1 20v-6h6"/>
                                                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                                                </svg>
                                            </button>
                                            <button class="message-action-btn" onclick="copyMessage(${assistantMsgIndex})" title="Copy">
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                                                </svg>
                                            </button>
                                            <button class="message-action-btn delete-btn" onclick="deleteFromMessage(${assistantMsgIndex})" title="Delete from here">
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                                </svg>
                                            </button>
                                        `;
                                    }
                                }
                                
                                // Cancel stream and cleanup
                                await reader.cancel();
                                scrollToBottom();
                                await loadConversations();
                                return;
                            }
                            
                            // Non-plan done handling: Preserve existing tool boxes
                            const existingToolBoxes = contentDiv.querySelectorAll('.plan-steps-box, .tool-call-box');
                            const toolBoxesToPreserve = [];
                            existingToolBoxes.forEach(box => {
                                box.remove();  // Detach from DOM
                                toolBoxesToPreserve.push(box);
                            });
                            
                            // Final update with all special tokens
                            const parsed = parseSpecialTokens(fullContent);
                            let finalHTML = '';
                            
                            if (parsed.think) {
                                finalHTML += createCoTHTML(parsed.think);
                            }
                            if (parsed.tools) {
                                finalHTML += createToolsHTML(parsed.tools);
                            }
                            // Skip toolCalls if we have existing tool boxes (already displayed)
                            if (toolBoxesToPreserve.length === 0 && parsed.toolCalls) {
                                finalHTML += createToolCallsHTML(parsed.toolCalls);
                            }
                            if (parsed.toolResults) {
                                finalHTML += createToolResultsHTML(parsed.toolResults);
                            }
                            if (parsed.toolContent) {
                                finalHTML += createToolContentHTML(parsed.toolContent);
                            }
                            if (parsed.fim) {
                                finalHTML += createFimHTML(parsed.fim);
                            }
                            if (parsed.args) {
                                finalHTML += createArgsHTML(parsed.args);
                            }
                            if (parsed.callId) {
                                finalHTML += createCallIdBadge(parsed.callId);
                            }
                            
                            let answerHTML = renderMarkdown(parsed.answer);
                            answerHTML = answerHTML.replace(/\{\{IMG_PLACEHOLDER\}\}/g, createImgPlaceholder());
                            answerHTML = answerHTML.replace(/\{\{AUDIO_PLACEHOLDER\}\}/g, createAudioPlaceholder());
                            finalHTML += `<div class="answer-content">${answerHTML}</div>`;
                            
                            contentDiv.innerHTML = finalHTML;
                            
                            // Re-append preserved tool boxes at the top
                            toolBoxesToPreserve.forEach(box => {
                                contentDiv.insertBefore(box, contentDiv.firstChild);
                            });
                            
                            renderMath();
                            
                            // Cancel stream and exit loop
                            await reader.cancel();
                            scrollToBottom();
                            // Don't reload conversation - content is already rendered with plan boxes
                            // Only refresh sidebar to update title
                            await loadConversations();
                            
                            // Update currentMessages to fix action buttons (edit/copy/delete/regenerate)
                            // Since we don't call loadConversation, we need to manually update the array
                            currentMessages.push({ role: 'user', content: displayContent });
                            currentMessages.push({ role: 'assistant', content: fullContent });
                            
                            // Update DOM elements with correct indices for action buttons
                            const messageElements = messagesWrapper.querySelectorAll('.message');
                            const userMsgIndex = currentMessages.length - 2;
                            const assistantMsgIndex = currentMessages.length - 1;
                            
                            if (messageElements.length >= 2) {
                                const userMsgEl = messageElements[messageElements.length - 2];
                                const assistantMsgEl = messageElements[messageElements.length - 1];
                                
                                // Re-render user message with correct index
                                if (userMsgEl) {
                                    userMsgEl.outerHTML = createMessageHTML(currentMessages[userMsgIndex], userMsgIndex);
                                }
                                
                                // Re-render assistant message (plan box is already created by createMessageHTML from saved content)
                                if (assistantMsgEl) {
                                    assistantMsgEl.outerHTML = createMessageHTML(currentMessages[assistantMsgIndex], assistantMsgIndex);
                                }
                            }
                            
                            return;
                        }
                        
                        if (data.error) {
                            contentDiv.innerHTML = `<div class="error">Error: ${escapeHtml(data.error)}</div>`;
                            await reader.cancel();
                            return;
                        }
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }
            
            scrollToBottom();
        }
        
        // Reload conversations to update title
        await loadConversations();
        
        // Update currentMessages for fallback case (if data.done wasn't received)
        if (!currentMessages.find(m => m.role === 'user' && m.content === displayContent)) {
            currentMessages.push({ role: 'user', content: displayContent });
            currentMessages.push({ role: 'assistant', content: fullContent });
            
            // Update DOM elements with correct indices for action buttons
            const messageElements = messagesWrapper.querySelectorAll('.message');
            const userMsgIndex = currentMessages.length - 2;
            const assistantMsgIndex = currentMessages.length - 1;
            
            if (messageElements.length >= 2) {
                const userMsgEl = messageElements[messageElements.length - 2];
                const assistantMsgEl = messageElements[messageElements.length - 1];
                
                if (userMsgEl) {
                    userMsgEl.outerHTML = createMessageHTML(currentMessages[userMsgIndex], userMsgIndex);
                }
                
                if (assistantMsgEl) {
                    assistantMsgEl.outerHTML = createMessageHTML(currentMessages[assistantMsgIndex], assistantMsgIndex);
                }
            }
        }
        
    } catch (error) {
        if (error.name === 'AbortError') {
            // User stopped generation - keep partial response
            console.log('Generation stopped by user');
            // Wait for server to save partial response before reloading
            await new Promise(r => setTimeout(r, 500));
            // Reload to show Edit/Regenerate buttons and update sidebar title
            await loadConversation(currentConversationId);
            await loadConversations();
        } else {
            console.error('Error sending message:', error);
            contentDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
    } finally {
        isStreaming = false;
        currentAbortController = null;
        setStreamingUI(false);
    }
}

function appendMessage(message) {
    welcomeMessage.style.display = 'none';
    
    const messageDiv = document.createElement('div');
    messageDiv.innerHTML = createMessageHTML(message);
    const messageElement = messageDiv.firstElementChild;
    messagesWrapper.appendChild(messageElement);
    
    scrollToBottom();
    return messageElement;
}

function updateAssistantMessage(contentDiv, content) {
    // Show streaming content with partial special token parsing
    const parsed = parseSpecialTokens(content);
    
    // Check for existing elements to avoid recreating them
    const existingPlanBox = contentDiv.querySelector('.plan-steps-box');
    const existingAnswerDiv = contentDiv.querySelector('.answer-content');
    const existingStreamingIndicator = contentDiv.querySelector('.streaming-indicator');
    
    // Render main answer with placeholders
    let answerHTML = renderMarkdown(parsed.answer);
    answerHTML = answerHTML.replace(/\{\{IMG_PLACEHOLDER\}\}/g, createImgPlaceholder());
    answerHTML = answerHTML.replace(/\{\{AUDIO_PLACEHOLDER\}\}/g, createAudioPlaceholder());
    
    // If we already have the structure, just update the answer content (reduces flicker)
    if (existingAnswerDiv) {
        existingAnswerDiv.innerHTML = answerHTML;
        
        // Ensure streaming indicator exists
        if (!existingStreamingIndicator) {
            const indicator = document.createElement('span');
            indicator.className = 'streaming-indicator';
            indicator.innerHTML = '<span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span>';
            contentDiv.appendChild(indicator);
        }
        return;
    }
    
    // First time setup - create structure
    let html = '';
    
    // Show thinking indicator during streaming
    if (parsed.think) {
        html += `<div class="cot-container">
            <button class="cot-toggle" onclick="toggleSpecialToken(this)">
                <span class="cot-icon">✦</span>
                <span class="cot-label" style="color: var(--text-cot);">생각하는 과정 표시</span>
                <span class="cot-arrow">▼</span>
            </button>
        </div>`;
    }
    
    // Show tool calls indicator during streaming (only if no plan box exists)
    if (parsed.toolCalls && !existingPlanBox) {
        html += `<div class="special-badge tool-calls-badge">
            <span class="special-icon">⚡</span>
            <span class="special-label">Tool Call...</span>
        </div>`;
    }
    
    // Show FIM indicator during streaming
    if (parsed.fim) {
        html += `<div class="special-badge">
            <span class="special-icon">📝</span>
            <span class="special-label">Code Generation...</span>
        </div>`;
    }
    
    html += `<div class="answer-content">${answerHTML}</div>`;
    html += '<span class="streaming-indicator"><span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span></span>';
    
    // Preserve plan box if it exists
    if (existingPlanBox) {
        existingPlanBox.remove();
        contentDiv.innerHTML = html;
        contentDiv.insertBefore(existingPlanBox, contentDiv.firstChild);
    } else {
        contentDiv.innerHTML = html;
    }
}

// ============================================
// Model Info
// ============================================

async function getModelInfo() {
    if (DEMO_MODE) {
        modelName.textContent = 'Biomni-Demo';
        return;
    }
    
    try {
        const response = await fetch('/api/model');
        if (!response.ok) throw new Error('Failed to get model info');
        
        const data = await response.json();
        modelName.textContent = data.model || 'Unknown';
    } catch (error) {
        console.error('Error getting model info:', error);
        modelName.textContent = 'Error loading';
    }
}

// ============================================
// Utilities
// ============================================

function getTheme() {
    return localStorage.getItem('theme') || 'soft-minimal';
}

function setTheme(theme) {
    localStorage.setItem('theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
}

function getUserDisplayName() {
    return localStorage.getItem('userDisplayName') || 'You';
}

function setUserDisplayName(name) {
    localStorage.setItem('userDisplayName', name || 'You');
}

// ============================================
// Background Image & Blur Management
// ============================================

function setBgImage(dataUrl) {
    localStorage.setItem('bgImage', dataUrl);
    applyBgImage();
}

function getBgImage() {
    return localStorage.getItem('bgImage');
}

function clearBgImage() {
    localStorage.removeItem('bgImage');
    applyBgImage();
    updateBgPreview();
}

function applyBgImage() {
    const bgLayer = document.getElementById('backgroundLayer');
    const bgImage = getBgImage();
    if (bgImage) {
        bgLayer.style.backgroundImage = `url(${bgImage})`;
    } else {
        bgLayer.style.backgroundImage = '';
    }
}

function selectBgImage() {
    document.getElementById('bgImageInput').click();
}

function updateBgPreview() {
    const preview = document.getElementById('bgPreview');
    const bgImage = getBgImage();
    if (bgImage) {
        preview.style.backgroundImage = `url(${bgImage})`;
        preview.textContent = '';
    } else {
        preview.style.backgroundImage = '';
        preview.textContent = 'No image selected';
    }
}

function setBgBlur(value) {
    localStorage.setItem('bgBlur', value);
    applyBgBlur();
}

function getBgBlur() {
    return localStorage.getItem('bgBlur') || '30';
}

function applyBgBlur() {
    const blur = getBgBlur();
    document.documentElement.style.setProperty('--bg-blur', blur + 'px');
}

// ============================================
// UI Opacity Management
// ============================================

function setBgOpacity(value) {
    localStorage.setItem('bgOpacity', value);
    applyBgOpacity();
}

function getBgOpacity() {
    return localStorage.getItem('bgOpacity') || '80';
}

function applyBgOpacity() {
    const opacity = getBgOpacity();
    document.documentElement.style.setProperty('--bg-opacity', opacity / 100);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
        return 'Yesterday';
    } else if (diffDays < 7) {
        return date.toLocaleDateString([], { weekday: 'short' });
    } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
}

function scrollToBottom(force = false) {
    // 시간 기반 잠금: 잠금 시간이 지났거나 force면 스크롤
    if (force || Date.now() > scrollLockUntil) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text || '');
    }
    return text || '';
}

function renderMath() {
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(messagesWrapper, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\[', right: '\\]', display: true },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false
        });
    }
}

// ============================================
// Loading State
// ============================================

function showLoading(text = 'Loading...') {
    loadingOverlay.querySelector('.loading-text').textContent = text;
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

// ============================================
// Streaming Control
// ============================================

function setStreamingUI(streaming) {
    if (streaming) {
        sendBtn.classList.add('streaming');
        sendBtn.disabled = false;
        sendIcon.style.display = 'none';
        stopIcon.style.display = 'block';
        sendBtn.title = 'Stop generation';
    } else {
        sendBtn.classList.remove('streaming');
        sendBtn.disabled = false;
        sendIcon.style.display = 'block';
        stopIcon.style.display = 'none';
        sendBtn.title = 'Send message (Enter)';
    }
}

function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        // Also notify server to stop (skip in demo mode)
        if (!DEMO_MODE) {
            fetch('/api/stop', { method: 'POST' }).catch(() => {});
        }
    }
    isStreaming = false;
    setStreamingUI(false);
}

// ============================================
// Edit and Regenerate
// ============================================

async function editMessage(index) {
    if (!currentConversationId || isStreaming) return;
    
    const message = currentMessages[index];
    if (!message || message.role !== 'user') return;
    
    // 메시지 요소 찾기
    const messageEl = document.querySelector(`.message[data-index="${index}"]`);
    if (!messageEl) return;
    
    // 이미 편집 중이면 무시
    if (messageEl.classList.contains('editing')) return;
    
    // 편집 모드 진입
    editingIndex = index;
    messageEl.classList.add('editing');
    
    const contentDiv = messageEl.querySelector('.answer-content');
    
    // 현재 너비 저장 및 min-width로 설정하여 너비 유지
    const currentWidth = contentDiv.offsetWidth;
    contentDiv.style.minWidth = currentWidth + 'px';
    
    const originalContent = message.content;
    
    // textarea로 교체
    contentDiv.innerHTML = `
        <textarea class="edit-textarea">${escapeHtml(originalContent)}</textarea>
        <div class="edit-actions">
            <button class="edit-btn save" onclick="saveEdit(${index})">Save & Send</button>
            <button class="edit-btn cancel" onclick="cancelEdit(${index})">Cancel</button>
        </div>
    `;
    
    const textarea = contentDiv.querySelector('.edit-textarea');
    
    // 내용에 맞게 높이 자동 조절
    function autoResize() {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }
    autoResize();
    textarea.addEventListener('input', autoResize);
    
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    
    // Enter로 저장, ESC로 취소
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            saveEdit(index);
        }
        if (e.key === 'Escape') {
            cancelEdit(index);
        }
    });
}

async function saveEdit(index) {
    const messageEl = document.querySelector(`.message[data-index="${index}"]`);
    if (!messageEl) return;
    
    const textarea = messageEl.querySelector('.edit-textarea');
    if (!textarea) return;
    
    const newContent = textarea.value.trim();
    if (!newContent) return;
    
    editingIndex = -1;
    
    // Truncate and send
    if (DEMO_MODE) {
        // Demo mode: truncate local messages
        const conv = conversations.find(c => c.id === currentConversationId);
        if (conv && conv.messages) {
            conv.messages = conv.messages.slice(0, index);
            currentMessages = [...conv.messages];
        }
        renderMessages(currentMessages);
        await sendMessage(newContent);
        return;
    }
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/truncate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from_index: index })
        });
        
        if (response.ok) {
            // UI 먼저 업데이트 (이전 메시지들 삭제 반영)
            await loadConversation(currentConversationId);
        }
        
        // 그 다음 새 메시지 전송
        await sendMessage(newContent);
    } catch (error) {
        console.error('Error saving edit:', error);
    }
}

function cancelEdit(index) {
    editingIndex = -1;
    loadConversation(currentConversationId);
}

async function copyMessage(index) {
    if (!currentConversationId) return;
    
    const message = currentMessages[index];
    if (!message) return;
    
    try {
        await navigator.clipboard.writeText(message.content);
        // 간단한 피드백 (버튼 아이콘 변경)
        const btn = document.querySelector(`.message:nth-child(${index + 1}) .message-action-btn[onclick*="copyMessage"]`);
        if (btn) {
            const originalHTML = btn.innerHTML;
            btn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
            `;
            setTimeout(() => {
                btn.innerHTML = originalHTML;
            }, 1500);
        }
    } catch (error) {
        console.error('Failed to copy:', error);
    }
}

async function deleteFromMessage(index) {
    if (!currentConversationId) return;
    
    // 생성 중이면 먼저 중지
    if (isStreaming) {
        stopGeneration();
        await new Promise(r => setTimeout(r, 500));  // 잠시 대기
    }
    
    // 확인 다이얼로그
    if (!confirm('Delete this message and all following messages?')) return;
    
    if (DEMO_MODE) {
        // Demo mode: truncate local messages
        const conv = conversations.find(c => c.id === currentConversationId);
        if (conv && conv.messages) {
            conv.messages = conv.messages.slice(0, index);
            currentMessages = [...conv.messages];
        }
        renderMessages(currentMessages);
        return;
    }
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/truncate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from_index: index })
        });
        
        if (response.ok) {
            await loadConversation(currentConversationId);
        }
    } catch (error) {
        console.error('Error deleting messages:', error);
    }
}

async function regenerateFrom(index) {
    if (!currentConversationId || isStreaming) return;
    
    // Get the user message before this assistant message
    const prevMessage = currentMessages[index - 1];
    if (!prevMessage || prevMessage.role !== 'user') return;
    
    const userContent = prevMessage.content;
    
    // Truncate from the user message (index - 1)
    // sendMessage will re-add the user message
    if (DEMO_MODE) {
        // Demo mode: truncate and regenerate
        const conv = conversations.find(c => c.id === currentConversationId);
        if (conv && conv.messages) {
            conv.messages = conv.messages.slice(0, index - 1);
            currentMessages = [...conv.messages];
        }
        renderMessages(currentMessages);
        await sendMessage(userContent);
        return;
    }
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/truncate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ from_index: index - 1 })
        });
        
        if (response.ok) {
            // Reload and regenerate
            await loadConversation(currentConversationId);
            // Send the same user message again
            await sendMessage(userContent);
        }
    } catch (error) {
        console.error('Error regenerating:', error);
    }
}

// ============================================
// Modal Functions
// ============================================

function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// Rename Modal
document.getElementById('renameBtn').addEventListener('click', async () => {
    if (!currentConversationId) {
        alert('No conversation selected');
        return;
    }
    
    // Get current title
    const conv = conversations.find(c => c.id === currentConversationId);
    document.getElementById('renameInput').value = conv?.title || '';
    openModal('renameModal');
    document.getElementById('renameInput').focus();
});

async function saveRename() {
    const newName = document.getElementById('renameInput').value.trim();
    if (!newName) return;
    
    if (DEMO_MODE) {
        // Demo mode: rename in local array
        const conv = conversations.find(c => c.id === currentConversationId);
        if (conv) {
            conv.title = newName;
        }
        renderConversationsList();
        closeModal('renameModal');
        return;
    }
    
    try {
        const response = await fetch(`/api/conversation/${currentConversationId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newName })
        });
        
        if (!response.ok) throw new Error('Failed to rename');
        
        await loadConversations();
        closeModal('renameModal');
    } catch (error) {
        console.error('Error renaming:', error);
        alert('Failed to rename conversation');
    }
}

// System Prompt Modal
document.getElementById('systemPromptBtn').addEventListener('click', async () => {
    if (DEMO_MODE) {
        document.getElementById('systemPromptInput').value = 'You are a helpful AI assistant for biomedical research.';
        openModal('systemPromptModal');
        return;
    }
    
    try {
        const response = await fetch('/api/system_prompt');
        if (!response.ok) throw new Error('Failed to get system prompt');
        
        const data = await response.json();
        document.getElementById('systemPromptInput').value = data.system_prompt || '';
        openModal('systemPromptModal');
    } catch (error) {
        console.error('Error getting system prompt:', error);
    }
});

async function saveSystemPrompt() {
    const systemPrompt = document.getElementById('systemPromptInput').value;
    
    if (DEMO_MODE) {
        closeModal('systemPromptModal');
        return;
    }
    
    try {
        const response = await fetch('/api/system_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ system_prompt: systemPrompt })
        });
        
        if (!response.ok) throw new Error('Failed to save');
        
        closeModal('systemPromptModal');
    } catch (error) {
        console.error('Error saving system prompt:', error);
        alert('Failed to save system prompt');
    }
}

// Settings Modal
document.getElementById('settingsBtn').addEventListener('click', async () => {
    // Demo mode: use default settings
    const data = DEMO_MODE ? { temperature: 1.0, max_length: 32768, top_k: 50, max_context: 32768 } : null;
    
    if (!DEMO_MODE) {
        try {
            const response = await fetch('/api/settings');
            if (!response.ok) throw new Error('Failed to get settings');
            Object.assign(data, await response.json());
        } catch (error) {
            console.error('Error getting settings:', error);
            return;
        }
    }
    
    // 현재 테마 버튼 강조
    const currentTheme = getTheme();
    document.querySelectorAll('.theme-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.theme === currentTheme);
    });
    // 배경 이미지 미리보기 업데이트
    updateBgPreview();
    // 블러 슬라이더 값 업데이트
    const blur = getBgBlur();
    document.getElementById('settingBlur').value = blur;
    document.getElementById('blurValue').textContent = blur;
    // 투명도 슬라이더 값 업데이트
    const opacity = getBgOpacity();
    document.getElementById('settingOpacity').value = opacity;
    document.getElementById('opacityValue').textContent = opacity;
    
    document.getElementById('settingUserName').value = getUserDisplayName();
    document.getElementById('settingTemperature').value = data.temperature || 1.0;
    document.getElementById('settingMaxLength').value = data.max_length || 32768;
    document.getElementById('settingTopK').value = data.top_k || 50;
    
    // Max Images 슬라이더 값 업데이트
    const maxImages = getMaxImages();
    document.getElementById('settingMaxImages').value = maxImages;
    document.getElementById('maxImagesValue').textContent = maxImages;
    
    // Max Context 값 업데이트
    const maxContext = data.max_context || 32768;
    document.getElementById('settingMaxContext').value = maxContext;
    document.getElementById('maxContextWarning').style.display = 'none';
    
    openModal('settingsModal');
});

// 테마 버튼 클릭 이벤트 (즉시 적용 및 저장)
document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // 모든 버튼에서 active 제거
        document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
        // 클릭된 버튼에 active 추가
        btn.classList.add('active');
        // 테마 적용
        setTheme(btn.dataset.theme);
    });
});

// 배경 이미지 파일 선택 이벤트
document.getElementById('bgImageInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            setBgImage(event.target.result);
            updateBgPreview();
        };
        reader.readAsDataURL(file);
    }
});

// 블러 슬라이더 이벤트 (실시간 적용)
document.getElementById('settingBlur').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('blurValue').textContent = value;
    setBgBlur(value);
});

// 투명도 슬라이더 이벤트 (실시간 적용)
document.getElementById('settingOpacity').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('opacityValue').textContent = value;
    setBgOpacity(value);
});

// Max Images 슬라이더 이벤트
document.getElementById('settingMaxImages').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('maxImagesValue').textContent = value;
});

// Max Context 입력 이벤트 (256k 초과 경고)
document.getElementById('settingMaxContext').addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    const warning = document.getElementById('maxContextWarning');
    if (value > 262144) {
        e.target.value = 262144;
        warning.style.display = 'block';
    } else {
        warning.style.display = 'none';
    }
});

async function saveSettings() {
    const userName = document.getElementById('settingUserName').value.trim() || 'You';
    const temperature = parseFloat(document.getElementById('settingTemperature').value);
    const maxLength = parseInt(document.getElementById('settingMaxLength').value);
    const topK = parseInt(document.getElementById('settingTopK').value);
    const maxImages = parseInt(document.getElementById('settingMaxImages').value);
    let maxContext = parseInt(document.getElementById('settingMaxContext').value);
    
    // Cap max_context at 256k
    if (maxContext > 262144) {
        maxContext = 262144;
    }
    if (maxContext < 1024) {
        maxContext = 1024;
    }
    
    // Save user name to localStorage
    setUserDisplayName(userName);
    
    // Save maxImages to localStorage
    localStorage.setItem('maxImages', maxImages.toString());
    
    if (DEMO_MODE) {
        // Demo mode: just close modal
        closeModal('settingsModal');
        return;
    }
    
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ temperature, max_length: maxLength, top_k: topK, max_context: maxContext })
        });
        
        if (!response.ok) throw new Error('Failed to save');
        
        // Reload messages to update display names
        if (currentConversationId) {
            await loadConversation(currentConversationId);
        }
        
        closeModal('settingsModal');
    } catch (error) {
        console.error('Error saving settings:', error);
        alert('Failed to save settings');
    }
}

// Close modal on background click
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }
});

// Enter key to save in rename modal
document.getElementById('renameInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        saveRename();
    }
});

// ============================================
// Tool Call / Plan Box Functions
// ============================================

let toolCallBoxes = {}; // Track tool call boxes by call_id

/**
 * Create a plan steps box (for create_plan tool calls)
 * Renders steps as numbered cards without raw JSON
 */
function createPlanStepsBox(toolCall) {
    const box = document.createElement('div');
    box.className = 'plan-steps-box';
    box.id = 'current-plan-box';
    
    const args = toolCall.arguments || {};
    const goal = args.goal || '';
    const steps = args.steps || [];
    
    // Goal header
    let goalHTML = goal ? `<div class="plan-goal">${escapeHtml(goal)}</div>` : '';
    
    // Steps with result placeholders and toggle
    let stepsHTML = '';
    steps.forEach((step, index) => {
        const stepNum = index + 1;
        stepsHTML += `
            <div class="plan-step pending" data-step-id="${stepNum}" data-tool="${escapeHtml(step.tool || '')}">
                <div class="step-header">
                    <div class="step-header-main" onclick="toggleStepResult(this.parentElement)" ondblclick="scrollToStepOutput(${index})">
                        <div class="step-indicator">${stepNum}</div>
                        <div class="step-content">
                            <div class="step-name">${escapeHtml(step.name || '')}</div>
                            <div class="step-tool">${escapeHtml(step.tool || '')}</div>
                        </div>
                    </div>
                    <div class="step-actions">
                        <button class="step-action-btn" onclick="event.stopPropagation(); retryStep(${stepNum})" title="재시도">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6"/>
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); editStepResult(${stepNum})" title="결과 수정">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                        <button class="step-action-btn" onclick="event.stopPropagation(); askAboutStep(${stepNum})" title="질문하기">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                        </button>
                    </div>
                    <div class="step-toggle" onclick="toggleStepResult(this.closest('.step-header'))" style="visibility: hidden;">▼</div>
                </div>
                <div class="step-result" style="display: none;"></div>
            </div>
        `;
    });
    
    box.innerHTML = `${goalHTML}<div class="plan-steps">${stepsHTML}</div>`;
    toolCallBoxes['create_plan'] = box.id;
    return box;
}

/**
 * Toggle collapse/expand of step result section
 */
function toggleStepResult(header) {
    const step = header.closest('.plan-step');
    const result = step.querySelector('.step-result');
    const toggle = step.querySelector('.step-toggle');
    
    if (result && result.innerHTML.trim()) {
        const isHidden = result.style.display === 'none';
        result.style.display = isHidden ? 'block' : 'none';
        if (toggle) {
            toggle.textContent = isHidden ? '▲' : '▼';
        }
    }
}

/**
 * Toggle collapse/expand of think section
 */
function toggleThinkSection(toggle) {
    const section = toggle.closest('.think-section-minimal');
    section.classList.toggle('collapsed');
    toggle.textContent = section.classList.contains('collapsed') ? 'Think ▶' : 'Think ▼';
}

// ============================================
// Step Action Functions (재시도, 수정, 질문)
// ============================================

// Store for step edits (수정된 결과 저장 - 재전송 아님!)
let stepEdits = {};

/**
 * Retry a step (재시도 - LLM이 tool select부터 다시 진행)
 */
async function retryStep(stepNum) {
    if (!currentConversationId || isStreaming) return;
    
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const tool = step.dataset.tool;
    const stepName = step.querySelector('.step-name')?.textContent || '';
    const resultEl = step.querySelector('.step-result');
    const originalResult = resultEl?.innerText || '';
    const userEdit = stepEdits[stepNum] || null;
    
    // Collect plan context
    const planBox = step.closest('.plan-steps-box');
    const planGoal = planBox?.querySelector('.plan-goal')?.textContent || '';
    
    // Collect previous steps with results
    const allSteps = planBox?.querySelectorAll('.plan-step') || [];
    const previousSteps = [];
    allSteps.forEach(s => {
        const sNum = parseInt(s.dataset.stepId);
        if (sNum < stepNum) {
            previousSteps.push({
                num: sNum,
                name: s.querySelector('.step-name')?.textContent || '',
                result: s.querySelector('.step-result')?.innerText || ''
            });
        }
    });
    
    try {
        // Show loading state
        step.classList.remove('completed', 'error');
        step.classList.add('running');
        resultEl.innerHTML = '<div class="step-loading">LLM이 재생성 중...</div>';
        resultEl.style.display = 'block';
        
        const response = await fetch('/retry_step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conv_id: currentConversationId,
                step_num: stepNum,
                tool: tool,
                step_name: stepName,
                original_result: originalResult,
                user_edit: userEdit,
                plan_goal: planGoal,
                previous_steps: previousSteps
            })
        });
        
        if (response.ok) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let llmText = '';
            let toolResultContent = '';
            let buffer = '';  // Buffer for incomplete SSE lines
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();  // Keep incomplete line for next chunk
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            // LLM token streaming
                            if (data.token) {
                                llmText += data.token;
                                // Show LLM thinking (optional)
                            }
                            
                            // Tool call started
                            if (data.tool_call) {
                                resultEl.innerHTML = `<div class="step-loading">${data.tool_call.name} 실행 중...</div>`;
                            }
                            
                            // Tool result received
                            if (data.tool_result) {
                                toolResultContent = formatStepResult(data.tool_result.result);
                                resultEl.innerHTML = toolResultContent;
                            }
                            
                            // Legacy: direct result (for backward compatibility)
                            if (data.result) {
                                toolResultContent = formatStepResult(data.result);
                                resultEl.innerHTML = toolResultContent;
                            }
                            
                            if (data.done) {
                                step.classList.remove('running');
                                step.classList.add('completed');
                                step.querySelector('.step-indicator').textContent = '✓';
                            }
                            
                            if (data.error) {
                                step.classList.remove('running');
                                step.classList.add('error');
                                resultEl.innerHTML = `<div class="step-error">${escapeHtml(data.error)}</div>`;
                            }
                        } catch (e) {}
                    }
                }
            }
        }
    } catch (error) {
        console.error('Retry step error:', error);
        step.classList.remove('running');
        step.classList.add('error');
        resultEl.innerHTML = `<div class="step-error">재시도 실패: ${escapeHtml(error.message)}</div>`;
    }
}

/**
 * Edit step result (결과 수정 UI 표시 - 재전송이 아닌 저장만!)
 * Think와 메타데이터(token/시간)는 수정 불가
 */
function editStepResult(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const resultEl = step.querySelector('.step-result');
    if (!resultEl) return;
    
    // Check if already in edit mode
    if (step.querySelector('.step-edit-container')) return;
    
    // Get editable text only (exclude Think and metadata)
    let editableText = stepEdits[stepNum] || '';
    if (!editableText) {
        // Clone result element to manipulate
        const clone = resultEl.cloneNode(true);
        
        // Remove Think section (not editable - doesn't affect next steps)
        const thinkSection = clone.querySelector('.think-section-minimal');
        if (thinkSection) thinkSection.remove();
        
        // Remove metadata (duration, tokens - not editable)
        const metaSection = clone.querySelector('.result-meta-minimal');
        if (metaSection) metaSection.remove();
        
        // Remove result title/summary (not editable)
        const labelSection = clone.querySelector('.section-label-minimal');
        if (labelSection) labelSection.remove();
        
        editableText = clone.innerText.trim();
    }
    
    // Create edit container
    const editContainer = document.createElement('div');
    editContainer.className = 'step-edit-container';
    editContainer.innerHTML = `
        <textarea class="step-edit-textarea" placeholder="결과를 수정하거나 보완하세요...">${escapeHtml(editableText)}</textarea>
        <div class="step-edit-actions">
            <button class="step-edit-btn cancel" onclick="cancelStepEdit(${stepNum})">취소</button>
            <button class="step-edit-btn save" onclick="saveStepEdit(${stepNum})">저장</button>
        </div>
    `;
    
    // Insert after result
    resultEl.style.display = 'none';
    step.appendChild(editContainer);
    
    // Focus textarea
    const textarea = editContainer.querySelector('textarea');
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

/**
 * Save step edit (수정 내용 저장 - 재전송하지 않음!)
 */
function saveStepEdit(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const textarea = step.querySelector('.step-edit-textarea');
    const editContainer = step.querySelector('.step-edit-container');
    const resultEl = step.querySelector('.step-result');
    
    if (textarea && textarea.value.trim()) {
        // Save to stepEdits (메모리에 저장, 재전송 아님)
        stepEdits[stepNum] = textarea.value.trim();
        
        // Add edited badge to step name if not exists
        const stepName = step.querySelector('.step-name');
        if (stepName && !stepName.querySelector('.step-edited-badge')) {
            const badge = document.createElement('span');
            badge.className = 'step-edited-badge';
            badge.textContent = '수정됨';
            stepName.appendChild(badge);
        }
    }
    
    // Remove edit container and show result
    if (editContainer) editContainer.remove();
    if (resultEl) resultEl.style.display = 'block';
}

/**
 * Cancel step edit
 */
function cancelStepEdit(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const editContainer = step.querySelector('.step-edit-container');
    const resultEl = step.querySelector('.step-result');
    
    if (editContainer) editContainer.remove();
    if (resultEl) resultEl.style.display = 'block';
}

/**
 * Ask about entire plan (Plan 전체 참고하여 질문)
 */
function askAboutPlan(btn) {
    const planBox = btn.closest('.plan-steps-box');
    const planGoal = planBox?.querySelector('.plan-goal')?.textContent || '';
    
    // Collect all steps with results
    const allSteps = planBox?.querySelectorAll('.plan-step') || [];
    const planSteps = [];
    allSteps.forEach(s => {
        planSteps.push({
            num: parseInt(s.dataset.stepId),
            name: s.querySelector('.step-name')?.textContent || '',
            tool: s.dataset.tool || '',
            result: s.querySelector('.step-result')?.innerText || ''
        });
    });
    
    // Set context for plan-level question
    currentStepQuestion = { 
        stepNum: 0,  // 0 = plan level
        tool: 'plan',
        stepName: 'Plan 전체',
        context: '',
        previousSteps: [],
        planGoal,
        planSteps
    };
    
    // Add tag pill
    const inputTags = document.getElementById('inputTags');
    inputTags.innerHTML = `
        <span class="input-tag" data-step="plan">
            Plan 전체
            <span class="input-tag-remove" onclick="removeStepTag()">×</span>
        </span>
    `;
    
    messageInput.value = '';
    messageInput.placeholder = 'Plan 전체에 대해 질문하세요...';
    messageInput.focus();
    messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Ask about step (메인 입력창에 태그 추가 방식)
 */
function askAboutStep(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const tool = step.dataset.tool || '';
    const stepName = step.querySelector('.step-name')?.textContent || '';
    const resultEl = step.querySelector('.step-result');
    const context = resultEl?.innerText || '';
    
    // Collect plan info (goal + all steps structure)
    const planBox = step.closest('.plan-steps-box');
    const planGoal = planBox?.querySelector('.plan-goal')?.textContent || '';
    
    // Collect all steps structure (for plan context)
    const allStepsInPlan = planBox?.querySelectorAll('.plan-step') || [];
    const planSteps = [];
    allStepsInPlan.forEach(s => {
        const sNum = parseInt(s.dataset.stepId);
        planSteps.push({
            num: sNum,
            name: s.querySelector('.step-name')?.textContent || '',
            tool: s.dataset.tool || ''
        });
    });
    
    // Collect previous steps with results (for detailed context)
    const previousSteps = [];
    allStepsInPlan.forEach(s => {
        const sNum = parseInt(s.dataset.stepId);
        if (sNum < stepNum) {
            previousSteps.push({
                num: sNum,
                name: s.querySelector('.step-name')?.textContent || '',
                result: s.querySelector('.step-result')?.innerText || ''
            });
        }
    });
    
    // Store context for when message is sent
    currentStepQuestion = { 
        stepNum, tool, stepName, context, 
        previousSteps,
        planGoal,      // Plan 목표
        planSteps      // 전체 plan 구조 (새 plan 작성 시 참고용)
    };
    
    // Add tag pill (instead of text)
    const inputTags = document.getElementById('inputTags');
    inputTags.innerHTML = `
        <span class="input-tag" data-step="${stepNum}">
            Step ${stepNum}
            <span class="input-tag-remove" onclick="removeStepTag()">×</span>
        </span>
    `;
    
    // Clear input and focus
    messageInput.value = '';
    messageInput.placeholder = `Step ${stepNum}에 대해 질문하세요...`;
    messageInput.focus();
    
    // Scroll to input area
    messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Remove step tag from input
 */
function removeStepTag() {
    const inputTags = document.getElementById('inputTags');
    inputTags.innerHTML = '';
    currentStepQuestion = null;
    messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
}

/**
 * Send question about step to LLM
 */
async function sendStepQuestion(stepNum) {
    const step = document.querySelector(`.plan-step[data-step-id="${stepNum}"]`);
    if (!step) return;
    
    const input = step.querySelector('.step-question-input');
    const question = input?.value.trim();
    if (!question) return;
    
    const tool = step.dataset.tool;
    const stepName = step.querySelector('.step-name')?.textContent || '';
    const resultEl = step.querySelector('.step-result');
    const stepContext = resultEl?.innerText || '';
    
    // Disable input
    input.disabled = true;
    const sendBtn = step.querySelector('.step-question-send');
    if (sendBtn) sendBtn.disabled = true;
    
    // Remove existing answer
    const existingAnswer = step.querySelector('.step-question-answer');
    if (existingAnswer) existingAnswer.remove();
    
    // Create answer container
    const answerDiv = document.createElement('div');
    answerDiv.className = 'step-question-answer';
    answerDiv.innerHTML = '<span class="loading-dots">답변 생성 중...</span>';
    step.appendChild(answerDiv);
    
    try {
        const response = await fetch('/step_question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conv_id: currentConversationId,
                step_num: stepNum,
                tool: tool,
                step_name: stepName,
                step_context: stepContext,
                question: question
            })
        });
        
        if (response.ok) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let answerText = '';
            let buffer = '';  // Buffer for incomplete SSE lines
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();  // Keep incomplete line for next chunk
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.token) {
                                answerText += data.token;
                                answerDiv.innerHTML = renderMarkdown(answerText);
                            }
                            if (data.done) {
                                // Clear input
                                input.value = '';
                            }
                        } catch (e) {}
                    }
                }
            }
        }
    } catch (error) {
        console.error('Step question error:', error);
        answerDiv.innerHTML = `<span class="error">오류: ${escapeHtml(error.message)}</span>`;
    } finally {
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.focus();
    }
}

/**
 * Send step question from main input (메인 입력창에서 @StepN: 태그로 질문)
 */
async function sendStepQuestionFromMain(question) {
    if (!currentStepQuestion || !currentConversationId) {
        // tag 제거하고 return
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
        currentStepQuestion = null;
        return;
    }
    
    const { stepNum, tool, stepName, context, previousSteps, planGoal, planSteps } = currentStepQuestion;
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // 바로 tag 제거 (보내는 시점에)
    document.getElementById('inputTags').innerHTML = '';
    messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
    
    // Hide welcome message
    if (welcomeMessage) welcomeMessage.style.display = 'none';
    
    // Show user's question in chat (using createMessageHTML)
    // Use marker that will be converted to tag pill after rendering
    const userMsgContent = `{{STEP_TAG:${stepNum}}} ${question}`;
    
    // Add to currentMessages for proper indexing
    currentMessages.push({role: 'user', content: userMsgContent});
    const userIndex = currentMessages.length - 1;
    
    const userHTML = createMessageHTML({role: 'user', content: userMsgContent}, userIndex);
    messagesWrapper.insertAdjacentHTML('beforeend', userHTML);
    scrollToBottom();
    
    // Create assistant message placeholder (using createMessageHTML)
    // Add placeholder to currentMessages
    currentMessages.push({role: 'assistant', content: ''});
    const assistantIndex = currentMessages.length - 1;
    
    const assistantHTML = createMessageHTML({role: 'assistant', content: ''}, assistantIndex);
    messagesWrapper.insertAdjacentHTML('beforeend', assistantHTML);
    const assistantMsgEl = messagesWrapper.lastElementChild;
    const contentDiv = assistantMsgEl.querySelector('.answer-content');
    
    // Safety check
    if (!contentDiv) {
        console.error('contentDiv not found in assistant message');
        // tag 제거하고 return
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
        currentStepQuestion = null;
        return;
    }
    
    // Show streaming indicator
    contentDiv.innerHTML = '<span class="streaming-indicator"><span class="streaming-dot"></span><span class="streaming-dot"></span><span class="streaming-dot"></span></span>';
    
    isStreaming = true;
    setStreamingUI(true);
    
    try {
        const response = await fetch('/step_question', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conv_id: currentConversationId,
                step_num: stepNum,
                tool: tool,
                step_name: stepName,
                step_context: context,
                previous_steps: previousSteps,
                plan_goal: planGoal,
                plan_steps: planSteps,
                question: question
            })
        });
        
        console.log('Step question response:', response.status, response.ok);
        
        if (response.ok) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let answerText = '';
            let buffer = '';  // Buffer for incomplete SSE lines
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();  // Keep incomplete line for next chunk
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.token) {
                                answerText += data.token;
                                contentDiv.innerHTML = renderMarkdown(answerText);
                                scrollToBottom();
                            }
                            if (data.error) {
                                contentDiv.innerHTML = `<span class="error">오류: ${escapeHtml(data.error)}</span>`;
                            }
                        } catch (e) {}
                    }
                }
            }
            
            // Final render
            if (answerText) {
                contentDiv.innerHTML = renderMarkdown(answerText);
                // Update currentMessages with final answer
                currentMessages[assistantIndex] = {role: 'assistant', content: answerText};
            }
            
            // Reload sidebar to show updated conversation
            loadConversations();
        } else {
            contentDiv.innerHTML = '<span class="error">요청 실패</span>';
        }
    } catch (error) {
        console.error('Step question error:', error);
        contentDiv.innerHTML = `<span class="error">오류: ${escapeHtml(error.message)}</span>`;
    } finally {
        isStreaming = false;
        setStreamingUI(false);
        currentStepQuestion = null;
        // 확실하게 tag 제거
        document.getElementById('inputTags').innerHTML = '';
        messageInput.placeholder = 'Type your message... (Shift+Enter for new line)';
    }
}

/**
 * Format tool result for display in a plan step
 * Minimal text-only design without backgrounds, borders, or icons
 */
function formatStepResult(toolResult) {
    if (!toolResult) return '';
    
    let html = '';
    
    // Handle string result (legacy)
    if (typeof toolResult === 'string') {
        return `<div class="section-text">${escapeHtml(toolResult)}</div>`;
    }
    
    // Thought section - collapsible "Think" (light gray, collapsed by default)
    if (toolResult.thought) {
        html += `
            <div class="think-section-minimal collapsed">
                <span class="think-toggle" onclick="toggleThinkSection(this)">Think ▶</span>
                <div class="think-content">${escapeHtml(toolResult.thought)}</div>
            </div>
        `;
    }
    
    // Action section - removed from UI (shown only in terminal DEBUG)
    
    // Result section (결과) - minimal
    const result = toolResult.result || toolResult;
    if (result && (result.title || result.details)) {
        const detailsHtml = result.details && Array.isArray(result.details) 
            ? `<ul class="result-details-minimal">${result.details.map(d => `<li>${escapeHtml(d)}</li>`).join('')}</ul>` 
            : '';
        
        // Graph output if available
        let graphHtml = '';
        if (result.has_graph) {
            if (result.graph_type === 'efficiency') {
                graphHtml = createEfficiencyChart(result);
            } else if (result.graph_type === 'timeline') {
                graphHtml = createTimelineChart(result);
            }
        }
        
        const metaHtml = (result.duration || result.tokens) 
            ? `<div class="result-meta-minimal">${result.duration || ''}${result.duration && result.tokens ? ' · ' : ''}${result.tokens ? result.tokens + ' tokens' : ''}</div>`
            : '';
        
        html += `
            <div class="step-section-minimal">
                <span class="section-label-minimal">${escapeHtml(result.title || '결과')}</span>
                ${detailsHtml}
                ${graphHtml}
                ${metaHtml}
            </div>
        `;
    } else if (!html && result) {
        // Fallback to JSON if no structured content
        html = `<pre class="result-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    }
    
    return html;
}

/**
 * Update a plan step with its tool result
 * Returns true if the result was successfully associated with a plan step
 */
function updatePlanStepResult(toolName, toolResult) {
    const planBox = document.getElementById('current-plan-box');
    if (!planBox) return false;
    
    // First try matching by step number (more reliable)
    let stepEl = null;
    if (toolResult.step) {
        stepEl = planBox.querySelector(`[data-step-id="${toolResult.step}"]`);
    }
    // Fallback: find the step with matching tool name - prefer running step first
    if (!stepEl) {
        stepEl = planBox.querySelector(`.plan-step.running[data-tool="${toolName}"]`) ||
                 planBox.querySelector(`.plan-step.pending[data-tool="${toolName}"]`) ||
                 planBox.querySelector(`[data-tool="${toolName}"]`);
    }
    if (!stepEl) return false;
    
    // Update step status
    stepEl.classList.remove('pending', 'running');
    stepEl.classList.add(toolResult.success ? 'completed' : 'error');
    
    // Update indicator
    const indicator = stepEl.querySelector('.step-indicator');
    if (indicator) {
        indicator.textContent = toolResult.success ? '✓' : '!';
    }
    
    // Update tool name display (tool is selected at execution time)
    const toolEl = stepEl.querySelector('.step-tool');
    if (toolEl && toolName) {
        toolEl.textContent = toolName;
        stepEl.dataset.tool = toolName;
    }
    
    // Show result - pass full toolResult to get thought, action, and result
    const resultEl = stepEl.querySelector('.step-result');
    if (resultEl && toolResult) {
        resultEl.style.display = 'block';
        resultEl.innerHTML = formatStepResult(toolResult);
        
        // Update toggle arrow to indicate content is available (▲ = expanded)
        const toggle = stepEl.querySelector('.step-toggle');
        if (toggle) {
            toggle.style.visibility = 'visible';
            toggle.textContent = '▲';
        }
    }
    
    return true;
}

/**
 * Update plan box with completed results from plan_complete data
 * Called when done event includes plan_complete
 */
function updatePlanBoxWithResults(planBox, planData) {
    const results = planData.results || [];
    
    results.forEach(r => {
        const stepEl = planBox.querySelector(`[data-step-id="${r.step}"]`);
        if (stepEl) {
            // Update step status
            stepEl.classList.remove('pending', 'running');
            stepEl.classList.add(r.success ? 'completed' : 'error');
            
            // Update indicator
            const indicator = stepEl.querySelector('.step-indicator');
            if (indicator) {
                indicator.textContent = r.success ? '✓' : '!';
            }
            
            // Update result section with full data (thought, action, result)
            const resultEl = stepEl.querySelector('.step-result');
            if (resultEl && (r.thought || r.action || r.result)) {
                resultEl.style.display = 'block';
                resultEl.innerHTML = formatStepResult({
                    success: r.success,
                    thought: r.thought,
                    action: r.action,
                    result: r.result
                });
                
                // Show toggle (▲ = expanded)
                const toggle = stepEl.querySelector('.step-toggle');
                if (toggle) {
                    toggle.style.visibility = 'visible';
                    toggle.textContent = '▲';
                }
            }
        }
    });
}

/**
 * Create a tool call box (shown when LLM calls a tool)
 */
function createToolCallBox(toolCall) {
    const box = document.createElement('div');
    box.className = 'tool-call-box';
    box.dataset.toolName = toolCall.name;
    box.dataset.status = toolCall.status || 'running';
    
    const toolId = `tool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    box.id = toolId;
    toolCallBoxes[toolCall.name] = toolId;
    
    // Tool icon mapping
    const toolIcons = {
        'pubmed_search': '📚',
        'ncbi_gene': '🧬',
        'crispr_designer': '✂️',
        'protocol_builder': '📋',
        'create_plan': '📝',
        'execute_step': '▶️'
    };
    
    const icon = toolIcons[toolCall.name] || '🔧';
    
    box.innerHTML = `
        <div class="tool-call-header">
            <span class="tool-icon">${icon}</span>
            <span class="tool-name">${escapeHtml(toolCall.name)}</span>
            <span class="tool-status ${toolCall.status || 'running'}">${toolCall.status === 'running' ? '실행 중...' : '완료'}</span>
        </div>
        <div class="tool-call-args">
            <pre>${escapeHtml(JSON.stringify(toolCall.arguments, null, 2))}</pre>
        </div>
        <div class="tool-call-result" style="display: none;"></div>
    `;
    
    return box;
}

/**
 * Update a tool result box with the result data
 */
function updateToolResultBox(toolResult) {
    const toolName = toolResult.tool || toolResult.name || toolResult.executed_tool;
    
    // First, try to update plan step if this tool is part of a plan
    if (updatePlanStepResult(toolName, toolResult)) {
        return; // Result shown in plan step
    }
    
    // Fallback: show in separate tool box
    const boxId = toolCallBoxes[toolName];
    
    if (!boxId) return;
    
    const box = document.getElementById(boxId);
    if (!box) return;
    
    // Update status
    box.dataset.status = toolResult.success ? 'completed' : 'error';
    const statusEl = box.querySelector('.tool-status');
    if (statusEl) {
        statusEl.className = `tool-status ${toolResult.success ? 'completed' : 'error'}`;
        statusEl.textContent = toolResult.success ? '완료' : '오류';
    }
    
    // Show result
    const resultEl = box.querySelector('.tool-call-result');
    if (resultEl && toolResult.result) {
        resultEl.style.display = 'block';
        
        const result = toolResult.result;
        let resultHTML = '';
        
        // Add thought if present
        if (toolResult.thought) {
            resultHTML += `<div class="tool-thought">💭 ${escapeHtml(toolResult.thought)}</div>`;
        }
        
        // Add action if present
        if (toolResult.action) {
            resultHTML += `<div class="tool-action">🔧 ${escapeHtml(toolResult.action)}</div>`;
        }
        
        // Add result content
        if (result.title) {
            resultHTML += `<div class="tool-result-title">✅ ${escapeHtml(result.title)}</div>`;
        }
        
        if (result.details && Array.isArray(result.details)) {
            resultHTML += '<ul class="tool-result-details">';
            for (const detail of result.details) {
                resultHTML += `<li>${escapeHtml(detail)}</li>`;
            }
            resultHTML += '</ul>';
        }
        
        // Add duration/tokens info if present
        if (result.duration || result.tokens) {
            resultHTML += `<div class="tool-meta">`;
            if (result.duration) resultHTML += `⏱️ ${result.duration} `;
            if (result.tokens) resultHTML += `📊 ${result.tokens} tokens`;
            resultHTML += `</div>`;
        }
        
        resultEl.innerHTML = resultHTML;
    }
}

/**
 * Create minimal efficiency bar chart (sgRNA efficiency distribution)
 */
function createEfficiencyChart(result) {
    // Generate bar data based on avg_efficiency or random
    const avgEff = result.avg_efficiency || 0.72;
    const bars = [];
    for (let i = 0; i < 6; i++) {
        const value = 0.4 + (i * 0.1);
        // Distribution: more bars near avg efficiency
        const height = Math.max(20, Math.min(95, 
            100 - Math.abs(value - avgEff) * 200 + Math.random() * 30
        ));
        bars.push({ value: value.toFixed(1), height: Math.round(height) });
    }
    
    const barsHtml = bars.map(b => 
        `<div class="mini-bar" style="height: ${b.height}%;" title="${b.value}"></div>`
    ).join('');
    
    const labelsHtml = bars.map(b => 
        `<span class="mini-bar-label">${b.value}</span>`
    ).join('');
    
    return `
        <div class="mini-graph-container">
            <div class="mini-graph-title">sgRNA 효율 점수 분포</div>
            <div class="mini-chart">
                ${barsHtml}
            </div>
            <div class="mini-chart-labels">
                ${labelsHtml}
            </div>
        </div>
    `;
}

/**
 * Create minimal timeline chart (experiment schedule)
 */
function createTimelineChart(result) {
    const weeks = result.duration_weeks || 6;
    const phases = [];
    
    // Determine phases based on experiment type
    if (result.experiment_type === 'crispr_screen' || !result.experiment_type) {
        const p1 = Math.ceil(weeks / 3);
        const p2 = Math.ceil(2 * weeks / 3);
        phases.push({ weeks: p1, label: '클로닝', color: '#8B4513' });
        phases.push({ weeks: p2 - p1, label: '형질도입', color: '#4A5568' });
        phases.push({ weeks: weeks - p2, label: '분석', color: '#22C55E' });
    } else {
        // Generic phases
        const half = Math.ceil(weeks / 2);
        phases.push({ weeks: half, label: '준비', color: '#8B4513' });
        phases.push({ weeks: weeks - half, label: '실행', color: '#22C55E' });
    }
    
    // Build week boxes
    let weekBoxes = '';
    let weekNum = 1;
    let labelHtml = '';
    
    phases.forEach((phase, pi) => {
        for (let i = 0; i < phase.weeks && weekNum <= weeks; i++) {
            weekBoxes += `<div class="timeline-week" style="background: ${phase.color};">W${weekNum}</div>`;
            weekNum++;
        }
        labelHtml += `<span class="timeline-phase-label" style="flex: ${phase.weeks};">${phase.label}</span>`;
    });
    
    return `
        <div class="mini-graph-container">
            <div class="mini-graph-title">${weeks}주 실험 타임라인</div>
            <div class="mini-timeline">
                ${weekBoxes}
            </div>
            <div class="mini-timeline-labels">
                ${labelHtml}
            </div>
        </div>
    `;
}

/**
 * Create a plan box showing all steps
 */
function createPlanBox(plan) {
    const box = document.createElement('div');
    box.className = 'plan-box';
    box.id = 'current-plan-box';
    
    let stepsHTML = '';
    for (const step of plan.steps || []) {
        const statusClass = step.status || 'pending';
        const statusIcon = statusClass === 'completed' ? '✓' : statusClass === 'running' ? '···' : step.id;
        
        stepsHTML += `
            <div class="plan-step ${statusClass}" data-step-id="${step.id}">
                <div class="step-indicator">${statusIcon}</div>
                <div class="step-content">
                    <div class="step-name">${escapeHtml(step.name)}</div>
                    <div class="step-tool">${escapeHtml(step.tool)}</div>
                </div>
            </div>
        `;
    }
    
    box.innerHTML = `
        <div class="plan-header">
            <span class="plan-icon">📋</span>
            <span class="plan-title">실행 계획</span>
            <span class="plan-progress">${plan.current_step || 0} / ${plan.total_steps || plan.steps?.length || 0}</span>
        </div>
        <div class="plan-goal">${escapeHtml(plan.goal || '')}</div>
        <div class="plan-steps">${stepsHTML}</div>
    `;
    
    return box;
}

/**
 * Update plan step status
 */
function updatePlanStep(stepId, status) {
    const planBox = document.getElementById('current-plan-box');
    if (!planBox) return;
    
    const stepEl = planBox.querySelector(`[data-step-id="${stepId}"]`);
    if (!stepEl) return;
    
    stepEl.className = `plan-step ${status}`;
    const indicator = stepEl.querySelector('.step-indicator');
    if (indicator) {
        indicator.textContent = status === 'completed' ? '✓' : status === 'running' ? '···' : stepId;
    }
}

/**
 * Animate plan steps completion (DEMO MODE)
 */
async function animatePlanSteps(steps) {
    const planBox = document.getElementById('current-plan-box');
    if (!planBox) return;
    
    for (let i = 0; i < steps.length; i++) {
        await new Promise(r => setTimeout(r, 800)); // 0.8초 대기
        
        const stepEl = planBox.querySelector(`.plan-step[data-step-id="${i + 1}"]`);
        if (!stepEl) continue;
        
        // running 상태로 변경
        stepEl.classList.remove('pending');
        stepEl.classList.add('running');
        const indicator = stepEl.querySelector('.step-indicator');
        if (indicator) indicator.textContent = '···';
        
        await new Promise(r => setTimeout(r, 500)); // 0.5초 후 완료
        
        // completed 상태로 변경
        stepEl.classList.remove('running');
        stepEl.classList.add('completed');
        if (indicator) indicator.textContent = '✓';
        
        // 결과 추가
        const resultEl = stepEl.querySelector('.step-result');
        if (resultEl && MOCK_STEP_RESULTS[i]) {
            resultEl.innerHTML = `<div class="step-result-content">${MOCK_STEP_RESULTS[i].summary}</div>`;
            resultEl.style.display = 'block';
            
            // toggle 버튼 보이기
            const toggle = stepEl.querySelector('.step-toggle');
            if (toggle) toggle.style.visibility = 'visible';
        }
        
        // Detail Panel 데이터 업데이트
        if (MOCK_STEP_OUTPUTS[i]) {
            detailPanelData.results[i] = MOCK_STEP_OUTPUTS[i];
            detailPanelData.currentStep = i + 1;
        }
        if (MOCK_STEP_CODES[i]) {
            detailPanelData.codes[i] = MOCK_STEP_CODES[i];
        }
        
        // Outputs 탭 업데이트
        renderOutputs();
        
        // Code step selector 업데이트
        updateCodeStepSelector();
        
        // All 뷰 업데이트 (currentCodeStep이 'all'이면)
        if (currentCodeStep === 'all') {
            renderCode('all');
        }
    }
}

// ============================================
// Detail Panel Functions
// ============================================

/**
 * Initialize Detail Panel event listeners
 */
function setupDetailPanelListeners() {
    // Close button
    if (detailClose) {
        detailClose.addEventListener('click', closeDetailPanel);
    }
    
    // Toggle button
    if (detailToggle) {
        detailToggle.addEventListener('click', toggleDetailPanel);
    }
    
    // Tab switching
    document.querySelectorAll('.detail-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchDetailTab(tab.dataset.tab);
        });
    });
    
    // Resize handle
    if (detailResizeHandle) {
        setupDetailResize();
    }
    
    // Regenerate buttons
    const regeneratePlan = document.getElementById('regeneratePlan');
    if (regeneratePlan) {
        regeneratePlan.addEventListener('click', () => requestAnalyzePlan(true));
    }
    
    const regenerateCode = document.getElementById('regenerateCode');
    if (regenerateCode) {
        regenerateCode.addEventListener('click', () => {
            if (currentCodeStep !== null) {
                requestCodeGen(currentCodeStep, true);
            }
        });
    }
    
    // Copy code button
    const copyCodeBtn = document.getElementById('copyCodeBtn');
    if (copyCodeBtn) {
        copyCodeBtn.addEventListener('click', copyCurrentCode);
    }
}

/**
 * Open Detail Panel with plan data
 */
function openDetailPanel(planData) {
    if (!detailPanel) return;
    
    // Initialize data
    detailPanelData.goal = planData.goal || '';
    detailPanelData.steps = planData.steps || [];
    detailPanelData.results = [];
    detailPanelData.codes = {};
    detailPanelData.summaryCode = null;
    detailPanelData.analysis = '';
    detailPanelData.currentStep = 0;
    
    // Show panel
    detailPanel.style.display = 'flex';
    // Set CSS variable on :root for consistent usage across toggle and panel
    document.documentElement.style.setProperty('--detail-panel-width', detailPanelWidth + 'px');
    
    if (detailResizeHandle) {
        detailResizeHandle.style.display = 'block';
    }
    
    if (detailToggle) {
        detailToggle.style.display = 'flex';
        detailToggle.classList.add('panel-open');
    }
    
    detailPanelOpen = true;
    
    // Request initial analysis
    requestAnalyzePlan();
    
    // Switch to Analysis Plan tab
    switchDetailTab('plan');
}

/**
 * Close Detail Panel
 */
function closeDetailPanel() {
    if (!detailPanel) return;
    
    detailPanel.style.display = 'none';
    
    if (detailResizeHandle) {
        detailResizeHandle.style.display = 'none';
    }
    
    if (detailToggle) {
        detailToggle.classList.remove('panel-open');
        // Keep toggle visible for reopening
    }
    
    detailPanelOpen = false;
}

/**
 * Toggle Detail Panel
 */
function toggleDetailPanel() {
    if (detailPanelOpen) {
        closeDetailPanel();
    } else {
        // Reopen panel
        if (detailPanel) {
            detailPanel.style.display = 'flex';
            if (detailResizeHandle) {
                detailResizeHandle.style.display = 'block';
            }
            if (detailToggle) {
                detailToggle.classList.add('panel-open');
            }
            detailPanelOpen = true;
        }
    }
}

/**
 * Hide Detail Panel completely (when no plan)
 */
function hideDetailPanel() {
    if (detailPanel) {
        detailPanel.style.display = 'none';
    }
    if (detailResizeHandle) {
        detailResizeHandle.style.display = 'none';
    }
    if (detailToggle) {
        detailToggle.style.display = 'none';
        detailToggle.classList.remove('panel-open');
    }
    detailPanelOpen = false;
    
    // Reset detail panel data
    detailPanelData = {
        goal: '',
        steps: [],
        results: {},
        codes: {},
        summaryCode: null,
        analysis: null,
        currentStep: 0
    };
}

/**
 * Switch Detail Panel tabs
 */
function switchDetailTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.detail-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update tab content
    document.querySelectorAll('.detail-tab-content').forEach(content => {
        content.style.display = 'none';
        content.classList.remove('active');
    });
    
    const activeContent = document.getElementById('tab' + tabName.charAt(0).toUpperCase() + tabName.slice(1));
    if (activeContent) {
        activeContent.style.display = 'flex';
        activeContent.classList.add('active');
    }
}

/**
 * Setup resize drag functionality
 */
function setupDetailResize() {
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    
    detailResizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = detailPanel.offsetWidth;
        detailResizeHandle.classList.add('resizing');
        // Add resizing class to toggle to disable transition
        if (detailToggle) {
            detailToggle.classList.add('resizing');
        }
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const diff = startX - e.clientX;
        const newWidth = Math.max(250, Math.min(startWidth + diff, window.innerWidth * 0.6));
        
        detailPanelWidth = newWidth;
        // Update CSS variable on :root for consistent usage
        document.documentElement.style.setProperty('--detail-panel-width', newWidth + 'px');
        detailPanel.style.width = newWidth + 'px';
        
        // Sync toggle button position (no transition during resize)
        if (detailToggle && detailPanelOpen) {
            detailToggle.style.right = (newWidth + 4) + 'px';
        }
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            detailResizeHandle.classList.remove('resizing');
            // Remove resizing class from toggle
            if (detailToggle) {
                detailToggle.classList.remove('resizing');
                // Clear inline style, let CSS handle via variable
                detailToggle.style.right = '';
            }
            // Clear inline width, let CSS handle via variable
            detailPanel.style.width = '';
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

/**
 * Request analyze_plan tool call
 */
async function requestAnalyzePlan(force = false) {
    const planLoading = document.getElementById('planLoading');
    const planContent = document.getElementById('planContent');
    const regenerateBtn = document.getElementById('regeneratePlan');
    
    // DEMO MODE: mock 분석 결과 표시
    if (DEMO_MODE) {
        if (planLoading) planLoading.style.display = 'none';
        if (planContent) {
            planContent.innerHTML = renderMarkdown(MOCK_PLAN_ANALYSIS);
        }
        if (regenerateBtn) regenerateBtn.style.display = 'block';
        detailPanelData.analysis = MOCK_PLAN_ANALYSIS;
        return;
    }
    
    if (!force && detailPanelData.analysis) {
        // Already have analysis
        return;
    }
    
    // Show loading
    if (planLoading) planLoading.style.display = 'flex';
    if (planContent) planContent.innerHTML = '';
    if (regenerateBtn) regenerateBtn.style.display = 'none';
    
    try {
        // Build steps with results
        const stepsWithResults = detailPanelData.steps.map((step, i) => ({
            name: step.name,
            tool: step.tool,
            description: step.description || '',
            status: i < detailPanelData.currentStep ? 'completed' : 
                   i === detailPanelData.currentStep ? 'running' : 'pending',
            result: detailPanelData.results[i] || null
        }));
        
        const response = await fetch('/api/tool_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tool: 'analyze_plan',
                args: {
                    goal: detailPanelData.goal,
                    steps: stepsWithResults,
                    current_step: detailPanelData.currentStep
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.result && data.result.analysis) {
            detailPanelData.analysis = data.result.analysis;
            renderAnalysisPlan(data.result.analysis);
        } else {
            if (planContent) {
                planContent.innerHTML = '<div class="error">분석 생성 실패</div>';
            }
        }
    } catch (error) {
        console.error('analyze_plan error:', error);
        if (planContent) {
            planContent.innerHTML = `<div class="error">오류: ${escapeHtml(error.message)}</div>`;
        }
    } finally {
        if (planLoading) planLoading.style.display = 'none';
        if (regenerateBtn) regenerateBtn.style.display = 'inline-flex';
    }
}

/**
 * Render Analysis Plan content (markdown)
 */
function renderAnalysisPlan(analysis) {
    const planContent = document.getElementById('planContent');
    if (!planContent) return;
    
    // Use existing markdown renderer
    planContent.innerHTML = renderMarkdown(analysis);
}

/**
 * Request code_gen tool call for a step
 */
async function requestCodeGen(stepIndex, force = false) {
    const codeLoading = document.getElementById('codeLoading');
    const codeContent = document.getElementById('codeContent');
    const codeActions = document.getElementById('codeActions');
    
    // Check if already have code for this step
    if (!force && detailPanelData.codes[stepIndex]) {
        renderCode(stepIndex);
        return;
    }
    
    const stepResult = detailPanelData.results[stepIndex];
    if (!stepResult) {
        if (codeContent) {
            codeContent.innerHTML = '<div class="code-empty-state">이 Step의 결과가 아직 없습니다.</div>';
        }
        return;
    }
    
    // Show loading
    if (codeLoading) codeLoading.style.display = 'flex';
    if (codeActions) codeActions.style.display = 'none';
    
    const step = detailPanelData.steps[stepIndex];
    const toolName = step?.tool || 'unknown';
    
    try {
        const response = await fetch('/api/tool_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tool: 'code_gen',
                args: {
                    task: `${toolName} 결과를 시각화하는 Python 코드를 생성하세요. matplotlib과 seaborn을 사용하고, 데이터 테이블도 pandas로 생성하세요.`,
                    language: 'python',
                    context: JSON.stringify(stepResult.result || stepResult, null, 2)
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.result) {
            detailPanelData.codes[stepIndex] = data.result;
            renderCode(stepIndex);
        } else {
            if (codeContent) {
                codeContent.innerHTML = '<div class="error">코드 생성 실패</div>';
            }
        }
    } catch (error) {
        console.error('code_gen error:', error);
        if (codeContent) {
            codeContent.innerHTML = `<div class="error">오류: ${escapeHtml(error.message)}</div>`;
        }
    } finally {
        if (codeLoading) codeLoading.style.display = 'none';
    }
}

/**
 * Request summary code generation after plan completion
 */
async function requestSummaryCodeGen() {
    if (detailPanelData.results.length === 0) return;
    
    try {
        const allResults = detailPanelData.results.map((r, i) => ({
            step: i + 1,
            tool: detailPanelData.steps[i]?.tool || 'unknown',
            result: r.result || r
        }));
        
        const response = await fetch('/api/tool_call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tool: 'code_gen',
                args: {
                    task: '모든 연구 단계의 결과를 종합하여 시각화하는 Python 코드를 생성하세요. 전체 결과 요약 그래프, 테이블 등을 포함하세요.',
                    language: 'python',
                    context: JSON.stringify(allResults, null, 2)
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.result) {
            detailPanelData.summaryCode = data.result;
            updateCodeStepSelector();
        }
    } catch (error) {
        console.error('Summary code_gen error:', error);
    }
}

/**
 * Update code step selector buttons
 */
function updateCodeStepSelector() {
    const selector = document.getElementById('codeStepSelector');
    if (!selector) return;
    
    let html = '';
    
    // All 버튼 추가 (DEMO MODE)
    if (DEMO_MODE && Object.keys(detailPanelData.codes).length > 0) {
        const activeClass = currentCodeStep === 'all' ? ' active' : '';
        html += `<button class="code-step-btn${activeClass}" data-step="all">All</button>`;
    }
    
    // Step buttons
    for (let i = 0; i < detailPanelData.steps.length; i++) {
        const hasCode = detailPanelData.codes[i];
        const hasResult = detailPanelData.results[i];
        const step = detailPanelData.steps[i];
        const activeClass = currentCodeStep === i ? ' active' : '';
        const disabledAttr = hasResult ? '' : ' disabled';
        
        html += `<button class="code-step-btn${activeClass}" data-step="${i}"${disabledAttr}>
            Step ${i + 1}: ${escapeHtml(step?.tool || '')}
        </button>`;
    }
    
    // Summary button
    if (detailPanelData.summaryCode) {
        const activeClass = currentCodeStep === 'summary' ? ' active' : '';
        html += `<button class="code-step-btn${activeClass}" data-step="summary">종합</button>`;
    }
    
    selector.innerHTML = html;
    
    // Add click handlers
    selector.querySelectorAll('.code-step-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const stepVal = btn.dataset.step;
            if (stepVal === 'all') {
                currentCodeStep = 'all';
                updateCodeStepSelector();
                renderCode('all');
            } else if (stepVal === 'summary') {
                currentCodeStep = 'summary';
                updateCodeStepSelector();
                renderCode('summary');
            } else {
                currentCodeStep = parseInt(stepVal);
                updateCodeStepSelector();
                if (DEMO_MODE) {
                    renderCode(currentCodeStep);
                } else {
                    requestCodeGen(currentCodeStep);
                }
            }
        });
    });
}

/**
 * Render code for a step or summary
 */
function renderCode(stepIndexOrSummary) {
    const codeContent = document.getElementById('codeContent');
    const codeActions = document.getElementById('codeActions');
    if (!codeContent) return;
    
    let codeData;
    let title;
    
    if (stepIndexOrSummary === 'all') {
        // Step별 구분된 코드 블록 표시
        let blocksHTML = '';
        for (let i = 0; i < detailPanelData.steps.length; i++) {
            if (detailPanelData.codes[i]) {
                const step = detailPanelData.steps[i];
                const stepTitle = `Step ${i + 1}: ${step?.tool || ''}`;
                const highlightedCode = highlightPythonSyntax(detailPanelData.codes[i].code);
                blocksHTML += `
                    <div class="code-block" style="margin-bottom: 16px;">
                        <div class="code-block-header">
                            <span class="code-block-title">${escapeHtml(stepTitle)}</span>
                            <span class="code-block-lang">python</span>
                        </div>
                        <div class="code-block-body">${highlightedCode}</div>
                    </div>
                `;
            }
        }
        codeContent.innerHTML = blocksHTML;
        if (codeActions) codeActions.style.display = 'flex';
        currentCodeStep = 'all';
        return;
    } else if (stepIndexOrSummary === 'summary') {
        codeData = detailPanelData.summaryCode;
        title = '종합 분석 코드';
    } else {
        codeData = detailPanelData.codes[stepIndexOrSummary];
        const step = detailPanelData.steps[stepIndexOrSummary];
        title = `Step ${stepIndexOrSummary + 1}: ${step?.tool || ''} 시각화`;
    }
    
    if (!codeData || !codeData.code) {
        codeContent.innerHTML = '<div class="code-empty-state">코드가 없습니다.</div>';
        if (codeActions) codeActions.style.display = 'none';
        return;
    }
    
    // Simple syntax highlighting
    const highlightedCode = highlightPythonSyntax(codeData.code);
    
    codeContent.innerHTML = `
        <div class="code-block">
            <div class="code-block-header">
                <span class="code-block-title">${escapeHtml(title)}</span>
                <span class="code-block-lang">${codeData.language || 'python'}</span>
            </div>
            <div class="code-block-body">${highlightedCode}</div>
        </div>
    `;
    
    if (codeActions) codeActions.style.display = 'flex';
    currentCodeStep = stepIndexOrSummary;
}

/**
 * Simple Python syntax highlighting
 */
function highlightPythonSyntax(code) {
    // Escape HTML first
    let escaped = escapeHtml(code);
    
    // Keywords
    const keywords = ['import', 'from', 'as', 'def', 'class', 'return', 'if', 'else', 'elif', 
                      'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None',
                      'try', 'except', 'finally', 'with', 'lambda', 'yield', 'pass', 'break', 'continue'];
    
    // Comments (# ...)
    escaped = escaped.replace(/(#.*)$/gm, '<span class="code-comment">$1</span>');
    
    // Strings (both single and double quotes)
    escaped = escaped.replace(/(&quot;.*?&quot;|&#39;.*?&#39;|"[^"]*"|'[^']*')/g, '<span class="code-string">$1</span>');
    
    // Numbers
    escaped = escaped.replace(/\b(\d+\.?\d*)\b/g, '<span class="code-number">$1</span>');
    
    // Keywords
    keywords.forEach(kw => {
        const regex = new RegExp(`\\b(${kw})\\b`, 'g');
        escaped = escaped.replace(regex, '<span class="code-keyword">$1</span>');
    });
    
    // Function calls
    escaped = escaped.replace(/\b([a-zA-Z_]\w*)\s*\(/g, '<span class="code-function">$1</span>(');
    
    return escaped;
}

/**
 * Copy current code to clipboard
 */
async function copyCurrentCode() {
    const copyBtn = document.getElementById('copyCodeBtn');
    
    let codeData;
    if (currentCodeStep === 'summary') {
        codeData = detailPanelData.summaryCode;
    } else if (currentCodeStep !== null) {
        codeData = detailPanelData.codes[currentCodeStep];
    }
    
    if (!codeData || !codeData.code) return;
    
    try {
        await navigator.clipboard.writeText(codeData.code);
        
        if (copyBtn) {
            copyBtn.classList.add('copied');
            copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"/>
            </svg> Copied!`;
            
            setTimeout(() => {
                copyBtn.classList.remove('copied');
                copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg> Copy`;
            }, 2000);
        }
    } catch (err) {
        console.error('Copy failed:', err);
    }
}

/**
 * Render Outputs tab content
 */
function renderOutputs() {
    const outputsContent = document.getElementById('outputsContent');
    if (!outputsContent) return;
    
    if (detailPanelData.results.length === 0) {
        outputsContent.innerHTML = '<div class="outputs-empty-state">Step 결과가 여기에 표시됩니다.</div>';
        return;
    }
    
    let html = '';
    
    for (let i = 0; i < detailPanelData.results.length; i++) {
        const result = detailPanelData.results[i];
        const step = detailPanelData.steps[i];
        const toolResult = result.result || result;
        
        html += `
            <div class="output-step-section" id="step-output-${i}">
                <div class="output-step-header">
                    <span class="output-step-number">${i + 1}</span>
                    <span class="output-step-title">${escapeHtml(step?.name || 'Step')}</span>
                    <span class="output-step-tool">${escapeHtml(step?.tool || '')}</span>
                </div>
                <div class="output-content">
                    ${renderToolResultDetail(toolResult)}
                </div>
            </div>
        `;
    }
    
    outputsContent.innerHTML = html;
}

/**
 * Render detailed tool result for Outputs tab
 */
function renderToolResultDetail(result) {
    if (!result) return '<div class="error">결과 없음</div>';
    
    let html = '';
    
    // Title
    if (result.title) {
        html += `<div class="output-title">${escapeHtml(result.title)}</div>`;
    }
    
    // Details list
    if (result.details && Array.isArray(result.details)) {
        html += '<ul class="output-details">';
        result.details.forEach(detail => {
            html += `<li>${escapeHtml(String(detail))}</li>`;
        });
        html += '</ul>';
    }
    
    // Table data (gene_table, paper_list, efficiency_data)
    if (result.gene_table && Array.isArray(result.gene_table)) {
        html += renderOutputTable(['Gene', 'Function', 'Location'], result.gene_table, ['gene', 'function', 'location']);
    }
    
    if (result.paper_list && Array.isArray(result.paper_list)) {
        html += renderOutputTable(['Title', 'Authors', 'Year'], result.paper_list.slice(0, 10), ['title', 'authors', 'year']);
    }
    
    if (result.efficiency_data && Array.isArray(result.efficiency_data)) {
        html += renderOutputTable(['Gene', 'Score'], result.efficiency_data.slice(0, 10), ['gene', 'score']);
    }
    
    // Summary
    if (result.summary) {
        html += `<div class="output-summary">${escapeHtml(result.summary)}</div>`;
    }
    
    // Meta info (duration, tokens)
    const meta = [];
    if (result.duration) meta.push(`${result.duration}초`);
    if (result.tokens) meta.push(`${result.tokens} tokens`);
    
    if (meta.length > 0) {
        html += `<div class="output-meta">${meta.join(' | ')}</div>`;
    }
    
    // Fallback to JSON if nothing else
    if (!html) {
        html = `<pre class="result-json">${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    }
    
    return html;
}

/**
 * Render table for output data
 */
function renderOutputTable(headers, data, keys) {
    if (!data || data.length === 0) return '';
    
    let html = '<table class="output-table"><thead><tr>';
    headers.forEach(h => {
        html += `<th>${escapeHtml(h)}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    data.forEach(row => {
        html += '<tr>';
        keys.forEach(key => {
            html += `<td>${escapeHtml(String(row[key] || ''))}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    return html;
}

/**
 * Add tool result to Detail Panel and trigger updates
 */
function addToolResultToDetailPanel(stepIndex, result) {
    // Store result
    detailPanelData.results[stepIndex] = result;
    detailPanelData.currentStep = stepIndex + 1;
    
    // Update Outputs tab
    renderOutputs();
    
    // Update code step selector
    updateCodeStepSelector();
    
    // Trigger code generation for this step
    requestCodeGen(stepIndex);
    
    // Refresh analysis
    requestAnalyzePlan(true);
}

/**
 * Handle plan completion
 */
function onPlanComplete() {
    // Generate summary code
    requestSummaryCodeGen();
    
    // Final analysis update
    requestAnalyzePlan(true);
}

/**
 * Scroll to step in Outputs tab
 */
function scrollToStepOutput(stepIndex) {
    // Switch to Outputs tab
    switchDetailTab('outputs');
    
    // Scroll to the step section
    setTimeout(() => {
        const stepEl = document.getElementById(`step-output-${stepIndex}`);
        if (stepEl) {
            stepEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 100);
}

// Initialize Detail Panel listeners after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setupDetailPanelListeners();
});
