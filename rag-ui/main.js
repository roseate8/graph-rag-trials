/**
 * Main JavaScript for RAG UI - connects to Flask backend API
 */

class RAGInterface {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000/api';
        this.logStream = null;
        this.init();
    }

    init() {
        this.bindEventListeners();
        this.initializeLogStreaming();
        this.loadConversationHistory();
        
        // Check system health on startup
        this.checkSystemHealth();
    }

    bindEventListeners() {
        // Query form submission
        const queryForm = document.getElementById('queryForm');
        if (queryForm) {
            queryForm.addEventListener('submit', (e) => this.handleQuerySubmit(e));
        }

        // API key form submission
        const apiKeyForm = document.getElementById('apiKeyForm');
        if (apiKeyForm) {
            apiKeyForm.addEventListener('submit', (e) => this.handleApiKeySubmit(e));
        }

        // API key toggle visibility
        const toggleApiKeyBtn = document.getElementById('toggleApiKeyBtn');
        if (toggleApiKeyBtn) {
            toggleApiKeyBtn.addEventListener('click', () => this.toggleApiKeyVisibility());
        }

        // History panel
        const historyBtn = document.getElementById('historyBtn');
        const closeHistoryBtn = document.getElementById('closeHistoryBtn');
        
        if (historyBtn) {
            historyBtn.addEventListener('click', () => this.toggleHistoryPanel());
        }
        
        if (closeHistoryBtn) {
            closeHistoryBtn.addEventListener('click', () => this.closeHistoryPanel());
        }
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const health = await response.json();
            
            if (health.rag_connected) {
                this.showStatus('System connected and ready', 'success');
            } else {
                this.showStatus('RAG system not connected', 'warning');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.showStatus('Backend server not available', 'error');
        }
    }

    async handleApiKeySubmit(e) {
        e.preventDefault();
        
        const apiKeyInput = document.getElementById('apiKeyInput');
        const loadingIndicator = document.getElementById('apiKeyLoadingIndicator');
        const submitBtn = document.getElementById('saveApiKeyBtn');
        
        if (!apiKeyInput.value.trim()) {
            this.showStatus('Please enter an API key', 'error');
            return;
        }

        // Show loading state
        loadingIndicator.classList.add('active');
        submitBtn.disabled = true;

        try {
            const response = await fetch(`${this.apiBaseUrl}/validate-api-key`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    api_key: apiKeyInput.value.trim()
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus('API key validated successfully', 'success');
                // Clear the input for security
                apiKeyInput.value = '';
            } else {
                this.showStatus(result.error || 'Failed to validate API key', 'error');
            }
        } catch (error) {
            console.error('API key validation failed:', error);
            this.showStatus('Failed to validate API key', 'error');
        } finally {
            loadingIndicator.classList.remove('active');
            submitBtn.disabled = false;
        }
    }

    async handleQuerySubmit(e) {
        e.preventDefault();
        
        const queryInput = document.getElementById('queryInput');
        const searchBtn = document.getElementById('searchBtn');
        const modelSelect = document.getElementById('modelSelect');
        const temperatureInput = document.getElementById('temperatureInput');
        const topKInput = document.getElementById('topKInput');
        
        const query = queryInput.value.trim();
        
        if (!query) {
            this.showStatus('Please enter a query', 'error');
            return;
        }

        // Show loading state
        this.setLoadingState(true);
        searchBtn.disabled = true;
        
        // Clear previous results
        this.clearResults();
        this.clearLogs();

        try {
            const response = await fetch(`${this.apiBaseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(topKInput.value),
                    model: modelSelect.value,
                    temperature: parseFloat(temperatureInput.value)
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result);
                this.showStatus('Query processed successfully', 'success');
                
                // Refresh conversation history
                this.loadConversationHistory();
            } else {
                this.showStatus(result.error || 'Failed to process query', 'error');
                this.displayError(result.error || 'An error occurred processing your query.');
            }
        } catch (error) {
            console.error('Query failed:', error);
            this.showStatus('Failed to process query', 'error');
            this.displayError('Failed to connect to the server. Please check if the backend is running.');
        } finally {
            this.setLoadingState(false);
            searchBtn.disabled = false;
        }
    }

    displayResults(result) {
        this.displayAnswer(result.response);
        this.displaySources(result.sources);
        this.displayMetrics(result.metrics);
    }

    displayAnswer(response) {
        const answerContent = document.getElementById('answerContent');
        if (answerContent) {
            answerContent.textContent = response;
            answerContent.classList.remove('empty-state');
        }
    }

    displaySources(sources) {
        const sourcesContent = document.getElementById('sourcesContent');
        if (!sourcesContent) return;

        // Store sources for later use in toggle functionality
        this.currentSources = sources;

        if (!sources || sources.length === 0) {
            sourcesContent.innerHTML = '<div class="empty-state">No sources available</div>';
            return;
        }

        sourcesContent.innerHTML = sources.map((source, index) => {
            const metadata = source.metadata || {};
            
            // Helper function to format metadata values
            const formatMetadataValue = (key, value) => {
                if (!value) return null;
                
                if (Array.isArray(value)) {
                    return value.length > 0 ? value.join(', ') : null;
                } else if (typeof value === 'object') {
                    // For structural_metadata and entity_metadata, show key properties
                    if (key === 'structural_metadata') {
                        const props = [];
                        if (value.element_type) props.push(`type: ${value.element_type}`);
                        if (value.page_number) props.push(`page: ${value.page_number}`);
                        if (value.is_heading) props.push('heading');
                        return props.length > 0 ? props.join(', ') : null;
                    } else if (key === 'entity_metadata') {
                        const props = [];
                        if (value.organizations?.length) props.push(`orgs: ${value.organizations.join(', ')}`);
                        if (value.locations?.length) props.push(`locations: ${value.locations.join(', ')}`);
                        if (value.products?.length) props.push(`products: ${value.products.join(', ')}`);
                        if (value.events?.length) props.push(`events: ${value.events.join(', ')}`);
                        return props.length > 0 ? props.join(' | ') : null;
                    }
                    return JSON.stringify(value);
                }
                return String(value);
            };
            
            // DEBUG: Log all metadata to see what's actually available
            console.log('Chunk metadata:', metadata);

            // Create metadata tags for all available metadata
            const metadataTags = Object.entries(metadata)
                .filter(([key, value]) => key !== 'chunk_type' && value !== null && value !== undefined && value !== '')
                .map(([key, value]) => {
                    const formattedValue = formatMetadataValue(key, value);
                    if (!formattedValue) return null;
                    
                    // Use friendly names for display
                    const friendlyNames = {
                        'chunk_id': 'ID',
                        'doc_id': 'Document',
                        'word_count': 'Words',
                        'section_path': 'Section',
                        'regions': 'Regions',
                        'product_version': 'Version',
                        'folder_path': 'Path',
                        'structural_metadata': 'Structure',
                        'entity_metadata': 'Entities'
                    };
                    
                    const displayName = friendlyNames[key] || key;
                    return `<span class="metadata-tag" title="${key}: ${formattedValue}">
                        <span class="metadata-key">${displayName}:</span> 
                        <span class="metadata-value">${formattedValue}</span>
                    </span>`;
                })
                .filter(tag => tag !== null)
                .join('');

            return `
                <div class="source-chunk">
                    <div class="source-header">
                        <span class="chunk-type">${source.chunk_type || 'unknown'}</span>
                    </div>
                    <div class="source-content-wrapper">
                        <div class="source-snippet" id="snippet-${index}">
                            ${source.snippet}
                        </div>
                        ${source.full_content !== source.snippet ? 
                            `<button class="see-more-btn" data-index="${index}">See more</button>` 
                            : ''
                        }
                    </div>
                    <div class="source-scores">
                        <span class="score">Similarity: ${source.score.toFixed(4)}</span>
                        ${source.rerank_score !== null && source.rerank_score !== undefined ? 
                            `<span class="score rerank-score">Rerank: ${source.rerank_score.toFixed(4)}</span>` 
                            : ''
                        }
                        ${source.rerank_probability !== null && source.rerank_probability !== undefined ? 
                            `<span class="score rerank-prob">Prob: ${(source.rerank_probability * 100).toFixed(1)}%</span>` 
                            : ''
                        }
                    </div>
                    ${metadataTags ? `<div class="metadata-tags">${metadataTags}</div>` : ''}
                </div>
            `;
        }).join('');
        
        // Add event listeners to "see more" buttons
        setTimeout(() => {
            document.querySelectorAll('.see-more-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const index = parseInt(e.target.getAttribute('data-index'));
                    this.toggleSourceContent(index);
                });
            });
        }, 100);
    }

    displayMetrics(metrics) {
        const metricsMap = {
            'totalTime': metrics.total_time,
            'tokensUsed': metrics.tokens_used,
            'cpuPeak': metrics.cpu_peak,
            'memoryPeak': metrics.memory_peak
        };

        Object.entries(metricsMap).forEach(([elementId, value]) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = value || '--';
            }
        });
    }

    displayError(message) {
        const answerContent = document.getElementById('answerContent');
        if (answerContent) {
            answerContent.innerHTML = `<div style="color: #F56868; font-style: italic;">${message}</div>`;
        }
    }

    toggleSourceContent(index) {
        const snippetElement = document.getElementById(`snippet-${index}`);
        const sourceChunk = snippetElement.closest('.source-chunk');
        const contentWrapper = sourceChunk.querySelector('.source-content-wrapper');
        const seeMoreBtn = sourceChunk.querySelector('.see-more-btn');
        const isExpanded = snippetElement.classList.contains('expanded');
        
        if (!this.currentSources || !this.currentSources[index]) {
            console.error('Source data not available for index:', index);
            return;
        }
        
        const source = this.currentSources[index];
        
        // Lock all parent containers to prevent width changes
        const sidebar = document.querySelector('.sidebar');
        const sourcesCard = document.querySelector('.sources-card');
        const sourcesContent = document.getElementById('sourcesContent');
        
        // Save current widths
        const sidebarWidth = sidebar.offsetWidth;
        const cardWidth = sourcesCard.offsetWidth;
        const contentWidth = sourcesContent.offsetWidth;
        const chunkWidth = sourceChunk.offsetWidth;
        const wrapperWidth = contentWrapper.offsetWidth;
        
        // Lock widths to prevent layout shifts
        sidebar.style.width = `${sidebarWidth}px`;
        sourcesCard.style.width = `${cardWidth}px`;
        sourcesContent.style.width = `${contentWidth}px`;
        sourceChunk.style.width = `${chunkWidth}px`;
        contentWrapper.style.width = `${wrapperWidth}px`;
        
        if (isExpanded) {
            // Show snippet (collapse)
            snippetElement.classList.remove('expanded');
            seeMoreBtn.textContent = 'See more';
            
            // Wait for transition to complete before changing content
            setTimeout(() => {
                snippetElement.innerHTML = source.snippet;
                
                // Reset widths after animation completes
                setTimeout(() => {
                    sidebar.style.width = '';
                    sourcesCard.style.width = '';
                    sourcesContent.style.width = '';
                    sourceChunk.style.width = '';
                    contentWrapper.style.width = '';
                }, 300);
            }, 50);
        } else {
            // Show full content (expand)
            snippetElement.classList.add('expanded');
            seeMoreBtn.textContent = 'See less';
            
            // Change content immediately when expanding
            snippetElement.innerHTML = source.full_content;
            
            // Reset widths after animation completes
            setTimeout(() => {
                sidebar.style.width = '';
                sourcesCard.style.width = '';
                sourcesContent.style.width = '';
                sourceChunk.style.width = '';
                contentWrapper.style.width = '';
            }, 300);
        }
    }

    initializeLogStreaming() {
        // Initialize Server-Sent Events for real-time log streaming
        if (typeof EventSource !== 'undefined') {
            try {
                this.logStream = new EventSource(`${this.apiBaseUrl}/logs/stream`);
                
                this.logStream.onmessage = (event) => {
                    try {
                        const logEntry = JSON.parse(event.data);
                        if (logEntry.type !== 'heartbeat') {
                            this.appendLog(logEntry);
                        }
                    } catch (e) {
                        console.error('Error parsing log entry:', e);
                    }
                };

                this.logStream.onerror = (error) => {
                    console.error('Log stream error:', error);
                    // Try to reconnect after a delay
                    setTimeout(() => {
                        if (this.logStream.readyState === EventSource.CLOSED) {
                            this.initializeLogStreaming();
                        }
                    }, 5000);
                };
            } catch (error) {
                console.error('Failed to initialize log streaming:', error);
            }
        }
    }

    appendLog(logEntry) {
        const logsContent = document.getElementById('logsContent');
        if (!logsContent) return;

        const timestamp = new Date(logEntry.timestamp).toLocaleTimeString();
        const logLine = `[${timestamp}] ${logEntry.level}: ${logEntry.message}\n`;
        
        logsContent.textContent += logLine;
        
        // Auto-scroll to bottom
        const logsContainer = logsContent.parentElement;
        if (logsContainer) {
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    }

    clearLogs() {
        const logsContent = document.getElementById('logsContent');
        if (logsContent) {
            logsContent.textContent = '';
        }
    }

    async loadConversationHistory() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/conversation-history`);
            const history = await response.json();
            
            this.displayConversationHistory(history);
        } catch (error) {
            console.error('Failed to load conversation history:', error);
        }
    }

    displayConversationHistory(history) {
        const chatsList = document.getElementById('chatsList');
        if (!chatsList) return;

        if (!history || history.length === 0) {
            chatsList.innerHTML = '<div class="empty-state">No saved chats</div>';
            return;
        }

        chatsList.innerHTML = history
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
            .map(chat => {
                const timestamp = new Date(chat.timestamp).toLocaleString();
                const shortQuery = chat.query.length > 80 ? 
                    chat.query.substring(0, 80) + '...' : 
                    chat.query;

                return `
                    <div class="chat-item" onclick="ragInterface.showChatDetails(${chat.id})">
                        <div class="chat-timestamp">${timestamp}</div>
                        <div class="chat-query">${shortQuery}</div>
                    </div>
                `;
            }).join('');
    }

    async showChatDetails(chatId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/conversation-history/${chatId}`);
            const chat = await response.json();
            
            const chatDetails = document.getElementById('chatDetails');
            if (!chatDetails) return;

            chatDetails.innerHTML = `
                <div class="detail-section">
                    <h4>Query</h4>
                    <div class="detail-content">${chat.query}</div>
                </div>
                <div class="detail-section">
                    <h4>Response</h4>
                    <div class="detail-content">${chat.response}</div>
                </div>
                <div class="detail-section">
                    <h4>Sources (${chat.sources.length})</h4>
                    <div class="detail-content">
                        ${chat.sources.map(source => `
                            <div style="margin-bottom: 8px; padding: 8px; border: 1px solid #525252;">
                                <strong>${source.chunk_type}</strong> (Score: ${source.score.toFixed(4)})<br>
                                ${source.snippet}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;

            // Highlight selected chat
            document.querySelectorAll('.chat-item').forEach(item => {
                item.classList.remove('selected');
            });
            event.target.closest('.chat-item').classList.add('selected');

        } catch (error) {
            console.error('Failed to load chat details:', error);
        }
    }

    toggleHistoryPanel() {
        const historyPanel = document.getElementById('historyPanel');
        if (historyPanel) {
            historyPanel.classList.toggle('open');
        }
    }

    closeHistoryPanel() {
        const historyPanel = document.getElementById('historyPanel');
        if (historyPanel) {
            historyPanel.classList.remove('open');
        }
    }

    toggleApiKeyVisibility() {
        const apiKeyInput = document.getElementById('apiKeyInput');
        const eyeIcon = document.querySelector('.eye-icon');
        
        if (!apiKeyInput || !eyeIcon) return;

        if (apiKeyInput.type === 'password') {
            apiKeyInput.type = 'text';
            eyeIcon.classList.remove('hidden');
            eyeIcon.classList.add('visible');
        } else {
            apiKeyInput.type = 'password';
            eyeIcon.classList.remove('visible');
            eyeIcon.classList.add('hidden');
        }
    }

    setLoadingState(loading) {
        const answerContent = document.getElementById('answerContent');
        if (answerContent) {
            if (loading) {
                answerContent.innerHTML = '<div class="thinking-indicator">Thinking...</div>';
                answerContent.classList.add('loading');
            } else {
                answerContent.classList.remove('loading');
            }
        }
    }

    clearResults() {
        const answerContent = document.getElementById('answerContent');
        const sourcesContent = document.getElementById('sourcesContent');
        
        if (answerContent) {
            answerContent.innerHTML = '<div class="empty-state">Submit a query to get started</div>';
            answerContent.classList.remove('loading');
        }
        
        if (sourcesContent) {
            sourcesContent.innerHTML = '<div class="empty-state">No sources available</div>';
        }

        // Reset metrics
        ['totalTime', 'tokensUsed', 'cpuPeak', 'memoryPeak'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = '--';
            }
        });
    }

    showStatus(message, type) {
        // Log to console
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Create toast notification
        const toast = this.createToast(message, type);
        document.body.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Remove toast after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }

    createToast(message, type) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${this.getToastIcon(type)}</span>
            <span class="toast-message">${message}</span>
        `;
        return toast;
    }

    getToastIcon(type) {
        switch(type) {
            case 'success': return '✓';
            case 'error': return '✕';
            case 'warning': return '⚠';
            default: return 'ℹ';
        }
    }
}

// Initialize the RAG interface when the page loads
let ragInterface;
document.addEventListener('DOMContentLoaded', () => {
    ragInterface = new RAGInterface();
});