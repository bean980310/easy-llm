css="""
/* Updated style.css */

.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem;
    background-color: #f8fafc;
    min-height: 100vh;
}

.header-container {
    background-color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}

.main-title {
    margin: 0;
    color: #1a1a1a;
}

.language-selector {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem;
}

.session-container {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.session-dropdown {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem;
}

.icon-button, .icon-button-delete {
    padding: 0.5rem;
    border-radius: 8px;
    border: none;
    background-color: #f3f4f6;
    cursor: pointer;
    transition: all 0.2s;
}

.icon-button:hover {
    background-color: #e5e7eb;
}

.icon-button-delete:hover {
    background-color: #fee2e2;
    color: #dc2626;
}

.model-container {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.chat-interface {
    background-color: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    min-height: 600px;
}

.system-message {
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 1rem;
}

.chat-window {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background-color: #ffffff;
    padding: 1rem;
    height: 500px;
    overflow-y: auto;
}

.chat-window .user-message {
    background-color: #f0f9ff;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}

.chat-window .assistant-message {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}

.input-area {
    margin-top: 1rem;
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 0.5rem;
}

.message-input {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.75rem;
}

.send-button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    transition: background-color 0.2s;
}

.send-button:hover {
    background-color: #2563eb;
}

.side-panel {
    padding-left: 1rem;
}

.profile-image {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}

.character-dropdown, .preset-dropdown {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
}

.status-bar {
    margin-top: 1rem;
    padding: 0.5rem;
    background-color: #f8fafc;
    border-radius: 8px;
    color: #6b7280;
    font-size: 0.875rem;
}

/* 다크 모드 지원 */
@media (prefers-color-scheme: dark) {
    .main-container {
        background-color: #1a1a1a;
    }
    
    .header-container, .session-container, .model-container, .chat-interface {
        background-color: #2d2d2d;
        border-color: #404040;
    }
    
    .main-title {
        color: #ffffff;
    }
    
    .chat-window {
        background-color: #2d2d2d;
        border-color: #404040;
    }
    
    .chat-window .user-message {
        background-color: #3b4252;
    }
    
    .chat-window .assistant-message {
        background-color: #2e3440;
    }
    
    .system-message, .input-area {
        background-color: #2d2d2d;
        border-color: #404040;
    }
    
    .message-input {
        background-color: #1a1a1a;
        border-color: #404040;
        color: #ffffff;
    }
    
    .status-bar {
        background-color: #2d2d2d;
        color: #9ca3af;
    }
}
"""