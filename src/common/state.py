from dataclasses import dataclass
from typing import Any, Optional, Dict, List
import gradio as gr

@dataclass
class SharedState:
    """공유 상태를 관리하는 클래스"""
    model_dropdown: Optional[gr.Dropdown] = None
    language_dropdown: Optional[gr.Dropdown] = None

@dataclass
class AppState:
    """애플리케이션 전역 상태 관리"""
    session_id: str = ""
    history: List[Dict[str, Any]] = None
    custom_model_path: str = ""
    selected_device: str = "cpu"
    selected_language: str = "ko"
    seed: int = 42

    # UI 컴포넌트 참조
    session_select_dropdown: Optional[gr.Dropdown] = None
    chatbot: Optional[gr.Chatbot] = None
    system_message_box: Optional[gr.Textbox] = None
    profile_image: Optional[gr.Image] = None
    
# 전역 상태 인스턴스
shared_state = SharedState()

app_state = AppState()