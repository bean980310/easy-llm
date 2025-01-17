import gradio as gr
from src.common.translations import _, translation_manager
from typing import Any, Optional, Dict, List
from src.common.state import app_state

def register_ui_components(components: Dict[str, Any]) -> None:
    """UI 컴포넌트들을 전역 상태에 등록"""
    for key, component in components.items():
        setattr(app_state, key, component)