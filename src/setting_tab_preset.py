import gradio as gr
from database import preset_exists, handle_add_preset, load_system_presets, save_chat_history_db

import logging

from src.preset_images import PRESET_IMAGES

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def on_add_preset_click(name, content):
    if preset_exists(name.strip()):
        # 프리셋이 이미 존재하면 덮어쓰기 확인을 요청
        return "", gr.update(visible=True), gr.update(visible=True), "⚠️ 해당 프리셋이 이미 존재합니다. 덮어쓰시겠습니까?"
    else:
        success, message = handle_add_preset(name.strip(), content.strip())
        if success:
            return message, gr.update(visible=False), gr.update(visible=False), ""
        else:
            return message, gr.update(visible=False), gr.update(visible=False), ""
        
 # 프리셋 적용 이벤트 수정
def apply_preset(name, session_id, history, language=None):
    if not name:
        return "❌ 적용할 프리셋을 선택해주세요.", history, gr.update()
                
    if language is None:
        language = "ko"
                    
    presets = load_system_presets(language)
    content = presets.get(name, "")
    if not content:
        return "❌ 선택한 프리셋에 내용이 없습니다.", history, gr.update()
        
    # 현재 세션의 히스토리를 초기화하고 시스템 메시지 추가
    new_history = [{"role": "system", "content": content}]
    success = save_chat_history_db(new_history, session_id=session_id)
    if not success:
        return "❌ 프리셋 적용 중 오류가 발생했습니다.", history, gr.update()
    logger.info(f"'{name}' 프리셋을 적용하여 세션을 초기화했습니다.")
                
    image_path = PRESET_IMAGES.get(name)
    return f"✅ '{name}' 프리셋이 적용되었습니다.", new_history, gr.update(value=content), gr.update(value=image_path) if image_path else gr.update()