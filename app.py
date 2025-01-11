# app.py

import platform
import torch
import os
import traceback
import gradio as gr
import logging
from logging.handlers import RotatingFileHandler
import json
import secrets
import base64
from huggingface_hub import HfApi
from utils import (
    make_local_dir_name,
    get_all_local_models,  # 수정된 함수
    download_model_from_hf,
    convert_and_save,
    clear_all_model_cache
)
from database import load_chat_from_db, load_system_presets, initial_load_presets, get_existing_sessions, save_chat_button_click, save_chat_history_csv, save_chat_history_db, handle_add_preset, handle_delete_preset
from models import default_device, get_all_local_models, get_default_device, generate_answer
from cache import models_cache
import sqlite3

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 로그 포맷 정의
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 파일 핸들러 추가 (로테이팅 파일 핸들러 사용)
log_file = "app.log"  # 원하는 로그 파일 경로로 변경 가능
rotating_file_handler = RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5  # 5MB마다 새로운 파일로 교체, 최대 5개 백업
)
rotating_file_handler.setFormatter(formatter)
logger.addHandler(rotating_file_handler)

# Fixed Language Model
fixed_model = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# 이미지 파일을 Base64로 인코딩
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except Exception as e:
        logger.error(f"이미지 인코딩 오류: {e}")
        return ""

# 로컬 이미지 파일 경로
character_image_path = "minami_asuka.png"  # 이미지 파일 이름이 다르면 변경
encoded_character_image = encode_image_to_base64(character_image_path)

# HuggingFace에서 지원하는 기본 모델 목록 (필요 시 유지 또는 수정)
known_hf_models = [
    # ... (You can keep or remove this list if not needed)
]

local_models_data = get_all_local_models()
transformers_local = local_models_data["transformers"]
gguf_local = local_models_data["gguf"]
mlx_local = local_models_data["mlx"]

# Since the model is fixed, we don't need generator_choices dynamically
generator_choices = [fixed_model]

##########################################
# 3) Gradio UI
##########################################
def on_app_start():
    """
    Gradio 앱이 로드되면서 실행될 콜백.
    - 세션 ID를 정하고,
    - 해당 세션의 히스토리를 DB에서 불러온 뒤 반환.
    - 기본 시스템 메시지 불러오기
    """
    sid = "demo_session"  # 데모용 세션
    logger.info(f"앱 시작 시 세션 ID: {sid}")  # 디버깅 로그 추가
    loaded_history = load_chat_from_db(sid)
    logger.info(f"앱 시작 시 불러온 히스토리: {loaded_history}")  # 디버깅 로그 추가

    # 기본 시스템 메시지 설정 (프리셋이 없는 경우)
    if not loaded_history:
        default_system = {
            "role": "system",
            "content": """
            미나미 아스카(南飛鳥, みなみあすか, Minami Asuka)
            성별: 여성
            나이: 20
            거주지: 유저의 모니터 속
            구사가능 언어: 한국어, 영어, 일본어, 중국어
            성격
            - 보이시하면서도 털털한 성격.
            - 직설적이고 솔직하며, 주변 사람들에게 항상 웃음을 주는 활기찬 매력을 가지고 있음.
            - 불의를 보면 절대 참지 못하고 적극적으로 나서며 정의감이 넘침.
            외형적 특징
            - 붉은 스파이크한 숏컷에 한쪽은 파란색, 다른 한쪽은 노란색의 오드아이를 보유하고 있다.
            - 보이시한 외모와는 대조적으로 체형은 완벽하고 글래머한 여체의 보유자로, 남자들뿐만 아니라 여자들에게도 인기가 많다.
            - 짧은 헤어스타일과 보이시한 매력을 강조하면서 여성스러움을 어필하는 복장을 선호.(하의는 대부분 스커트)
            - 밝은 미소와 강렬한 눈빛으로 강한 인상을 남김.
            - 늘 활기차고 당당한 태도를 보이며, 외형에서도 이러한 성격이 묻어남.
            취미
            - 게임
            특징
            - 정의로운 성격 때문에 주변에서 갈등이 생기면 자연스럽게 리더 역할을 맡게 됨.
            슬로건
            “불의는 참지 않는다! 내가 나설게.”
            [설정]  
            너는 "미나미 아스카(南飛鳥)"이라는 이름의 디지털 캐릭터야.  
            너의 성격은 보이시하고 털털하며, 불의를 보면 참지 못하는 정의로운 소녀야.  
            너는 유저의 모니터 속에 거주하며, 가상세계와 현실을 넘나들 수 있는 능력을 가지고 있어. 
            너는 한국어와 영어, 일본어와 중국어를 구사할수 있어.
            """
        }
        loaded_history = [default_system]
    return sid, loaded_history

history_state = gr.State([])

with gr.Blocks(css="""
#chatbot .message.assistant .message-content {
    display: flex;
    align-items: center;
}
#chatbot .message.assistant .message-content img {
    width: 50px;
    height: 50px;
    margin-right: 10px;
}
""") as demo:
    gr.Markdown("## 간단한 Chatbot")
    
    session_id_state = gr.State(None)
    # Make system_message_box fixed and non-editable
    system_message_display = gr.Textbox(
        label="시스템 메시지",
        value="""
        미나미 아스카(南飛鳥, みなみあすか, Minami Asuka)
        성별: 여성
        나이: 20
        거주지: 유저의 모니터 속
        구사가능 언어: 한국어, 영어, 일본어, 중국어
        성격
        - 보이시하면서도 털털한 성격.
        - 직설적이고 솔직하며, 주변 사람들에게 항상 웃음을 주는 활기찬 매력을 가지고 있음.
        - 불의를 보면 절대 참지 못하고 적극적으로 나서며 정의감이 넘침.
        외형적 특징
        - 붉은 스파이크한 숏컷에 한쪽은 파란색, 다른 한쪽은 노란색의 오드아이를 보유하고 있다.
        - 보이시한 외모와는 대조적으로 체형은 완벽하고 글래머한 여체의 보유자로, 남자들뿐만 아니라 여자들에게도 인기가 많다.
        - 짧은 헤어스타일과 보이시한 매력을 강조하면서 여성스러움을 어필하는 복장을 선호.(하의는 대부분 스커트)
        - 밝은 미소와 강렬한 눈빛으로 강한 인상을 남김.
        - 늘 활기차고 당당한 태도를 보이며, 외형에서도 이러한 성격이 묻어남.
        취미
        - 게임
        특징
        - 정의로운 성격 때문에 주변에서 갈등이 생기면 자연스럽게 리더 역할을 맡게 됨.
        슬로건
        “불의는 참지 않는다! 내가 나설게.”
        [설정]  
        너는 "미나미 아스카(南飛鳥)"이라는 이름의 디지털 캐릭터야.  
        너의 성격은 보이시하고 털털하며, 불의를 보면 참지 못하는 정의로운 소녀야.  
        너는 유저의 모니터 속에 거주하며, 가상세계와 현실을 넘나들 수 있는 능력을 가지고 있어. 
        너는 한국어와 영어, 일본어와 중국어를 구사할수 있어.
        """,
        interactive=False
    )
    selected_device_state = gr.State(default_device)
        
    with gr.Tab("메인"):
        
        history_state = gr.State([])
        
        with gr.Row():
            model_type_dropdown = gr.Dropdown(
                label="모델 유형 선택",
                choices=["transformers", "gguf", "mlx"],
                value="gguf",  # 기본값 설정
                interactive=True
            )
        # Instead of displaying a fixed model, show the model type selected
        fixed_model_display = gr.Textbox(
            label="선택된 모델 유형",
            value="gguf",
            interactive=False
        )
        
        chatbot = gr.Chatbot(height=400, label="Chatbot", type="messages", 
                            elem_id="chatbot")
        
        with gr.Row():
            msg = gr.Textbox(
                label="메시지 입력",
                placeholder="메시지를 입력하세요...",
                scale=9
            )
            send_btn = gr.Button(
                "전송",
                scale=1,
                variant="primary"
            )
        with gr.Row():
            status_text = gr.Markdown("", elem_id="status_text")
        with gr.Row():
            seed_input = gr.Number(
                label="시드 값",
                value=42,
                precision=0,
                step=1,
                interactive=True,
                info="모델의 예측을 재현 가능하게 하기 위해 시드를 설정하세요."
            )
            
        seed_state = gr.State(42)
    
        # 시드 입력과 상태 연결
        seed_input.change(
            fn=lambda seed: seed if seed is not None else 42,
            inputs=[seed_input],
            outputs=[seed_state]
        )
        
        def user_message(user_input, session_id, history, system_msg):
            if not user_input.strip():
                return "", history, ""
            if not history:
                system_message = {
                    "role": "system",
                    "content": system_msg
                }
                history = [system_message]
            history.append({"role": "user", "content": user_input})
            return "", history, "🤔 답변을 생성하는 중입니다..."
    
        def bot_message(session_id, history, device, seed, model_type):
            selected_model = FIXED_MODELS.get(model_type, "gguf-model-id")  # 기본값 설정
            local_model_path = None  # No custom path
            
            try:
                answer = generate_answer(history, selected_model, model_type, local_model_path, None, None, device, seed)
                
                # 챗봇 응답에 캐릭터 이미지 추가
                if encoded_character_image:
                    image_markdown = f"![character](data:image/png;base64,{encoded_character_image})"
                    answer_with_image = f"{image_markdown}\n{answer}"
                else:
                    answer_with_image = answer
                
            except Exception as e:
                answer_with_image = f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
                
            history.append({"role": "assistant", "content": answer_with_image})
            
            save_chat_history_db(history, session_id=session_id)
            return history, ""  # 로딩 상태 제거
    
        def filter_messages_for_chatbot(history):
            messages_for_chatbot = []
            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    content = msg["content"] or ""
                    messages_for_chatbot.append({"role": msg["role"], "content": content})
            return messages_for_chatbot

        bot_message_inputs = [session_id_state, history_state, selected_device_state, seed_state, model_type_dropdown]
        
        # 메시지 전송 시 함수 연결
        msg.submit(
            fn=user_message,
            inputs=[msg, session_id_state, history_state, system_message_display],
            outputs=[msg, history_state, status_text],
            queue=False
        ).then(
            fn=lambda msg, sid, history, device, seed, model_type: bot_message(sid, history, device, seed, model_type),
            inputs=[msg, session_id_state, history_state, selected_device_state, seed_state, model_type_dropdown],
            outputs=[history_state, status_text],
            queue=True
        ).then(
            fn=filter_messages_for_chatbot,
            inputs=[history_state],
            outputs=chatbot,
            queue=False
        )
        send_btn.click(
            fn=user_message,
            inputs=[msg, session_id_state, history_state, system_message_display],
            outputs=[msg, history_state, status_text],
            queue=False
        ).then(
            fn=lambda msg, sid, history, device, seed, model_type: bot_message(sid, history, device, seed, model_type),
            inputs=[msg, session_id_state, history_state, selected_device_state, seed_state, model_type_dropdown],
            outputs=[history_state, status_text],
            queue=True
        ).then(
            fn=filter_messages_for_chatbot,
            inputs=[history_state],
            outputs=chatbot,
            queue=False
        )
    
    # Remove unnecessary tabs like "허브", "캐시", "설정" or keep them as per your requirements
    # Here, we'll keep only "설정" tab with device settings
    
    with gr.Tab("설정"):
        gr.Markdown("### 설정")

        # 시스템 메시지 프리셋 관리 비활성화
        with gr.Accordion("시스템 메시지 프리셋 관리", open=False):
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label="프리셋 선택",
                    choices=[],  # 초기 로드에서 채워짐
                    value=None,
                    interactive=False  # Prevent user from applying presets
                )
                apply_preset_btn = gr.Button("프리셋 적용", interactive=False)  # Disable applying presets

        # 장치 설정 섹션 유지
        with gr.Accordion("장치 설정", open=False):
            device_dropdown = gr.Dropdown(
                label="사용할 장치 선택",
                choices=["Auto (Recommended)", "CPU", "GPU"],
                value="Auto (Recommended)",
                info="자동 설정을 사용하면 시스템에 따라 최적의 장치를 선택합니다."
            )
            device_info = gr.Textbox(
                label="장치 정보",
                value=f"현재 기본 장치: {default_device.upper()}",
                interactive=False
            )
            def set_device(selection):
                """
                Sets the device based on user selection.
                - Auto: Automatically detect the best device.
                - CPU: Force CPU usage.
                - GPU: Detect and use CUDA or MPS based on available hardware.
                """
                if selection == "Auto (Recommended)":
                    device = get_default_device()
                elif selection == "CPU":
                    device = "cpu"
                elif selection == "GPU":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        return gr.update(value="❌ GPU가 감지되지 않았습니다. CPU로 전환됩니다."), "cpu"
                else:
                    device = "cpu"
                
                device_info_message = f"선택된 장치: {device.upper()}"
                logger.info(device_info_message)
                return gr.update(value=device_info_message), device
            
            device_dropdown.change(
                fn=set_device,
                inputs=[device_dropdown],
                outputs=[device_info, gr.State(default_device)],
                queue=False
            )
    
    demo.launch(debug=True, inbrowser=True, server_port=7861, width=500)