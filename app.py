# app.py

import platform
import torch
import os
import traceback
import torch
import gradio as gr
import logging
from logging.handlers import RotatingFileHandler
import json
import secrets
from huggingface_hub import HfApi
import sqlite3
from utils import (
    make_local_dir_name,
    get_all_local_models,  # 수정된 함수
    download_model_from_hf,
    convert_and_save,
    clear_all_model_cache
)
from database import (
    load_chat_from_db, 
    load_system_presets, 
    initial_load_presets, 
    get_existing_sessions, 
    save_chat_button_click, 
    save_chat_history_csv, 
    save_chat_history_db, 
    handle_add_preset, 
    handle_delete_preset, 
    preset_exists,
    get_preset_choices,
    delete_session_history,
    delete_all_sessions)
from models import default_device, get_all_local_models, get_default_device, generate_answer, generate_stable_diffusion_prompt_cached
from cache import models_cache
from translations import translation_manager, _, detect_system_language

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

api_models = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o"
    # 필요 시 추가
]
    
# HuggingFace에서 지원하는 기본 모델 목록
known_hf_models = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-11B-Vision",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Bllossom/llama-3.2-Korean-Bllossom-3B",
    "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
    "Bllossom/llama-3.1-Korean-Bllossom-Vision-8B",
    "THUDM/glm-4-9b-chat",
    "THUDM/glm-4-9b-chat-hf",
    "THUDM/glm-4-9b-chat-1m",
    "THUDM/glm-4-9b-chat-1m-hf",
    "THUDM/glm-4v-9b",
    "huggyllama/llama-7b",
    "OrionStarAI/Orion-14B-Base",
    "OrionStarAI/Orion-14B-Chat",
    "OrionStarAI/Orion-14B-LongChat",
    "CohereForAI/aya-23-8B",
    "CohereForAI/aya-23-35B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "EleutherAI/polyglot-ko-1.3b"
]
    
local_models_data = get_all_local_models()
transformers_local = local_models_data["transformers"]
gguf_local = local_models_data["gguf"]
mlx_local = local_models_data["mlx"]
    
generator_choices = api_models + transformers_local + gguf_local + mlx_local + ["사용자 지정 모델 경로 변경"]
generator_choices = list(dict.fromkeys(generator_choices))  # 중복 제거
generator_choices = sorted(generator_choices)  # 정렬

default_language = detect_system_language()

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
            "content": system_message_box.value  # 현재 시스템 메시지 박스의 값을 사용
        }
        loaded_history = [default_system]
    return sid, loaded_history

def process_message(user_input, session_id, history, system_msg, selected_model, custom_path, image, api_key, device, seed):
    """
    사용자 메시지를 처리하고 봇 응답을 생성하는 통합 함수.

    Args:
        user_input (str): 사용자가 입력한 메시지.
        session_id (str): 현재 세션 ID.
        history (list): 채팅 히스토리.
        system_msg (str): 시스템 메시지.
        selected_model (str): 선택된 모델 이름.
        custom_path (str): 사용자 지정 모델 경로.
        image (PIL.Image or None): 이미지 입력 (비전 모델용).
        api_key (str or None): API 키 (API 모델용).
        device (str): 사용할 장치 ('cpu', 'cuda', 등).
        seed (int): 시드 값.

    Returns:
        tuple: 업데이트된 입력 필드, 히스토리, Chatbot 컴포넌트, 상태 메시지.
    """
    if not user_input.strip():
        # 빈 입력일 경우 아무 것도 하지 않음
        return "", history, filter_messages_for_chatbot(history), ""

    if not history:
        # 히스토리가 없을 경우 시스템 메시지로 초기화
        system_message = {
            "role": "system",
            "content": system_msg
        }
        history = [system_message]

    # 사용자 메시지 추가
    history.append({"role": "user", "content": user_input})

    try:
        # 봇 응답 생성
        answer = generate_answer(
            history=history,
            selected_model=selected_model,
            model_type="api" if selected_model in api_models else (
                "transformers" if selected_model in transformers_local else (
                    "gguf" if selected_model in gguf_local else (
                        "mlx" if selected_model in mlx_local else "transformers"
                    )
                )
            ),
            local_model_path=custom_path if selected_model == "사용자 지정 모델 경로 변경" else None,
            image_input=image,  # image 인자 전달
            api_key=api_key,
            device=device,
            seed=seed
        )

        # 응답을 히스토리에 추가
        history.append({"role": "assistant", "content": answer})

        # 데이터베이스에 히스토리 저장
        save_chat_history_db(history, session_id=session_id)

        # 상태 메시지 초기화
        status = ""

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        history.append({"role": "assistant", "content": f"❌ 오류 발생: {str(e)}"})
        status = "❌ 오류가 발생했습니다. 로그를 확인하세요."

    # 업데이트된 히스토리를 Chatbot 형식으로 변환
    chatbot_history = filter_messages_for_chatbot(history)

    return "", history, chatbot_history, status
    

def filter_messages_for_chatbot(history):
    """
    채팅 히스토리를 Gradio Chatbot 컴포넌트에 맞는 형식으로 변환

    Args:
        history (list): 전체 채팅 히스토리

    Returns:
        list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    messages_for_chatbot = []
    for msg in history:
        if msg["role"] in ("user", "assistant"):
            content = msg["content"] or ""
            messages_for_chatbot.append({"role": msg["role"], "content": content})
    return messages_for_chatbot

def reset_session(history, chatbot, system_message_default):
    """
    특정 세션을 초기화하는 함수.
    """
    try:
        # 데이터베이스에서 해당 세션 삭제
        success = delete_session_history("demo_session")  # 현재 세션 ID 사용
        if not success:
            return gr.update(), history, filter_messages_for_chatbot(history), "❌ 세션 초기화에 실패했습니다."

        # 히스토리 초기화 (기본 시스템 메시지로 재설정)
        default_system = {
            "role": "system",
            "content": system_message_default  # 기본 시스템 메시지 내용
        }
        new_history = [default_system]

        # 데이터베이스에 초기 히스토리 저장
        save_chat_history_db(new_history, session_id="demo_session")  # 기본 세션 ID 사용

        # 챗봇 UI 업데이트
        chatbot_history = filter_messages_for_chatbot(new_history)

        return "", new_history, chatbot_history, "✅ 세션이 초기화되었습니다."

    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}", exc_info=True)
        return "", history, filter_messages_for_chatbot(history), f"❌ 세션 초기화 중 오류가 발생했습니다: {str(e)}"

def reset_all_sessions(history, chatbot, system_message_default):
    """
    모든 세션을 초기화하는 함수.
    """
    try:
        # 데이터베이스에서 모든 세션 삭제
        success = delete_all_sessions()
        if not success:
            return gr.update(), history, filter_messages_for_chatbot(history), "❌ 모든 세션 초기화에 실패했습니다."

        # 히스토리 초기화 (기본 시스템 메시지로 재설정)
        default_system = {
            "role": "system",
            "content": system_message_default  # 기본 시스템 메시지 내용
        }
        new_history = [default_system]

        # 데이터베이스에 초기 히스토리 저장
        save_chat_history_db(new_history, session_id="demo_session")  # 기본 세션 ID 사용

        # 챗봇 UI 업데이트
        chatbot_history = filter_messages_for_chatbot(new_history)

        return "", new_history, chatbot_history, "✅ 모든 세션이 초기화되었습니다."

    except Exception as e:
        logger.error(f"Error resetting all sessions: {str(e)}", exc_info=True)
        return "", history, filter_messages_for_chatbot(history), f"❌ 모든 세션 초기화 중 오류가 발생했습니다: {str(e)}"
    
with gr.Blocks() as demo:
    history_state = gr.State([])
    overwrite_state = gr.State(False) 

    # 단일 history_state와 selected_device_state 정의 (중복 제거)
    custom_model_path_state = gr.State("")
    session_id_state = gr.State()
    selected_device_state = gr.State(default_device)
    seed_state = gr.State(42)  # 시드 상태 전역 정의
    selected_language_state = gr.State(default_language)
    
    reset_confirmation = gr.State(False)
    reset_all_confirmation = gr.State(False)
    
    title=gr.Markdown(f"## {_('main_title')}")
    language_dropdown = gr.Dropdown(
        label=_('language_select'),
        choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
        value=translation_manager.get_language_display_name(default_language),
        interactive=True,
        info=_('language_info')
    )
    
    system_message_box = gr.Textbox(
        label=_("system_message"),
        value=_("system_message_default"),
        placeholder=_("system_message_placeholder")
    )
    
    with gr.Tab(_("tab_main")):
        initial_choices = api_models + transformers_local + gguf_local + mlx_local + ["사용자 지정 모델 경로 변경"]
        initial_choices = list(dict.fromkeys(initial_choices))
        initial_choices = sorted(initial_choices)  # 정렬 추가
        
        with gr.Row():
            model_type_dropdown = gr.Radio(
                label=_("model_type_label"),
                choices=["all", "transformers", "gguf", "mlx"],
                value="all",
            )
        
        model_dropdown = gr.Dropdown(
            label=_("model_select_label"),
            choices=initial_choices,
            value=initial_choices[0] if len(initial_choices) > 0 else None,
        )
        
        api_key_text = gr.Textbox(
            label=_("api_key_label"),
            placeholder="sk-...",
            visible=False  # 기본적으로 숨김
        )
        image_info = gr.Markdown("", visible=False)
        with gr.Column():
            with gr.Row():
                image_input = gr.Image(label=_("image_upload_label"), type="pil", visible=False)
                chatbot = gr.Chatbot(height=400, label="Chatbot", type="messages")
            with gr.Row():
                msg = gr.Textbox(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder"),
                    scale=9
                )
                send_btn = gr.Button(
                    value=_("send_button"),
                    scale=1,
                    variant="primary"
                )
            with gr.Row():
                status_text = gr.Markdown("", elem_id="status_text")
            with gr.Row():
                seed_input = gr.Number(
                    label=_("seed_label"),
                    value=42,
                    precision=0,
                    step=1,
                    interactive=True,
                    info=_("seed_info")
                )
                        
            # 시드 입력과 상태 연결
            seed_input.change(
                fn=lambda seed: seed if seed is not None else 42,
                inputs=[seed_input],
                outputs=[seed_state]
            )
            
            with gr.Row():
                reset_btn = gr.Button(
                    value=_("reset_session_button"),  # "세션 초기화"에 해당하는 번역 키
                    variant="secondary",
                    scale=1
                )
                reset_all_btn = gr.Button(
                    value=_("reset_all_sessions_button"),  # "모든 세션 초기화"에 해당하는 번역 키
                    variant="secondary",
                    scale=1
                )
                
            # 초기화 확인 메시지 및 버튼 추가 (숨김 상태로 시작)
            with gr.Row(visible=False) as reset_confirm_row:
                reset_confirm_msg = gr.Markdown("⚠️ **정말로 현재 세션을 초기화하시겠습니까? 모든 대화 기록이 삭제됩니다.**")
                reset_yes_btn = gr.Button("✅ 예", variant="danger")
                reset_no_btn = gr.Button("❌ 아니요", variant="secondary")

            with gr.Row(visible=False) as reset_all_confirm_row:
                reset_all_confirm_msg = gr.Markdown("⚠️ **정말로 모든 세션을 초기화하시겠습니까? 모든 대화 기록이 삭제됩니다.**")
                reset_all_yes_btn = gr.Button("✅ 예", variant="danger")
                reset_all_no_btn = gr.Button("❌ 아니요", variant="secondary")
        
        
        # 함수: OpenAI API Key와 사용자 지정 모델 경로 필드의 가시성 제어
        def toggle_api_key_visibility(selected_model):
            """
            OpenAI API Key 입력 필드의 가시성을 제어합니다.
            """
            api_visible = selected_model in api_models
            return gr.update(visible=api_visible)

        def toggle_image_input_visibility(selected_model):
            """
            이미지 입력 필드의 가시성을 제어합니다.
            """
            image_visible = (
                "vision" in selected_model.lower() or
                "qwen2-vl" in selected_model.lower() or
                selected_model in [
                    "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
                    "THUDM/glm-4v-9b",
                    "openbmb/MiniCPM-Llama3-V-2_5"
                ]
            )
            return gr.update(visible=image_visible)
        
        # 모델 선택 변경 시 가시성 토글
        model_dropdown.change(
            fn=lambda selected_model: (
                toggle_api_key_visibility(selected_model),
                toggle_image_input_visibility(selected_model)
            ),
            inputs=[model_dropdown],
            outputs=[api_key_text, image_input]
        )
        def update_model_list(selected_type):
            local_models_data = get_all_local_models()
            transformers_local = local_models_data["transformers"]
            gguf_local = local_models_data["gguf"]
            mlx_local = local_models_data["mlx"]
            
            # "전체 목록"이면 => API 모델 + 모든 로컬 모델 + "사용자 지정 모델 경로 변경"
            if selected_type == "all":
                all_models = api_models + transformers_local + gguf_local + mlx_local + ["사용자 지정 모델 경로 변경"]
                # 중복 제거 후 정렬
                all_models = sorted(list(dict.fromkeys(all_models)))
                return gr.update(choices=all_models, value=all_models[0] if all_models else None)
            
            # 개별 항목이면 => 해당 유형의 로컬 모델 + "사용자 지정 모델 경로 변경"만
            if selected_type == "transformers":
                updated_list = transformers_local + ["사용자 지정 모델 경로 변경"]
            elif selected_type == "gguf":
                updated_list = gguf_local + ["사용자 지정 모델 경로 변경"]
            elif selected_type == "mlx":
                updated_list = mlx_local + ["사용자 지정 모델 경로 변경"]
            else:
                # 혹시 예상치 못한 값이면 transformers로 처리(또는 None)
                updated_list = transformers_local + ["사용자 지정 모델 경로 변경"]
            
            updated_list = sorted(list(dict.fromkeys(updated_list)))
            return gr.update(choices=updated_list, value=updated_list[0] if updated_list else None)
        
        model_type_dropdown.change(
            fn=update_model_list,
            inputs=[model_type_dropdown],
            outputs=[model_dropdown]
        )
        
        # .load()를 사용해, 페이지 로딩시 자동으로 on_app_start()가 실행되도록 연결
        demo.load(
            fn=on_app_start,
            inputs=[],
            outputs=[session_id_state, history_state],
            queue=False
        )
        
        bot_message_inputs = [session_id_state, history_state, model_dropdown, custom_model_path_state, image_input, api_key_text, selected_device_state, seed_state]
        
        def change_language(selected_lang):
            """언어 변경 처리 함수"""
            lang_map = {
                "한국어": "ko",
                "日本語": "ja",
                "中文(简体)": "zh_CN",
                "中文(繁體)": "zh_TW",
                "English": "en"
            }
            lang_code = lang_map.get(selected_lang, "ko")
            translation_manager.set_language(lang_code)

            return [
                gr.update(value=f"## {_('main_title')}"),
                gr.update(label=_('language_select'),
                info=_('language_info')),
                gr.update(
                    label=_("system_message"),
                    value=_("system_message_default"),
                    placeholder=_("system_message_placeholder")
                ),
                gr.update(label=_("model_type_label")),
                gr.update(label=_("model_select_label")),
                gr.update(label=_("api_key_label")),
                gr.update(label=_("image_upload_label")),
                gr.update(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder")
                ),
                gr.update(value=_("send_button")),
                gr.update(label=_("seed_label"), info=_("seed_info")),
                gr.update(value=_("reset_session_button"))
            ]

        # 언어 변경 이벤트 연결
        language_dropdown.change(
            fn=change_language,
            inputs=[language_dropdown],
            outputs=[
                title,
                language_dropdown,
                system_message_box,
                model_type_dropdown,
                model_dropdown,
                api_key_text,
                image_input,
                msg,
                send_btn,
                seed_input,
                reset_btn
            ]
        )
        # 메시지 전송 시 함수 연결
        msg.submit(
            fn=process_message,
            inputs=[
                msg,  # 사용자 입력
                session_id_state,
                history_state,
                system_message_box,
                model_dropdown,
                custom_model_path_state,
                image_input,
                api_key_text,
                selected_device_state,
                seed_state
            ],
            outputs=[
                msg,            # 사용자 입력 필드 초기화
                history_state,  # 히스토리 업데이트
                chatbot,        # Chatbot UI 업데이트
                status_text     # 상태 메시지 업데이트
            ],
            queue=False
        ).then(
            fn=filter_messages_for_chatbot,
            inputs=[history_state],
            outputs=chatbot,
            queue=False
        )

        send_btn.click(
            fn=process_message,
            inputs=[
                msg, 
                session_id_state, 
                history_state, 
                system_message_box, 
                model_dropdown, 
                custom_model_path_state, 
                image_input, 
                api_key_text, 
                selected_device_state, 
                seed_state
            ],
            outputs=[
                msg, 
                history_state, 
                chatbot, 
                status_text
            ],
            queue=False
        ).then(
            fn=filter_messages_for_chatbot,            # 추가된 부분
            inputs=[history_state],
            outputs=chatbot,                           # chatbot에 최종 전달
            queue=False
        )
        
        # 초기화 버튼 클릭 시 확인 메시지 표시
        reset_btn.click(
            fn=lambda: True,
            inputs=[],
            outputs=[reset_confirmation],
            queue=False
        ).then(
            fn=lambda confirm: gr.update(visible=confirm),
            inputs=[reset_confirmation],
            outputs=[reset_confirm_row],
            queue=False
        )

        reset_all_btn.click(
            fn=lambda: True,
            inputs=[],
            outputs=[reset_all_confirmation],
            queue=False
        ).then(
            fn=lambda confirm: gr.update(visible=confirm),
            inputs=[reset_all_confirmation],
            outputs=[reset_all_confirm_row],
            queue=False
        )

        # "예" 버튼 클릭 시 세션 초기화 수행
        reset_yes_btn.click(
            fn=reset_session,  # 이미 정의된 reset_session 함수
            inputs=[history_state, chatbot, system_message_box],
            outputs=[
                msg,            # 사용자 입력 필드 초기화
                history_state,  # 히스토리 업데이트
                chatbot,        # Chatbot UI 업데이트
                status_text     # 상태 메시지 업데이트
            ],
            queue=False
        ).then(
            fn=lambda: gr.update(visible=False),  # 확인 메시지 숨김
            inputs=[],
            outputs=[reset_confirm_row],
            queue=False
        )

        # "아니요" 버튼 클릭 시 확인 메시지 숨김
        reset_no_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[reset_confirm_row],
            queue=False
        )

        # "모든 세션 초기화"의 "예" 버튼 클릭 시 모든 세션 초기화 수행
        reset_all_yes_btn.click(
            fn=reset_all_sessions,  # 이미 정의된 reset_all_sessions 함수
            inputs=[history_state, chatbot, system_message_box],
            outputs=[
                msg,            # 사용자 입력 필드 초기화
                history_state,  # 히스토리 업데이트
                chatbot,        # Chatbot UI 업데이트
                status_text     # 상태 메시지 업데이트
            ],
            queue=False
        ).then(
            fn=lambda: gr.update(visible=False),  # 확인 메시지 숨김
            inputs=[],
            outputs=[reset_all_confirm_row],
            queue=False
        )

        # "모든 세션 초기화"의 "아니요" 버튼 클릭 시 확인 메시지 숨김
        reset_all_no_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[reset_all_confirm_row],
            queue=False
        )
            
    with gr.Tab(_("download_tab")):
        download_title=gr.Markdown(f"""### {_("download_title")}
        {_("download_description")}
        {_("download_description_detail")}""")
        
        with gr.Column():
            # 다운로드 모드 선택 (라디오 버튼)
            download_mode = gr.Radio(
                label=_("download_mode_label"),
                choices=["Predefined", "Custom Repo ID"],
                value="Predefined",
                container=True,
            )
            # 모델 선택/입력 영역
            with gr.Column(visible=True) as predefined_column:
                predefined_dropdown = gr.Dropdown(
                    label=_("model_select_label"),
                    choices=sorted(known_hf_models),
                    value=known_hf_models[0] if known_hf_models else None,
                    info=_("model_select_info")
                )
                
            with gr.Column(visible=False) as custom_column:
                custom_repo_id_box = gr.Textbox(
                    label="Custom Model ID",
                    placeholder=_("custom_model_id_placeholder"),
                    info=_("custom_model_id_info")
                )
                
            # 다운로드 설정
            with gr.Row():
                with gr.Column(scale=2):
                    target_path = gr.Textbox(
                        label=_("save_path_label"),
                        placeholder="./models/my-model",
                        value="",
                        interactive=True,
                        info=_("save_path_info")
                    )
                with gr.Column(scale=1):
                    use_auth = gr.Checkbox(
                        label=_("auth_required_label"),
                        value=False,
                        info=_("auth_required_info")
                    )
            
            with gr.Column(visible=False) as auth_column:
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    placeholder="hf_...",
                    type="password",
                    info=_("hf_token_info")
                )
            
            # 다운로드 버튼과 진행 상태
            with gr.Row():
                download_btn = gr.Button(
                    value=_("download_start_button"),
                    variant="primary",
                    scale=2
                )
                cancel_btn = gr.Button(
                    value=_("download_cancel_button"),
                    variant="stop",
                    scale=1,
                    interactive=False
                )
                
            # 상태 표시
            download_status = gr.Markdown("")
            progress_bar = gr.Progress(
                track_tqdm=True,  # tqdm progress bars를 추적
            )
            
            # 다운로드 결과와 로그
            with gr.Accordion(_("download_details_label"), open=False):
                download_info = gr.TextArea(
                    label=_("download_log_label"),
                    interactive=False,
                    max_lines=10,
                    autoscroll=True
                )

        # UI 동작 제어를 위한 함수들
        def toggle_download_mode(mode):
            """다운로드 모드에 따라 UI 컴포넌트 표시/숨김"""
            return [
                gr.update(visible=(mode == "Predefined")),  # predefined_column
                gr.update(visible=(mode == "Custom Repo ID"))  # custom_column
            ]
            
        download_mode.change(
            fn=toggle_download_mode,
            inputs=[download_mode],
            outputs=[predefined_column, custom_column]
        )

        def toggle_auth(use_auth_val):
            """인증 필요 여부에 따라 토큰 입력창 표시/숨김"""
            return gr.update(visible=use_auth_val)
        
        use_auth.change(
            fn=toggle_auth,
            inputs=[use_auth],
            outputs=[auth_column]
        )

        def download_with_progress(mode, predefined_choice, custom_repo, target_dir, use_auth_val, token):
            try:
                repo_id = predefined_choice if mode == "Predefined" else custom_repo.strip()
                if not repo_id:
                    yield (
                        _("download_error_no_model"),  # status
                        gr.update(interactive=True),  # download_btn
                        gr.update(interactive=False),  # cancel_btn
                        "다운로드가 시작되지 않았습니다.",  # download_info
                        gr.Dropdown.update()
                    )
                    return  

                # 모델 유형 결정
                if "gguf" in repo_id.lower():
                    model_type = "gguf"
                elif "mlx" in repo_id.lower():
                    model_type = "mlx"
                else:
                    model_type = "transformers"

                # 진행 상태 초기화
                yield (
                    _("download_preparing"),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    f"모델: {repo_id}\n준비 중...",
                    gr.Dropdown.update()
                )

                # 실제 다운로드 수행
                yield (
                    _("download_in_progress"),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    "다운로드를 진행 중입니다...",
                    gr.Dropdown.update()
                )
                result = download_model_from_hf(
                    repo_id,
                    target_dir or os.path.join("./models", model_type, make_local_dir_name(repo_id)),
                    model_type=model_type,
                    token=token if use_auth_val else None
                )

                # 다운로드 완료 후 UI 업데이트
                yield (
                    _("download_complete") if "실패" not in result else _("download_failed"),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    result,
                    gr.Dropdown.update(choices=sorted(api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"] + ["사용자 지정 모델 경로 변경"]))
                )

            except Exception as e:
                yield (
                    _("download_error"),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    f"오류: {str(e)}\n\n{traceback.format_exc()}",
                    gr.Dropdown.update()
                )

        # Gradio에서 async 함수를 지원하는지 확인 후, 연결
        download_btn.click(
            fn=download_with_progress,
            inputs=[
                download_mode,
                predefined_dropdown,
                custom_repo_id_box,
                target_path,
                use_auth,
                hf_token
            ],
            outputs=[
                download_status,
                download_btn,
                cancel_btn,
                download_info,
                model_dropdown
            ]
        )
        
        def change_language(selected_lang: str):
            """언어 변경 처리 함수"""
            lang_map = {
                "한국어": "ko",
                "日本語": "ja",
                "中文(简体)": "zh_CN",
                "中文(繁體)": "zh_TW",
                "English": "en"
            }
            lang_code = lang_map.get(selected_lang, "ko")
            translation_manager.set_language(lang_code)
            
            return [
                gr.update(value=f"""### {_("download_title")}
                {_("download_description")}
                {_("download_description_detail")}"""),
                gr.update(label=_("download_mode_label")),
                gr.update(label=_("model_select_label"), info=_('model_select_info')),
                gr.update(
                    placeholder=_("custom_model_id_placeholder"),
                    info=_("custom_model_id_info")
                ),
                gr.update(
                    label=_("save_path_label"),
                    info=_("save_path_info")
                ),
                gr.update(
                    label=_("auth_required_label"),
                    info=_("auth_required_info")
                ),
                gr.update(
                    label=_("hf_token_label"),
                    info=_("hf_token_info")
                ),
                gr.update(value=_("download_start_button")),
                gr.update(value=_("download_cancel_button")),
                gr.update(label=_("download_log_label"))
            ]
        language_dropdown.change(
            fn=change_language,
            inputs=[language_dropdown],
            outputs=[
                download_title,
                download_mode,
                predefined_dropdown,
                custom_repo_id_box,
                target_path,
                use_auth,
                hf_token,
                download_btn,
                cancel_btn,
                download_info
        ]
    )
    with gr.Tab(_("hub_tab_title")):
        hub_title=gr.Markdown(f"""### {_("hub_description")}
        {_("hub_description_detail")}
        {_("hub_search_description")}""")
        
        with gr.Row():
            search_box = gr.Textbox(
                label=_("hub_search_label"),
                placeholder=_("hub_search_placeholder"),
                scale=4
            )
            search_btn = gr.Button(_("hub_search_button"), scale=1)
            
        with gr.Row():
            with gr.Column(scale=1):
                model_type_filter = gr.Dropdown(
                    label=_("hub_model_type_label"),
                    choices=["All", "Text Generation", "Vision", "Audio", "Other"],
                    value="All"
                )
                language_filter = gr.Dropdown(
                    label=_("hub_language_label"),
                    choices=["All", "Korean", "English", "Chinese", "Japanese", "Multilingual"],
                    value="All"
                )
                library_filter = gr.Dropdown(
                    label=_("hub_library_label"),
                    choices=["All", "Transformers", "GGUF", "MLX"],
                    value="All"
                )
            with gr.Column(scale=3):
                model_list = gr.Dataframe(
                    headers=["Model ID", "Description", "Downloads", "Likes"],
                    label=_("hub_model_list_label"),
                    interactive=False
                )
        
        with gr.Row():
            selected_model = gr.Textbox(
                label=_("hub_selected_model_label"),
                interactive=False
            )
            
        # 다운로드 설정
        with gr.Row():
            with gr.Column(scale=2):
                target_path = gr.Textbox(
                    label=_("hub_save_path_label"),
                    placeholder=_("hub_save_path_placeholder"),
                    value="",
                    interactive=True,
                    info=_("hub_save_path_info")
                )
            with gr.Column(scale=1):
                use_auth = gr.Checkbox(
                    label=_("hub_auth_required_label"),
                    value=False,
                    info=_("hub_auth_required_info")
                )
        
        with gr.Column(visible=False) as auth_column:
            hf_token = gr.Textbox(
                label=_("hub_token_label"),
                placeholder=_("hub_token_placeholder"),
                type="password",
                info=_("hub_token_info")
            )
        
        # 다운로드 버튼과 진행 상태
        with gr.Row():
            download_btn = gr.Button(
                _("hub_download_button"),
                variant="primary",
                scale=2
            )
            cancel_btn = gr.Button(
                _("hub_cancel_button"),
                variant="stop",
                scale=1,
                interactive=False
            )
            
        # 상태 표시
        download_status = gr.Markdown("")
        progress_bar = gr.Progress(track_tqdm=True)
        
        # 다운로드 결과와 로그
        with gr.Accordion(_("hub_details_label"), open=False):
            download_info = gr.TextArea(
                label=_("hub_download_log_label"),
                interactive=False,
                max_lines=10,
                autoscroll=True
            )

        def search_models(query, model_type, language, library):
            """허깅페이스 허브에서 모델 검색"""
            try:
                api = HfApi()
                # 검색 필터 구성
                filter_str = ""
                if model_type != "All":
                    filter_str += f"task_{model_type.lower().replace(' ', '_')}"
                if language != "All":
                    if filter_str:
                        filter_str += " AND "
                    filter_str += f"language_{language.lower()}"
                if library != "All":
                    filter_str += f"library_{library.lower()}"

                # 모델 검색 수행
                models = api.list_models(
                    filter=filter_str if filter_str else None,
                    limit=100,
                    sort="lastModified",
                    direction=-1
                )

                filtered_models = [model for model in models if query.lower() in model.id.lower()]

                # 결과 데이터프레임 구성
                model_list_data = []
                for model in filtered_models:
                    description = model.cardData.get('description', '') if model.cardData else 'No description available.'
                    short_description = (description[:100] + "...") if len(description) > 100 else description
                    model_list_data.append([
                        model.id,
                        short_description,
                        model.downloads,
                        model.likes
                    ])
                return model_list_data
            except Exception as e:
                logger.error(f"모델 검색 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
                return [["오류 발생", str(e), "", ""]]

        def select_model(evt: gr.SelectData, data):
            """데이터프레임에서 모델 선택"""
            selected_model_id = data.at[evt.index[0], "Model ID"]  # 선택된 행의 'Model ID' 컬럼 값
            return selected_model_id
        
        def toggle_auth(use_auth_val):
            """인증 필요 여부에 따라 토큰 입력창 표시/숨김"""
            return gr.update(visible=use_auth_val)
        
        def download_model_with_progress(model_id, target_dir, use_auth_val, token, progress=gr.Progress()):
            """진행률 표시와 함께 모델 다운로드 수행"""
            try:
                if not model_id:
                    yield (
                        _('download_error_no_model'),
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        "다운로드가 시작되지 않았습니다.",
                        gr.Dropdown.update()
                    )
                    return
                
                # 모델 유형 결정
                model_type = "transformers"  # 기본값
                if "gguf" in model_id.lower():
                    model_type = "gguf"
                elif "mlx" in model_id.lower():
                    model_type = "mlx"

                # 진행 상태 초기화
                progress(0, desc="준비 중...")
                yield (
                    _('download_preparing'),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    f"모델: {model_id}\n준비 중...",
                    gr.Dropdown.update()
                )

                # 실제 다운로드 수행
                progress(0.5, desc=_('download_in_progress'))
                result = download_model_from_hf(
                    model_id,
                    target_dir or os.path.join("./models", model_type, make_local_dir_name(model_id)),
                    model_type=model_type,
                    token=token if use_auth_val else None
                )

                # 다운로드 완료 후 UI 업데이트
                progress(1.0, desc="완료")
                local_models_data = get_all_local_models()
                local_models = (
                    local_models_data["transformers"] +
                    local_models_data["gguf"] +
                    local_models_data["mlx"]
                )
                new_choices = api_models + local_models + ["사용자 지정 모델 경로 변경"]
                new_choices = list(dict.fromkeys(new_choices))
                new_choices = sorted(new_choices)

                yield (
                    _('download_complete') if "실패" not in result else _('download_failed'),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    result,
                    gr.Dropdown.update(choices=new_choices)
                )

            except Exception as e:
                yield (
                    _('download_error'),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    f"오류: {str(e)}\n\n{traceback.format_exc()}",
                    gr.Dropdown.update()
                )

        # 이벤트 연결
        search_btn.click(
            fn=search_models,
            inputs=[search_box, model_type_filter, language_filter, library_filter],
            outputs=model_list
        )
        
        model_list.select(
            fn=select_model,
            inputs=[model_list],
            outputs=selected_model
        )
        
        use_auth.change(
            fn=toggle_auth,
            inputs=use_auth,
            outputs=auth_column
        )
        
        download_btn.click(
            fn=download_model_with_progress,
            inputs=[
                selected_model,
                target_path,
                use_auth,
                hf_token
            ],
            outputs=[
                download_status,
                download_btn,
                cancel_btn,
                download_info,
                model_dropdown
            ]
        )
        
        def change_language(selected_lang: str):
            """언어 변경 처리 함수"""
            lang_map = {
                "한국어": "ko",
                "日本語": "ja",
                "中文(简体)": "zh_CN",
                "中文(繁體)": "zh_TW",
                "English": "en"
            }
            lang_code = lang_map.get(selected_lang, "ko")
            translation_manager.set_language(lang_code)
            
            return [
                gr.update(value=f"""### {_("hub_description")}
                    {_("hub_description_detail")}
                    {_("hub_search_description")}"""),
                gr.update(label=_("hub_search_label"),
                        placeholder=_("hub_search_placeholder")),
                gr.update(value=_("hub_search_button")),
                gr.update(label=_("hub_model_type_label"),
                        choices=["All", "Text Generation", "Vision", "Audio", "Other"]),
                gr.update(label=_("hub_language_label"),
                        choices=["All", "Korean", "English", "Chinese", "Japanese", "Multilingual"]),
                gr.update(label=_("hub_library_label"),
                        choices=["All", "Transformers", "GGUF", "MLX"]),
                gr.update(label=_("hub_model_list_label")),
                gr.update(label=_("hub_selected_model_label")),
                gr.update(label=_("hub_save_path_label"),
                        placeholder=_("hub_save_path_placeholder"),
                        info=_("hub_save_path_info")),
                gr.update(label=_("hub_auth_required_label"),
                        info=_("hub_auth_required_info")),
                gr.update(label=_("hub_token_label"),
                        placeholder=_("hub_token_placeholder"),
                        info=_("hub_token_info")),
                gr.update(value=_("hub_download_button")),
                gr.update(value=_("hub_cancel_button")),
                gr.update(label=_("hub_download_log_label"))
            ]
            
        language_dropdown.change(
            fn=change_language,
            inputs=[language_dropdown],
            outputs=[
                hub_title,
                search_box,
                search_btn,
                model_type_filter,
                language_filter,
                library_filter,
                model_list,
                selected_model,
                target_path,
                use_auth,
                hf_token,
                download_btn,
                cancel_btn,
                download_info
            ]
        )
        
    with gr.Tab("캐시"):
        with gr.Row():
            with gr.Column():
                refresh_button = gr.Button("모델 목록 새로고침")
                refresh_info = gr.Textbox(label="새로고침 결과", interactive=False)
            with gr.Column():
                clear_all_btn = gr.Button("모든 모델 캐시 삭제")
                clear_all_result = gr.Textbox(label="결과", interactive=False)

        def refresh_model_list():
            """
            수동 새로고침 시 호출되는 함수.
            - 새로 scan_local_models()
            - DropDown 모델 목록 업데이트
            """
            # 새로 스캔
            new_local_models = get_all_local_models()
            # 새 choices: API 모델 + 로컬 모델 + 사용자 지정 모델 경로 변경
            api_models = [
                "gpt-3.5-turbo",
                "gpt-4o-mini",
                "gpt-4o"
                # 필요 시 추가
            ]
            local_models = new_local_models["transformers"] + new_local_models["gguf"] + new_local_models["mlx"]
            new_choices = api_models + local_models + ["사용자 지정 모델 경로 변경"]
            new_choices = list(dict.fromkeys(new_choices))
            new_choices = sorted(new_choices)  # 정렬 추가
            # 반환값:
            logger.info("모델 목록 새로고침")
            return gr.update(choices=new_choices), "모델 목록을 새로고침 했습니다."
            
        refresh_button.click(
            fn=refresh_model_list,
            inputs=[],
            outputs=[model_dropdown, refresh_info]
        )
        clear_all_btn.click(
            fn=clear_all_model_cache,
            inputs=[],
            outputs=clear_all_result
        )
    with gr.Tab("유틸리티"):
        gr.Markdown("### 모델 비트 변환기")
        gr.Markdown("Transformers와 BitsAndBytes를 사용하여 모델을 8비트로 변환합니다.")
        
        with gr.Row():
            model_id = gr.Textbox(label="HuggingFace 모델 ID", placeholder="예: gpt2")
            output_dir = gr.Textbox(label="저장 디렉토리", placeholder="./converted_models/gpt2_8bit")
        with gr.Row():
            quant_type = gr.Radio(choices=["float8", "int8", "int4"], label="변환 유형", value="int8")
        with gr.Row():
            push_to_hub = gr.Checkbox(label="Hugging Face Hub에 푸시", value=False)
        
        convert_button = gr.Button("모델 변환 시작")
        output = gr.Textbox(label="결과")
        
        convert_button.click(fn=convert_and_save, inputs=[model_id, output_dir, push_to_hub, quant_type], outputs=output)
    with gr.Tab("설정"):
        gr.Markdown("### 설정")

        # 사용자 지정 모델 경로 설정 섹션
        with gr.Accordion("사용자 지정 모델 경로 설정", open=False):
            custom_path_text = gr.Textbox(
                label="사용자 지정 모델 경로",
                placeholder="./models/custom-model",
            )
            apply_custom_path_btn = gr.Button("경로 적용")

            # custom_path_text -> custom_model_path_state 저장
            def update_custom_path(path):
                return path

            apply_custom_path_btn.click(
                fn=update_custom_path,
                inputs=[custom_path_text],
                outputs=[custom_model_path_state]
            )

         # 시스템 메시지 프리셋 관리 섹션 추가
        with gr.Accordion("시스템 메시지 프리셋 관리", open=False):
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label="프리셋 선택",
                    choices=[],  # 초기 로드에서 채워짐
                    value=None,
                    interactive=True
                )
                apply_preset_btn = gr.Button("프리셋 적용")
        
            with gr.Row():
                preset_name = gr.Textbox(
                    label="새 프리셋 이름",
                    placeholder="예: 친절한 비서",
                    interactive=True
                )
                preset_content = gr.Textbox(
                    label="프리셋 내용",
                    placeholder="프리셋으로 사용할 시스템 메시지를 입력하세요.",
                    lines=4,
                    interactive=True
                )
        
            with gr.Row():
                add_preset_btn = gr.Button("프리셋 추가", variant="primary")
                delete_preset_btn = gr.Button("프리셋 삭제", variant="secondary")
        
            preset_info = gr.Textbox(
                label="프리셋 관리 결과",
                interactive=False
            )
        
            # 덮어쓰기 확인을 위한 컴포넌트 추가 (처음에는 숨김)
            with gr.Row():
                confirm_overwrite_btn = gr.Button("확인", variant="primary", visible=False)
                cancel_overwrite_btn = gr.Button("취소", variant="secondary", visible=False)
        
            overwrite_message = gr.Textbox(
                label="덮어쓰기 메시지",
                value="",
                interactive=False
            )
        
            # 프리셋 Dropdown 초기화
            demo.load(
                fn=initial_load_presets,
                inputs=[],
                outputs=[preset_dropdown],
                queue=False
            )
        
            # 프리셋 추가 버튼 클릭 시
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
            
            add_preset_btn.click(
                fn=on_add_preset_click,
                inputs=[preset_name, preset_content],
                outputs=[preset_info, confirm_overwrite_btn, cancel_overwrite_btn, overwrite_message]
            )
        
            # 덮어쓰기 확인 버튼 클릭 시
            def confirm_overwrite(name, content):
                success, message = handle_add_preset(name.strip(), content.strip(), overwrite=True)
                if success:
                    return message, gr.update(visible=False), gr.update(visible=False), ""
                else:
                    return message, gr.update(visible=False), gr.update(visible=False), ""
            
            confirm_overwrite_btn.click(
                fn=confirm_overwrite,
                inputs=[preset_name, preset_content],
                outputs=[preset_info, confirm_overwrite_btn, cancel_overwrite_btn, overwrite_message]
            )
        
            # 덮어쓰기 취소 버튼 클릭 시
            def cancel_overwrite():
                return "❌ 덮어쓰기가 취소되었습니다.", gr.update(visible=False), gr.update(visible=False), ""
            
            cancel_overwrite_btn.click(
                fn=cancel_overwrite,
                inputs=[],
                outputs=[preset_info, confirm_overwrite_btn, cancel_overwrite_btn, overwrite_message]
            )
        
            # 프리셋 삭제 버튼 클릭 시
            def on_delete_preset_click(name):
                if not name:
                    return "❌ 삭제할 프리셋을 선택해주세요.", gr.update(choices=get_preset_choices())
                success, message = handle_delete_preset(name)
                if success:
                    return message, gr.update(choices=get_preset_choices())
                else:
                    return message, gr.update(choices=get_preset_choices())
        
            delete_preset_btn.click(
                fn=on_delete_preset_click,
                inputs=[preset_dropdown],
                outputs=[preset_info, preset_dropdown]
            )
        
            # 프리셋 적용 이벤트 수정
            def apply_preset(name, session_id, history):
                if not name:
                    return "❌ 적용할 프리셋을 선택해주세요.", history, gr.update()
                presets = load_system_presets()
                content = presets.get(name, "")
                if not content:
                    return "❌ 선택한 프리셋에 내용이 없습니다.", history, gr.update()
        
                # 현재 세션의 히스토리를 초기화하고 시스템 메시지 추가
                new_history = [{"role": "system", "content": content}]
                logger.info(f"'{name}' 프리셋을 적용하여 세션을 초기화했습니다.")
                return f"✅ '{name}' 프리셋이 적용되었습니다.", new_history, gr.update(value=content)
        
            apply_preset_btn.click(
                fn=apply_preset,
                inputs=[preset_dropdown, session_id_state, history_state],
                outputs=[preset_info, history_state, system_message_box]
            ).then(
                fn=filter_messages_for_chatbot,
                inputs=[history_state],
                outputs=chatbot
            )
        # 채팅 기록 저장 섹션
        with gr.Accordion("채팅 기록 저장", open=False):
            save_button = gr.Button("채팅 기록 저장", variant="secondary")
            save_info = gr.Textbox(label="저장 결과", interactive=False)

            save_csv_button = gr.Button("채팅 기록 CSV 저장", variant="secondary")
            save_csv_info = gr.Textbox(label="CSV 저장 결과", interactive=False)

            save_db_button = gr.Button("채팅 기록 DB 저장", variant="secondary")
            save_db_info = gr.Textbox(label="DB 저장 결과", interactive=False)

            def save_chat_button_click_csv(history):
                if not history:
                    return "채팅 이력이 없습니다."
                saved_path = save_chat_history_csv(history)
                if saved_path is None:
                    return "❌ 채팅 기록 CSV 저장 실패"
                else:
                    return f"✅ 채팅 기록 CSV가 저장되었습니다: {saved_path}"
                
            def save_chat_button_click_db(history):
                if not history:
                    return "채팅 이력이 없습니다."
                ok = save_chat_history_db(history, session_id="demo_session")
                if ok:
                    return f"✅ DB에 채팅 기록이 저장되었습니다 (session_id=demo_session)"
                else:
                    return "❌ DB 저장 실패"

            save_csv_button.click(
                fn=save_chat_button_click_csv,
                inputs=[history_state],
                outputs=save_csv_info
            )

            # save_button이 클릭되면 save_chat_button_click 실행
            save_button.click(
                fn=save_chat_button_click,
                inputs=[history_state],
                outputs=save_info
            )
            
            save_db_button.click(
                fn=save_chat_button_click_db,
                inputs=[history_state],
                outputs=save_db_info
            )

        # 채팅 히스토리 재로드 섹션
        with gr.Accordion("채팅 히스토리 재로드", open=False):
            upload_json = gr.File(label="대화 JSON 업로드", file_types=[".json"])
            load_info = gr.Textbox(label="로딩 결과", interactive=False)
            
            def load_chat_from_json(json_file):
                """
                업로드된 JSON 파일을 파싱하여 history_state에 주입
                """
                if not json_file:
                    return [], "파일이 없습니다."
                try:
                    with open(json_file.name, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        return [], "JSON 구조가 올바르지 않습니다. (list 형태가 아님)"
                    # data를 그대로 history_state로 반환
                    return data, "✅ 대화가 로딩되었습니다."
                except Exception as e:
                    logger.error(f"JSON 로드 오류: {e}")
                    return [], f"❌ 로딩 실패: {e}"

            upload_json.change(
                fn=load_chat_from_json,
                inputs=[upload_json],
                outputs=[history_state, load_info]
            )

        # 세션 관리 섹션
        with gr.Accordion("세션 관리", open=False):
            gr.Markdown("### 세션 관리")
            with gr.Row():
                refresh_sessions_btn = gr.Button("세션 목록 갱신")
                existing_sessions_dropdown = gr.Dropdown(
                    label="기존 세션 목록",
                    choices=[],  # 초기에는 비어 있다가, 버튼 클릭 시 갱신
                    value=None,
                    interactive=True
                )
            
            with gr.Row():
                create_new_session_btn = gr.Button("새 세션 생성")
                apply_session_btn = gr.Button("세션 적용")
                delete_session_btn = gr.Button("세션 삭제")
            
            # 삭제 확인을 위한 컴포넌트 추가
            confirm_delete_checkbox = gr.Checkbox(
                label="정말로 이 세션을 삭제하시겠습니까?",
                value=False,
                interactive=True,
                visible=False  # 기본적으로 숨김
            )
            confirm_delete_btn = gr.Button(
                "삭제 확인",
                variant="stop",
                visible=False  # 기본적으로 숨김
            )
            
            session_manage_info = gr.Textbox(
                label="세션 관리 결과",
                interactive=False
            )
            
            current_session_display = gr.Textbox(
                label="현재 세션 ID",
                value="",
                interactive=False
            )

            session_id_state.change(
                fn=lambda sid: f"현재 세션: {sid}" if sid else "세션 없음",
                inputs=[session_id_state],
                outputs=[current_session_display]
            )
            
            def refresh_sessions():
                """
                세션 목록 갱신: DB에서 세션 ID들을 불러와서 Dropdown에 업데이트
                """
                sessions = get_existing_sessions()
                logger.info(f"가져온 세션 목록: {sessions}")  # 디버깅용 로그 추가
                if not sessions:
                    return gr.update(choices=[], value=None), "DB에 세션이 없습니다."
                return gr.update(choices=sessions, value=sessions[0]), "세션 목록을 불러왔습니다."
            
            def create_new_session():
                """
                새 세션 ID를 생성하고 session_id_state에 반영.
                """
                new_sid = secrets.token_hex(8)  # 새 세션 ID 생성
                logger.info(f"새 세션 생성됨: {new_sid}")
                
                # 기본 시스템 메시지 설정
                system_message = {
                    "role": "system",
                    "content": system_message_box.value  # 현재 시스템 메시지 박스의 값을 사용
                }
                
                # 새 세션에 시스템 메시지 저장
                save_chat_history_db([system_message], session_id=new_sid)
                
                return new_sid, f"새 세션 생성: {new_sid}"
        
            def apply_session(chosen_sid):
                """
                Dropdown에서 선택된 세션 ID로, DB에서 history를 불러오고, session_id_state를 갱신
                """
                if not chosen_sid:
                    return [], None, "세션 ID를 선택하세요."
                loaded_history = load_chat_from_db(chosen_sid)
                logger.info(f"불러온 히스토리: {loaded_history}")  # 디버깅 로그 추가
                # history_state에 반영하고, session_id_state도 업데이트
                return loaded_history, chosen_sid, f"세션 {chosen_sid}이 적용되었습니다."
            def delete_session(chosen_sid, current_sid):
                """
                선택된 세션을 DB에서 삭제합니다.
                현재 활성 세션은 삭제할 수 없습니다.
                """
                if not chosen_sid:
                    return "❌ 삭제할 세션을 선택하세요.", False, gr.update()
                
                if chosen_sid == current_sid:
                    return "❌ 현재 활성 세션은 삭제할 수 없습니다.", False, gr.update()
                
                try:
                    conn = sqlite3.connect("chat_history.db")
                    cursor = conn.cursor()
                    # 삭제하기 전에 세션이 존재하는지 확인
                    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (chosen_sid,))
                    count = cursor.fetchone()[0]
                    if count == 0:
                        return f"❌ 세션 '{chosen_sid}'이(가) DB에 존재하지 않습니다.", False, gr.update(visible=False)
                    
                    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (chosen_sid,))
                    conn.commit()
                    conn.close()
                    logger.info(f"세션 삭제 완료: {chosen_sid}")
                    return f"✅ 세션 '{chosen_sid}'이(가) 삭제되었습니다.", False, gr.update(visible=False)
                except sqlite3.OperationalError as oe:
                    logger.error(f"DB 운영 오류: {oe}")
                    return f"❌ DB 운영 오류 발생: {oe}", False, gr.update(visible=False)
                except Exception as e:
                    logger.error(f"세션 삭제 오류: {e}")
                    return f"❌ 세션 삭제 실패: {e}", False, gr.update(visible=False)
            
            # 버튼 이벤트 연결
            def initiate_delete():
                return gr.update(visible=True), gr.update(visible=True)
            
            # 삭제 확인 버튼 클릭 시 실제 삭제 수행
            def confirm_delete(chosen_sid, current_sid, confirm):
                if not confirm:
                    return "❌ 삭제가 취소되었습니다.", False, gr.update(visible=False)
                return delete_session(chosen_sid, current_sid)
    
            refresh_sessions_btn.click(
                fn=refresh_sessions,
                inputs=[],
                outputs=[existing_sessions_dropdown, session_manage_info]
            )
            
            create_new_session_btn.click(
                fn=create_new_session,
                inputs=[],
                outputs=[session_id_state, session_manage_info]
            ).then(
                fn=lambda: [],  # 새 세션 생성 시 히스토리 초기화
                inputs=[],
                outputs=[history_state]
            ).then(
                fn=filter_messages_for_chatbot,  # 초기화된 히스토리를 Chatbot에 반영
                inputs=[history_state],
                outputs=[chatbot]
            )
            
            apply_session_btn.click(
                fn=apply_session,
                inputs=[existing_sessions_dropdown],
                outputs=[history_state, session_id_state, session_manage_info]
            ).then(
                fn=filter_messages_for_chatbot, # (2) 불러온 history를 Chatbot 형식으로 필터링
                inputs=[history_state],
                outputs=chatbot                 # (3) Chatbot 업데이트
            )
            
            delete_session_btn.click(
                fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
                inputs=[],
                outputs=[confirm_delete_checkbox, confirm_delete_btn]
            )
            
            # 삭제 확인 버튼 클릭 시 실제 삭제 수행
            confirm_delete_btn.click(
                fn=confirm_delete,
                inputs=[existing_sessions_dropdown, session_id_state, confirm_delete_checkbox],
                outputs=[session_manage_info, confirm_delete_checkbox, confirm_delete_btn]
            ).then(
                fn=refresh_sessions,  # 세션 삭제 후 목록 새로고침
                inputs=[],
                outputs=[existing_sessions_dropdown, session_manage_info]
            )
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
    with gr.Tab("SD Prompt 생성"):
        gr.Markdown("# Stable Diffusion 프롬프트 생성기")
        
        with gr.Row():
            user_input_sd = gr.Textbox(
                label="이미지 설명",
                placeholder="예: 해질녘의 아름다운 해변 풍경",
                lines=2
            )
            generate_prompt_btn = gr.Button("프롬프트 생성")
        
        with gr.Row():
            selected_model_sd = gr.Dropdown(
                label="언어 모델 선택",
                choices=generator_choices,
                value="gpt-3.5-turbo",
                interactive=True
            )
            model_type_sd = gr.Dropdown(
                label="모델 유형",
                choices=["api", "transformers", "gguf", "mlx"],
                value="api",
                interactive=False  # 자동 설정되므로 사용자가 변경하지 못하도록 설정
            )
        
        api_key_sd = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            visible=True
        )
        
        prompt_output_sd = gr.Textbox(
            label="생성된 프롬프트",
            placeholder="여기에 생성된 프롬프트가 표시됩니다...",
            lines=4,
            interactive=False
        )
        
        # 사용자 지정 모델 경로 입력 필드
        custom_model_path_sd = gr.Textbox(
            label="사용자 지정 모델 경로",
            placeholder="./models/custom-model",
            visible=False
        )
        
        # 모델 선택 시 모델 유형 자동 설정 및 API Key 필드 가시성 제어
        def update_model_type(selected_model):
            if selected_model in api_models:
                model_type = "api"
                api_visible = True
                custom_visible = False
            elif selected_model in transformers_local:
                model_type = "transformers"
                api_visible = False
                custom_visible = False
            elif selected_model in gguf_local:
                model_type = "gguf"
                api_visible = False
                custom_visible = False
            elif selected_model in mlx_local:
                model_type = "mlx"
                api_visible = False
                custom_visible = False
            elif selected_model == "사용자 지정 모델 경로 변경":
                model_type = "transformers"  # 기본값 설정 (필요 시 수정)
                api_visible = False
                custom_visible = True
            else:
                model_type = "transformers"
                api_visible = False
                custom_visible = False
            
            return gr.update(value=model_type), gr.update(visible=api_visible), gr.update(visible=custom_visible)
        
        selected_model_sd.change(
            fn=update_model_type,
            inputs=[selected_model_sd],
            outputs=[model_type_sd, api_key_sd, custom_model_path_sd]
        )
        
        # 프롬프트 생성 버튼 클릭 시 함수 연결
        generate_prompt_btn.click(
            fn=generate_stable_diffusion_prompt_cached,
            inputs=[user_input_sd, selected_model_sd, model_type_sd, custom_model_path_sd, api_key_sd],
            outputs=prompt_output_sd
        )

demo.launch(debug=True, inbrowser=True, server_port=7861, width=800)