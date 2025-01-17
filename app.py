# app.py

import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import gradio as gr
import logging
from logging.handlers import RotatingFileHandler
import json
import secrets
import sqlite3
from utils import (
    get_all_local_models,  # 수정된 함수
    convert_and_save,
    clear_all_model_cache
)
from database import (
    initialize_database,
    add_system_preset,
    delete_system_preset,
    ensure_demo_session,
    load_chat_from_db, 
    load_system_presets, 
    get_existing_sessions, 
    save_chat_button_click, 
    save_chat_history_csv, 
    save_chat_history_db, 
    handle_add_preset, 
    handle_delete_preset, 
    get_preset_choices,
    insert_default_presets)
from models import default_device, get_all_local_models, generate_stable_diffusion_prompt_cached
from cache import models_cache
from translations import translation_manager, _, detect_system_language, TranslationManager
from persona_speech_manager import PersonaSpeechManager

from src.main_tab import (
    api_models, 
    transformers_local, 
    gguf_local, 
    mlx_local,
    MainTab,
    generator_choices,
    characters,
    get_speech_manager,
    update_system_message_and_profile,
)
from src.download_tab import create_download_tab
from src.setting_tab_preset import on_add_preset_click, apply_preset
from src.device_setting import set_device

from presets import __all__ as preset_modules
import json

from css import css

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

default_language = detect_system_language()

# translation_manager = TranslationManager(default_language)

main_tab=MainTab()

def initialize_speech_manager():
    return PersonaSpeechManager(translation_manager, characters)

def handle_character_change(selected_character, language, speech_manager: PersonaSpeechManager):
    try:
        speech_manager.set_character_and_language(selected_character, language)
        system_message = speech_manager.get_system_message()
        return system_message, gr.update(value=speech_manager.characters[selected_character]["profile_image"])
    except ValueError as e:
        logger.error(str(e))
        return "❌ 선택한 캐릭터가 유효하지 않습니다.", gr.update(value=None)
    
def load_presets_from_files(presets_dir: str) -> List[Dict[str, Any]]:
    """
    presets 디렉토리 내의 모든 프리셋 파일을 로드하여 프리셋 리스트를 반환합니다.
    각 프리셋은 여러 언어로 정의될 수 있습니다.
    """
    presets = []
    presets_path = Path(presets_dir)
    for preset_file in presets_path.glob("*.py"):
        module_name = preset_file.stem
        try:
            module = importlib.import_module(f"presets.{module_name}")
            # __all__ 에 정의된 프리셋 변수들 로드
            for preset_var in getattr(module, "__all__", []):
                preset = getattr(module, preset_var, None)
                if preset:
                    # 각 언어별로 분리하여 추가
                    for lang, content in preset.items():
                        presets.append({
                            "name": preset_var,
                            "language": lang,
                            "content": content.strip()
                        })
        except Exception as e:
            logger.error(f"프리셋 파일 {preset_file} 로드 중 오류 발생: {e}")
    return presets

def update_presets_on_start(presets_dir: str):
    """
    앱 시작 시 presets 디렉토리의 프리셋을 로드하고 데이터베이스를 업데이트합니다.
    """
    # 현재 데이터베이스에 저장된 프리셋 로드
    existing_presets = load_system_presets()  # {(name, language): content, ...}

    # 파일에서 로드한 프리셋
    loaded_presets = load_presets_from_files(presets_dir)

    loaded_preset_keys = set()
    for preset in loaded_presets:
        name = preset["name"]
        language = preset["language"]
        content = preset["content"]
        loaded_preset_keys.add((name, language))
        existing_content = existing_presets.get((name, language))

        if not existing_content:
            # 새로운 프리셋 추가
            success, message = add_system_preset(name, language, content)
            if success:
                logger.info(f"새 프리셋 추가: {name} ({language})")
            else:
                logger.warning(f"프리셋 추가 실패: {name} ({language}) - {message}")
        elif existing_content != content:
            # 기존 프리셋 내용 업데이트
            success, message = add_system_preset(name, language, content, overwrite=True)
            if success:
                logger.info(f"프리셋 업데이트: {name} ({language})")
            else:
                logger.warning(f"프리셋 업데이트 실패: {name} ({language}) - {message}")

    # 데이터베이스에 있지만 파일에는 없는 프리셋 삭제 여부 결정
    for (name, language) in existing_presets.keys():
        if (name, language) not in loaded_preset_keys:
            success, message = delete_system_preset(name, language)
            if success:
                logger.info(f"프리셋 삭제: {name} ({language})")
            else:
                logger.warning(f"프리셋 삭제 실패: {name} ({language}) - {message}")
                
def get_last_used_session():
    try:
        with sqlite3.connect("chat_history.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id
                FROM sessions
                WHERE last_activity IS NOT NULL
                ORDER BY last_activity DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None
    except Exception as e:
        logger.error(f"마지막 사용 세션 조회 오류: {e}")
        return None
                
##########################################
# 3) Gradio UI
##########################################
def initialize_app():
    """
    애플리케이션 초기화 함수.
    - 기본 프리셋 삽입
    - 세션 초기화
    """
    initialize_database()
    ensure_demo_session()
    insert_default_presets(translation_manager, overwrite=True)
    return on_app_start(default_language)

def on_app_start(language=None):  # language 매개변수에 기본값 설정
    """
    Gradio 앱이 로드되면서 실행될 콜백.
    """
    if language is None:
        language = default_language
        
    # (1) 마지막으로 사용된 세션 ID 조회
    last_sid = get_last_used_session()
    if last_sid:
        sid = last_sid
        logger.info(f"마지막 사용 세션: {sid}")
    else:
        sid = "demo_session"
        logger.info("마지막 사용 세션이 없어 demo_session 사용")
        
    loaded_history = load_chat_from_db(sid)
    logger.info(f"앱 시작 시 불러온 히스토리: {loaded_history}")
    
    sessions = get_existing_sessions()
    logger.info(f"불러온 세션 목록: {sessions}")

    presets = load_system_presets(language=language)
    logger.info(f"로드된 프리셋: {presets}")
    
    if not loaded_history:
        system_presets = load_system_presets(language)
        if len(system_presets) > 0:
            preset_name = list(system_presets.keys())[0]
            default_system = {
                "role": "system",
                "content": system_presets[preset_name]
            }
            loaded_history = [default_system]
        else:
            default_system = {
                "role": "system",
                "content": "당신은 유용한 AI 비서입니다."
            }
            loaded_history = [default_system]
            
    return (
        sid, 
        loaded_history,
        gr.update(choices=sessions, value=sid if sessions else None),
        f"현재 세션: {sid}"
    )

def on_character_and_language_select(character_name, language):
    """
    캐릭터와 언어 선택 시 호출되는 함수.
    - 캐릭터와 언어 설정 적용
    - 시스템 메시지 프리셋 업데이트
    """
    try:
        speech_manager_state.set_character_and_language(character_name, language)
        system_message = speech_manager_state.get_system_message()
        return system_message
    except ValueError as ve:
        logger.error(f"Character setting error: {ve}")
        return "시스템 메시지 로딩 중 오류가 발생했습니다."

def parse_args():
    parser = argparse.ArgumentParser(description="Easy-LLM Application Setting")
    
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Gradio 서버가 실행될 포트 번호를 지정합니다. (default: %(default)d)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="모델의 예측을 재현 가능하게 하기 위한 시드 값을 지정합니다. (default: %(default)d)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버깅 모드를 활성화합니다. (default: %(default)s)"
    )
    
    parser.add_argument(
        "--inbrowser",
        type=bool,
        default=True,
        help="Gradio 앱을 브라우저에서 실행합니다. (default: %(default)s)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="gradio 앱의 공유 링크를 생성합니다."
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default=default_language,
        choices=["ko", "ja", "en", "zh_CN", "zh_TW"],
        help="애플리케이션의 기본 언어를 지정합니다. (default: %(default)s)"
    )
    
    return parser.parse_args()


args=parse_args()

refresh_session_list=main_tab.refresh_sessions()

with gr.Blocks(css=css) as demo:
    speech_manager_state = gr.State(initialize_speech_manager)
    
    session_id, loaded_history, session_dropdown, session_label=on_app_start()
    last_sid_state=gr.State()
    history_state = gr.State(loaded_history)
    session_list_state = gr.State()
    overwrite_state = gr.State(False) 

    # 단일 history_state와 selected_device_state 정의 (중복 제거)
    custom_model_path_state = gr.State("")
    session_id_state = gr.State()
    selected_device_state = gr.State(default_device)
    seed_state = gr.State(args.seed)  # 시드 상태 전역 정의
    selected_language_state = gr.State(default_language)
    
    reset_confirmation = gr.State(False)
    reset_all_confirmation = gr.State(False)
    
    initial_choices = api_models + transformers_local + gguf_local + mlx_local
    initial_choices = list(dict.fromkeys(initial_choices))
    initial_choices = sorted(initial_choices)  # 정렬 추가
    
    with gr.Column(elem_classes="container"):
        with gr.Row(elem_classes="header-container"):
            with gr.Column(scale=3):
                title = gr.Markdown(f"## {_('main_title')}", elem_classes="title")
                session_select_info = gr.Markdown("선택된 세션이 표시됩니다.")
            with gr.Column(scale=1):
                language_dropdown = gr.Dropdown(
                    label=_('language_select'),
                    choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
                    value=translation_manager.get_language_display_name(default_language),
                    interactive=True,
                    info=_('language_info'),
                    container=False,
                    elem_classes="custom-dropdown"
                )
        with gr.Row(elem_classes="session-container"):
            session_select_dropdown = gr.Dropdown(
                label="세션 선택",
                choices=[],  # 앱 시작 시 혹은 별도의 로직으로 세션 목록을 채움
                value=None,
                interactive=True,
                container=False,
                scale=8,
                elem_classes="session-dropdown"
            )
            add_session_icon_btn = gr.Button("📝", elem_classes="icon-button", scale=1, variant="secondary")
            delete_session_icon_btn = gr.Button("🗑️", elem_classes="icon-button-delete", scale=1, variant="stop")
        
        with gr.Row(elem_classes="model-container"):
            with gr.Column(scale=5):
                model_type_dropdown = gr.Radio(
                    label=_("model_type_label"),
                    choices=["all", "transformers", "gguf", "mlx"],
                    value="all",
                    elem_classes="model-dropdown"
                )
            with gr.Column(scale=10):
                model_dropdown = gr.Dropdown(
                label=_("model_select_label"),
                choices=initial_choices,
                value=initial_choices[0] if len(initial_choices) > 0 else None,
                elem_classes="model-dropdown"
                )
                api_key_text = gr.Textbox(
                    label=_("api_key_label"),
                    placeholder="sk-...",
                    visible=False,
                    elem_classes="api-key-input"
                )
        with gr.Row(elem_classes="chat-interface"):
            with gr.Column(scale=7):
                system_message_box = gr.Textbox(
                    label=_("system_message"),
                    value=_("system_message_default"),
                    placeholder=_("system_message_placeholder"),
                    elem_classes="system-message"
                )
                
                chatbot = gr.Chatbot(
                    height=400, 
                    label="Chatbot", 
                    type="messages", 
                    elem_classes=["chat-messages"]
                )
                
                with gr.Row(elem_classes="input-area"):
                    msg = gr.Textbox(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder"),
                    scale=9,
                    show_label=False,
                    elem_classes="message-input"
                    )
                    send_btn = gr.Button(
                        value=_("send_button"),
                        scale=1,
                        variant="primary",
                        elem_classes="send-button"
                    )
                    image_input = gr.Image(label=_("image_upload_label"), type="pil", visible=False)
            with gr.Column(scale=3, elem_classes="side-panel"):
                profile_image = gr.Image(
                    label=_('profile_image_label'),
                    visible=True,
                    interactive=False,
                    show_label=True,
                    width=400,
                    height=400,
                    value=characters[list(characters.keys())[0]]["profile_image"],
                    elem_classes="profile-image"
                )
                character_dropdown = gr.Dropdown(
                    label="캐릭터 선택",
                    choices=list(characters.keys()),
                    value=list(characters.keys())[0],
                    interactive=True,
                    info="대화할 캐릭터를 선택하세요.",
                    elem_classes='profile-image'
                )
                with gr.Accordion("고급 설정", open=False):
                    seed_input = gr.Number(
                        label=_("seed_label"),
                        value=42,
                        precision=0,
                        step=1,
                        interactive=True,
                        info=_("seed_info"),
                        elem_classes="seed-input"
                    )
                    preset_dropdown = gr.Dropdown(
                        label="프리셋 선택",
                        choices=get_preset_choices(default_language),
                        value=list(get_preset_choices(default_language))[0] if get_preset_choices(default_language) else None,
                        interactive=True,
                        elem_classes="preset-dropdown"
                    )
                    change_preset_button = gr.Button("프리셋 변경")
                    character_conversation_dropdown = gr.CheckboxGroup(
                        label="대화할 캐릭터 선택",
                        choices=get_preset_choices(default_language),  # 추가 캐릭터 이름
                        value=list(get_preset_choices(default_language))[0] if get_preset_choices(default_language) else None,
                        interactive=True
                    )
                    start_conversation_button = gr.Button("대화 시작")
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
        with gr.Row(elem_classes="status-bar"):
            status_text = gr.Markdown("Ready", elem_id="status_text")
            image_info = gr.Markdown("", visible=False)

    # 아래는 변경 이벤트 등록
    def apply_session_immediately(chosen_sid):
        """
        메인탭에서 세션이 선택되면 바로 main_tab.apply_session을 호출해 세션 적용.
        """
        return main_tab.apply_session(chosen_sid)

    def init_session_dropdown(sessions):
        if not sessions:
            return gr.update(choices=[], value=None)
        return gr.update(choices=sessions, value=sessions[0])
        
    def create_session():
        # 실제로는 main_tab.create_new_session(...) 같은 함수 호출
        new_sid, info = main_tab.create_new_session(system_message_box.value)
        return new_sid, info
        
    add_session_icon_btn.click(
        fn=create_session,
        inputs=[],
        outputs=[]  # 필요하다면 session_id_state, session_select_dropdown 등 갱신
    ).then(
        fn=main_tab.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown]  # 세션 목록 즉시 갱신
    )
        
    def delete_selected_session(chosen_sid):
        # 선택된 세션을 삭제 (주의: None 또는 ""인 경우 처리)
        result_msg, _, updated_dropdown = main_tab.delete_session(chosen_sid, "demo_session")
        return result_msg, updated_dropdown
        
    delete_session_icon_btn.click(
        fn=lambda: delete_selected_session(session_select_dropdown.value),
        inputs=[],
        outputs=[]  # 필요 시 Textbox나 Dropdown 업데이트
    ).then(
        fn=main_tab.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown]
    )
                        
    # 시드 입력과 상태 연결
    seed_input.change(
        fn=lambda seed: seed if seed is not None else 42,
        inputs=[seed_input],
        outputs=[seed_state]
    )
            
    # 프리셋 변경 버튼 클릭 시 호출될 함수 연결
    change_preset_button.click(
        fn=main_tab.handle_change_preset,
        inputs=[preset_dropdown, history_state, selected_language_state],
        outputs=[history_state, system_message_box, profile_image]
    )
            
    character_dropdown.change(
        fn=update_system_message_and_profile,
        inputs=[character_dropdown, language_dropdown, speech_manager_state],
        outputs=[system_message_box, profile_image]
    )
        
    # 모델 선택 변경 시 가시성 토글
    model_dropdown.change(
        fn=lambda selected_model: (
            main_tab.toggle_api_key_visibility(selected_model),
            main_tab.toggle_image_input_visibility(selected_model)
        ),
        inputs=[model_dropdown],
        outputs=[api_key_text, image_input]
    )
        
    model_type_dropdown.change(
        fn=main_tab.update_model_list,
        inputs=[model_type_dropdown],
        outputs=[model_dropdown]
    )
        
    bot_message_inputs = [session_id_state, history_state, model_dropdown, custom_model_path_state, image_input, api_key_text, selected_device_state, seed_state]
        
    def update_character_languages(selected_language, selected_character):
        """
        인터페이스 언어에 따라 선택된 캐릭터의 언어를 업데이트합니다.
        """
        speech_manager = get_speech_manager(session_id_state)
        if selected_language in characters[selected_character]["languages"]:
            # 인터페이스 언어가 캐릭터의 지원 언어에 포함되면 해당 언어로 설정
            speech_manager.current_language = selected_language
        else:
            # 지원하지 않는 언어일 경우 기본 언어로 설정
            speech_manager.current_language = characters[selected_character]["default_language"]
        return gr.update()

        
    def change_language(selected_lang, selected_character):
        """언어 변경 처리 함수"""
        lang_map = {
            "한국어": "ko",
            "日本語": "ja",
            "中文(简体)": "zh_CN",
            "中文(繁體)": "zh_TW",
            "English": "en"
        }
        lang_code = lang_map.get(selected_lang, "ko")
        if translation_manager.set_language(lang_code):
            if selected_lang in characters[selected_character]["languages"]:
                speech_manager_state.current_language = selected_lang
            else:
                speech_manager_state.current_language = characters[selected_character]["languages"][0]
            system_presets = load_system_presets(lang_code)
                
            if len(system_presets) > 0:
                preset_name = list(system_presets.keys())[0]
                system_content = system_presets[preset_name]
            else:
                system_content = _("system_message_default")

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
                gr.update(value=_("reset_session_button")),
                gr.update(value=_("reset_all_sessions_button")),
            ]
        else:
            # 언어 변경 실패 시 아무 것도 하지 않음
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # 언어 변경 이벤트 연결
    language_dropdown.change(
        fn=change_language,
        inputs=[language_dropdown, character_dropdown],
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
            reset_btn,
            reset_all_btn,
        ]
    )
        # 메시지 전송 시 함수 연결
    msg.submit(
        fn=main_tab.process_message,
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
            seed_state,
            selected_language_state,
            character_dropdown
        ],
        outputs=[
            msg,            # 사용자 입력 필드 초기화
            history_state,  # 히스토리 업데이트
            chatbot,        # Chatbot UI 업데이트
            status_text     # 상태 메시지 업데이트
        ],
        queue=False
    ).then(
        fn=main_tab.filter_messages_for_chatbot,
        inputs=[history_state],
        outputs=chatbot,
        queue=False
    )

    send_btn.click(
        fn=main_tab.process_message,
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
            seed_state,
            selected_language_state,
            character_dropdown
        ],
        outputs=[
            msg, 
            history_state, 
            chatbot, 
            status_text
        ],
        queue=False
    ).then(
        fn=main_tab.filter_messages_for_chatbot,            # 추가된 부분
        inputs=[history_state],
        outputs=chatbot,                           # chatbot에 최종 전달
        queue=False
    )
        
    start_conversation_button.click(
        fn=main_tab.process_character_conversation,
        inputs=[
            history_state,
            character_conversation_dropdown,
            model_type_dropdown, 
            model_dropdown,
            custom_model_path_state,
            image_input,
            api_key_text,
            selected_device_state,
            seed_state
        ],
        outputs=[history_state, profile_image]
    ).then(
        fn=main_tab.filter_messages_for_chatbot,  # 히스토리를 채팅창에 표시하기 위한 필터링
        inputs=[history_state],
        outputs=[chatbot]
    )
        
    # 초기화 버튼 클릭 시 확인 메시지 표시
    reset_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
        inputs=[],
        outputs=[reset_confirm_row, reset_yes_btn, reset_no_btn]
    )
    reset_all_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
        inputs=[],
        outputs=[reset_all_confirm_row, reset_all_yes_btn, reset_all_no_btn]
    )

    # "예" 버튼 클릭 시 세션 초기화 수행
    reset_yes_btn.click(
        fn=main_tab.reset_session,
        inputs=[history_state, chatbot, system_message_box, selected_language_state, session_id_state],
        outputs=[msg, history_state, chatbot, status_text],
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
        fn=main_tab.reset_all_sessions,
        inputs=[history_state, chatbot, system_message_box, selected_language_state],
        outputs=[msg, history_state, chatbot, status_text],
        queue=False
    ).then(
        fn=lambda: gr.update(visible=False),  # 확인 메시지 숨김
        inputs=[],
        outputs=[reset_all_confirm_row],
        queue=False
    ).then(
        fn=main_tab.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown]
    )

    # "모든 세션 초기화"의 "아니요" 버튼 클릭 시 확인 메시지 숨김
    reset_all_no_btn.click(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[reset_all_confirm_row],
        queue=False
    )
        
    demo.load(
        fn=main_tab.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown],
        queue=False
    )
        
    session_select_dropdown.change(
        fn=apply_session_immediately,
        inputs=[session_select_dropdown],
        outputs=[history_state, session_id_state, session_select_info]
    ).then(
        fn=main_tab.filter_messages_for_chatbot,
        inputs=[history_state],
        outputs=[chatbot]
    )
        
            
    create_download_tab()
        
    with gr.Tab(_("cache_tab_title")):
        with gr.Row():
            with gr.Column():
                refresh_button = gr.Button(_("refresh_model_list_button"))
                refresh_info = gr.Textbox(label=_("refresh_info_label"), interactive=False)
            with gr.Column():
                clear_all_btn = gr.Button(_("cache_clear_all_button"))
                clear_all_result = gr.Textbox(label=_("clear_all_result_label"), interactive=False)

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
            new_choices = api_models + local_models
            new_choices = list(dict.fromkeys(new_choices))
            new_choices = sorted(new_choices)  # 정렬 추가
            # 반환값:
            logger.info(_("refresh_model_list_button"))
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
                gr.update(value=_("refresh_model_list_button")),
                gr.update(label=_("refresh_info_label")),
                gr.update(value=_("cache_clear_all_button")),
                gr.update(label=_("clear_all_result_label"))
            ]

        language_dropdown.change(
            fn=change_language,
            inputs=[language_dropdown],
            outputs=[
                refresh_button,
                refresh_info,
                clear_all_btn,
                clear_all_result
            ]
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

        with gr.Tabs():
            # 사용자 지정 모델 경로 설정 섹션
            with gr.Tab("사용자 지정 모델 경로 설정"):
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
            with gr.Tab("시스템 메시지 프리셋 관리"):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="프리셋 선택",  # 필요 시 번역 키로 변경
                        choices=get_preset_choices(default_language),
                        value=get_preset_choices(default_language)[0] if get_preset_choices(default_language) else None
                    )
                    refresh_preset_button = gr.Button("프리셋 목록 갱신")
                    refresh_preset_button.click(
                        fn=main_tab.refresh_preset_list,
                        inputs=[selected_language_state],
                        outputs=[preset_dropdown]
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
                    fn=main_tab.initial_load_presets,
                    inputs=[],
                    outputs=[preset_dropdown],
                    queue=False
                )
                
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
                        return "❌ 삭제할 프리셋을 선택해주세요.", gr.update(visible=False), gr.update(choices=get_preset_choices(default_language))
                    confirmation_msg = f"⚠️ 정말로 '{name}' 프리셋을 삭제하시겠습니까?"
                    return confirmation_msg, gr.update(visible=True), gr.update(choices=get_preset_choices(default_language))
                
                
                # 삭제 확인 버튼 추가
                delete_confirm_btn = gr.Button("삭제 확인", variant="danger", visible=False)
                delete_cancel_btn = gr.Button("삭제 취소", variant="secondary", visible=False)
                delete_preset_info = gr.Textbox(label="프리셋 삭제 결과", interactive=False)
                
                with gr.Row(visible=False) as delete_preset_confirm_row:
                    delete_preset_confirm_msg = gr.Markdown("⚠️ **정말로 선택한 프리셋을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.**")
                    delete_preset_yes_btn = gr.Button("✅ 예", variant="danger")
                    delete_preset_no_btn = gr.Button("❌ 아니요", variant="secondary")
                
                # 프리셋 삭제 확인 버튼 클릭 시 실제 삭제 수행
                def confirm_delete_preset(name, confirm):
                    if confirm:
                        success, message = handle_delete_preset(name, default_language)
                        if success:
                            return message, gr.update(visible=False), gr.update(choices=get_preset_choices(default_language))
                        else:
                            return f"❌ {message}", gr.update(visible=False), gr.update(choices=get_preset_choices(default_language))
                    else:
                        return "❌ 삭제가 취소되었습니다.", gr.update(visible=False), gr.update(choices=get_preset_choices(default_language))
                    
                # 프리셋 삭제 버튼과 확인 버튼의 상호작용 연결
                delete_preset_btn.click(
                    fn=lambda : (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
                    inputs=[preset_dropdown],
                    outputs=[delete_preset_confirm_row, delete_preset_yes_btn, delete_preset_no_btn]
                )
                
                delete_preset_yes_btn.click(
                    fn=confirm_delete_preset,
                    inputs=[preset_dropdown, gr.State(True)],  # confirm=True
                    outputs=[preset_info, delete_preset_confirm_row, preset_dropdown]
                )

                # 프리셋 삭제 취소 버튼 클릭 시
                delete_preset_no_btn.click(
                    fn=lambda: ("❌ 삭제가 취소되었습니다.", gr.update(visible=False), preset_dropdown),
                    inputs=[],
                    outputs=[preset_info, delete_preset_confirm_row, preset_dropdown]
                )
            
                apply_preset_btn.click(
                    fn=apply_preset,
                    inputs=[preset_dropdown, session_id_state, history_state, selected_language_state],
                    outputs=[preset_info, history_state, system_message_box, profile_image]
                ).then(
                    fn=main_tab.filter_messages_for_chatbot,
                    inputs=[history_state],
                    outputs=chatbot
                )
            # 채팅 기록 저장 섹션
            with gr.Tab("채팅 기록 저장"):
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
            with gr.Tab("채팅 히스토리 재로드"):
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
            with gr.Tab("세션 관리"):
                gr.Markdown("### 세션 관리")
                with gr.Row():
                    refresh_sessions_btn = gr.Button("세션 목록 갱신")
                    existing_sessions_dropdown = gr.Dropdown(
                        label="기존 세션 목록",
                        choices=[],  # 초기에는 비어 있다가, 버튼 클릭 시 갱신
                        value=None,
                        interactive=True
                    )
                    current_session_display = gr.Textbox(
                        label="현재 세션 ID",
                        value="",
                        interactive=False
                    )
                
                with gr.Row():
                    create_new_session_btn = gr.Button("새 세션 생성")
                    apply_session_btn = gr.Button("세션 적용")
                    delete_session_btn = gr.Button("세션 삭제")
                
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
                
                refresh_sessions_btn.click(
                    fn=main_tab.refresh_sessions,
                    inputs=[],
                    outputs=[existing_sessions_dropdown]
                ).then(
                    fn=main_tab.refresh_sessions,
                    inputs=[],
                    outputs=[session_select_dropdown]
                )
                
                # (2) 새 세션 생성
                create_new_session_btn.click(
                    fn=lambda: main_tab.create_new_session(system_message_box.value),
                    inputs=[],
                    outputs=[session_id_state, session_manage_info]
                ).then(
                    fn=lambda: [],
                    inputs=[],
                    outputs=[history_state]
                ).then(
                    fn=main_tab.filter_messages_for_chatbot,
                    inputs=[history_state],
                    outputs=[chatbot]
                ).then(
                    fn=main_tab.refresh_sessions,
                    inputs=[],
                    outputs=[session_select_dropdown]
                )
                
                apply_session_btn.click(
                    fn=main_tab.apply_session,
                    inputs=[existing_sessions_dropdown],
                    outputs=[history_state, session_id_state, session_manage_info]
                ).then(
                    fn=main_tab.filter_messages_for_chatbot,
                    inputs=[history_state],
                    outputs=[chatbot]
                ).then(
                    fn=main_tab.refresh_sessions,
                    inputs=[],
                    outputs=[session_select_dropdown]
                )
                
                with gr.Row(visible=False) as delete_session_confirm_row:
                        delete_session_confirm_msg = gr.Markdown("⚠️ **정말로 선택한 세션을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.**")
                        delete_session_yes_btn = gr.Button("✅ 예", variant="danger")
                        delete_session_no_btn = gr.Button("❌ 아니요", variant="secondary")

                # “세션 삭제” 버튼 클릭 시, 확인창(문구/버튼) 보이기
                delete_session_btn.click(
                    fn=lambda: (
                        gr.update(visible=True),
                        gr.update(visible=True), 
                        gr.update(visible=True)
                    ),
                    inputs=[],
                    outputs=[delete_session_confirm_row, delete_session_yes_btn, delete_session_no_btn]
                )

                # (5) 예 버튼 → 실제 세션 삭제
                delete_session_yes_btn.click(
                    fn=main_tab.delete_session,
                    inputs=[existing_sessions_dropdown, session_id_state],
                    outputs=[session_manage_info, delete_session_confirm_msg, existing_sessions_dropdown]
                ).then(
                    fn=lambda: (gr.update(visible=False)),
                    inputs=[],
                    outputs=[delete_session_confirm_row],
                    queue=False
                ).then(
                    fn=main_tab.refresh_sessions,
                    inputs=[],
                    outputs=[session_select_dropdown]
                )

                # “아니요” 버튼: “취소되었습니다” 메시지 + 문구/버튼 숨기기
                delete_session_no_btn.click(
                    fn=lambda: (
                        "❌ 삭제가 취소되었습니다.",
                        gr.update(visible=False)
                    ),
                    inputs=[],
                    outputs=[session_manage_info, delete_session_confirm_row],
                    queue=False
                )
            with gr.Tab("장치 설정"):
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
        
    demo.load(
        fn=on_app_start,
        inputs=[], # 언어 상태는 이미 초기화됨
        outputs=[session_id_state, history_state, existing_sessions_dropdown,
        current_session_display],
        queue=False
    )

if __name__=="__main__":
    
    initialize_app()
    translation_manager.current_language=args.language
    
    demo.queue().launch(debug=args.debug, share=args.share, inbrowser=args.inbrowser, server_port=args.port, width=800)