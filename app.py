# app.py
import platform
import torch
import os
import traceback
import gradio as gr
import logging
from logging.handlers import RotatingFileHandler
import uuid  # 고유한 세션 ID 생성을 위해 추가
import base64
from huggingface_hub import HfApi
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
    handle_delete_preset
)
from models import (
    default_device, 
    get_all_local_models, 
    get_default_device, 
    generate_answer, 
    FIXED_MODELS, 
    get_fixed_model_id
)
from cache import models_cache
import sqlite3

from translations import translation_manager, _, detect_system_language, get_system_message
from minami_asuka_char_set import DEFAULT_CHARACTER_SETTINGS
import i18n
import locale

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

# 이미지 파일을 Base64로 인코딩 (별도로 처리)
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
            
DEFAULT_SYSTEM_MESSAGES = DEFAULT_CHARACTER_SETTINGS
local_models_data = get_all_local_models()
transformers_local = local_models_data["transformers"]
gguf_local = local_models_data["gguf"]
mlx_local = local_models_data["mlx"]

# 고정된 모델 목록에서 mlx 모델 가져오기
generator_choices = [FIXED_MODELS.get("mlx", "default-mlx-model")]

default_language = detect_system_language()

##########################################
# Gradio UI
##########################################

def on_app_start():
    """
    Gradio 앱이 로드되면서 실행될 콜백.
    - 고유한 세션 ID를 생성하고,
    - 해당 세션의 히스토리를 DB에서 불러온 뒤 반환.
    - 기본 시스템 메시지 불러오기
    """
    sid = str(uuid.uuid4())  # 고유한 세션 ID 생성
    logger.info(f"앱 시작 시 세션 ID: {sid}")  # 디버깅 로그 추가
    loaded_history = load_chat_from_db(sid)
    logger.info(f"앱 시작 시 불러온 히스토리: {loaded_history}")  # 디버깅 로그 추가

    # 기본 시스템 메시지 설정 (프리셋이 없는 경우)
    if not loaded_history:
        default_system = {
            "role": "system",
            "content": get_system_message()
        }
        loaded_history = [default_system]
    return sid, loaded_history

def filter_messages_for_chatbot(history):
    """
    채팅 히스토리를 Gradio Chatbot 컴포넌트에 맞는 형식으로 변환

    Args:
        history (list): 전체 채팅 히스토리

    Returns:
        list: [(user_msg, bot_msg), ...] 형식의 메시지 리스트
    """
    if history is None:
        return []
        
    messages = []
    current_user_msg = None
    
    for msg in history:
        if msg["role"] == "user":
            current_user_msg = msg["content"]
        elif msg["role"] == "assistant" and current_user_msg is not None:
            messages.append((current_user_msg, msg["content"]))
            current_user_msg = None
        # system 메시지는 무시
    
    # 마지막 user 메시지가 아직 응답을 받지 않은 경우
    if current_user_msg is not None:
        messages.append((current_user_msg, None))
    
    return messages

def process_message(message, session_id, history, system_msg, device, seed_val, model_type_val):
    """
    사용자 메시지 처리 및 봇 응답 생성을 통합한 함수
    """
    if not message.strip():
        return "", history, filter_messages_for_chatbot(history), ""
        
    if not history:
        history = [{"role": "system", "content": system_msg}]
        
    # 사용자 메시지 추가
    history.append({"role": "user", "content": message})
    chatbot_messages = filter_messages_for_chatbot(history)  # 중간 상태 업데이트
    
    try:
        answer = generate_answer(
            history=history,
            model_type=model_type_val,
            device=device,
            seed=seed_val,
            language=default_language  # 다국어 지원을 위해 현재 언어 사용
        )
        
        # 이미지를 응답에 포함시키지 않음
        answer_with_image = answer
            
        history.append({"role": "assistant", "content": answer_with_image})
        
        # DB에 저장
        save_chat_history_db(history, session_id=session_id)
        
        return "", history, filter_messages_for_chatbot(history), ""
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return "", history, chatbot_messages, f"❌ 오류 발생: {str(e)}"

history_state = gr.State([])
overwrite_state = gr.State(False) 

# 단일 history_state와 selected_device_state 정의 (중복 제거)
session_id_state = gr.State()
history_state = gr.State([])
selected_device_state = gr.State(default_device)
seed_state = gr.State(42)  # 시드 상태 전역 정의
selected_language_state = gr.State(default_language)

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
    title = gr.Markdown(value=f"## {_('title')}")
    
    # 언어 선택 드롭다운 추가
    language_dropdown = gr.Dropdown(
        label=_('language_select'),
        choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
        value=translation_manager.get_language_display_name(default_language),
        interactive=True,
        info=_('language_info')
    )
    
    # 시스템 메시지 박스
    system_message_display = gr.Textbox(
        label=_('system_message_label'),
        value=DEFAULT_SYSTEM_MESSAGES[default_language],
        interactive=False
    )
    
    with gr.Tab(_('main_tab')):
        with gr.Row():
            model_type = gr.Dropdown(
                label=_('select_model'),
                choices=["transformers", "gguf", "mlx"],
                value="gguf",
                interactive=True
            )
        
        fixed_model_display = gr.Textbox(
            label=_('selected_model'),
            value=get_fixed_model_id("gguf"),
            interactive=False
        )
        
        with gr.Row():
            chatbot = gr.Chatbot(
                height=400,
                label=_('chatbot_label'),
                elem_id="chatbot"
            )
            # 프로필 이미지를 표시할 Image 컴포넌트 추가
            profile_image = gr.Image(
                value=character_image_path,
                label=_('profile_image_label'),
                visible=True,
                interactive=False,
                width="500px",
                height="500px"
            )
        
        with gr.Row():
            msg = gr.Textbox(
                label=_('input_placeholder'),
                placeholder=_('input_placeholder'),
                scale=9
            )
            send = gr.Button(
                value=_('send_button'),  # 'label' 대신 'value' 사용
                scale=1,
                variant="primary"
            )
        
        status = gr.Markdown("", elem_id="status_text")
        
        with gr.Row():
            seed_input = gr.Number(
                label=_('seed_value'),
                value=42,
                precision=0,
                step=1,
                interactive=True,
                info=_('seed_info')
            )
        
        # 시드 입력과 상태 연결
        seed_input.change(
            fn=lambda seed: seed if seed is not None else 42,
            inputs=[seed_input],
            outputs=[seed_state]
        )
        
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
            
            # 시스템 메시지 업데이트
            system_message = get_system_message()
            
            # UI 컴포넌트 업데이트
            return [
                gr.update(value=f"## {_('title')}"),
                gr.update(label=_('language_select'),info=_('language_info')),
                gr.update(label=_('system_message_label')),
                gr.update(label=_('select_model')),
                gr.update(label=_('selected_model')),
                gr.update(label=_('chatbot_label')),
                gr.update(label=_('profile_image_label')),
                gr.update(label=_('input_placeholder'), placeholder=_('input_placeholder')),
                gr.update(value=_('send_button')),
                gr.update(label=_('seed_value'),info=_('seed_info')),
                gr.update(value=system_message),
            ]

        # 언어 변경 이벤트 연결
        language_dropdown.change(
            fn=change_language,
            inputs=[language_dropdown],
            outputs=[
                title,
                language_dropdown,
                system_message_display,
                model_type,
                fixed_model_display,
                chatbot,
                profile_image,
                msg,
                send,
                seed_input,
                system_message_display
            ]
        )

         # 이벤트 핸들러 연결
        msg.submit(
            fn=process_message,
            inputs=[msg, session_id_state, history_state, system_message_display, selected_device_state, seed_state, model_type],
            outputs=[msg, history_state, chatbot, status]
        )
        
        send.click(
            fn=process_message,
            inputs=[msg, session_id_state, history_state, system_message_display, selected_device_state, seed_state, model_type],
            outputs=[msg, history_state, chatbot, status]
        )
    
        # 세션 초기화
        demo.load(
            fn=on_app_start,
            inputs=[],
            outputs=[session_id_state, history_state],
            queue=False
        )
    
    # "설정" 탭 유지
    with gr.Tab(_('settings_tab')):
        setting_title=gr.Markdown(f"### {_('settings_title')}")

        # 시스템 메시지 프리셋 관리 비활성화
        with gr.Accordion(_('preset_management'), open=False):
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label=_('preset_select'),
                    choices=[],  # 초기 로드에서 채워짐
                    value=None,
                    interactive=False  # Prevent user from applying presets
                )
                apply_preset_btn = gr.Button(_('preset_apply'), interactive=False)  # Disable applying presets

        # 세션 관리 섹션
        with gr.Accordion(_('session_management'), open=False):
            gr.Markdown(f"### {_('session_management')}")
            with gr.Row():
                refresh_sessions_btn = gr.Button(_('session_list_refresh'))
                existing_sessions_dropdown = gr.Dropdown(
                    label=_('existing_sessions'),
                    choices=[],  # 초기에는 비어 있다가, 버튼 클릭 시 갱신
                    value=None,
                    interactive=True
                )
            
            with gr.Row():
                create_new_session_btn = gr.Button(_('create_new_session'))
                apply_session_btn = gr.Button(_('apply_session'))
                delete_session_btn = gr.Button(_('delete_session'))
            
            # 삭제 확인을 위한 컴포넌트 추가
            confirm_delete_checkbox = gr.Checkbox(
                label=_('delete_confirm'),
                value=False,
                interactive=True,
                visible=False  # 기본적으로 숨김
            )
            confirm_delete_btn = gr.Button(
                _('confirm_delete'),
                variant="stop",
                visible=False  # 기본적으로 숨김
            )
            
            session_manage_info = gr.Textbox(
                label=_('session_manage_result'),
                interactive=False
            )
            
            current_session_display = gr.Textbox(
                label=_('current_session'),
                value="",
                interactive=False
            )

            # 현재 세션 ID 표시 업데이트
            session_id_state.change(
                fn=lambda sid: _('current_session_display').format(sid=sid) if sid else _('no_session'),
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
                new_sid = str(uuid.uuid4())  # 새 세션 ID 생성
                logger.info(f"새 세션 생성됨: {new_sid}")
                
                # 기본 시스템 메시지 설정
                system_message = {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_MESSAGES["ko"]  # 기본 언어를 한국어로 설정
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
                    with sqlite3.connect("chat_history.db") as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (chosen_sid,))
                        count = cursor.fetchone()[0]
                        
                        if count == 0:
                            logger.warning(f"세션 '{chosen_sid}'이(가) DB에 존재하지 않습니다.")
                            return f"❌ 세션 '{chosen_sid}'이(가) DB에 존재하지 않습니다.", False, gr.update(visible=False)
                        
                        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (chosen_sid,))
                        conn.commit()
                        
                    logger.info(f"세션 삭제 완료: {chosen_sid}")
                    return f"✅ 세션 '{chosen_sid}'이(가) 삭제되었습니다.", False, gr.update(visible=False)
                    
                except sqlite3.OperationalError as oe:
                    logger.critical(f"DB 운영 오류: {oe}")
                    return f"❌ DB 운영 오류 발생: {oe}", False, gr.update(visible=False)
                except Exception as e:
                    logger.error(f"세션 삭제 오류: {e}", exc_info=True)
                    return f"❌ 세션 삭제 실패: {e}", False, gr.update(visible=False)
    
            
            def initiate_delete():
                return gr.update(visible=True), gr.update(visible=True)
            
            def confirm_delete(chosen_sid, current_sid, confirm):
                if not confirm:
                    return "❌ 삭제가 취소되었습니다.", False, gr.update(visible=False)
                return delete_session(chosen_sid, current_sid)
        
            # 버튼 이벤트 연결
            refresh_sessions_btn.click(
                fn=refresh_sessions,
                inputs=[],
                outputs=[existing_sessions_dropdown, session_manage_info]
            )
            
            def on_new_session_created(sid, info):
                """새 세션 생성 시 초기 히스토리 생성"""
                history = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGES["ko"]}]
                return history, filter_messages_for_chatbot(history)
    
            # 기존의 이벤트 핸들러 수정
            create_new_session_btn.click(
                fn=create_new_session,
                inputs=[],
                outputs=[session_id_state, session_manage_info]
            ).then(
                fn=on_new_session_created,
                inputs=[session_id_state, session_manage_info],
                outputs=[history_state, chatbot]
            )
    
            def on_session_applied(loaded_history, sid, info):
                """세션 적용 시 채팅 표시 업데이트"""
                return loaded_history, filter_messages_for_chatbot(loaded_history), info
    
            apply_session_btn.click(
                fn=apply_session,
                inputs=[existing_sessions_dropdown],
                outputs=[history_state, session_id_state, session_manage_info]
            ).then(
                fn=lambda h, s, i: (h, filter_messages_for_chatbot(h), i),
                inputs=[history_state, session_id_state, session_manage_info],
                outputs=[history_state, chatbot, session_manage_info]
            )
            
            delete_session_btn.click(
                fn=initiate_delete,
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
                
                # 시스템 메시지 업데이트
                system_message = get_system_message()
                
                # UI 컴포넌트 업데이트
                return [
                    # 설정 탭 UI 요소들 업데이트
                    gr.update(value=f"### {_('settings_title')}"),
                    gr.update(label=_('preset_select')),
                    gr.update(value=_('preset_apply')),
                    gr.update(value=_('session_list_refresh')),
                    gr.update(label=_('existing_sessions')),
                    gr.update(value=_('create_new_session')),
                    gr.update(value=_('apply_session')),
                    gr.update(value=_('delete_session')),
                    gr.update(label=_('delete_confirm')),
                    gr.update(value=_('confirm_delete')),
                    gr.update(label=_('session_manage_result')),
                    gr.update(label=_('current_session'))
                ]

            # language_dropdown.change 이벤트 업데이트
            language_dropdown.change(
                fn=change_language,
                inputs=[language_dropdown],
                outputs=[
                    # 설정 탭 UI 요소들 추가
                    setting_title,
                    preset_dropdown,
                    apply_preset_btn,
                    refresh_sessions_btn,
                    existing_sessions_dropdown,
                    create_new_session_btn,
                    apply_session_btn,
                    delete_session_btn,
                    confirm_delete_checkbox,
                    confirm_delete_btn,
                    session_manage_info,
                    current_session_display
                ]
            )
    
    # 장치 설정 섹션 유지
    with gr.Tab(_('device_settings')):
        device_dropdown = gr.Dropdown(
            label=_('device_select'),
            choices=['Auto(recommanded)', "CPU", "GPU"],
            value='Auto(recommanded)'
        )
        device_info = gr.Textbox(
            label='Device Info',
            value=f"Current Device: {default_device.upper()}",
            interactive=False
        )
        
        def set_device(selection):
            try:
                if selection == 'Auto(recommanded)':
                    device = get_default_device()
                elif selection == "CPU":
                    device = "cpu"
                elif selection == "GPU":
                    if torch and torch.cuda.is_available():
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
            except Exception as e:
                logger.error(f"Device selection error: {e}")
                return gr.update(value="❌ 장치 설정 중 오류가 발생했습니다."), "cpu"

        device_dropdown.change(
            fn=set_device,
            inputs=[device_dropdown],
            outputs=[device_info, selected_device_state],
            queue=False
        )
        
        def change_device_language(selected_lang):
            """장치 설정 탭의 언어 변경을 위한 함수"""
            lang_map = {
                "한국어": "ko",
                "日本語": "ja",
                "中文(简体)": "zh_CN",
                "中文(繁體)": "zh_TW",
                "English": "en"
            }
            lang_code = lang_map.get(selected_lang, "ko")
            translation_manager.set_language(lang_code)

            # 장치 선택기의 label만 업데이트
            return gr.update(
                label=_('device_select')
            )

        # language_dropdown.change 이벤트 연결 부분을 수정
        language_dropdown.change(
            fn=change_device_language,
            inputs=[language_dropdown],
            outputs=[device_dropdown]
        )

demo.launch(debug=True, inbrowser=True, server_port=7861, width=800)