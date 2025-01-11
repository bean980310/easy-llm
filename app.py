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

# HuggingFace에서 지원하는 기본 모델 목록 (필요 시 유지 또는 수정)
known_hf_models = [
    # ... (필요에 따라 유지 또는 제거 가능)
]

DEFAULT_SYSTEM_MESSAGE="""
    미나미 아스카(南飛鳥, みなみあすか, Minami Asuka)
    성별: 여성
    나이: 20
    거주지: 유저의 모니터 속
    구사가능 언어: 한국어, 영어, 일본어, 중국어
    성격
    - 보이시면서도 털털한 성격.
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

local_models_data = get_all_local_models()
transformers_local = local_models_data["transformers"]
gguf_local = local_models_data["gguf"]
mlx_local = local_models_data["mlx"]

# 고정된 모델 목록에서 mlx 모델 가져오기
generator_choices = [FIXED_MODELS.get("mlx", "default-mlx-model")]

##########################################
# Gradio UI
##########################################
def user_message(user_input, session_id, history, system_msg):
    """사용자 메시지 처리 함수"""
    if not user_input.strip():
        return "", history, ""
    
    # history가 None인 경우 빈 리스트로 초기화
    if history is None:
        history = []
        
    if not history:
        system_message = {
            "role": "system",
            "content": system_msg
        }
        history = [system_message]
    
    history.append({"role": "user", "content": user_input})
    return "", history, "🤔 답변을 생성하는 중입니다..."

def bot_message(session_id, history, device, seed, model_type):  # async 제거
    """봇 메시지 생성 함수"""
    if model_type is None:
        logger.error("모델 유형이 선택되지 않았습니다.")
        return history, "❌ 모델 유형이 선택되지 않았습니다."
    
    # history가 None인 경우 처리
    if history is None:
        history = []
    
    selected_model = get_fixed_model_id(model_type)
    logger.debug(f"Selected model_type: {model_type}, model_id: {selected_model}")
    
    if not selected_model:
        logger.error(f"모델 유형 '{model_type}'에 대한 고정된 모델 ID를 찾을 수 없습니다.")
        return history, "❌ 지원되지 않는 모델 유형입니다."
    
    try:
        # async/await 제거하고 동기 호출로 변경
        answer = generate_answer(history, model_type, None, None, None, device, seed)
        
        # 이미지를 응답에 포함시키지 않음
        answer_with_image = answer
            
    except MemoryError:
        logger.critical("메모리 부족 오류 발생")
        return history, "❌ 메모리 부족 오류가 발생했습니다. 시스템 관리자에게 문의하세요."
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        return history, f"❌ 오류 발생: {str(e)}"
    
    if not history:
        history = []
    
    history.append({"role": "assistant", "content": answer_with_image})
    
    if not session_id:
        logger.error("세션 ID가 None입니다.")
        return history, "❌ 세션 ID가 유효하지 않습니다."
    
    # async/await 제거
    save_chat_history_db(history, session_id=session_id)
    logger.debug(f"DB에 채팅 히스토리 저장 완료 (session_id={session_id})")
    return history, ""

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
            "content": DEFAULT_SYSTEM_MESSAGE
        }
        loaded_history = [default_system]
    return sid, loaded_history

# 단일 history_state와 selected_device_state 정의 (중복 제거)
history_state = gr.State([])
selected_device_state = gr.State(default_device)
seed_state = gr.State(42)  # 시드 상태 전역 정의

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
    
    # 모든 State를 먼저 정의
    session_id = gr.State()
    history = gr.State([])
    device = gr.State(default_device)
    seed = gr.State(42)
    
    # 시스템 메시지 박스
    system_message_display = gr.Textbox(
        label="시스템 메시지",
        value=DEFAULT_SYSTEM_MESSAGE,
        interactive=False
    )
    
    with gr.Tab("메인"):
        with gr.Row():
            model_type = gr.Dropdown(
                label="모델 유형 선택",
                choices=["transformers", "gguf", "mlx"],
                value="gguf",
                interactive=True
            )
            
        fixed_model_display = gr.Textbox(
            label="선택된 모델 유형",
            value=get_fixed_model_id("gguf"),
            interactive=False
        )
        
        with gr.Row():
            chatbot = gr.Chatbot(
                height=400,
                label="Chatbot",
                elem_id="chatbot"
            )
            # 프로필 이미지를 표시할 Image 컴포넌트 추가
            profile_image = gr.Image(
                value=character_image_path,
                label="프로필 이미지",
                visible=True,
                interactive=False
            )
        
        with gr.Row():
            msg = gr.Textbox(
                label="메시지 입력",
                placeholder="메시지를 입력하세요...",
                scale=9
            )
            send = gr.Button("전송", scale=1, variant="primary")
            
        status = gr.Markdown("", elem_id="status_text")
        
        with gr.Row():
            seed_input = gr.Number(
                label="시드 값",
                value=42,
                precision=0,
                step=1,
                interactive=True
            )

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
                    seed=seed_val
                )
                
                # 이미지를 응답에 포함시키지 않음
                answer_with_image = answer
                    
                history.append({"role": "assistant", "content": answer_with_image})
                
                # DB에 저장
                save_chat_history_db(history, session_id=session_id)
                
                return "", history, filter_messages_for_chatbot(history), ""
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                return "", history, chatbot_messages, f"오류 발생: {str(e)}"

        # 이벤트 핸들러 연결
        msg.submit(
            process_message,
            inputs=[msg, session_id, history, system_message_display, device, seed, model_type],
            outputs=[msg, history, chatbot, status]
        )
        
        send.click(
            process_message,
            inputs=[msg, session_id, history, system_message_display, device, seed, model_type],
            outputs=[msg, history, chatbot, status]
        )
        
        # 시드 업데이트 핸들러
        seed_input.change(
            lambda x: x,
            inputs=[seed_input],
            outputs=[seed]
        )
        
        # 모델 타입 변경 핸들러
        model_type.change(
            lambda x: get_fixed_model_id(x),
            inputs=[model_type],
            outputs=[fixed_model_display]
        )

        # 세션 초기화
        demo.load(
            on_app_start,
            outputs=[session_id, history]
        )
    
    # "설정" 탭 유지
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

            # 현재 세션 ID 표시 업데이트
            session_id.change(
                fn=lambda sid: f"현재 세션: {sid}" if sid else "세션 없음",
                inputs=[session_id],
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
                    "content": DEFAULT_SYSTEM_MESSAGE
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
                history = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}]
                return history, filter_messages_for_chatbot(history)
    
            # 기존의 이벤트 핸들러 수정
            create_new_session_btn.click(
                fn=create_new_session,
                inputs=[],
                outputs=[session_id, session_manage_info]
            ).then(
                fn=on_new_session_created,
                inputs=[session_id, session_manage_info],
                outputs=[history, chatbot]
            )
    
            def on_session_applied(loaded_history, sid, info):
                """세션 적용 시 채팅 표시 업데이트"""
                return loaded_history, filter_messages_for_chatbot(loaded_history), info
    
            apply_session_btn.click(
                fn=apply_session,
                inputs=[existing_sessions_dropdown],
                outputs=[history, session_id, session_manage_info]
            ).then(
                fn=lambda h, s, i: (h, filter_messages_for_chatbot(h), i),
                inputs=[history, session_id, session_manage_info],
                outputs=[history, chatbot, session_manage_info]
            )
            
            delete_session_btn.click(
                fn=initiate_delete,
                inputs=[],
                outputs=[confirm_delete_checkbox, confirm_delete_btn]
            )
            
            # 삭제 확인 버튼 클릭 시 실제 삭제 수행
            confirm_delete_btn.click(
                fn=confirm_delete,
                inputs=[existing_sessions_dropdown, session_id, confirm_delete_checkbox],
                outputs=[session_manage_info, confirm_delete_checkbox, confirm_delete_btn]
            ).then(
                fn=refresh_sessions,  # 세션 삭제 후 목록 새로고침
                inputs=[],
                outputs=[existing_sessions_dropdown, session_manage_info]
            )
    
    # 장치 설정 섹션 유지
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
            outputs=[device_info, selected_device_state],
            queue=False
        )

demo.launch(debug=True, inbrowser=True, server_port=7861, width=500)