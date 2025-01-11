# app.py

import os
import shutil
import traceback
import torch
import gc
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import logging
from logging.handlers import RotatingFileHandler
from model_handlers import (
    GGUFModelHandler,MiniCPMLlama3V25Handler, GLM4Handler, GLM4VHandler, VisionModelHandler,
    Aya23Handler, GLM4HfHandler, OtherModelHandler, QwenHandler, MlxModelHandler, MlxVisionHandler
)
import json
import datetime
import csv
import secrets
from huggingface_hub import HfApi, list_models
from utils import (
    make_local_dir_name,
    get_all_local_models,  # 수정된 함수
    scan_local_models,
    get_model_list_from_hf_hub,
    download_model_from_hf,
    ensure_model_available,
    convert_and_save,
    
)
from cache import models_cache
import sqlite3

def get_existing_sessions():
    """
    DB에서 이미 존재하는 모든 session_id 목록을 가져옴 (중복 없이).
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM chat_history ORDER BY session_id ASC")
        rows = cursor.fetchall()
        conn.close()
        session_ids = [r[0] for r in rows]
        return session_ids
    except Exception as e:
        logger.error(f"세션 목록 조회 오류: {e}")
        return []
def save_chat_history_db(history, session_id="session_1"):
    """
    채팅 히스토리를 SQLite DB에 저장합니다.
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        for msg in history:
            cursor.execute("""
                INSERT INTO chat_history (session_id, role, content)
                VALUES (?, ?, ?)
            """, (session_id, msg.get("role"), msg.get("content")))
        
        conn.commit()
        conn.close()
        logger.info(f"DB에 채팅 히스토리 저장 완료 (session_id={session_id})")
        return True
    except Exception as e:
        logger.error(f"DB 저장 중 오류: {e}")
        return False
    
def save_chat_history(history):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"chat_history_{timestamp}.json"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"채팅 히스토리 저장 완료: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"채팅 히스토리 저장 중 오류: {e}")
        return None

def save_chat_history_csv(history):
    """
    채팅 히스토리를 CSV 형태로 저장
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"chat_history_{timestamp}.csv"
    try:
        # CSV 파일 열기
        with open(file_name, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 작성
            writer.writerow(["role", "content"])
            # 각 메시지 row 작성
            for msg in history:
                writer.writerow([msg.get("role"), msg.get("content")])
        logger.info(f"채팅 히스토리 CSV 저장 완료: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"채팅 히스토리 CSV 저장 중 오류: {e}")
        return None
    
def save_chat_button_click(history):
    if not history:
        return "채팅 이력이 없습니다."
    saved_path = save_chat_history(history)
    if saved_path is None:
        return "❌ 채팅 기록 저장 실패"
    else:
        return f"✅ 채팅 기록이 저장되었습니다: {saved_path}"
    
# 예: session_id를 함수 인자로 전달받아 DB로부터 해당 세션 데이터만 불러오기
def load_chat_from_db(session_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    history = []
    for row in rows:
        role, content = row
        history.append({"role": role, "content": content})
    return history

##########################################
# 1) 유틸 함수들
##########################################

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

# 메모리 상에 로드된 모델들을 저장하는 캐시
LOCAL_MODELS_ROOT = "./models"

def build_model_cache_key(model_id: str, model_type: str, local_path: str = None) -> str:
    """
    models_cache에 사용될 key를 구성.
    - 만약 model_id == 'Local (Custom Path)' 이고 local_path가 주어지면 'local::{local_path}'
    - 그 외에는 'auto::{model_type}::{local_dir}::hf::{model_id}' 형태.
    """
    if model_id == "Local (Custom Path)" and local_path:
        return f"local::{local_path}"
    elif model_type == "api":
        return f"api::{model_id}"
    else:
        local_dirname = make_local_dir_name(model_id)
        local_dirpath = os.path.join("./models", model_type, local_dirname)
        return f"auto::{model_type}::{local_dirpath}::hf::{model_id}"

def clear_model_cache(model_id: str, local_path: str = None) -> str:
    """
    특정 모델에 대한 캐시를 제거 (models_cache에서 해당 key를 삭제).
    - 만약 해당 key가 없으면 '이미 없음' 메시지 반환
    - 성공 시 '캐시 삭제 완료' 메시지
    """
    # 모델 유형을 결정해야 합니다.
    if model_id in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]:
        model_type = "api"
    else:
        # 로컬 모델의 기본 유형을 transformers로 설정 (필요 시 수정)
        model_type = "transformers"
    key = build_model_cache_key(model_id, model_type, local_path)
    if key in models_cache:
        del models_cache[key]
        msg = f"[cache] 모델 캐시 제거: {key}"
        logger.info(msg)
        return msg
    else:
        msg = f"[cache] 이미 캐시에 없거나, 로드된 적 없음: {key}"
        logger.info(msg)
        return msg

def refresh_model_list():
    new_local_models = get_all_local_models()
    api_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    local_models = (
        new_local_models["transformers"] + 
        new_local_models["gguf"] + 
        new_local_models["mlx"]
    )
    new_choices = api_models + local_models
    new_choices = sorted(list(dict.fromkeys(new_choices)))
    return gr.update(choices=new_choices), "모델 목록을 새로고침했습니다."


def clear_all_model_cache():
    """
    현재 메모리에 로드된 모든 모델 캐시(models_cache)를 한 번에 삭제.
    필요하다면, 로컬 폴더의 .cache들도 일괄 삭제할 수 있음.
    """
    for key, handler in list(models_cache.items()):
        # 혹시 model, tokenizer 등 메모리를 점유하는 속성이 있으면 제거
        if hasattr(handler, "model"):
            del handler.model
        if hasattr(handler, "tokenizer"):
            del handler.tokenizer
        # 필요 시 handler 내부의 다른 자원들(예: embeddings 등)도 정리
        
    # 1) 메모리 캐시 전부 삭제
    count = len(models_cache)
    models_cache.clear()
    logger.info(f"[*] 메모리 캐시 삭제: {count}개 모델")

    # 2) (선택) 로컬 폴더 .cache 삭제
    #    예: ./models/*/.cache 폴더 전부 삭제
    #    원치 않으면 주석처리
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    gc.collect()
        
    cache_deleted = 0
    for subdir, models in get_all_local_models().items():
        for folder in models:
            folder_path = os.path.join(LOCAL_MODELS_ROOT, subdir, folder)
            if os.path.isdir(folder_path):
                cache_path = os.path.join(folder_path, ".cache")
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                    cache_deleted += 1
    logger.info(f"[*] 로컬 폴더 .cache 삭제: {cache_deleted}개 폴더 삭제")
    return f"[cache all] {count}개 모델 캐시 삭제 완료. 로컬 폴더 .cache {cache_deleted}개 삭제."

##########################################
# 2) 모델 로드 & 추론 로직
##########################################

def load_model(selected_model, model_type, quantization_bit="Q8_0", local_model_path=None, api_key=None):
    """
    모델 로드 함수. 특정 모델에 대한 로드 로직을 외부 핸들러로 분리.
    """
    model_id = selected_model
    if model_type not in ["transformers", "gguf", "mlx", "api"]:
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None
    if model_type == "api":
        # API 모델은 별도의 로드가 필요 없으므로 핸들러 생성 안함
        return None
    if model_type == "gguf":
        # GGUF 모델 로딩 로직
        if not ensure_model_available(model_id, local_model_path, model_type):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = GGUFModelHandler(
            model_id=model_id,
            quantization_bit=quantization_bit,
            local_model_path=local_model_path,
            model_type=model_type
        )
        models_cache[build_model_cache_key(model_id, model_type, quantization_bit, local_model_path)] = handler
        return handler
    elif model_type == "mlx":
        if "vision" in model_id.lower() or "qwen2-vl" in model_id.lower():
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = MlxVisionHandler(
                model_id=model_id,
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        else:
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = MlxModelHandler(
                model_id=model_id,
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
    else:
        if model_id == "openbmb/MiniCPM-Llama3-V-2_5":
            # 모델 존재 확인 및 다운로드
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = MiniCPMLlama3V25Handler(
                model_id=model_id,
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        elif model_id in [
            "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
        ] or ("vision" in model_id.lower() and model_id != "Bllossom/llama-3.1-Korean-Bllossom-Vision-8B"):
            # 모델 존재 확인 및 다운로드
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = VisionModelHandler(
                model_id=model_id,
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        elif model_id == "THUDM/glm-4v-9b":
            # 모델 존재 확인 및 다운로드
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = GLM4VHandler(
                model_id=model_id,
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        elif model_id == "THUDM/glm-4-9b-chat":
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = GLM4Handler(
                model_id=model_id,
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        elif model_id in ["THUDM/glm-4-9b-chat-hf", "THUDM/glm-4-9b-chat-1m-hf"]:
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = GLM4HfHandler(
                model_id=model_id,  # model_id가 정의되어 있어야 합니다.
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        elif model_id in ["bean980310/glm-4-9b-chat-hf_float8", "genai-archive/glm-4-9b-chat-hf_int8"]:
            # 'fp8' 특화 핸들러 로직 추가
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = GLM4HfHandler(
                model_id=model_id,  # model_id가 정의되어 있어야 합니다.
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        elif model_id in ["CohereForAI/aya-23-8B", "CohereForAI/aya-23-35B"]:
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = Aya23Handler(
                model_id=model_id,  # model_id가 정의되어 있어야 합니다.
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        elif "qwen" in model_id.lower():
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = QwenHandler(
                model_id=model_id,  # model_id가 정의되어 있어야 합니다.
                local_model_path=local_model_path,
                model_type=model_type
            )
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler
        else:
            if not ensure_model_available(model_id, local_model_path, model_type):
                logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
                return None
            handler = OtherModelHandler(model_id, local_model_path=local_model_path, model_type=model_type)
            models_cache[build_model_cache_key(model_id, model_type)] = handler
            return handler

def generate_answer(history, selected_model, model_type, local_model_path=None, image_input=None, api_key=None):
    """
    사용자 히스토리를 기반으로 답변 생성.
    """
    if not history:
        system_message = {
            "role": "system",
            "content": "당신은 유용한 AI 비서입니다."
        }
        history = [system_message]
    
    cache_key = build_model_cache_key(selected_model, model_type, local_path=local_model_path)
    handler = models_cache.get(cache_key)
    
    if model_type == "api":
        if not api_key:
            logger.error("OpenAI API Key가 missing.")
            return "OpenAI API Key가 필요합니다."
        openai.api_key = api_key
        messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
        logger.info(f"[*] OpenAI API 요청: {messages}")
        
        try:
            response = openai.ChatCompletion.create(
                model=selected_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            answer = response.choices[0].message["content"]
            logger.info(f"[*] OpenAI 응답: {answer}")
            return answer
        except Exception as e:
            logger.error(f"OpenAI API 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
    
    else:
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, model_type, local_model_path=local_model_path)
        
        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."
        
        logger.info(f"[*] Generating answer using {handler.__class__.__name__}")
        try:
            if isinstance(handler, VisionModelHandler):
                answer = handler.generate_answer(history, image_input)
            else:
                answer = handler.generate_answer(history)
            return answer
        except Exception as e:
            logger.error(f"모델 추론 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"

##########################################
# 3) Gradio UI
##########################################

with gr.Blocks() as demo:
    gr.Markdown("## 간단한 Chatbot")
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
        "openbmb/MiniCPM-Llama3-V-2_5",
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
    
    custom_model_path_state = gr.State("")
    session_id_state = gr.State(None)
    system_message_box = gr.Textbox(
        label="시스템 메시지",
        value="당신은 유용한 AI 비서입니다.",
        placeholder="대화의 성격, 말투 등을 정의하세요."
    )
        
    with gr.Tab("메인"):
        
        history_state = gr.State([])
        
        initial_choices = api_models + transformers_local + gguf_local + mlx_local + ["사용자 지정 모델 경로 변경"]
        initial_choices = list(dict.fromkeys(initial_choices))
        initial_choices = sorted(initial_choices)  # 정렬 추가
        
        with gr.Row():
            model_type_dropdown = gr.Radio(
                label="모델 유형 선택",
                choices=["all", "transformers", "gguf", "mlx"],
                value="all",
            )
        
        model_dropdown = gr.Dropdown(
            label="모델 선택",
            choices=initial_choices,
            value=initial_choices[0] if len(initial_choices) > 0 else None,
        )
        
        api_key_text = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-...",
            visible=False  # 기본적으로 숨김
        )
        image_info = gr.Markdown("", visible=False)
        with gr.Column():
            with gr.Row():
                image_input = gr.Image(label="이미지 업로드 (선택)", type="pil", visible=False)
                chatbot = gr.Chatbot(height=400, label="Chatbot", type="messages")
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
        history_state = gr.State([])
        
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
        
        def on_app_start():
            """
            Gradio 앱이 로드되면서 실행될 콜백.
            - 세션 ID를 정하고,
            - 해당 세션의 히스토리를 DB에서 불러온 뒤 반환.
            """
            # 여기서 필요한 테이블 생성 등을 해도 됨
            # create_table_if_not_exists()  # 필요 시 구현
            
            sid = "demo_session"  # 데모용으로 고정. 실제로는 secrets.token_hex() 등을 쓸 수 있음.
            loaded_history = load_chat_from_db(sid)
            # 만약 DB에 기록이 없으면 빈 리스트가 반환될 것
            return sid, loaded_history
        
        # .load()를 사용해, 페이지 로딩시 자동으로 on_app_start()가 실행되도록 연결
        demo.load(
            fn=on_app_start,
            inputs=[],
            outputs=[session_id_state, history_state],
            queue=False
        )
        
        def user_message(user_input, history, system_msg):
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
    
        def bot_message(session_id, history, selected_model, custom_path, image, api_key):
            # 모델 유형 결정
            local_model_path = None
            if selected_model in api_models:
                model_type = "api"
                local_model_path = None
            elif selected_model == "사용자 지정 모델 경로 변경":
                # 사용자 지정 모델 경로 사용
                model_type = "transformers"  # 기본 모델 유형 설정, 필요 시 수정
                local_model_path = custom_path
            else:
                # 로컬 모델 유형 결정 (transformers, gguf, mlx)
                if selected_model in transformers_local:
                    model_type = "transformers"
                elif selected_model in gguf_local:
                    model_type = "gguf"
                elif selected_model in mlx_local:
                    model_type = "mlx"
                else:
                    model_type = "transformers"  # 기본값
                local_model_path = None  # 기본 로컬 경로 사용
                
            try:
                answer = generate_answer(history, selected_model, model_type, local_model_path, image, api_key)
            except Exception as e:
                answer = f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
                
            history.append({"role": "assistant", "content": answer})
            
            save_chat_history_db(history, session_id=session_id)
            return history, ""  # 로딩 상태 제거
    

        def filter_messages_for_chatbot(history):
            messages_for_chatbot = []
            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    content = msg["content"] or ""
                    messages_for_chatbot.append({"role": msg["role"], "content": content})
            return messages_for_chatbot

        # 메시지 전송 시 함수 연결
        msg.submit(
            fn=user_message,
            inputs=[msg, session_id_state, history_state, system_message_box],  # 세 번째 파라미터 추가
            outputs=[msg, history_state, status_text],
            queue=False
        ).then(
            fn=bot_message,
            inputs=[session_id_state, history_state, model_dropdown, custom_model_path_state, image_input, api_key_text],
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
            inputs=[msg, session_id_state, history_state, system_message_box],
            outputs=[msg, history_state, status_text],
            queue=False
        ).then(
            fn=bot_message,
            inputs=[session_id_state, history_state, model_dropdown, custom_model_path_state, image_input, api_key_text],
            outputs=[history_state, status_text],
            queue=True
        ).then(
            fn=filter_messages_for_chatbot,            # 추가된 부분
            inputs=[history_state],
            outputs=chatbot,                           # chatbot에 최종 전달
            queue=False
        )
    
    with gr.Tab("다운로드"):
        gr.Markdown("""### 모델 다운로드
        HuggingFace에서 모델을 다운로드하고 로컬에 저장합니다. 
        미리 정의된 모델 목록에서 선택하거나, 커스텀 모델 ID를 직접 입력할 수 있습니다.""")
        
        with gr.Column():
            # 다운로드 모드 선택 (라디오 버튼)
            download_mode = gr.Radio(
                label="다운로드 방식 선택",
                choices=["Predefined", "Custom Repo ID"],
                value="Predefined",
                container=True,
            )
            # 모델 선택/입력 영역
            with gr.Column(visible=True) as predefined_column:
                predefined_dropdown = gr.Dropdown(
                    label="모델 선택",
                    choices=sorted(known_hf_models),
                    value=known_hf_models[0] if known_hf_models else None,
                    info="지원되는 모델 목록입니다."
                )
                
            with gr.Column(visible=False) as custom_column:
                custom_repo_id_box = gr.Textbox(
                    label="Custom Model ID",
                    placeholder="예) facebook/opt-350m",
                    info="HuggingFace의 모델 ID를 입력하세요 (예: organization/model-name)"
                )
                
            # 다운로드 설정
            with gr.Row():
                with gr.Column(scale=2):
                    target_path = gr.Textbox(
                        label="저장 경로",
                        placeholder="./models/my-model",
                        value="",
                        interactive=True,
                        info="비워두면 자동으로 경로가 생성됩니다."
                    )
                with gr.Column(scale=1):
                    use_auth = gr.Checkbox(
                        label="인증 필요",
                        value=False,
                        info="비공개 또는 gated 모델 다운로드 시 체크"
                    )
            
            with gr.Column(visible=False) as auth_column:
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    placeholder="hf_...",
                    type="password",
                    info="HuggingFace에서 발급받은 토큰을 입력하세요."
                )
            
            # 다운로드 버튼과 진행 상태
            with gr.Row():
                download_btn = gr.Button(
                    "다운로드 시작",
                    variant="primary",
                    scale=2
                )
                cancel_btn = gr.Button(
                    "취소",
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
            with gr.Accordion("상세 정보", open=False):
                download_info = gr.TextArea(
                    label="다운로드 로그",
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

        def toggle_auth(use_auth_val):
            """인증 필요 여부에 따라 토큰 입력창 표시/숨김"""
            return gr.update(visible=use_auth_val)

        def download_with_progress(mode, predefined_choice, custom_repo, target_dir, use_auth_val, token):
            try:
                repo_id = predefined_choice if mode == "Predefined" else custom_repo.strip()
                if not repo_id:
                    yield (
                        "❌ 모델 ID를 입력해주세요.",  # status
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
                    "🔄 다운로드 준비 중...",
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    f"모델: {repo_id}\n준비 중...",
                    gr.Dropdown.update()
                )

                # 실제 다운로드 수행
                yield (
                    "🔄 다운로드 중...",
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    "다운로드를 진행 중입니다...",
                    gr.Dropdown.update()
                )
                result = download_model_from_hf(
                    repo_id,
                    target_dir or os.path.join("./models", model_type, make_local_dir_name(repo_id)),
                    model_type=model_type
                )

                # 다운로드 완료 후 UI 업데이트
                yield (
                    "✅ 다운로드 완료!" if "실패" not in result else "❌ 다운로드 실패",
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    result,
                    gr.Dropdown.update(choices=sorted(api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"] + ["사용자 지정 모델 경로 변경"]))
                )

            except Exception as e:
                yield (
                    "❌ 오류 발생",
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    f"오류: {str(e)}\n\n{traceback.format_exc()}",
                    gr.Dropdown.update()
                )

        # 이벤트 연결
        download_mode.change(
            fn=toggle_download_mode,
            inputs=download_mode,
            outputs=[predefined_column, custom_column]
        )
        
        use_auth.change(
            fn=toggle_auth,
            inputs=use_auth,
            outputs=[auth_column]
        )
        
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
                model_dropdown  # model_dropdown을 업데이트하도록 변경
            ]
        )
    with gr.Tab("허브"):
        gr.Markdown("""### 허깅페이스 허브 모델 검색
        허깅페이스 허브에서 모델을 검색하고 다운로드할 수 있습니다. 
        키워드로 검색하거나 필터를 사용하여 원하는 모델을 찾을 수 있습니다.""")
        
        with gr.Row():
            search_box = gr.Textbox(
                label="검색어",
                placeholder="모델 이름, 태그 또는 키워드를 입력하세요",
                scale=4
            )
            search_btn = gr.Button("검색", scale=1)
            
        with gr.Row():
            with gr.Column(scale=1):
                model_type_filter = gr.Dropdown(
                    label="모델 유형",
                    choices=["All", "Text Generation", "Vision", "Audio", "Other"],
                    value="All"
                )
                language_filter = gr.Dropdown(
                    label="언어",
                    choices=["All", "Korean", "English", "Chinese", "Japanese", "Multilingual"],
                    value="All"
                )
                library_filter = gr.Dropdown(
                    label="라이브러리",
                    choices=["All", "Transformers", "GGUF", "MLX"],
                    value="All"
                )
            with gr.Column(scale=3):
                model_list = gr.Dataframe(
                    headers=["Model ID", "Description", "Downloads", "Likes"],
                    label="검색 결과",
                    interactive=False
                )
        
        with gr.Row():
            selected_model = gr.Textbox(
                label="선택된 모델",
                interactive=False
            )
            
        # 다운로드 설정
        with gr.Row():
            with gr.Column(scale=2):
                target_path = gr.Textbox(
                    label="저장 경로",
                    placeholder="./models/my-model",
                    value="",
                    interactive=True,
                    info="비워두면 자동으로 경로가 생성됩니다."
                )
            with gr.Column(scale=1):
                use_auth = gr.Checkbox(
                    label="인증 필요",
                    value=False,
                    info="비공개 또는 gated 모델 다운로드 시 체크"
                )
        
        with gr.Column(visible=False) as auth_column:
            hf_token = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_...",
                type="password",
                info="HuggingFace에서 발급받은 토큰을 입력하세요."
            )
        
        # 다운로드 버튼과 진행 상태
        with gr.Row():
            download_btn = gr.Button(
                "다운로드",
                variant="primary",
                scale=2
            )
            cancel_btn = gr.Button(
                "취소",
                variant="stop",
                scale=1,
                interactive=False
            )
            
        # 상태 표시
        download_status = gr.Markdown("")
        progress_bar = gr.Progress(track_tqdm=True)
        
        # 다운로드 결과와 로그
        with gr.Accordion("상세 정보", open=False):
            download_info = gr.TextArea(
                label="다운로드 로그",
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
                model_list = []
                for model in filtered_models:
                    description = model.cardData.get('description', '') if model.cardData else 'No description available.'
                    short_description = (description[:100] + "...") if len(description) > 100 else description
                    model_list.append([
                        model.id,
                        short_description,
                        model.downloads,
                        model.likes
                    ])
                return model_list
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
                        "❌ 모델을 선택해주세요.",
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
                    "🔄 다운로드 준비 중...",
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    f"모델: {model_id}\n준비 중...",
                    gr.Dropdown.update()
                )

                # 실제 다운로드 수행
                progress(0.5, desc="다운로드 중...")
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
                    "✅ 다운로드 완료!" if "실패" not in result else "❌ 다운로드 실패",
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    result,
                    gr.Dropdown.update(choices=new_choices)
                )

            except Exception as e:
                yield (
                    "❌ 오류 발생",
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
        gr.Markdown("### 사용자 지정 모델 경로 설정")
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
        gr.Markdown("### 채팅 기록 저장")
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
        
        gr.Markdown('### 채팅 히스토리 재로드')
        
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
        
        session_manage_info = gr.Textbox(
            label="세션 관리 결과",
            interactive=False
        )
        
        def refresh_sessions():
            """
            세션 목록 갱신: DB에서 세션 ID들을 불러와서 Dropdown에 업데이트
            """
            sessions = get_existing_sessions()
            if not sessions:
                return gr.update(choices=[], value=None), "DB에 세션이 없습니다."
            return gr.update(choices=sessions, value=sessions[0]), "세션 목록을 불러왔습니다."
        
        def create_new_session():
            """
            새 세션 ID를 만든 뒤, session_id_state에 반영
            """
            new_sid = secrets.token_hex(8)
            # 실제로는 DB에 넣을 필요는 없으며, 채팅 최초 저장 시 자동으로 들어갈 것
            return new_sid, f"새 세션 생성: {new_sid}"

        def apply_session(chosen_sid):
            """
            Dropdown에서 선택된 세션 ID로, DB에서 history를 불러오고, session_id_state를 갱신
            """
            if not chosen_sid:
                return [], None, "세션 ID를 선택하세요."
            loaded_history = load_chat_from_db(chosen_sid)
            # history_state에 반영하고, session_id_state도 업데이트
            return loaded_history, chosen_sid, f"세션 {chosen_sid}이 적용되었습니다."
        
        # 버튼 이벤트 연결
        refresh_sessions_btn.click(
            fn=refresh_sessions,
            inputs=[],
            outputs=[existing_sessions_dropdown, session_manage_info]
        )
        
        create_new_session_btn.click(
            fn=create_new_session,
            inputs=[],
            outputs=[session_id_state, session_manage_info]
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

demo.launch(debug=True, inbrowser=True, server_port=7861, width=500)