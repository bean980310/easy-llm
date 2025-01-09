import os
import shutil
import traceback
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import logging
from logging.handlers import RotatingFileHandler
from model_handlers import (
    MiniCPMLlama3V25Handler, GLM4Handler, GLM4VHandler, VisionModelHandler,
    Aya23Handler, GLM4HfHandler, OtherModelHandler
)
from utils import (
    make_local_dir_name,
    scan_local_models,
    download_model_from_hf,
    ensure_model_available,
    convert_and_save
)
from cache import models_cache 
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
    elif "gpt" in model_id:
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
    key = build_model_cache_key(model_id, local_path)
    if key in models_cache:
        del models_cache[key]
        msg = f"[cache] 모델 캐시 제거: {key}"
        logger.info(msg)
        return msg
    else:
        msg = f"[cache] 이미 캐시에 없거나, 로드된 적 없음: {key}"
        logger.info(msg)
        return msg
    
def clear_all_model_cache():
    """
    현재 메모리에 로드된 모든 모델 캐시(models_cache)를 한 번에 삭제.
    필요하다면, 로컬 폴더의 .cache들도 일괄 삭제할 수 있음.
    """
    # 1) 메모리 캐시 전부 삭제
    count = len(models_cache)
    models_cache.clear()
    logger.info(f"[*] 메모리 캐시 삭제: {count}개 모델")

    # 2) (선택) 로컬 폴더 .cache 삭제
    #    예: ./models/*/.cache 폴더 전부 삭제
    #    원치 않으면 주석처리
    cache_deleted = 0
    for folder in os.listdir(LOCAL_MODELS_ROOT):
        folder_path = os.path.join(LOCAL_MODELS_ROOT, folder)
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

def get_terminators(tokenizer):
    """
    모델별 종료 토큰 ID를 반환하는 함수
    """
    if "glm" in str(tokenizer.__class__).lower():
        # GLM 모델용 특수 처리
        return [tokenizer.eos_token_id]  # GLM의 EOS 토큰 사용
    else:
        # 기존 다른 모델들을 위한 처리
        return [
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
        ]
# app.py

def load_model(model_id, model_type, local_model_path=None, api_key=None):
    """
    모델 로드 함수. 특정 모델에 대한 로드 로직을 외부 핸들러로 분리.
    """
    if model_type not in ["transformers", "gguf", "mlx"]:
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None
    if model_id == "openbmb/MiniCPM-Llama3-V-2_5":
        # 모델 존재 확인 및 다운로드
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = MiniCPMLlama3V25Handler(model_dir=local_model_path or f"./models/{make_local_dir_name(model_id)}")
        models_cache[model_id] = handler
        return handler
    elif model_id in [
        "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
    ] or "vision" in model_id.lower() and model_id != "Bllossom/llama-3.1-Korean-Bllossom-Vision-8B":
        # 모델 존재 확인 및 다운로드
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = VisionModelHandler(model_dir=local_model_path or f"./models/{make_local_dir_name(model_id)}")
        models_cache[model_id] = handler
        return handler
    elif model_id == "THUDM/glm-4v-9b":
        # 모델 존재 확인 및 다운로드
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = GLM4VHandler(model_dir=local_model_path or f"./models/{make_local_dir_name(model_id)}")
        models_cache[model_id] = handler
        return handler
    elif model_id == "THUDM/glm-4-9b-chat":
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = GLM4Handler(model_dir=local_model_path or f"./models/{make_local_dir_name(model_id)}")
        models_cache[model_id] = handler
        return handler
    elif model_id in ["THUDM/glm-4-9b-chat-hf", "THUDM/glm-4-9b-chat-1m-hf"]:
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = GLM4HfHandler(model_dir=local_model_path or f"./models/{make_local_dir_name(model_id)}")
        models_cache[model_id] = handler
        return handler
    elif model_id in ["bean980310/glm-4-9b-chat-hf_float8", "genai-archive/glm-4-9b-chat-hf_int8"]:
        # 'fp8' 특화 핸들러 로직 추가
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = GLM4HfHandler(model_dir=local_model_path or f"./models/{make_local_dir_name(model_id)}")
        models_cache[model_id] = handler
        return handler
    elif model_id == "CohereForAI/aya-23-8B" or model_id == "CohereForAI/aya-23-35B":
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = Aya23Handler(model_dir=local_model_path or f"./models/{make_local_dir_name(model_id)}")
        models_cache[model_id] = handler
        return handler
    else:
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None
        handler = OtherModelHandler(model_id, local_model_path=local_model_path, model_type=model_type)
        models_cache[model_id] = handler
        return handler

# app.py

def generate_answer(history, selected_model, model_type, local_model_path=None, image_input=None, api_key=None):
    """
    사용자 히스토리를 기반으로 답변 생성.
    """
    cache_key = build_model_cache_key(selected_model, model_type, local_path=local_model_path)
    model_cache = models_cache.get(cache_key, {})
    
    if "gpt" in selected_model:
        if not api_key:
            logger.error("OpenAI API Key is missing.")
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
    
    elif selected_model in [
        "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
    ] or "vision" in selected_model.lower():
        handler: VisionModelHandler = models_cache.get(selected_model)
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, local_model_path=local_model_path)
        
        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."
        
        logger.info(f"[*] Generating answer using VisionModelHandler")
        answer = handler.generate_answer(history, image_input)
        return answer
    
    elif selected_model == "openbmb/MiniCPM-Llama3-V-2_5":
        handler: MiniCPMLlama3V25Handler = models_cache.get(selected_model)
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, local_model_path=local_model_path)
        
        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."
        
        logger.info(f"[*] Generating answer using MiniCPMLlama3V25Handler")
        # image_input 파라미터 전달 추가
        logger.info(f"[*] Image input provided: {image_input is not None}")
        answer = handler.generate_answer(history, image_input=image_input)
        return answer
    
    elif selected_model == "THUDM/glm-4v-9b":
        handler: GLM4VHandler = models_cache.get(selected_model)
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, local_model_path=local_model_path)

        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."

        logger.info(f"[*] Generating answer using GLM4VHandler")
        answer = handler.generate_answer(history)
        return answer
    elif selected_model == "THUDM/glm-4-9b-chat":
        handler: GLM4Handler = models_cache.get(selected_model)
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, local_model_path=local_model_path)

        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."

        logger.info(f"[*] Generating answer using GLM4Handler")
        answer = handler.generate_answer(history)
        return answer
    elif selected_model in ["THUDM/glm-4-9b-chat-hf", "THUDM/glm-4-9b-chat-1m-hf", "bean980310/glm-4-9b-chat-hf_float8", "genai-archive/glm-4-9b-chat-hf_int8"] :
        handler: GLM4HfHandler = models_cache.get(selected_model)
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, local_model_path=local_model_path)

        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."

        logger.info(f"[*] Generating answer using GLM4Handler")
        answer = handler.generate_answer(history)
        return answer
    elif selected_model in ["CohereForAI/aya-23-8B", "CohereForAI/aya-23-35B"]:
        handler: Aya23Handler = models_cache.get(selected_model)
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, local_model_path=local_model_path)

        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."
        
        logger.info(f"[*] Generating answer using Aya23Handler")
        answer = handler.generate_answer(history)
        return answer
    else:
        handler: OtherModelHandler = models_cache.get(selected_model)
        if not handler:
            logger.info(f"[*] 모델 로드 중: {selected_model}")
            handler = load_model(selected_model, local_model_path=local_model_path)

        if not handler:
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."
        logger.info(f"[*] Generating answer using default handler for model: {selected_model}")
        answer = handler.generate_answer(history)
        return answer

##########################################
# 3) Gradio UI
##########################################

with gr.Blocks() as demo:
    gr.Markdown("## 간단한 Chatbot")
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
        "bean980310/glm-4-9b-chat-hf_float8",
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
        "EleutherAI/polyglot-ko-1.3b",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo"
    ]
    with gr.Tab("메인"):
        local_model_folders = scan_local_models()
        initial_choices = known_hf_models + local_model_folders + ["Local (Custom Path)"]
        initial_choices = list(dict.fromkeys(initial_choices))

        with gr.Row():
            model_type_dropdown = gr.Radio(
                label="모델 유형 선택",
                choices=["transformers", "gguf", "mlx"],
                value="transformers",
                inline=True
            )
            
        model_dropdown = gr.Dropdown(
            label="모델 선택",
            choices=initial_choices,
            value=initial_choices[0] if len(initial_choices) > 0 else None,
        )
        local_path_text = gr.Textbox(
            label="(Local Path) 로컬 폴더 경로",
            placeholder="./models/my-llama",
            visible=False  # 기본적으로 숨김
        )
        api_key_text = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-...",
            visible=False  # 기본적으로 숨김
        )
        image_info = gr.Markdown("", visible=False)
        with gr.Column():
            # image_input을 먼저 정의합니다.
            with gr.Row():
                image_input = gr.Image(label="이미지 업로드 (선택)", type="pil", visible=False)  # 초기 상태 숨김
                chatbot = gr.Chatbot(height=400, label="Chatbot", type="messages")  # 'type' 파라미터 설정
            with gr.Row():
                msg = gr.Textbox(
                    label="메시지 입력",
                    placeholder="메시지를 입력하세요...",
                    scale=9  # 90% 차지
                )
                send_btn = gr.Button(
                    "전송",
                    scale=1,  # 10% 차지
                    variant="primary"
                )
            with gr.Row():
                status_text = gr.Markdown("", elem_id="status_text")
        history_state = gr.State([])
        
        def toggle_api_key_display(selected_model):
            """
            OpenAI API Key 입력 필드와 로컬 경로 입력 필드의 가시성을 제어합니다.
            """
            api_visible = "gpt" in selected_model
            local_path_visible = selected_model == "Local (Custom Path)"
            return gr.update(visible=api_visible), gr.update(visible=local_path_visible)
    
        def toggle_image_input(selected_model):
            """
            이미지 업로드 필드와 정보 메시지의 가시성을 제어합니다.
            """
            requires_image = (
                "vision" in selected_model.lower() or
                selected_model in [
                    "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
                    "THUDM/glm-4v-9b",
                    "openbmb/MiniCPM-Llama3-V-2_5"
                ]
            )
            if requires_image:
                return gr.update(visible=True), "이미지를 업로드해주세요."
            else:
                return gr.update(visible=False), "이미지 입력이 필요하지 않습니다."
    
        def update_model_dropdown(model_type, _):
            """
            모델 유형에 따라 모델 드롭다운의 선택지를 업데이트합니다.
            """
            return scan_local_models(model_type=model_type)

        model_type_dropdown.change(
            fn=update_model_dropdown,
            inputs=[model_type_dropdown, model_dropdown],
            outputs=[model_dropdown]
        )

        model_dropdown.change(
            fn=toggle_api_key_display,
            inputs=[model_dropdown],
            outputs=[api_key_text, local_path_text]
        ).then(
            fn=toggle_image_input,
            inputs=[model_dropdown],
            outputs=[image_input, image_info]  # 인스턴스를 사용
        )
        def user_message(user_input, history):
            if not user_input.strip():
                return "", history, ""
            history = history + [{"role": "user", "content": user_input}]
            return "", history, "🤔 답변을 생성하는 중입니다..."

        def bot_message(history, selected_model, local_model_path, image, api_key):
            try:
                answer = generate_answer(history, selected_model, local_model_path, image, api_key)
            except Exception as e:
                answer = f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"
            history = history + [{"role": "assistant", "content": answer}]
            return history, ""  # 로딩 상태 제거

        # 메시지 전송 시 함수 연결
        msg.submit(
            fn=user_message,
            inputs=[msg, history_state],
            outputs=[msg, history_state, status_text],
            queue=False  # 사용자 입력은 즉시 처리
        ).then(
            fn=bot_message,
            inputs=[history_state, model_dropdown, local_path_text, image_input, api_key_text],
            outputs=[history_state, status_text],
            queue=True  # 모델 생성은 큐에서 처리
        ).then(
            fn=lambda h: h,
            inputs=history_state,
            outputs=chatbot,
            queue=False  # UI 업데이트는 즉시 처리
        )
        send_btn.click(
            fn=user_message,
            inputs=[msg, history_state],
            outputs=[msg, history_state, status_text],
            queue=False
        ).then(
            fn=bot_message,
            inputs=[history_state, model_dropdown, local_path_text, image_input, api_key_text],
            outputs=[history_state, status_text],
            queue=True
        ).then(
            fn=lambda h: h,
            inputs=history_state,
            outputs=chatbot,
            queue=False
        )
    with gr.Tab("다운로드"):
        gr.Markdown("""### 모델 다운로드
        HuggingFace에서 모델을 다운로드하고 로컬에 저장합니다. 
        미리 정의된 모델 목록에서 선택하거나, 커스텀 모델 ID를 직접 입력할 수 있습니다.""")
        
        with gr.Column():
            # 다운로드 모드 선택 (라디오 버튼을 세그먼트로 변경)
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
            return {
                auth_column: use_auth_val
            }

        def update_download_ui(
            status: str = "",
            btn_enabled: bool = True,
            cancel_enabled: bool = False,
            info: str = "",
            model_list = None
        ):
            """UI 업데이트를 위한 헬퍼 함수"""
            updates = {
                "download_status": status,
                "download_btn": gr.Button(interactive=btn_enabled),
                "cancel_btn": gr.Button(interactive=cancel_enabled),
                "download_info": info
            }
            if model_list is not None:
                updates["model_dropdown"] = gr.Dropdown(choices=model_list)
            return updates

        def download_with_progress(mode, predefined_choice, custom_repo, target_dir, use_auth_val, token, progress=gr.Progress()):
            """진행률 표시와 함께 모델 다운로드 수행"""
            try:
                repo_id = predefined_choice if mode == "Predefined" else custom_repo.strip()
                if not repo_id:
                    yield (
                        "❌ 모델 ID를 입력해주세요.",  # status
                        gr.Button(interactive=True),  # download_btn
                        gr.Button(interactive=False),  # cancel_btn
                        "다운로드가 시작되지 않았습니다.",  # download_info
                        None  # model_dropdown (no update)
                    )
                    return
                
                if "gguf" in repo_id.lower():
                    model_type = "gguf"
                elif "mlx" in repo_id.lower():
                    model_type = "mlx"
                else:
                    model_type = "transformers"


                # 진행 상태 초기화
                progress(0, desc="준비 중...")
                yield (
                    "🔄 다운로드 준비 중...",  # status
                    gr.Button(interactive=False),  # download_btn
                    gr.Button(interactive=True),  # cancel_btn
                    f"모델: {repo_id}\n준비 중...",  # download_info
                    None  # model_dropdown (no update)
                )

                # 실제 다운로드 수행
                progress(0.5, desc="다운로드 중...")
                result = download_model_from_hf(
                    repo_id,
                    target_dir or os.path.join("./models", model_type, make_local_dir_name(repo_id)),
                    model_type=model_type
                )


                # 다운로드 완료 후 UI 업데이트
                progress(1.0, desc="완료")
                yield (
                    "✅ 다운로드 완료!" if "실패" not in result else "❌ 다운로드 실패",  # status
                    gr.Button(interactive=True),  # download_btn
                    gr.Button(interactive=False),  # cancel_btn
                    result,  # download_info
                    gr.Dropdown(choices=scan_local_models())  # model_dropdown update
                )

            except Exception as e:
                yield (
                    "❌ 오류 발생",  # status
                    gr.Button(interactive=True),  # download_btn
                    gr.Button(interactive=False),  # cancel_btn
                    f"오류: {str(e)}\n\n{traceback.format_exc()}",  # download_info
                    None  # model_dropdown (no update)
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
            new_local_models = scan_local_models()
            # 새 choices: 기존 HF 모델 + 새 local 모델 + Local (Custom Path)
            new_choices = known_hf_models + new_local_models + ["Local (Custom Path)"]
            new_choices = list(dict.fromkeys(new_choices))
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
            quant_type = gr.Radio(choices=["float8", "int8"], label="변환 유형", value="int8")
        with gr.Row():
            push_to_hub = gr.Checkbox(label="Hugging Face Hub에 푸시", value=False)
        
        convert_button = gr.Button("모델 변환 시작")
        output = gr.Textbox(label="결과")
        
        convert_button.click(fn=convert_and_save, inputs=[model_id, output_dir, push_to_hub, quant_type], outputs=output)

demo.launch(debug=True, inbrowser=True, server_port=7861)