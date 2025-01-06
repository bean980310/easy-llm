import os
import shutil
from pathlib import Path
from PIL import Image
from threading import Thread
import torch
import gradio as gr
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor,
    MllamaForConditionalGeneration,
    TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
)
from huggingface_hub import snapshot_download
import openai
import logging
from logging.handlers import RotatingFileHandler
from model_handlers import MiniCPMLlama3V25Handler, GLM4VHandler, VisionModelHandler
from utils import (
    make_local_dir_name,
    scan_local_models,
    remove_hf_cache,
    download_model_from_hf,
    ensure_model_available
)
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
models_cache = {}

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 마지막 생성된 토큰이 stop_ids에 있는지 확인
        if input_ids[0][-1] in self.stop_ids:
            return True
        return False

def build_model_cache_key(model_id: str, local_path: str = None) -> str:
    """
    models_cache에 사용될 key를 구성.
    - 만약 model_id == 'Local (Custom Path)' 이고 local_path가 주어지면 'local::{local_path}'
    - 그 외에는 'auto::{local_dir}::hf::{model_id}' 형태.
    """
    if model_id == "Local (Custom Path)" and local_path:
        return f"local::{local_path}"
    elif "gpt" in model_id:
        return f"api::{model_id}"
    else:
        local_dirname = make_local_dir_name(model_id)
        local_dirpath = os.path.join("./models", local_dirname)
        return f"auto::{local_dirpath}::hf::{model_id}"

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
    return [
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

# app.py

def load_model(model_id, local_model_path=None, api_key=None):
    """
    모델 로드 함수. 특정 모델에 대한 로드 로직을 외부 핸들러로 분리.
    """
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
    ] or "vision" in model_id.lower():
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
    else:
        # 기존 로직 유지
        logger.info(f"[*] Loading model: {model_id}")
        local_dirname = make_local_dir_name(model_id)
        local_dirpath = os.path.join(LOCAL_MODELS_ROOT, local_dirname)

        # 모델 존재 확인 및 다운로드
        if not ensure_model_available(model_id, local_model_path):
            logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
            return None

        # 로컬 폴더에서 로드
        logger.info(f"[*] 로컬 폴더 로드: {local_dirpath}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_dirpath, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                local_dirpath,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            models_cache[model_id] = {"tokenizer": tokenizer, "model": model}
            logger.info(f"[*] 모델 로드 완료: {model_id}")
            return models_cache[model_id]
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return None

# app.py

def generate_answer(history, selected_model, local_model_path=None, image_input=None, api_key=None):
    """
    사용자 히스토리를 기반으로 답변 생성.
    """
    cache_key = build_model_cache_key(selected_model, local_model_path)
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
            logger.error(f"OpenAI API 오류: {str(e)}")
            return f"오류 발생: {str(e)}"
    
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
        answer = handler.generate_answer(history)
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
    else:
        # 다른 모델 처리
        handler = load_model(selected_model, local_model_path=local_model_path)
        if not handler:
            return "모델 로드에 실패했습니다."

        # 기존 로직
        tokenizer = handler.get("tokenizer")
        model = handler.get("model")
        if not tokenizer or not model:
            logger.error("토크나이저 또는 모델이 로드되지 않았습니다.")
            return "토크나이저 또는 모델이 로드되지 않았습니다."

        terminators = get_terminators(tokenizer)
        prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
        logger.info(f"[*] Prompt messages for other models: {prompt_messages}")
        
        try:
            input_ids = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            logger.info("[*] 입력 템플릿 적용 완료")
        except Exception as e:
            logger.error(f"입력 템플릿 적용 중 오류 발생: {str(e)}")
            return f"입력 템플릿 적용 중 오류 발생: {str(e)}"

        try:
            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            logger.info("[*] 모델 생성 완료")
        except Exception as e:
            logger.error(f"모델 생성 중 오류 발생: {str(e)}")
            return f"모델 생성 중 오류 발생: {str(e)}"

        try:
            generated_text = tokenizer.decode(
                outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            logger.info(f"[*] 생성된 텍스트: {generated_text}")
        except Exception as e:
            logger.error(f"출력 디코딩 중 오류 발생: {str(e)}")
            return f"출력 디코딩 중 오류 발생: {str(e)}"
        
        return generated_text.strip()

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
        "EleutherAI/polyglot-ko-1.3b",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo"
    ]
    with gr.Tab("메인"):
        local_model_folders = scan_local_models()
        initial_choices = known_hf_models + local_model_folders + ["Local (Custom Path)"]
        initial_choices = list(dict.fromkeys(initial_choices))

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
            msg = gr.Textbox(label="메시지 입력")
        send_btn = gr.Button("보내기")
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
    
        # 모델 드롭다운 변경 시 함수 연결
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
                return "", history
                # 사용자 메시지 추가
            history = history + [{"role": "user", "content": user_input}]
            return "", history

        def bot_message(history, selected_model, local_model_path, image, api_key):
            try:
                answer = generate_answer(history, selected_model, local_model_path, image, api_key)
            except Exception as e:
                answer = f"오류 발생: {str(e)}"
            # 봇 응답 추가
            history = history + [{"role": "assistant", "content": answer}]
            return history

        # 메시지 전송 시 함수 연결
        msg.submit(
            fn=user_message,
            inputs=[msg, history_state],
            outputs=[msg, history_state]
        ).then(
            fn=bot_message,
            inputs=[history_state, model_dropdown, local_path_text, image_input, api_key_text],
            outputs=history_state
        ).then(
            fn=lambda h: h,
            inputs=history_state,
            outputs=chatbot
        )

        send_btn.click(
            fn=user_message,
            inputs=[msg, history_state],
            outputs=[msg, history_state]
        ).then(
            fn=bot_message,
            inputs=[history_state, model_dropdown, local_path_text, image_input, api_key_text],
            outputs=history_state
        ).then(
            fn=lambda h: h,
            inputs=history_state,
            outputs=chatbot
        )
    with gr.Tab("다운로드"):
        gr.Markdown("### 모델 다운로드")
        download_mode = gr.Radio(
            label="다운로드 모드 선택",
            choices=["Predefined", "Custom Repo ID"],
            value="Predefined"
        )
        with gr.Column():
            with gr.Row():
                predefined_dropdown = gr.Dropdown(
                    label="Predefined Model",
                    choices=known_hf_models,
                    value=known_hf_models[0],
                )
                custom_repo_id_box = gr.Textbox(label="Custom Repo ID", placeholder="예) Bllossom/llama-3.2-Korean-Bllossom-3B")
            download_btn = gr.Button("모델 다운로드")
            download_info = gr.Textbox(label="다운로드 결과", interactive=False)

        def download_and_update(mode, predefined_choice, custom_repo):
            """
            1) 모델 다운로드
            2) 모델 목록 자동 새로고침
            """
            if mode == "Predefined":
                repo_id = predefined_choice
            else:
                repo_id = custom_repo.strip()
            local_name = make_local_dir_name(repo_id)
            target_dir = os.path.join(LOCAL_MODELS_ROOT, local_name)
            msg1 = download_model_from_hf(repo_id.strip(), target_dir)
            # 다운로드 후, 모델 목록도 자동 갱신
            new_local_models = scan_local_models()
            new_choices = known_hf_models + new_local_models + ["Local (Custom Path)"]
            new_choices = list(dict.fromkeys(new_choices))
            # DropDown update + 결과 메시지
            return (
                gr.update(choices=new_choices),  # DropDown 갱신
                f"{msg1}\n[Auto-Refresh] 모델 목록을 업데이트했습니다."
            )

        download_btn.click(
            fn=download_and_update,
            inputs=[download_mode, predefined_dropdown, custom_repo_id_box],
            outputs=[model_dropdown, download_info],
            queue=False
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

demo.launch(debug=True, inbrowser=True, server_port=7861)