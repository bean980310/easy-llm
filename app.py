import os
import shutil
from pathlib import Path
from PIL import Image
from threading import Thread
import requests
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForVision2Seq, LlavaForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from huggingface_hub import snapshot_download
import openai

##########################################
# 1) 유틸 함수들
##########################################

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

def make_local_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")

def convert_folder_to_modelid(folder_name: str) -> str:
    """
    Bllossom__llama-3.2-Korean-Bllossom-3B
    => Bllossom/llama-3.2-Korean-Bllossom-3B
    """
    return folder_name.replace("__", "/")

def scan_local_models(root=LOCAL_MODELS_ROOT):
    """
    ./models 폴더 아래를 스캔하여,
    'config.json' 파일이 존재하는 서브폴더를 모델 폴더로 간주,
    ['폴더이름1', '폴더이름2', ...] 형태의 리스트를 반환.
    예: './models/Bllossom__llama-3.2-Korean-Bllossom-3B' 라면
        폴더 이름 'Bllossom__llama-3.2-Korean-Bllossom-3B'를 목록에 추가.
    """
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)

    local_model_ids = []
    for folder in os.listdir(root):
        full_path = os.path.join(root, folder)
        if os.path.isdir(full_path) and 'config.json' in os.listdir(full_path):
            model_id = convert_folder_to_modelid(folder)
            local_model_ids.append(model_id)
    return local_model_ids

def remove_hf_cache(model_id):
    """
    model_id에 대응하는 Hugging Face Hub 캐시 폴더를 찾아 삭제한다.
    예) '~/.cache/huggingface/hub/models--Bllossom--llama-3.2-Korean-Bllossom-3B'
    
    (아래 예시는 .cache 폴더를 ./models/{user}__{name}/.cache 형태로 가정하여 처리)
    필요에 맞게 수정 가능.
    """
    if "/" in model_id:
        user, name = model_id.split("/", maxsplit=1)
        cache_dirname = f"./models/{user}__{name}"
    else:
        cache_dirname = f"./models/{model_id}"

    cache_path = os.path.join(cache_dirname, ".cache")  # 예시로 ".cache" 폴더를 사용
    if os.path.isdir(cache_path):
        print(f"[*] 캐시 폴더 삭제: {cache_path}")
        shutil.rmtree(cache_path)
    else:
        print(f"[*] 캐시 폴더 없음: {cache_path}")


def download_model_from_hf(hf_repo_id: str, target_dir: str) -> str:
    """
    Hugging Face repo id를 target_dir에 스냅샷 다운로드.
    - remove_cache_after=True면, 다운로드 후 remove_hf_cache를 호출.
    - 반환값: 결과 메시지
    """
    if os.path.isdir(target_dir):
        return f"[알림] 이미 다운로드됨: {hf_repo_id} → {target_dir}"
    
    os.makedirs(target_dir, exist_ok=True)
    print(f"[*] 모델 '{hf_repo_id}'을(를) '{target_dir}'에 다운로드 중...")
    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=target_dir,
        ignore_patterns=["*.md", ".gitattributes", "original/", "LICENSE.txt", "LICENSE"]
    )
    remove_hf_cache(hf_repo_id)
    print(f"[+] 다운로드 & 저장 완료: {target_dir}")
    
    return f"모델 '{hf_repo_id}' 다운로드 완료 → 로컬 폴더 '{target_dir}'에 저장했습니다."

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
        return f"[cache] 모델 캐시 제거: {key}"
    else:
        return f"[cache] 이미 캐시에 없거나, 로드된 적 없음: {key}"
    
def clear_all_model_cache():
    """
    현재 메모리에 로드된 모든 모델 캐시(models_cache)를 한 번에 삭제.
    필요하다면, 로컬 폴더의 .cache들도 일괄 삭제할 수 있음.
    """
    # 1) 메모리 캐시 전부 삭제
    count = len(models_cache)
    models_cache.clear()

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

    return f"[cache all] {count}개 모델 캐시 삭제 완료. 로컬 폴더 .cache {cache_deleted}개 삭제."


##########################################
# 2) 모델 로드 & 추론 로직
##########################################

def get_terminators(tokenizer):
    return [
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

def load_model(model_id, local_model_path=None, api_key=None):
    """
    1) Local (Custom Path) -> local_model_path
    2) 그렇지 않다면 -> ./models/{model_id 치환} 폴더가 없으면 snapshot_download
    """
    cache_key = build_model_cache_key(model_id, local_model_path)

    # 캐시가 있으면 바로 반환
    if cache_key in models_cache:
        print(f"[*] 메모리 캐시 사용: {cache_key}")
        return models_cache[cache_key]["tokenizer"], models_cache[cache_key]["model"]
    
    # 로컬 경로(사용자 지정)
    if model_id == "Local (Custom Path)" and local_model_path:
        print(f"[*] 사용자 로컬 경로에서 모델 로드: {local_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        models_cache[cache_key] = {"tokenizer": tokenizer, "model": model}
        return tokenizer, model
    
    if "gpt" in model_id:
        if not api_key:
            raise ValueError("OpenAI API Key가 필요합니다.")
        # OpenAI 모델의 경우, 로컬에 모델을 로드하지 않고 API 호출에 필요한 설정을 캐시에 저장
        models_cache[cache_key] = {"api_key": api_key}
        return None, None  # 외부 API 모델은 로컬 모델이 아니므로 None 반환

    # 미리 지정된 모델 ID
    local_dirname = make_local_dir_name(model_id)
    local_dirpath = os.path.join(LOCAL_MODELS_ROOT, local_dirname)

    # 없다면 다운로드
    if not os.path.isdir(local_dirpath):
        print(f"[*] 폴더가 없어 다운로드 진행: {model_id} -> {local_dirpath}")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dirpath,
            ignore_patterns=["README.md", ".gitattributes"]
        )
        remove_hf_cache(model_id)
        print(f"[*] 다운로드 & 캐시정리 완료: {local_dirpath}")

    # 로컬 폴더에서 로드
    print(f"[*] 로컬 폴더 로드: {local_dirpath}")
    if "vision" in model_id.lower() or model_id=="Bllossom/llama-3.2-Korean-Bllossom-AICA-5B":
        processor=AutoProcessor.from_pretrained(local_dirpath, trust_remote_code=True)
        model=MllamaForConditionalGeneration.from_pretrained(
            local_dirpath, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", trust_remote_code=True)
        model.tie_weights()
        models_cache[cache_key] = {"processor": processor, "model": model}
        return processor, model
    elif model_id=="openbmb/MiniCPM-Llama3-V-2_5":
        tokenizer = AutoTokenizer.from_pretrained(local_dirpath, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            local_dirpath,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        models_cache[cache_key] = {"tokenizer": tokenizer, "model": model}
        return tokenizer, model
    elif model_id=="THUDM/glm-4v-9b":
        tokenizer = AutoTokenizer.from_pretrained(local_dirpath, 
                                                  trust_remote_code=True, 
                                                  encode_special_tokens=True)
        model = AutoModel.from_pretrained(
            local_dirpath,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        models_cache[cache_key] = {"tokenizer": tokenizer, "model": model}
        return tokenizer, model
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_dirpath, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_dirpath,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        models_cache[cache_key] = {"tokenizer": tokenizer, "model": model}
        return tokenizer, model


def generate_answer(history, selected_model, local_model_path=None, image_input=None, api_key=None):
    
    cache_key = build_model_cache_key(selected_model, local_model_path)
    model_cache = models_cache.get(cache_key, {})
    
    if "gpt" in selected_model:
        if not api_key:
            raise ValueError("OpenAI API Key가 필요합니다.")
        openai.api_key = api_key
        messages = []
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        if image_input:
            pass
        
        response = openai.ChatCompletion.create(
            model=selected_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9
        )
        return response.choices[0].message["content"]
    elif selected_model=="Bllossom/llama-3.2-Korean-Bllossom-AICA-5B" or "vision" in selected_model or "Vision" in selected_model:
        processor, model=load_model(selected_model, local_model_path)
        tokenizer=processor.tokenizer
        terminators=get_terminators(tokenizer)
        
        image=image_input
        
        prompt_messages = []
        for user_msg, bot_msg in history:
            if user_msg:
                prompt_messages.append({"role": "user", "content": [
                    {"type":"image"},
                    {"type":"text", "text": user_msg}]})
            if bot_msg:
                prompt_messages.append({"role": "assistant", "content": bot_msg})
                
        input_text=processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        if image_input:
            inputs=processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
        else:
            inputs=tokenizer(
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        
        generated_text=processor.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return generated_text.strip()
    elif selected_model == "openbmb/MiniCPM-Llama3-V-2_5":
        tokenizer, model=load_model(selected_model, local_model_path)
        prompt_messages=[]
        for user_msg, bot_msg in history:
            if user_msg:
                prompt_messages.append({"role": "user", 'content':[image_input, user_msg]})
            if bot_msg:
                prompt_messages.append({"role": "assistant", "content": bot_msg})
             
        if image_input:
            res = model.chat(
                image=image_input,
                msgs=prompt_messages,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7,
                stream=True
            )
        else:
            res = model.chat(
                image=None,
                msgs=prompt_messages,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7,
                stream=True
            )
        generated_text = ""
        for new_text in res:
            generated_text += new_text
        return generated_text.strip()
        
    elif selected_model == "THUDM/glm-4v-9b":
        tokenizer, model=load_model(selected_model, local_model_path)
        prompt_messages=[]
        for user_msg, bot_msg in history:
            if user_msg:
                prompt_messages.append({"role": "user", "image": image_input, "content": user_msg})
            if bot_msg:
                prompt_messages.append({"role": "assistant", "content": bot_msg})
                
        inputs=tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, 
                                             tokenize=True, return_tensors="pt",
                                             return_dict=True).to(next(model.parameters()).device)
        
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        input_ids = inputs["input_ids"]
        
        stop_ids = model.config.eos_token_id
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
        generate_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.6,
            "stopping_criteria": stopping_criteria,
            "repetition_penalty": 1.2,
            "eos_token_id": [151329, 151336, 151338],
        }
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        response = ""
        for new_token in streamer:
            if new_token:
                response += new_token
        generated_text = response
        
        return generated_text.strip()
    else:
        # 모델 로드
        tokenizer, model = load_model(selected_model, local_model_path)

        # 종결 토큰
        terminators = get_terminators(tokenizer)

        # Gradio Chatbot 형식 → 모델 입력 형식 변환
        prompt_messages = []
        for user_msg, bot_msg in history:
            if user_msg:
                prompt_messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                prompt_messages.append({"role": "assistant", "content": bot_msg})
            
        input_ids = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        generated_text = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
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

        with gr.Column():
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="모델 선택",
                    choices=initial_choices,
                    value=initial_choices[0] if len(initial_choices) > 0 else None,
                )
                local_path_text = gr.Textbox(
                    label="(Local Path) 로컬 폴더 경로",
                    placeholder="./models/my-llama"
                )
        api_key_text = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-...",
            visible=False  # 기본적으로 숨김
        )
        def toggle_api_key_display(selected_model):
            return gr.update(visible="gpt" in selected_model)
    
        model_dropdown.change(
            fn=toggle_api_key_display,
            inputs=[model_dropdown],
            outputs=[api_key_text]
        )
        
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

        # -----------------------------------------
        # (C) Chatbot 섹션
        # -----------------------------------------
        with gr.Row():
            image_input = gr.Image(label="이미지 업로드 (선택)", type="pil")
            with gr.Column():
                chatbot = gr.Chatbot(height=400, label="Chatbot")
                msg = gr.Textbox(label="메시지 입력")
        send_btn = gr.Button("보내기")
        history_state = gr.State([])

        def user_message(user_input, history):
            if not user_input.strip():
                return gr.update(), history
            history = history + [[user_input, None]]
            return "", history

        def bot_message(history, selected_model, local_model_path, image, api_key):
            answer = generate_answer(history, selected_model, local_model_path, image, api_key)
            history[-1][1] = answer
            return history
        def toggle_image_input(selected_model):
            if "vision" in selected_model.lower() or selected_model == "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B" or selected_model == "THUDM/glm-4v-9b" or selected_model == "openbmb/MiniCPM-Llama3-V-2_5":
                return gr.update(visible=True), "이미지를 업로드해주세요."
            else:
                return gr.update(visible=False), "이미지 입력이 필요하지 않습니다."

        model_dropdown.change(
            fn=toggle_api_key_display,
            inputs=[model_dropdown],
            outputs=[api_key_text]
        ).then(
            fn=toggle_image_input,
            inputs=[model_dropdown],
            outputs=[image_input, gr.Markdown("description")]
        )
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

demo.launch(debug=True, inbrowser=True, server_port=7861)