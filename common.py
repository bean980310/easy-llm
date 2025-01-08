# common.py 상단에 추가
import os
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from utils import make_local_dir_name, ensure_model_available, get_terminators
from cache import models_cache # app.py에서 정의된 models_cache를 참조

logger = logging.getLogger(__name__)

# 메모리 캐시를 관리할 수 있도록 import
models_cache = {}

def load_default_model(model_id: str, local_model_path: str = None):
    """
    기본 모델을 로드하는 함수.
    """
    logger.info(f"[*] Loading model: {model_id}")
    local_dirname = make_local_dir_name(model_id)
    local_dirpath = os.path.join("./models", local_dirname)

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
        logger.error(f"모델 로드 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
        return None

def generate_default_answer(history, selected_model, local_model_path=None):
    """
    기본 모델을 사용하여 답변을 생성하는 함수.
    """
    handler = load_default_model(selected_model, local_model_path=local_model_path)
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
        logger.error(f"입력 템플릿 적용 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
        return f"입력 템플릿 적용 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

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
        logger.error(f"모델 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
        return f"모델 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

    try:
        generated_text = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        logger.info(f"[*] 생성된 텍스트: {generated_text}")
    except Exception as e:
        logger.error(f"출력 디코딩 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
        return f"출력 디코딩 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"

    return generated_text.strip()