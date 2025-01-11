# models.py

import random
import numpy as np
import torch
import logging
import traceback
import openai

from cache import models_cache
from model_handlers import (
    GGUFModelHandler, MlxModelHandler, TransformersModelHandler
)
from utils import ensure_model_available, build_model_cache_key, get_all_local_models

logger = logging.getLogger(__name__)

LOCAL_MODELS_ROOT = "./models"

def get_default_device() -> str:
    """
    Automatically selects the best available device:
    - CUDA if NVIDIA GPU is available.
    - MPS if Apple Silicon (M-Series) is available.
    - CPU otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Set default device
default_device = get_default_device()
logger.info(f"Default device set to: {default_device}")

FIXED_MODELS = {
    "transformers": "Qwen/Qwen2.5-7B-Instruct",  # 예: "gpt-2"
    "gguf": "Qwen/Qwen2.5-7B-Instruct-GGUF",                  # 예: "gguf-model"
    "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit"                     # 예: "mlx-model"
}

def get_fixed_model_id(model_type: str) -> str:
    """
    Returns the fixed model ID based on the model type.
    """
    return FIXED_MODELS.get(model_type, "mlx-community/Qwen2.5-7B-Instruct-4bit")  # 기본값 설정

def load_model(selected_model: str, model_type: str, quantization_bit: str = "Q8_0", local_model_path: str = None, device: str = "cpu") -> object:
    """
    모델 로드 함수. 모델 유형에 따라 적절한 핸들러를 사용하여 모델을 로드하고 캐시에 저장.
    모델 유형은 transformers, gguf, mlx 중 선택 가능.
    """
    supported_model_types = ["transformers", "gguf", "mlx"]
    if model_type not in supported_model_types:
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None

    model_id = FIXED_MODELS.get(model_type)
    if not model_id:
        logger.error(f"모델 유형 '{model_type}'에 대한 고정된 모델 ID가 설정되지 않았습니다.")
        return None

    handler = None

    # Check if handler already in cache
    cache_key = build_model_cache_key(model_id, model_type, quantization_bit, local_model_path)
    if cache_key in models_cache:
        logger.info(f"캐시에서 모델 '{model_id}'을(를) 로드합니다.")
        return models_cache[cache_key]

    # Ensure model is available (download if not)
    if not ensure_model_available(model_id, local_model_path, model_type):
        logger.error(f"모델 '{model_id}'을(를) 다운로드할 수 없습니다.")
        return None

    # Initialize appropriate handler based on model type
    if model_type == "gguf":
        handler = GGUFModelHandler(
            model_id=model_id,
            quantization_bit=quantization_bit,
            local_model_path=local_model_path,
            model_type=model_type
        )
    elif model_type == "mlx":
        handler = MlxModelHandler(
            model_id=model_id,
            local_model_path=local_model_path,
            model_type=model_type
        )
    elif model_type == "transformers":
        handler = TransformersModelHandler(
            model_id=model_id,
            local_model_path=local_model_path,
            model_type=model_type,
            device=device
        )
    else:
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None

    # Cache the handler
    models_cache[cache_key] = handler
    logger.info(f"모델 '{model_id}'을(를) 로드하고 캐시에 저장했습니다.")
    return handler

def generate_answer(
    history: list,
    selected_model: str,
    model_type: str,
    local_model_path: str = None,
    image_input: str = None,
    api_key: str = None,
    device: str = "cpu",
    seed: int = 42
) -> str:
    """
    사용자 히스토리를 기반으로 답변 생성.
    모델 유형에 따라 적절한 핸들러를 사용하여 답변을 생성.
    """
    # Seed 설정
    set_seed(seed)

    if not history:
        system_message = {
            "role": "system",
            "content": "당신은 고정된 시스템 메시지를 가진 유용한 AI 비서입니다."
        }
        history = [system_message]

    handler = load_model(selected_model, model_type, local_model_path=local_model_path, device=device)

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
            logger.error("모델 핸들러가 로드되지 않았습니다.")
            return "모델 핸들러가 로드되지 않았습니다."

        logger.info(f"[*] Generating answer using {handler.__class__.__name__}")
        try:
            answer = handler.generate_answer(history)
            return answer
        except Exception as e:
            logger.error(f"모델 추론 오류: {str(e)}\n\n{traceback.format_exc()}")
            return f"오류 발생: {str(e)}\n\n{traceback.format_exc()}"

def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)