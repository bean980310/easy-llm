# utils.py

import os
import shutil
from pathlib import Path
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

def make_local_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")

def convert_folder_to_modelid(folder_name: str) -> str:
    return folder_name.replace("__", "/")

def scan_local_models(root="./models"):
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)

    local_model_ids = []
    for folder in os.listdir(root):
        full_path = os.path.join(root, folder)
        if os.path.isdir(full_path) and 'config.json' in os.listdir(full_path):
            model_id = convert_folder_to_modelid(folder)
            local_model_ids.append(model_id)
    logger.info(f"Scanned local models: {local_model_ids}")
    return local_model_ids

def remove_hf_cache(model_id):
    if "/" in model_id:
        user, name = model_id.split("/", maxsplit=1)
        cache_dirname = f"./models/{user}__{name}"
    else:
        cache_dirname = f"./models/{model_id}"

    cache_path = os.path.join(cache_dirname, ".cache")
    if os.path.isdir(cache_path):
        logger.info(f"[*] Removing cache folder: {cache_path}")
        shutil.rmtree(cache_path)
    else:
        logger.info(f"[*] No cache folder found: {cache_path}")

def download_model_from_hf(hf_repo_id: str, target_dir: str) -> bool:
    """
    모델을 Hugging Face Hub에서 다운로드합니다.
    성공 시 True, 실패 시 False를 반환합니다.
    """
    if os.path.isdir(target_dir):
        logger.info(f"[*] 이미 다운로드됨: {hf_repo_id} → {target_dir}")
        return True
    
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"[*] 모델 '{hf_repo_id}'을(를) '{target_dir}'에 다운로드 중...")
    try:
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=target_dir,
            ignore_patterns=["*.md", ".gitattributes", "original/", "LICENSE.txt", "LICENSE"]
        )
        remove_hf_cache(hf_repo_id)
        logger.info(f"[+] 다운로드 & 저장 완료: {target_dir}")
        return True
    except Exception as e:
        logger.error(f"모델 '{hf_repo_id}' 다운로드 실패: {str(e)}")
        return False

def ensure_model_available(model_id: str, local_model_path: str = None) -> bool:
    """
    모델이 로컬에 존재하는지 확인하고, 없으면 다운로드합니다.
    - model_id: Hugging Face 모델 ID 또는 "Local (Custom Path)"
    - local_model_path: 로컬 경로 (필요한 경우)
    - 반환값: 모델이 성공적으로 로컬에 있는 경우 True, 그렇지 않으면 False
    """
    if model_id == "Local (Custom Path)":
        if local_model_path and os.path.isdir(local_model_path):
            logger.info(f"[*] 사용자 지정 경로에서 모델을 찾았습니다: {local_model_path}")
            return True
        else:
            logger.error(f"[!] 지정된 로컬 경로에 모델이 없습니다: {local_model_path}")
            return False

    # 모델 디렉토리 이름 변환
    local_dirname = make_local_dir_name(model_id)
    target_dir = os.path.join("./models", local_dirname)

    if os.path.isdir(target_dir) and 'config.json' in os.listdir(target_dir):
        logger.info(f"[*] 모델 '{model_id}'이(가) 로컬에 이미 존재합니다.")
        return True
    else:
        logger.info(f"[*] 모델 '{model_id}'이(가) 로컬에 없습니다. 다운로드를 시도합니다.")
        success = download_model_from_hf(model_id, target_dir)
        return success