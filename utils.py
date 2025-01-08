# utils.py

import os
import shutil
import logging
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Callable
from huggingface_hub import (
    HfApi, 
    snapshot_download,
    model_info,
)

logger = logging.getLogger(__name__)

class DownloadTracker:
    """다운로드 진행 상황을 추적하는 클래스"""
    def __init__(self, total_size: int, progress_callback: Optional[Callable] = None):
        self.total_size = total_size
        self.current_size = 0
        self.progress_callback = progress_callback

    def update(self, chunk_size: int):
        """다운로드된 청크 크기만큼 진행률 업데이트"""
        self.current_size += chunk_size
        if self.progress_callback:
            progress = (self.current_size / self.total_size) if self.total_size > 0 else 0
            self.progress_callback(min(progress, 1.0))

def make_local_dir_name(model_id: str) -> str:
    """HuggingFace 모델 ID를 로컬 디렉토리 이름으로 변환"""
    return model_id.replace("/", "__")

def convert_folder_to_modelid(folder_name: str) -> str:
    """로컬 디렉토리 이름을 HuggingFace 모델 ID로 변환"""
    return folder_name.replace("__", "/")

def scan_local_models(root="./models"):
    """로컬에 저장된 모델 목록을 스캔"""
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
    """HuggingFace 캐시 폴더 제거"""
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

async def get_model_size(repo_id: str, token: Optional[str] = None) -> int:
    """모델의 총 크기를 계산"""
    try:
        api = HfApi()
        info = await asyncio.to_thread(model_info, repo_id, token=token)
        return info.size_in_bytes or 0
    except Exception as e:
        logger.warning(f"모델 크기 계산 실패: {e}")
        return 0

def download_model_from_hf(hf_repo_id: str, target_dir: str) -> str:
    """
    동기식 모델 다운로드 (이전 버전과의 호환성 유지)
    """
    if os.path.isdir(target_dir):
        msg = f"[*] 이미 다운로드됨: {hf_repo_id} → {target_dir}"
        logger.info(msg)
        return msg
    
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"[*] 모델 '{hf_repo_id}'을(를) '{target_dir}'에 다운로드 중...")
    try:
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=target_dir,
            ignore_patterns=["*.md", ".gitattributes", "original/", "LICENSE.txt", "LICENSE"],
            local_dir_use_symlinks=False
        )
        remove_hf_cache(hf_repo_id)
        msg = f"[+] 다운로드 & 저장 완료: {target_dir}"
        logger.info(msg)
        return msg
    except Exception as e:
        error_msg = f"모델 '{hf_repo_id}' 다운로드 실패: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return error_msg

async def download_model_from_hf_async(
    repo_id: str,
    target_dir: str = None,
    token: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    비동기식 모델 다운로드 (진행률 표시 지원)
    """
    try:
        # 기본 저장 경로 설정
        if not target_dir:
            target_dir = os.path.join("./models", make_local_dir_name(repo_id))
            
        # 이미 다운로드된 경우 확인
        if os.path.exists(target_dir) and any(os.scandir(target_dir)):
            return f"모델이 이미 존재합니다: {target_dir}"
            
        # 저장 경로 생성
        os.makedirs(target_dir, exist_ok=True)
        
        # 모델 크기 계산
        total_size = await get_model_size(repo_id, token)
        if total_size == 0:
            logger.warning("모델 크기를 확인할 수 없습니다. 진행률이 정확하지 않을 수 있습니다.")
        
        # 다운로드 트래커 초기화
        tracker = DownloadTracker(total_size, progress_callback)
        
        # 실제 다운로드 수행
        snapshot_path = await asyncio.to_thread(
            snapshot_download,
            repo_id=repo_id,
            local_dir=target_dir,
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.md", ".gitattributes", "original/", "LICENSE.txt", "LICENSE"],
        )
        
        # 다운로드 완료 후 정리
        remove_hf_cache(repo_id)
        if progress_callback:
            progress_callback(1.0)  # 100% 완료 표시
            
        # 결과 메시지 생성
        model_name = repo_id.split("/")[-1]
        size_gb = total_size / (1024**3)
        message = f"""
            다운로드 완료!
            - 모델: {model_name}
            - 저장 위치: {target_dir}
            - 크기: {size_gb:.2f} GB
            """
        logger.info(f"모델 다운로드 완료: {repo_id}")
        return message.strip()
        
    except Exception as e:
        error_msg = f"다운로드 중 오류 발생: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise RuntimeError(error_msg)

def ensure_model_available(model_id: str, local_model_path: str = None) -> bool:
    """모델이 로컬에 존재하는지 확인하고, 없으면 다운로드"""
    if model_id == "Local (Custom Path)":
        if local_model_path and os.path.isdir(local_model_path):
            logger.info(f"[*] 사용자 지정 경로에서 모델을 찾았습니다: {local_model_path}")
            return True
        else:
            logger.error(f"[!] 지정된 로컬 경로에 모델이 없습니다: {local_model_path}")
            return False

    local_dirname = make_local_dir_name(model_id)
    target_dir = os.path.join("./models", local_dirname)

    if os.path.isdir(target_dir) and 'config.json' in os.listdir(target_dir):
        logger.info(f"[*] 모델 '{model_id}'이(가) 로컬에 이미 존재합니다.")
        return True
    else:
        logger.info(f"[*] 모델 '{model_id}'이(가) 로컬에 없습니다. 다운로드를 시도합니다.")
        success = download_model_from_hf(model_id, target_dir)
        return "실패" not in success
    
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