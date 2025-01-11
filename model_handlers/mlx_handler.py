# mlx_handler.py

import logging
import traceback
import os
from utils import make_local_dir_name
from mlx_lm import load, generate

logger = logging.getLogger(__name__)

class MlxModelHandler:
    def __init__(self, model_id, local_model_path=None, model_type="mlx"):
        self.model_dir = local_model_path or os.path.join("./models", model_type, make_local_dir_name(model_id))
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            logger.info(f"모델 로드 시작: {self.model_dir}")
            self.model, self.tokenizer = load(self.model_dir, tokenizer_config={"eos_token": "<|im_end|>"})
            logger.info("모델 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def generate_answer(self, history):
        """
        히스토리를 기반으로 응답 생성
        """
        try:
            # 히스토리 길이 제한 (최근 대화만 유지)
            max_history_items = 10  # 최근 10개의 메시지만 유지
            recent_history = history[-max_history_items:] if len(history) > max_history_items else history
            
            # 대화 템플릿 적용
            text = self.tokenizer.apply_chat_template(
                conversation=recent_history,
                tokenize=False,
                add_generation_prompt=True
            )
            # logger.debug(f"생성된 프롬프트: {text[:200]}...")  # 로깅 추가
            
            token_count = len(self.tokenizer.encode(text))
            max_sequence_length = 131072  # 모델의 최대 시퀀스 길이
            
            if token_count > max_sequence_length:
                # 필요시 히스토리 항목 수를 줄이거나 요약 적용
                recent_history = history[-5:]  # 예: 최근 5개로 줄이기
                text = self.tokenizer.apply_chat_template(
                    conversation=recent_history,
                    tokenize=False,
                    add_generation_prompt=True
                )
                token_count = len(self.tokenizer.encode(text))
            # 기본 생성 매개변수만 사용
            response = generate(
                self.model,
                self.tokenizer,
                prompt=text,
                verbose=True,
                max_tokens=512,
                temp=0.7,
                top_p=0.8,
            )
            
            logger.debug(f"생성된 응답: {response[:200]}...")  # 로깅 추가
            return response.strip()
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}\n{traceback.format_exc()}")
            raise Exception(f"응답 생성 실패: {str(e)}")