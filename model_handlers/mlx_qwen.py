import logging
import traceback
import os
from utils import make_local_dir_name

from mlx_lm import load, generate

logger = logging.getLogger(__name__)

class MlxQwenHandler:
    def __init__(self, model_id, local_model_path=None, model_type="mlx"):
        self.model_dir = local_model_path or os.path.join("./models", model_type, make_local_dir_name(model_id))
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        self.model, self.tokenizer = load(self.model_dir, tokenizer_config={"eos_token": "<|im_end|>"})
    
    def generate_answer(self, history):
         # prompt = self.history_to_prompt(history)  # << 이 부분은 사용하지 않음
        text = self.tokenizer.apply_chat_template(
            conversation=history,      # <-- 문자열 대신 딕셔너리 리스트를 직접 전달
            tokenize=False,
            add_generation_prompt=True
        )
        response = generate(self.model, self.tokenizer, prompt=text, verbose=True, top_p=0.8, temp=0.7, repetition_penalty=1.05, max_tokens=512)
        
        return response