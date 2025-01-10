import logging
import traceback
import os
from utils import make_local_dir_name

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

logger = logging.getLogger(__name__)

class MlxVisionHandler:
    def __init__(self, model_id, local_model_path=None, model_type="mlx"):
        self.model_dir = local_model_path or os.path.join("./models", model_type, make_local_dir_name(model_id))
        self.processor = None
        self.config = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        self.model, self.processor = load(self.model_dir)
        self.config = load_config(self.model_dir)
        
    def generate_answer(self, history, *image_inputs):
        prompt=self.history_to_prompt(history)
        
        if image_inputs:
            images=image_inputs
            formatted_prompt=apply_chat_template(
                self.processor, self.config, prompt, num_images=len(images)
            )
            
            output = generate(self.model, self.processor, formatted_prompt, images, verbose=False)
            return output
        else:
            images=None
            formatted_prompt=apply_chat_template(
                self.processor, self.config, prompt, num_images=len(images)
            )
            output = generate(self.model, self.processor, formatted_prompt, images=None, verbose=False)
            return output
    
    def history_to_prompt(self, history):
        """
        대화 히스토리를 프롬프트로 변환
        """
        prompt = ""
        for message in history:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt