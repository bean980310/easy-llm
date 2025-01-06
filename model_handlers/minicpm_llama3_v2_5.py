# model_handlers/minicpm_llama3_v2_5.py

import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class MiniCPMLlama3V25Handler:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            
            logger.info(f"[*] Loading model from {self.model_dir}")
            self.model = AutoModel.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load MiniCPM-Llama3-V-2_5 model: {str(e)}")
            raise

    def generate_answer(self, history):
        try:
            # 메시지 히스토리를 하나의 프롬프트로 결합
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            inputs = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)
            
            logger.info("[*] Generating response...")
            outputs = self.model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            logger.info(f"[*] Generated text: {generated_text}")
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            return f"Error during answer generation: {str(e)}"