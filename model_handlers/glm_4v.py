# model_handlers/glm4v_handler.py

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class GLM4VHandler:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True, encode_special_tokens=True)
            
            logger.info(f"[*] Loading model from {self.model_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load GLM4V Model: {str(e)}")
            raise

    def generate_answer(self, history):
        try:
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Prompt messages for GLM: {prompt_messages}")
            
            inputs = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True, 
                tokenize=True, 
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            logger.info("[*] GLM input template applied successfully")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.8,
                repetition_penalty=1.2,
                eos_token_id=[151329, 151336, 151338],
                stopping_criteria=self.get_stopping_criteria()
            )
            logger.info("[*] GLM model generated the response")
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            logger.info(f"[*] Generated text: {generated_text}")
            
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during GLM answer generation: {str(e)}")
            return f"Error during GLM answer generation: {str(e)}"

    def get_stopping_criteria(self):
        return [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]