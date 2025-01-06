# model_handlers/vision_model_handler.py

import torch
import logging
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class VisionModelHandler:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            
            logger.info(f"[*] Loading processor from {self.model_dir}")
            self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
            
            logger.info(f"[*] Loading model from {self.model_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load Vision Model: {str(e)}")
            raise

    def generate_answer(self, history, image_input=None):
        try:
            prompt_messages = []
            for msg in history:
                if msg['role'] == 'user':
                    if image_input:
                        prompt_messages.append({
                            "role": "user", 
                            "content": "Please see the attached image."
                        })
                        prompt_messages.append({
                            "role": "user",
                            "content": msg['content']
                        })
                    else:
                        prompt_messages.append({"role": "user", "content": msg['content']})
                elif msg['role'] == 'assistant':
                    prompt_messages.append({"role": "assistant", "content": msg['content']})
            
            logger.info(f"[*] Prompt messages: {prompt_messages}")
            
            if image_input:
                inputs = self.processor(
                    image_input,
                    prompt_messages,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                logger.info("[*] Image input processed successfully")
            else:
                inputs = self.tokenizer(
                    [msg['content'] for msg in prompt_messages if msg['role'] in ['user', 'assistant']],
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                logger.info("[*] Text input processed successfully")
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            terminators = self.get_terminators()
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            logger.info("[*] Model generated the response")
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            logger.info(f"[*] Generated text: {generated_text}")
            
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}")
            return f"Error during answer generation: {str(e)}"

    def get_terminators(self):
        return [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]