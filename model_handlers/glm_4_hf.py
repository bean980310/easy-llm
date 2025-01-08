import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

class GLM4HfHandler:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, 
                encode_special_tokens=True
            )
            
            logger.info(f"[*] Loading model from {self.model_dir}")
            if "fp8" in self.model_dir:
                self.model = LLM(model=self.model_dir, quantization="fp8")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load GLM4 Model: {str(e)}\n\n{traceback.format_exc()}")
            raise
    def generate_answer(self, history):
        try:
            # 메시지 처리
            prompt_messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            logger.info(f"[*] Prompt messages for GLM: {prompt_messages}")
            
            # 입력 처리
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True, 
                tokenize=True, 
                return_tensors="pt",
                return_dict=True
            )['input_ids'][0]  # vLLM은 텍스트 입력을 직접 받으므로 토큰화된 input_ids 대신 텍스트 사용
            
            logger.info("[*] GLM input template applied successfully")
            
            # 모델 유형 분기
            if isinstance(self.model, LLM):
                logger.info("[*] Using vLLM for FP8 quantization")
                # vLLM을 사용하는 경우
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=128,
                    do_sample=False
                )
                response = self.model.generate(prompt, sampling_params)
                generated_text = response.generations[0].text.strip()
                logger.info(f"[*] Generated text (vLLM): {generated_text}")
            
            elif isinstance(self.model, AutoModelForCausalLM):
                logger.info("[*] Using Transformers for standard quantization")
                # Transformers를 사용하는 경우
                inputs = self.tokenizer(
                    prompt_messages,
                    add_special_tokens=True,
                    return_tensors="pt"
                ).to(self.model.device)
                input_len = inputs['input_ids'].shape[1]
                
                # 생성 설정
                generation_config = {
                    "input_ids": inputs['input_ids'],
                    "attention_mask": inputs['attention_mask'],
                    "max_new_tokens": 128,
                    "do_sample": False,
                }
                
                # 텍스트 생성
                outputs = self.model.generate(**generation_config)
                logger.info("[*] GLM model generated the response (Transformers)")
                
                # 결과 처리
                generated_text = self.tokenizer.decode(
                    outputs[0][input_len:],
                    skip_special_tokens=True
                ).strip()
                logger.info(f"[*] Generated text (Transformers): {generated_text}")
            
            else:
                raise TypeError("Unsupported model type for generation")
            
            return generated_text
            
        except Exception as e:
            error_msg = f"Error during GLM answer generation: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg