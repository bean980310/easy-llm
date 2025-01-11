import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
# We support int4_weight_only, int8_weight_only and int8_dynamic_activation_int8_weight
# More examples and documentations for arguments can be found in https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques
quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# compile the quantized model to get speedup
import torchao
torchao.quantization.utils.recommended_inductor_config_setter()
quantized_model = torch.compile(quantized_model, mode="max-autotune")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# benchmark the performance
import torch.utils.benchmark as benchmark

def benchmark_fn(f, *args, **kwargs):
    # Manual warmup
    for _ in range(5):
        f(*args, **kwargs)
        
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"

MAX_NEW_TOKENS = 1000
print("int4wo-128 model:", benchmark_fn(quantized_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS))

bf16_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
bf16_model = torch.compile(bf16_model, mode="max-autotune")
print("bf16 model:", benchmark_fn(bf16_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS))
