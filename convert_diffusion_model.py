import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm

input_dir="/tmp/ComfyUI/models/checkpoints"

safetensors_files=list(Path(input_dir).rglob("*.safetensors"))

def convert_model_to_float8(file_path: str):
    print(f"[*] Converting {file_path}")
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata=f.metadata()
            metadata = metadata if metadata is not None else {}
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return False
    
    try:
        sd_pruned={}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in tqdm(f.keys(), desc="Converting tensor"):
                tensor=f.get_tensor(k)
                sd_pruned[k]=tensor.to(torch.float8_e4m3fn)
                
        output_dir=os.path.dirname(file_path)
        model_name=os.path.splitext(os.path.basename(file_path))[0]
        output_filename=f"{model_name}.fp8.safetensors"
        output_path=os.path.join(output_dir, output_filename)
        save_file(sd_pruned, output_path, metadata={"format":"pt",**metadata})
        print(f"[*] Converted {file_path} to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        
def convert_model_to_int8(file_path: str):
    print(f"[*] Converting {file_path}")
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata=f.metadata()
            metadata = metadata if metadata is not None else {}
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return False
    
    try:
        sd_pruned={}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in tqdm(f.keys(), desc="Converting tensor"):
                tensor=f.get_tensor(k)
                sd_pruned[k]=tensor.to(torch.int8)
                
        output_dir=os.path.dirname(file_path)
        model_name=os.path.splitext(os.path.basename(file_path))[0]
        output_filename=f"{model_name}.int8.safetensors"
        output_path=os.path.join(output_dir, output_filename)
        save_file(sd_pruned, output_path, metadata={"format":"pt",**metadata})
        print(f"[*] Converted {file_path} to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        
def convert_model_to_qint8(file_path: str):
    print(f"[*] Converting {file_path}")
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata=f.metadata()
            metadata = metadata if metadata is not None else {}
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return False
    
    try:
        sd_pruned={}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in tqdm(f.keys(), desc="Converting tensor"):
                tensor=f.get_tensor(k)
                sd_pruned[k]=tensor.to(torch.qint8)
                
        output_dir=os.path.dirname(file_path)
        model_name=os.path.splitext(os.path.basename(file_path))[0]
        output_filename=f"{model_name}.qint8.safetensors"
        output_path=os.path.join(output_dir, output_filename)
        save_file(sd_pruned, output_path, metadata={"format":"pt",**metadata})
        print(f"[*] Converted {file_path} to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error converting {file_path}: {e}")