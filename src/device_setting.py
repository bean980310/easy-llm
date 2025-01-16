import torch
import platform
import logging
import gradio as gr

from models import get_default_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_device(selection):
    """
    Sets the device based on user selection.
    - Auto: Automatically detect the best device.
    - CPU: Force CPU usage.
    - GPU: Detect and use CUDA or MPS based on available hardware.
    """
    if selection == "Auto (Recommended)":
        device = get_default_device()
    elif selection == "CPU":
        device = "cpu"
    elif selection == "GPU":
        if torch.cuda.is_available():
            device = "cuda"
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = "mps"
        else:
            return gr.update(value="❌ GPU가 감지되지 않았습니다. CPU로 전환됩니다."), "cpu"
    else:
        device = "cpu"
                
    device_info_message = f"선택된 장치: {device.upper()}"
    logger.info(device_info_message)
    return gr.update(value=device_info_message), device