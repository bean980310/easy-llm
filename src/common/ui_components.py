import gradio as gr
from src.common.translations import _, translation_manager, detect_system_language
from src.common.state import shared_state
from src.common.local_models import transformers_local, gguf_local, mlx_local
from src.common.api_models import api_models

default_language = detect_system_language()

initial_choices = api_models + transformers_local + gguf_local + mlx_local
initial_choices = list(dict.fromkeys(initial_choices))
initial_choices = sorted(initial_choices)  # 정렬 추가
    
def create_shared_model_dropdown():
    model_dropdown = gr.Dropdown(
        label=_("model_select_label"),
        choices=initial_choices,
        value=initial_choices[0] if len(initial_choices) > 0 else None,
        elem_classes="model-dropdown"
    )

    shared_state.model_dropdown = model_dropdown

    return model_dropdown

def create_shared_lauguage_dropdown():
    language_dropdown = gr.Dropdown(
        label=_('language_select'),
        choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
        value=translation_manager.get_language_display_name(default_language),
        interactive=True,
        info=_('language_info'),
        container=False,
        elem_classes="custom-dropdown"
    )
    
    shared_state.language_dropdown = language_dropdown
    
    return language_dropdown