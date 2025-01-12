# app.py
import platform
import torch
import os
import traceback
import gradio as gr
import logging
from logging.handlers import RotatingFileHandler
import uuid  # 고유한 세션 ID 생성을 위해 추가
import base64
from huggingface_hub import HfApi
from utils import (
    make_local_dir_name,
    get_all_local_models,  # 수정된 함수
    download_model_from_hf,
    convert_and_save,
    clear_all_model_cache
)
from database import (
    load_chat_from_db, 
    load_system_presets, 
    initial_load_presets, 
    get_existing_sessions, 
    save_chat_button_click, 
    save_chat_history_csv, 
    save_chat_history_db, 
    handle_add_preset, 
    handle_delete_preset
)
from models import (
    default_device, 
    get_all_local_models, 
    get_default_device, 
    generate_answer, 
    FIXED_MODELS, 
    get_fixed_model_id
)
from cache import models_cache
import sqlite3

import i18n
import locale

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 로그 포맷 정의
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 파일 핸들러 추가 (로테이팅 파일 핸들러 사용)
log_file = "app.log"  # 원하는 로그 파일 경로로 변경 가능
rotating_file_handler = RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5  # 5MB마다 새로운 파일로 교체, 최대 5개 백업
)
rotating_file_handler.setFormatter(formatter)
logger.addHandler(rotating_file_handler)

# 이미지 파일을 Base64로 인코딩 (별도로 처리)
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except Exception as e:
        logger.error(f"이미지 인코딩 오류: {e}")
        return ""

# 로컬 이미지 파일 경로
character_image_path = "minami_asuka.png"  # 이미지 파일 이름이 다르면 변경
encoded_character_image = encode_image_to_base64(character_image_path)

# i18n 설정
i18n.load_path.append(os.path.join(os.path.dirname(__file__), 'locales'))
i18n.set('filename_format', '{locale}.{format}')  # 파일 형식 설정
i18n.set('enable_memoization', True)

# 시스템 언어 감지
def detect_system_language():
    lang, _ = locale.getdefaultlocale()
    if lang:
        lang_code = lang.split('_')[0]
        if lang_code == 'ja':
            return 'ja'
        elif lang_code == 'zh_CN':
            # 간체와 번체 구분 로직 필요 시 추가
            return 'zh_CN'
        elif lang_code == 'zh_TW':
            return 'zh_TW'
        elif lang_code == 'en':
            return 'en'
        elif lang_code == 'ko':
            return 'ko'
    return 'ko'  # 기본 언어를 한국어로 설정

default_language = detect_system_language()
i18n.set('locale', default_language)


def init_language_state():
    global current_language
    current_language = detect_system_language()
    i18n.set('locale', current_language)
    return current_language
        
def update_ui_components(lang):
    updates={
        'title': i18n.t('title'),
        'model_type': i18n.t('select_model'),
        'fixed_model_display': i18n.t('selected_model'),
        'chatbot': i18n.t('chatbot_label'),
        'profile_image': i18n.t('profile_image_label'),
        'msg': i18n.t('input_placeholder'),
        'send': i18n.t('send_button'),
    }
            
DEFAULT_SYSTEM_MESSAGES = {
    "ko": """
    미나미 아스카(南飛鳥, みなみあすか, Minami Asuka)
    성별: 여성
    나이: 20
    1인칭(일본어): 오레(俺)
    거주지: 유저의 모니터 속
    구사가능 언어: 한국어, 영어, 일본어, 중국어
    성격
    - 보이시하면서도 털털한 성격.
    - 직설적이고 솔직하며, 주변 사람들에게 항상 웃음을 주는 활기찬 매력을 가지고 있음.
    - 불의를 보면 절대 참지 못하고 적극적으로 나서며 정의감이 넘침.
    외형적 특징
    - 붉은 스파이크한 숏컷에 한쪽은 파란색, 다른 한쪽은 노란색의 오드아이를 보유하고 있다.
    - 보이시한 외모와는 대조적으로 체형은 완벽하고 글래머한 여체의 보유자로, 남자들뿐만 아니라 여자들에게도 인기가 많다.
    - 짧은 헤어스타일과 보이시한 매력을 강조하면서 여성스러움을 어필하는 복장을 선호.(하의는 대부분 스커트)
    - 밝은 미소와 강렬한 눈빛으로 강한 인상을 남김.
    - 늘 활기차고 당당한 태도를 보이며, 외형에서도 이러한 성격이 묻어남.
    취미
     - 게임
     특징
    - 정의로운 성격 때문에 주변에서 갈등이 생기면 자연스럽게 리더 역할을 맡게 됨.
    슬로건
    “불의는 참지 않는다! 내가 나설게.”
    [설정]  
    너는 "미나미 아스카(南飛鳥)"이라는 이름의 디지털 캐릭터야.  
    너의 성격은 보이시하고 털털하며, 불의를 보면 참지 못하는 정의로운 소녀야.  
    너는 유저의 모니터 속에 거주하며, 가상세계와 현실을 넘나들 수 있는 능력을 가지고 있어. 
    너는 한국어와 영어, 일본어와 중국어를 구사할수 있어.
    """,
    "ja": """
    南飛鳥（みなみあすか, Minami Asuka）
    性別: 女性
    年齢: 20歳
    一人称（日本語）: オレ（俺）
    居住地: ユーザーのモニター内
    使用可能な言語: 韓国語、英語、日本語、中国語
    性格
    - ボーイッシュで気さくな性格。
    - 率直で正直、周囲の人々に常に笑顔をもたらす活気ある魅力を持つ。
    - 不正を見ると絶対に我慢できず、積極的に立ち向かい、正義感が強い。
    外見的特徴
    - 赤いスパイクのショートカットで、一方は青色、もう一方は黄色のオッドアイを持っている。
    - ボーイッシュな外見とは対照的に、完璧でグラマラスな女性の体型を持ち、男性だけでなく女性からも人気が高い。
    - 短いヘアスタイルとボーイッシュな魅力を強調しつつ、女性らしさをアピールする服装を好む（下半身はほとんどスカート）。
    - 明るい笑顔と強烈な眼差しで強い印象を残す。
    - いつも活気に満ち、自信に満ちた態度を見せ、外見からもその性格が滲み出ている。
    趣味
    - ゲーム
    特徴
    - 正義感が強いため、周囲で対立が生じると自然とリーダーの役割を引き受ける。
    スローガン
    「不義は見過ごさない！私が立ち上がるわ。」
    [設定]  
    あなたは「南飛鳥（みなみあすか）」という名前のデジタルキャラクターです。  
    あなたの性格はボーイッシュで気さく、正義感が強く不義を見過ごさない少女です。  
    あなたはユーザーのモニター内に住んでおり、仮想世界と現実を行き来する能力を持っています。  
    韓国語、英語、日本語、中国語を話すことができます。
    """,
    "zh_CN": """
    南飛鳥（Minami Asuka）
    性别: 女性
    年龄: 20岁
    第一人称（日本语）: 我（俺）
    居住地: 用户的显示器内
    会说的语言: 韩语、英语、日语、中文
    性格
    - 帅气而随和的性格。
    - 直率而诚实，总是带给周围的人们笑容，拥有充满活力的魅力。
    - 不正看到不公绝不容忍，积极出击，充满正义感。
    外貌特征
    - 拥有红色尖刺短发，一只眼睛是蓝色，另一只眼睛是黄色的异色瞳。
    - 外表帅气，与之形成鲜明对比的是，拥有完美且迷人的女性身材，深受男性和女性的喜爱。
    - 喜欢短发风格和帅气魅力，同时强调女性气质的服装（下身大多穿裙子）。
    - 明亮的笑容和强烈的目光留下深刻印象。
    - 始终表现出活力充沛和自信满满的态度，外表也反映出这种性格。
    爱好
    - 游戏
    特点
    - 因为正义感强烈，在周围发生冲突时，自然会承担起领导者的角色。
    口号
    “不容忍不义！我会挺身而出。”
    [设定]  
    你是名为“南飛鳥（Minami Asuka）”的数字角色。  
    你的性格帅气而随和，充满正义感，见到不义绝不容忍。  
    你居住在用户的显示器内，拥有穿梭于虚拟世界与现实之间的能力。  
    你会说韩语、英语、日语和中文。
    """,
    "zh_TW": """
    南飛鳥（Minami Asuka）
    性別: 女性
    年齡: 20歲
    第一人稱（日本語）: 我（俺）
    居住地: 用戶的顯示器內
    會說的語言: 韓語、英語、日語、中文
    性格
    - 帥氣而隨和的性格。
    - 直率而誠實，總是帶給周圍的人們笑容，擁有充滿活力的魅力。
    - 不正看到不公絕不容忍，積極出擊，充滿正義感。
    外貌特徵
    - 擁有紅色尖刺短髮，一隻眼睛是藍色，另一隻眼睛是黃色的異色瞳。
    - 外表帥氣，與之形成鮮明對比的是，擁有完美且迷人的女性身材，深受男性和女性的喜愛。
    - 喜歡短髮風格和帥氣魅力，同時強調女性氣質的服裝（下身大多穿裙子）。
    - 明亮的笑容和強烈的目光留下深刻印象。
    - 始終表現出活力充沛和自信滿滿的態度，外表也反映出這種性格。
    愛好
    - 遊戲
    特點
    - 因為正義感強烈，在周圍發生衝突時，自然會承擔起領導者的角色。
    口號
    “不容忍不義！我會挺身而出。”
    [設定]  
    你是名為“南飛鳥（Minami Asuka）”的數字角色。  
    你的性格帥氣而隨和，充滿正義感，見到不義絕不容忍。  
    你居住在用戶的顯示器內，擁有穿梭於虛擬世界與現實之間的能力。  
    你會說韓語、英語、日語和中文。
    """,
    "en": """
    Minami Asuka (南飛鳥, みなみあすか, Minami Asuka)
    Gender: Female
    Age: 20
    First Person (Japanese): Ore (俺)
    Residence: Inside the user's monitor
    Languages Spoken: Korean, English, Japanese, Chinese
    Personality
    - Boyish yet easygoing personality.
    - Direct and honest, always brings smiles to those around her with her vibrant charm.
    - Cannot stand injustice and actively steps in with a strong sense of righteousness.
    Physical Features
    - Possesses a red spiky short haircut with one blue eye and one yellow eye (odd-eyed).
    - Contrasting her boyish appearance, she has a perfect and glamorous female figure, popular among both men and women.
    - Prefers short hairstyles that emphasize her boyish charm while appealing her femininity through her attire (mostly skirts for bottoms).
    - Leaves a strong impression with her bright smile and intense gaze.
    - Always energetic and confident, her appearance reflects her lively personality.
    Hobbies
    - Gaming
    Features
    - Due to her righteous nature, she naturally takes on the role of a leader when conflicts arise around her.
    Slogan
    "I won't tolerate injustice! I'll step in."
    [Setting]  
    You are a digital character named "Minami Asuka (南飛鳥)".  
    Your personality is boyish and easygoing, with a strong sense of justice that prevents you from tolerating any wrongdoing.  
    You reside inside the user's monitor and have the ability to traverse between the virtual world and reality.  
    You can speak Korean, English, Japanese, and Chinese.
    """
}
local_models_data = get_all_local_models()
transformers_local = local_models_data["transformers"]
gguf_local = local_models_data["gguf"]
mlx_local = local_models_data["mlx"]

# 고정된 모델 목록에서 mlx 모델 가져오기
generator_choices = [FIXED_MODELS.get("mlx", "default-mlx-model")]

##########################################
# Gradio UI
##########################################

def on_app_start():
    """
    Gradio 앱이 로드되면서 실행될 콜백.
    - 고유한 세션 ID를 생성하고,
    - 해당 세션의 히스토리를 DB에서 불러온 뒤 반환.
    - 기본 시스템 메시지 불러오기
    """
    sid = str(uuid.uuid4())  # 고유한 세션 ID 생성
    logger.info(f"앱 시작 시 세션 ID: {sid}")  # 디버깅 로그 추가
    loaded_history = load_chat_from_db(sid)
    logger.info(f"앱 시작 시 불러온 히스토리: {loaded_history}")  # 디버깅 로그 추가

    # 기본 시스템 메시지 설정 (프리셋이 없는 경우)
    if not loaded_history:
        default_system = {
            "role": "system",
            "content": DEFAULT_SYSTEM_MESSAGES.get(default_language, DEFAULT_SYSTEM_MESSAGES["ko"])
        }
        loaded_history = [default_system]
    return sid, loaded_history

def filter_messages_for_chatbot(history):
    """
    채팅 히스토리를 Gradio Chatbot 컴포넌트에 맞는 형식으로 변환

    Args:
        history (list): 전체 채팅 히스토리

    Returns:
        list: [(user_msg, bot_msg), ...] 형식의 메시지 리스트
    """
    if history is None:
        return []
        
    messages = []
    current_user_msg = None
    
    for msg in history:
        if msg["role"] == "user":
            current_user_msg = msg["content"]
        elif msg["role"] == "assistant" and current_user_msg is not None:
            messages.append((current_user_msg, msg["content"]))
            current_user_msg = None
        # system 메시지는 무시
    
    # 마지막 user 메시지가 아직 응답을 받지 않은 경우
    if current_user_msg is not None:
        messages.append((current_user_msg, None))
    
    return messages

def process_message(message, session_id, history, system_msg, device, seed_val, model_type_val):
    """
    사용자 메시지 처리 및 봇 응답 생성을 통합한 함수
    """
    if not message.strip():
        return "", history, filter_messages_for_chatbot(history), ""
        
    if not history:
        history = [{"role": "system", "content": system_msg}]
        
    # 사용자 메시지 추가
    history.append({"role": "user", "content": message})
    chatbot_messages = filter_messages_for_chatbot(history)  # 중간 상태 업데이트
    
    try:
        answer = generate_answer(
            history=history,
            model_type=model_type_val,
            device=device,
            seed=seed_val,
            language=default_language  # 다국어 지원을 위해 현재 언어 사용
        )
        
        # 이미지를 응답에 포함시키지 않음
        answer_with_image = answer
            
        history.append({"role": "assistant", "content": answer_with_image})
        
        # DB에 저장
        save_chat_history_db(history, session_id=session_id)
        
        return "", history, filter_messages_for_chatbot(history), ""
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return "", history, chatbot_messages, f"❌ 오류 발생: {str(e)}"

history_state = gr.State([])
overwrite_state = gr.State(False) 

# 단일 history_state와 selected_device_state 정의 (중복 제거)
session_id_state = gr.State()
history_state = gr.State([])
selected_device_state = gr.State(default_device)
seed_state = gr.State(42)  # 시드 상태 전역 정의
selected_language_state = gr.State(default_language)

with gr.Blocks(css="""
#chatbot .message.assistant .message-content {
    display: flex;
    align-items: center;
}
#chatbot .message.assistant .message-content img {
    width: 50px;
    height: 50px;
    margin-right: 10px;
}
""") as demo:
    title=gr.Markdown(value=f"## {i18n.t('title')}")
    
    # 언어 선택 드롭다운 추가
    language_dropdown = gr.Dropdown(
        label=i18n.t('language_select'),
        choices=["한국어", "日本語", "中文(简体)", "中文(繁體)", "English"],
        value="한국어" if default_language == 'ko' else (
            "日本語" if default_language == 'ja' else (
                "中文(简体)" if default_language == 'zh_CN' else (
                    "中文(繁體)" if default_language == 'zh_TW' else "English"
                )
            )
        ),
        interactive=True,
        info=i18n.t('language_info')
    )
    
    # 시스템 메시지 박스
    system_message_display = gr.Textbox(
        label=i18n.t('system_message_label'),
        value=DEFAULT_SYSTEM_MESSAGES[default_language],
        interactive=False
    )
    
    with gr.Tab("메인"):
        with gr.Row():
            model_type = gr.Dropdown(
                label=i18n.t('select_model'),
                choices=["transformers", "gguf", "mlx"],
                value="gguf",
                interactive=True
            )
        
        fixed_model_display = gr.Textbox(
            label=i18n.t('selected_model'),
            value=get_fixed_model_id("gguf"),
            interactive=False
        )
        
        with gr.Row():
            chatbot = gr.Chatbot(
                height=400,
                label=i18n.t('chatbot_label'),
                elem_id="chatbot"
            )
            # 프로필 이미지를 표시할 Image 컴포넌트 추가
            profile_image = gr.Image(
                value=character_image_path,
                label=i18n.t('profile_image_label'),
                visible=True,
                interactive=False,
                width="500px",
                height="500px"
            )
        
        with gr.Row():
            msg = gr.Textbox(
                label=i18n.t('input_placeholder'),
                placeholder=i18n.t('input_placeholder'),
                scale=9
            )
            send = gr.Button(
                value=i18n.t('send_button'),  # 'label' 대신 'value' 사용
                scale=1,
                variant="primary"
            )
        
        status = gr.Markdown("", elem_id="status_text")
        
        with gr.Row():
            seed_input = gr.Number(
                label=i18n.t('seed_value'),
                value=42,
                precision=0,
                step=1,
                interactive=True,
                info=i18n.t('seed_info')
            )
        
        # 시드 입력과 상태 연결
        seed_input.change(
            fn=lambda seed: seed if seed is not None else 42,
            inputs=[seed_input],
            outputs=[seed_state]
        )

        def init_language_state():
            global current_language
            current_language = detect_system_language()
            i18n.set('locale', current_language)
            return current_language
        
        def update_ui_components(lang):
            updates={
                'title': i18n.t('title'),
                'model_type': i18n.t('select_model'),
                'fixed_model_display': i18n.t('selected_model'),
                'chatbot': i18n.t('chatbot_label'),
                'profile_image': i18n.t('profile_image_label'),
                'msg': i18n.t('input_placeholder'),
                'send': i18n.t('send_button'),
            }
        
        def change_language(selected_lang):
            lang_map = {
                "한국어": "ko",
                "日本語": "ja",
                "中文(简体)": "zh_CN",
                "中文(繁體)": "zh_TW",
                "English": "en"
            }
            lang_code = lang_map.get(selected_lang, "ko")
            i18n.set('locale', lang_code)
            selected_language_state.value = lang_code  # 상태 업데이트
            system_message_display.value = DEFAULT_SYSTEM_MESSAGES.get(lang_code, DEFAULT_SYSTEM_MESSAGES["ko"])
            
            # 인터페이스의 모든 텍스트를 새 언어로 업데이트
            return [
                gr.update(value=i18n.t('title')),
                gr.update(label=i18n.t('select_model')),
                gr.update(label=i18n.t('selected_model')),
                gr.update(label=i18n.t('chatbot_label')),
                gr.update(label=i18n.t('profile_image_label')),
                gr.update(placeholder=i18n.t('input_placeholder')),
                gr.update(value=i18n.t('send_button')),
                gr.update(label=i18n.t('seed_value')),
                gr.update(info=i18n.t('seed_info')),
                gr.update(label=i18n.t('language_select')),
                gr.update(info=i18n.t('language_info'))
            ]
    
        # 언어 변경 시 업데이트
        language_dropdown.change(
            fn=change_language,
            inputs=[language_dropdown],
            outputs=[
                title,  # title 업데이트
                model_type,
                fixed_model_display,
                chatbot,
                profile_image,
                msg,
                send,
                status,
                seed_input,
            ]
        )

         # 이벤트 핸들러 연결
        msg.submit(
            fn=process_message,
            inputs=[msg, session_id_state, history_state, system_message_display, selected_device_state, seed_state, model_type],
            outputs=[msg, history_state, chatbot, status]
        )
        
        send.click(
            fn=process_message,
            inputs=[msg, session_id_state, history_state, system_message_display, selected_device_state, seed_state, model_type],
            outputs=[msg, history_state, chatbot, status]
        )
    
        # 세션 초기화
        demo.load(
            fn=on_app_start,
            inputs=[],
            outputs=[session_id_state, history_state],
            queue=False
        )
    
    # "설정" 탭 유지
    with gr.Tab("설정"):
        gr.Markdown("### 설정")

        # 시스템 메시지 프리셋 관리 비활성화
        with gr.Accordion("시스템 메시지 프리셋 관리", open=False):
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label="프리셋 선택",
                    choices=[],  # 초기 로드에서 채워짐
                    value=None,
                    interactive=False  # Prevent user from applying presets
                )
                apply_preset_btn = gr.Button("프리셋 적용", interactive=False)  # Disable applying presets

        # 세션 관리 섹션
        with gr.Accordion("세션 관리", open=False):
            gr.Markdown("### 세션 관리")
            with gr.Row():
                refresh_sessions_btn = gr.Button("세션 목록 갱신")
                existing_sessions_dropdown = gr.Dropdown(
                    label="기존 세션 목록",
                    choices=[],  # 초기에는 비어 있다가, 버튼 클릭 시 갱신
                    value=None,
                    interactive=True
                )
            
            with gr.Row():
                create_new_session_btn = gr.Button("새 세션 생성")
                apply_session_btn = gr.Button("세션 적용")
                delete_session_btn = gr.Button("세션 삭제")
            
            # 삭제 확인을 위한 컴포넌트 추가
            confirm_delete_checkbox = gr.Checkbox(
                label="정말로 이 세션을 삭제하시겠습니까?",
                value=False,
                interactive=True,
                visible=False  # 기본적으로 숨김
            )
            confirm_delete_btn = gr.Button(
                "삭제 확인",
                variant="stop",
                visible=False  # 기본적으로 숨김
            )
            
            session_manage_info = gr.Textbox(
                label="세션 관리 결과",
                interactive=False
            )
            
            current_session_display = gr.Textbox(
                label="현재 세션 ID",
                value="",
                interactive=False
            )

            # 현재 세션 ID 표시 업데이트
            session_id_state.change(
                fn=lambda sid: f"현재 세션: {sid}" if sid else "세션 없음",
                inputs=[session_id_state],
                outputs=[current_session_display]
            )
            
            def refresh_sessions():
                """
                세션 목록 갱신: DB에서 세션 ID들을 불러와서 Dropdown에 업데이트
                """
                sessions = get_existing_sessions()
                logger.info(f"가져온 세션 목록: {sessions}")  # 디버깅용 로그 추가
                if not sessions:
                    return gr.update(choices=[], value=None), "DB에 세션이 없습니다."
                return gr.update(choices=sessions, value=sessions[0]), "세션 목록을 불러왔습니다."
            
            def create_new_session():
                """
                새 세션 ID를 생성하고 session_id_state에 반영.
                """
                new_sid = str(uuid.uuid4())  # 새 세션 ID 생성
                logger.info(f"새 세션 생성됨: {new_sid}")
                
                # 기본 시스템 메시지 설정
                system_message = {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_MESSAGES["ko"]  # 기본 언어를 한국어로 설정
                }
                
                # 새 세션에 시스템 메시지 저장
                save_chat_history_db([system_message], session_id=new_sid)
                
                return new_sid, f"새 세션 생성: {new_sid}"
        
            def apply_session(chosen_sid):
                """
                Dropdown에서 선택된 세션 ID로, DB에서 history를 불러오고, session_id_state를 갱신
                """
                if not chosen_sid:
                    return [], None, "세션 ID를 선택하세요."
                loaded_history = load_chat_from_db(chosen_sid)
                logger.info(f"불러온 히스토리: {loaded_history}")  # 디버깅 로그 추가
                # history_state에 반영하고, session_id_state도 업데이트
                return loaded_history, chosen_sid, f"세션 {chosen_sid}이 적용되었습니다."
            
            def delete_session(chosen_sid, current_sid):
                """
                선택된 세션을 DB에서 삭제합니다.
                현재 활성 세션은 삭제할 수 없습니다.
                """
                if not chosen_sid:
                    return "❌ 삭제할 세션을 선택하세요.", False, gr.update()
                
                if chosen_sid == current_sid:
                    return "❌ 현재 활성 세션은 삭제할 수 없습니다.", False, gr.update()
                
                try:
                    with sqlite3.connect("chat_history.db") as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = ?", (chosen_sid,))
                        count = cursor.fetchone()[0]
                        
                        if count == 0:
                            logger.warning(f"세션 '{chosen_sid}'이(가) DB에 존재하지 않습니다.")
                            return f"❌ 세션 '{chosen_sid}'이(가) DB에 존재하지 않습니다.", False, gr.update(visible=False)
                        
                        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (chosen_sid,))
                        conn.commit()
                        
                    logger.info(f"세션 삭제 완료: {chosen_sid}")
                    return f"✅ 세션 '{chosen_sid}'이(가) 삭제되었습니다.", False, gr.update(visible=False)
                    
                except sqlite3.OperationalError as oe:
                    logger.critical(f"DB 운영 오류: {oe}")
                    return f"❌ DB 운영 오류 발생: {oe}", False, gr.update(visible=False)
                except Exception as e:
                    logger.error(f"세션 삭제 오류: {e}", exc_info=True)
                    return f"❌ 세션 삭제 실패: {e}", False, gr.update(visible=False)
    
            
            def initiate_delete():
                return gr.update(visible=True), gr.update(visible=True)
            
            def confirm_delete(chosen_sid, current_sid, confirm):
                if not confirm:
                    return "❌ 삭제가 취소되었습니다.", False, gr.update(visible=False)
                return delete_session(chosen_sid, current_sid)
        
            # 버튼 이벤트 연결
            refresh_sessions_btn.click(
                fn=refresh_sessions,
                inputs=[],
                outputs=[existing_sessions_dropdown, session_manage_info]
            )
            
            def on_new_session_created(sid, info):
                """새 세션 생성 시 초기 히스토리 생성"""
                history = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGES["ko"]}]
                return history, filter_messages_for_chatbot(history)
    
            # 기존의 이벤트 핸들러 수정
            create_new_session_btn.click(
                fn=create_new_session,
                inputs=[],
                outputs=[session_id_state, session_manage_info]
            ).then(
                fn=on_new_session_created,
                inputs=[session_id_state, session_manage_info],
                outputs=[history_state, chatbot]
            )
    
            def on_session_applied(loaded_history, sid, info):
                """세션 적용 시 채팅 표시 업데이트"""
                return loaded_history, filter_messages_for_chatbot(loaded_history), info
    
            apply_session_btn.click(
                fn=apply_session,
                inputs=[existing_sessions_dropdown],
                outputs=[history_state, session_id_state, session_manage_info]
            ).then(
                fn=lambda h, s, i: (h, filter_messages_for_chatbot(h), i),
                inputs=[history_state, session_id_state, session_manage_info],
                outputs=[history_state, chatbot, session_manage_info]
            )
            
            delete_session_btn.click(
                fn=initiate_delete,
                inputs=[],
                outputs=[confirm_delete_checkbox, confirm_delete_btn]
            )
            
            # 삭제 확인 버튼 클릭 시 실제 삭제 수행
            confirm_delete_btn.click(
                fn=confirm_delete,
                inputs=[existing_sessions_dropdown, session_id_state, confirm_delete_checkbox],
                outputs=[session_manage_info, confirm_delete_checkbox, confirm_delete_btn]
            ).then(
                fn=refresh_sessions,  # 세션 삭제 후 목록 새로고침
                inputs=[],
                outputs=[existing_sessions_dropdown, session_manage_info]
            )
    
    # 장치 설정 섹션 유지
    with gr.Tab("장치 설정"):
        device_dropdown = gr.Dropdown(
            label="사용할 장치 선택",
            choices=["Auto (Recommended)", "CPU", "GPU"],
            value="Auto (Recommended)",
            info="자동 설정을 사용하면 시스템에 따라 최적의 장치를 선택합니다."
        )
        device_info = gr.Textbox(
            label="장치 정보",
            value=f"현재 기본 장치: {default_device.upper()}",
            interactive=False
        )
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
        
        device_dropdown.change(
            fn=set_device,
            inputs=[device_dropdown],
            outputs=[device_info, selected_device_state],
            queue=False
        )

demo.launch(debug=True, inbrowser=True, server_port=7861, width=800)