# translations.py

from typing import Dict, Any
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TranslationManager:
    
    LANGUAGE_NAMES = {
        'ko': '한국어',
        'en': 'English',
        'ja': '日本語',
        'zh_CN': '简体中文',
        'zh_TW': '繁體中文'
    }
    
    def __init__(self, default_language: str = 'ko'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.load_translations()

    def load_translations(self):
        """Load all translation files from the translations directory"""
        translations_dir = Path('translations')
        if not translations_dir.exists():
            translations_dir.mkdir(parents=True)
            self._create_default_translations()
        
        for lang_file in translations_dir.glob('*.json'):
            lang_code = lang_file.stem
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                logger.info(f"Loaded translations for {lang_code}")
            except Exception as e:
                logger.error(f"Error loading translations for {lang_code}: {e}")

    def _create_default_translations(self):
        """Create default translation files if they don't exist"""
        default_translations = {
            'ko': {
                'main_title': '간단한 Chatbot',
                'system_message': '시스템 메시지',
                'system_message_default': '당신은 유용한 AI 비서입니다.',
                'system_message_placeholder': '대화의 성격, 말투 등을 정의하세요.',
                'tab_main': '메인',
                'model_type_label': '모델 유형 선택',
                'model_select_label': '모델 선택',
                'api_key_label': 'OpenAI API 키',
                'image_upload_label': '이미지 업로드 (선택)',
                'message_input_label': '메시지 입력',
                'message_placeholder': '메시지를 입력하세요...',
                'send_button': '전송',
                'seed_label': '시드 값',
                'seed_info': '모델의 예측을 재현 가능하게 하기 위해 시드를 설정하세요.',
                # 추가 번역키들...
            },
            'en': {
                'main_title': 'Simple Chatbot',
                'system_message': 'System Message',
                'system_message_default': 'You are a useful AI assistant.',
                'system_message_placeholder': 'Define conversation characteristics, tone, etc.',
                'tab_main': 'Main',
                'model_type_label': 'Select Model Type',
                'model_select_label': 'Select Model',
                'api_key_label': 'OpenAI API Key',
                'image_upload_label': 'Upload Image (Optional)',
                'message_input_label': 'Enter Message',
                'message_placeholder': 'Type your message...',
                'send_button': 'Send',
                'seed_label': 'Seed Value',
                'seed_info': 'Set a seed value to make model predictions reproducible.',
                # Additional translation keys...
            },
            'ja': {
                'main_title': 'シンプルなチャットボット',
                'system_message': 'システムメッセージ',
                'system_message_default': 'あなたは有用なAI秘書です。',
                'system_message_placeholder': '会話の性格、話し方などを定義してください。',
                'tab_main': 'メイン',
                'model_type_label': 'モデルタイプの選択',
                'model_select_label': 'モデルの選択',
                'api_key_label': 'OpenAI APIキー',
                'image_upload_label': '画像のアップロード（オプション）',
                'message_input_label': 'メッセージを入力',
                'message_placeholder': 'メッセージを入力してください...',
                'send_button': '送信',
                'seed_label': 'シード値',
                'seed_info': 'モデルの予測を再現可能にするためにシード値を設定してください。',
                # 追加の翻訳キー...
            },
            'zh_CN': {
                'main_title': '简单聊天机器人',
                'system_message': '系统消息',
                'system_message_default': '你是一个有用的AI秘书。',
                'system_message_placeholder': '定义对话特征、语气等。',
                'tab_main': '主干',
                'model_type_label': '选择模型类型',
                'model_select_label': '选择模型',
                'api_key_label': 'OpenAI API密钥',
                'image_upload_label': '上传图片（可选）',
                'message_input_label': '输入消息',
                'message_placeholder': '请输入消息...',
                'send_button': '发送',
                'seed_label': '种子值',
                'seed_info': '设置种子值以使模型预测可重现。',
                # 其他翻译键...
            },
            'zh_TW': {
                'main_title': '簡單聊天機器人',
                'system_message': '系統消息',
                'system_message_default': '你是一個有用的AI祕書。',
                'system_message_placeholder': '定義對話特徵、語氣等。',
                'tab_main': '主幹',
                'model_type_label': '選擇模型類型',
                'model_select_label': '選擇模型',
                'api_key_label': 'OpenAI API密鑰',
                'image_upload_label': '上傳圖片（可選）',
                'message_input_label': '輸入消息',
                'message_placeholder': '請輸入消息...',
                'send_button': '發送',
                'seed_label': '種子值',
                'seed_info': '設置種子值以使模型預測可重現。',
                # 其他翻譯鍵...
            }
        }

        for lang, translations in default_translations.items():
            file_path = Path('translations') / f'{lang}.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            logger.info(f"Created default translations for {lang}")

    def set_language(self, language_code: str):
        """Set the current language"""
        if language_code in self.translations:
            self.current_language = language_code
            logger.info(f"Language changed to {language_code}")
            return True
        logger.warning(f"Language {language_code} not found, using default language")
        return False

    def get(self, key: str, **kwargs) -> str:
        """Get translation for a key in the current language"""
        try:
            translation = self.translations[self.current_language].get(
                key,
                self.translations[self.default_language].get(key, key)
            )
            return translation.format(**kwargs) if kwargs else translation
        except Exception as e:
            logger.error(f"Translation error for key '{key}': {e}")
            return key

    def add_translation(self, language_code: str, translations: Dict[str, str]):
        """Add or update translations for a language"""
        if language_code not in self.translations:
            self.translations[language_code] = {}
        
        self.translations[language_code].update(translations)
        
        # Save to file
        file_path = Path('translations') / f'{language_code}.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.translations[language_code], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Added/updated translations for {language_code}")

    def get_available_languages(self) -> list:
        """Get list of available language codes"""
        return list(self.translations.keys())

# Global instance
translation_manager = TranslationManager()

# Helper function for easier access
def _(key: str, **kwargs) -> str:
    """Shorthand for translation_manager.get()"""
    return translation_manager.get(key, **kwargs)