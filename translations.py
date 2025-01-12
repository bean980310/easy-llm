# translations.py

from typing import Dict, Any
import json
import os
from pathlib import Path
import logging
import locale

from minami_asuka_char_set import DEFAULT_CHARACTER_SETTINGS

logger = logging.getLogger(__name__)

class MinamiTranslationManager:
    def __init__(self, default_language: str = 'ko'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.character_settings: Dict[str, str] = {}
        self.load_translations()
        self.load_character_settings()

    def load_translations(self):
        """UI 요소의 번역 파일들을 로드"""
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

    def load_character_settings(self):
        """캐릭터 설정 로드"""
        settings_path = Path('character_settings')
        if not settings_path.exists():
            settings_path.mkdir(parents=True)
            self._create_default_character_settings()
        
        for lang_file in settings_path.glob('*.txt'):
            lang_code = lang_file.stem
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.character_settings[lang_code] = f.read()
                logger.info(f"Loaded character settings for {lang_code}")
            except Exception as e:
                logger.error(f"Error loading character settings for {lang_code}: {e}")

    def _create_default_translations(self):
        """기본 UI 번역 생성"""
        default_translations = {
            'ko': {
                'title': '미나미 아스카와 대화하기',
                'language_select': '언어 선택',
                'language_info': '인터페이스 언어를 선택하세요',
                'system_message_label': '시스템 메시지',
                'main_tab': '메인',
                'select_model': '모델 선택',
                'selected_model': '선택된 모델',
                'chatbot_label': '채팅',
                'profile_image_label': '프로필 이미지',
                'input_placeholder': '메시지를 입력하세요...',
                'send_button': '전송',
                'seed_value': '시드 값',
                'seed_info': '모델의 예측을 재현 가능하게 하기 위해 시드를 설정하세요.',
                'device_settings': '장치 설정',
                'device_select': '사용할 장치 선택',
                'device_auto': '자동 (권장)',
                'device_info': '자동 설정을 사용하면 시스템에 따라 최적의 장치를 선택합니다.',
                'settings_tab': '설정',
                'settings_title': '설정',
                'preset_management': '시스템 메시지 프리셋 관리',
                'preset_select': '프리셋 선택',
                'preset_apply': '프리셋 적용',
                'session_management': '세션 관리',
                'session_list_refresh': '세션 목록 갱신',
                'existing_sessions': '기존 세션 목록',
                'create_new_session': '새 세션 생성',
                'apply_session': '세션 적용',
                'delete_session': '세션 삭제',
                'delete_confirm': '정말로 이 세션을 삭제하시겠습니까?',
                'confirm_delete': '삭제 확인',
                'session_manage_result': '세션 관리 결과',
                'current_session': '현재 세션 ID',
                'no_session': '세션 없음',
                'current_session_display': '현재 세션: {sid}'
            },
            'ja': {
                'title': '南飛鳥との会話',
                'language_select': '言語選択',
                'language_info': 'インターフェース言語を選択してください',
                'system_message_label': 'システムメッセージ',
                'main_tab': 'メイン',
                'select_model': 'モデル選択',
                'selected_model': '選択されたモデル',
                'chatbot_label': 'チャット',
                'profile_image_label': 'プロフィール画像',
                'input_placeholder': 'メッセージを入力してください...',
                'send_button': '送信',
                'seed_value': 'シード値',
                'seed_info': 'モデルの予測を再現可能にするためにシード値を設定してください',
                'device_settings': 'デバイス設定',
                'device_select': '使用するデバイスを選択',
                'device_auto': '自動（推奨）',
                'device_info': '自動設定を使用すると、システムに応じて最適なデバイスを選択します',
                'settings_tab': '設定',
                'settings_title': '設定',
                'preset_management': 'システムメッセージプリセット管理',
                'preset_select': 'プリセット選択',
                'preset_apply': 'プリセット適用',
                'session_management': 'セッション管理',
                'session_list_refresh': 'セッション一覧更新',
                'existing_sessions': '既存のセッション一覧',
                'create_new_session': '新規セッション作成',
                'apply_session': 'セッション適用',
                'delete_session': 'セッション削除',
                'delete_confirm': '本当にこのセッションを削除しますか？',
                'confirm_delete': '削除確認',
                'session_manage_result': 'セッション管理結果',
                'current_session': '現在のセッションID',
                'no_session': 'セッションなし',
                'current_session_display': '現在のセッション: {sid}'
            },
            'zh_CN': {
                'title': '与南飞鸟对话',
                'language_select': '选择语言',
                'language_info': '选择界面语言',
                'system_message_label': '系统消息',
                'main_tab': '主干',
                'select_model': '选择模型',
                'selected_model': '已选择的模型',
                'chatbot_label': '聊天',
                'profile_image_label': '头像',
                'input_placeholder': '请输入消息...',
                'send_button': '发送',
                'seed_value': '种子值',
                'seed_info': '设置种子值以使模型预测可重现',
                'device_settings': '设备设置',
                'device_select': '选择使用设备',
                'device_auto': '自动（推荐）',
                'device_info': '使用自动设置将根据系统选择最优设备',
                'settings_tab': '设置',
                'settings_title': '设置',
                'preset_management': '系统消息预设管理',
                'preset_select': '选择预设',
                'preset_apply': '应用预设',
                'session_management': '会话管理',
                'session_list_refresh': '刷新会话列表',
                'existing_sessions': '现有会话列表',
                'create_new_session': '创建新会话',
                'apply_session': '应用会话',
                'delete_session': '删除会话',
                'delete_confirm': '确定要删除这个会话吗？',
                'confirm_delete': '确认删除',
                'session_manage_result': '会话管理结果',
                'current_session': '当前会话ID',
                'no_session': '无会话',
                'current_session_display': '当前会话: {sid}'
            },
            'zh_TW': {
                'title': '與南飛鳥對話',
                'language_select': '選擇語言',
                'language_info': '選擇界面語言',
                'system_message_label': '系統訊息',
                'main_tab': '主幹',
                'select_model': '選擇模型',
                'selected_model': '已選擇的模型',
                'chatbot_label': '聊天',
                'profile_image_label': '頭像',
                'input_placeholder': '請輸入訊息...',
                'send_button': '發送',
                'seed_value': '種子值',
                'seed_info': '設置種子值以使模型預測可重現',
                'device_settings': '裝置設定',
                'device_select': '選擇使用裝置',
                'device_auto': '自動（推薦）',
                'device_info': '使用自動設定將根據系統選擇最優裝置',
                'settings_tab': '設定',
                'settings_title': '設定',
                'preset_management': '系統訊息預設管理',
                'preset_select': '選擇預設',
                'preset_apply': '套用預設',
                'session_management': '會話管理',
                'session_list_refresh': '重新整理會話清單',
                'existing_sessions': '現有會話清單',
                'create_new_session': '建立新會話',
                'apply_session': '套用會話',
                'delete_session': '刪除會話',
                'delete_confirm': '確定要刪除這個會話嗎？',
                'confirm_delete': '確認刪除',
                'session_manage_result': '會話管理結果',
                'current_session': '目前會話ID',
                'no_session': '無會話',
                'current_session_display': '目前會話: {sid}'
            },
            'en': {
                'title': 'Chat with Minami Asuka',
                'language_select': 'Select Language',
                'language_info': 'Choose interface language',
                'system_message_label': 'System Message',
                'main_tab': 'Main',
                'select_model': 'Select Model',
                'selected_model': 'Selected Model',
                'chatbot_label': 'Chat',
                'profile_image_label': 'Profile Image',
                'input_placeholder': 'Type your message...',
                'send_button': 'Send',
                'seed_value': 'Seed Value',
                'seed_info': 'Set a seed value to make model predictions reproducible',
                'device_settings': 'Device Settings',
                'device_select': 'Select Device',
                'device_auto': 'Auto (Recommended)',
                'device_info': 'Automatic setting will choose the optimal device for your system',
                'settings_tab': 'Settings',
                'settings_title': 'Settings',
                'preset_management': 'System Message Preset Management',
                'preset_select': 'Select Preset',
                'preset_apply': 'Apply Preset',
                'session_management': 'Session Management',
                'session_list_refresh': 'Refresh Session List',
                'existing_sessions': 'Existing Sessions',
                'create_new_session': 'Create New Session',
                'apply_session': 'Apply Session',
                'delete_session': 'Delete Session',
                'delete_confirm': 'Are you sure you want to delete this session?',
                'confirm_delete': 'Confirm Delete',
                'session_manage_result': 'Session Management Result',
                'current_session': 'Current Session ID',
                'no_session': 'No Session',
                'current_session_display': 'Current Session: {sid}'
            }
        }

        for lang, translations in default_translations.items():
            file_path = Path('translations') / f'{lang}.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            logger.info(f"Created default translations for {lang}")

    def _create_default_character_settings(self):
        """기본 캐릭터 설정 파일 생성"""
        for lang, setting in DEFAULT_CHARACTER_SETTINGS.items():
            file_path = Path('character_settings') / f'{lang}.txt'
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(setting)
            logger.info(f"Created default character setting for {lang}")

    def set_language(self, language_code: str) -> bool:
        """현재 언어 설정"""
        if language_code in self.translations:
            self.current_language = language_code
            logger.info(f"Language changed to {language_code}")
            return True
        logger.warning(f"Language {language_code} not found, using default language")
        return False

    def get(self, key: str, **kwargs) -> str:
        """UI 텍스트 번역 가져오기"""
        try:
            translation = self.translations[self.current_language].get(
                key,
                self.translations[self.default_language].get(key, key)
            )
            return translation.format(**kwargs) if kwargs else translation
        except Exception as e:
            logger.error(f"Translation error for key '{key}': {e}")
            return key

    def get_character_setting(self) -> str:
        """현재 언어의 캐릭터 설정 가져오기"""
        return self.character_settings.get(
            self.current_language,
            self.character_settings[self.default_language]
        )

    def get_available_languages(self) -> list:
        """사용 가능한 언어 코드 목록"""
        return list(self.translations.keys())

    def get_language_display_name(self, lang_code: str) -> str:
        """언어 코드에 대한 표시 이름"""
        display_names = {
            'ko': '한국어',
            'ja': '日本語',
            'zh_CN': '中文(简体)',
            'zh_TW': '中文(繁體)',
            'en': 'English'
        }
        return display_names.get(lang_code, lang_code)

# 시스템 언어 감지
def detect_system_language() -> str:
    lang, _ = locale.getdefaultlocale()
    if lang:
        lang_code = lang.split('_')[0]
        if lang_code == 'zh':
            # 중국어의 경우 간체/번체 구분
            if lang.lower() == 'zh_tw':
                return 'zh_TW'
            return 'zh_CN'
        if lang_code in ['ko', 'ja', 'en']:
            return lang_code
    return 'ko'

# 글로벌 인스턴스
translation_manager = MinamiTranslationManager(default_language=detect_system_language())

# 간편한 접근을 위한 헬퍼 함수
def _(key: str, **kwargs) -> str:
    """UI 텍스트 번역을 위한 단축 함수"""
    return translation_manager.get(key, **kwargs)

def get_system_message() -> str:
    """현재 언어의 캐릭터 설정 반환"""
    return translation_manager.get_character_setting()