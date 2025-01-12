# translations.py

import locale
from typing import Dict, Optional, Union
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
class TranslationManager:
    def __init__(self, default_language: str = 'ko'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.load_translations()

    def load_translations(self):
        """번역 파일 로드"""
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
        """기본 번역 파일 생성"""
        default_translations = {
            'ko': {
                'main_title': '간단한 Chatbot',
                'language_select': '언어 선택',
                'language_info': '인터페이스 언어를 선택하세요',
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
                'download_tab': '다운로드',
                'download_title': '모델 다운로드',
                'download_description': 'HuggingFace에서 모델을 다운로드하고 로컬에 저장합니다.',
                'download_description_detail': '미리 정의된 모델 목록에서 선택하거나, 커스텀 모델 ID를 직접 입력할 수 있습니다.',
                'download_mode_label': '다운로드 방식 선택',
                'download_mode_predefined': '미리 정의된 모델',
                'download_mode_custom': '커스텀 모델 ID',
                'model_select_label': '모델 선택',
                'model_select_info': '지원되는 모델 목록입니다.',
                'custom_model_id_label': '커스텀 모델 ID',
                'custom_model_id_placeholder': '예) facebook/opt-350m',
                'custom_model_id_info': 'HuggingFace의 모델 ID를 입력하세요 (예: organization/model-name)',
                'save_path_label': '저장 경로',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '비워두면 자동으로 경로가 생성됩니다.',
                'auth_required_label': '인증 필요',
                'auth_required_info': '비공개 또는 gated 모델 다운로드 시 체크',
                'hf_token_label': 'HuggingFace Token',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': 'HuggingFace에서 발급받은 토큰을 입력하세요.',
                'download_start_button': '다운로드 시작',
                'download_cancel_button': '취소',
                'download_details_label': '상세 정보',
                'download_log_label': '다운로드 로그',
                'download_preparing': '🔄 다운로드 준비 중...',
                'download_in_progress': '🔄 다운로드 중...',
                'download_complete': '✅ 다운로드 완료!',
                'download_failed': '❌ 다운로드 실패',
                'download_error': '❌ 오류 발생',
                'download_error_no_model': '❌ 모델 ID를 입력해주세요.',
                # 추가 번역키들...
            },
            'en': {
                'main_title': 'Simple Chatbot',
                'language_select': 'Select Language',
                'language_info': 'Choose interface language',
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
                'download_tab': 'Download',
                'download_title': 'Model Download',
                'download_description': 'Download models from HuggingFace and save them locally.',
                'download_description_detail': 'You can select from a predefined list of models or directly enter a custom model ID.',
                'download_mode_label': 'Download Method',
                'download_mode_predefined': 'Predefined Models',
                'download_mode_custom': 'Custom Model ID',
                'model_select_label': 'Model Select',
                'model_select_info': 'List of supported models.',
                'custom_model_id_label': 'Custom Model ID',
                'custom_model_id_placeholder': 'e.g., facebook/opt-350m',
                'custom_model_id_info': 'Enter a HuggingFace model ID (e.g., organization/model-name)',
                'save_path_label': 'Save Path',
                'save_path_placeholder': './models/my-model',
                'save_path_info': 'Leave empty for automatic path generation.',
                'auth_required_label': 'Authentication Required',
                'auth_required_info': 'Check for private or gated model downloads',
                'hf_token_label': 'HuggingFace Token',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': 'Enter your HuggingFace token.',
                'download_start_button': 'Start Download',
                'download_cancel_button': 'Cancel',
                'download_details_label': 'Details',
                'download_log_label': 'Download Log',
                'download_preparing': '🔄 Preparing download...',
                'download_in_progress': '🔄 Downloading...',
                'download_complete': '✅ Download complete!',
                'download_failed': '❌ Download failed',
                'download_error': '❌ Error occurred',
                'download_error_no_model': '❌ Please enter a model ID.',
                # Additional translation keys...
            },
            'ja': {
                'main_title': 'シンプルなチャットボット',
                'language_select': '言語選択',
                'language_info': 'インターフェース言語を選択してください',
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
                'download_tab': 'ダウンロード',
                'download_title': 'モデルダウンロード',
                'download_description': 'HuggingFaceからモデルをダウンロードしてローカルに保存します。',
                'download_description_detail': '事前定義されたモデルリストから選択するか、カスタムモデルIDを直接入力できます。',
                'download_mode_label': 'ダウンロード方式',
                'download_mode_predefined': '事前定義モデル',
                'download_mode_custom': 'カスタムモデルID',
                'model_select_label': 'モデル選択',
                'model_select_info': 'サポートされているモデルのリストです。',
                'custom_model_id_label': 'カスタムモデルID',
                'custom_model_id_placeholder': '例）facebook/opt-350m',
                'custom_model_id_info': 'HuggingFaceのモデルIDを入力してください（例：organization/model-name）',
                'save_path_label': '保存パス',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '空欄の場合、自動的にパスが生成されます。',
                'auth_required_label': '認証必要',
                'auth_required_info': 'プライベートまたはゲート付きモデルのダウンロード時にチェック',
                'hf_token_label': 'HuggingFaceトークン',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': 'HuggingFaceで発行されたトークンを入力してください。',
                'download_start_button': 'ダウンロード開始',
                'download_cancel_button': 'キャンセル',
                'download_details_label': '詳細情報',
                'download_log_label': 'ダウンロードログ',
                'download_preparing': '🔄 ダウンロード準備中...',
                'download_in_progress': '🔄 ダウンロード中...',
                'download_complete': '✅ ダウンロード完了！',
                'download_failed': '❌ ダウンロード失敗',
                'download_error': '❌ エラーが発生しました',
                'download_error_no_model': '❌ モデルIDを入力してください。',
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
                'download_tab': '下载',
                'download_title': '模型下载',
                'download_description': '从HuggingFace下载模型并保存到本地。',
                'download_description_detail': '您可以从预定义模型列表中选择，或直接输入自定义模型ID。',
                'download_mode_label': '下载方式',
                'download_mode_predefined': '预定义模型',
                'download_mode_custom': '自定义模型ID',
                'model_select_label': '选择型号',
                'model_select_info': '支持的模型列表。',
                'custom_model_id_label': '自定义模型ID',
                'custom_model_id_placeholder': '例如：facebook/opt-350m',
                'custom_model_id_info': '输入HuggingFace模型ID（例如：organization/model-name）',
                'save_path_label': '保存路径',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '留空将自动生成路径。',
                'auth_required_label': '需要认证',
                'auth_required_info': '下载私有或受限模型时勾选',
                'hf_token_label': 'HuggingFace令牌',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': '输入您的HuggingFace令牌。',
                'download_start_button': '开始下载',
                'download_cancel_button': '取消',
                'download_details_label': '详细信息',
                'download_log_label': '下载日志',
                'download_preparing': '🔄 准备下载中...',
                'download_in_progress': '🔄 下载中...',
                'download_complete': '✅ 下载完成！',
                'download_failed': '❌ 下载失败',
                'download_error': '❌ 发生错误',
                'download_error_no_model': '❌ 请输入模型ID。',
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
                'download_tab': '下載',
                'download_title': '模型下載',
                'download_description': '從HuggingFace下載模型並儲存到本地。',
                'download_description_detail': '您可以從預定義模型列表中選擇，或直接輸入自訂模型ID。',
                'download_mode_label': '下載方式',
                'download_mode_predefined': '預定義模型',
                'download_mode_custom': '自訂模型ID',
                'model_select_label': '選擇型號',
                'model_select_info': '支援的模型列表。',
                'custom_model_id_label': '自訂模型ID',
                'custom_model_id_placeholder': '例如：facebook/opt-350m',
                'custom_model_id_info': '輸入HuggingFace模型ID（例如：organization/model-name）',
                'save_path_label': '儲存路徑',
                'save_path_placeholder': './models/my-model',
                'save_path_info': '留空將自動生成路徑。',
                'auth_required_label': '需要認證',
                'auth_required_info': '下載私有或受限模型時勾選',
                'hf_token_label': 'HuggingFace權杖',
                'hf_token_placeholder': 'hf_...',
                'hf_token_info': '輸入您的HuggingFace權杖。',
                'download_start_button': '開始下載',
                'download_cancel_button': '取消',
                'download_details_label': '詳細信息',
                'download_log_label': '下載日誌',
                'download_preparing': '🔄 準備下載中...',
                'download_in_progress': '🔄 下載中...',
                'download_complete': '✅ 下載完成！',
                'download_failed': '❌ 下載失敗',
                'download_error': '❌ 發生錯誤',
                'download_error_no_model': '❌ 請輸入模型ID。',
            }
        }

        for lang, translations in default_translations.items():
            file_path = Path('translations') / f'{lang}.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            logger.info(f"Created default translations for {lang}")

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
translation_manager = TranslationManager(default_language=detect_system_language())

# 간편한 접근을 위한 헬퍼 함수
def _(key: str, **kwargs) -> str:
    """UI 텍스트 번역을 위한 단축 함수"""
    return translation_manager.get(key, **kwargs)