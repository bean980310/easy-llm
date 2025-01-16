# persona_speech_manager.py
import logging
from typing import Dict
from database import load_system_presets

import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class PersonaSpeechManager:
    def __init__(self, translation_manager, characters: Dict[str, Dict[str, str]]):
        """
        :param characters: 캐릭터 이름을 키로 하고, 각 캐릭터의 설정을 값으로 가지는 딕셔너리
                           예: {
                               "친구": {"default_tone": "반말", "languages": "ko"},
                               "선생님": {"default_tone": "존댓말", "languages": "ko"},
                               "영어친구": {"default_tone": "casual", "languages": "en"}
                           }
        """
        self.translation_manager = translation_manager
        self.characters = characters
        self.current_character = None  # 현재 선택된 캐릭터
        self.current_language = None
        self.current_system_preset = None
        self.current_tone = None
    
    def set_character_and_language(self, character_name: str, language: str):
        if character_name in self.characters:
            self.current_character = character_name
            if language in self.characters[character_name]["languages"]:
                self.current_language = language
            else:
                self.current_language = self.characters[character_name]["default_language"]
                logger.warning(f"Language '{language}' not supported by '{character_name}'. Using default language '{self.current_language}'.")
            
            self.current_tone = self.characters[character_name].get("default_tone", "존댓말")
            
            # 시스템 메시지 프리셋 불러오기
            presets = load_system_presets(self.current_language)
            preset_name = self.characters[character_name]["preset_name"]
            if preset_name in presets:
                self.current_system_preset = presets[preset_name]
                logger.info(f"Loaded system preset for {preset_name} in language {self.current_language}")
            else:
                self.current_system_preset = "당신은 유용한 AI 비서입니다."
                logger.warning(f"Preset '{preset_name}' not found for language {self.current_language}. Using default.")
        else:
            raise ValueError(f"캐릭터 '{character_name}'이(가) 존재하지 않습니다.")
    
    def get_system_message(self) -> str:
        """현재 캐릭터의 시스템 메시지 프리셋을 반환합니다."""
        if not self.current_character:
            return "당신은 유용한 AI 비서입니다."
            
        try:
            # 현재 언어로 프리셋 불러오기
            presets = load_system_presets(self.current_language)
            preset_name = self.characters[self.current_character]["preset_name"]
            
            if preset_name in presets:
                logger.info(f"Loaded system message for {self.current_character} in {self.current_language}")
                return presets[preset_name]
            else:
                logger.warning(f"Preset {preset_name} not found for language {self.current_language}")
                return "당신은 유용한 AI 비서입니다."
                
        except Exception as e:
            logger.error(f"Error loading system message: {e}")
            return "당신은 유용한 AI 비서입니다."

    def get_available_presets(self, language: str) -> Dict[str, str]:
        return load_system_presets(language)
    
    def update_tone(self, user_input: str):
        """
        사용자 입력을 기반으로 현재 캐릭터의 말투를 업데이트
        """
        if "존댓말로 말해줘" in user_input:
            self.current_tone = "존댓말"
        elif "반말로 해도 돼" in user_input:
            self.current_tone = "반말"
        elif "カジュアルにして" in user_input:  # 일본어 캐주얼 전환
            self.current_tone = "カジュアル"
        elif "フォーマルにして" in user_input:  # 일본어 포멀 전환
            self.current_tone = "フォーマル"
        elif "随便说" in user_input:  # 중국어 간체 캐주얼 전환
            self.current_tone = "随便"
        elif "正式一点" in user_input:  # 중국어 간체 포멀 전환
            self.current_tone = "正式"
        elif "隨便說" in user_input:  # 중국어 번체 캐주얼 전환
            self.current_tone = "隨便"
        elif "正式一點" in user_input:  # 중국어 번체 포멀 전환
            self.current_tone = "正式"
    
    def generate_response(self, content: str) -> str:
        """
        설정된 말투와 언어에 따라 응답을 생성
        """
        if not self.current_character:
            raise ValueError("캐릭터가 설정되지 않았습니다.")

        tone = self.current_tone if hasattr(self, 'current_tone') else self.characters[self.current_character]["default_tone"]
        language = self.current_language

        if language == "ko":
            if tone == "반말":
                return self.convert_to_casual(content)
            elif tone == "존댓말":
                return self.convert_to_formal(content)
        elif language == "ja":
            if tone == "カジュアル":
                return self.convert_to_casual_japanese(content)
            elif tone == "フォーマル":
                return self.convert_to_formal_japanese(content)
        elif language == "zh_CN":
            if tone == "随便":
                return self.convert_to_casual_simplified_chinese(content)
            elif tone == "正式":
                return self.convert_to_formal_simplified_chinese(content)
        elif language == "zh_TW":
            if tone == "隨便":
                return self.convert_to_casual_traditional_chinese(content)
            elif tone == "正式":
                return self.convert_to_formal_traditional_chinese(content)
        elif language == "en":
            if tone == "casual":
                return self.convert_to_casual_english(content)
            elif tone == "formal":
                return self.convert_to_formal_english(content)
        # 다른 언어의 변환 로직 추가 가능
        return content

    # 한국어 변환 메서드들
    def convert_to_casual(self, content: str) -> str:
        """
        한국어 존댓말을 반말로 변환 (정규 표현식 사용)
        """
        # 예시: "입니다"를 "야"로 변환
        content = re.sub(r'\b입니다\b', '야', content)
        content = re.sub(r'\b예요\b', '야', content)
        content = re.sub(r'\b합니다\b', '해', content)
        content = re.sub(r'\b요\b', '', content)
        return content.strip()


    def convert_to_formal(self, content: str) -> str:
        """
        한국어 반말을 존댓말로 변환 (정규 표현식 사용)
        """
        # 예시: "야"를 "입니다"로 변환
        content = re.sub(r'\b야\b', '입니다', content)
        content = re.sub(r'\b해\b', '합니다', content)
        content = re.sub(r'$', '요', content)  # 문장 끝에 "요" 추가
        return content.strip()

    # 영어 변환 메서드들
    def convert_to_casual_english(self, content: str) -> str:
        """
        영어 formal을 casual로 변환 (예시)
        """
        replacements = {
            "I am": "I'm",
            "do not": "don't",
            "cannot": "can't",
            "would like": "wanna",
            "could you": "could you please",
            # 추가적인 변환 규칙
        }
        for formal, casual in replacements.items():
            content = content.replace(formal, casual)
        return content

    def convert_to_formal_english(self, content: str) -> str:
        """
        영어 casual을 formal로 변환 (예시)
        """
        replacements = {
            "I'm": "I am",
            "don't": "do not",
            "can't": "cannot",
            "wanna": "would like to",
            "could you": "could you please",
            # 추가적인 변환 규칙
        }
        for casual, formal in replacements.items():
            content = content.replace(casual, formal)
        return content

    # 일본어 변환 메서드들
    def convert_to_casual_japanese(self, content: str) -> str:
        """
        일본어 포멀을 캐주얼로 변환 (예시)
        """
        replacements = {
            "です": "だよ",
            "ます": "るよ",
            "ございます": "あげるよ",
            "いただきます": "もらうよ",
            # 추가적인 변환 규칙
        }
        for formal, casual in replacements.items():
            content = content.replace(formal, casual)
        return content

    def convert_to_formal_japanese(self, content: str) -> str:
        """
        일본어 캐주얼을 포멀로 변환 (예시)
        """
        replacements = {
            "だよ": "です",
            "るよ": "ます",
            "あげるよ": "ございます",
            "もらうよ": "いただきます",
            # 추가적인 변환 규칙
        }
        for casual, formal in replacements.items():
            content = content.replace(casual, formal)
        return content

    # 중국어 간체 변환 메서드들
    def convert_to_casual_simplified_chinese(self, content: str) -> str:
        """
        중국어 간체 正式을 随便으로 변환 (예시)
        """
        replacements = {
            "您好": "嘿",
            "请问": "请",
            "谢谢": "谢谢你",
            "不客气": "没事",
            # 추가적인 변환 규칙
        }
        for formal, casual in replacements.items():
            content = content.replace(formal, casual)
        return content

    def convert_to_formal_simplified_chinese(self, content: str) -> str:
        """
        중국어 간체 随便을 正式으로 변환 (예시)
        """
        replacements = {
            "嘿": "您好",
            "请": "请问",
            "谢谢你": "谢谢",
            "没事": "不客气",
            # 추가적인 변환 규칙
        }
        for casual, formal in replacements.items():
            content = content.replace(casual, formal)
        return content

    # 중국어 번체 변환 메서드들
    def convert_to_casual_traditional_chinese(self, content: str) -> str:
        """
        중국어 번체 正式을 隨便으로 변환 (예시)
        """
        replacements = {
            "您好": "嘿",
            "請問": "請",
            "謝謝": "謝謝你",
            "不客氣": "沒事",
            # 추가적인 변환 규칙
        }
        for formal, casual in replacements.items():
            content = content.replace(formal, casual)
        return content

    def convert_to_formal_traditional_chinese(self, content: str) -> str:
        """
        중국어 번체 隨便을 正式으로 변환 (예시)
        """
        replacements = {
            "嘿": "您好",
            "請": "請問",
            "謝謝你": "謝謝",
            "沒事": "不客氣",
            # 추가적인 변환 규칙
        }
        for casual, formal in replacements.items():
            content = content.replace(casual, formal)
        return content

    def process_input(self, user_input: str, base_response: str) -> str:
        """
        사용자 입력 처리 후 응답 생성
        """
        self.update_tone(user_input)
        return self.generate_response(base_response)