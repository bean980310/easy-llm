from typing import Optional, List, Tuple, Dict, Any
import sqlite3
import logging
from contextlib import contextmanager
import gradio as gr
import json
import datetime
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect("chat_history.db", timeout=10)
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise DatabaseError(f"Failed to connect to database: {e}")
    finally:
        if conn:
            conn.close()

def initialize_presets_db() -> None:
    """Initialize system message presets table"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    content TEXT NOT NULL,
                    UNIQUE(name, language)
                )
            """)
            conn.commit()
            logger.info("System message presets table initialized.")
    except DatabaseError as e:
        logger.error(f"Failed to initialize presets DB: {e}")
        raise


# 앱 시작 시 DB 초기화 함수 호출
initialize_presets_db()

def insert_default_presets(translation_manager):
    default_presets = ['MINAMI_ASUKA_PRESET', 'MAKOTONO_AOI_PRESET', 'AINO_KOITO_PRESET']
    languages = translation_manager.get_available_languages()
    
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    
    for preset in default_presets:
        for lang in languages:
            # 이미 존재하는지 확인
            cursor.execute("SELECT * FROM system_presets WHERE name = ? AND language = ?", (preset, lang))
            if cursor.fetchone() is None:
                # translation_manager에서 해당 프리셋의 내용을 가져옴
                if preset == 'MINAMI_ASUKA_PRESET':
                    content = translation_manager.get_character_setting('minami_asuka')
                elif preset == 'MAKOTONO_AOI_PRESET':
                    content = translation_manager.get_character_setting('makotono_aoi')
                elif preset == 'AINO_KOITO_PRESET':
                    content = translation_manager.get_character_setting('aino_koito')
                else:
                    content = "Default content"
                
                cursor.execute("INSERT INTO system_presets (name, language, content) VALUES (?, ?, ?)", (preset, lang, content))
                logger.info(f"Inserted default preset: {preset} (language: {lang})")
    
    conn.commit()
    conn.close()
    
# 시스템 메시지 프리셋 불러오기
def load_system_presets(language):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, content FROM system_presets WHERE language = ? ORDER BY name ASC", (language,))
        rows = cursor.fetchall()
        conn.close()
        presets = {name: content for name, content in rows}
        return presets
    except Exception as e:
        logger.error(f"시스템 메시지 프리셋 불러오기 오류: {e}")
        return {}

# 새로운 시스템 메시지 프리셋 추가
def add_system_preset(name, language, content, overwrite=False):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        if overwrite:
            cursor.execute("""
                UPDATE system_presets SET content = ?
                WHERE name = ? AND language = ?
            """, (content, name, language))
            logger.info(f"프리셋 업데이트: {name} (언어: {language})")
        else:
            cursor.execute("""
                INSERT INTO system_presets (name, language, content) 
                VALUES (?, ?, ?)
            """, (name, language, content))
            logger.info(f"프리셋 추가: {name} (언어: {language})")
        conn.commit()
        conn.close()
        return True, "프리셋이 성공적으로 저장되었습니다."
    except sqlite3.IntegrityError:
        logger.warning(f"프리셋 '{name}' (언어: {language})이(가) 이미 존재합니다.")
        return False, "프리셋이 이미 존재합니다."
    except Exception as e:
        logger.error(f"시스템 메시지 프리셋 저장 오류: {e}")
        return False, f"오류 발생: {e}"

# 시스템 메시지 프리셋 삭제
def delete_system_preset(name, language):
    default_presets = ['MINAMI_ASUKA_PRESET', 'MAKOTONO_AOI_PRESET', 'AINO_KOITO_PRESET']
    if name in default_presets:
        logger.warning(f"기본 프리셋 '{name}' (언어: {language}) 삭제 시도됨. 삭제 불가.")
        return False, "기본 프리셋은 삭제할 수 없습니다."
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM system_presets WHERE name = ? AND language = ?", (name, language))
        conn.commit()
        conn.close()
        logger.info(f"시스템 메시지 프리셋 삭제: {name} (언어: {language})")
        return True, "프리셋이 성공적으로 삭제되었습니다."
    except Exception as e:
        logger.error(f"시스템 메시지 프리셋 삭제 오류: {e}")
        return False, f"오류 발생: {e}"
    
def preset_exists(name, language):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM system_presets WHERE name = ? AND language = ?", (name, language))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except Exception as e:
        logger.error(f"프리셋 존재 여부 확인 오류: {e}")
        return False
    
def get_preset_choices(language):
    presets = load_system_presets(language)
    return sorted(presets.keys())

# 초기 로드 시 프리셋 Dropdown 업데이트
def initial_load_presets(language):
    presets = get_preset_choices(language)
    return gr.update(choices=presets)

# 프리셋 추가 핸들러
def handle_add_preset(name, language, content, confirm_overwrite=False):
    if not name.strip() or not content.strip():
        return "❌ 프리셋 이름과 내용을 모두 입력해주세요.", gr.update(choices=get_preset_choices(language)), False
    
    exists = preset_exists(name.strip(), language)
    
    if exists and not confirm_overwrite:
        # 프리셋이 존재하지만 덮어쓰기 확인이 이루어지지 않은 경우
        return "⚠️ 해당 프리셋이 이미 존재합니다. 덮어쓰시겠습니까?", gr.update(choices=get_preset_choices(language)), True  # 추가 출력: 덮어쓰기 필요
    
    success, message = add_system_preset(name.strip(), language, content.strip(), overwrite=exists)
    if success:
        presets = get_preset_choices(language)
        return message, gr.update(choices=presets), False  # 덮어쓰기 완료
    else:
        return message, gr.update(choices=get_preset_choices(language)), False


# 프리셋 삭제 핸들러
def handle_delete_preset(name, language):
    if not name:
        return "❌ 삭제할 프리셋을 선택해주세요.", gr.update(choices=get_preset_choices(language))
    success, message = delete_system_preset(name, language)
    if success:
        presets = get_preset_choices(language)
        return message, gr.update(choices=presets)
    else:
        return message, gr.update(choices=get_preset_choices(language))
    
def get_existing_sessions() -> List[str]:
    """Get list of existing session IDs"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT session_id FROM chat_history ORDER BY session_id ASC")
            return [row[0] for row in cursor.fetchall()]
            
    except DatabaseError as e:
        logger.error(f"Error retrieving sessions: {e}")
        return []
    
def save_chat_history_db(history: List[Dict[str, Any]], session_id: str = "session_1") -> bool:
    """Save chat history to SQLite database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            for msg in history:
                # Check for duplicate messages
                cursor.execute("""
                    SELECT COUNT(*) FROM chat_history
                    WHERE session_id = ? AND role = ? AND content = ?
                """, (session_id, msg.get("role"), msg.get("content")))
                
                if cursor.fetchone()[0] == 0:
                    cursor.execute("""
                        INSERT INTO chat_history (session_id, role, content)
                        VALUES (?, ?, ?)
                    """, (session_id, msg.get("role"), msg.get("content")))
            
            conn.commit()
            logger.info(f"Chat history saved successfully (session_id={session_id})")
            return True
            
    except sqlite3.IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        return False
    except sqlite3.OperationalError as e:
        logger.error(f"Database operational error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving chat history: {e}")
        return False
    
def save_chat_history(history):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"chat_history_{timestamp}.json"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"채팅 히스토리 저장 완료: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"채팅 히스토리 저장 중 오류: {e}")
        return None

def save_chat_history_csv(history):
    """
    채팅 히스토리를 CSV 형태로 저장
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"chat_history_{timestamp}.csv"
    try:
        # CSV 파일 열기
        with open(file_name, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 작성
            writer.writerow(["role", "content"])
            # 각 메시지 row 작성
            for msg in history:
                writer.writerow([msg.get("role"), msg.get("content")])
        logger.info(f"채팅 히스토리 CSV 저장 완료: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"채팅 히스토리 CSV 저장 중 오류: {e}")
        return None
    
def save_chat_button_click(history):
    if not history:
        return "채팅 이력이 없습니다."
    saved_path = save_chat_history(history)
    if saved_path is None:
        return "❌ 채팅 기록 저장 실패"
    else:
        return f"✅ 채팅 기록이 저장되었습니다: {saved_path}"
    
# 예: session_id를 함수 인자로 전달받아 DB로부터 해당 세션 데이터만 불러오기
def load_chat_from_db(session_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    history = []
    for row in rows:
        role, content = row
        history.append({"role": role, "content": content})
    return history

def delete_session_history(session_id):
    """
    특정 세션 ID와 연결된 모든 채팅 기록을 데이터베이스에서 삭제합니다.
    
    Args:
        session_id (str): 삭제할 세션의 ID.
    
    Returns:
        bool: 삭제 성공 여부.
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        logger.info(f"Session '{session_id}' history deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to delete session '{session_id}' history: {e}")
        return False
    
# 모든 세션과 채팅 기록 삭제 함수 추가
def delete_all_sessions():
    """
    데이터베이스의 모든 세션과 채팅 기록을 삭제합니다.
    
    Returns:
        bool: 삭제 성공 여부.
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history")
        conn.commit()
        conn.close()
        logger.info("모든 세션과 채팅 기록이 성공적으로 삭제되었습니다.")
        return True
    except Exception as e:
        logger.error(f"모든 세션 삭제 오류: {e}")
        return False