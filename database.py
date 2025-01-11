import sqlite3
import logging
import gradio as gr
import json
import datetime
import csv

logger = logging.getLogger(__name__)

# DB 초기화 시 시스템 메시지 프리셋 테이블 생성
def initialize_presets_db():
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("시스템 메시지 프리셋 테이블 초기화 완료.")
    except Exception as e:
        logger.error(f"시스템 메시지 프리셋 DB 초기화 오류: {e}")

# 앱 시작 시 DB 초기화 함수 호출
initialize_presets_db()

# 시스템 메시지 프리셋 불러오기
def load_system_presets():
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, content FROM system_presets ORDER BY name ASC")
        rows = cursor.fetchall()
        conn.close()
        presets = {name: content for name, content in rows}
        return presets
    except Exception as e:
        logger.error(f"시스템 메시지 프리셋 불러오기 오류: {e}")
        return {}

# 새로운 시스템 메시지 프리셋 추가
def add_system_preset(name, content):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_presets (name, content) 
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET content=excluded.content
        """, (name, content))
        conn.commit()
        conn.close()
        logger.info(f"프리셋 추가/업데이트: {name}")
        return True, "프리셋이 성공적으로 추가/업데이트되었습니다."
    except Exception as e:
        logger.error(f"시스템 메시지 프리셋 추가/업데이트 오류: {e}")
        return False, f"오류 발생: {e}"

# 시스템 메시지 프리셋 삭제
def delete_system_preset(name):
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM system_presets WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        logger.info(f"시스템 메시지 프리셋 삭제: {name}")
        return True, "프리셋이 성공적으로 삭제되었습니다."
    except Exception as e:
        logger.error(f"시스템 메시지 프리셋 삭제 오류: {e}")
        return False, f"오류 발생: {e}"
def get_preset_choices():
    presets = load_system_presets()
    return sorted(presets.keys())

# 초기 로드 시 프리셋 Dropdown 업데이트
def initial_load_presets():
    presets = get_preset_choices()
    return gr.update(choices=presets)

# 프리셋 추가 핸들러
def handle_add_preset(name, content):
    if not name.strip() or not content.strip():
        return "❌ 프리셋 이름과 내용을 모두 입력해주세요.", gr.update(choices=get_preset_choices())
    success, message = add_system_preset(name.strip(), content.strip())
    if success:
        presets = get_preset_choices()
        return message, gr.update(choices=presets)
    else:
        return message, gr.update(choices=get_preset_choices())

# 프리셋 삭제 핸들러
def handle_delete_preset(name):
    if not name:
        return "❌ 삭제할 프리셋을 선택해주세요.", gr.update(choices=get_preset_choices())
    success, message = delete_system_preset(name)
    if success:
        presets = get_preset_choices()
        return message, gr.update(choices=presets)
    else:
        return message, gr.update(choices=get_preset_choices())
    
def get_existing_sessions():
    """
    DB에서 이미 존재하는 모든 session_id 목록을 가져옴 (중복 없이).
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM chat_history ORDER BY session_id ASC")
        rows = cursor.fetchall()
        conn.close()
        session_ids = [r[0] for r in rows]
        return session_ids
    except Exception as e:
        logger.error(f"세션 목록 조회 오류: {e}")
        return []

def save_chat_history_db(history, session_id):
    if session_id is None:
        logger.error("세션 ID가 None입니다. 채팅 히스토리를 저장할 수 없습니다.")
        return
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        for message in history:
            cursor.execute("""
                INSERT INTO chat_history (session_id, role, content)
                VALUES (?, ?, ?)
            """, (session_id, message["role"], message["content"]))
        conn.commit()
        logger.info(f"DB에 채팅 히스토리 저장 완료 (session_id={session_id})")
    except sqlite3.IntegrityError as e:
        logger.error(f"IntegrityError: {e}")
    except Exception as e:
        logger.error(f"Error saving chat history to DB: {e}")
    finally:
        conn.close()
    
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
