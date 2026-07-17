"""
Слой хранения — SQLite.

Хранит документы, их чанки и лог диалогов с ботом (для /api/analytics).
Файл базы данных создаётся рядом (rag.db) при первом запуске.
"""
import sqlite3
from contextlib import contextmanager

DB_PATH = "rag.db"


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                used_chunk_ids TEXT,
                answered INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def add_document(filename: str) -> int:
    with _conn() as conn:
        cur = conn.execute("INSERT INTO documents (filename) VALUES (?)", (filename,))
        return cur.lastrowid


def add_chunks(document_id: int, chunks: list[str]):
    with _conn() as conn:
        conn.executemany(
            "INSERT INTO chunks (document_id, content) VALUES (?, ?)",
            [(document_id, c) for c in chunks],
        )


def list_documents():
    with _conn() as conn:
        rows = conn.execute("""
            SELECT d.id, d.filename, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.id DESC
        """).fetchall()
        return [dict(r) for r in rows]


def delete_document(document_id: int):
    with _conn() as conn:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))


def get_all_chunks():
    """Все чанки со всех документов — используется для пересборки TF-IDF индекса."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT c.id, c.content, c.document_id, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
        """).fetchall()
        return [dict(r) for r in rows]


def log_chat(question: str, answer: str, used_chunk_ids: list[int], answered: bool):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO chats (question, answer, used_chunk_ids, answered) VALUES (?, ?, ?, ?)",
            (question, answer, ",".join(map(str, used_chunk_ids)), int(answered)),
        )


def get_analytics():
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM chats").fetchone()["n"]
        answered = conn.execute("SELECT COUNT(*) AS n FROM chats WHERE answered = 1").fetchone()["n"]
        unanswered = total - answered
        recent = conn.execute(
            "SELECT question, answer, answered, created_at FROM chats ORDER BY id DESC LIMIT 10"
        ).fetchall()
        return {
            "total_questions": total,
            "answered": answered,
            "unanswered": unanswered,
            "answer_rate": round(answered / total, 3) if total else None,
            "recent_chats": [dict(r) for r in recent],
        }
