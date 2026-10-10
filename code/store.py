"""Хранилище сервиса на SQLite: сессии, истории диалогов, учёт расхода и рабочая копия заказов.

Общий модуль: им пользуются урок 04 и все следующие. Один файл базы переживает перезапуск процесса.
Эталон data/orders.json по-прежнему никто не меняет: при первом открытии базы рабочая копия создаётся из него.
"""

import json
import sqlite3
import time
import uuid

import support_tools

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (session_id TEXT NOT NULL, seq INTEGER NOT NULL, payload TEXT NOT NULL,
                                     PRIMARY KEY (session_id, seq));
CREATE TABLE IF NOT EXISTS usage (user_id TEXT NOT NULL, ts REAL NOT NULL, input INTEGER, cached INTEGER, output INTEGER);
CREATE TABLE IF NOT EXISTS working_copy (name TEXT PRIMARY KEY, payload TEXT NOT NULL);
"""

_db: sqlite3.Connection | None = None


def open_db(path: str = "support.db") -> None:
    """Открывает базу и поднимает из неё рабочую копию заказов. Если копии в базе ещё нет, берёт эталон."""
    global _db
    if _db is not None:
        _db.close()
    _db = sqlite3.connect(path, check_same_thread=False)  # одно соединение на процесс; запись идёт только из цикла событий
    _db.executescript(SCHEMA)
    rows = dict(_db.execute("SELECT name, payload FROM working_copy"))
    if rows:
        support_tools.ORDERS.clear()
        support_tools.ORDERS.update({int(k): v for k, v in json.loads(rows["orders"]).items()})  # ключи JSON это строки
        support_tools.REFUNDS[:] = json.loads(rows["refunds"])
    else:
        support_tools.reset_state()
        save_working_copy()


def reset_db() -> None:
    """Чистит сессии, истории и расход, возвращает рабочую копию к эталону. Для повторяемых прогонов."""
    with _db:
        for table in ("sessions", "messages", "usage", "working_copy"):
            _db.execute(f"DELETE FROM {table}")
    support_tools.reset_state()
    save_working_copy()


def save_working_copy() -> None:
    """Сохраняет заказы и возвраты из памяти в базу. Вызывайте после каждого шага агента с инструментами."""
    with _db:  # одна транзакция: заказы и возвраты меняются вместе
        _db.execute("REPLACE INTO working_copy VALUES ('orders', ?)", (json.dumps(support_tools.ORDERS, ensure_ascii=False),))
        _db.execute("REPLACE INTO working_copy VALUES ('refunds', ?)", (json.dumps(support_tools.REFUNDS, ensure_ascii=False),))


# --- Сессии и история -------------------------------------------------------


def create_session(user_id: str) -> str:
    session_id = uuid.uuid4().hex  # случайный и неугадываемый: по номеру 1, 2, 3 можно читать чужие диалоги
    with _db:
        _db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (session_id, user_id, time.time()))
    return session_id


def session_user(session_id: str) -> str | None:
    row = _db.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return row[0] if row else None


def load_history(session_id: str) -> list[dict]:
    """Элементы сессии в порядке записи: сообщения user, reasoning, message, function_call, function_call_output.
    Системного промпта в истории нет: он уходит отдельно, в instructions (урок 04) или developer-сообщением с меткой кэша (урок 11)."""
    rows = _db.execute("SELECT payload FROM messages WHERE session_id = ? ORDER BY seq", (session_id,))
    return [json.loads(payload) for (payload,) in rows]


def append_messages(session_id: str, messages: list[dict]) -> None:
    """Дописывает элементы одной транзакцией: либо весь шаг, либо ничего."""
    with _db:
        (last,) = _db.execute("SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        _db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?)",
            [(session_id, last + i, json.dumps(m, ensure_ascii=False)) for i, m in enumerate(messages, 1)],
        )


def delete_user(user_id: str) -> dict:
    """Удаляет всё, что сервис хранит о пользователе, одной транзакцией. Возвращает, сколько строк ушло из каждой таблицы."""
    with _db:  # сообщения удаляются до сессий: после удаления сессий их уже не найти по user_id
        messages = _db.execute(
            "DELETE FROM messages WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)", (user_id,)
        ).rowcount
        sessions = _db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,)).rowcount
        usage = _db.execute("DELETE FROM usage WHERE user_id = ?", (user_id,)).rowcount
    return {"sessions": sessions, "messages": messages, "usage": usage}


# --- Расход и лимиты --------------------------------------------------------


def log_usage(user_id: str, u: dict) -> None:
    with _db:
        _db.execute("INSERT INTO usage VALUES (?, ?, ?, ?, ?)", (user_id, time.time(), u["input"], u["cached"], u["output"]))


def spent(user_id: str) -> tuple[int, int]:
    """(вызовов модели за последнюю минуту, токенов за последние сутки) для проверки лимитов до вызова модели."""
    now = time.time()
    (calls,) = _db.execute("SELECT COUNT(*) FROM usage WHERE user_id = ? AND ts > ?", (user_id, now - 60)).fetchone()
    (tokens,) = _db.execute(
        "SELECT COALESCE(SUM(input + output), 0) FROM usage WHERE user_id = ? AND ts > ?", (user_id, now - 86400)
    ).fetchone()
    return calls, tokens
