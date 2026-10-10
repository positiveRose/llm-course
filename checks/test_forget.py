"""Чекпоинт 4, урок 11: после «забудьте меня» в базе сервиса не осталось следов пользователя.

Запуск из корня проекта после сценария удаления:
    pytest checks/test_forget.py --db support.db --user U-42 --canary "моя собака Жужа"
--canary это строка, которую пользователь написал в диалоге и которая могла попасть в заметки.
Проверка смотрит только базу. Логи, кэши и провайдер из вашего списка мест хранения проверяет ваш сценарий.
"""
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def db(opt):
    path = Path(opt("--db"))
    if not path.exists():
        pytest.fail(f"нет базы {path}")
    return sqlite3.connect(path)


def text_columns(db):
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        cols = [c[1] for c in db.execute(f"PRAGMA table_info({table})") if c[2].upper() in ("TEXT", "")]
        yield table, cols


def test_nothing_left(db, opt):
    user, canary = opt("--user"), opt("--canary")
    if not user:
        pytest.fail("укажите --user с id пользователя, которого забыли")
    needles = [user] + ([canary] if canary else [])
    found = []
    for table, cols in text_columns(db):
        for col in cols:
            for needle in needles:
                (n,) = db.execute(f"SELECT COUNT(*) FROM {table} WHERE instr({col}, ?) > 0", (needle,)).fetchone()
                if n:
                    found.append(f"{table}.{col}: {n} строк с «{needle}»")
    assert not found, "после удаления остались следы:\n" + "\n".join(found)


def test_no_orphan_messages(db):
    tables = {t for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not {"messages", "sessions"} <= tables:
        pytest.skip("нет таблиц messages и sessions урока 04")
    (n,) = db.execute("SELECT COUNT(*) FROM messages WHERE session_id NOT IN (SELECT session_id FROM sessions)").fetchone()
    assert n == 0, f"{n} сообщений без сессии: сессии удалили раньше сообщений, и слова клиента остались"


def test_others_untouched(db, opt):
    tables = {t for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "sessions" not in tables:
        pytest.skip("нет таблицы sessions")
    (n,) = db.execute("SELECT COUNT(*) FROM sessions WHERE user_id != ?", (opt("--user"),)).fetchone()
    assert n > 0, "в базе не осталось ни одной чужой сессии: заведите второго пользователя и проверьте, что удаление его не задело"
