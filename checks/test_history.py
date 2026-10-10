"""Чекпоинт 1, урок 04: история в базе после диалога с перезапуском.

Запуск из корня проекта после диалога: pytest checks/test_history.py --db support.db
"""
import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def sessions(opt):
    path = Path(opt("--db"))
    if not path.exists():
        pytest.fail(f"нет базы {path}: сначала проведите диалог с отменой заказа и перезапуском сервера")
    db = sqlite3.connect(path)
    rows = db.execute("SELECT session_id, seq, payload FROM messages ORDER BY session_id, seq").fetchall()
    out = {}
    for session_id, seq, payload in rows:
        out.setdefault(session_id, []).append((seq, json.loads(payload)))
    if not out:
        pytest.fail("в таблице messages пусто")
    return out


def test_no_system_prompt(sessions):
    for sid, items in sessions.items():
        for seq, item in items:
            assert item.get("role") not in ("system", "developer"), (
                f"сессия {sid[:8]}, seq {seq}: системный промпт попал в историю, а он уходит отдельно")


def test_every_call_has_output(sessions):
    for sid, items in sessions.items():
        calls = [i["call_id"] for _, i in items if i.get("type") == "function_call"]
        outputs = [i["call_id"] for _, i in items if i.get("type") == "function_call_output"]
        missing = set(calls) - set(outputs)
        orphans = set(outputs) - set(calls)
        assert not missing, f"сессия {sid[:8]}: вызовы без результата {missing}, API откажется принять такую историю"
        assert not orphans, f"сессия {sid[:8]}: результаты без вызова {orphans}"


def test_seq_is_continuous(sessions):
    for sid, items in sessions.items():
        seqs = [s for s, _ in items]
        assert seqs == list(range(1, len(seqs) + 1)), (
            f"сессия {sid[:8]}: seq идут {seqs[:10]}…, а должны 1, 2, 3 без пропусков и без начала заново после перезапуска")


def test_dialogue_used_a_tool(sessions):
    assert any(i.get("type") == "function_call" for items in sessions.values() for _, i in items), (
        "в истории нет ни одного вызова инструмента: проведите диалог, который начинается с отмены заказа")
