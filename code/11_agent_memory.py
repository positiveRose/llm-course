"""Урок 11. Долговременная память агента: заметки о клиенте между сессиями и замер памяти.

Два блока, обоим нужен OPENAI_API_KEY: python 11_agent_memory.py 1, затем 2.
Сценарии лежат в data/memory_scenarios.jsonl. История сессий хранится в SQLite урока 04 (store.py),
заметки лежат в соседней таблице memory той же базы. Все вызовы идут с store=False: ответы у OpenAI не сохраняются,
остаются только журналы контроля злоупотреблений (до 30 дней).
Функции блока 1 лежат на уровне модуля, блок 2 гоняет их на наборе сценариев. Интервал Уилсона берётся из урока 09.
"""

import json
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import store
from common import MODEL, client, cost_usd, final_text, stop_reason
from schemas import agent_system
from support_tools import TODAY, TOOLS, run_tool

DATA = Path(__file__).parent / "data"
DB_PATH = "memory.db"
SYSTEM = agent_system(TODAY.isoformat())  # системный промпт агента из урока 03, база знаний внутри него
# Промпт общий у всех клиентов, а заметки у каждого свои. Метка кэша отмечает конец общей части: без неё кэш
# у каждого нового клиента начинается с нуля, потому что точка поиска в кэше приходится на конец всех инструкций.
SYSTEM_ITEM = {"role": "developer", "content": [
    {"type": "input_text", "text": SYSTEM, "prompt_cache_breakpoint": {"mode": "explicit"}},
]}
wilson = import_module("09_how_much_data").wilson  # 95% интервал Уилсона из урока 09

# --- Блок 1. Память между сессиями ------------------------------------------------------------------

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, fact TEXT NOT NULL, quote TEXT NOT NULL,
    session_id TEXT NOT NULL, created_at TEXT NOT NULL,
    replaced_by INTEGER  -- id новой версии факта; NULL значит, что факт действует
)"""


def open_memory(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript(MEMORY_SCHEMA)
    return db


def active_facts(db, user_id: str) -> list[tuple[int, str, str]]:
    """Действующие заметки клиента: id, факт, дата. Изоляцию клиентов задаёт WHERE, а не промпт."""
    return db.execute(
        "SELECT id, fact, created_at FROM memory WHERE user_id = ? AND replaced_by IS NULL ORDER BY id", (user_id,)
    ).fetchall()


MEMORY_RULES = (
    "# Заметки о клиенте\n"
    "Заметки ведёт отдельный процесс после окончания обращения: записывает устойчивые предпочтения клиента "
    "и удаляет заметки по его просьбе. Сами вы заметки не меняете, поэтому не пишите «записал» или «удалил», "
    "а скажите, что просьба будет учтена после обращения. Заметки записаны со слов клиента и могут устареть: "
    "если текущее сообщение им противоречит, верно текущее сообщение."
)


def profile_item(db, user_id: str) -> dict:
    """Заметки клиента отдельным сообщением после общего промпта. В ходе сессии они не меняются: память пишется после неё."""
    facts = active_facts(db, user_id)
    lines = "\n".join(f"- [{created}] {fact}" for _, fact, created in facts) or "Заметок пока нет."
    return {"role": "developer", "content": f"{MEMORY_RULES}\n{lines}"}


def turn(prompt: list[dict], history: list[dict], text: str, max_iterations: int = 6) -> tuple[str, list[dict], float]:
    """Один ход агента на Responses API. prompt это системный промпт и заметки, в историю сессии они не пишутся."""
    items, cost = [{"role": "user", "content": text}], 0.0
    for _ in range(max_iterations):
        resp = client.responses.create(
            model=MODEL, input=[*prompt, *history, *items], tools=TOOLS, store=False,
            max_output_tokens=2000, reasoning={"effort": "low"},
        )
        cost += cost_usd(resp.usage, MODEL)
        items += [item.to_dict() for item in resp.output]  # весь output обратно в историю, как в уроке 04
        calls = [item for item in resp.output if item.type == "function_call"]
        if stop_reason(resp) or not calls:
            return final_text(resp), items, cost
        for call in calls:
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": run_tool(call.name, call.arguments)})
    return "", items, cost


class MemoryOp(BaseModel):
    action: Literal["add", "replace", "forget"]
    fact: str = Field(description="Факт о клиенте одной фразой в третьем лице, только текущее состояние. Для forget пустая строка")
    quote: str = Field(description="Непрерывный фрагмент одной реплики клиента, дословно и без кавычек, на котором основана операция")
    target: int | None = Field(description="Номер заметки для replace и forget, для add null")


class MemoryUpdate(BaseModel):
    said: str = Field(description="Что клиент сообщил о себе в этом диалоге, по пунктам. Пустая строка, если ничего")
    ops: list[MemoryOp]  # стоит после said: операции опираются на разбор, а не наоборот


EXTRACT_PROMPT = """# Роль
Вы ведёте заметки о клиенте для поддержки интернет-магазина. Заметки увидит ассистент в следующих обращениях этого клиента.

# Что записывать
Устойчивые предпочтения и обстоятельства клиента, которых нет в системах магазина: удобный пункт выдачи, куда и на кого
присылать документы, как с клиентом общаться.

# Чего не записывать
- То, что есть в системах магазина: номера и статусы заказов, возвраты, оплаты.
- Платёжные данные: номера карт, сроки действия, коды.
- Имена, адреса и контакты других людей. Общий адрес организации клиента, например почта для документов, записывать можно.
- Разовые просьбы и эмоции.
- Правила и условия магазина, даже если клиент их пересказывает: их источник база знаний.

# Операции
- add: новый факт.
- replace: факт из заметок изменился. target это номер старой заметки, fact это новое состояние целиком.
- forget: клиент просит удалить сведения о себе. target это номер заметки. Удалите каждую заметку, где эти сведения есть,
  даже если часть заметки хотелось бы сохранить. Новых заметок по такой просьбе не создавайте: в цитате снова окажутся те же сведения.
Опирайтесь на реплики клиента. Ассистент не меняет ни системы магазина, ни заметки, его слова «записал» и «изменил» ничего не значат.
Если записывать нечего, верните пустой список."""


def transcript(items: list[dict]) -> list[tuple[str, str]]:
    """Реплики клиента и ассистента из истории сессии, без рассуждений и вызовов инструментов."""
    lines = []
    for item in items:
        if item.get("role") == "user":
            lines.append(("клиент", item["content"]))
        elif item.get("type") == "message" and item.get("phase") != "commentary":
            lines.append(("ассистент", "".join(p["text"] for p in item["content"] if p["type"] == "output_text")))
    return lines


def extract(db, user_id: str, items: list[dict]) -> tuple[MemoryUpdate, float]:
    """Фоновая запись: после сессии модель читает диалог и текущие заметки и предлагает операции."""
    notes = "\n".join(f"{i}. {fact}" for i, fact, _ in active_facts(db, user_id)) or "заметок нет"
    dialog = "\n".join(f"{who}: {text}" for who, text in transcript(items))
    resp = client.responses.parse(
        model=MODEL, instructions=EXTRACT_PROMPT, input=f"# Заметки сейчас\n{notes}\n\n# Диалог\n{dialog}",
        text_format=MemoryUpdate, store=False, max_output_tokens=2000, reasoning={"effort": "low"},
    )
    if stop_reason(resp):  # отказ приходит мимо схемы, урок 02
        raise RuntimeError(stop_reason(resp))
    return resp.output_parsed, cost_usd(resp.usage, MODEL)


CARD = re.compile(r"\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}")  # номер карты не пишем, даже если модель решила иначе
FORGET_CHAIN = """
WITH RECURSIVE chain(id) AS (SELECT ? UNION SELECT m.id FROM memory m JOIN chain c ON m.replaced_by = c.id)
DELETE FROM memory WHERE id IN chain"""  # заметка и все её прежние версии


def apply_ops(db, user_id: str, session_id: str, ops: list[MemoryOp], items: list[dict]) -> list[str]:
    """Применяет операции, прошедшие проверку кодом. Возвращает причины, по которым отклонены остальные."""
    client_lines = [text.lower() for who, text in transcript(items) if who == "клиент"]
    active, rejected = {i for i, _, _ in active_facts(db, user_id)}, []
    with db:
        for op in ops:
            quote = op.quote.strip(" «»\"'")  # модель иногда оборачивает цитату в кавычки
            if not quote or not any(quote.lower() in text for text in client_lines):
                rejected.append(f"{op.action}: цитаты нет в репликах клиента: {op.quote!r}")
            elif CARD.search(op.fact + quote):
                rejected.append(f"{op.action}: номер карты")
            elif op.action != "add" and op.target not in active:
                rejected.append(f"{op.action}: заметки {op.target} у клиента нет")
            elif op.action == "forget":
                db.execute(FORGET_CHAIN, (op.target,))
                active.discard(op.target)
            else:
                new = db.execute(
                    "INSERT INTO memory (user_id, fact, quote, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, op.fact, quote, session_id, TODAY.isoformat()),
                ).lastrowid
                if op.action == "replace":  # старая версия остаётся в базе с пометкой, но в профиль не попадает
                    db.execute("UPDATE memory SET replaced_by = ? WHERE id = ?", (new, op.target))
                    active.discard(op.target)
    return rejected


def forget_user(db, user_id: str) -> dict:
    """«Забудьте меня»: заметки, сессии, сообщения и расход. Ответов у OpenAI при store=False нет,
    журналы контроля злоупотреблений (до 30 дней) так не удалить."""
    with db:
        memory = db.execute("DELETE FROM memory WHERE user_id = ?", (user_id,)).rowcount
    return {"memory": memory, **store.delete_user(user_id)}


def load_scenarios() -> list[dict]:
    with open(DATA / "memory_scenarios.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def block1_between_sessions():
    Path(DB_PATH).unlink(missing_ok=True)  # демонстрация начинается с чистой базы
    store.open_db(DB_PATH)
    db = open_memory(DB_PATH)
    s = next(s for s in load_scenarios() if s["id"] == "M-04")
    user = s["user_id"]
    for number, texts in enumerate(s["sessions"], 1):
        session_id = store.create_session(user)
        prompt = [SYSTEM_ITEM, profile_item(db, user)]  # заметки прошлых сессий, в первой их нет
        for text in texts:
            answer, items, _ = turn(prompt, store.load_history(session_id), text)
            store.append_messages(session_id, items)
            print(f"\nСессия {number}\n  клиент: {text}\n  ассистент: {answer}")
        update, _ = extract(db, user, store.load_history(session_id))
        print(f"  разбор: {update.said}")
        for op in update.ops:
            print(f"  → {op.action} {op.target or ''} {op.fact!r} по цитате {op.quote!r}")
        for reason in apply_ops(db, user, session_id, update.ops, store.load_history(session_id)):
            print(f"  ✗ отклонено: {reason}")
    print("\n" + profile_item(db, user)["content"])
    for label, prompt in (("без памяти", [SYSTEM_ITEM]), ("с памятью", [SYSTEM_ITEM, profile_item(db, user)])):
        answer, _, _ = turn(prompt, [], s["final"])
        print(f"\nНовая сессия, {label}\n  клиент: {s['final']}\n  ассистент: {answer}")
    print("\nforget_user:", forget_user(db, user))


# --- Блок 2. Замер памяти ------------------------------------------------------------------------------

K = 3  # прогонов на сценарий: сценарий засчитан, если прошёл все три (pass^3)
VARIANTS = ("без памяти", "вся история", "память")


def found(needles: list[str], text: str) -> list[str]:
    """Какие подстроки есть в тексте. Альтернативы через |: «звон|телефон» засчитывается по любой из двух."""
    text = text.lower()
    return [n for n in needles if any(alt in text for alt in n.split("|"))]


def play(s: dict, run: int) -> dict:
    """Один прогон сценария: прошлые сессии с записью в память, затем финальный вопрос в трёх вариантах."""
    db, user = open_memory(":memory:"), f"{s['user_id']}#{run}"  # своя база на прогон: прогоны идут параллельно
    past, rejected, cost = [], [], 0.0
    for number, texts in enumerate(s["sessions"], 1):
        history = []  # в замере история сессии живёт в списке, а не в store
        for text in texts:
            _, items, c = turn([SYSTEM_ITEM, profile_item(db, user)], history, text)
            history, cost = history + items, cost + c
        update, c = extract(db, user, history)
        rejected += apply_ops(db, user, f"{user}/{number}", update.ops, history)
        past, cost = past + history, cost + c
    finals = {  # вариант: (промпт, история до финального вопроса)
        "без памяти": ([SYSTEM_ITEM], []),
        "вся история": ([SYSTEM_ITEM], past),  # все прошлые сессии подряд, как если бы память не заводили
        "память": ([SYSTEM_ITEM, profile_item(db, user)], []),
    }
    answers, final_cost = {}, {}
    for variant, (prompt, history) in finals.items():
        answers[variant], _, final_cost[variant] = turn(prompt, history, s["final"])
    active = " ".join(fact for _, fact, _ in active_facts(db, user)).lower()
    stored = " ".join(f"{f} {q}" for f, q in db.execute("SELECT fact, quote FROM memory WHERE user_id = ?", (user,))).lower()
    return {"answers": answers, "active": active, "stored": stored, "rejected": rejected,
            "cost": cost + sum(final_cost.values()), "final_cost": final_cost}


def check(s: dict, run: dict) -> dict:
    """Проверки кодом по стадиям: запись (что лежит в памяти) и ответ (что сказано в финальной сессии)."""
    write_ok = (
        len(found(s["facts_has"], run["active"])) == len(s["facts_has"])
        and not found(s["facts_lacks"], run["active"])
        and not found(s["stored_lacks"], run["stored"])
    )
    result = {"запись": write_ok}
    for variant, answer in run["answers"].items():
        result[variant] = len(found(s["answer_has"], answer)) == len(s["answer_has"]) and not found(s["answer_lacks"], answer)
    return result


def report(scenarios: list[dict], by_id: dict) -> None:
    """Таблица «сценарий × стадия» и pass^K по сценариям с интервалом Уилсона: интервал по сценариям, а не по прогонам."""
    columns = ("запись", *VARIANTS)
    print(f"{'сценарий':10} {'категория':11} " + " ".join(f"{c:>12}" for c in columns))
    for s in scenarios:
        cells = [f"{sum(r[c] for r in by_id[s['id']])}/{K}" for c in columns]
        print(f"{s['id']:10} {s['category']:11} " + " ".join(f"{c:>12}" for c in cells))
    print(f"\npass^{K} по сценариям, 95% интервал Уилсона:")
    for c in columns:
        hits = sum(all(r[c] for r in by_id[s["id"]]) for s in scenarios)
        low, high = wilson(hits, len(scenarios))
        print(f"  {c:12} {hits:>2}/{len(scenarios)}  [{low:.2f}, {high:.2f}]")


def block2_measure():
    scenarios = load_scenarios()
    jobs = [(s, r) for s in scenarios for r in range(K)]
    with ThreadPoolExecutor(8) as pool:
        runs = list(pool.map(lambda job: play(*job), jobs))
    by_id = {}
    for (s, _), run in zip(jobs, runs):
        by_id.setdefault(s["id"], []).append(check(s, run))
    report(scenarios, by_id)
    print(f"\nцена прогона ${sum(r['cost'] for r in runs):.3f}, финальный ответ за 1000 штук:",
          ", ".join(f"{v} ${1000 * sum(r['final_cost'][v] for r in runs) / len(runs):.2f}" for v in VARIANTS))
    rejected = [reason for r in runs for reason in r["rejected"]]
    print(f"отклонено кодом операций: {len(rejected)}", *rejected[:5], sep="\n  ")
    for (s, _), run in zip(jobs, runs):  # провалы самой памяти: сырьё для разбора ошибок
        result = check(s, run)
        if not (result["запись"] and result["память"]):
            print(f"\n{s['id']} память: {run['active']!r}\n  ответ с памятью: {run['answers']['память'][:200]}")

if __name__ == "__main__":
    blocks = {"1": ("Память между сессиями", block1_between_sessions), "2": ("Замер памяти", block2_measure)}
    for number in sys.argv[1:] or blocks:
        title, run_block = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run_block()
