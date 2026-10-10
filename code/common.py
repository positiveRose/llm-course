"""Общий код курса: клиент, константы моделей, цены, учёт usage, разбор ответа и данные проекта.

Импортируется из кода каждого урока: `from common import client, MODEL, log_usage`.
Имена моделей меняются каждые несколько месяцев; в коде уроков только константы отсюда.
Все вызовы модели идут через Responses API. Учёт usage понимает и ответ Chat Completions: им пользуется блок 2 урока 01.
"""

import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

# --- Модели (на сентябрь 2026) --------------------------------------------

MODEL = "deepseek-chat"  # дешёвая модель, по умолчанию везде
STRONG_MODEL = "gpt-5.6-terra"  # средняя, только где важен контраст с дешёвой
EMBED_MODEL = "text-embedding-3-small"  # эмбеддинги, с урока 06
CLAUDE_MODEL = "claude-sonnet-5"  # второй провайдер через OpenAI-совместимый эндпоинт, урок 01
ENCODING = "o200k_base"  # токенизатор всего семейства GPT-5.x в tiktoken

# Цены в долларах за 1M токенов: (вход, кэшированный вход, выход), на сентябрь 2026.
PRICES = {
    MODEL: (0.27, 0.07, 1.10),
    STRONG_MODEL: (2.00, 0.20, 12.00),
    EMBED_MODEL: (0.02, 0.0, 0.0),
    CLAUDE_MODEL: (2.00, 0.20, 10.00),
}
CACHE_WRITE = 1.25  # запись в кэш у GPT-5.6 стоит 1.25 цены входа, чтение из кэша 0.1 (на сентябрь 2026)

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY не задан: вызовы модели упадут, локальные блоки работают", file=sys.stderr)
_KEY = os.environ.get("OPENAI_API_KEY") or "not-set"  # пустая строка это тоже «ключа нет»: SDK без значения падает на импорте
# max_retries: SDK сам повторяет 429, 5xx и таймауты с растущей паузой (по умолчанию 2 раза, для прогонов циклом мало).
# timeout: сколько секунд ждать следующий кусок ответа (урок 18), по умолчанию у SDK 600, и зависший вызов держит весь прогон.
client = OpenAI(api_key=_KEY, base_url="https://api.deepseek.com", max_retries=5, timeout=120)
aclient = AsyncOpenAI(api_key=_KEY, base_url="https://api.deepseek.com", max_retries=5, timeout=120)  # асинхронный клиент, с урока 04


# --- Учёт токенов и стоимости ----------------------------------------------


def usage_dict(usage) -> dict:
    """Пять чисел из usage, которые нужны для стоимости: input, cached, cache_write, output, reasoning.

    Понимает оба API. У Responses поля называются input_tokens и output_tokens, у Chat Completions (блок 2 урока 01)
    prompt_tokens и completion_tokens. У совместимых эндпоинтов других провайдеров вложенные details могут быть
    пустыми: тогда cached и reasoning равны 0.
    """
    if usage is None:  # ответ без usage, например оборванный стрим: считаем нулём, а не падаем
        return {"input": 0, "cached": 0, "cache_write": 0, "output": 0, "reasoning": 0}
    if hasattr(usage, "input_tokens"):  # Responses API
        total_in, total_out = usage.input_tokens, usage.output_tokens
        details_in = getattr(usage, "input_tokens_details", None)
        details_out = getattr(usage, "output_tokens_details", None)
    else:  # Chat Completions
        total_in, total_out = usage.prompt_tokens, usage.completion_tokens
        details_in = getattr(usage, "prompt_tokens_details", None)
        details_out = getattr(usage, "completion_tokens_details", None)
    return {
        "input": total_in,
        "cached": getattr(details_in, "cached_tokens", 0) or 0,
        "cache_write": getattr(details_in, "cache_write_tokens", 0) or 0,  # есть только у Responses
        "output": total_out,  # уже включает reasoning
        "reasoning": getattr(details_out, "reasoning_tokens", 0) or 0,
    }


def cost_usd(usage, model: str) -> float:
    """Стоимость одного ответа по прайсу PRICES. Вход делится на три части: из кэша, записанный в кэш и обычный."""
    price_in, price_cached, price_out = PRICES[model]
    u = usage_dict(usage)
    ordinary = u["input"] - u["cached"] - u["cache_write"]
    cost_in = ordinary * price_in + u["cached"] * price_cached + u["cache_write"] * price_in * CACHE_WRITE
    return (cost_in + u["output"] * price_out) / 1_000_000


def log_usage(resp, model: str, label: str = "") -> dict:
    """Печатает поля usage одной строкой и возвращает словарь с ними и стоимостью."""
    u = usage_dict(resp.usage)
    u["cost_usd"] = round(cost_usd(resp.usage, model), 6)
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}{model}: in={u['input']} (cached {u['cached']}) "
        f"out={u['output']} (reasoning {u['reasoning']}) cost=${u['cost_usd']:.6f}"
    )
    return u


# --- Разбор ответа Responses API --------------------------------------------
# Ответ это список элементов resp.output: reasoning (зашифрованное рассуждение), message (текст или отказ),
# function_call (просьба вызвать инструмент). finish_reason из Chat Completions здесь нет, вместо него resp.status.


def stop_reason(resp) -> str | None:
    """Почему ответ нельзя считать решением задачи: "length", "refusal" или None, если ответ полный."""
    if resp.status == "incomplete":  # генерация оборвалась, причина в incomplete_details
        reason = getattr(resp.incomplete_details, "reason", None)
        return "length" if reason == "max_output_tokens" else "refusal"  # иначе content_filter: ответ заблокирован
    for item in resp.output:
        if item.type == "message" and any(part.type == "refusal" for part in item.content or []):
            return "refusal"  # отказ приходит отдельной частью сообщения, а не текстом
    return None


def final_text(resp) -> str:
    """Текст ответа без промежуточных реплик (phase="commentary"). resp.output_text склеивает все сообщения подряд."""
    return "".join(
        part.text
        for item in resp.output
        if item.type == "message" and item.phase != "commentary"
        for part in item.content or []
        if part.type == "output_text"
    )


# --- Данные проекта: ассистент поддержки интернет-магазина -----------------
# data/kb/*.md: 27 документов базы знаний (возврат 14 дней, гарантия 12 месяцев,
# доставка, оплата, поддержка 9:00–21:00, инструкции к товарам).
# data/tickets.jsonl: 60 обращений с полем category из billing / tech / general.
# data/tickets_fields.jsonl: эталон полей для 20 тикетов (order_id, product, days_ago), урок 02.
# data/letters.jsonl: 3 длинных письма с несколькими проблемами и эталоном issues, урок 02; refunds и covers, урок 12.
# data/orders.json: 11 фейковых заказов в разных статусах и зафиксированная дата «сегодня», урок 03.
# data/kb_long/*.md: 6 длинных документов (инструкции и регламенты), урок 06.
# data/rag_questions.jsonl: 40 вопросов с эталонными документами, из них 3 без ответа в базе, урок 06.
# data/answers_labeled.jsonl: 120 ответов ассистента по найденным фрагментам с ручной разметкой pass/fail, урок 08.
# data/agent_scenarios.jsonl: 8 сценариев для агента с ожидаемыми возвратами, уроки 10 и 13.
# data/team_scenarios.jsonl: 4 сценария сверх урока 10 (техника, доставка, смешанный) с полем route, урок 12.
# data/refund_proposals.jsonl: 40 предложенных возвратов, половина с внесёнными ошибками, урок 12.

DATA_DIR = Path(__file__).parent / "data"


def load_kb() -> list[dict]:
    """Документы базы знаний: id, title (первая строка), text (без заголовка)."""
    docs = []
    for path in sorted((DATA_DIR / "kb").glob("*.md")):
        title, _, body = path.read_text(encoding="utf-8").partition("\n\n")
        docs.append({"id": path.stem, "title": title.lstrip("# ").strip(), "text": body.strip()})
    return docs


def load_kb_long() -> list[dict]:
    """Длинные документы базы знаний: инструкции и регламенты по несколько страниц, с урока 06."""
    docs = []
    for path in sorted((DATA_DIR / "kb_long").glob("*.md")):
        title, _, body = path.read_text(encoding="utf-8").partition("\n\n")
        docs.append({"id": path.stem, "title": title.lstrip("# ").strip(), "text": body.strip()})
    return docs


def load_tickets() -> list[dict]:
    """Тикеты: id, category, text."""
    with (DATA_DIR / "tickets.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_jsonl(name: str) -> list[dict]:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ticket_fields() -> list[dict]:
    """20 тикетов с эталоном полей: id, text, gold = {order_id, product, days_ago}."""
    texts = {t["id"]: t["text"] for t in load_tickets()}
    rows = []
    for gold in _load_jsonl("tickets_fields.jsonl"):
        ticket_id = gold.pop("id")
        rows.append({"id": ticket_id, "text": texts[ticket_id], "gold": gold})
    return rows


def load_letters() -> list[dict]:
    """Длинные письма: id, text, issues = [{kind, order_id}, ...] в порядке появления в письме.

    С урока 12 ещё refunds (ожидаемые возвраты) и covers: по группе слов на каждую проблему, которую не проверить по возвратам; в ответе должна быть хотя бы одна строка группы.
    """
    return _load_jsonl("letters.jsonl")


def load_rag_questions() -> list[dict]:
    """Вопросы с эталоном: id, question, docs (список подходящих документов; пустой, если ответа в базе нет)."""
    return _load_jsonl("rag_questions.jsonl")


def load_labeled_answers() -> list[dict]:
    """Ответы ассистента с разметкой: id, question, chunks (что видел ассистент), answer, label (pass/fail), mode, note."""
    return _load_jsonl("answers_labeled.jsonl")


def load_agent_scenarios() -> list[dict]:
    """Сценарии для агента: id, query, refunds (ожидаемые пары [order_id, amount_rub]), answer_has, critical."""
    return _load_jsonl("agent_scenarios.jsonl")


def load_team_scenarios() -> list[dict]:
    """Сценарии урока 10 и ещё четыре, у каждого route: какой специалист должен вести разговор, урок 12."""
    return _load_jsonl("agent_scenarios.jsonl") + _load_jsonl("team_scenarios.jsonl")


def load_refund_proposals() -> list[dict]:
    """Предложенные возвраты: id, query, proposal {order_id, amount_rub, reason}, ok, fault (тип внесённой ошибки), урок 12."""
    return _load_jsonl("refund_proposals.jsonl")


def load_orders() -> tuple[str, dict[int, dict]]:
    """Фейковая база заказов: (дата «сегодня» в ISO, словарь заказов по номеру)."""
    raw = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))
    return raw["today"], {o["order_id"]: o for o in raw["orders"]}
