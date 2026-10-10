"""Урок 00. Как устроена LLM для инженера: токены, контекст, сэмплинг, цена.

Три блока урока. Блок 1 работает без ключа, блокам 2 и 3 нужен OPENAI_API_KEY.
Вызовы модели идут через Responses API: client.responses.create.
Запуск: python 00_llm_basics.py (все блоки) или python 00_llm_basics.py 2 (только второй)
"""

import os


def block1_tokens():
    import tiktoken
    from common import ENCODING, MODEL, PRICES, load_kb

    enc = tiktoken.get_encoding(ENCODING)

    samples = {
        "русский": "Списали деньги дважды за заказ 48213, верните лишнее.",
        "english": "You charged me twice for order 48213, please refund the extra.",
        "json": '{"order_id": 48213, "category": "billing", "refund": true}',
    }
    for name, text in samples.items():
        ids = enc.encode(text)
        print(f"{name:8} {len(text):3} символов -> {len(ids):3} токенов, {len(text) / len(ids):.1f} симв/токен")
        print("        ", [enc.decode([i]) for i in ids])  # на какие куски разрезан текст

    # База знаний целиком: сколько токенов уйдёт, если класть её в каждый запрос
    kb_text = "\n\n".join(d["text"] for d in load_kb())
    kb_tokens = len(enc.encode(kb_text))
    price_in = PRICES[MODEL][0]
    print(f"\nбаза знаний: {len(kb_text)} символов, {kb_tokens} токенов")
    print(f"в каждом запросе на {MODEL}: ${kb_tokens * price_in / 1_000_000:.6f} за вход")
    print(f"на 100 000 запросов в месяц: ${kb_tokens * price_in / 1_000_000 * 100_000:.2f}")

    # Одна фраза на пяти языках, новый словарь против предыдущего: таблица из раздела 2.2
    old_enc = tiktoken.get_encoding("cl100k_base")
    sentences = {
        "английский": "You can return an item within 14 days of delivery if it has not been used, and we will refund it.",
        "русский": "Товар можно вернуть в течение 14 дней после доставки, если им не пользовались, и мы вернём деньги.",
        "немецкий": "Sie können einen Artikel innerhalb von 14 Tagen nach der Lieferung zurückgeben, "
                    "wenn er nicht benutzt wurde, und wir erstatten den Betrag.",
        "китайский": "商品未经使用的话,可以在送达后14天内退货,我们会退款。",
        "хинди": "यदि सामान का उपयोग नहीं किया गया है, तो आप डिलीवरी के 14 दिनों के भीतर उसे वापस कर सकते हैं, और हम पैसे लौटा देंगे।",
    }
    print(f"\n{'язык':11} {'символов':>8} {'байт':>5} {ENCODING:>11} {'cl100k_base':>11}")
    for name, text in sentences.items():
        print(f"{name:11} {len(text):>8} {len(text.encode()):>5} {len(enc.encode(text)):>11} {len(old_enc.encode(text)):>11}")


def block2_read_usage():
    from common import MODEL, client, load_kb, log_usage

    SYSTEM = (
        "Вы ассистент поддержки интернет-магазина. Отвечайте кратко и только по базе знаний.\n\n"
        + "\n\n".join(f"## {d['title']}\n{d['text']}" for d in load_kb())
    )
    QUESTION = "Холодильник привезли 12 дней назад на 4 этаж без лифта. Он не подошёл по размеру: могу вернуть и сколько я заплатил за подъём?"

    for effort in ("none", "high"):  # вопрос с проверкой срока и расчётом: модели есть о чём подумать
        resp = client.responses.create(
            model=MODEL,
            instructions=SYSTEM,  # системный промпт: правила и база знаний
            input=QUESTION,  # строка это одно сообщение с ролью user
            max_output_tokens=2000,  # включает reasoning-токены, не только видимый ответ
            reasoning={"effort": effort},
            store=False,  # не хранить ответ у провайдера: историю курс ведёт сам
        )
        print(f"--- effort={effort}")
        print(resp.output_text)
        print("status:", resp.status, resp.incomplete_details)
        print("output:", [item.type for item in resp.output])  # элементы ответа: reasoning, message
        print(resp.usage.model_dump())  # сырой объект: все поля, включая вложенные details
        log_usage(resp, MODEL, effort)


def block3_effort_and_sampling():
    import time

    from common import MODEL, client, cost_usd, load_tickets, usage_dict

    tickets = load_tickets()[:5]
    SYSTEM = "Определите категорию обращения в поддержку. Ответьте одним словом: billing, tech или general."

    def classify(text, **params):
        t0 = time.perf_counter()
        resp = client.responses.create(
            model=MODEL,
            instructions=SYSTEM,
            input=text,
            max_output_tokens=1000,
            store=False,
            **params,
        )
        return resp, time.perf_counter() - t0

    runs = {
        "none": {"reasoning": {"effort": "none"}},
        "none, t=0": {"reasoning": {"effort": "none"}, "temperature": 0},
        "none, t=1.5": {"reasoning": {"effort": "none"}, "temperature": 1.5},
        "low": {"reasoning": {"effort": "low"}},
        "high": {"reasoning": {"effort": "high"}},
        "high, t=1.5": {"reasoning": {"effort": "high"}, "temperature": 1.5},
    }
    print(f"{'режим':13} {'верно':>5} {'out':>5} {'reasoning':>9} {'$':>9} {'сек':>5}")
    for name, params in runs.items():
        correct = out = reasoning = 0
        cost = seconds = 0.0
        try:
            for t in tickets:
                resp, dt = classify(t["text"], **params)
                u = usage_dict(resp.usage)
                answer = resp.output_text.strip().lower()  # пустая строка, если текста нет
                correct += t["category"] in answer
                out += u["output"]
                reasoning += u["reasoning"]
                cost += cost_usd(resp.usage, MODEL)
                seconds += dt
        except Exception as e:  # например, HTTP 400: temperature при включённом reasoning
            print(f"{name:13} ошибка: {str(e)[:90]}")
            continue
        print(f"{name:13} {correct:>3}/{len(tickets)} {out:>5} {reasoning:>9} {cost:>9.5f} {seconds:>5.1f}")


if __name__ == "__main__":
    import sys

    blocks = {
        "1": ("Токены (без ключа)", block1_tokens),
        "2": ("Читаем usage", block2_read_usage),
        "3": ("Reasoning effort и temperature", block3_effort_and_sampling),
    }
    for number in sys.argv[1:] or blocks:  # без аргументов идут все блоки, «python 00_llm_basics.py 2» запускает только второй
        title, run = blocks[number]
        if number not in ['1'] and not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
