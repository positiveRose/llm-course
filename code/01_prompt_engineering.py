"""Урок 01. Промпт-инжиниринг: базовые техники, провайдеры, reasoning-модели.

Три блока урока. Блоки 1 и 3 идут через Responses API, блок 2 через Chat Completions:
этот формат понимают совместимые эндпоинты других провайдеров. Всем блокам нужен OPENAI_API_KEY;
второй половине блока 2 нужен ещё ANTHROPIC_API_KEY, без него она пропускается. Другие SDK не нужны.
Запуск: python 01_prompt_engineering.py (все блоки) или python 01_prompt_engineering.py 2 (только второй)
"""

import os


def block1_router_prompts():
    from collections import Counter

    from common import MODEL, client, cost_usd, load_tickets

    PROMPT_VAGUE = "Определите категорию обращения: billing, tech, general."

    PROMPT_CLEAR = """# Роль
Вы маршрутизатор обращений в поддержку интернет-магазина.

# Категории
- billing: деньги. Оплата, списания, возврат денег, рассрочка, промокоды, бонусы, документы для бухгалтерии.
- tech: неисправность или настройка купленного товара, сбои сайта и приложения.
- general: всё остальное. Доставка, самовывоз, часы работы, личный кабинет, отмена заказа.

# Правила
- Обращение может касаться нескольких тем: выбирайте ту, из-за которой человек написал.
- Отвечайте одним словом в нижнем регистре, без пояснений и знаков препинания."""

    EXAMPLES = """

# Примеры
<ticket>Промокод на скидку не сработал при оплате, деньги списали полностью.</ticket>
<category>billing</category>
<ticket>Можно ли продлить хранение заказа в пункте выдачи ещё на несколько дней?</ticket>
<category>general</category>
<ticket>Стиральная машина не сливает воду и показывает ошибку E21.</ticket>
<category>tech</category>"""

    def classify(system_prompt, text):
        resp = client.responses.create(
            model=MODEL,
            instructions=system_prompt,
            input=text,
            max_output_tokens=20,
            reasoning={"effort": "none"},
            temperature=0,  # измерение должно повторяться от прогона к прогону
            store=False,
        )
        answer = resp.output_text.strip().lower()
        return answer, cost_usd(resp.usage, MODEL)

    def evaluate(system_prompt, tickets):
        """Прогоняет размеченные тикеты: модель видит только text, метка category остаётся у нас."""
        rows, cost = [], 0.0
        for t in tickets:
            answer, c = classify(system_prompt, t["text"])
            rows.append({"id": t["id"], "expected": t["category"], "answer": answer})
            cost += c
        return rows, cost

    def report(name, rows, cost):
        hits = [r for r in rows if r["answer"] == r["expected"]]  # точное совпадение: формат тоже часть задачи
        print(f"{name:13} accuracy {len(hits)}/{len(rows)} = {len(hits) / len(rows):.0%}  ${cost:.5f}")
        total, right = Counter(r["expected"] for r in rows), Counter(r["expected"] for r in hits)
        print("    по классам:", ", ".join(f"{c} {right[c]}/{total[c]}" for c in sorted(total)))
        for r in [r for r in rows if r["answer"] != r["expected"]][:3]:  # первые ошибки текстом
            print(f"    {r['id']} ожидали {r['expected']}, получили {r['answer'][:30]!r}")

    def diff(before, after):
        """Сравнивает два прогона по id тикета: что правка починила и что сломала."""
        was_right = {r["id"]: r["answer"] == r["expected"] for r in before}
        fixed = [r["id"] for r in after if r["answer"] == r["expected"] and not was_right[r["id"]]]
        broken = [r["id"] for r in after if r["answer"] != r["expected"] and was_right[r["id"]]]
        print(f"    починено: {fixed or 'ничего'}, сломано: {broken or 'ничего'}")

    tickets = load_tickets()[:20]
    prompts = {
        "расплывчатый": PROMPT_VAGUE,
        "структурный": PROMPT_CLEAR,
        "с примерами": PROMPT_CLEAR + EXAMPLES,
    }
    results = {name: evaluate(prompt, tickets) for name, prompt in prompts.items()}
    for name, (rows, cost) in results.items():
        report(name, rows, cost)
    print("примеры против структурного:")
    diff(results["структурный"][0], results["с примерами"][0])


def block2_two_providers():
    from openai import OpenAI

    from common import CLAUDE_MODEL, MODEL, client, load_kb, log_usage

    def build_system() -> str:
        docs = "\n".join(
            f'<document id="{d["id"]}" title="{d["title"]}">\n{d["text"]}\n</document>' for d in load_kb()
        )
        return f"""# Роль
Вы ассистент поддержки интернет-магазина. Отвечаете клиентам в чате.

# Правила
- Отвечайте только по документам из раздела «База знаний». Если ответа там нет, скажите об этом и предложите написать в поддержку.
- Сроки и суммы называйте точно так, как они записаны в документах.
- Ответ: два-четыре предложения без заголовков и списков.

# База знаний
{docs}"""

    SYSTEM = build_system()
    QUESTION = "Купил чайник 10 дней назад, не подошёл по цвету. Могу вернуть, и когда придут деньги?"

    # Здесь Chat Completions, а не Responses: этот формат понимают совместимые эндпоинты других провайдеров
    def ask(api, model, label, **extra):
        resp = api.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}],
            max_completion_tokens=300,
            **extra,
        )
        print(f"{label}:", resp.choices[0].message.content)
        log_usage(resp, model, label)

    ask(client, MODEL, "openai", reasoning_effort="none")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY не задан, второй провайдер пропущен")
    else:
        # Тот же SDK и тот же запрос: меняются только адрес, ключ и имя модели
        claude = OpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com/v1/", max_retries=5, timeout=120
        )
        # reasoning_effort этот эндпоинт молча игнорирует, размышление выключается параметром провайдера
        ask(claude, CLAUDE_MODEL, "claude", extra_body={"thinking": {"type": "disabled"}})


def block3_cot_vs_effort():
    import time

    from common import MODEL, client, cost_usd, load_kb, usage_dict

    KB = "\n\n".join(f"## {d['title']}\n{d['text']}" for d in load_kb())
    BASE = "Вы ассистент поддержки интернет-магазина. Отвечайте кратко и только по базе знаний ниже.\n\n" + KB
    COT = BASE + (
        "\n\nПрежде чем ответить, рассуждайте шаг за шагом: выпишите правила из базы знаний, "
        "которые относятся к вопросу. Итог дайте в конце отдельной строкой после слова «Ответ:»."
    )

    # Вопрос и подстроки, которые обязаны быть в ответе. Проверка грубая, строгую сделаем в уроке 08
    CASES = [
        ("Купил чайник 10 дней назад, не подошёл по цвету. Могу вернуть?", ["14"]),
        ("Наушники сломались через 8 месяцев после покупки, гарантия ещё действует?", ["12"]),
        ("Заказ на 4500 рублей, курьером по городу. Сколько стоит доставка?", ["300"]),
        ("Кабель из комплекта перестал заряжать через 8 месяцев, гарантия на него есть?", ["6 мес"]),
        # Два вопроса на три-четыре факта базы знаний с арифметикой: только здесь модели есть что обдумывать
        ("Заказал холодильник за 29 500 рублей и чехол за 500 рублей одним заказом, холодильник поднимут "
         "на пятый этаж без лифта. Сколько я доплачу за подъём, сколько бонусов начислят за товары "
         "и сколько из них получится списать при следующем заказе на 2000 рублей?", ["750", "900", "600"]),
        ("Холодильник за 30 000 рублей подняли на третий этаж без лифта. Хочу обменять его на модель "
         "за 30 800: сколько доплачу за новый холодильник, сколько снова заплачу за подъём "
         "и кто оплачивает вывоз старого?", ["800", "450"]),
    ]

    def final(answer):
        """Итог после последнего «Ответ:». Факт, который модель выписала в черновик, в зачёт не идёт."""
        return answer.rsplit("Ответ:", 1)[-1]

    def ask(system, question, effort):
        t0 = time.perf_counter()
        resp = client.responses.create(
            model=MODEL,
            instructions=system,
            input=question,
            max_output_tokens=1500,
            reasoning={"effort": effort},  # temperature здесь нет: при включённом effort API отвечает на него ошибкой 400
            store=False,
        )
        return resp, time.perf_counter() - t0

    runs = {
        "none, прямой": (BASE, "none"),
        "none, CoT": (COT, "none"),
        "low, прямой": (BASE, "low"),
        "low, CoT": (COT, "low"),
    }
    print(f"{'режим':13} {'верно':>5} {'out':>5} {'reasoning':>9} {'$':>9} {'сек':>5}")
    for name, (system, effort) in runs.items():
        correct = out = reasoning = 0
        cost = seconds = 0.0
        for question, facts in CASES:
            resp, dt = ask(system, question, effort)
            answer = final(resp.output_text)
            correct += all(fact in answer for fact in facts)
            u = usage_dict(resp.usage)
            out += u["output"]
            reasoning += u["reasoning"]
            cost += cost_usd(resp.usage, MODEL)
            seconds += dt
        print(f"{name:13} {correct:>3}/{len(CASES)} {out:>5} {reasoning:>9} {cost:>9.5f} {seconds:>5.1f}")


if __name__ == "__main__":
    import sys

    blocks = {
        "1": ("Три промпта для роутера", block1_router_prompts),
        "2": ("Один запрос к двум провайдерам", block2_two_providers),
        "3": ("CoT в промпте против reasoning_effort", block3_cot_vs_effort),
    }
    for number in sys.argv[1:] or blocks:  # без аргументов идут все блоки, «python 01_prompt_engineering.py 2» запускает только второй
        title, run = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
