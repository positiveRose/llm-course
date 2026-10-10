"""Урок 03. Воркфлоу и ReAct-агент без фреймворков.

Три блока урока, всем нужен OPENAI_API_KEY. Общие детали лежат рядом:
schemas.py (схемы и промпты из уроков 01–02) и support_tools.py (инструменты ассистента).
Все вызовы модели идут через Responses API, как во всём курсе.
Запуск: python 03_agents_and_workflows.py (все блоки) или python 03_agents_and_workflows.py 2 (только второй)
"""

import os


def block1_ticket_workflow():
    from concurrent.futures import ThreadPoolExecutor

    from common import MODEL, client, load_kb, load_tickets, stop_reason
    from schemas import EXTRACT_PROMPT, ROUTER_PROMPT, Route, Triage, Verdict

    KB = "\n\n".join(f"## {d['title']}\n{d['text']}" for d in load_kb())
    HANDLERS = {  # роутинг: у каждой категории свой узкий промпт
        "billing": "Вы специалист по оплатам и возвратам. Называйте сроки и суммы точно по базе знаний.",
        "tech": "Вы инженер техподдержки. Дайте шаги диагностики из базы знаний и скажите, когда случай гарантийный.",
        "general": "Вы консультант магазина. Отвечайте коротко и по базе знаний.",
    }
    JUDGE = "Проверьте ответ поддержки: все сроки, суммы и условия должны совпадать с базой знаний.\n\n" + KB

    def parse(system, user, schema):
        resp = client.responses.parse(
            model=MODEL,
            instructions=system,
            input=user,
            text_format=schema,
            store=False,  # ответ у провайдера не хранится, историю курс ведёт сам
            max_output_tokens=500,
            reasoning={"effort": "none"},
            temperature=0,  # этапы конвейера должны повторяться от прогона к прогону, урок 01
        )
        stop = stop_reason(resp)
        if stop:  # отказ приходит мимо схемы, урок 02; обрыв посреди JSON parse бросит ValidationError сам
            raise RuntimeError(stop)
        return resp.output_parsed

    def draft(ticket, category, feedback):
        user = ticket if not feedback else f"{ticket}\n\nЗамечания к прошлому ответу, учтите их:\n{feedback}"
        resp = client.responses.create(
            model=MODEL,
            instructions=f"{HANDLERS[category]} Два-четыре предложения.\n\n{KB}",
            input=user,
            store=False,
            max_output_tokens=400,
            reasoning={"effort": "none"},
        )
        stop = stop_reason(resp)
        if stop:  # обрыв или отказ: черновика нет, как в parse
            raise RuntimeError(stop)
        return resp.output_text

    def handle(ticket):
        with ThreadPoolExecutor() as pool:  # параллелизация: два независимых вызова идут одновременно
            route = pool.submit(parse, ROUTER_PROMPT, ticket, Route)
            fields = pool.submit(parse, EXTRACT_PROMPT, ticket, Triage)
        category, fields = route.result().category, fields.result()
        print(f"  категория {category}, поля {fields.model_dump()}")
        if category == "billing" and fields.needs_order and fields.order_id is None:  # гейт: обычный код между этапами
            return "Подскажите, пожалуйста, номер заказа, и мы сразу проверим оплату."
        feedback = None
        for attempt in (1, 2):  # генерация с проверкой, число кругов ограничено
            answer = draft(ticket, category, feedback)
            verdict = parse(JUDGE, f"Обращение: {ticket}\n\nОтвет поддержки: {answer}", Verdict)
            print(f"  круг {attempt}: passed={verdict.passed} {verdict.feedback[:80]}")
            if verdict.passed:
                break
            feedback = verdict.feedback
        return answer

    tickets = {t["id"]: t["text"] for t in load_tickets()}
    for ticket_id in ("T-055", "T-030", "T-008", "T-007"):
        print(f"{ticket_id}: {tickets[ticket_id]}")
        print("  ->", handle(tickets[ticket_id]))


def block2_one_tool_call():
    import json

    from common import MODEL, client
    from support_tools import get_order

    TOOLS = [{
        "type": "function",
        "name": "get_order",
        "description": "Возвращает заказ по номеру: состав, суммы, статус, дату получения.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "integer", "description": "Номер заказа из обращения клиента"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        "strict": True,  # тот же строгий режим, что в уроке 02, но для аргументов вызова
    }]
    SYSTEM = "Вы ассистент поддержки интернет-магазина. Данные о заказах берите только из инструментов."
    items = [{"role": "user", "content": "Заказ 51002 третий день «в пути», когда привезут?"}]

    resp = client.responses.create(
        model=MODEL, instructions=SYSTEM, input=items, tools=TOOLS, store=False, reasoning={"effort": "low"}
    )
    print("status:", resp.status, "| output_text:", repr(resp.output_text))  # completed, а текста нет
    for item in resp.output:  # элементы ответа: reasoning, message, function_call
        print(" ", item.to_dict() if item.type == "function_call" else item.type)

    items += [item.to_dict() for item in resp.output]  # весь output попадает в историю раньше результатов
    for call in [item for item in resp.output if item.type == "function_call"]:  # вызовов может быть несколько
        args = json.loads(call.arguments)  # arguments приходит строкой с JSON
        result = get_order(**args)  # функцию выполняет ваш код, а не модель
        items.append({"type": "function_call_output", "call_id": call.call_id,  # связь по call_id, а не по id
                      "output": json.dumps(result, ensure_ascii=False)})

    final = client.responses.create(
        model=MODEL, instructions=SYSTEM, input=items, tools=TOOLS, store=False, reasoning={"effort": "low"}
    )
    print("ответ:", final.output_text)


def block3_agent_loop():
    from common import MODEL, client, cost_usd, final_text, load_kb, stop_reason, usage_dict
    from support_tools import TODAY, TOOLS, reset_state, run_tool

    SYSTEM = (
        "# Роль\nВы ассистент поддержки интернет-магазина. Решаете обращение клиента до конца.\n\n"
        "# Правила\n"
        "- Правила магазина берите из базы знаний ниже, данные о заказах только из инструментов.\n"
        "- Если инструмент вернул ошибку, объясните клиенту причину и предложите, что можно сделать вместо этого.\n"
        "- Ответ клиенту: два-четыре предложения, с суммами и сроками.\n\n"
        "# База знаний\n" + "\n\n".join(f"## {d['title']}\n{d['text']}" for d in load_kb())
        + f"\n\n# Сегодня\n{TODAY.isoformat()}"  # изменчивое стоит в конце: всё, что выше, остаётся кэшируемым префиксом
    )

    def run_agent(query, items=None, max_iterations=6):
        items = [] if items is None else items  # свой список продолжает диалог
        items.append({"role": "user", "content": query})
        total = 0.0
        for step in range(1, max_iterations + 1):
            resp = client.responses.create(
                model=MODEL, instructions=SYSTEM, input=items, tools=TOOLS, store=False,
                max_output_tokens=2000, reasoning={"effort": "low"},
            )
            u = usage_dict(resp.usage)
            total += cost_usd(resp.usage, MODEL)
            print(f"  шаг {step}: вход {u['input']}, из кэша {u['cached']}, выход {u['output']}")
            print("    элементы:", [item.type for item in resp.output])
            if stop_reason(resp):  # обрыв по лимиту или отказ это не «задача решена»
                break  # недописанный шаг в историю не кладём: вызов без результата сломал бы следующий ход
            items += [item.to_dict() for item in resp.output]  # весь output обратно в input: история только дописывается
            calls = [item for item in resp.output if item.type == "function_call"]
            if not calls:  # вызовов нет: модель считает задачу решённой
                print(f"  итого ${total:.5f}")
                return final_text(resp)  # текст без промежуточных реплик модели
            for call in calls:
                result = run_tool(call.name, call.arguments)  # ошибка вернётся модели, а не уронит цикл
                print(f"    {call.name}({call.arguments}) -> {result[:90]}")
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
        return "Не получилось решить обращение, передаю его оператору."  # кончились шаги, лимит токенов или отказ

    reset_state()  # инструменты меняют рабочую копию заказов; сброс к эталону даёт каждому запуску одинаковый старт
    for query in (
        "Хочу вернуть чайник из заказа 49880, не подошёл по цвету. Оформите возврат денег.",
        "Верните деньги за ноутбук из заказа 51877, он мне не понравился.",
        "Заказ 53110. Если я верну чехол, сколько денег вернётся и сколько бонусов останется за пылесос?",
    ):
        print(query)
        print("  ->", run_agent(query))


if __name__ == "__main__":
    import sys

    blocks = {
        "1": ("Воркфлоу обработки тикета", block1_ticket_workflow),
        "2": ("Один вызов инструмента", block2_one_tool_call),
        "3": ("Цикл агента", block3_agent_loop),
    }
    for number in sys.argv[1:] or blocks:  # без аргументов идут все блоки, «python 03_agents_and_workflows.py 2» запускает только второй
        title, run = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
