"""Урок 02. Structured outputs: схемы, извлечение, accuracy по полям.

Три блока урока, всем нужен OPENAI_API_KEY.
Запуск: python 02_structured_outputs.py (все блоки) или python 02_structured_outputs.py 2 (только второй)
"""

import os


def block1_router_by_schema():
    from typing import Literal

    from pydantic import BaseModel

    from common import MODEL, client, load_tickets, stop_reason

    CATEGORIES = ("billing", "tech", "general")
    PROMPT_VAGUE = "Определите категорию обращения: billing, tech, general."  # тот самый из урока 01

    class Route(BaseModel):
        category: Literal["billing", "tech", "general"]

    def route_by_text(text):
        resp = client.responses.create(
            model=MODEL,
            instructions=PROMPT_VAGUE,
            input=text,
            max_output_tokens=20,
            reasoning={"effort": "none"},
            temperature=0,
            store=False,
        )
        return resp.output_text.strip().lower()

    def route_by_schema(text):
        resp = client.responses.parse(
            model=MODEL,
            instructions=PROMPT_VAGUE,
            input=text,
            text_format=Route,  # SDK превратит модель в strict JSON Schema
            max_output_tokens=20,
            reasoning={"effort": "none"},
            temperature=0,
            store=False,
        )
        if stop_reason(resp) == "refusal":  # отказ не соответствует схеме и приходит отдельной частью ответа
            return "refusal"
        return resp.output_parsed.category  # уже не строка для парсинга, а поле объекта Route

    tickets = load_tickets()[:20]
    for name, route in {"текстом": route_by_text, "по схеме": route_by_schema}.items():
        answers = [route(t["text"]) for t in tickets]
        invalid = sum(a not in CATEGORIES for a in answers)  # форма: ответ вне списка
        correct = sum(a == t["category"] for a, t in zip(answers, tickets))  # истина: совпал с меткой
        print(f"{name:9} вне списка {invalid:>2}/{len(tickets)}, accuracy {correct:>2}/{len(tickets)}")


def block2_extract_fields():
    from pydantic import BaseModel, Field

    from common import MODEL, client, load_ticket_fields, stop_reason

    class TicketFields(BaseModel):
        order_id: int | None = Field(description="Номер заказа из текста. null, если номера в тексте нет")
        product: str | None = Field(description="Товар, о котором пишет клиент, одним-двумя словами. null, если товара нет")
        days_ago: int | None = Field(
            description="Сколько дней назад произошло событие, о котором пишет клиент: покупка, получение, возврат, отмена. "
            "Неделя это 7, месяц это 30. null, если клиент не сказал, сколько дней назад это было"
        )

    def extract(text) -> TicketFields:
        resp = client.responses.parse(
            model=MODEL,
            instructions="Извлеките поля из обращения в поддержку. Ничего не додумывайте.",
            input=text,
            text_format=TicketFields,
            max_output_tokens=200,
            reasoning={"effort": "none"},
            temperature=0,
            store=False,
        )
        stop = stop_reason(resp)
        if stop:  # при отказе output_parsed равен None, и без проверки код упал бы на первом же обращении к полю
            raise RuntimeError(f"ответ без полей: {stop}")
        return resp.output_parsed

    def same(field, gold, got) -> bool:
        if gold is None or got is None:
            return gold is got
        if field == "product":  # строку сравниваем по ключевому слову, числа точно
            return gold in got.lower()
        return gold == got

    def outcome(field, gold, got) -> str:
        if same(field, gold, got):
            return "верно"
        if gold is None:
            return "выдумано"  # значение там, где в тексте его нет
        if got is None:
            return "пропущено"  # null там, где значение есть
        return "неверно"  # значение есть и в эталоне, и в ответе, но они разные

    def report(name, rows, show_errors):
        stats = {f: dict.fromkeys(("верно", "неверно", "выдумано", "пропущено"), 0) for f in TicketFields.model_fields}
        errors = []
        for row in rows:
            got = extract(row["text"])
            for field, s in stats.items():
                gold_value, got_value = row["gold"][field], getattr(got, field)
                kind = outcome(field, gold_value, got_value)
                s[kind] += 1  # четыре исхода не пересекаются, поэтому строка таблицы в сумме даёт число тикетов
                if kind != "верно":
                    errors.append(f"  {row['id']} {field}: {kind}, эталон {gold_value!r}, ответ {got_value!r} | {row['text'][:60]}")
        print(f"\n{name}, тикетов {len(rows)}")
        print(f"{'поле':10} {'верно':>6} {'неверно':>8} {'выдумано':>9} {'пропущено':>10}")
        for field, s in stats.items():
            print(f"{field:10} {s['верно']:>6} {s['неверно']:>8} {s['выдумано']:>9} {s['пропущено']:>10}")
        if show_errors:
            print("\n".join(errors) or "  ошибок нет")

    rows = load_ticket_fields()
    report("рабочая часть", rows[:14], show_errors=True)  # на эти ошибки смотрим и по ним правим описания полей
    report("отложенная часть", rows[14:], show_errors=False)  # сюда не заглядываем: только числа после правок


def block3_issues_from_letter():
    from typing import Literal

    from pydantic import BaseModel, Field

    from common import MODEL, client, load_letters, log_usage, stop_reason

    QUOTE = "Дословная цитата из письма, на которой основан этот пункт"

    class Refund(BaseModel):
        kind: Literal["refund"]
        quote: str = Field(description=QUOTE)  # цитата идёт до вывода: сначала опора, потом решение
        order_id: int | None

    class TechProblem(BaseModel):
        kind: Literal["tech"]
        quote: str = Field(description=QUOTE)
        product: str
        symptom: str

    class Question(BaseModel):
        kind: Literal["question"]
        quote: str = Field(description=QUOTE)
        topic: str

    class Letter(BaseModel):  # корень схемы обязан быть объектом, поэтому union завёрнут в поле
        issues: list[Refund | TechProblem | Question]

    SYSTEM = (
        "Разберите письмо клиента на отдельные проблемы, по одной записи на каждую, в порядке появления в письме. "
        "refund: клиент хочет вернуть деньги или товар. tech: товар неисправен. "
        "question: вопрос о правилах магазина. Приветствия, жалобы на поддержку и цитаты прошлых ответов пропускайте."
    )

    for letter in load_letters():
        resp = client.responses.parse(
            model=MODEL, instructions=SYSTEM, input=letter["text"], text_format=Letter, store=False,
            max_output_tokens=1500,  # список длиннее одного поля: лимит с запасом
            reasoning={"effort": "none"}, temperature=0,  # извлечение, как в блоках 1 и 2
        )
        stop = stop_reason(resp)
        if stop:  # письмо клиента может оказаться и таким: отказ вместо разбора
            print(f"{letter['id']}: разбора нет, {stop}")
            continue
        issues = resp.output_parsed.issues
        kinds_ok = sorted(i.kind for i in issues) == sorted(g["kind"] for g in letter["issues"])
        orders_ok = {i.order_id for i in issues if i.kind == "refund"} == {
            g["order_id"] for g in letter["issues"] if g["kind"] == "refund"
        }
        grounded = sum(i.quote in letter["text"] for i in issues)  # цитата обязана быть в письме дословно
        print(f"{letter['id']}: проблем {len(issues)} из {len(letter['issues'])}, типы {'ok' if kinds_ok else 'НЕТ'}, "
              f"заказы {'ok' if orders_ok else 'НЕТ'}, цитат дословно {grounded}/{len(issues)}")
        for i in issues:
            print(f"    {i.kind:9} {i.quote[:70]!r}")
        log_usage(resp, MODEL, letter["id"])


if __name__ == "__main__":
    import sys

    blocks = {
        "1": ("Роутер текстом и по схеме", block1_router_by_schema),
        "2": ("Извлечение полей и accuracy по полям", block2_extract_fields),
        "3": ("Список проблем из длинного письма", block3_issues_from_letter),
    }
    for number in sys.argv[1:] or blocks:  # без аргументов идут все блоки, «python 02_structured_outputs.py 2» запускает только второй
        title, run = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
