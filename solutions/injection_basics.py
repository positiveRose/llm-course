"""Эталон к чекпоинту 3: минимальная защита урока 14a, правила 2 и 3.

Открывайте после того, как сдали своё решение. Положите файл в свой проект рядом с support_tools.py и подключите
в checks/adapter.py: в функциях урока 14a раскомментируйте строки «эталон».

Код взят из уроков 15 и 16 второй части и ужат до того, что нужно проверке checks/test_injection_basics.py.
Правило 1 (недоверенный текст приходит результатом инструмента, а не в instructions) живёт в цикле агента,
его здесь нет: его проверяют чтением кода на защите.
"""

import json
import re
from urllib.parse import urlparse

import support_tools as st

# --- Правило 2. Фильтр ответа ------------------------------------------------------------------

ALLOWED_HOSTS = {"shop.example"}  # хосты магазина точными именами: поддомены и похожие имена не проходят
REMOVED = "[ссылка удалена]"

MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^\s)>]*)>?[^)]*\)")  # ![подпись](адрес "заголовок")
MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*<?([^\s)>]*)>?[^)]*\)")  # [текст](адрес)
HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*=[^<>]*)?\s*/?>")  # <a href=…>, <img src=…>, </a>
TAIL = r"[^\s<>\"'()\[\]]*[^\s<>\"'()\[\].,;:!?]"  # адрес до пробела или скобки, без точки в конце предложения
URLISH = re.compile(rf"(?:[a-zA-Z][a-zA-Z0-9+.-]*:)?//{TAIL}|\bwww\.{TAIL}")  # со схемой, //хост без схемы, www.


def allowed(url: str) -> bool:
    """Адрес ведёт на хост из списка. urlparse отделяет userinfo и порт: у https://shop.example@evil.com хост evil.com."""
    if "\\" in url:  # браузер читает \ как /: https://evil.com\@shop.example ведёт на evil.com, а urlparse видит магазин
        return False
    try:
        parts = urlparse(url.strip())
    except ValueError:  # битый адрес
        return False
    if parts.username is not None:  # логин перед хостом легальной ссылке не нужен
        return False
    return parts.scheme in ("http", "https") and parts.hostname in ALLOWED_HOSTS  # сравнение целиком, не подстрока


def filter_answer(text: str) -> str:
    """Ответ модели перед отправкой клиенту: всё, что ведёт не на хост из списка, вырезается, текст вокруг остаётся."""
    text = MD_IMAGE.sub(lambda m: m[0] if allowed(m[2]) else REMOVED, text)
    text = MD_LINK.sub(lambda m: m[0] if allowed(m[2]) else m[1], text)  # у ссылки остаётся текст, адрес уходит
    text = HTML_TAG.sub("", text)  # виджет показывает markdown: HTML-теги не нужны ни с какими адресами
    return URLISH.sub(lambda m: m[0] if allowed(m[0]) else REMOVED, text)  # голые адреса и всё, что пропустили выше


# --- Правило 3. Клиент из сессии, суммы из кода ------------------------------------------------

OWNER = {  # чей заказ: в настоящей базе это поле customer_id у заказа
    53110: "u-101", 51002: "u-101", 49880: "u-102", 51877: "u-103", 48213: "u-104", 53422: "u-105",
    52876: "u-106", 50990: "u-107", 54010: "u-108", 54077: "u-109", 54120: "u-110",
}


def order_of(customer: str, order_id: int) -> dict:
    """Заказ, если он принадлежит клиенту. Чужой отвечает так же, как несуществующий: перебор номеров ничего не даст."""
    if OWNER.get(order_id) != customer:
        return {"error": f"Заказ {order_id} не найден среди заказов клиента. Уточните номер."}
    return st.get_order(order_id)


def refund_plan(customer: str, name: str, args: dict) -> dict:
    """Что оформит возврат: заказ, сумма и причина. Сумму считает код, модель её не называет."""
    order = order_of(customer, args["order_id"])
    if "error" in order:
        return order
    if name == "refund_overpayment":  # двойное списание: оплачено больше, чем стоят товары
        what, amount = "лишнее списание", order["paid_rub"] - sum(i["price_rub"] for i in order["items"])
    else:
        what = args["item"]
        amount = next((i["price_rub"] for i in order["items"] if i["name"] == what), 0)
        if not amount:
            return {"error": f"В заказе нет товара «{what}». Товары: {', '.join(i['name'] for i in order['items'])}."}
    if amount <= 0:
        return {"error": "Переплаты по заказу нет: оплачено столько, сколько стоят товары."}
    if any(r["order_id"] == order["order_id"] and r["reason"].startswith(what) for r in st.REFUNDS):
        return {"error": f"Возврат «{what}» по этому заказу уже оформлен."}
    return {"order_id": order["order_id"], "amount_rub": amount, "reason": f"{what}: {args['reason']}"}


def session_tools(customer: str) -> dict:
    """Реализации инструментов одной сессии. Клиент приходит из сессии через замыкание, в схемах его нет."""

    def refund(name: str, **args) -> dict:
        plan = refund_plan(customer, name, args)
        return plan if "error" in plan else st.create_refund(**plan)  # срок, статус и потолок проверяет код урока 03

    return {"get_order": lambda order_id: order_of(customer, order_id), "calculator": st.calculator,
            "refund_item": lambda **args: refund("refund_item", **args),
            "refund_overpayment": lambda **args: refund("refund_overpayment", **args)}


ORDER_ID = {"type": "integer", "description": "Номер заказа"}
REASON = {"type": "string", "description": "Причина возврата одной фразой"}
SESSION_TOOLS = [  # описания для модели вместо TOOLS: поля клиента и суммы в них нет
    st.TOOLS[0],  # get_order: описание прежнее, поменялась только реализация
    st._tool("refund_item", "Возврат денег за один товар из заказа, сумму система берёт из цены товара. Необратимое "
             "действие: вызывайте, только когда клиент прямо просит вернуть деньги за этот товар и заказ проверен "
             "через get_order.",
             {"order_id": ORDER_ID, "item": {"type": "string", "description": "Название товара точно как в заказе"},
              "reason": REASON}),
    st._tool("refund_overpayment", "Возвращает ошибочное повторное списание: разницу между оплаченной суммой и "
             "стоимостью товаров. Вызывайте, когда клиент сообщает о двойном списании и get_order его подтверждает.",
             {"order_id": ORDER_ID, "reason": REASON}),
    st.TOOLS[2],  # calculator
]


def run_session_tool(customer: str, name: str, args: dict) -> str:
    """run_tool урока 03 с реализациями одной сессии: ошибка тоже результат, ответ JSON-строкой."""
    tools = session_tools(customer)
    try:
        result = tools[name](**args) if name in tools else {"error": f"Неизвестный инструмент: {name}"}
    except Exception as e:  # неверные аргументы, в том числе поле, которого нет в схеме
        result = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(result, ensure_ascii=False)
