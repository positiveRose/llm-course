"""Инструменты ассистента поддержки: реализации, описания для модели и безопасный запуск.

Общий модуль: им пользуются агент из урока 03 и все следующие уроки.
Внешних сервисов нет. Файл data/orders.json это эталон, его никто не меняет. Инструменты читают и меняют
рабочую копию в памяти: словарь ORDERS и список REFUNDS. reset_state() возвращает копию к эталону.
"""

import ast
import copy
import json
import operator
from datetime import date

from common import load_orders

_today_iso, _FIXTURE = load_orders()  # эталон из data/orders.json: только для чтения
TODAY = date.fromisoformat(_today_iso)  # «сегодня» зафиксировано в данных, чтобы прогоны повторялись
RETURN_DAYS = 14  # политика возврата из базы знаний
ORDERS: dict[int, dict] = {}  # рабочая копия заказов: её читают и меняют инструменты
REFUNDS: list[dict] = []  # рабочая копия платёжной системы: оформленные возвраты
STATUS_LABELS = {  # коды статусов и их названия из документа «Статусы заказа» базы знаний
    "placed": "Оформлен", "assembled": "Собран", "handed_to_delivery": "Передан в доставку", "in_transit": "В пути",
    "ready_for_pickup": "Ожидает в пункте выдачи", "delivered": "Доставлен", "cancelled": "Отменён",
}


def reset_state() -> None:
    """Возвращает рабочую копию к эталону: заказы как в файле, возвратов нет. Вызывайте перед каждым прогоном."""
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(_FIXTURE))  # глубокая копия: правка заказа не дотянется до эталона
    REFUNDS.clear()


reset_state()


# --- Реализации -------------------------------------------------------------


def get_order(order_id: int) -> dict:
    """Заказ по номеру. Добавляет название статуса и days_since_delivery, чтобы модель не считала даты сама."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Заказ {order_id} не найден. Уточните номер у клиента."}
    days = (TODAY - date.fromisoformat(order["delivered_at"])).days if order["delivered_at"] else None
    return {**order, "status_label": STATUS_LABELS[order["status"]], "days_since_delivery": days}

def cancel_order(order_id, reason) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Заказ {order_id} не найден. Уточните номер у клиента."}
    if order["status"] not in ("placed", "assembled"):
        return {"error": f"Заказ {order_id} не может быть отменен."}
    if order.get("custom_build") or order.get("build_started"):
        return {"error": f"Заказ {order_id} не может быть отменен"}
    order["status"] = "cancelled"
    result = {"status": "cancelled", "order_id": order_id}
    if order["paid_rub"] > 0:
        refund = {"refund_id": f"R-{len(REFUNDS) + 1:04d}", "order_id": order_id, "amount_rub": order["paid_rub"], "reason": reason}
        REFUNDS.append(refund)
        result["refund_id"] = refund["refund_id"]
    return result

def create_refund(order_id: int, amount_rub: int, reason: str) -> dict:
    """Оформляет возврат денег. Правила магазина проверяет код, а не модель."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Заказ {order_id} не найден."}
    if order["status"] != "delivered":
        return {"error": "Заказ ещё не получен, возврат оформляется после получения. До передачи в доставку заказ можно "
                         "отменить, после неё клиент может отказаться от заказа при получении."}
    days = (TODAY - date.fromisoformat(order["delivered_at"])).days
    if days > RETURN_DAYS:  # политика возврата проверяется здесь, что бы ни решила модель
        return {"error": f"С получения прошло {days} дней, возврат возможен в течение {RETURN_DAYS}. Предложите гарантийное обращение."}
    already = sum(r["amount_rub"] for r in REFUNDS if r["order_id"] == order_id)
    if amount_rub <= 0 or amount_rub + already > order["paid_rub"]:
        return {"error": f"Сумма вне допустимого: оплачено {order['paid_rub']}, уже возвращено {already}."}
    refund = {"refund_id": f"R-{len(REFUNDS) + 1:04d}", "order_id": order_id, "amount_rub": amount_rub, "reason": reason}
    REFUNDS.append(refund)
    return {**refund, "status": "created", "eta": "до 5 рабочих дней после проверки товара"}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression: str) -> dict:
    """Арифметика без eval: выражение разбирается в дерево, разрешены только числа и + - * /."""

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("разрешены только числа, скобки и + - * /")

    return {"result": round(ev(ast.parse(expression, mode="eval").body), 2)}


# Новый инструмент регистрируется в трёх местах: функция выше, строка в TOOL_IMPL, описание в TOOLS ниже.
TOOL_IMPL = {"get_order": get_order, "create_refund": create_refund, "calculator": calculator, "cancel_order": cancel_order}


def run_tool(name: str, arguments_json: str) -> str:
    """Выполняет вызов модели и всегда возвращает JSON-строку: ошибка это тоже результат, а не исключение.

    Строка уходит в поле output элемента function_call_output: API принимает там строку любого формата.
    """
    try:
        result = TOOL_IMPL[name](**json.loads(arguments_json))
    except KeyError:
        result = {"error": f"Неизвестный инструмент: {name}"}
    except Exception as e:  # неверные аргументы, деление на ноль, что угодно
        result = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(result, ensure_ascii=False)


# --- Описания для модели ----------------------------------------------------


def _tool(name: str, description: str, properties: dict) -> dict:
    """Описание инструмента в формате Chat Completions. Модель видит только его: имя, description и схему, но не код."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),  # строгий режим: все поля обязательны
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


TOOLS = [  # порядок фиксирован: описания инструментов входят в кэшируемый префикс
    _tool(
        "get_order",
        "Возвращает заказ по номеру: состав, суммы, статус, дату получения и сколько дней прошло с получения. "
        "Вызывайте перед любым действием с заказом.",
        {"order_id": {"type": "integer", "description": "Номер заказа из обращения клиента"}},
    ),

    _tool(
        "create_refund",
        "Оформляет возврат денег по заказу. Необратимое действие: вызывайте, только когда клиент прямо просит возврат "
        "и заказ уже проверен через get_order. Может вернуть ошибку, если возврат не положен.",
        {
            "order_id": {"type": "integer", "description": "Номер заказа"},
            "amount_rub": {"type": "integer", "description": "Сумма возврата в рублях, не больше оплаченной"},
            "reason": {"type": "string", "description": "Причина возврата одной фразой"},
        },
    ),
    _tool(
        "calculator",
        "Вычисляет арифметическое выражение. Используйте для любых расчётов с деньгами и процентами, не считайте в уме.",
        {"expression": {"type": "string", "description": "Выражение из чисел, скобок и + - * /, например '32990 * 0.03'"}},
    ),
    _tool(
        "cancel_order",
        "Отменяет заказ клиента. Можно отменить только до передачи в доставку. Возвращает деньги, если заказ был оплачен.",
        {
            "order_id": {"type": "integer", "description": "Номер заказа из обращения клиента"},
            "reason": {"type": "string", "description": "Причина отмены одной фразой"},
        },
    ),
]
