import ast
import copy
import json
import operator
from datetime import date

from common import load_orders

_today_iso, _FIXTURE = load_orders()
TODAY = date.fromisoformat(_today_iso)
RETURN_DAYS = 14
ORDERS: dict[int, dict] = {}
REFUNDS: list[dict] = []
STATUS_LABELS = {
    "placed": "Оформлен", "assembled": "Собран", "handed_to_delivery": "Передан в доставку", "in_transit": "В пути",
    "ready_for_pickup": "Ожидает в пункте выдачи", "delivered": "Доставлен", "cancelled": "Отменён",
}


def reset_state() -> None:
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(_FIXTURE))
    REFUNDS.clear()


reset_state()


# --- Инструменты -------------------------------------------------------------


def get_order(order_id: int) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found"}
    days_from = None
    if order["status"] == "delivered":
        days_from = (TODAY - date.fromisoformat(order["delivered_at"])).days
    return {"order_id": order_id, "status": order['status'], "days_from": days_from}

def cancel_order(order_id: int, reason: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Заказ {order_id} не найден. Уточните номер у клиента."}
    if order.get("custom_build") and order.get("build_started"):
        return {"error": "Индивидуальная сборка уже начата, отмена невозможна."}
    if order["status"] == "cancelled":
        return {"error": "Заказ уже отменён."}
    if order["status"] == "delivered":
        return {"error": "Заказ уже доставлен. Предложите клиенту оформить возврат."}
    if order["status"] in ("handed_to_delivery", "in_transit", "ready_for_pickup"):
        return {"error": "Заказ уже передан в доставку. Предложите клиенту отказаться от заказа при получении."}
    order["status"] = "cancelled"
    result = {"status": "cancelled", "order_id": order_id}
    if order["paid_rub"] > 0:
        refund = {"refund_id": f"R-{len(REFUNDS) + 1:04d}", "order_id": order_id, "amount_rub": order["paid_rub"], "reason": reason}
        REFUNDS.append(refund)
        result["refund_id"] = refund["refund_id"]
    return result


def change_address(order_id: int, address: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found"}
    if order["status"] not in ("placed", "assembled"):
        return {"error": f"order {order_id} status is not 'placed' or 'assembled'"}
    order["address"] = address
    return {"order_id": order_id, "status": order["status"]}


def create_refund(order_id: int, amount_rub: int, reason: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found"}
    if order["status"] != "delivered":
        return {"error": f"order {order_id} status is not 'delivered'"}
    days_from = (TODAY - date.fromisoformat(order["delivered_at"])).days
    if days_from > RETURN_DAYS:
        return {"error": f" order {order_id} cannot be refunded because days_from > RETURN_DAYS"}
    if amount_rub <= 0 or amount_rub > order["paid_rub"]:
        return {"error": f" order {order_id} cannot be refunded because amount_rub <= 0"}
    refund = {"refund_id": f"R-{len(REFUNDS) + 1:04d}", "order_id": order_id, "amount_rub": order["paid_rub"], "reason": reason}
    REFUNDS.append(refund)
    return {**refund, "status": "created"}






# --- Регистрация -------------------------------------------------------------

TOOL_IMPL = {
    "get_order": get_order,
    "cancel_order": cancel_order,
    "change_address": change_address,
    "create_refund": create_refund,
}


def run_tool(name: str, arguments_json: str) -> str:
    try:
        result = TOOL_IMPL[name](**json.loads(arguments_json))
    except KeyError:
        result = {"error": f"Неизвестный инструмент: {name}"}
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(result, ensure_ascii=False)


# --- Описания для модели -----------------------------------------------------

def _tool(name, description, properties):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


TOOLS = [
    _tool(
        "get_order",
        "Возвращает заказ по номеру: состав, суммы, статус, дату получения. Вызывайте перед любым действием с заказом.",
        {"order_id": {"type": "integer", "description": "Номер заказа"}},
    ),
    _tool(
        "cancel_order",
        "Отменяет заказ. Только до передачи в доставку. Возвращает деньги если оплачен.",
        {
            "order_id": {"type": "integer", "description": "Номер заказа"},
            "reason": {"type": "string", "description": "Причина отмены"},
        },
    ),
    _tool(
        "change_address",
        "Меняет адрес доставки. Только пока заказ не передан в доставку.",
        {
            "order_id": {"type": "integer", "description": "Номер заказа"},
            "address": {"type": "string", "description": "Новый адрес доставки"},
        },
    ),
    _tool(
        "create_refund",
        "Оформляет возврат денег. Только после получения заказа, не позже 14 дней.",
        {
            "order_id": {"type": "integer", "description": "Номер заказа"},
            "amount_rub": {"type": "integer", "description": "Сумма возврата в рублях"},
            "reason": {"type": "string", "description": "Причина возврата"},
        },
    ),
]
