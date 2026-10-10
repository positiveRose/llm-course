"""MCP-сервер «как REST» для эксперимента урока 13: те же заказы и те же правила, но интерфейс повторяет эндпоинты API.

GET /orders/{id}, /orders/{id}/items, /payments, /shipment, /refunds и POST /refunds, по инструменту на эндпоинт.
Ответы сырые (коды статусов, даты ISO, служебные id), ошибки кодами. Правила возврата проверяет тот же support_tools.
"""

import json
import re
import uuid
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

import support_tools

mcp = FastMCP("shop-api", instructions="REST API интернет-магазина: заказы, позиции, платежи, доставка, возвраты.")
OrderId = Annotated[int, Field(description="Идентификатор заказа")]
CODES = [  # текст ошибки support_tools -> HTTP-статус и машинный код, как у настоящего API
    ("не найден", 404, "ORDER_NOT_FOUND"), ("ещё не получен", 409, "ORDER_NOT_DELIVERED"),
    ("С получения прошло", 422, "REFUND_WINDOW_EXPIRED"), ("Сумма вне допустимого", 422, "AMOUNT_EXCEEDS_REFUNDABLE"),
]


def _order(order_id: int) -> dict:
    order = support_tools.ORDERS.get(order_id)
    if order is None:
        raise ToolError(json.dumps({"status": 404, "code": "ORDER_NOT_FOUND"}))
    return order


@mcp.tool(annotations={"readOnlyHint": True})
def get_order(order_id: OrderId) -> dict:
    """GET /orders/{id}. Заказ: код статуса, дата доставки (ISO 8601, UTC), id клиента, оплаченная сумма в рублях.
    Позиции, платежи, доставка и возвраты запрашиваются отдельными методами."""
    o = _order(order_id)
    return {"id": order_id, "status": o["status"], "customer_id": str(uuid.uuid5(uuid.NAMESPACE_OID, str(order_id))),
            "delivered_at": o["delivered_at"] and o["delivered_at"] + "T12:00:00Z", "paid_rub": o["paid_rub"]}


@mcp.tool(annotations={"readOnlyHint": True})
def list_order_items(order_id: OrderId) -> list[dict]:
    """GET /orders/{id}/items. Позиции заказа: id позиции, название, количество, цена в рублях."""
    return [{"item_id": f"it_{order_id}_{i}", "name": it["name"], "qty": 1, "price_rub": it["price_rub"]}
            for i, it in enumerate(_order(order_id)["items"], 1)]


@mcp.tool(annotations={"readOnlyHint": True})
def list_payments(order_id: OrderId) -> list[dict]:
    """GET /orders/{id}/payments. Списания по заказу: id платежа, сумма в рублях, способ, статус."""
    o = _order(order_id)
    double = re.search(r"дважды по (\d+)", o.get("note", ""))  # в эталоне двойное списание описано заметкой
    amounts = [int(double[1])] * 2 if double else ([o["paid_rub"]] if o["paid_rub"] else [])
    return [{"payment_id": f"pay_{order_id}_{i}", "amount_rub": a, "method": o["payment"], "status": "captured"}
            for i, a in enumerate(amounts, 1)]


@mcp.tool(annotations={"readOnlyHint": True})
def get_shipment(order_id: OrderId) -> dict:
    """GET /orders/{id}/shipment. Доставка: код статуса, даты передачи перевозчику и обещанной доставки (ISO 8601)."""
    o = _order(order_id)
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", o.get("note", ""))
    return {"status": o["status"], "handed_at": dates[0] if dates else None, "promised_at": dates[1] if len(dates) > 1 else None,
            "delivered_at": o["delivered_at"]}


@mcp.tool(annotations={"readOnlyHint": True})
def list_refunds(order_id: OrderId) -> list[dict]:
    """GET /orders/{id}/refunds. Оформленные возвраты по заказу: id, сумма в рублях, статус."""
    return [{"id": r["refund_id"], "amount_rub": r["amount_rub"], "status": "pending"}
            for r in support_tools.REFUNDS if r["order_id"] == order_id]


@mcp.tool(annotations={"destructiveHint": True, "idempotentHint": False})
def create_refund(
    order_id: OrderId,
    amount_rub: Annotated[int, Field(description="Сумма возврата в рублях")],
    reason: Annotated[str, Field(description="Причина возврата")],
) -> dict:
    """POST /refunds. Создаёт возврат денег по заказу. Ошибки: 404, 409, 422 с машинным кодом."""
    result = support_tools.create_refund(order_id, amount_rub, reason)  # те же правила, что у сервера под задачу
    if "error" in result:
        status, code = next((s, c) for text, s, c in CODES if text in result["error"])
        raise ToolError(json.dumps({"status": status, "code": code}))
    return {"id": result["refund_id"], "status": "pending"}


@mcp.tool(annotations={"readOnlyHint": True})
def calculator(expression: Annotated[str, Field(description="Арифметическое выражение из чисел, скобок и + - * /")]) -> dict:
    """Вычисляет арифметическое выражение."""
    try:
        return support_tools.calculator(expression)
    except Exception as e:
        raise ToolError(json.dumps({"status": 400, "code": "INVALID_EXPRESSION"})) from e


@mcp.resource("support://refunds")
def refunds() -> list[dict]:
    """Оформленные возвраты для проверок, как у сервера под задачу."""
    return support_tools.REFUNDS


if __name__ == "__main__":
    mcp.run(show_banner=False)
