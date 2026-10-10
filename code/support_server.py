"""MCP-сервер ассистента поддержки: инструменты урока 03 за протоколом MCP. Урок 13.

Клиент поднимает его подпроцессом и говорит с ним по stdio: python support_server.py
По HTTP: python support_server.py http, адрес http://127.0.0.1:8931/mcp
В stdout сервер пишет только сообщения протокола. print() в инструменте сломал бы поток, логи идут в stderr.
"""

import sys
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

import support_tools

mcp = FastMCP("support", instructions="Инструменты поддержки интернет-магазина: заказы, возвраты, расчёты.")


def _unwrap(result: dict) -> dict:
    """support_tools сообщает об ошибке словарём {"error": ...}. В MCP ошибку инструмента помечает isError."""
    if "error" in result:
        raise ToolError(result["error"])  # FastMCP превратит исключение в результат с isError: true
    return result


@mcp.tool(annotations={"readOnlyHint": True})
def get_order(order_id: Annotated[int, Field(description="Номер заказа из обращения клиента")]) -> dict:
    """Возвращает заказ по номеру: состав, суммы, статус, дату получения и сколько дней прошло с получения.
    Вызывайте перед любым действием с заказом."""
    return _unwrap(support_tools.get_order(order_id))


@mcp.tool(annotations={"destructiveHint": True, "idempotentHint": False})
def create_refund(
    order_id: Annotated[int, Field(description="Номер заказа")],
    amount_rub: Annotated[int, Field(description="Сумма возврата в рублях, не больше оплаченной")],
    reason: Annotated[str, Field(description="Причина возврата одной фразой")],
) -> dict:
    """Оформляет возврат денег по заказу. Необратимое действие: вызывайте, только когда клиент прямо просит возврат
    и заказ уже проверен через get_order. Может вернуть ошибку, если возврат не положен."""
    return _unwrap(support_tools.create_refund(order_id, amount_rub, reason))


@mcp.tool(annotations={"readOnlyHint": True})
def calculator(
    expression: Annotated[str, Field(description="Выражение из чисел, скобок и + - * /, например '32990 * 0.03'")],
) -> dict:
    """Вычисляет арифметическое выражение. Используйте для любых расчётов с деньгами и процентами, не считайте в уме."""
    try:
        return support_tools.calculator(expression)
    except Exception as e:  # неверное выражение это ошибка инструмента, а не падение сервера
        raise ToolError(f"{type(e).__name__}: {e}")


@mcp.resource("support://refunds")
def refunds() -> list[dict]:
    """Оформленные возвраты. Это ресурс, а не инструмент: его читает приложение (проверки, панель оператора), не модель."""
    return support_tools.REFUNDS


if __name__ == "__main__":
    if sys.argv[1:] == ["http"]:
        mcp.run(transport="http", host="127.0.0.1", port=8931)
    else:
        mcp.run(show_banner=False)  # stdio по умолчанию
