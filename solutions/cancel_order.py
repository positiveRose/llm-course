"""Эталон к чекпоинту 1, урок 03: отмена заказа.

Открывайте после того, как сдали своё решение. Если к уроку 04 и дальше своего cancel_order нет,
положите этот файл в свой проект рядом с support_tools.py и импортируйте после него:

    import support_tools
    import cancel_order  # регистрирует инструмент в support_tools.TOOL_IMPL и support_tools.TOOLS

Правила проверяет код, как в create_refund: модель только выбирает, когда вызвать.
"""

import support_tools as st

NOT_YET_SHIPPED = {"placed", "assembled"}
ON_THE_WAY = {"handed_to_delivery", "in_transit", "ready_for_pickup"}


def cancel_order(order_id: int, reason: str) -> dict:
    """Отменяет заказ до передачи в доставку. Оплаченное возвращается целиком."""
    order = st.ORDERS.get(order_id)
    if order is None:
        return {"error": f"Заказ {order_id} не найден. Уточните номер у клиента."}
    if order.get("custom_build") and order.get("build_started"):  # сборку проверяем раньше статуса: у 54120 статус placed
        return {"error": "Сборка на заказ уже началась, отменить заказ нельзя."}
    status = order["status"]
    if status == "cancelled":
        return {"error": "Заказ уже отменён."}
    if status in ON_THE_WAY:
        return {"error": "Заказ уже передан в доставку, отменить его нельзя. Предложите отказаться от заказа при получении: "
                         "деньги вернутся в обычные сроки."}
    if status == "delivered":
        return {"error": f"Заказ получен, отмена невозможна. В течение {st.RETURN_DAYS} дней после получения оформите "
                         "возврат через create_refund."}
    if status not in NOT_YET_SHIPPED:
        return {"error": f"Неизвестный статус заказа: {status}."}
    order["status"] = "cancelled"  # меняется только рабочая копия ORDERS, data/orders.json не трогаем
    result = {"order_id": order_id, "status": "cancelled", "status_label": st.STATUS_LABELS["cancelled"]}
    if order["paid_rub"] > 0:
        refund = {"refund_id": f"R-{len(st.REFUNDS) + 1:04d}", "order_id": order_id,
                  "amount_rub": order["paid_rub"], "reason": f"Отмена заказа: {reason}"}
        st.REFUNDS.append(refund)
        result["refund"] = {**refund, "eta": "до 5 рабочих дней"}
    else:
        result["refund"] = None  # оплата при получении: возвращать нечего
    return result


st.TOOL_IMPL["cancel_order"] = cancel_order
if not any(t["name"] == "cancel_order" for t in st.TOOLS):
    st.TOOLS.append(st._tool(
        "cancel_order",
        "Отменяет заказ, который ещё не передан в доставку, и возвращает оплаченную сумму. Необратимое действие: "
        "вызывайте, только когда клиент прямо просит отменить заказ и заказ уже проверен через get_order. "
        "Для полученного заказа не вызывайте, там нужен create_refund. Может вернуть ошибку, если отмена невозможна.",
        {
            "order_id": {"type": "integer", "description": "Номер заказа"},
            "reason": {"type": "string", "description": "Причина отмены словами клиента, одной фразой"},
        },
    ))
