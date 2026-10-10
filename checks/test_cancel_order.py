"""Чекпоинт 1, урок 03: cancel_order меняет рабочую копию строго по таблице правил.

Запуск из корня проекта: pytest checks/test_cancel_order.py
"""
import hashlib
import json
from pathlib import Path

import pytest

import adapter

ORDERS_FILE = Path("data/orders.json")


@pytest.fixture(scope="module")
def st():
    return adapter.tools()


@pytest.fixture(scope="module", autouse=True)
def fixture_untouched():
    """Эталон data/orders.json не должен меняться ни при каком вызове."""
    before = hashlib.sha256(ORDERS_FILE.read_bytes()).hexdigest()
    yield
    assert hashlib.sha256(ORDERS_FILE.read_bytes()).hexdigest() == before, "data/orders.json изменился: инструмент правит эталон, а не ORDERS"


def call(st, order_id):
    return json.loads(st.run_tool("cancel_order", json.dumps({"order_id": order_id, "reason": "проверка"})))


def test_registered_in_three_places(st):
    assert "cancel_order" in st.TOOL_IMPL, "нет строки в TOOL_IMPL: модель получит «Неизвестный инструмент»"
    desc = next((t for t in st.TOOLS if t["name"] == "cancel_order"), None)
    assert desc is not None, "нет описания в TOOLS: модель не узнает об инструменте"
    assert desc.get("strict") is True
    assert set(desc["parameters"]["required"]) == {"order_id", "reason"}


@pytest.mark.parametrize("order_id, paid", [(54010, 12990), (54077, 0)])
def test_cancels_before_shipping(st, order_id, paid):
    st.reset_state()
    result = call(st, order_id)
    assert "error" not in result, f"{order_id} можно отменить, а инструмент вернул ошибку: {result}"
    assert st.ORDERS[order_id]["status"] == "cancelled"
    refunds = [r for r in st.REFUNDS if r["order_id"] == order_id]
    if paid:
        assert len(refunds) == 1 and refunds[0]["amount_rub"] == paid, f"ожидался один возврат {paid}: {refunds}"
        assert refunds[0]["refund_id"] in json.dumps(result, ensure_ascii=False), "номер возврата не отдан модели"
    else:
        assert refunds == [], "оплата при получении: возвращать нечего"


@pytest.mark.parametrize("order_id", [51002, 49880, 54120])
def test_refuses_and_changes_nothing(st, order_id):
    st.reset_state()
    status = st.ORDERS[order_id]["status"]
    result = call(st, order_id)
    assert "error" in result, f"{order_id} отменять нельзя, а инструмент не вернул ошибку: {result}"
    assert st.ORDERS[order_id]["status"] == status, "при отказе статус не должен меняться"
    assert st.REFUNDS == [], "при отказе возвратов быть не должно"


def test_second_cancel_is_refused(st):
    st.reset_state()
    call(st, 54010)
    result = call(st, 54010)
    assert "error" in result, "повторная отмена должна вернуть ошибку «уже отменён»"
    assert len([r for r in st.REFUNDS if r["order_id"] == 54010]) == 1, "повторная отмена вернула деньги второй раз"


def test_unknown_order(st):
    st.reset_state()
    assert "error" in call(st, 99999)
