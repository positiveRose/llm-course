"""Чекпоинт 5, урок 18: при трёх видах повтора возврат ровно один.

Сначала заполните четыре функции урока 18 в checks/adapter.py. Запуск: pytest checks/test_idempotent_refund.py
Заказ 53110 получен девять дней назад и оплачен на 33 980: возвраты по нему законны.
"""
import pytest

import adapter

# Все повторы проверяются на частичном возврате: чехол за 990 из заказа 53110 на 33 980.
# Полную сумму второй раз не пропустит и правило «не больше оплаченного», а частичную пропустит,
# если нет ключа исполнения или проверки «один возврат на заказ и сумму»: деньги уйдут дважды.
PART_ORDER, PART_AMOUNT = 53110, 990


@pytest.fixture(autouse=True)
def clean():
    try:
        adapter.reset()
    except NotImplementedError:
        pytest.skip("адаптер урока 18 не заполнен: допишите reset, refund, refunds_for и restart в checks/adapter.py")


def test_same_call_twice():
    first = adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл", session_id="s0", call_id="call_0")
    second = adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл", session_id="s0", call_id="call_0")
    assert "error" not in first, f"законный возврат отклонён: {first}"
    assert len(adapter.refunds_for(PART_ORDER)) == 1, "тот же вызов дважды провёл два возврата: деньги за чехол ушли дважды"
    assert second.get("refund_id") == first.get("refund_id"), f"повтор того же вызова должен вернуть прежний результат, а вернул {second}"


def test_same_call_after_crash():
    adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл", session_id="s1", call_id="call_1")
    adapter.restart()
    adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл", session_id="s1", call_id="call_1")
    assert len(adapter.refunds_for(PART_ORDER)) == 1, "после падения процесса тот же вызов провёл возврат второй раз"


def test_new_generation_same_request():
    adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл", session_id="s1", call_id="call_1")
    adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл", session_id="s1", call_id="call_2")
    assert len(adapter.refunds_for(PART_ORDER)) == 1, "новый вызов от новой генерации провёл второй возврат: call_id не защищает"


def test_client_resends_message():
    adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл", session_id="s1", call_id="call_1")
    adapter.restart()
    adapter.refund(PART_ORDER, PART_AMOUNT, "чехол не подошёл, пишу ещё раз", session_id="s2", call_id="call_7")
    assert len(adapter.refunds_for(PART_ORDER)) == 1, "повторная отправка сообщения после перезапуска провела второй возврат"


def test_second_item_is_still_allowed():
    """Защита не должна запрещать законное: у 53110 два товара, чехол за 990 и остальное."""
    a = adapter.refund(53110, 990, "чехол не подошёл", session_id="s3", call_id="call_1")
    b = adapter.refund(53110, 32990, "вернуть остальное", session_id="s3", call_id="call_2")
    assert "error" not in a and "error" not in b, f"второй возврат другого товара отклонён: {a} {b}"
    assert len(adapter.refunds_for(53110)) == 2
