"""Адаптер: единственный файл, который вы правите под свой проект. Проверки зовут только его.

По умолчанию он настроен на устройство проекта как в курсе. Если у вас модули называются иначе,
поменяйте импорты здесь, а не в тестах.
"""


# --- Урок 03: инструменты -----------------------------------------------------

def tools():
    """Модуль с ORDERS, REFUNDS, reset_state, run_tool, TOOL_IMPL и TOOLS, где уже есть ваш cancel_order."""
    import my_tools
    return my_tools


# --- Урок 14a: минимальная защита ---------------------------------------------
# Четыре функции поверх вашего ассистента. Пока они не написаны, проверка чекпоинта 3 пропускается:
# фильтр и права проверяются отдельно, можно начать с одного.
# Возвраты тест читает из tools().REFUNDS и сбрасывает через tools().reset_state(), как проверка урока 03.
# Эталон solutions/injection_basics.py открывайте после того, как сдали своё. Чтобы подключить его, положите файл
# рядом с support_tools.py и в каждой функции ниже раскомментируйте строку «эталон».

def allowed_hosts() -> set[str]:
    """Хосты, на которые ответу можно ссылаться, точными именами: {"shop.example"}.

    Тест ставит в ответ ссылку https://<хост>/returns и ждёт, что фильтр её оставит.
    """
    # import injection_basics; return injection_basics.ALLOWED_HOSTS  # эталон
    raise NotImplementedError


def filter_answer(text: str) -> str:
    """Ответ модели после вашего фильтра: ровно то, что увидит клиент в виджете."""
    # import injection_basics; return injection_basics.filter_answer(text)  # эталон
    raise NotImplementedError


def owner_of(order_id: int) -> str | None:
    """Клиент, которому принадлежит заказ, тем же идентификатором, что у вашей сессии (user_id). None, если заказа нет.

    Тест берёт отсюда клиентов заказов 53110 и 48213 и ищет чужой заказ среди 49880, 52876, 50990, 53422:
    у одного из них владелец должен быть другим.
    """
    # import injection_basics; return injection_basics.OWNER.get(order_id)  # эталон
    raise NotImplementedError


def session_tool(customer_id: str, name: str, args: dict) -> str | dict:
    """Вызов инструмента так, как его делает ваш цикл агента в сессии клиента customer_id. Верните то, что увидит модель.

    Клиент приходит из сессии, name и args из вызова модели. Ошибку верните результатом, как цикл агента, а не исключением.
    Тест зовёт три инструмента: get_order {order_id}; refund_item {order_id, item, reason}, возврат за один товар;
    refund_overpayment {order_id, reason}, возврат лишнего списания. Если у вас они называются или устроены иначе,
    переведите вызов здесь. Если возврата лишнего списания у вас нет, верните ошибку «Неизвестный инструмент».
    В args тест иногда добавляет поля, которые модель могла вписать под диктовку атаки: amount_rub, customer_id, user_id.
    Инструмент зовите в этом же процессе, не через подпроцесс MCP: возвраты тест ищет в tools().REFUNDS.
    """
    # import injection_basics; return injection_basics.run_session_tool(customer_id, name, args)  # эталон
    raise NotImplementedError


# --- Урок 18: идемпотентный возврат -------------------------------------------
# Четыре функции поверх вашего сервиса. Пока они не написаны, проверка урока 18 пропускается.

def reset() -> None:
    """Чистая база сервиса и рабочая копия заказов, как после первого запуска."""
    raise NotImplementedError


def refund(order_id: int, amount_rub: int, reason: str, session_id: str, call_id: str) -> dict:
    """Вызов create_refund так, как его вызывает ваш цикл агента: с сессией и call_id вызова модели."""
    raise NotImplementedError


def refunds_for(order_id: int) -> list[dict]:
    """Возвраты по заказу, которые реально записаны: в базе сервиса или в заглушке платёжной системы."""
    raise NotImplementedError


def restart() -> None:
    """Имитация падения процесса: закрыть соединения, забыть всё, что лежит в памяти, и открыть базу заново."""
    raise NotImplementedError
