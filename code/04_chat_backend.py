"""Урок 04. Агент как сервис: сессии, история, стриминг, async.

Блоки запускаются по одному: python 04_chat_backend.py 1, затем 2.
Блок 3 это сервер: python 04_chat_backend.py serve, запросы к нему шлются из второго терминала.
Рядом лежат store.py (SQLite: сессии, истории, расход, рабочая копия заказов) и support_tools.py (инструменты).
Вызовы модели идут через Chat Completions API: история как messages, системный промпт первым сообщением.
Нужны OPENAI_API_KEY и пакеты fastapi, uvicorn.
"""

import asyncio
import json
import os
import sys
import time


def block1_session_history():
    import store
    import support_tools
    from common import MODEL, client, usage_dict
    from schemas import agent_system
    from support_tools import TODAY, TOOLS, run_tool

    SYSTEM = agent_system(TODAY.isoformat())

    def chat(session_id, text, max_iterations=6):
        history = store.load_history(session_id)
        # системный промпт первым, потом история, потом новое сообщение
        messages = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": text}]
        turn = [{"role": "user", "content": text}]  # только новые сообщения этого хода — пишем в базу
        answer = None
        for _ in range(max_iterations):
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, max_tokens=2000,
            )
            u = usage_dict(resp.usage)
            print(f"    вход {u['input']}, из кэша {u['cached']}, выход {u['output']}")
            msg = resp.choices[0].message
            finish = resp.choices[0].finish_reason

            # собираем dict ассистента для истории
            msg_dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(msg_dict)
            turn.append(msg_dict)

            if finish == "stop" or not msg.tool_calls:
                answer = msg.content or ""
                break

            for tc in msg.tool_calls:
                result = run_tool(tc.function.name, tc.function.arguments)
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result}
                messages.append(tool_msg)
                turn.append(tool_msg)

        if answer is None:
            answer = "Не получилось решить обращение, передаю его оператору."
            turn.append({"role": "assistant", "content": answer})
        store.append_messages(session_id, turn)
        store.save_working_copy()
        return answer

    store.open_db("support.db")
    store.reset_db()
    session_id = store.create_session("user-1")
    for text in ("Хочу вернуть чайник из заказа 49880, не подошёл по цвету. Оформите возврат денег.",
                 "А когда придут деньги?"):
        print(f"> {text}")
        print("  ->", chat(session_id, text))

    support_tools.reset_state()
    store.open_db("support.db")
    print("после перезапуска: элементов в истории", len(store.load_history(session_id)), "| возвраты", support_tools.REFUNDS)
    print("> Напомните, на какую сумму оформлен возврат?")
    print("  ->", chat(session_id, "Напомните, на какую сумму оформлен возврат?"))


# --- Блок 2. Асинхронный агент со стримингом. Две функции ниже нужны и блоку 3 ------------------


async def run_tools(tool_calls):
    """Выполняет все вызовы шага одновременно. tool_calls — объекты SDK из message.tool_calls."""
    from support_tools import run_tool

    results = await asyncio.gather(*(
        asyncio.to_thread(run_tool, tc.function.name, tc.function.arguments) for tc in tool_calls
    ))
    return [{"role": "tool", "tool_call_id": tc.id, "content": r} for tc, r in zip(tool_calls, results)]


def _to_db_format(step):
    """Chat Completions → Responses API формат для хранения в БД."""
    result = []
    for msg in step:
        if msg.get("role") == "tool":
            result.append({"type": "function_call_output", "call_id": msg["tool_call_id"], "output": msg["content"]})
        elif msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                result.append({"type": "function_call", "call_id": tc["id"], "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]})
        else:
            result.append(msg)
    return result


def _from_db_format(items):
    """Responses API формат → Chat Completions для отправки в модель."""
    result = []
    i = 0
    while i < len(items):
        item = items[i]
        if item.get("type") == "function_call":
            tool_calls = []
            while i < len(items) and items[i].get("type") == "function_call":
                fc = items[i]
                tool_calls.append({"id": fc["call_id"], "type": "function", "function": {"name": fc["name"], "arguments": fc["arguments"]}})
                i += 1
            result.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            continue
        elif item.get("type") == "function_call_output":
            result.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        else:
            result.append(item)
        i += 1
    return result


async def agent_events(system, items, max_iterations=6):
    """Один ход агента как поток событий: ("text", кусок), ("tool", имя), ("usage", dict), ("step", элементы шага)."""
    from common import MODEL, aclient, usage_dict
    from support_tools import TOOLS

    for _ in range(max_iterations):
        messages = [{"role": "system", "content": system}, *items]
        resp = await aclient.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, max_tokens=2000,
        )

        yield "usage", usage_dict(resp.usage)

        msg = resp.choices[0].message
        # model_dump сохраняет все поля включая thought_signature для Gemini
        msg_dict = msg.model_dump(exclude_none=True)

        if msg.content:
            yield "text", msg.content

        if not msg.tool_calls:
            yield "step", [msg_dict]
            return

        for tc in msg.tool_calls:
            yield "tool", tc.function.name

        tool_results = await run_tools(msg.tool_calls)
        step = [msg_dict] + tool_results
        items.extend(step)
        yield "step", step

    fallback = "Не получилось решить обращение, передаю его оператору."
    yield "text", fallback
    yield "step", [{"role": "assistant", "content": fallback}]




def block2_async_streaming():
    from schemas import agent_system
    from support_tools import TODAY

    SYSTEM = agent_system(TODAY.isoformat())

    async def ask(query, show):
        start, first = time.perf_counter(), None
        async for kind, data in agent_events(SYSTEM, [{"role": "user", "content": query}]):
            if kind == "text" and first is None:
                first = time.perf_counter() - start
            if show and kind in ("text", "tool"):
                print(data if kind == "text" else f"[{data}] ", end="", flush=True)
        return first, time.perf_counter() - start

    async def main():
        first, total = await ask("Заказ 51002 третий день «в пути», когда привезут?", show=True)
        print(f"\nпервый токен через {first:.1f} с, весь ответ за {total:.1f} с")
        queries = ["Какой статус у заказа 53422?", "Что в заказе 52876 и когда его доставили?", "Сколько стоит подъём холодильника на пятый этаж?"]
        start = time.perf_counter()
        times = await asyncio.gather(*(ask(q, show=False) for q in queries))
        print(f"три запроса параллельно: {time.perf_counter() - start:.1f} с, по отдельности заняли бы {sum(t for _, t in times):.1f} с")

    asyncio.run(main())


# --- Блок 3. Сервис: два эндпоинта, стриминг через SSE, лимиты, отмена ----------------------------


def create_app():
    from collections import defaultdict

    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    import store
    from schemas import agent_system
    from support_tools import TODAY

    CALLS_PER_MINUTE, TOKENS_PER_DAY = 20, 300_000
    SYSTEM = agent_system(TODAY.isoformat())
    locks = defaultdict(asyncio.Lock)
    app = FastAPI()

    class MessageIn(BaseModel):
        text: str = Field(min_length=1, max_length=4000)

    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    @app.post("/sessions")
    async def create_session(x_user_id: str = Header()):
        return {"session_id": store.create_session(x_user_id)}

    @app.post("/sessions/{session_id}/messages")
    async def post_message(session_id: str, body: MessageIn, x_user_id: str = Header()):
        if store.session_user(session_id) != x_user_id:
            raise HTTPException(404, "Сессия не найдена")
        calls, tokens = store.spent(x_user_id)
        if calls >= CALLS_PER_MINUTE or tokens >= TOKENS_PER_DAY:
            raise HTTPException(429, "Лимит запросов исчерпан, попробуйте позже")

        async def events():
            async with locks[session_id]:
                store.append_messages(session_id, [{"role": "user", "content": body.text}])
                try:
                    async for kind, data in agent_events(SYSTEM, _from_db_format(store.load_history(session_id))):
                        if kind == "text":
                            yield sse("delta", {"text": data})
                        elif kind == "tool":
                            yield sse("tool", {"name": data})
                        elif kind == "usage":
                            store.log_usage(x_user_id, data)
                        elif kind == "step":
                            store.append_messages(session_id, _to_db_format(data))
                            store.save_working_copy()
                    yield sse("done", {})
                except asyncio.CancelledError:
                    print(f"сессия {session_id[:8]}: клиент ушёл, генерация остановлена")
                    raise
                except Exception as exc:
                    print(f"сессия {session_id[:8]}: {type(exc).__name__}: {exc}")
                    yield sse("error", {"message": "Не удалось получить ответ, повторите сообщение"})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Нужен OPENAI_API_KEY: все блоки урока вызывают модель")
    elif sys.argv[1:] == ["serve"]:
        import uvicorn

        import store

        store.open_db("support.db")
        uvicorn.run(create_app(), host="127.0.0.1", port=8000)
    elif sys.argv[1:] == ["1"]:
        block1_session_history()
    elif sys.argv[1:] == ["2"]:
        block2_async_streaming()
    else:
        print("Запуск по блокам: python 04_chat_backend.py 1 | 2 | serve")
