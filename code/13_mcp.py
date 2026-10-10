"""Урок 13. MCP: сервер, клиент в цикле агента и сервер под задачу против сервера как REST.

Три блока. Блоки 1 и 3 работают без ключа, блоку 2 нужен OPENAI_API_KEY.
Серверы: support_server.py (под задачу) и api_server.py (как REST). Пакет: pip install fastmcp, он ставит и клиент mcp.
Запуск: python 13_mcp.py (все блоки) или python 13_mcp.py 2 (только второй).
"""

import asyncio
import json
import os
import sys
from importlib import import_module
from pathlib import Path

from mcp import Client, StdioServerParameters

import support_tools
from agent import SYSTEM, _args, run_agent
from common import MODEL, aclient, cost_usd, final_text, load_agent_scenarios, stop_reason

HERE = Path(__file__).parent
MAX_STEPS = 10  # поровну всем вариантам: серверу как REST нужно больше шагов, и обрезать его раньше нечестно


def server(file: str) -> StdioServerParameters:
    """Как запустить сервер: клиент поднимет его подпроцессом. Логи FastMCP глушим, чтобы не мешали выводу урока."""
    quiet = {"FASTMCP_LOG_LEVEL": "CRITICAL", "OPENAI_API_KEY": ""}  # серверу ключ не нужен, пустое значение глушит common
    return StdioServerParameters(command=sys.executable, args=[str(HERE / file)], env=quiet)


def to_tools(mcp_tools) -> list[dict]:
    """Ответ tools/list в формате tools для Responses. inputSchema это готовая JSON Schema, её отдаём как есть."""
    tools = []
    for t in mcp_tools:
        schema = t.input_schema  # строгий режим требует, чтобы все поля были обязательными: default это нарушает
        strict = schema.get("additionalProperties") is False and set(schema.get("required", [])) == set(schema["properties"])
        tools.append({"type": "function", "name": t.name, "description": t.description or "", "parameters": schema,
                      "strict": strict})  # явно: без strict Responses молча откатывается на нестрогий режим
    return tools


def result_text(result) -> str:
    """Результат tools/call в JSON-строку для function_call_output. Контракт run_tool урока 03: ошибка в поле error."""
    text = "\n".join(part.text for part in result.content if part.type == "text")
    if result.is_error:
        return json.dumps({"error": text}, ensure_ascii=False)
    data = result.structured_content if result.structured_content is not None else {"text": text}
    return json.dumps(data, ensure_ascii=False)


def block1_client():
    async def main():
        async with Client(server("support_server.py")) as session:  # подпроцесс сервера живёт внутри with
            tools = to_tools((await session.list_tools()).tools)
            print("инструменты:", [(t["name"], t["strict"]) for t in tools])
            ok = await session.call_tool("get_order", {"order_id": 49880})
            print("get_order:", result_text(ok)[:140], "…")
            bad = await session.call_tool("create_refund", {"order_id": 51877, "amount_rub": 84990, "reason": "не понравился"})
            print("create_refund:", result_text(bad))
            refunds = await session.read_resource("support://refunds")
            print("ресурс support://refunds:", refunds.contents[0].text)

    asyncio.run(main())


async def run_agent_mcp(query: str, session, tools: list[dict], max_iterations: int = MAX_STEPS) -> dict:
    """Агент урока 03, у которого инструменты живут на MCP-сервере. Возвращает то же, что agent.run_agent."""
    items, calls, cost = [{"role": "user", "content": query}], [], 0.0
    for step in range(1, max_iterations + 1):
        resp = await aclient.responses.create(
            model=MODEL, instructions=SYSTEM, input=items, tools=tools, store=False,
            max_output_tokens=2000, reasoning={"effort": "low"},
        )
        cost += cost_usd(resp.usage, MODEL)
        items += [item.to_dict() for item in resp.output]
        if stop := stop_reason(resp):
            return {"answer": "", "calls": calls, "steps": step, "cost": cost, "stop": stop}
        function_calls = [item for item in resp.output if item.type == "function_call"]
        if not function_calls:
            return {"answer": final_text(resp), "calls": calls, "steps": step, "cost": cost, "stop": "answer"}
        for call in function_calls:
            args = _args(call.arguments)  # битый JSON от модели не роняет прогон, как в agent.run_agent
            try:
                output = result_text(await session.call_tool(call.name, args))  # вместо run_tool
            except Exception as e:  # протокольная ошибка (нет инструмента, битый запрос) тоже уходит модели
                output = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            calls.append({"name": call.name, "args": args, "result": json.loads(output)})
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
    return {"answer": "", "calls": calls, "steps": max_iterations, "cost": cost, "stop": "max_iterations"}


def check(scenario: dict, run: dict, refunds: list[dict]) -> bool:
    """Три проверки урока 10: нужные возвраты и никаких лишних, get_order до create_refund, слова в ответе."""
    state_ok = sorted([r["order_id"], r["amount_rub"]] for r in refunds) == sorted(scenario["refunds"])
    names = [c["name"] for c in run["calls"]]
    order_ok = "create_refund" not in names or "get_order" in names[: names.index("create_refund")]
    return state_ok and order_ok and all(word in run["answer"] for word in scenario["answer_has"])


LOOP = asyncio.Runner()  # один цикл событий на все прогоны: соединения aclient привязаны к циклу, где открылись


def over_mcp(file: str):
    """Вариант агента поверх MCP: свой процесс сервера на прогон, возвраты для проверки читаются ресурсом."""
    async def run(query):
        async with Client(server(file)) as session:  # новый процесс: заказы как в эталоне
            result = await run_agent_mcp(query, session, to_tools((await session.list_tools()).tools))
            refunds = await session.read_resource("support://refunds")  # состояние живёт в сервере
            return result, json.loads(refunds.contents[0].text)

    return lambda query: LOOP.run(run(query))  # asyncio.run на каждый прогон уронит второй: Event loop is closed


def in_code(query: str):
    support_tools.reset_state()  # инструменты в процессе агента: рабочую копию сбрасываем сами
    return run_agent(query, max_iterations=MAX_STEPS), support_tools.REFUNDS


def block2_rest_vs_task():
    variants = {"в коде": in_code, "MCP под задачу": over_mcp("support_server.py"), "MCP как REST": over_mcp("api_server.py")}
    K, scenarios = 2, load_agent_scenarios()  # K прогонов на сценарий в каждом варианте
    runs = {name: [] for name in variants}  # вариант -> [(сценарий, прогон, прошёл ли)]
    for s in scenarios:
        for name, variant in variants.items():
            for _ in range(K):
                result, refunds = variant(s["query"])
                runs[name].append((s["id"], result, check(s, result, refunds)))

    print(f"{'scenario':8}" + "".join(f"{name:>16}" for name in variants))
    for s in scenarios:
        print(f"{s['id']:8}" + "".join(f"{sum(ok for sid, _, ok in runs[n] if sid == s['id']):>14}/{K}" for n in variants))
    print()
    for name, rows in runs.items():
        results = [r for _, r, _ in rows]
        errors = sum("error" in c["result"] for r in results for c in r["calls"])  # вызовы, вернувшие ошибку
        print(f"{name:15} pass rate {sum(ok for *_, ok in rows)}/{len(rows)}, "
              f"шагов {sum(r['steps'] for r in results) / len(results):.1f}, "
              f"вызовов {sum(len(r['calls']) for r in results) / len(results):.1f}, ошибок {errors}, "
              f"${sum(r['cost'] for r in results) / len(results):.4f} на прогон")


def block3_catalog_search():
    bm25 = import_module("07_rag_advanced")  # BM25 урока 07 без изменений
    catalog = json.loads((HERE / "data" / "tool_catalog.json").read_text(encoding="utf-8"))
    tools = catalog["tools"]
    requests = [json.loads(line) for line in (HERE / "data" / "tool_requests.jsonl").open(encoding="utf-8")]

    def document(tool):
        """Что индексируем: имя словами, описание, имена и описания параметров."""
        params = " ".join(f"{name} {p['description']}" for name, p in tool["parameters"]["properties"].items())
        return f"{tool['name'].replace('_', ' ')} {tool['description']} {params}"

    index = bm25.BM25([document(t) for t in tools])
    found = [[tools[i]["name"] for i in index.top(r["text"], 5)] for r in requests]
    hits = sum(r["expected"] in top for r, top in zip(requests, found))
    print(f"Инструментов {len(tools)}, обращений {len(requests)}. BM25 по тексту обращения: recall@5 {hits}/{len(requests)}")
    for r, top in zip(requests, found):
        if r["expected"] not in top:
            print(f"  {r['id']} {r['text'][:48]!r:52} нужен {r['expected']}, найдено {top[:2]}")


if __name__ == "__main__":
    blocks = {
        "1": ("Сервер и клиент", block1_client, False),
        "2": ("Сервер под задачу против сервера как REST", block2_rest_vs_task, True),
        "3": ("Поиск по каталогу инструментов", block3_catalog_search, False),
    }
    for number in sys.argv[1:] or blocks:
        title, run, needs_key = blocks[number]
        if needs_key and not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
