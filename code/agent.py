"""Агент из урока 03 одной функцией, которая возвращает не только ответ, но и журнал прогона.

С урока 10 журнал нужен проверкам: какие инструменты вызывались, с какими аргументами, сколько шагов и денег ушло.
Состояние заказов агент меняет через support_tools, поэтому перед каждым прогоном вызывайте reset_state().
Вызовы идут через Responses API: история своя (store=False), системный промпт в instructions.
"""

import json

from common import MODEL, client, cost_usd, final_text, stop_reason
from schemas import agent_system
from support_tools import TODAY, TOOLS, run_tool

SYSTEM = agent_system(TODAY.isoformat())


def _args(arguments_json: str) -> dict:
    """Аргументы вызова для журнала. Битый JSON от модели не должен ронять прогон."""
    try:
        return json.loads(arguments_json)
    except json.JSONDecodeError:
        return {"_raw": arguments_json}


def run_agent(query: str, model: str = MODEL, max_iterations: int = 6) -> dict:
    """Один ход агента. Возвращает answer, calls (имя, аргументы, результат), steps, cost и stop: почему цикл закончился."""
    items = [{"role": "user", "content": query}]  # история хода; системный промпт идёт отдельно, в instructions
    calls, cost = [], 0.0
    for step in range(1, max_iterations + 1):
        resp = client.responses.create(
            model=model, instructions=SYSTEM, input=items, tools=TOOLS, store=False,  # историю храним сами
            max_output_tokens=2000, reasoning={"effort": "low"},
        )
        cost += cost_usd(resp.usage, model)
        items += [item.to_dict() for item in resp.output]  # весь output обратно в input: reasoning, реплики, вызовы
        stop = stop_reason(resp)
        if stop:  # обрыв по лимиту или отказ это не «задача решена»
            return {"answer": "", "calls": calls, "steps": step, "cost": cost, "stop": stop}
        tool_calls = [item for item in resp.output if item.type == "function_call"]
        if not tool_calls:
            return {"answer": final_text(resp), "calls": calls, "steps": step, "cost": cost, "stop": "answer"}
        for call in tool_calls:
            result = run_tool(call.name, call.arguments)  # ошибка инструмента возвращается модели
            calls.append({"name": call.name, "args": _args(call.arguments), "result": json.loads(result)})
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": result})
    return {"answer": "", "calls": calls, "steps": max_iterations, "cost": cost, "stop": "max_iterations"}
