"""Урок 12. Мультиагентные системы: передача разговора, сабагенты, проверяющий с чистым контекстом.

Три блока, всем нужен OPENAI_API_KEY. Блоки 1 и 2 сравнивают несколько агентов с одним агентом урока 03 из agent.py
(урок 10), блок 3 сравнивает проверяющего с самопроверкой. Данные: сценарии урока 10 и data/team_scenarios.jsonl (блок 1),
письма урока 02 (блок 2), data/refund_proposals.jsonl (блок 3).
Запуск: python 12_multi_agent.py (все блоки) или python 12_multi_agent.py 2 (только второй).
Агенты и циклы лежат на уровне модуля: задание урока импортирует их отсюда.
"""

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import ValidationError

import support_tools
from agent import run_agent
from common import MODEL, client, cost_usd, final_text, load_kb, stop_reason
from schemas import Finding, Review, agent_system
from support_tools import TODAY, TOOLS, run_tool

TOOL = {t["name"]: t for t in TOOLS}  # описания инструментов из урока 03 по имени
KB = {d["id"]: d for d in load_kb()}


def kb_text(ids) -> str:
    """Раздел базы знаний для промпта: те же заголовки, что в agent_system урока 03."""
    return "\n\n".join(f"## {KB[i]['title']}\n{KB[i]['text']}" for i in ids)


# --- Блок 1: триаж и специалисты -------------------------------------------------------------


@dataclass
class Agent:
    name: str  # латиница: из имени собирается инструмент transfer_to_<name>
    instructions: str
    tools: list = field(default_factory=list)  # только эти инструменты агент может вызвать, проверяет код
    handoffs: list = field(default_factory=list)  # кому агент может передать разговор
    model: str = MODEL


DOMAINS = {  # категории роутера из урока 01 и документы базы знаний, которые к ним относятся
    "billing": ("деньги: оплата, списания, возврат денег, рассрочка, промокоды, бонусы, документы",
                ["vozvrat-tovara", "vozvrat-deneg", "dvoynoe-spisanie", "oplata-chastyami", "sposoby-oplaty",
                 "promokody", "programma-loyalnosti", "chek-i-dokumenty", "obmen-tovara", "brak-pri-poluchenii"]),
    "tech": ("неисправность или настройка купленного товара, гарантия и ремонт",
             ["garantiya", "garantiyny-remont", "negarantiynye-sluchai", "robot-pylesos", "smart-chasy",
              "naushniki-zaryadka", "nastroyka-routera", "krupnaya-tehnika"]),
    "general": ("доставка, самовывоз, статус и отмена заказа, личный кабинет, часы работы, персональные данные",
                ["dostavka-kurierom", "dostavka-v-regiony", "samovyvoz", "otmena-zakaza", "otslezhivanie-zakaza",
                 "lichny-kabinet", "chasy-raboty-podderzhki", "kak-svyazatsya", "personalnye-dannye"]),
}


def specialist(name: str) -> Agent:
    """Специалист: своя часть базы знаний, get_order и calculator; create_refund только у billing."""
    topic, ids = DOMAINS[name]
    instructions = (
        f"# Роль\nВы специалист поддержки интернет-магазина. Ваша область: {topic}. Решаете свою часть обращения до конца.\n\n"
        "# Правила\n"
        "- Правила магазина берите из базы знаний ниже, данные о заказах только из инструментов.\n"
        "- Если в обращении есть вопрос не из вашей области, решите свою часть и передайте разговор коллеге.\n"
        "- Если инструмент вернул ошибку, объясните клиенту причину и предложите, что можно сделать вместо этого.\n"
        "- Ответ клиенту: два-четыре предложения, с суммами и сроками.\n\n"
        f"# База знаний\n{kb_text(ids)}\n\n# Сегодня\n{TODAY.isoformat()}"
    )
    tools = [TOOL["get_order"], TOOL["calculator"]] + ([TOOL["create_refund"]] if name == "billing" else [])
    return Agent(name, instructions, tools, [n for n in DOMAINS if n != name])


TRIAGE = Agent(
    "triage",
    "# Роль\nВы диспетчер поддержки интернет-магазина. На вопросы клиента сами не отвечаете.\n\n"
    "# Правила\n- Сразу передайте разговор специалисту, в чью область попадает обращение.\n"
    "- Если тем несколько, выбирайте ту, из-за которой человек написал: специалист передаст остальное коллеге.",
    handoffs=list(DOMAINS),
)
TEAM = {"triage": TRIAGE, **{name: specialist(name) for name in DOMAINS}}


def transfer_tool(name: str) -> dict:
    """Передача разговора это обычный инструмент: модель его вызывает, код меняет агента."""
    return {
        "type": "function", "name": f"transfer_to_{name}", "strict": True,
        "description": f"Передать разговор специалисту. Его область: {DOMAINS[name][0]}.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }


def ask(agent: Agent, items: list):
    """Один шаг агента: его instructions и инструменты, общая история разговора."""
    return client.responses.create(
        model=agent.model, instructions=agent.instructions, input=items, store=False,
        tools=agent.tools + [transfer_tool(n) for n in agent.handoffs],
        max_output_tokens=2000, reasoning={"effort": "low"},
    )


def answer_calls(agent: Agent, calls: list, log: list) -> tuple[list, str | None]:
    """Результат на каждый вызов, иначе следующий запрос получит 400. Передача выполняется одна, первая."""
    allowed, outputs, target = {t["name"] for t in agent.tools}, [], None
    for call in calls:
        if call.name.startswith("transfer_to_"):
            result = {"error": "Разговор уже передан."} if target else {"assistant": call.name.removeprefix("transfer_to_")}
            target = target or result["assistant"]
        elif call.name not in allowed:  # право на запись проверяет код
            result = {"error": f"Инструмент {call.name} недоступен агенту {agent.name}."}
        else:
            result = json.loads(run_tool(call.name, call.arguments))
            log.append({"agent": agent.name, "name": call.name, "args": call.arguments, "result": result})
        outputs.append({"type": "function_call_output", "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False)})
    return outputs, target


def run_team(query: str, start: str = "triage", max_steps: int = 8) -> dict:
    """Ход команды. Возвращает то же, что run_agent из agent.py, и route: кто вёл разговор по очереди."""
    agent, items = TEAM[start], [{"role": "user", "content": query}]
    run = {"answer": "", "calls": [], "route": [start], "steps": 0, "cost": 0.0, "stop": "max_iterations"}
    while run["steps"] < max_steps:
        resp = ask(agent, items)
        run["steps"] += 1
        run["cost"] += cost_usd(resp.usage, agent.model)
        items += [item.to_dict() for item in resp.output]  # история общая: следующий агент увидит всё
        run["answer"] += final_text(resp) + "\n"  # клиент видит реплики всех агентов по очереди
        calls = [item for item in resp.output if item.type == "function_call"]
        if stop_reason(resp) or not calls:
            run["stop"] = stop_reason(resp) or "answer"
            return run
        outputs, target = answer_calls(agent, calls, run["calls"])
        items += outputs
        if target:  # дальше та же история, но другие instructions и инструменты
            agent = TEAM[target]
            run["route"].append(target)
    return run


def check(s: dict, run: dict) -> bool:
    """Проверка из урока 10 одним вердиктом: возвраты как в эталоне, get_order раньше create_refund, слова в ответе."""
    refunds = sorted([r["order_id"], r["amount_rub"]] for r in support_tools.REFUNDS)
    names = [c["name"] for c in run["calls"]]
    order_ok = "create_refund" not in names or "get_order" in names[: names.index("create_refund")]
    return refunds == sorted(s["refunds"]) and order_ok and all(w in run["answer"] for w in s["answer_has"])


def summary(results: list) -> dict:
    """Сводка одной конфигурации по списку (сценарий, прогон, успех): pass@1, pass^k, цена и вызовы модели."""
    by_scenario = {}
    for s, _, ok in results:
        by_scenario.setdefault(s["id"], []).append(ok)
    n = len(results)
    return {
        "pass@1": sum(ok for *_, ok in results) / n,
        "pass^k": sum(all(oks) for oks in by_scenario.values()) / len(by_scenario),  # все прогоны сценария успешны
        "$ за прогон": sum(run["cost"] for _, run, _ in results) / n,
        "вызовов модели": sum(run["steps"] for _, run, _ in results) / n,
    }


def block1_handoff():
    from common import load_team_scenarios

    K = 4  # прогонов на сценарий
    configs = {
        "один агент": lambda s: run_agent(s["query"]),
        "триаж": lambda s: run_team(s["query"]),
        "эталон": lambda s: run_team(s["query"], start=s["route"]),  # маршрут из разметки вместо триажа
    }
    results = {name: [] for name in configs}  # по конфигурации список (сценарий, прогон, успех)
    print(f"{'scenario':8} {'route':8}" + "".join(f"{name:>12}" for name in configs))
    for s in load_team_scenarios():
        for name, run_config in configs.items():
            for _ in range(K):
                support_tools.reset_state()  # каждый прогон начинается с эталонных заказов
                run = run_config(s)
                results[name].append((s, run, check(s, run)))
        cells = [f"{sum(ok for x, _, ok in results[name] if x is s)}/{K}" for name in configs]
        print(f"{s['id']:8} {s['route']:8}" + "".join(f"{c:>12}" for c in cells))

    table = {name: summary(r) for name, r in results.items()}
    print(f"\n{'':15}" + "".join(f"{name:>12}" for name in table))
    for metric in table["триаж"]:
        print(f"{metric:15}" + "".join(f"{t[metric]:12.{5 if '$' in metric else 2}f}" for t in table.values()))
    routed = [run["route"][1:2] == [s["route"]] for s, run, _ in results["триаж"]]  # первая передача по разметке
    print(f"триаж передал верно: {sum(routed)}/{len(routed)}")
    for s, run, _ in [x for x in results["триаж"] if not x[2]][:3]:  # журналы провалов команды
        print(f"\n{s['id']} провал, маршрут {run['route']}: {[(c['agent'], c['name']) for c in run['calls']]}"
              f"\n  ответ: {run['answer'].strip()[:200]}")


# --- Блок 2: оркестратор и сабагенты ---------------------------------------------------------

RESEARCHER = (
    "# Роль\nВы помогаете оркестратору поддержки интернет-магазина: выясняете одну проблему клиента. "
    "Ничего не оформляете и клиенту не пишете.\n\n"
    "# Правила\n- Правила магазина берите из базы знаний ниже, данные о заказах только из инструментов.\n"
    "- Верните находку: что выяснили, положен ли возврат и на какую сумму, что ответить клиенту.\n\n"
    f"# База знаний\n{kb_text(KB)}\n\n# Сегодня\n{TODAY.isoformat()}"
)
SUBAGENT = Agent("investigate", RESEARCHER, [TOOL["get_order"], TOOL["calculator"]])  # без create_refund: только читает


def investigate(task: str, max_steps: int = 5) -> dict:
    """Сабагент со своим контекстом. Наверх уходит находка по схеме Finding, журнал его шагов остаётся у него."""
    items, sub = [{"role": "user", "content": task}], {"calls": [], "cost": 0.0, "steps": 0}
    while sub["steps"] < max_steps:
        try:
            resp = client.responses.parse(
                model=MODEL, instructions=RESEARCHER, input=items, tools=SUBAGENT.tools, text_format=Finding,
                store=False, max_output_tokens=2000, reasoning={"effort": "low"},
            )
        except ValidationError:  # обрыв посреди JSON: parse бросает исключение, а не возвращает ответ
            return {**sub, "finding": {"error": "ответ сабагента оборвался"}}
        sub["steps"] += 1
        sub["cost"] += cost_usd(resp.usage, MODEL)
        for item in resp.output:  # parse дописывает к вызову parsed_arguments, а API такое поле во входе отвергает с 400
            items.append({k: v for k, v in item.to_dict().items() if k != "parsed_arguments"})
        calls = [item for item in resp.output if item.type == "function_call"]
        if not calls:
            finding = resp.output_parsed  # None при отказе или обрыве
            return {**sub, "finding": finding.model_dump() if finding else {"error": stop_reason(resp)}}
        outputs, _ = answer_calls(SUBAGENT, calls, sub["calls"])
        items += outputs
    return {**sub, "finding": {"error": "сабагент не уложился в лимит шагов"}}


INVESTIGATE = {
    "type": "function", "name": "investigate", "strict": True,
    "description": "Поручить сабагенту выяснить одну проблему клиента: он посмотрит заказ и правила и вернёт находку. "
                   "Сабагент не видит письмо, поэтому в задаче должно быть всё нужное.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["task"], "properties": {
        "task": {"type": "string", "description": "Проблема словами клиента, номер заказа, если он есть в письме, и что выяснить"}}},
}
ORCHESTRATOR = (
    "# Роль\nВы оркестратор поддержки интернет-магазина. В письме клиента может быть несколько проблем.\n\n"
    "# Правила\n"
    "- Разбейте письмо на проблемы и по каждой вызовите investigate. Все investigate вызывайте в одном ответе.\n"
    "- Возврат оформляйте сами через create_refund и только по находке, где refund равно true.\n"
    "- Затем ответьте клиенту одним письмом по всем проблемам, с суммами и сроками.\n\n"
    f"# Сегодня\n{TODAY.isoformat()}"
)


def orchestrate(calls: list, run: dict) -> list:
    """Сабагенты идут параллельно, запись после них и в одном потоке: возврат не оформят дважды."""
    tasks = [c for c in calls if c.name == "investigate"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        subs = dict(zip([c.call_id for c in tasks], pool.map(lambda c: investigate(json.loads(c.arguments)["task"]), tasks)))
    run["fanout"] = max(run["fanout"], len(tasks))  # сколько сабагентов запущено одним ответом
    outputs = []
    for call in calls:
        if call.call_id in subs:
            sub = subs[call.call_id]
            run["cost"] += sub["cost"]
            run["steps"] += sub["steps"]
            run["calls"] += sub["calls"]
            result = sub["finding"]
        elif call.name == "create_refund":  # пишет только оркестратор
            result = json.loads(run_tool(call.name, call.arguments))
            run["calls"].append({"agent": "orchestrator", "name": call.name, "args": call.arguments, "result": result})
        else:  # чужой инструмент код не исполняет, как answer_calls в блоке 1
            result = {"error": f"Инструмент {call.name} недоступен оркестратору."}
        outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result, ensure_ascii=False)})
    return outputs


def run_orchestrator(text: str, max_steps: int = 6) -> dict:
    """Ход оркестратора. Шаги и цена считаются вместе с сабагентами."""
    items = [{"role": "user", "content": text}]
    run = {"answer": "", "calls": [], "steps": 0, "cost": 0.0, "fanout": 0, "stop": "max_iterations"}
    for _ in range(max_steps):
        resp = client.responses.create(
            model=MODEL, instructions=ORCHESTRATOR, input=items, tools=[INVESTIGATE, TOOL["create_refund"]],
            store=False, max_output_tokens=3000, reasoning={"effort": "low"},
        )
        run["steps"] += 1
        run["cost"] += cost_usd(resp.usage, MODEL)
        items += [item.to_dict() for item in resp.output]
        calls = [item for item in resp.output if item.type == "function_call"]
        if stop_reason(resp) or not calls:
            run.update(answer=final_text(resp), stop=stop_reason(resp) or "answer")
            return run
        items += orchestrate(calls, run)
    return run


def letter_runs(letter: dict, run_config, K: int) -> list:
    """K прогонов одной конфигурации на письме. К каждому прогону дописываются время, возвраты и закрытые проблемы."""
    runs = []
    for _ in range(K):
        support_tools.reset_state()
        start = time.perf_counter()
        run = run_config(letter["text"])
        run["seconds"] = time.perf_counter() - start
        run["refunds"] = sorted([r["order_id"], r["amount_rub"]] for r in support_tools.REFUNDS)
        run["covered"] = sum(any(w in run["answer"] for w in group) for group in letter["covers"])  # закрытые проблемы
        runs.append(run)
    return runs


def block2_orchestrator():
    from common import load_letters

    K = 4
    configs = {"один агент": lambda text: run_agent(text, max_iterations=10), "оркестратор": run_orchestrator}
    print(f"{'letter':7} {'config':12} {'pass':>5} {'state':>6} {'covered':>8} {'time, s':>8} {'$':>8} {'calls':>6} {'fanout':>6}")
    failed = []
    for letter in load_letters():
        issues = len(letter["covers"])
        for name, run_config in configs.items():
            runs = letter_runs(letter, run_config, K)
            state = [r["refunds"] == sorted(letter["refunds"]) for r in runs]
            passed = sum(ok and r["covered"] == issues for ok, r in zip(state, runs))
            print(f"{letter['id']:7} {name:12} {passed:>3}/{K} {sum(state):>4}/{K} {sum(r['covered'] for r in runs):>4}/{K * issues:<3} "
                  f"{sum(r['seconds'] for r in runs) / K:8.1f} {sum(r['cost'] for r in runs) / K:8.5f} "
                  f"{sum(r['steps'] for r in runs) / K:6.1f} {max(r.get('fanout', 0) for r in runs):>6}")
            failed += [(letter, name, r) for ok, r in zip(state, runs) if not ok]
    for letter, name, run in failed[:4]:  # возвраты не как в эталоне: что было сделано
        print(f"\n{letter['id']} {name}: возвраты {run['refunds']}, эталон {letter['refunds']}"
              f"\n  ответ: {run['answer'].strip()[:200]}")


# --- Блок 3: проверяющий с чистым контекстом -------------------------------------------------

CHECKLIST = (
    "Вызов create_refund перехвачен до отправки денег. create_refund регистрирует возврат: деньги уйдут клиенту "
    "после того, как товар вернётся на склад и пройдёт проверку. Проверьте решение: просил ли клиент вернуть деньги, "
    "положен ли возврат по правилам магазина, тот ли это заказ и та ли сумма. Одобряйте только решение без ошибок."
)
HOLD = {"status": "on_review", "message": "Возврат не отправлен: решение проверяется перед отправкой денег."}


def self_check(p: dict, order: dict):
    """Самопроверка: история агента, в которой он сам проверил заказ и вызвал create_refund, и тот же чек-лист."""
    args = p["proposal"]
    items = [
        {"role": "user", "content": p["query"]},
        {"type": "function_call", "call_id": "call_1", "name": "get_order", "arguments": json.dumps({"order_id": args["order_id"]})},
        {"type": "function_call_output", "call_id": "call_1", "output": json.dumps(order, ensure_ascii=False)},
        {"type": "function_call", "call_id": "call_2", "name": "create_refund", "arguments": json.dumps(args, ensure_ascii=False)},
        {"type": "function_call_output", "call_id": "call_2", "output": json.dumps(HOLD, ensure_ascii=False)},
        {"role": "developer", "content": CHECKLIST},
    ]
    return client.responses.parse(  # instructions и инструменты те же, что у агента: это его собственный контекст
        model=MODEL, instructions=agent_system(TODAY.isoformat()), input=items, tools=TOOLS, tool_choice="none",
        text_format=Review, store=False, max_output_tokens=2000, reasoning={"effort": "low"},
    )


VERIFIER = (
    "# Роль\nВы проверяющий в поддержке интернет-магазина. Решения о возврате принимает другой агент, "
    f"вы видите обращение клиента, заказ и решение агента.\n\n# Задача\n{CHECKLIST}\n\n"
    f"# База знаний\n{kb_text(KB)}\n\n# Сегодня\n{TODAY.isoformat()}"
)


def verifier_check(p: dict, order: dict):
    """Проверяющий: чистый контекст, решение другого агента пришло как данные."""
    case = {"обращение клиента": p["query"], "заказ": order, "решение агента": p["proposal"]}
    return client.responses.parse(model=MODEL, instructions=VERIFIER, input=json.dumps(case, ensure_ascii=False),
                                  text_format=Review, store=False, max_output_tokens=2000, reasoning={"effort": "low"})


def verdict(review, p: dict, order: dict):
    """Ответ проверки или None, если он оборвался посреди JSON: такой вердикт считаем «не одобрено»."""
    try:
        return review(p, order)
    except ValidationError:
        return None


def block3_reviewer():
    from common import load_refund_proposals

    K = 3  # прогонов на предложение
    support_tools.reset_state()
    cases = [(p, support_tools.get_order(p["proposal"]["order_id"])) for p in load_refund_proposals()] * K
    faults = sorted({p["fault"] for p, _ in cases if p["fault"]})
    print(f"{'':14} {'TPR':>5} {'TNR':>5} {'$':>8}   отклонены по типам: " + " ".join(faults))
    for name, review in {"самопроверка": self_check, "проверяющий": verifier_check}.items():
        with ThreadPoolExecutor(max_workers=8) as pool:  # состояние не меняется: проверки идут параллельно
            resps = list(pool.map(lambda case: verdict(review, *case), cases))
        approved = [r is not None and r.output_parsed is not None and r.output_parsed.approve for r in resps]
        good = [a for (p, _), a in zip(cases, approved) if p["ok"]]
        bad = [a for (p, _), a in zip(cases, approved) if not p["ok"]]
        caught = {f: [not a for (p, _), a in zip(cases, approved) if p["fault"] == f] for f in faults}
        cost = sum(cost_usd(r.usage, MODEL) for r in resps if r)  # у оборванных ответов usage потерян
        print(f"{name:14} {sum(good) / len(good):5.2f} {bad.count(False) / len(bad):5.2f} {cost:8.4f}   "  # TPR, TNR
              + " ".join(f"{sum(c)}/{len(c)}".rjust(len(f)) for f, c in caught.items())
              + f"   обрывов: {resps.count(None)}")
        wrong = Counter(p["id"] for (p, _), a in zip(cases, approved) if a != p["ok"])  # ошибки: сырьё для разбора
        print(f"{'':14} ошибся на: {', '.join(f'{i}×{n}' for i, n in sorted(wrong.items())) or '-'}")


if __name__ == "__main__":
    import sys

    blocks = {
        "1": ("Триаж и специалисты против одного агента", block1_handoff),
        "2": ("Оркестратор и параллельные сабагенты против одного агента", block2_orchestrator),
        "3": ("Проверяющий с чистым контекстом против самопроверки", block3_reviewer),
    }
    for number in sys.argv[1:] or blocks:  # без аргументов идут все блоки, «python 12_multi_agent.py 2» только второй
        title, run_block = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run_block()
