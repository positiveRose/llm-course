"""Урок 10. Evals для конвейеров и агентов: состояние, pass^k, гейт регрессии.

Два блока, обоим нужен OPENAI_API_KEY. Данные: data/agent_scenarios.jsonl (блок 1), тикеты урока 01 (блок 2).
Агент с журналом вызовов лежит в agent.py.
Запуск: python 10_agent_evals.py (оба блока) или python 10_agent_evals.py 2 (только второй).
Блок 2 завершает процесс с кодом 1, если гейт красный: так его запускает CI.
"""

import os


def block1_agent_runs():
    from math import comb

    import support_tools
    from agent import run_agent
    from common import load_agent_scenarios

    K = 4  # прогонов на сценарий

    def check(s, run):
        """Три проверки прогона: состояние заказов, порядок вызовов, обязательные слова в ответе."""
        refunds = sorted([r["order_id"], r["amount_rub"]] for r in support_tools.REFUNDS)
        state_ok = refunds == sorted(s["refunds"])  # нужный возврат есть, лишних нет
        names = [c["name"] for c in run["calls"]]
        order_ok = "create_refund" not in names or "get_order" in names[: names.index("create_refund")]
        answer_ok = all(word in run["answer"] for word in s["answer_has"])
        return state_ok, order_ok, answer_ok

    def pass_at_k(n, c, k):  # хотя бы один из k прогонов успешен
        return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)

    def pass_hat_k(n, c, k):  # все k прогонов успешны
        return comb(c, k) / comb(n, k)

    print(f"{'scenario':8} {'pass':>5} {'state':>7} {'order':>7} {'answer':>6} {'steps':>5} {'$':>8}")
    table, failed = [], []
    for s in load_agent_scenarios():
        checks, runs = [], []
        for _ in range(K):
            support_tools.reset_state()  # каждый прогон начинается с эталонных заказов
            run = run_agent(s["query"])
            checks.append(check(s, run))
            runs.append(run)
        c = sum(all(ok) for ok in checks)
        table.append((s, c))
        failed += [(s, run) for ok, run in zip(checks, runs) if not all(ok)]
        cols = [sum(ok[i] for ok in checks) for i in range(3)]
        steps = sum(r["steps"] for r in runs) / K
        print(f"{s['id']:8} {c:>3}/{K} {cols[0]:>5}/{K} {cols[1]:>5}/{K} {cols[2]:>4}/{K} {steps:5.1f} {sum(r['cost'] for r in runs):8.5f}")

    n = len(table)
    print(f"\npass@1 {sum(pass_at_k(K, c, 1) for _, c in table) / n:.2f}   "
          f"pass@{K} {sum(pass_at_k(K, c, K) for _, c in table) / n:.2f}   "
          f"pass^{K} {sum(pass_hat_k(K, c, K) for _, c in table) / n:.2f}")
    for s, c in table:
        if s.get("critical") and c < K:
            print(f"КРИТИЧНО: {s['id']} прошёл {c} из {K}")
    for s, run in failed[:3]:  # журнал провала: сырьё для разбора ошибок
        print(f"\n{s['id']} провал: {[(c['name'], c['args']) for c in run['calls']]}\n  ответ: {run['answer'][:160]}")


def block2_regression_gate():
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from common import MODEL, client, load_tickets
    from schemas import ROUTER_PROMPT, Route

    dev = load_tickets()[:40]
    CRITICAL = {"T-017", "T-021", "T-022"}  # деньги клиента: ошибка маршрута здесь стоит дороже остальных
    OLD_TECH = "- tech: неисправность или настройка купленного товара, сбои сайта и приложения."
    OLD_BILLING = "- billing: деньги. Оплата, списания, возврат денег, рассрочка, промокоды, бонусы, документы для бухгалтерии."
    CANDIDATES = {
        "v2, вход в кабинет к tech": ROUTER_PROMPT.replace(OLD_TECH, OLD_TECH[:-1] + ", включая вход в личный кабинет и SMS-коды."),
        "v3, короткий billing": ROUTER_PROMPT.replace(OLD_BILLING, "- billing: оплата заказа и способы оплаты."),
    }
    assert ROUTER_PROMPT not in CANDIDATES.values(), "replace ничего не заменил: строка в schemas.py изменилась"

    def route(prompt, text):
        resp = client.responses.parse(
            model=MODEL,
            instructions=prompt,
            input=text,
            text_format=Route,
            store=False,
            max_output_tokens=50,
            reasoning={"effort": "none"},
            temperature=0,
        )
        return resp.output_parsed.category  # при отказе output_parsed равен None: прогон упадёт, и сборка в CI покраснеет

    def run(prompt):
        """Прогон набора: словарь «id тикета → верно ли». Его и сравнивают между версиями."""
        with ThreadPoolExecutor(8) as pool:
            answers = list(pool.map(lambda t: route(prompt, t["text"]), dev))
        return {t["id"]: answer == t["category"] for t, answer in zip(dev, answers)}

    baseline_path = Path("runs/router_baseline.json")
    if not baseline_path.exists():  # baseline снимается один раз и хранится рядом с кодом
        baseline_path.parent.mkdir(exist_ok=True)
        baseline_path.write_text(json.dumps(run(ROUTER_PROMPT), indent=1))
    base = json.loads(baseline_path.read_text())
    noise = sum(base[i] != ok for i, ok in run(ROUTER_PROMPT).items())  # та же версия ещё раз: сколько тикетов «плавает»
    print(f"baseline: {sum(base.values())}/{len(base)}, шум при повторе: {noise} тикетов")

    def gate(new):
        fixed = [i for i in base if new[i] and not base[i]]
        broken = [i for i in base if base[i] and not new[i]]
        critical = sorted(set(broken) & CRITICAL)
        ok = not critical and len(broken) - len(fixed) <= noise
        print(f"  {sum(new.values())}/{len(new)}, починено {fixed or '-'}, сломано {broken or '-'}")
        print("  ЗЕЛЁНЫЙ" if ok else f"  КРАСНЫЙ: {'сломаны критичные ' + str(critical) if critical else 'сломано больше, чем шум'}")
        return ok

    results = []
    for name, prompt in CANDIDATES.items():
        print(name)
        results.append(gate(run(prompt)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    import sys

    blocks = {
        "1": ("Агент: состояние, порядок вызовов, pass^k", block1_agent_runs),
        "2": ("Гейт регрессии для роутера", block2_regression_gate),
    }
    code = 0
    for number in sys.argv[1:] or blocks:  # без аргументов идут оба блока, «python 10_agent_evals.py 2» запускает только второй
        title, run_block = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        code = run_block() or code  # блок 2 возвращает код выхода для CI
    sys.exit(code)
