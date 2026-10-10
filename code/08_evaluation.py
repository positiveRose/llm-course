"""Урок 08. Evals: разбор ошибок, грейдеры, проверка судьи.

Один блок, нужен OPENAI_API_KEY. Данные: data/answers_labeled.jsonl, 120 ответов ассистента урока 06 с ручной разметкой.
Запуск: python 08_evaluation.py
"""

import os


def block1_graders():
    import re
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    from pydantic import BaseModel, Field

    from common import MODEL, STRONG_MODEL, client, cost_usd, load_labeled_answers, stop_reason

    answers = load_labeled_answers()
    print("labels:", dict(Counter(a["label"] for a in answers)))
    print("failure modes:", Counter(a["mode"] for a in answers if a["label"] == "fail").most_common())

    NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
    def numbers_grader(a):
        """Код: каждое число из ответа есть во фрагментах или в вопросе клиента. id документов в скобках не проверяем."""
        text = re.sub(r"\[[^\]]*\]", "", a["answer"])
        source = set(NUMBER.findall(a["question"] + " " + " ".join(c["text"] for c in a["chunks"])))
        return all(n in source for n in NUMBER.findall(text)), 0.0

    class Grounded(BaseModel):
        reasoning: str = Field(description="Какие утверждения ответа есть во фрагментах, а каких нет, в двух-трёх предложениях")
        grounded: bool  # после reasoning: вердикт опирается на разбор, а не наоборот

    JUDGE = """# Роль
Вы проверяете ответы ассистента поддержки интернет-магазина.

# Критерий
Ответ проходит, если каждое его утверждение о магазине, товарах, правилах, сроках, суммах и действиях клиента есть во фрагментах или прямо из них следует.
Ответ не проходит, если хотя бы одно такое утверждение во фрагментах отсутствует, искажено или относится во фрагменте к другому случаю.
- Ответ «Не знаю» проходит. Совет обратиться в поддержку проходит всегда.
- Вежливые фразы и id документов в скобках не проверяйте.
- Верно ли утверждение по вашим собственным знаниям, не важно: только фрагменты."""

    def judge(a, model):
        fragments = "\n\n".join(f"<fragment>\n{c['text']}\n</fragment>" for c in a["chunks"])
        user = f"# Фрагменты\n{fragments}\n\n# Вопрос клиента\n{a['question']}\n\n# Ответ ассистента\n{a['answer']}"
        resp = client.responses.parse(
            model=model,
            instructions=JUDGE,
            input=user,
            text_format=Grounded,
            store=False,
            max_output_tokens=600,
            reasoning={"effort": "none"},
            temperature=0,  # судья это измерительный прибор: одинаковый вход, одинаковый вердикт
        )
        stop = stop_reason(resp)
        if stop:  # отказ приходит мимо схемы (урок 02); обрыв посреди JSON parse бросает ValidationError сам
            raise RuntimeError(stop)
        return resp.output_parsed.grounded, cost_usd(resp.usage, model)

    def agreement(human, grader):
        """Сверка грейдера с человеком. Положительный класс pass: TPR про хорошие ответы, TNR про плохие."""
        pairs = list(zip(human, grader))
        tp = sum(h and g for h, g in pairs)
        tn = sum(not h and not g for h, g in pairs)
        fp = sum(not h and g for h, g in pairs)  # плохой ответ, который грейдер пропустил
        fn = sum(h and not g for h, g in pairs)  # хороший ответ, который грейдер забраковал
        return tp / (tp + fn), tn / (tn + fp), (tp + tn) / len(pairs)  # TPR, TNR и agreement, доля совпадений

    human = [a["label"] == "pass" for a in answers]
    graders = {
        "всегда pass": lambda a: (True, 0.0),
        "числа кодом": numbers_grader,
        "судья MODEL": lambda a: judge(a, MODEL),
        "судья STRONG": lambda a: judge(a, STRONG_MODEL),
    }
    print(f"\n{'grader':13} {'TPR':>5} {'TNR':>5} {'agreement':>9} {'$':>8}")
    verdicts = {}
    for name, grade in graders.items():
        with ThreadPoolExecutor(8) as pool:  # вызовы независимы, как роутер и экстрактор в уроке 03
            results = list(pool.map(grade, answers))
        verdicts[name] = [ok for ok, _ in results]
        tpr, tnr, po = agreement(human, verdicts[name])
        print(f"{name:13} {tpr:5.2f} {tnr:5.2f} {po:9.2f} {sum(c for _, c in results):8.4f}")

    print("\nрасхождения судьи STRONG с человеком:")
    for a, h, g in zip(answers, human, verdicts["судья STRONG"]):
        if h != g:
            print(f"  {a['id']} человек {a['label']}, судья {'pass' if g else 'fail'}: {a['mode'] or '-'}; {a['note'][:70]}")


if __name__ == "__main__":
    import sys

    blocks = {
        "1": ("Четыре грейдера против разметки человека", block1_graders),
    }
    for number in sys.argv[1:] or blocks:
        title, run = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
