"""Урок 07. Продвинутый поиск: гибрид, реранкинг и агентский RAG.

Блоки запускаются по одному: python 07_rag_advanced.py 1, затем 2 и 3.
Индекс берётся из урока 06: сначала выполните python 06_rag_basics.py 2.
Нужен OPENAI_API_KEY. Из пакетов нужен только numpy.
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np

STOP = set("а бы в во для до же за и из или к как ко ли на не но о об от по при с со то у что это я мне мой".split())  # служебные слова
K_RRF = 60  # константа сглаживания в RRF: гасит вклад верхних позиций, значение по умолчанию из статьи


def terms(text: str) -> list[str]:
    """Слова в нижнем регистре. Русские слова обрезаны до шести букв вместо стемминга: грубо, зато без зависимостей.

    Всё, где есть цифра, латиница или подчёркивание, остаётся целиком: коды ошибок, артикулы, имена функций.
    Ради таких точных строк BM25 и нужен, а обрезка склеила бы «1234567» с «1234568».
    """
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]  # стоп-слова ничего не различают
    return [w if re.search(r"[0-9a-z_]", w) else w[:6] for w in words]


class BM25:
    """Лексический поиск: насколько слова запроса редки и часто ли они встречаются в документе."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [terms(t) for t in texts]
        self.k1, self.b = k1, b  # k1 гасит вклад повторов слова, b учитывает длину документа
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg = float(self.lengths.mean())
        self.freqs = [Counter(d) for d in self.docs]
        seen = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in seen.items()}

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.docs), dtype=np.float32)
        for word in terms(query):
            if word not in self.idf:
                continue  # слова, которого нет ни в одном документе, для BM25 не существует
            freq = np.array([f[word] for f in self.freqs], dtype=np.float32)
            norm = freq + self.k1 * (1 - self.b + self.b * self.lengths / self.avg)
            result += self.idf[word] * freq * (self.k1 + 1) / norm
        return result

    def top(self, query: str, n: int = 20) -> list[int]:
        """Номера лучших n документов. Нулевой счёт значит «ни одного общего слова»: такие в выдачу не идут,
        иначе argsort вернёт произвольные чанки, и они на равных войдут в слияние."""
        scores = self.scores(query)
        return [int(i) for i in np.argsort(-scores)[:n] if scores[i] > 0]


def rrf(*rankings: list[int], k: int = K_RRF) -> list[int]:
    """Слияние ранжированных списков по позициям: у документа тем выше вес, чем выше он в каждом списке."""
    score: Counter = Counter()
    for ranking in rankings:
        for position, index in enumerate(ranking, 1):
            score[index] += 1 / (k + position)
    return [i for i, _ in score.most_common()]


def block1_hybrid():
    from importlib import import_module

    from common import load_rag_questions

    rag = import_module("06_rag_basics")
    chunks, matrix = rag.load_index()
    bm25 = BM25([f"{c['title']}. {c['section']}\n{c['text']}" for c in chunks])
    questions = [q for q in load_rag_questions() if q["docs"] and not q.get("multi")]  # многотемные меряются иначе, блок 3

    def rank_vector(query, top=20):
        return list(np.argsort(-(rag.embed([query])[0] @ matrix.T))[:top])

    def rank_bm25(query, top=20):
        return bm25.top(query, top)

    def measure(name, rank):
        hits = {1: 0, 3: 0}
        reciprocal = 0.0
        for q in questions:
            ranking = rank(q["question"])
            found = [chunks[i]["doc"] in q["docs"] for i in ranking]
            for k in hits:
                hits[k] += any(found[:k])
            reciprocal += 1 / (found.index(True) + 1) if True in found else 0
        n = len(questions)
        print(f"{name:12} recall@1 {hits[1]:2}/{n}  recall@3 {hits[3]:2}/{n}  MRR {reciprocal / n:.3f}")

    measure("вектор", rank_vector)
    measure("BM25", rank_bm25)
    measure("гибрид", lambda q: rrf(rank_vector(q), rank_bm25(q)))

    query = "что значит E05"  # слепое пятно эмбеддингов из урока 06: редкая строка вместо смысла
    print(f"\nзапрос «{query}», верхние три чанка: раздел и есть ли в тексте E05")
    vector, lexical = rank_vector(query), rank_bm25(query)
    for name, ranking in (("вектор", vector), ("BM25", lexical), ("гибрид", rrf(vector, lexical))):
        print(f"  {name:8} {[(chunks[i]['section'], 'E05' in chunks[i]['text']) for i in ranking[:3]]}")


def block2_rerank():
    from importlib import import_module

    from common import MODEL, client, cost_usd, load_rag_questions, stop_reason
    from schemas import Ranking

    rag = import_module("06_rag_basics")
    chunks, matrix = rag.load_index()
    bm25 = BM25([f"{c['title']}. {c['section']}\n{c['text']}" for c in chunks])
    questions = [q for q in load_rag_questions() if q["docs"] and not q.get("multi")]  # многотемные меряются иначе, блок 3

    def candidates(query, top=10):
        vector = list(np.argsort(-(rag.embed([query])[0] @ matrix.T))[:20])
        return rrf(vector, bm25.top(query))[:top]

    spent = {"usd": 0.0, "calls": 0}

    def rerank(query, indexes):
        """Списочный реранкер на чат-модели: замена отдельной модели-реранкера, см. раздел 4.1."""
        listing = "\n\n".join(f"[{n}] {chunks[i]['title']}. {chunks[i]['section']}\n{chunks[i]['text'][:400]}"
                              for n, i in enumerate(indexes, 1))
        resp = client.responses.parse(
            model=MODEL, store=False,
            instructions="Вы ранжируете фрагменты документации по полезности для ответа на вопрос. "
                         "Верните номера фрагментов от самого полезного к самому бесполезному.",
            input=f"Вопрос: {query}\n\n{listing}",
            text_format=Ranking, max_output_tokens=500, reasoning={"effort": "none"}, temperature=0,
        )
        spent["usd"] += cost_usd(resp.usage, MODEL)
        spent["calls"] += 1
        if stop_reason(resp):  # отказ приходит мимо схемы (урок 02): остаётся порядок гибрида
            return indexes
        valid = [n for n in resp.output_parsed.order if 1 <= n <= len(indexes)]
        order = list(dict.fromkeys(valid))  # повторы номеров убираем, порядок первого появления сохраняется
        order += [n for n in range(1, len(indexes) + 1) if n not in order]  # модель могла потерять номер
        return [indexes[n - 1] for n in order]

    for name, rank in (("гибрид", candidates), ("гибрид + реранкер", lambda q: rerank(q, candidates(q)))):
        hits = {1: 0, 3: 0}
        reciprocal = 0.0
        started = time.perf_counter()
        for q in questions:
            found = [chunks[i]["doc"] in q["docs"] for i in rank(q["question"])]
            for k in hits:
                hits[k] += any(found[:k])
            reciprocal += 1 / (found.index(True) + 1) if True in found else 0
        n = len(questions)
        print(f"{name:18} recall@1 {hits[1]:2}/{n}  recall@3 {hits[3]:2}/{n}  MRR {reciprocal / n:.3f}"
              f"  {time.perf_counter() - started:5.1f} с")
    print(f"реранкер: вызовов модели {spent['calls']}, ${spent['usd']:.4f}")


def block3_agentic_search():
    from importlib import import_module

    from common import MODEL, client, cost_usd, final_text, load_rag_questions, stop_reason

    rag = import_module("06_rag_basics")
    chunks, matrix = rag.load_index()
    bm25 = BM25([f"{c['title']}. {c['section']}\n{c['text']}" for c in chunks])

    def search(query, k=4):
        vector = list(np.argsort(-(rag.embed([query])[0] @ matrix.T))[:20])
        return [chunks[i] for i in rrf(vector, bm25.top(query))[:k]]

    TOOLS = [{
        "type": "function", "name": "search_kb",
        "description": "Ищет фрагменты документов магазина. Формулируйте запрос словами документа, а не клиента. "
                       "Если вопрос про несколько тем, вызывайте по разу на каждую тему.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
                       "required": ["query"], "additionalProperties": False},
        "strict": True}]
    SYSTEM = ("Вы ассистент поддержки интернет-магазина. Отвечайте только по найденным фрагментам.\n"
              "- Если нужных сведений нет, так и скажите. Не догадывайтесь.\n"
              "- В конце ответа перечислите id использованных документов в квадратных скобках.")

    def answer(question, max_iterations=4):
        items = [{"role": "user", "content": question}]  # системный промпт идёт отдельно, в instructions
        queries, found, spent = [], set(), 0.0
        for _ in range(max_iterations):
            resp = client.responses.create(model=MODEL, instructions=SYSTEM, input=items, tools=TOOLS, store=False,
                                           max_output_tokens=1000, reasoning={"effort": "low"})
            spent += cost_usd(resp.usage, MODEL)
            items += [item.to_dict() for item in resp.output]  # весь output обратно в input: reasoning и вызовы
            stop = stop_reason(resp)
            if stop:  # обрыв по лимиту или отказ: исключения нет, но и ответа нет
                return f"Ответа нет, причина остановки: {stop}.", queries, found, spent
            calls = [item for item in resp.output if item.type == "function_call"]
            if not calls:
                return final_text(resp), queries, found, spent
            for call in calls:
                query = json.loads(call.arguments)["query"]
                hits = search(query)
                queries.append(query)
                found |= {c["doc"] for c in hits}
                payload = "\n\n".join(f'<fragment doc="{c["doc"]}" section="{c["section"]}">\n{c["text"]}\n</fragment>' for c in hits)
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": payload})
        return "Не удалось собрать ответ за отведённое число шагов.", queries, found, spent

    one_pass = wide_pass = agentic = 0
    for q in [q for q in load_rag_questions() if q.get("multi")]:  # здесь docs значит «нужны все», а не «любой из»
        plain = {c["doc"] for c in search(q["question"])}  # один поиск по вопросу как есть, как в уроке 06
        wide = {c["doc"] for c in search(q["question"], k=12)}  # контроль: объём, который агент набирает за три запроса
        text, queries, found, spent = answer(q["question"])
        one_pass += set(q["docs"]) <= plain  # <= на множествах: все нужные документы входят в найденные
        wide_pass += set(q["docs"]) <= wide
        agentic += set(q["docs"]) <= found
        print(f"{q['id']}: {q['question']}")
        print(f"  один поиск нашёл всё нужное: k=4 {set(q['docs']) <= plain}, k=12 {set(q['docs']) <= wide}")
        print(f"  запросы агента: {queries}")
        print(f"  агент нашёл всё нужное: {set(q['docs']) <= found}, ${spent:.5f}")
        print(f"  ответ: {' '.join(text.split())[:220]}\n")
    print(f"оба нужных документа найдены: один поиск k=4 {one_pass}/5, один поиск k=12 {wide_pass}/5, агент {agentic}/5")


if __name__ == "__main__":
    blocks = {
        "1": ("Гибридный поиск и метрики ранжирования", block1_hybrid),
        "2": ("Реранкинг", block2_rerank),
        "3": ("Поиск как инструмент агента", block3_agentic_search),
    }
    for number in sys.argv[1:] or blocks:
        title, run = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
