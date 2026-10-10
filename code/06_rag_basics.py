"""Урок 06. RAG: эмбеддинги, чанкинг, векторный индекс и ответ по найденному.

Блоки запускаются по одному: python 06_rag_basics.py 1, затем 2 и 3. Блок 2 строит индекс, блок 3 им пользуется.
Индекс лежит в rag_index.db рядом с кодом, в git его не кладут.
Нужен OPENAI_API_KEY. Дополнительных пакетов, кроме numpy, нет.
"""

import os
import re
import sqlite3
import sys

import numpy as np

INDEX_DB = "rag_index.db"


def embed(texts: list[str], dimensions: int = 512) -> np.ndarray:
    """Эмбеддинги списка текстов одним запросом. Возвращает матрицу (len(texts), dimensions)."""
    from common import EMBED_MODEL, client

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=dimensions)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)  # векторы уже нормированы провайдером


def block1_embeddings():
    query = "как вернуть деньги за товар, который не подошёл"
    candidates = [
        "Товар надлежащего качества можно вернуть в течение 14 дней, деньги приходят на карту за 1-10 рабочих дней.",
        "Подъём крупной техники на этаж без лифта стоит 150 рублей за этаж.",
        "Возврат денег: сроки, способы и что делать, если деньги не пришли.",
    ]
    vectors = embed([query] + candidates)  # один запрос на весь список: быстрее и расходует один запрос из лимита
    print(f"размерность вектора {vectors.shape[1]}, длина вектора {np.linalg.norm(vectors[0]):.3f}")
    for text, score in zip(candidates, vectors[0] @ vectors[1:].T):  # у нормированных векторов скалярное произведение это косинус
        print(f"  {score:.3f}  {text[:70]}")

    print("\nслепое пятно: строки одного вида выглядят одинаково")
    error_query = "пылесос пишет E22"
    codes = [
        "E22 - робот не нашёл базу за отведённое время. Проверьте свободное пространство вокруг станции.",
        "E05 - не установлен или не распознан пылесборник. Достаньте и установите его заново до щелчка.",
        "E12 - перегрев двигателя. Дайте роботу остыть 30 минут и очистите воздуховод.",
    ]
    vectors = embed([error_query] + codes)
    for text, score in sorted(zip(codes, vectors[0] @ vectors[1:].T), key=lambda p: -p[1]):
        print(f"  {score:.3f}  {text[:70]}")


# --- Блок 2. Чанкинг, индекс и измерение поиска. Функции ниже нужны и блоку 3 -------------------


def chunk(doc: dict, size: int = 120, overlap: int = 20) -> list[dict]:
    """Режет документ по абзацам, набирая куски примерно по size слов с перекрытием overlap слов.

    Заголовок документа и заголовок раздела запоминаются в чанке: без них фрагмент теряет контекст.
    """
    section, carry, current, out = doc["title"], [], [], []  # carry это хвост предыдущего чанка, current это новые абзацы

    def close():
        out.append((section, "\n\n".join(carry + current)))

    for part in re.split(r"\n\n+", doc["text"]):
        if part.startswith("## "):  # граница раздела: чанк закрывается под старым заголовком, перекрытие через неё не тянем
            if current:
                close()
            section, carry, current = part[3:].strip(), [], []
            continue
        if sum(len(c.split()) for c in carry + current) + len(part.split()) > size and current:
            close()
            tail = " ".join(" ".join(carry + current).split()[-overlap:])  # хвост уезжает в начало следующего чанка
            carry, current = ([tail] if overlap else []), []
        current.append(part)
    if current:
        close()
    return [{"doc": doc["id"], "title": doc["title"], "section": s, "text": t} for s, t in out]


def build_index(chunks: list[dict], path: str = INDEX_DB) -> None:
    """Считает эмбеддинги чанков пачками и складывает их в SQLite вместе с текстом и метаданными."""
    db = sqlite3.connect(path)
    db.executescript("DROP TABLE IF EXISTS chunks;"
                     "CREATE TABLE chunks (doc TEXT, title TEXT, section TEXT, text TEXT, vector BLOB);")
    with db:
        for i in range(0, len(chunks), 64):  # пачками: у эндпоинта есть лимит на размер запроса
            batch = chunks[i:i + 64]
            vectors = embed([f"{c['title']}. {c['section']}\n{c['text']}" for c in batch])  # в индекс едет заголовок
            db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                           [(c["doc"], c["title"], c["section"], c["text"], v.tobytes()) for c, v in zip(batch, vectors)])
    db.close()


def load_index(path: str = INDEX_DB) -> tuple[list[dict], np.ndarray]:
    db = sqlite3.connect(path)
    rows = db.execute("SELECT doc, title, section, text, vector FROM chunks").fetchall()
    db.close()
    chunks = [{"doc": d, "title": t, "section": s, "text": x} for d, t, s, x, _ in rows]
    return chunks, np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])


def search(query: str, chunks: list[dict], matrix: np.ndarray, k: int = 5) -> list[dict]:
    """Полный перебор: на сотне чанков это доли миллисекунды. ANN нужен начиная с сотен тысяч."""
    scores = embed([query])[0] @ matrix.T
    return [{**chunks[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k]]


def block2_chunk_and_measure():
    from common import load_kb, load_kb_long, load_rag_questions

    docs = load_kb() + load_kb_long()
    questions = [q for q in load_rag_questions() if q["docs"] and not q.get("multi")]  # без ответа в базе: блок 3; многотемные: урок 07

    def recall(chunks, matrix, k):
        hits = 0
        for q in questions:
            found = {c["doc"] for c in search(q["question"], chunks, matrix, k)}
            hits += bool(found & set(q["docs"]))
        return hits

    for size, overlap in ((400, 0), (120, 0), (120, 20), (60, 0)):  # пара 120/0 и 120/20 отличается только перекрытием
        chunks = [c for d in docs for c in chunk(d, size, overlap)]
        build_index(chunks)
        chunks, matrix = load_index()
        lengths = [len(c["text"].split()) for c in chunks]
        print(f"size={size:3} overlap={overlap:2}: чанков {len(chunks):3}, слов медиана {int(np.median(lengths)):3} "
              f"| recall@1 {recall(chunks, matrix, 1)}/{len(questions)} recall@3 {recall(chunks, matrix, 3)}/{len(questions)}")

    build_index([c for d in docs for c in chunk(d)])  # в индексе оставляем вариант по умолчанию: им пользуется блок 3
    chunks, matrix = load_index()
    for q in questions:  # разбор ошибок: где нужный документ не первый и что поиск поставил выше него
        found = search(q["question"], chunks, matrix, 3)
        if found[0]["doc"] not in q["docs"]:
            print(f"\nпромах при k=1, {q['id']}: {q['question']}")
            for c in found:
                mark = "+" if c["doc"] in q["docs"] else " "  # плюсом помечен чанк эталонного документа
                print(f"  {mark} {c['score']:.3f} {c['doc']} / {c['section']}")


def block3_answer_with_sources():
    from common import MODEL, client, final_text, load_rag_questions

    SYSTEM = (
        "Вы ассистент поддержки интернет-магазина. Отвечайте только по документам из раздела «Источники».\n"
        "- Если ответа в документах нет, ответьте ровно «Не знаю» и ничего не добавляйте.\n"
        "- Сроки и суммы называйте точно так, как они записаны в документах.\n"
        "- В конце ответа укажите id использованных документов в квадратных скобках.\n"
        "- Два-четыре предложения."
    )
    chunks, matrix = load_index()

    def answer(question, k=5):
        found = search(question, chunks, matrix, k)
        sources = "\n\n".join(f'<document id="{c["doc"]}" section="{c["section"]}">\n{c["text"]}\n</document>' for c in found)
        resp = client.responses.create(
            model=MODEL, instructions=f"{SYSTEM}\n\n# Источники\n{sources}", input=question, store=False,
            max_output_tokens=400, reasoning={"effort": "none"}, temperature=0,
        )
        return final_text(resp), found

    questions = {q["id"]: q for q in load_rag_questions()}
    for qid in ("Q-01", "Q-19", "Q-24", "Q-37", "Q-39"):
        q = questions[qid]
        text, found = answer(q["question"])
        cited = set(re.findall(r"[\w-]+", text.rsplit("[", 1)[-1])) if "[" in text else set()
        print(f"{qid}: {q['question']}")
        print(f"  найдено: {[c['doc'] for c in found]}, сходство первого {found[0]['score']:.3f}")
        print(f"  эталон:  {q['docs'] or 'ответа в базе нет'}")
        print(f"  ответ:   {text}")
        print(f"  ссылается на найденный документ: {bool(cited & {c['doc'] for c in found})}\n")


if __name__ == "__main__":
    blocks = {
        "1": ("Эмбеддинги и близость", block1_embeddings),
        "2": ("Чанкинг, индекс и измерение поиска", block2_chunk_and_measure),
        "3": ("Ответ по найденному с источниками", block3_answer_with_sources),
    }
    for number in sys.argv[1:] or blocks:
        title, run = blocks[number]
        if not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
