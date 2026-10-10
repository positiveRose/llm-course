"""Урок 05. Документы и мультимодальный ввод: из файла в чистый текст с метаданными.

Блоки запускаются по одному: python 05_documents.py 1, затем 2, 3 и 4.
Блоки 1, 2 и 4 работают без ключа, блоку 3 нужен OPENAI_API_KEY и, для сравнения, установленный tesseract.
Из пакетов нужен pymupdf.
Исходные файлы лежат в data/incoming/, результат блока 2 пишется в data/parsed.jsonl, блока 4 в data/parsed_book.jsonl.
Книгу для блока 4 собирает tools/make_book_fixture.py.
"""

import base64
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

INCOMING = Path(__file__).parent / "data" / "incoming"
PARSED = Path(__file__).parent / "data" / "parsed.jsonl"
PARSED_BOOK = Path(__file__).parent / "data" / "parsed_book.jsonl"
HEADER_BAND, FOOTER_BAND = 70, 750  # верх и низ страницы в пунктах (1/72 дюйма): туда попадают колонтитулы


def clean(text: str) -> str:
    """Чинит типичные артефакты извлечения: неразрывные пробелы, дефис в виде U+00AD, пробел нулевой ширины, склейку слов."""
    text = text.replace("\xa0", " ").replace("\xad", "-").replace("\u200b", "")  # в PDF символом \xad часто записан обычный дефис
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # слово, разорванное переносом в конце строки
    text = re.sub(r"(?<![.!?:;»\n])\n(?![\n-])", " ", text)  # перенос строки внутри предложения это пробел
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def page_blocks(path: Path, crop: bool = True) -> list[list[str]]:
    """Текстовые блоки каждой страницы PDF. При crop=True отсекает колонтитулы по координате блока."""
    import pymupdf

    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        blocks = page.get_text("blocks")
        if crop:
            blocks = [b for b in blocks if b[1] > HEADER_BAND and b[3] < FOOTER_BAND]
        blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))  # порядок чтения: сверху вниз, внутри строки слева направо
        pages.append([b[4].strip() for b in blocks if b[4].strip()])
    doc.close()
    return pages


def drop_repeating(pages: list[list[str]]) -> set[str]:
    """Блоки, которые встречаются больше чем на половине страниц. Так колонтитул ловится без координат."""
    seen = Counter(block for page in pages for block in set(page))  # set: повтор внутри страницы считается один раз
    return {block for block, n in seen.items() if n > len(pages) / 2 and len(pages) > 2}


def block1_pdf():
    import pymupdf

    for name in ("reglament-garantii.pdf", "reglament-garantii-skan.pdf"):
        doc = pymupdf.open(INCOMING / name)
        chars = [len(p.get_text().strip()) for p in doc]
        print(f"{name}: {doc.page_count} страниц, символов на странице {chars}")
        print("   текстовый слой есть" if sum(chars) / doc.page_count > 100 else "   текстового слоя нет: это скан, блок 3")
        doc.close()

    raw = page_blocks(INCOMING / "reglament-garantii.pdf", crop=False)
    print("\nбез фильтра, стык страниц 1 и 2:")
    for block in raw[0][-2:] + raw[1][:2]:
        print(f"   {clean(block)[:96]!r}")
    boilerplate = drop_repeating(raw)
    print(f"повторяется на большинстве страниц: {len(boilerplate)} блоков")
    for block in sorted(boilerplate):
        print(f"   {block[:80]!r}")  # сырой вид: в нём видны \xa0 и \xad, с которыми работает clean

    pages = page_blocks(INCOMING / "reglament-garantii.pdf")
    print(f"\nс фильтром по координатам блоков на страницах остаётся {[len(p) for p in pages]} блоков")
    body = clean("\n\n".join(block for page in pages for block in page if block not in boilerplate))
    print(f"чистого текста {len(body)} символов, первые абзацы:")
    for paragraph in body.split("\n\n")[:3]:
        print(f"   {paragraph[:96]}")


class Extractor(HTMLParser):
    """Достаёт из HTML заголовки и абзацы основного текста, выбрасывая служебные блоки."""

    SKIP = {"script", "style", "nav", "header", "footer", "aside", "form"}
    KEEP = {"h1", "h2", "h3", "p", "th", "td", "li"}

    def __init__(self):
        super().__init__()
        self.parts, self.skip_depth, self.tag, self.buf = [], 0, None, []

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class") or ""
        wrapper = tag in self.SKIP or any(c in classes for c in ("promo", "sidebar", "related"))
        if wrapper or self.skip_depth and tag in ("div", "aside"):  # вложенный div считаем тоже, иначе выйдем из пропуска раньше
            self.skip_depth += 1  # служебный блок вместе со всем содержимым
        elif tag in self.KEEP and not self.skip_depth:
            self.tag, self.buf = tag, []

    def handle_endtag(self, tag):
        if tag in self.SKIP or self.skip_depth and tag in ("div", "aside"):
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag == self.tag:
            text = clean(" ".join(self.buf))
            if text:
                self.parts.append({"h1": "# ", "h2": "## ", "h3": "## "}.get(tag, "") + text)
            self.tag, self.buf = None, []

    def handle_data(self, data):
        if self.tag and not self.skip_depth:
            self.buf.append(data.replace("\xad", ""))  # в HTML это настоящий мягкий перенос (&shy;), в тексте он не нужен


def block2_html_and_metadata():
    raw = (INCOMING / "usloviya-dostavki.html").read_text(encoding="utf-8")
    parser = Extractor()
    parser.feed(raw)
    print(f"html: {len(raw)} символов, после чистки {sum(len(p) for p in parser.parts)}")
    for part in parser.parts[:6]:
        print(f"   {part[:90]}")

    documents, parsed_at = [], date.today().isoformat()  # дата разбора: по ней потом поймут, от какой версии текст
    for name, title in (("reglament-garantii.pdf", "Регламент гарантийного обслуживания"),
                        ("pravila-loyalnosti.pdf", "Правила программы лояльности")):
        pages = page_blocks(INCOMING / name)
        boilerplate = drop_repeating(pages)  # координатный фильтр уже сработал, этот страхует на других файлах
        for number, page in enumerate(pages, 1):  # номер страницы это происхождение: по нему потом найдут оригинал
            text = clean("\n\n".join(block for block in page if block not in boilerplate))
            if text:
                documents.append({"id": f"{Path(name).stem}-p{number}", "title": title, "text": text,
                                  "source": {"file": name, "type": "pdf", "page": number, "parsed_at": parsed_at}})
    documents.append({"id": "usloviya-dostavki-web", "title": "Условия доставки",
                      "text": "\n\n".join(p for p in parser.parts if not p.startswith("# ")),
                      "source": {"file": "usloviya-dostavki.html", "type": "html", "page": None, "parsed_at": parsed_at}})
    with PARSED.open("w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n{PARSED.name}: {len(documents)} документов")
    for d in documents:
        print(f"   {d['id']:34} {len(d['text']):5} символов  {d['source']['type']} стр. {d['source']['page']}")


def ocr(png: bytes, lang: str = "rus") -> str | None:
    """Распознаёт страницу через tesseract. Возвращает None, если его нет в системе или нет языкового пакета."""
    try:
        done = subprocess.run(["tesseract", "-", "-", "-l", lang], input=png, capture_output=True, timeout=120)
    except FileNotFoundError:
        return None
    if done.returncode:  # чаще всего не установлен пакет языка: список покажет tesseract --list-langs
        print("   tesseract:", done.stderr.decode("utf-8", "replace").strip().splitlines()[0])
        return None
    return clean(done.stdout.decode("utf-8", "replace"))


def recognized_by_model(png: bytes) -> tuple[str, object]:
    """Просит модель переписать страницу в Markdown. Картинка уходит в сообщение как data-ссылка."""
    from common import MODEL, client, final_text

    url = f"data:image/png;base64,{base64.b64encode(png).decode()}"
    resp = client.responses.create(
        model=MODEL, store=False,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": "Перепишите текст страницы в Markdown. Заголовки разделов отметьте ##. "
                                           "Колонтитулы и номер страницы пропустите. Ничего не добавляйте от себя."},
            {"type": "input_image", "image_url": url, "detail": "original"}]}],  # original: без уменьшения
        max_output_tokens=2000, reasoning={"effort": "none"}, temperature=0,
    )
    return clean(final_text(resp)), resp


def block3_scan():
    import pymupdf

    from common import MODEL, cost_usd, usage_dict

    scan = pymupdf.open(INCOMING / "reglament-garantii-skan.pdf")
    png = scan[0].get_pixmap(dpi=150).tobytes("png")  # модель и tesseract принимают картинку, а не PDF
    original = clean("\n\n".join(page_blocks(INCOMING / "reglament-garantii.pdf")[0]))  # эталон: та же страница с текстовым слоем
    gold = set(re.findall(r"\w+", original.lower()))
    print(f"страница {len(png) // 1024} КБ png, эталон {len(original)} символов, {len(gold)} разных слов")

    for name in ("tesseract", MODEL):
        started = time.perf_counter()
        if name == "tesseract":
            text, price = ocr(png), 0.0
            if text is None:
                print("tesseract: не установлен или нет пакета rus, пропускаем")
                continue
        else:
            text, resp = recognized_by_model(png)
            price = cost_usd(resp.usage, MODEL)
            print(f"   вход запроса {usage_dict(resp.usage)['input']} токенов, почти все на картинку")
        found = set(re.findall(r"\w+", text.lower()))
        print(f"{name}: {len(text)} символов, слов эталона найдено {len(found & gold) / len(gold):.0%}, "
              f"лишних слов {len(found - gold)}, {time.perf_counter() - started:.1f} с, ${price:.5f}")
        print("   " + " ".join(text.split())[:150])


def mask(text: str) -> str:
    """Текст блока без различий в цифрах и пробелах: «12 Справочник» и «14 Справочник» дают одну строку."""
    return re.sub(r"\d+", "#", " ".join(text.split()))


def head_bands(doc, zone: float = 0.12) -> tuple[float, float]:
    """Где в этой книге живут колонтитулы. Ищем у краёв страницы блоки, которые повторяются с точностью до цифр."""
    edge = [(mask(b[4]), b[1], b[3], page.rect.height) for page in doc for b in page.get_text("blocks")
            if b[3] < page.rect.height * zone or b[1] > page.rect.height * (1 - zone)]  # zone это доля высоты, не пункты
    repeated = {text for text, n in Counter(text for text, *_ in edge).items() if n >= 2}
    top = max((y1 for text, y0, y1, h in edge if text in repeated and y1 < h / 2), default=0)
    bottom = min((y0 for text, y0, y1, h in edge if text in repeated and y0 > h / 2), default=float("inf"))
    return top, bottom  # всё, что выше top или ниже bottom, это колонтитул, даже если он встретился один раз


def book_blocks(doc, bands: tuple[float, float], columns: bool = True) -> list[tuple[int, str]]:
    """Блоки книги в порядке чтения, с номером страницы. Колонтитулы сняты, колонки прочитаны по очереди."""
    out = []
    for number, page in enumerate(doc, 1):
        blocks = [b for b in page.get_text("blocks") if b[4].strip() and b[1] > bands[0] and b[3] < bands[1]]
        middle = page.rect.width / 2
        two = columns and sum(b[0] > middle for b in blocks) >= 2  # в одну колонку блоки не начинаются правее середины
        blocks.sort(key=lambda b: (two and b[0] > middle, round(b[1] / 10), b[0]))
        out += [(number, b[4].strip()) for b in blocks]
    return out


def glue_torn(blocks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Абзац, разорванный границей страницы или колонки, целиком отходит странице, где он начался."""
    out = []
    for number, text in blocks:
        if out and text[:1].islower():  # блок с маленькой буквы продолжает предложение предыдущего
            joint = "" if out[-1][1].endswith(("-", "\xad")) else " "  # «GPS-» и «тренировка» это одно слово
            out[-1] = (out[-1][0], out[-1][1] + joint + text)
        else:
            out.append((number, text))
    return out


def book_documents(path: Path) -> list[dict]:
    """Книга в формате parsed.jsonl: документ на страницу, в происхождении глава из оглавления PDF и дата разбора."""
    import pymupdf

    doc = pymupdf.open(path)
    toc = doc.get_toc()  # строки вида [уровень, заголовок, страница]
    sections = {name for level, name, _ in toc if level > 1}
    blocks = glue_torn(book_blocks(doc, head_bands(doc)))
    documents = []
    for number in sorted({n for n, _ in blocks}):
        parts = [clean(text) for n, text in blocks if n == number]
        chapter = [name for level, name, start in toc if level == 1 and start <= number]
        documents.append({"id": f"{path.stem}-p{number}", "title": doc.metadata["title"] or path.stem,
                          "text": "\n\n".join("## " + p if p in sections else p for p in parts),
                          "source": {"file": path.name, "type": "pdf", "page": number,
                                     "chapter": chapter[-1] if chapter else None,
                                     "parsed_at": date.today().isoformat()}})
    return documents


def layer_kind(page) -> str:
    """Три вида страницы: текстовая, скан без слоя и скан, поверх которого лежит слой распознавания."""
    picture = max((r.get_area() for img in page.get_images() for r in page.get_image_rects(img[0])), default=0)
    if len(page.get_text().strip()) < 100:
        return "скан без текстового слоя"
    return "скан со слоем распознавания" if picture > 0.8 * page.rect.get_area() else "текстовая"


def block4_book():
    import pymupdf

    from common import load_kb_long

    path = INCOMING / "spravochnik-pokupatelya.pdf"
    book = pymupdf.open(path)
    toc = book.get_toc()
    print(f"страница {book[0].rect.width:.0f}×{book[0].rect.height:.0f} пунктов, {book.page_count} страниц, "
          f"в оглавлении {len(toc)} строк, из них глав {sum(level == 1 for level, *_ in toc)}")
    for number in (11, 12, 13):  # первая страница главы, чётная и нечётная
        first, last = (sorted(book[number - 1].get_text("blocks"), key=lambda b: b[1])[i] for i in (0, -1))
        print(f"   стр. {number}: сверху y={first[1]:.0f}–{first[3]:.0f} {mask(first[4])[:44]!r}, "
              f"снизу y={last[1]:.0f}–{last[3]:.0f} {mask(last[4])[:24]!r}")
    print(f"повторяется больше чем на половине страниц: {len(drop_repeating(page_blocks(path, crop=False)))} блоков")
    bands = head_bands(book)
    print(f"колонтитулы этой книги: выше y={bands[0]:.0f} и ниже y={bands[1]:.0f}")
    scan = pymupdf.open(INCOMING / "spravochnik-skan-ocr.pdf")
    print(f"\n{path.name}: {layer_kind(book[8])}")
    print(f"{'spravochnik-skan-ocr.pdf'}: {layer_kind(scan[0])}, в слое {' '.join(scan[0].get_text().split())[:80]!r}")

    gold = []  # эталон: абзацы исходных документов, из которых свёрстана книга
    for d in load_kb_long():
        for part in re.split(r"\n\n+", d["text"]):
            if not part.startswith(("## ", "|")):  # заголовки и таблицу не считаем: мерим абзацы
                lines = part.splitlines() if re.match(r"^(- |\d+\. )", part) else [part]
                gold += [" ".join(re.sub(r"\*\*|^- ", "", line).split()) for line in lines]
    gold_words = set(re.findall(r"\w+", " ".join(gold).lower()))

    def report(name: str, blocks: list[tuple[int, str]]) -> list[str]:
        pages = [" ".join(clean("\n\n".join(t for n, t in blocks if n == number)).split())
                 for number in sorted({n for n, _ in blocks})]  # документ это страница, поэтому ищем абзац внутри одной
        found = set(re.findall(r"\w+", " ".join(pages).lower())) & gold_words
        whole = sum(any(p in page for page in pages) for p in gold)
        print(f"{name:32} абзацев целиком {whole:3}/{len(gold)} = {whole / len(gold):4.0%}   "
              f"слов эталона {len(found) / len(gold_words):4.0%}")
        return pages

    print()
    report("как в блоке 1: полосы 70 и 750", [(n, t) for n, page in enumerate(page_blocks(path), 1) for t in page])
    report("+ полосы по самой книге", book_blocks(book, bands, columns=False))
    report("+ колонки", book_blocks(book, bands))
    pages = report("+ склейка разорванных абзацев", glue_torn(book_blocks(book, bands)))
    print("не дошли целиком:")
    for p in (p for p in gold if not any(p in page for page in pages)):
        print(f"   {p[:100]}")

    documents = book_documents(path)
    with PARSED_BOOK.open("w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n{PARSED_BOOK.name}: {len(documents)} документов из {book.page_count} страниц")
    print(f"   происхождение: {documents[12]['source']}")


if __name__ == "__main__":
    blocks = {
        "1": ("PDF: текстовый слой, колонтитулы, чистка", block1_pdf),
        "2": ("HTML и метаданные происхождения", block2_html_and_metadata),
        "3": ("Скан: распознавание и модель", block3_scan),
        "4": ("Книга: другой формат страницы, главы, колонки", block4_book),
    }
    for number in sys.argv[1:] or blocks:
        title, run = blocks[number]
        if number == "3" and not os.environ.get("OPENAI_API_KEY"):
            print("\n=== Блок 3 пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        run()
