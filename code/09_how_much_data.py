"""Урок 09. Сколько нужно данных: шум выборки, интервалы, сравнение версий, шум модели.

Блоки запускаются по одному: python 09_how_much_data.py 1, затем 2 и 3.
Блоки 1 и 2 это симуляции, ключ им не нужен. Блоку 3 нужен OPENAI_API_KEY.
Новых пакетов нет: симуляции на стандартном random, гистограммы рисуются символами.
wilson, sign_test и bootstrap лежат на уровне модуля: их импортируют следующие уроки.
"""

import math
import os
import random
import sys
from collections import Counter

rng = random.Random(9)  # фиксированное зерно: симуляции дают одни и те же числа при каждом запуске
SETS = 10_000  # сколько воображаемых наборов собирает симуляция

# --- Блок 1. Шум выборки ---------------------------------------------------------------------


def draw(p: float, n: int) -> int:
    """Один набор из n примеров: сколько из них система решила, если на самом деле она решает долю p."""
    return sum(rng.random() < p for _ in range(n))


def bars(counts: Counter, total: int, width: int = 40) -> None:
    """Гистограмма символами: строка на значение, длина полосы пропорциональна частоте."""
    peak = max(counts.values())
    for value in range(min(counts), max(counts) + 1):
        print(f"{value:5} {'█' * round(counts[value] / peak * width):{width}} {counts[value] / total:6.1%}")


def rough(hits: int, n: int) -> tuple[float, float]:
    """Доля плюс-минус margin of error: хороша около 50%, у краёв ломается."""
    p = hits / n
    half = 2 * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% интервал Уилсона для доли hits из n: не ломается у краёв. В statsmodels это proportion_confint(method="wilson")."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def coverage(interval, p: float, n: int) -> float:
    """Доля воображаемых наборов, в которых интервал накрыл настоящую долю p. У честного 95% интервала около 0.95."""
    covered = 0
    for _ in range(SETS):
        low, high = interval(draw(p, n), n)
        covered += low <= p <= high
    return covered / SETS


def block1_sampling_noise():
    rng.seed(9)  # каждый блок с того же зерна: вывод не зависит от того, шёл ли перед ним другой блок
    print("настоящая accuracy 90%, наборы по 40 тикетов, сколько решено:")
    bars(Counter(draw(0.9, 40) for _ in range(SETS)), SETS)
    for n in (20, 40, 100, 400):
        seen = sorted(draw(0.9, n) / n for _ in range(SETS))
        print(f"n={n:3}: в 95% наборов accuracy от {seen[int(SETS * 0.025)]:.0%} до {seen[int(SETS * 0.975)]:.0%}")

    print("\nкак часто интервал накрывает настоящую долю, должно быть около 95%:")
    for p, n in ((0.5, 40), (0.9, 40), (0.97, 40), (0.97, 20)):
        print(f"p={p:.2f} n={n}: доля ± margin of error {coverage(rough, p, n):.0%}, Уилсон {coverage(wilson, p, n):.0%}")

    print("\nноль успехов из n: верхняя граница Уилсона и правило трёх")
    for n in (9, 27, 60, 300):
        print(f"0 из {n:3}: Уилсон до {wilson(0, n)[1]:.1%}, 3/n = {3 / n:.1%}")


# --- Блок 2. Две версии на одних примерах ------------------------------------------------------

DIFFICULTY = [0.98] * 30 + [0.8] * 6 + [0.5] * 2 + [0.2] * 2  # 40 тикетов dev: с какой вероятностью решается каждый
TEST = DIFFICULTY[::2]  # 20 тикетов test той же сложности, настоящая accuracy та же, 89%


def run(difficulty: list[float]) -> list[bool]:
    """Один прогон версии: решён ли каждый тикет."""
    return [rng.random() < p for p in difficulty]


def sign_test(fixed: int, broken: int) -> float:
    """Sign test: шанс увидеть fixed и больше починенных из fixed + broken, если правка ничего не меняет."""
    n = fixed + broken
    return sum(math.comb(n, i) for i in range(fixed, n + 1)) / 2 ** n


def bootstrap(values: list[float], rounds: int = 2000) -> tuple[float, float]:
    """95% интервал для среднего: примеры пересобираются с возвращением. Годится для любой метрики по примерам."""
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def compare(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Парное сравнение по тикетам: сколько B починила, сколько сломала и p из sign test."""
    fixed = sum(y and not x for x, y in zip(a, b))
    broken = sum(x and not y for x, y in zip(a, b))
    return fixed, broken, sign_test(fixed, broken)


def race(a_difficulty: list[float], b_difficulty: list[float], sets: int = 2000) -> tuple[float, float]:
    """Доля наборов, где B обогнала A на 2+ тикета, и доля, где sign test признал разницу (p < 0.05)."""
    ahead = detected = 0
    for _ in range(sets):
        a, b = run(a_difficulty), run(b_difficulty)
        ahead += sum(b) - sum(a) >= 2
        detected += compare(a, b)[2] < 0.05
    return ahead / sets, detected / sets


def block2_two_versions():
    rng.seed(9)
    ahead, detected = race(DIFFICULTY, DIFFICULTY)
    print(f"версии одинаковы: B впереди на 2+ тикета в {ahead:.0%} наборов, sign test поверил в {detected:.1%}")
    better = [min(0.99, p + 0.15) for p in DIFFICULTY]  # B лучше на трудных тикетах: 93.5% против 89%
    for times in (1, 4, 10):
        ahead, detected = race(DIFFICULTY * times, better * times)
        print(f"B лучше на 4.5 п.п., тикетов {40 * times:3}: впереди на 2+ в {ahead:.0%}, sign test заметил в {detected:.0%}")

    print("пять пар прогонов на 400 тикетах, настоящая разница +4.5%:")
    for _ in range(5):
        a, b = run(DIFFICULTY * 10), run(better * 10)
        low, high = bootstrap([int(y) - int(x) for x, y in zip(a, b)])  # разница accuracy по тем же тикетам
        print(f"  разница {(sum(b) - sum(a)) / 400:+.1%}, интервал от {low:+.1%} до {high:+.1%}")

    dev_best = test_best = 0.0
    for _ in range(1000):  # 20 вариантов промпта, на самом деле одинаковых: выбираем лучший на dev, проверяем на test
        dev_best += max(sum(run(DIFFICULTY)) for _ in range(20)) / 40 / 1000
        test_best += sum(run(TEST)) / 20 / 1000  # победитель ничем не лучше остальных, на test это обычный прогон
    print(f"лучший из 20 одинаковых вариантов: на dev {dev_best:.0%}, на test {test_best:.0%}, на самом деле 89%")


# --- Блок 3. Шум модели ----------------------------------------------------------------------


def block3_model_noise():
    from concurrent.futures import ThreadPoolExecutor

    from common import MODEL, client, load_tickets
    from schemas import ROUTER_PROMPT

    ONE_WORD = "\n- Отвечайте одним словом в нижнем регистре, без пояснений и знаков препинания."  # как в уроке 01
    RUNS = 5
    tickets = load_tickets()[:40]  # dev из урока 01

    def correct(ticket, temperature):
        resp = client.responses.create(
            model=MODEL, instructions=ROUTER_PROMPT + ONE_WORD, input=ticket["text"], store=False,
            max_output_tokens=20, reasoning={"effort": "none"}, temperature=temperature)  # temperature работает только при none
        return resp.output_text.strip().lower() == ticket["category"]

    for temperature in (0.0, 1.0):
        with ThreadPoolExecutor(max_workers=8) as pool:  # вызовы независимы, параллелим
            runs = [list(pool.map(lambda t: correct(t, temperature), tickets)) for _ in range(RUNS)]
        solved = [sum(r[i] for r in runs) for i in range(len(tickets))]  # сколько раз из RUNS решён каждый тикет
        print(f"\ntemperature={temperature}: accuracy по прогонам {[sum(r) for r in runs]} из 40")
        print(f"  тикетов, решённых 0..{RUNS} раз из {RUNS}:")
        bars(Counter(solved), len(tickets))
        print("  нестабильные:", [t["id"] for t, s in zip(tickets, solved) if 0 < s < RUNS] or "нет")
        naive, by_ticket = wilson(sum(solved), 40 * RUNS), bootstrap([s / RUNS for s in solved])
        print(f"  интервал: по {40 * RUNS} прогонам как по независимым {naive[0]:.0%}–{naive[1]:.0%}, "
              f"по тикетам {by_ticket[0]:.0%}–{by_ticket[1]:.0%}")
        print(f"  pass@{RUNS} {sum(s > 0 for s in solved)}/40, pass^{RUNS} {sum(s == RUNS for s in solved)}/40")


if __name__ == "__main__":
    blocks = {
        "1": ("Шум выборки", block1_sampling_noise, False),
        "2": ("Две версии на одних примерах", block2_two_versions, False),
        "3": ("Шум модели", block3_model_noise, True),
    }
    for number in sys.argv[1:] or blocks:  # «python 09_how_much_data.py 2» запускает только второй блок
        title, go, needs_key = blocks[number]
        if needs_key and not os.environ.get("OPENAI_API_KEY"):
            print(f"\n=== Блок {number} пропущен: нужен OPENAI_API_KEY")
            continue
        print(f"\n=== Блок {number}. {title}")
        go()
