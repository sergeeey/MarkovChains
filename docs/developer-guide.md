# Книга нового разработчика — ChernoffPy

> Этот документ позволяет за 30 минут войти в проект с нуля.
> Подробности — в `docs/architecture.md` и `docs/` сайте.

---

## 1. Что это и зачем

ChernoffPy — Python-библиотека для ценообразования финансовых деривативов через метод
оператор-полугрупп Чернова. Научная новизна: **гарантированная верхняя граница ошибки**
(`CertifiedBound`), которая доказуемо ограничивает отклонение от точной цены.

Математическая основа: теорема Галкина-Ремизова (Israel J. Math., 2025) — если схема Чернова
`C(t)` совпадает с `e^{tL}` по `k` первым производным при `t=0`, то ошибка `C(t/n)^n f`
убывает как `O(1/n^k)`. Скорость сходимости **доказана**, не просто наблюдаема численно.

---

## 2. Развернуть локально за 5 минут

```bash
git clone https://github.com/sergeeey/MarkovChains.git
cd MarkovChains/ChernoffPy

# Минимальная установка
pip install -e ".[dev]"

# Запустить тесты
pytest tests/ -q
# Ожидаемый результат: 539 passed

# С Numba-ускорением (опционально, медленная установка)
pip install -e ".[fast,dev]"

# Собрать документацию локально
pip install -e ".[docs]"
mkdocs serve          # → http://127.0.0.1:8000
```

---

## 3. Архитектура за 10 минут

```
chernoffpy/               ← математическое ядро (ПУБЛИЧНЫЙ API)
├── functions.py          ChernoffFunction ABC + 5 реализаций (BE, CN, Padé, G, S)
├── semigroups.py         HeatSemigroup — точное решение для сравнения
├── certified.py          CertifiedBound, n_steps_for_tolerance
├── analysis.py           convergence_table, compute_errors
├── accel.py              thomas_solve_batch, mixed_deriv_step (+ Numba JIT)
├── backends.py           NumPy / CuPy переключатель
└── finance/              ← финансовые приложения (не импортируют core обратно)
    ├── validation.py     MarketParams, GridConfig, PricingResult (dataclasses)
    ├── transforms.py     Wilmott substitution: BS ↔ Heat equation
    ├── european.py       EuropeanPricer
    ├── barrier*.py       BarrierFFTPricer, BarrierDSTPricer, double_barrier
    ├── american.py       AmericanPricer (early exercise projection)
    ├── heston*.py        HestonPricer, HestonFastPricer
    ├── bates.py          BatesPricer (Heston + Merton jumps)
    ├── local_vol.py      LocalVolPricer
    ├── greeks.py         Delta, Gamma, Vega, Theta, Rho (finite differences)
    ├── calibration.py    LocalVolSurface
    └── dividends.py      DividendSchedule (absolute / proportional)
```

**Главный инвариант:** `finance/` → `chernoffpy/`, обратно нельзя.

**Главный поток данных** (EuropeanPricer):

```
MarketParams → Wilmott substitution → u0 (heat space)
→ C(τ/n)^n u0 (n итераций Chernoff) → обратная замена → цена + CertifiedBound
```

---

## 4. Ключевые решения (почему так, а не иначе)

| Решение | Почему |
|---|---|
| `ChernoffFunction` — ABC, не callable | Нужны `order`, `name` для `certified.py`; рефлексия без dict |
| `BarrierDSTPricer` использует DST, не FFT | FFT → периодические ГУ → эффект Гиббса у барьера |
| `transforms.py` — отдельный модуль | Подстановка Уилмотта — источник числовых ошибок; тестируется изолированно |
| `CertifiedBound` в публичном API | Главное научное преимущество перед MC и FD; нельзя скрывать |
| Early exercise через `max(u, intrinsic)` | Изменение сетки несовместимо с `C(t/n)^n` — нужно одно и то же пространство |

Полные ADR — в `docs/architecture.md`.

---

## 5. Известные проблемы (не трогать без понимания)

| Проблема | Симптом | Обходной путь |
|---|---|---|
| `AmericanPricer` call + абсолютные дивиденды | Цена ~$148 вместо ~$9, выбрасывает `UserWarning` | Использовать `proportional=True` |
| `PadeChernoff(m, n)` с `m > n` | Не A-устойчив, расхождение на высоких частотах, `UserWarning` | Использовать `m ≤ n` |
| CuPy-бэкенд не работает в Finance-слое | `get_backend("cupy")` работает, но Heston/American используют NumPy напрямую | Только NumPy в Finance |

---

## 6. Как добавить новый приcer (пошагово)

1. Создать `chernoffpy/finance/my_pricer.py`
2. Принять `ChernoffFunction` как первый аргумент `__init__`
3. Использовать `MarketParams` и `GridConfig` из `validation.py`
4. Возвращать `PricingResult` или аналогичный dataclass
5. Добавить аналитический reference в `finance/*_analytical.py` для сравнения в тестах
6. Написать тесты в `tests/` (сравнение с аналитикой, convergence test)
7. Добавить в `__init__.py` и `docs/api/finance.md`

---

## 7. Как выпустить релиз

Полная инструкция: [`release-guide.md`](release-guide.md).

Кратко:
```bash
# 1. Обновить версию в chernoffpy/__init__.py и pyproject.toml
# 2. Обновить CHANGELOG.md
# 3. Закоммитить
git add -u && git commit -m "chore: bump version to v0.2.0"
# 4. Тег → CI публикует на PyPI автоматически
git tag v0.2.0 && git push origin main --tags
```

---

## 8. CI/CD: что происходит при push

| Событие | Jobs |
|---|---|
| Push/PR в `main` | `test` (8 матриц) + `test-numba` |
| Push в `main` | + `docs` (деплой GitHub Pages) |
| Push тега `v*.*.*` | + `publish` (PyPI через OIDC) |

Файл: `.github/workflows/ci.yml`

---

## 9. Структура тестов

```
tests/
├── test_functions.py       ChernoffFunction API, сходимость, порядки
├── test_semigroups.py      HeatSemigroup точность
├── test_certified.py       CertifiedBound логика
├── test_european.py        EuropeanPricer vs BS-exact
├── test_barrier*.py        барьерные типы vs аналитика
├── test_american.py        AmericanPricer, BAW, early exercise
├── test_heston*.py         Heston vs аналитическая CF
├── test_bates.py           Bates vs CF
├── test_greeks.py          греки vs finite differences
└── test_dividends.py       дивиденды, proportional/absolute
```

Запуск конкретной группы:
```bash
pytest tests/test_european.py -v
pytest tests/ -k "barrier" -v
```

---

## 10. Контакты и Bus Factor

| Роль | Кто |
|---|---|
| Автор кода | Sergey (`@sergeeey` на GitHub) |
| Математическая теория | И.Д. Ремизов, О.Е. Галкин (ссылки в `docs/theory.md`) |
| Issues / PR | https://github.com/sergeeey/MarkovChains/issues |

**Bus factor: 1.** При необходимости передачи — подготовить второго мейнтейнера через 2–3
совместных code review + разбор `docs/architecture.md`.

---

## Карта всей документации

```
ChernoffPy/
├── README.md                    Обзор, badges, быстрый старт, decision guide
├── CONTRIBUTING.md              Стаб → docs/contributing.md (для GitHub баннера)
├── docs/
│   ├── index.md                 Главная MkDocs
│   ├── getting-started.md       Установка, первые 5 примеров
│   ├── architecture.md          Архитектура, ADR, CI/CD, ограничения, TODO
│   ├── theory.md                Математика: формула Чернова, Уилмотт, ссылки
│   ├── developer-guide.md       ← этот файл: онбординг за 30 минут
│   ├── contributing.md          Как контрибьютить (setup, style, PR process)
│   ├── release-guide.md         Как выпускать релизы
│   ├── changelog.md             История версий
│   ├── user-guide/
│   │   ├── choosing-pricer.md   Decision guide (когда что использовать)
│   │   ├── european.md
│   │   ├── barriers.md
│   │   ├── american.md
│   │   ├── heston-bates.md
│   │   └── certified-bounds.md
│   └── api/
│       ├── functions.md         API ref: core (mkdocstrings)
│       ├── finance.md           API ref: pricers (mkdocstrings)
│       └── certified.md        API ref: bounds (mkdocstrings)
└── examples/
    ├── american_with_dividends.py
    └── heston_calibration.py
```
