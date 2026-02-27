# activeContext — ChernoffPy

**Последнее обновление:** 2026-02-27

## Статус проекта

Оценка: **~9.3/10** (было 8.7 при старте работы)

## Что сделано (все сессии)

### Сессия 1 — документация и примеры
- ✅ README: badges (CI, Coverage) + Decision Guide "когда какой pricer"
- ✅ CONTRIBUTING.md
- ✅ MkDocs docs/ (12 файлов: index, getting-started, 6 user-guide, 3 API ref, theory)
- ✅ mkdocs.yml (Material theme + mkdocstrings + MathJax)
- ✅ pyproject.toml: `[docs]` extras, Documentation URL
- ✅ examples/american_with_dividends.py
- ✅ examples/heston_calibration.py

### Сессия 2 — CI и деплой
- ✅ .github/workflows/ci.yml: `docs` job → автодеплой MkDocs на GitHub Pages при push в main
- ✅ .gitignore: добавлен `site/`
- ✅ Коммит всех файлов двух сессий (8e0db80)

### Сессия 3 — архитектурный документ + фиксы пробелов
- ✅ Architecture Document (в чате, не в файле)
- ✅ Fix #1: `requires-python = ">=3.10"` (было 3.11, CI тестирует с 3.10)
- ✅ Fix #4: `warnings.warn(UserWarning)` для AmericanPricer call + absolute dividends
- ✅ Fix #6: `publish` job в CI (OIDC Trusted Publisher, тег v*.*.*)
- ✅ Коммит (0413f87)

## Текущее состояние репозитория

- Branch: `main`
- Последний коммит: `0413f87` — fix: resolve three documented gaps from architecture audit
- Тесты: 539 pass, 0 fail
- Coverage: ≥ 80%

## Что осталось из пробелов (из Architecture Doc)

| # | Пробел | Приоритет |
|---|--------|-----------|
| 2 | PadeChernoff m>n: Warning есть, unit-теста на Warning нет | Средний |
| 3 | CuPy backend не интегрирован в Finance-слой | Средний |
| 5 | Версия не синхронизирована через автоматику (bumpversion) | Низкий |
| 7 | PhysicalG/S — нет ссылки на конкретную теорему в docstring | Низкий |
| 8 | adaptive_grid.py — нет unit-тестов | Средний |
| 9 | LocalVolPricer — нет CertifiedBound | Средний |
| 10 | GitHub Pages — нужно включить вручную в Settings | Низкий |

## PyPI: что нужно сделать один раз

1. PyPI → Account → Publishing → Add Trusted Publisher
   - GitHub owner: `sergeeey`, repo: `MarkovChains`, workflow: `ci.yml`, env: `pypi`
2. `git tag v0.1.0 && git push --tags` → CI опубликует автоматически

## Команды

```bash
# Тесты
pytest tests/ -q

# Документация локально
mkdocs serve

# Новый релиз
git tag v0.1.0 && git push --tags
```
