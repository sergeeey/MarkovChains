# Release Guide

Этот документ описывает полный процесс выпуска новой версии ChernoffPy от коммита до PyPI.

---

## Предварительные требования (один раз)

### GitHub Actions: OIDC Trusted Publisher

Современный способ публикации без хранения секретов в репозитории.
Настраивается один раз на PyPI:

1. Войти на [pypi.org](https://pypi.org)
2. **Account Settings → Publishing → Add a new pending publisher**
3. Заполнить форму:

| Поле | Значение |
|---|---|
| PyPI Project Name | `chernoffpy` |
| Owner | `sergeeey` |
| Repository name | `MarkovChains` |
| Workflow name | `ci.yml` |
| Environment name | `pypi` |

4. Сохранить

После этого теги `v*.*.*` будут публиковаться автоматически без паролей и токенов.

### GitHub Pages

Настраивается один раз в репозитории:

1. GitHub → репозиторий → **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **`gh-pages`**, Directory: **`/ (root)`**
4. Save

---

## Процесс релиза

### Шаг 1. Убедиться что всё зелёное

```bash
# Полный прогон тестов локально
pytest tests/ -q --tb=short

# Убедиться что нет незакоммиченных изменений
git status
```

Проверить GitHub Actions на ветке `main`: все jobs зелёные.

### Шаг 2. Обновить версию

Версия хранится в **двух местах** — обновить оба:

```bash
# chernoffpy/__init__.py
__version__ = "0.2.0"

# pyproject.toml
version = "0.2.0"
```

### Шаг 3. Обновить CHANGELOG.md

Формат [Keep a Changelog](https://keepachangelog.com/):

```markdown
## [0.2.0] - 2026-03-15

### Added
- ...

### Fixed
- ...

### Changed
- ...

### Breaking Changes
- ...
```

!!! warning "Breaking changes"
    Если есть ломающие изменения — описать явно в `### Breaking Changes`.
    Семантическое версионирование: `MAJOR.MINOR.PATCH`.
    Breaking change → MAJOR, новый функционал → MINOR, bugfix → PATCH.

### Шаг 4. Коммит

```bash
git add chernoffpy/__init__.py pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to v0.2.0"
git push origin main
```

Дождаться прохождения CI на `main`.

### Шаг 5. Создать тег и запустить публикацию

```bash
git tag v0.2.0
git push origin v0.2.0
```

После этого CI автоматически:

1. Запустит `test` (8 матриц)
2. При успехе — `publish` job:
   - `python -m build` → `dist/chernoffpy-0.2.0.tar.gz` + `.whl`
   - `pypa/gh-action-pypi-publish` → загрузка на PyPI через OIDC

### Шаг 6. Проверить результат

```bash
# Подождать 2-3 минуты после успешного CI
pip install chernoffpy==0.2.0 --upgrade

python -c "import chernoffpy; print(chernoffpy.__version__)"
# → 0.2.0
```

Проверить страницу: `https://pypi.org/project/chernoffpy/`

---

## Что делать при ошибке публикации

### Publish job упал: `403 Forbidden`

Trusted Publisher не настроен или настроен с ошибкой в имени workflow/environment.

```
Проверить: PyPI → Account → Publishing → список publisher-ов
Убедиться что Environment name = "pypi" (точно совпадает с ci.yml)
```

### Publish job упал: `File already exists`

PyPI не разрешает перезаписывать уже опубликованные версии.

```bash
# Нельзя переиспользовать тег. Исправить код, сделать новый тег:
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0
# ... исправить ...
git tag v0.2.1
git push origin v0.2.1
```

### Тег создан ошибочно (до коммита на main)

```bash
# Удалить тег локально и удалённо
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0
```

---

## Чек-лист перед каждым релизом

```
[ ] pytest tests/ -q — 0 failures
[ ] Версия обновлена в __init__.py
[ ] Версия обновлена в pyproject.toml
[ ] CHANGELOG.md содержит секцию для новой версии
[ ] Breaking changes описаны явно
[ ] CI на main зелёный
[ ] Trusted Publisher настроен на PyPI (первый раз)
[ ] GitHub Pages настроен (первый раз)
```

---

## Ссылки

- [PyPI Trusted Publishers документация](https://docs.pypi.org/trusted-publishers/)
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- CI pipeline: `.github/workflows/ci.yml`
