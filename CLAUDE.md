# ChernoffPy — Project Configuration

## Project

**ChernoffPy** — первая open-source библиотека для черновских аппроксимаций операторных полугрупп с анализом скоростей сходимости.

- **Теория:** Чернов (1968) + Галкин-Ремизов (Israel J. Math. 2025)
- **Версия:** 0.1.0 (alpha)
- **Python:** ≥3.10 (используем 3.11.13)
- **Зависимости:** numpy, scipy (только 2 runtime deps!)
- **Лицензия:** MIT

## Architecture

```
chernoffpy/
├── functions.py      — ChernoffFunction ABC + 5 implementations (core)
├── semigroups.py     — HeatSemigroup exact solutions (reference)
├── analysis.py       — Convergence rate tools
└── finance/          — European option pricing (Phase 1)
    ├── validation.py — Dataclasses (MarketParams, GridConfig, etc.)
    ├── transforms.py — BS ↔ Heat equation (Wilmott substitution)
    ├── european.py   — EuropeanPricer
    └── greeks.py     — Delta, Gamma, Vega, Theta, Rho
```

**Принцип:** finance/ зависит от core, НЕ наоборот. Core ничего не знает о финансах.

## Commands

```bash
# Тесты
pytest -v --tb=short                    # Все тесты
pytest tests/finance/ -v                # Только finance
pytest --cov=chernoffpy --cov-report=term-missing

# Установка
pip install -e ".[dev]"

# Пример
python examples/verify_galkin_remizov.py
```

## Key APIs

```python
# Core
from chernoffpy import CrankNicolson, HeatSemigroup, convergence_rate

# Finance
from chernoffpy.finance import EuropeanPricer, MarketParams, GridConfig, compute_greeks
```

## Agents

| Agent | File | Назначение |
|-------|------|-----------|
| **critic-auditor** | `.claude/agents/critic-auditor.md` | Скептик. Математическая корректность, численная стабильность, безопасность |
| **stakeholder** | `.claude/agents/stakeholder.md` | Заказчик. UX, коммерческая ценность, документация, приоритизация |
| **math-theorist** | `.claude/agents/math-theorist.md` | Теоретик. Функанализ, полугруппы, Чернов, Галкин-Ремизов, Паде |
| **quant-developer** | `.claude/agents/quant-developer.md` | Квант. BS, Greeks, PDE, finance roadmap (American, local vol) |
| **test-engineer** | `.claude/agents/test-engineer.md` | Тестировщик. Edge cases, convergence, property-based, regression guard |
| **architect** | `.claude/agents/architect.md` | Архитектор. API дизайн, расширяемость, dependency management |

### Использование агентов

```
> Используй subagent critic-auditor для аудита finance/transforms.py
> Используй subagent stakeholder для оценки Phase 2 roadmap
> Используй subagent math-theorist для проверки формулы Gamma
```

## Quality Standards

- Тесты: 117 passing (70 core + 47 finance)
- Новый код: coverage ≥ 80%
- Convergence rates: BE → α≈1, CN → α≈2, Padé[1/2] → α≈3
- Put-call parity: |C - P - (S - Ke^{-rT})| < 0.15
- Greeks vs BS analytical: tolerance 5-10%

## References

- Galkin, Remizov. Israel J. Math. 265:2, 929-943 (2025) — convergence rates
- Dragunova, Nikbakht, Remizov. arXiv:2301.05284 (2023) — numerical code
- Chernoff, P.R. J. Funct. Anal. 2(2), 238-242 (1968) — original theorem

## Local Context Files

- `E:\MarkovChains\INTELLIGENCE_REPORT.md` — 18 публикаций Ремизова, конкурентный ландшафт
- `E:\MarkovChains\ROADMAP_STRUCTURED.md` — 8-частный учебный план
- `E:\MarkovChains\SOLUTIONS.md` — 10 задач с решениями (72KB)
- `E:\MarkovChains\chernoff_semigroups_masterplan.md` — 18-месячный мастер-план
