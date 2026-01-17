# 🔴 P0 — Критические замечания (блокеры релиза)

**Версия:** 2.0.0  
**Дата:** 17 января 2025  
**Статус:** ⛔ Блокируют релиз

---

## P0-1: Битые директории в архиве

**Серьёзность:** 🔴 КРИТИЧЕСКАЯ  
**Влияние:** Сборка/установка может сломаться

### Проблема

В архиве присутствуют директории с невалидными именами:

```
src/llmteam/{tenancy/
src/llmteam/{tenancy/stores,audit/
src/llmteam/{tenancy/stores,audit/stores,context,ratelimit}/
tests/{tenancy,audit,context,ratelimit}/
```

### Причина

Вероятно, артефакт bash brace expansion при создании архива:
```bash
# Так делать НЕЛЬЗЯ:
mkdir -p src/llmteam/{tenancy,audit,context,ratelimit}
```

### Решение

```bash
# Удалить битые директории
find . -name "{*" -type d -exec rm -rf {} +

# Проверить
find . -name "{*" -o -name "*,*" -type d
```

### Критерий готовности

```bash
find . -name "{*" | wc -l  # Должно быть 0
```

---

## P0-2: Мусорные файлы в архиве

**Серьёзность:** 🔴 КРИТИЧЕСКАЯ  
**Влияние:** Размер пакета, утечка внутренних данных

### Проблема

В архиве присутствуют:
- `__pycache__/` директории (везде)
- `*.pyc` файлы
- `dist/llmteam-1.9.0-*` (старая версия!)
- `.pytest_cache/`

### Решение

1. **Добавить `.gitignore`:**
```gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.pytest_cache/
dist/
build/
*.egg-info/
.mypy_cache/
.ruff_cache/
```

2. **Очистить:**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf dist/ build/ *.egg-info/
```

3. **Пересобрать:**
```bash
python -m build
# Результат: dist/llmteam-2.0.0-py3-none-any.whl
```

### Критерий готовности

```bash
find . -name "__pycache__" | wc -l  # 0
ls dist/  # Только 2.0.0 артефакты
```

---

## P0-3: Несоответствие версий

**Серьёзность:** 🔴 КРИТИЧЕСКАЯ  
**Влияние:** Путаница при установке, неправильная версия

### Проблема

| Место | Версия |
|-------|--------|
| `pyproject.toml` | 2.0.0 ✅ |
| `__init__.py` | 2.0.0 ✅ |
| `dist/*.whl` | 1.9.0 ❌ |

### Решение

```bash
# Удалить старые артефакты
rm -rf dist/

# Пересобрать
pip install build
python -m build

# Проверить
ls dist/
# llmteam-2.0.0-py3-none-any.whl
# llmteam-2.0.0.tar.gz
```

### Критерий готовности

```bash
pip install dist/*.whl
python -c "import llmteam; assert llmteam.__version__ == '2.0.0'"
```

---

## P0-4: Отсутствуют обязательные файлы

**Серьёзность:** 🔴 КРИТИЧЕСКАЯ  
**Влияние:** Невозможно использовать в enterprise/on-prem

### Проблема

Отсутствуют файлы, обязательные для enterprise:

| Файл | Статус | Назначение |
|------|--------|------------|
| `LICENSE` | ❌ Отсутствует | Лицензия Apache 2.0 |
| `NOTICE` | ❌ Отсутствует | Атрибуция, copyright |
| `CHANGELOG.md` | ❌ Отсутствует | История изменений |
| `SECURITY.md` | ❌ Отсутствует | Политика безопасности |
| `SBOM.json` | ❌ Отсутствует | Software Bill of Materials |

### Решение

1. **LICENSE** — скачать Apache 2.0:
```bash
curl -o LICENSE https://www.apache.org/licenses/LICENSE-2.0.txt
```

2. **NOTICE:**
```
LLMTeam
Copyright 2024-2025 KirilinVS

Licensed under the Apache License, Version 2.0.
```

3. **CHANGELOG.md:**
```markdown
# Changelog

## [2.0.0] - 2025-01-17

### Added
- Canvas Integration (RFC #1-5)
- RuntimeContext injection
- Worktrail Events
- Step Catalog API
- Segment Runner

### Changed
- License model to Open Core

## [1.9.0] - 2025-01-15
...
```

4. **SECURITY.md:**
```markdown
# Security Policy

## Supported Versions
| Version | Supported |
|---------|-----------|
| 2.0.x   | ✅        |
| < 2.0   | ❌        |

## Reporting a Vulnerability
Email: security@llmteam.ai
Response time: 48 hours
```

5. **SBOM.json:**
```bash
pip install cyclonedx-bom
cyclonedx-py --format json -o SBOM.json
```

### Критерий готовности

```bash
ls LICENSE NOTICE CHANGELOG.md SECURITY.md SBOM.json
# Все 5 файлов присутствуют
```

---

## P0-5: Неправильная лицензия в pyproject.toml

**Серьёзность:** 🔴 КРИТИЧЕСКАЯ  
**Влияние:** Юридические проблемы

### Проблема

```toml
# Сейчас:
license = "MIT"
authors = [{ name = "llmteam contributors" }]

# Должно быть:
license = "Apache-2.0"
authors = [{ name = "KirilinVS", email = "LLMTeamai@gmail.com" }]
```

### Решение

Обновить `pyproject.toml`:

```toml
[project]
name = "llmteam"
version = "2.0.0"
description = "Enterprise AI Workflow Runtime - Multi-agent LLM pipelines"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    { name = "KirilinVS", email = "LLMTeamai@gmail.com" }
]
maintainers = [
    { name = "KirilinVS", email = "LLMTeamai@gmail.com" }
]

[project.urls]
Homepage = "https://llmteam.ai"
Documentation = "https://docs.llmteam.ai"
Repository = "https://github.com/llmteamai-rgb/LLMTeam"
Issues = "https://github.com/llmteamai-rgb/LLMTeam/issues"
Changelog = "https://github.com/llmteamai-rgb/LLMTeam/blob/main/CHANGELOG.md"
```

---

## P0-6: Опасная семантика условий в EdgeDefinition

**Серьёзность:** 🔴 КРИТИЧЕСКАЯ (БЕЗОПАСНОСТЬ)  
**Влияние:** Невалидное условие = выполнение всех переходов

### Проблема

Файл: `src/llmteam/canvas/runner.py`, строка 456:

```python
def _evaluate_condition(self, condition: str, output: Any) -> bool:
    if condition.lower() == "true":
        return True
    if condition.lower() == "false":
        return False
    
    if isinstance(output, dict):
        if condition in output:
            return bool(output[condition])
    
    return True  # ← ОПАСНО! Неизвестное условие = True
```

**Риск:** Любое невалидное условие (опечатка, инъекция) будет интерпретировано как `True`.

### Решение

1. **Добавить исключение** в `canvas/exceptions.py`:
```python
class InvalidConditionError(CanvasError):
    """Raised when condition expression is invalid or unsafe."""
    
    def __init__(self, condition: str, reason: str) -> None:
        self.condition = condition
        self.reason = reason
        super().__init__(f"Invalid condition '{condition}': {reason}")
```

2. **Переписать `_evaluate_condition`:**
```python
import re

CONDITION_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
FORBIDDEN_KEYWORDS = {'import', 'exec', 'eval', '__', 'lambda', 'compile', 'open'}

def _validate_condition(self, condition: str) -> None:
    """Validate condition at segment load time."""
    condition_lower = condition.lower()
    
    # Boolean literals are always OK
    if condition_lower in ('true', 'false'):
        return
    
    # Check for forbidden keywords
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in condition_lower:
            raise InvalidConditionError(condition, f"Forbidden keyword: {keyword}")
    
    # Simple field reference must match pattern
    if not CONDITION_PATTERN.match(condition):
        raise InvalidConditionError(condition, "Invalid characters in condition")

def _evaluate_condition(self, condition: str, output: Any) -> bool:
    """Evaluate condition with strict validation."""
    # Boolean literals
    if condition.lower() == "true":
        return True
    if condition.lower() == "false":
        return False
    
    # Field reference in output dict
    if isinstance(output, dict):
        if condition in output:
            return bool(output[condition])
        # Field not found = ERROR, not True!
        raise InvalidConditionError(
            condition, 
            f"Field '{condition}' not found in output"
        )
    
    # Cannot evaluate against non-dict = ERROR
    raise InvalidConditionError(
        condition,
        f"Cannot evaluate against output type {type(output).__name__}"
    )
```

3. **Валидировать при загрузке сегмента:**
```python
def _load_segment(self, segment: SegmentDefinition) -> None:
    for edge in segment.edges:
        if edge.condition:
            self._validate_condition(edge.condition)
```

### Критерий готовности

```python
# Тест 1: Невалидное условие → ошибка
runner._evaluate_condition("unknown_field", {"other": 1})
# InvalidConditionError: Field 'unknown_field' not found

# Тест 2: Инъекция → ошибка
runner._validate_condition("__import__('os')")
# InvalidConditionError: Forbidden keyword: __
```

---

## 📊 Сводка P0

| ID | Задача | Effort | Статус |
|----|--------|--------|--------|
| P0-1 | Удалить битые директории | 15 мин | ⏳ |
| P0-2 | Очистить __pycache__ и dist | 15 мин | ⏳ |
| P0-3 | Пересобрать wheel 2.0.0 | 10 мин | ⏳ |
| P0-4 | Создать LICENSE/NOTICE/CHANGELOG/SECURITY/SBOM | 2-3 часа | ⏳ |
| P0-5 | Исправить pyproject.toml | 15 мин | ⏳ |
| P0-6 | Исправить _evaluate_condition | 2-4 часа | ⏳ |

**Общий effort P0:** ~4-6 часов

---

## ✅ Definition of Done для P0

- [ ] `find . -name "{*" | wc -l` = 0
- [ ] `find . -name "__pycache__" | wc -l` = 0
- [ ] `ls dist/` содержит только `llmteam-2.0.0-*`
- [ ] `ls LICENSE NOTICE CHANGELOG.md SECURITY.md SBOM.json` — все присутствуют
- [ ] `pyproject.toml` содержит `license = "Apache-2.0"`
- [ ] `runner._evaluate_condition("bad", {})` → `InvalidConditionError`
