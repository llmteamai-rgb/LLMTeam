# CLAUDE.md

Этот файл содержит инструкции для Claude Code (claude.ai/code) при работе с кодом в этом репозитории.

## Обзор проекта

**llmteam** — Enterprise AI Workflow Runtime для построения multi-agent LLM пайплайнов с безопасностью, оркестрацией и workflow-возможностями.

Переименован из `llm-pipeline-smtrk` в v1.7.0. Python-пакет находится в поддиректории `llmteam/`.

**Текущая версия:** 1.9.0 (Workflow Runtime)

## Команды разработки

Все команды выполняются из директории `llmteam/`.

### Установка

```bash
cd llmteam
pip install -e ".[dev]"

# Проверка установки
$env:PYTHONPATH="src"; python -c "import llmteam; print(f'v{llmteam.__version__}')"
```

### Тестирование

**ВАЖНО:** Тесты требуют последовательного или ограниченно-параллельного выполнения для предотвращения исчерпания памяти.

```bash
# Рекомендуется: использовать test runner
python run_tests.py                    # Последовательно (безопаснее всего)
python run_tests.py --parallel 2       # Ограниченный параллелизм
python run_tests.py --module tenancy   # Один модуль
python run_tests.py --fast             # Только unit-тесты
python run_tests.py --coverage         # С coverage

# Вручную (PowerShell)
$env:PYTHONPATH="src"; pytest tests/tenancy/ -v

# Запуск конкретного теста
$env:PYTHONPATH="src"; pytest tests/tenancy/test_tenancy.py::TestTenantConfig::test_default_config -vv
```

**Избегать:** `pytest tests/ -n auto` — вызывает проблемы с памятью.

### Качество кода

```bash
mypy src/llmteam/          # Проверка типов
black src/ tests/          # Форматирование
ruff check src/ tests/     # Линтинг
```

## Архитектура

### Структура модулей (по версиям)

**v1.7.0 — Security Foundation:**
- `tenancy/` — Мультитенантная изоляция (TenantManager, TenantContext, TenantIsolatedStore)
- `audit/` — Аудит-трейл для compliance с SHA-256 цепочкой чексумм (AuditTrail, AuditRecord)
- `context/` — Безопасный контекст агентов с sealed data (SecureAgentContext, SealedData)
- `ratelimit/` — Rate limiting + circuit breaker (RateLimiter, CircuitBreaker, RateLimitedExecutor)

**v1.8.0 — Orchestration Intelligence:**
- `context/hierarchical.py` — Иерархическое распространение контекста (HierarchicalContext, ContextManager)
- `licensing/` — Лицензионные лимиты (LicenseManager, LicenseTier)
- `execution/` — Параллельное выполнение пайплайнов (PipelineExecutor, ExecutorConfig)
- `roles/` — Роли оркестрации (PipelineOrchestrator, GroupOrchestrator, ProcessMiningEngine)

**v1.9.0 — Workflow Runtime:**
- `actions/` — Внешние API/webhook вызовы (ActionExecutor, ActionRegistry)
- `human/` — Human-in-the-loop взаимодействие (HumanInteractionManager, approval/chat/escalation)
- `persistence/` — Snapshot-based пауза/возобновление (SnapshotManager, PipelineSnapshot)

### Ключевые паттерны

**Store Pattern:** Все хранилища используют инъекцию зависимостей:
- Абстрактный базовый класс определяет интерфейс
- `MemoryStore` для тестирования
- `PostgresStore` для продакшена
- Stores находятся в поддиректориях `stores/`

**Context Manager Pattern:** Операции в рамках тенанта:
```python
async with manager.context(tenant_id):
    # Все операции изолированы в рамках tenant_id
    pass
```

**Immutability для безопасности:**
- `AuditRecord` — неизменяемый с цепочкой чексумм
- `SealedData` — контейнер с доступом только для владельца

### Принципы безопасности

1. **Горизонтальная изоляция** — Агенты НИКОГДА не видят контексты друг друга
2. **Вертикальная видимость** — Оркестраторы видят только своих дочерних агентов (только parent→child)
3. **Sealed Data** — Только агент-владелец может получить доступ к sealed-полям
4. **Изоляция тенантов** — Полное разделение данных между тенантами

## Создание новых модулей

1. Создать директорию модуля с `__init__.py` содержащим экспорты
2. Добавить импорты в родительский `llmteam/__init__.py`
3. Создать тесты в `tests/{module}/test_{module}.py`
4. Добавить модуль в список `TEST_MODULES` в `run_tests.py`

### Async код

- Использовать `asyncio.Lock()` для thread-safety
- Тесты используют `asyncio_mode = "auto"` (декоратор `@pytest.mark.asyncio` не нужен)
- Все async методы должны последовательно использовать `async`/`await`

## Примеры интеграции

```python
from llmteam.tenancy import current_tenant, TenantContext, TenantManager
from llmteam.audit import AuditTrail, AuditEventType
from llmteam.context import SecureAgentContext, ContextAccessPolicy, HierarchicalContext
from llmteam.ratelimit import RateLimitedExecutor
from llmteam.roles import PipelineOrchestrator, GroupOrchestrator
from llmteam.actions import ActionExecutor
from llmteam.human import HumanInteractionManager
from llmteam.persistence import SnapshotManager
```

## Справочная документация

- `v170-security-foundation.md` — Спецификация v1.7.0
- `v180-orchestration-intelligence.md` — Спецификация v1.8.0
- `v190-workflow-runtime.md` — Спецификация v1.9.0
- `llmteam-v170-implementation-summary.md` — Заметки по реализации v1.7.0
- `llmteam-v190-implementation-summary.md` — Заметки по реализации v1.9.0
