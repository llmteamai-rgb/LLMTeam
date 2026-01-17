# 🔓 LLMTeam Open Core Implementation

## Обзор

Этот пакет содержит изменения для реализации Open Core модели лицензирования в LLMTeam.

## Структура файлов

```
open-core-changes/
├── README.md                    # Этот файл
├── PATCHES.md                   # Список всех изменений в существующих файлах
├── __init__.py                  # Обновлённый корневой __init__.py
└── licensing/
    ├── __init__.py              # Экспорты модуля licensing
    ├── models.py                # LicenseTier enum
    ├── manager.py               # LicenseManager, activate(), get_tier()
    └── decorators.py            # @professional_only, @enterprise_only
```

## Как применить изменения

### Шаг 1: Замените модуль licensing

```bash
# В директории проекта
rm -rf src/llmteam/licensing/
cp -r open-core-changes/licensing/ src/llmteam/licensing/
```

### Шаг 2: Замените корневой __init__.py

```bash
cp open-core-changes/__init__.py src/llmteam/__init__.py
```

### Шаг 3: Примените патчи из PATCHES.md

Откройте `PATCHES.md` и для каждого файла добавьте импорт и декоратор.

**Пример для tenancy/manager.py:**

```python
# Добавить в начало файла:
from llmteam.licensing import enterprise_only

# Добавить декоратор перед классом:
@enterprise_only
class TenantManager:
    ...
```

### Шаг 4: Проверьте работу

```python
import llmteam

# Без лицензии
print(llmteam.get_tier())  # LicenseTier.COMMUNITY

# Попытка использовать Enterprise feature
try:
    from llmteam import TenantManager
    tm = TenantManager()  # Ошибка!
except llmteam.FeatureNotLicensedError as e:
    print(e)
    # ╔══════════════════════════════════════════════════════════════╗
    # ║  🔒 FEATURE LOCKED: TenantManager                            ║
    # ╠══════════════════════════════════════════════════════════════╣
    # ║  This feature requires LLMTeam Enterprise license.           ║
    # ║                                                              ║
    # ║  Upgrade: https://llmteam.ai/pricing#enterprise              ║
    # ║  Contact: sales@llmteam.ai                                   ║
    # ╚══════════════════════════════════════════════════════════════╝

# С лицензией
llmteam.activate("LLMT-PRO-A1B2C3D4-20261231")
print(llmteam.get_tier())  # LicenseTier.PROFESSIONAL

# Теперь Professional features работают
from llmteam import ProcessMiningEngine
engine = ProcessMiningEngine()  # OK!
```

## Тiers и Features

### 🆓 COMMUNITY (бесплатно)

| Feature | Описание |
|---------|----------|
| Agent | Базовый агент |
| LLMTeam | Команда агентов (до 2 команд, 5 агентов) |
| Group | Группа команд |
| TeamOrchestrator | Оркестратор команды |
| CriticLoop | Паттерн критика |
| MemoryStore | In-memory хранилище |
| RateLimiter | Базовый rate limiter |
| CircuitBreaker | Circuit breaker |
| SecureAgentContext | Безопасный контекст |

### 💼 PROFESSIONAL ($99/месяц)

Всё из Community, плюс:

| Feature | Описание |
|---------|----------|
| ProcessMiningEngine | Анализ процессов, XES экспорт |
| PostgresSnapshotStore | PostgreSQL для снимков |
| HumanInteractionManager | Human-in-the-loop |
| ActionExecutor | Внешние действия (webhooks) |
| RateLimitedExecutor | Продвинутый rate limiting |
| До 10 команд | Увеличенные лимиты |
| До 20 агентов/команда | Увеличенные лимиты |

### 🏢 ENTERPRISE (custom pricing)

Всё из Professional, плюс:

| Feature | Описание |
|---------|----------|
| TenantManager | Multi-tenant изоляция |
| AuditTrail | Аудит для compliance |
| PostgresTenantStore | PostgreSQL для tenants |
| PostgresAuditStore | PostgreSQL для аудита |
| SSO Integration | Single Sign-On |
| Priority Support | Приоритетная поддержка |
| Unlimited | Без лимитов |

## Формат лицензионного ключа

```
LLMT-{TIER}-{HASH}-{EXPIRY}

Примеры:
- LLMT-COM-ABCD1234-20261231  (Community до 31.12.2026)
- LLMT-PRO-EFGH5678-20261231  (Professional до 31.12.2026)
- LLMT-ENT-IJKL9012-20271231  (Enterprise до 31.12.2027)
```

## Активация лицензии

### Способ 1: Через код

```python
import llmteam
llmteam.activate("LLMT-PRO-XXXX-20261231")
```

### Способ 2: Через переменную окружения

```bash
export LLMTEAM_LICENSE_KEY=LLMT-PRO-XXXX-20261231
```

### Способ 3: Через файл

```bash
mkdir -p ~/.llmteam
echo "LLMT-PRO-XXXX-20261231" > ~/.llmteam/license.key
```

## Проверка статуса

```python
import llmteam

# Вывести полный статус
llmteam.print_license_status()

# Или программно
info = llmteam.LicenseManager.instance().get_info()
print(info)
```

## Генерация тестовых ключей

Для разработки можно использовать ключи с любым хешем:

```python
# Professional до конца 2026
"LLMT-PRO-TEST1234-20261231"

# Enterprise до конца 2027
"LLMT-ENT-TEST5678-20271231"
```

В production версии нужно добавить серверную валидацию ключей.

## Интеграция с PyPI

После применения изменений:

```bash
# Собрать пакет
python -m build

# Загрузить на PyPI
twine upload dist/*
```

Пользователи смогут установить:

```bash
pip install llmteam
```

И использовать Community features бесплатно, а для Professional/Enterprise — активировать лицензию.
