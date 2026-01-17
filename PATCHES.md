# PATCHES.md
# Изменения для Open Core модели
# 
# Этот файл содержит все изменения, которые нужно внести в существующий код
# для реализации Open Core лицензирования.

## ============================================================================
## ФАЙЛ: src/llmteam/tenancy/manager.py
## ДЕЙСТВИЕ: Добавить декоратор @enterprise_only
## ============================================================================

### Было:
```python
class TenantManager:
    """Multi-tenant isolation manager."""
    
    def __init__(self, store: "TenantStore" = None):
        self.store = store or MemoryTenantStore()
```

### Стало:
```python
from llmteam.licensing import enterprise_only

@enterprise_only
class TenantManager:
    """
    Multi-tenant isolation manager.
    
    🔒 ENTERPRISE FEATURE
    Requires LLMTeam Enterprise license.
    """
    
    def __init__(self, store: "TenantStore" = None):
        self.store = store or MemoryTenantStore()
```


## ============================================================================
## ФАЙЛ: src/llmteam/audit/trail.py
## ДЕЙСТВИЕ: Добавить декоратор @enterprise_only
## ============================================================================

### Было:
```python
class AuditTrail:
    """Audit trail for compliance."""
    
    def __init__(self, store: "AuditStore" = None):
```

### Стало:
```python
from llmteam.licensing import enterprise_only

@enterprise_only
class AuditTrail:
    """
    Audit trail for compliance.
    
    🔒 ENTERPRISE FEATURE
    Requires LLMTeam Enterprise license.
    """
    
    def __init__(self, store: "AuditStore" = None):
```


## ============================================================================
## ФАЙЛ: src/llmteam/roles/process_mining.py
## ДЕЙСТВИЕ: Добавить декоратор @professional_only
## ============================================================================

### Было:
```python
class ProcessMiningEngine:
    """Process mining and analytics engine."""
    
    def __init__(self):
```

### Стало:
```python
from llmteam.licensing import professional_only

@professional_only
class ProcessMiningEngine:
    """
    Process mining and analytics engine.
    
    🔒 PROFESSIONAL FEATURE
    Requires LLMTeam Professional or Enterprise license.
    """
    
    def __init__(self):
```


## ============================================================================
## ФАЙЛ: src/llmteam/persistence/stores/postgres.py
## ДЕЙСТВИЕ: Добавить декоратор @professional_only
## ============================================================================

### Было:
```python
class PostgresSnapshotStore:
    """PostgreSQL-backed snapshot store."""
    
    def __init__(self, connection_string: str):
```

### Стало:
```python
from llmteam.licensing import professional_only

@professional_only
class PostgresSnapshotStore:
    """
    PostgreSQL-backed snapshot store.
    
    🔒 PROFESSIONAL FEATURE
    Requires LLMTeam Professional or Enterprise license.
    """
    
    def __init__(self, connection_string: str):
```


## ============================================================================
## ФАЙЛ: src/llmteam/tenancy/stores/postgres.py
## ДЕЙСТВИЕ: Добавить декоратор @enterprise_only
## ============================================================================

### Было:
```python
class PostgresTenantStore:
    """PostgreSQL-backed tenant store."""
    
    def __init__(self, connection_string: str):
```

### Стало:
```python
from llmteam.licensing import enterprise_only

@enterprise_only
class PostgresTenantStore:
    """
    PostgreSQL-backed tenant store.
    
    🔒 ENTERPRISE FEATURE
    Requires LLMTeam Enterprise license.
    """
    
    def __init__(self, connection_string: str):
```


## ============================================================================
## ФАЙЛ: src/llmteam/audit/stores/postgres.py
## ДЕЙСТВИЕ: Добавить декоратор @enterprise_only
## ============================================================================

### Было:
```python
class PostgresAuditStore:
    """PostgreSQL-backed audit store."""
    
    def __init__(self, connection_string: str):
```

### Стало:
```python
from llmteam.licensing import enterprise_only

@enterprise_only
class PostgresAuditStore:
    """
    PostgreSQL-backed audit store.
    
    🔒 ENTERPRISE FEATURE
    Requires LLMTeam Enterprise license.
    """
    
    def __init__(self, connection_string: str):
```


## ============================================================================
## ФАЙЛ: src/llmteam/human/manager.py
## ДЕЙСТВИЕ: Добавить декоратор @professional_only
## ============================================================================

### Было:
```python
class HumanInteractionManager:
    """Manager for human-in-the-loop interactions."""
    
    def __init__(self, store: "InteractionStore" = None):
```

### Стало:
```python
from llmteam.licensing import professional_only

@professional_only
class HumanInteractionManager:
    """
    Manager for human-in-the-loop interactions.
    
    🔒 PROFESSIONAL FEATURE
    Requires LLMTeam Professional or Enterprise license.
    """
    
    def __init__(self, store: "InteractionStore" = None):
```


## ============================================================================
## ФАЙЛ: src/llmteam/actions/executor.py
## ДЕЙСТВИЕ: Добавить декоратор @professional_only
## ============================================================================

### Было:
```python
class ActionExecutor:
    """Executor for external actions."""
    
    def __init__(self, registry: "ActionRegistry"):
```

### Стало:
```python
from llmteam.licensing import professional_only

@professional_only
class ActionExecutor:
    """
    Executor for external actions (webhooks, APIs).
    
    🔒 PROFESSIONAL FEATURE
    Requires LLMTeam Professional or Enterprise license.
    """
    
    def __init__(self, registry: "ActionRegistry"):
```


## ============================================================================
## ФАЙЛ: src/llmteam/ratelimit/executor.py
## ДЕЙСТВИЕ: Добавить декоратор @professional_only для advanced rate limiting
## ============================================================================

### Было:
```python
class RateLimitedExecutor:
    """Executor with rate limiting and circuit breaker."""
    
    def __init__(self, ...):
```

### Стало:
```python
from llmteam.licensing import professional_only

@professional_only
class RateLimitedExecutor:
    """
    Executor with rate limiting and circuit breaker.
    
    🔒 PROFESSIONAL FEATURE
    Requires LLMTeam Professional or Enterprise license.
    
    Note: Basic rate limiting via RateLimiter is available in Community.
    """
    
    def __init__(self, ...):
```


## ============================================================================
## СВОДНАЯ ТАБЛИЦА ЗАЩИЩЁННЫХ КЛАССОВ
## ============================================================================

| Класс | Tier | Файл |
|-------|------|------|
| TenantManager | ENTERPRISE | tenancy/manager.py |
| TenantContext | ENTERPRISE | tenancy/context.py |
| PostgresTenantStore | ENTERPRISE | tenancy/stores/postgres.py |
| AuditTrail | ENTERPRISE | audit/trail.py |
| PostgresAuditStore | ENTERPRISE | audit/stores/postgres.py |
| ProcessMiningEngine | PROFESSIONAL | roles/process_mining.py |
| PostgresSnapshotStore | PROFESSIONAL | persistence/stores/postgres.py |
| HumanInteractionManager | PROFESSIONAL | human/manager.py |
| ActionExecutor | PROFESSIONAL | actions/executor.py |
| RateLimitedExecutor | PROFESSIONAL | ratelimit/executor.py |


## ============================================================================
## КЛАССЫ БЕЗ ОГРАНИЧЕНИЙ (COMMUNITY)
## ============================================================================

| Класс | Файл | Описание |
|-------|------|----------|
| Agent | core/agent.py | Базовый агент |
| LLMTeam | core/team.py | Команда агентов |
| Group | core/group.py | Группа команд |
| TeamOrchestrator | orchestration/team_orch.py | Оркестратор команды |
| GroupOrchestrator | orchestration/group_orch.py | Оркестратор группы |
| CriticLoop | patterns/critic_loop.py | Паттерн критика |
| MemoryStore | persistence/stores/memory.py | In-memory хранилище |
| MemoryTenantStore | tenancy/stores/memory.py | In-memory для dev |
| MemoryAuditStore | audit/stores/memory.py | In-memory для dev |
| RateLimiter | ratelimit/limiter.py | Базовый rate limiter |
| CircuitBreaker | ratelimit/circuit.py | Circuit breaker |
| SecureAgentContext | context/security.py | Безопасный контекст |
