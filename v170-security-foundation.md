# 🔐 Пакет изменений v1.7.0: Security Foundation

## ⚠️ Переименование библиотеки

| Было | Стало |
|------|-------|
| `llm_pipeline_smtrk` | `llmteam` |

```python
# Было
from llm_pipeline_smtrk import create_pipeline
pip install llm-pipeline-smtrk

# Стало
from llmteam import create_pipeline
pip install llmteam
```

**Deprecation path:** Старый импорт работает 2 релиза с warning.

---

## 🎯 Цель релиза

Заложить security-фундамент для enterprise использования:
- Multi-tenant изоляция
- Audit trail для compliance
- Защита контекста агентов
- Rate limiting для внешних вызовов

---

## 📋 Состав пакета

| # | RFC | Файл | Effort |
|---|-----|------|--------|
| 1 | Tenant Isolation | `rfc-tenant-isolation.md` | 1 нед |
| 2 | Audit Trail | `rfc-audit-trail.md` | 1.5 нед |
| 3 | Context Security | `rfc-context-security.md` | 1.5 нед |
| 4 | Rate Limiting | `rfc-rate-limiting.md` | 1 нед |

**Общий effort: 5 недель**

---

## 🔗 Зависимости

```
┌─────────────────┐
│ Tenant Isolation│  ← Базовый, первым
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌─────────────┐
│ Audit │ │ Context     │
│ Trail │ │ Security    │
└───────┘ └─────────────┘
    
┌─────────────────┐
│ Rate Limiting   │  ← Независимый
└─────────────────┘
```

---

## 📑 RFC #1: Tenant Isolation

### Назначение

Полная изоляция данных между tenants в multi-tenant deployment.

### Модель данных

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set, Dict, Any
from contextvars import ContextVar
from datetime import datetime


# Context variable для текущего tenant
current_tenant: ContextVar[str] = ContextVar("current_tenant", default="")


class TenantTier(Enum):
    """Уровень подписки tenant."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantLimits:
    """Лимиты по tier."""
    max_concurrent_pipelines: int
    max_agents_per_pipeline: int
    max_requests_per_minute: int
    max_storage_gb: float
    max_runs_per_day: int
    features: Set[str]


TIER_LIMITS: Dict[TenantTier, TenantLimits] = {
    TenantTier.FREE: TenantLimits(
        max_concurrent_pipelines=1,
        max_agents_per_pipeline=5,
        max_requests_per_minute=10,
        max_storage_gb=1.0,
        max_runs_per_day=100,
        features={"basic_agents", "simple_pipelines"},
    ),
    TenantTier.STARTER: TenantLimits(
        max_concurrent_pipelines=2,
        max_agents_per_pipeline=10,
        max_requests_per_minute=60,
        max_storage_gb=10.0,
        max_runs_per_day=1000,
        features={"basic_agents", "simple_pipelines", "parallel_execution"},
    ),
    TenantTier.PROFESSIONAL: TenantLimits(
        max_concurrent_pipelines=10,
        max_agents_per_pipeline=50,
        max_requests_per_minute=300,
        max_storage_gb=100.0,
        max_runs_per_day=10000,
        features={"basic_agents", "simple_pipelines", "parallel_execution",
                  "external_actions", "human_interaction", "persistence"},
    ),
    TenantTier.ENTERPRISE: TenantLimits(
        max_concurrent_pipelines=999999,
        max_agents_per_pipeline=999999,
        max_requests_per_minute=999999,
        max_storage_gb=999999.0,
        max_runs_per_day=999999,
        features={"*"},  # All features
    ),
}


@dataclass
class TenantConfig:
    """Конфигурация tenant."""
    
    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE
    
    # Override лимитов
    max_concurrent_pipelines: Optional[int] = None
    max_agents_per_pipeline: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    
    # Features
    features_enabled: Set[str] = field(default_factory=set)
    features_disabled: Set[str] = field(default_factory=set)
    
    # Security
    allowed_actions: Set[str] = field(default_factory=set)
    blocked_actions: Set[str] = field(default_factory=set)
    
    # Data residency
    data_region: str = "default"
    encryption_key_id: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class TenantContext:
    """Context manager для работы в рамках tenant."""
    
    def __init__(self, tenant_id: str, config: TenantConfig = None):
        self.tenant_id = tenant_id
        self.config = config
        self._token = None
    
    def __enter__(self) -> "TenantContext":
        self._token = current_tenant.set(self.tenant_id)
        return self
    
    def __exit__(self, *args) -> None:
        if self._token:
            current_tenant.reset(self._token)
    
    async def __aenter__(self) -> "TenantContext":
        return self.__enter__()
    
    async def __aexit__(self, *args) -> None:
        self.__exit__(*args)


class TenantManager:
    """Управление tenants."""
    
    def __init__(self, store: "TenantStore"):
        self.store = store
        self._cache: Dict[str, TenantConfig] = {}
    
    async def get_tenant(self, tenant_id: str) -> TenantConfig:
        """Получить конфигурацию tenant."""
        if tenant_id not in self._cache:
            config = await self.store.get(tenant_id)
            if not config:
                raise TenantNotFoundError(f"Tenant {tenant_id} not found")
            self._cache[tenant_id] = config
        return self._cache[tenant_id]
    
    async def create_tenant(self, config: TenantConfig) -> TenantConfig:
        """Создать tenant."""
        await self.store.create(config)
        self._cache[config.tenant_id] = config
        return config
    
    def get_effective_limits(self, config: TenantConfig) -> TenantLimits:
        """Получить эффективные лимиты (tier + overrides)."""
        base = TIER_LIMITS[config.tier]
        return TenantLimits(
            max_concurrent_pipelines=config.max_concurrent_pipelines or base.max_concurrent_pipelines,
            max_agents_per_pipeline=config.max_agents_per_pipeline or base.max_agents_per_pipeline,
            max_requests_per_minute=config.max_requests_per_minute or base.max_requests_per_minute,
            max_storage_gb=base.max_storage_gb,
            max_runs_per_day=base.max_runs_per_day,
            features=self._resolve_features(config, base),
        )
    
    def _resolve_features(self, config: TenantConfig, base: TenantLimits) -> Set[str]:
        features = base.features.copy()
        features |= config.features_enabled
        features -= config.features_disabled
        return features
    
    async def check_limit(self, tenant_id: str, limit_type: str, current: int) -> bool:
        """Проверить лимит."""
        config = await self.get_tenant(tenant_id)
        limits = self.get_effective_limits(config)
        max_val = getattr(limits, f"max_{limit_type}", float('inf'))
        return current < max_val
    
    async def check_feature(self, tenant_id: str, feature: str) -> bool:
        """Проверить доступность feature."""
        config = await self.get_tenant(tenant_id)
        limits = self.get_effective_limits(config)
        return "*" in limits.features or feature in limits.features
    
    def context(self, tenant_id: str) -> TenantContext:
        """Создать контекст tenant."""
        config = self._cache.get(tenant_id)
        return TenantContext(tenant_id, config)


class TenantIsolatedStore:
    """Storage с автоматической изоляцией по tenant."""
    
    def __init__(self, inner_store: Any):
        self.inner = inner_store
    
    def _get_tenant_id(self) -> str:
        tenant_id = current_tenant.get()
        if not tenant_id:
            raise TenantContextError("No tenant context")
        return tenant_id
    
    async def get(self, key: str) -> Any:
        tenant_id = self._get_tenant_id()
        return await self.inner.get(f"{tenant_id}:{key}")
    
    async def set(self, key: str, value: Any) -> None:
        tenant_id = self._get_tenant_id()
        await self.inner.set(f"{tenant_id}:{key}", value)
    
    async def delete(self, key: str) -> None:
        tenant_id = self._get_tenant_id()
        await self.inner.delete(f"{tenant_id}:{key}")
    
    async def list(self, prefix: str = "") -> list[str]:
        tenant_id = self._get_tenant_id()
        keys = await self.inner.list(f"{tenant_id}:{prefix}")
        return [k[len(tenant_id) + 1:] for k in keys]
```

### Использование

```python
from llmteam.tenancy import TenantManager, TenantConfig, TenantTier

# Setup
tenant_manager = TenantManager(PostgresTenantStore(conn))

# Создание tenant
await tenant_manager.create_tenant(TenantConfig(
    tenant_id="acme_corp",
    name="Acme Corporation",
    tier=TenantTier.PROFESSIONAL,
    data_region="eu-west-1",
))

# Использование
async with tenant_manager.context("acme_corp"):
    result = await pipeline.run_async(input_data)
    # Все данные изолированы в рамках acme_corp
```

---

## 📑 RFC #2: Audit Trail

### Назначение

Immutable audit log для compliance (SOC2, ISO27001, HIPAA).

### Модель данных

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import hashlib
import json


class AuditEventType(Enum):
    """Типы audit событий."""
    
    # Pipeline
    PIPELINE_CREATED = "pipeline.created"
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    
    # Agent
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    
    # Security
    ACCESS_GRANTED = "security.access_granted"
    ACCESS_DENIED = "security.access_denied"
    CONTEXT_ACCESSED = "security.context_accessed"
    SEALED_DATA_ACCESSED = "security.sealed_data_accessed"
    
    # Config
    CONFIG_CHANGED = "config.changed"
    
    # Data
    DATA_EXPORTED = "data.exported"
    DATA_DELETED = "data.deleted"


class AuditSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditRecord:
    """Immutable audit запись."""
    
    # Identity
    record_id: str
    sequence_number: int
    timestamp: datetime
    
    # Event
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Context
    tenant_id: str = ""
    pipeline_id: str = ""
    run_id: str = ""
    agent_name: str = ""
    
    # Actor
    actor_type: str = ""      # user, agent, system
    actor_id: str = ""
    actor_ip: str = ""
    
    # Details
    action: str = ""
    resource_type: str = ""
    resource_id: str = ""
    old_value: Optional[dict] = None
    new_value: Optional[dict] = None
    
    # Result
    success: bool = True
    error_message: str = ""
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    
    # Integrity (chain)
    checksum: str = ""
    previous_checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        data = {
            "record_id": self.record_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "previous_checksum": self.previous_checksum,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        return self.checksum == self._compute_checksum()


@dataclass
class AuditQuery:
    """Фильтры для поиска."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tenant_id: Optional[str] = None
    event_types: Optional[list[AuditEventType]] = None
    actor_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    limit: int = 100
    offset: int = 0


class AuditTrail:
    """Главный класс для audit."""
    
    def __init__(self, store: "AuditStore", tenant_id: str = "default"):
        self.store = store
        self.tenant_id = tenant_id
        self._sequence = 0
        self._last_checksum = ""
    
    async def log(
        self,
        event_type: AuditEventType,
        *,
        actor_id: str = "",
        action: str = "",
        resource_type: str = "",
        resource_id: str = "",
        old_value: dict = None,
        new_value: dict = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        **kwargs,
    ) -> AuditRecord:
        """Записать событие."""
        self._sequence += 1
        
        record = AuditRecord(
            record_id=generate_uuid(),
            sequence_number=self._sequence,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            tenant_id=self.tenant_id,
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_value=old_value,
            new_value=new_value,
            previous_checksum=self._last_checksum,
            **kwargs,
        )
        
        self._last_checksum = record.checksum
        await self.store.append(record)
        return record
    
    async def query(self, query: AuditQuery) -> list[AuditRecord]:
        """Поиск записей."""
        return await self.store.query(query)
    
    async def verify_chain(self, start: int, end: int) -> tuple[bool, list[int]]:
        """Проверить целостность цепочки."""
        return await self.store.verify_chain(self.tenant_id, start, end)
    
    # Convenience methods
    async def log_access_denied(self, actor_id: str, resource: str, reason: str):
        return await self.log(
            AuditEventType.ACCESS_DENIED,
            actor_id=actor_id,
            resource_id=resource,
            severity=AuditSeverity.WARNING,
            success=False,
            error_message=reason,
        )
```

### Использование

```python
from llmteam.audit import AuditTrail, PostgresAuditStore

audit = AuditTrail(PostgresAuditStore(conn), tenant_id="acme_corp")

# Автоматически в pipeline
await audit.log(
    AuditEventType.PIPELINE_STARTED,
    actor_id="user@acme.com",
    resource_id="pipeline_123",
)

# Проверка целостности
valid, missing = await audit.verify_chain(1, 1000)
```

---

## 📑 RFC #3: Context Security

### Назначение

Безопасная модель видимости контекста без горизонтального доступа.

### Принципы

```
1. Агент видит ТОЛЬКО свой контекст
2. Оркестратор видит контексты СВОИХ агентов (по умолчанию)
3. Sealed данные — ТОЛЬКО владелец
4. Горизонтальная видимость — ЗАПРЕЩЕНА
```

### Модель данных

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Set


class VisibilityLevel(Enum):
    """Уровень видимости."""
    SELF_ONLY = "self_only"
    ORCHESTRATOR = "orchestrator"
    HIERARCHY = "hierarchy"
    # НЕТ "peers" — запрещено


class SensitivityLevel(Enum):
    """Чувствительность данных."""
    PUBLIC = "public"           # Виден всей иерархии
    INTERNAL = "internal"       # Виден оркестраторам
    CONFIDENTIAL = "confidential"  # Только прямой оркестратор
    SECRET = "secret"           # Только агент (sealed)
    TOP_SECRET = "top_secret"   # Sealed + encryption + audit


@dataclass
class ContextAccessPolicy:
    """Политика доступа к контексту."""
    
    default_visibility: VisibilityLevel = VisibilityLevel.ORCHESTRATOR
    denied_viewers: Set[str] = field(default_factory=set)
    allowed_viewers: Set[str] = field(default_factory=set)
    sealed_fields: Set[str] = field(default_factory=set)
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    audit_access: bool = False
    
    def can_access(
        self, 
        viewer_id: str, 
        viewer_role: str,
        field_name: str = None,
    ) -> tuple[bool, str]:
        """Проверить доступ."""
        
        # Agents никогда не видят друг друга
        if viewer_role == "agent":
            return False, "Horizontal access forbidden"
        
        # Sealed fields
        if field_name and field_name in self.sealed_fields:
            return False, f"Field '{field_name}' is sealed"
        
        # Explicit deny
        if viewer_id in self.denied_viewers and viewer_id not in self.allowed_viewers:
            return False, f"Viewer '{viewer_id}' denied"
        
        # SECRET/TOP_SECRET
        if self.sensitivity in (SensitivityLevel.SECRET, SensitivityLevel.TOP_SECRET):
            return False, "Context is sealed"
        
        # CONFIDENTIAL — только direct orchestrator
        if self.sensitivity == SensitivityLevel.CONFIDENTIAL:
            if viewer_role != "pipeline_orch":
                return False, "CONFIDENTIAL: direct orchestrator only"
        
        return True, "Granted"


@dataclass
class SealedData:
    """Контейнер для sealed данных."""
    
    _data: Any = field(repr=False)
    owner_id: str = ""
    
    def get(self, requester_id: str) -> Any:
        if requester_id != self.owner_id:
            raise PermissionError(f"Access denied to sealed data")
        return self._data
    
    def __repr__(self) -> str:
        return f"SealedData(owner={self.owner_id}, [REDACTED])"


@dataclass
class SecureAgentContext:
    """Контекст агента с security."""
    
    agent_id: str
    agent_name: str
    
    # Public
    confidence: float = 0.0
    status: str = "idle"
    error_count: int = 0
    
    # Internal
    reasoning_steps: list[str] = field(default_factory=list)
    
    # Sealed
    _sealed: dict[str, SealedData] = field(default_factory=dict, repr=False)
    
    # Policy
    access_policy: ContextAccessPolicy = field(default_factory=ContextAccessPolicy)
    
    def set_sealed(self, key: str, value: Any) -> None:
        """Сохранить sealed данные."""
        self._sealed[key] = SealedData(_data=value, owner_id=self.agent_id)
        self.access_policy.sealed_fields.add(key)
    
    def get_sealed(self, key: str, requester_id: str) -> Any:
        """Получить sealed данные (только owner)."""
        return self._sealed[key].get(requester_id)
    
    def get_visible_context(self, viewer_id: str, viewer_role: str) -> dict:
        """Получить отфильтрованный контекст."""
        allowed, reason = self.access_policy.can_access(viewer_id, viewer_role)
        
        if not allowed:
            return {
                "agent_id": self.agent_id,
                "access": "denied",
                "reason": reason,
            }
        
        result = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "confidence": self.confidence,
            "status": self.status,
            "error_count": self.error_count,
        }
        
        if self.access_policy.sensitivity in (SensitivityLevel.PUBLIC, SensitivityLevel.INTERNAL):
            result["reasoning_steps"] = self.reasoning_steps
        
        result["sealed_fields"] = list(self.access_policy.sealed_fields)
        
        return result
```

### Использование

```python
from llmteam.context import SecureAgentContext, ContextAccessPolicy, SensitivityLevel

# Создание агента с секретными данными
context = SecureAgentContext(
    agent_id="payment_001",
    agent_name="payment_processor",
    access_policy=ContextAccessPolicy(
        sensitivity=SensitivityLevel.CONFIDENTIAL,
        sealed_fields={"card_number", "cvv"},
        audit_access=True,
    ),
)

# В агенте
context.set_sealed("card_number", "4111-1111-1111-1111")

# Оркестратор получает фильтрованный контекст
visible = context.get_visible_context(
    viewer_id="pipeline_orch_1",
    viewer_role="pipeline_orch",
)
# sealed_fields видны как список, но значения недоступны
```

---

## 📑 RFC #4: Rate Limiting

### Назначение

Защита внешних API от перегрузки с Circuit Breaker.

### Модель данных

```python
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Callable, Any
import asyncio


class RateLimitStrategy(Enum):
    WAIT = "wait"
    REJECT = "reject"
    QUEUE = "queue"
    FALLBACK = "fallback"


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RateLimitConfig:
    """Конфигурация rate limiting."""
    
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 10
    
    strategy: RateLimitStrategy = RateLimitStrategy.WAIT
    max_wait_seconds: float = 30.0
    queue_size: int = 100
    
    # Retry
    retry_count: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential: bool = True


@dataclass
class CircuitBreakerConfig:
    """Конфигурация circuit breaker."""
    
    failure_threshold: int = 5
    failure_rate_threshold: float = 0.5
    sample_size: int = 10
    open_timeout: timedelta = timedelta(seconds=30)
    half_open_max_requests: int = 3


class RateLimiter:
    """Rate limiter с circuit breaker."""
    
    def __init__(
        self,
        name: str,
        config: RateLimitConfig,
        circuit_config: CircuitBreakerConfig = None,
    ):
        self.name = name
        self.config = config
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._semaphore = asyncio.Semaphore(config.burst_size)
    
    async def acquire(self) -> bool:
        """Получить разрешение."""
        if self._circuit_state == CircuitState.OPEN:
            if self.config.strategy == RateLimitStrategy.FALLBACK:
                return False
            raise CircuitOpenError(f"Circuit open for {self.name}")
        
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.max_wait_seconds,
            )
            return True
        except asyncio.TimeoutError:
            if self.config.strategy == RateLimitStrategy.REJECT:
                raise RateLimitExceeded(self.name)
            return False
    
    def release(self) -> None:
        self._semaphore.release()
    
    def record_success(self) -> None:
        self._failure_count = 0
        if self._circuit_state == CircuitState.HALF_OPEN:
            self._circuit_state = CircuitState.CLOSED
    
    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.circuit_config.failure_threshold:
            self._circuit_state = CircuitState.OPEN


class RateLimitedExecutor:
    """Executor с rate limiting."""
    
    def __init__(self):
        self._limiters: dict[str, RateLimiter] = {}
    
    def register(self, name: str, config: RateLimitConfig, circuit: CircuitBreakerConfig = None):
        self._limiters[name] = RateLimiter(name, config, circuit)
    
    async def execute(self, name: str, handler: Callable, *args, **kwargs) -> Any:
        limiter = self._limiters.get(name)
        if not limiter:
            return await handler(*args, **kwargs)
        
        config = limiter.config
        
        for attempt in range(config.retry_count + 1):
            try:
                if not await limiter.acquire():
                    return config.fallback_value if hasattr(config, 'fallback_value') else None
                
                try:
                    result = await handler(*args, **kwargs)
                    limiter.record_success()
                    return result
                finally:
                    limiter.release()
                    
            except Exception as e:
                limiter.record_failure()
                if attempt < config.retry_count:
                    delay = config.retry_base_delay * (2 ** attempt if config.retry_exponential else 1)
                    await asyncio.sleep(min(delay, config.retry_max_delay))
                    continue
                raise
```

### Использование

```python
from llmteam.ratelimit import RateLimitConfig, CircuitBreakerConfig, RateLimitedExecutor

executor = RateLimitedExecutor()

executor.register(
    "external_api",
    RateLimitConfig(
        requests_per_minute=100,
        strategy=RateLimitStrategy.QUEUE,
    ),
    CircuitBreakerConfig(
        failure_threshold=5,
        open_timeout=timedelta(seconds=60),
    ),
)

result = await executor.execute("external_api", call_api, params)
```

---

## 📅 План реализации

| Неделя | Задачи |
|--------|--------|
| 1 | Tenant Isolation: модели, TenantManager, TenantContext |
| 2 | Audit Trail: AuditRecord, AuditTrail, PostgresStore |
| 3 | Audit Trail: chain verification, query, интеграция |
| 4 | Context Security: модели, SecureAgentContext, политики |
| 5 | Rate Limiting: RateLimiter, CircuitBreaker, Executor |
| +0.5 | Тестирование, документация, миграция имени пакета |

**Итого: ~5.5 недель**

---

## 📁 Структура файлов

```
src/llmteam/
├── __init__.py                 # Новое имя пакета
├── _compat.py                  # Backward compatibility
│
├── tenancy/
│   ├── __init__.py
│   ├── models.py               # TenantConfig, TenantLimits
│   ├── manager.py              # TenantManager
│   ├── context.py              # TenantContext, current_tenant
│   ├── isolation.py            # TenantIsolatedStore
│   └── stores/
│       ├── postgres.py
│       └── memory.py
│
├── audit/
│   ├── __init__.py
│   ├── models.py               # AuditRecord, AuditQuery
│   ├── trail.py                # AuditTrail
│   └── stores/
│       ├── postgres.py
│       └── memory.py
│
├── context/
│   ├── __init__.py
│   ├── security.py             # ContextAccessPolicy, SealedData
│   ├── visibility.py           # VisibilityLevel, SensitivityLevel
│   └── secure_context.py       # SecureAgentContext
│
└── ratelimit/
    ├── __init__.py
    ├── config.py               # RateLimitConfig, CircuitBreakerConfig
    ├── limiter.py              # RateLimiter
    ├── circuit.py              # CircuitBreaker
    └── executor.py             # RateLimitedExecutor
```

---

## ✅ Критерии готовности

- [ ] Все тесты проходят
- [ ] Миграция имени пакета с deprecation warning
- [ ] Документация обновлена
- [ ] Примеры использования
- [ ] Security review пройден
- [ ] Performance benchmarks

---

## 🎯 Результат v1.7.0

```python
from llmteam import create_pipeline
from llmteam.tenancy import TenantManager, TenantConfig, TenantTier
from llmteam.audit import AuditTrail, PostgresAuditStore
from llmteam.context import SensitivityLevel
from llmteam.ratelimit import RateLimitConfig

# Multi-tenant setup
tenant_manager = TenantManager(store)
await tenant_manager.create_tenant(TenantConfig(
    tenant_id="acme",
    tier=TenantTier.PROFESSIONAL,
))

# Audit for compliance
audit = AuditTrail(PostgresAuditStore(conn))

# Pipeline с security
pipeline = (
    create_pipeline("secure_flow", llm=llm)
    .with_tenant_manager(tenant_manager)
    .with_audit(audit)
    .add_agent(
        "processor",
        sensitivity=SensitivityLevel.CONFIDENTIAL,
        sealed_fields=["secret_data"],
    )
)

# Run
async with tenant_manager.context("acme"):
    result = await pipeline.run_async(data)
```

---

**Версия: 1.7.0**
**Кодовое имя: Security Foundation**
