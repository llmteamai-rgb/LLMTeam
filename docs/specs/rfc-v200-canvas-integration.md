# 🎨 RFC v2.0.0: Canvas Integration (КорпОС Worktrail)

## Обзор

Мажорный релиз для интеграции LLMTeam с визуальным canvas-редактором КорпОС Worktrail.

**Цель:** LLMTeam segments могут создаваться, редактироваться и исполняться через визуальный UI без написания Python кода.

**Зависимость:** v1.9.0 (Workflow Runtime)

---

## 📋 Состав пакета

| # | Task ID | Название | Effort | Приоритет |
|---|---------|----------|--------|-----------|
| 1 | TASK-RT-01 | RuntimeContext Injection | 2 нед | P0 |
| 2 | TASK-EVT-01 | Worktrail Events | 1 нед | P0 |
| 3 | TASK-CAN-01 | Segment JSON Contract | 1 нед | P0 |
| 4 | TASK-CAN-02 | Step Catalog API | 1.5 нед | P0 |
| 5 | TASK-SEG-01 | Segment Runner | 1.5 нед | P0 |
| 6 | TASK-ISO-01 | Instance Namespacing | 0.5 нед | P1 |
| 7 | TASK-HITL-01 | Human Tasks Integration | 1 нед | P1 |
| 8 | TASK-DOC-01 | Integration Documentation | 1 нед | P1 |

**Общий effort:** ~10.5 недель

---

## 🔗 Зависимости

```
v1.9.0 (Workflow Runtime)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    Phase 1: Foundations                  │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │ TASK-RT-01       │    │ TASK-EVT-01      │          │
│  │ RuntimeContext   │    │ Worktrail Events │          │
│  └────────┬─────────┘    └────────┬─────────┘          │
│           │                       │                     │
└───────────┼───────────────────────┼─────────────────────┘
            │                       │
            ▼                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Phase 2: Contracts                    │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │ TASK-CAN-01      │    │ TASK-CAN-02      │          │
│  │ Segment JSON     │    │ Step Catalog     │          │
│  └────────┬─────────┘    └──────────────────┘          │
│           │                                             │
└───────────┼─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│                   Phase 3: Integration                   │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │ TASK-SEG-01      │    │ TASK-ISO-01      │          │
│  │ Segment Runner   │    │ Namespacing      │          │
│  └──────────────────┘    └──────────────────┘          │
│  ┌──────────────────┐                                  │
│  │ TASK-HITL-01     │                                  │
│  │ Human Tasks      │                                  │
│  └──────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│                   Phase 4: Documentation                 │
│  ┌──────────────────┐                                  │
│  │ TASK-DOC-01      │                                  │
│  │ Integration Docs │                                  │
│  └──────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📑 RFC #1: RuntimeContext Injection (TASK-RT-01)

### Назначение

Единая точка доступа к enterprise ресурсам. Шаги получают зависимости через injection, в конфигах — только ссылки/ID.

### Проблема

Текущее состояние:
```python
# Плохо: конфиг содержит живые объекты
pipeline = Pipeline(
    store=PostgresStore(connection_string),  # ❌ Живой объект
    llm=OpenAI(api_key="sk-xxx"),            # ❌ Секрет в коде
)
```

Нужно:
```python
# Хорошо: конфиг содержит только ссылки
{
    "store_ref": "main_store",      # ✅ Резолвится из RuntimeContext
    "llm_ref": "gpt4_client",       # ✅ Резолвится из RuntimeContext
    "secret_ref": "openai_key"      # ✅ Резолвится из SecretsProvider
}
```

### Модель данных

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol
from datetime import datetime
from contextvars import ContextVar


# ===== Protocols for Registries =====

class Store(Protocol):
    """Протокол для storage backends."""
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any) -> None: ...
    async def delete(self, key: str) -> None: ...


class Client(Protocol):
    """Протокол для внешних клиентов."""
    async def request(self, method: str, path: str, **kwargs) -> Any: ...


class SecretsProvider(Protocol):
    """Протокол для доступа к секретам."""
    async def get_secret(self, secret_id: str) -> str: ...


class LLMProvider(Protocol):
    """Протокол для LLM провайдеров."""
    async def complete(self, prompt: str, **kwargs) -> str: ...


# ===== Registries =====

@dataclass
class StoreRegistry:
    """Реестр storage backends."""
    
    _stores: Dict[str, Store] = field(default_factory=dict)
    
    def register(self, store_id: str, store: Store) -> None:
        """Зарегистрировать store."""
        self._stores[store_id] = store
    
    def get(self, store_id: str) -> Store:
        """Получить store по ID."""
        if store_id not in self._stores:
            raise ResourceNotFoundError(f"Store '{store_id}' not found")
        return self._stores[store_id]
    
    def list(self) -> list[str]:
        """Список всех store ID."""
        return list(self._stores.keys())


@dataclass
class ClientRegistry:
    """Реестр внешних клиентов (HTTP, gRPC, etc)."""
    
    _clients: Dict[str, Client] = field(default_factory=dict)
    
    def register(self, client_id: str, client: Client) -> None:
        self._clients[client_id] = client
    
    def get(self, client_id: str) -> Client:
        if client_id not in self._clients:
            raise ResourceNotFoundError(f"Client '{client_id}' not found")
        return self._clients[client_id]


@dataclass
class LLMRegistry:
    """Реестр LLM провайдеров."""
    
    _providers: Dict[str, LLMProvider] = field(default_factory=dict)
    
    def register(self, llm_id: str, provider: LLMProvider) -> None:
        self._providers[llm_id] = provider
    
    def get(self, llm_id: str) -> LLMProvider:
        if llm_id not in self._providers:
            raise ResourceNotFoundError(f"LLM provider '{llm_id}' not found")
        return self._providers[llm_id]


# ===== Runtime Context =====

@dataclass
class RuntimeContext:
    """
    Единая точка доступа к enterprise ресурсам.
    
    Передаётся в каждый шаг через injection.
    Содержит все зависимости, разрешённые по ID/ref.
    """
    
    # === Identity ===
    tenant_id: str
    instance_id: str                    # Уникальный ID инстанса workflow
    run_id: str                         # ID текущего запуска
    segment_id: str                     # ID сегмента (pipeline)
    
    # === Resource Registries ===
    stores: StoreRegistry = field(default_factory=StoreRegistry)
    clients: ClientRegistry = field(default_factory=ClientRegistry)
    llms: LLMRegistry = field(default_factory=LLMRegistry)
    secrets: Optional[SecretsProvider] = None
    
    # === Policies (из v1.7.0-v1.9.0) ===
    rate_limiter: Optional["RateLimitedExecutor"] = None
    audit_trail: Optional["AuditTrail"] = None
    
    # === Event Hooks ===
    on_step_start: Optional[Callable[["StepStartEvent"], None]] = None
    on_step_complete: Optional[Callable[["StepCompleteEvent"], None]] = None
    on_step_error: Optional[Callable[["StepErrorEvent"], None]] = None
    on_event: Optional[Callable[["WorktrailEvent"], None]] = None
    
    # === Timestamps ===
    created_at: datetime = field(default_factory=datetime.now)
    
    # === Helpers ===
    
    def resolve_store(self, store_ref: str) -> Store:
        """Resolve store by reference."""
        return self.stores.get(store_ref)
    
    def resolve_client(self, client_ref: str) -> Client:
        """Resolve client by reference."""
        return self.clients.get(client_ref)
    
    def resolve_llm(self, llm_ref: str) -> LLMProvider:
        """Resolve LLM provider by reference."""
        return self.llms.get(llm_ref)
    
    async def resolve_secret(self, secret_ref: str) -> str:
        """Resolve secret by reference."""
        if not self.secrets:
            raise ResourceNotFoundError("SecretsProvider not configured")
        return await self.secrets.get_secret(secret_ref)
    
    def child_context(self, step_id: str) -> "StepContext":
        """Create child context for a step."""
        return StepContext(
            runtime=self,
            step_id=step_id,
        )


@dataclass
class StepContext:
    """Контекст для конкретного шага."""
    
    runtime: RuntimeContext
    step_id: str
    
    # Step-local state
    _state: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tenant_id(self) -> str:
        return self.runtime.tenant_id
    
    @property
    def instance_id(self) -> str:
        return self.runtime.instance_id
    
    @property
    def run_id(self) -> str:
        return self.runtime.run_id
    
    def get_store(self, store_ref: str) -> Store:
        return self.runtime.resolve_store(store_ref)
    
    def get_client(self, client_ref: str) -> Client:
        return self.runtime.resolve_client(client_ref)
    
    def get_llm(self, llm_ref: str) -> LLMProvider:
        return self.runtime.resolve_llm(llm_ref)
    
    async def get_secret(self, secret_ref: str) -> str:
        return await self.runtime.resolve_secret(secret_ref)


# ===== Context Variable =====

current_runtime: ContextVar[RuntimeContext] = ContextVar(
    "current_runtime", 
    default=None
)


# ===== Context Manager =====

class RuntimeContextManager:
    """Менеджер RuntimeContext."""
    
    def __init__(self, context: RuntimeContext):
        self.context = context
        self._token = None
    
    def __enter__(self) -> RuntimeContext:
        self._token = current_runtime.set(self.context)
        return self.context
    
    def __exit__(self, *args) -> None:
        if self._token:
            current_runtime.reset(self._token)
    
    async def __aenter__(self) -> RuntimeContext:
        return self.__enter__()
    
    async def __aexit__(self, *args) -> None:
        self.__exit__(*args)


# ===== Exceptions =====

class ResourceNotFoundError(Exception):
    """Resource not found in registry."""
    pass


class SecretAccessDeniedError(Exception):
    """Access to secret denied."""
    pass
```

### Использование

```python
from llmteam.runtime import RuntimeContext, StoreRegistry, LLMRegistry

# === Setup (один раз при старте приложения) ===

stores = StoreRegistry()
stores.register("main_store", PostgresStore(conn))
stores.register("cache", RedisStore(redis_url))

llms = LLMRegistry()
llms.register("gpt4", OpenAIProvider(api_key_from_vault))
llms.register("claude", AnthropicProvider(api_key_from_vault))

runtime = RuntimeContext(
    tenant_id="acme",
    instance_id="inst_abc123",
    run_id="run_xyz789",
    segment_id="content_pipeline",
    stores=stores,
    llms=llms,
    secrets=VaultSecretsProvider(vault_url),
    audit_trail=audit_trail,
)

# === В шаге (получает только StepContext) ===

async def execute_step(ctx: StepContext, config: dict, input_data: dict):
    # Резолвим ресурсы по ref из конфига
    llm = ctx.get_llm(config["llm_ref"])           # "gpt4" → OpenAIProvider
    store = ctx.get_store(config["store_ref"])     # "main_store" → PostgresStore
    api_key = await ctx.get_secret(config["secret_ref"])  # "api_key" → value
    
    # Используем
    result = await llm.complete(prompt)
    await store.set(f"result:{ctx.run_id}", result)
    
    return result
```

### Изменения в существующих компонентах

```python
# Было (v1.9.0):
class ActionExecutor:
    def __init__(self, registry: ActionRegistry, rate_limiter: RateLimiter):
        self.registry = registry
        self.rate_limiter = rate_limiter  # Живой объект

# Стало (v2.0.0):
class ActionExecutor:
    async def execute(self, ctx: StepContext, action_config: dict) -> ActionResult:
        # Получаем rate_limiter из RuntimeContext
        rate_limiter = ctx.runtime.rate_limiter
        
        # Резолвим client по ref
        client = ctx.get_client(action_config["client_ref"])
        
        # Выполняем
        ...
```

---

## 📑 RFC #2: Worktrail Events (TASK-EVT-01)

### Назначение

Стандартизированные события для отображения прогресса в canvas UI.

### Модель данных

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncIterator
import json


class EventType(Enum):
    """Типы событий Worktrail."""
    
    # Segment lifecycle
    SEGMENT_STARTED = "segment.started"
    SEGMENT_COMPLETED = "segment.completed"
    SEGMENT_FAILED = "segment.failed"
    SEGMENT_CANCELLED = "segment.cancelled"
    SEGMENT_PAUSED = "segment.paused"
    SEGMENT_RESUMED = "segment.resumed"
    
    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_SKIPPED = "step.skipped"
    STEP_RETRYING = "step.retrying"
    
    # Human interaction
    HUMAN_TASK_CREATED = "human.task_created"
    HUMAN_TASK_ASSIGNED = "human.task_assigned"
    HUMAN_TASK_COMPLETED = "human.task_completed"
    HUMAN_TASK_ESCALATED = "human.task_escalated"
    
    # External actions
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    
    # Data flow
    DATA_PRODUCED = "data.produced"
    DATA_CONSUMED = "data.consumed"


class EventSeverity(Enum):
    """Severity level для фильтрации."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ErrorInfo:
    """Информация об ошибке."""
    
    error_type: str                 # "ValidationError"
    error_message: str              # "Field 'email' is required"
    error_code: Optional[str] = None  # "E001"
    stack_trace: Optional[str] = None
    recoverable: bool = False
    
    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "stack_trace": self.stack_trace,
            "recoverable": self.recoverable,
        }


@dataclass
class WorktrailEvent:
    """
    Стандартное событие Worktrail.
    
    Все поля кроме payload — обязательные.
    """
    
    # === Identity (всегда) ===
    event_id: str                   # UUID события
    event_type: EventType           # Тип события
    timestamp: datetime             # Когда произошло
    
    # === Context (всегда) ===
    tenant_id: str                  # ID тенанта
    instance_id: str                # ID инстанса workflow
    run_id: str                     # ID запуска
    segment_id: str                 # ID сегмента
    
    # === Step context (если применимо) ===
    step_id: Optional[str] = None   # ID шага
    step_type: Optional[str] = None # Тип шага ("llm_agent", "http_action")
    
    # === Metadata ===
    severity: EventSeverity = EventSeverity.INFO
    correlation_id: Optional[str] = None  # Для связи событий
    parent_event_id: Optional[str] = None  # Для иерархии
    
    # === Payload (зависит от типа) ===
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # === Error (для *_FAILED событий) ===
    error: Optional[ErrorInfo] = None
    
    # === Timing ===
    duration_ms: Optional[int] = None  # Для *_COMPLETED событий
    
    def to_dict(self) -> dict:
        """Serialize to dict for JSON."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "instance_id": self.instance_id,
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "step_id": self.step_id,
            "step_type": self.step_type,
            "severity": self.severity.value,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "payload": self.payload,
            "error": self.error.to_dict() if self.error else None,
            "duration_ms": self.duration_ms,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorktrailEvent":
        """Deserialize from dict."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tenant_id=data["tenant_id"],
            instance_id=data["instance_id"],
            run_id=data["run_id"],
            segment_id=data["segment_id"],
            step_id=data.get("step_id"),
            step_type=data.get("step_type"),
            severity=EventSeverity(data.get("severity", "info")),
            correlation_id=data.get("correlation_id"),
            parent_event_id=data.get("parent_event_id"),
            payload=data.get("payload", {}),
            error=ErrorInfo(**data["error"]) if data.get("error") else None,
            duration_ms=data.get("duration_ms"),
        )


# ===== Event Emitter =====

class EventEmitter:
    """Эмиттер событий."""
    
    def __init__(self, runtime: RuntimeContext):
        self.runtime = runtime
        self._sequence = 0
    
    def _make_event_id(self) -> str:
        self._sequence += 1
        return f"{self.runtime.run_id}:{self._sequence}"
    
    def emit(
        self,
        event_type: EventType,
        *,
        step_id: str = None,
        step_type: str = None,
        payload: dict = None,
        error: ErrorInfo = None,
        duration_ms: int = None,
        severity: EventSeverity = EventSeverity.INFO,
    ) -> WorktrailEvent:
        """Emit event."""
        event = WorktrailEvent(
            event_id=self._make_event_id(),
            event_type=event_type,
            timestamp=datetime.now(),
            tenant_id=self.runtime.tenant_id,
            instance_id=self.runtime.instance_id,
            run_id=self.runtime.run_id,
            segment_id=self.runtime.segment_id,
            step_id=step_id,
            step_type=step_type,
            severity=severity,
            payload=payload or {},
            error=error,
            duration_ms=duration_ms,
        )
        
        # Call hook if registered
        if self.runtime.on_event:
            self.runtime.on_event(event)
        
        return event
    
    # === Convenience methods ===
    
    def segment_started(self, payload: dict = None) -> WorktrailEvent:
        return self.emit(EventType.SEGMENT_STARTED, payload=payload)
    
    def segment_completed(self, duration_ms: int, payload: dict = None) -> WorktrailEvent:
        return self.emit(EventType.SEGMENT_COMPLETED, duration_ms=duration_ms, payload=payload)
    
    def segment_failed(self, error: ErrorInfo) -> WorktrailEvent:
        return self.emit(EventType.SEGMENT_FAILED, error=error, severity=EventSeverity.ERROR)
    
    def step_started(self, step_id: str, step_type: str, payload: dict = None) -> WorktrailEvent:
        return self.emit(EventType.STEP_STARTED, step_id=step_id, step_type=step_type, payload=payload)
    
    def step_completed(self, step_id: str, step_type: str, duration_ms: int, payload: dict = None) -> WorktrailEvent:
        return self.emit(EventType.STEP_COMPLETED, step_id=step_id, step_type=step_type, duration_ms=duration_ms, payload=payload)
    
    def step_failed(self, step_id: str, step_type: str, error: ErrorInfo) -> WorktrailEvent:
        return self.emit(EventType.STEP_FAILED, step_id=step_id, step_type=step_type, error=error, severity=EventSeverity.ERROR)


# ===== Event Store =====

class EventStore(Protocol):
    """Протокол для хранения событий."""
    
    async def append(self, event: WorktrailEvent) -> None:
        """Append event to store."""
        ...
    
    async def get_by_run(self, run_id: str) -> List[WorktrailEvent]:
        """Get all events for a run."""
        ...
    
    async def get_by_step(self, run_id: str, step_id: str) -> List[WorktrailEvent]:
        """Get events for a specific step."""
        ...


class MemoryEventStore:
    """In-memory event store."""
    
    def __init__(self):
        self._events: List[WorktrailEvent] = []
    
    async def append(self, event: WorktrailEvent) -> None:
        self._events.append(event)
    
    async def get_by_run(self, run_id: str) -> List[WorktrailEvent]:
        return [e for e in self._events if e.run_id == run_id]
    
    async def get_by_step(self, run_id: str, step_id: str) -> List[WorktrailEvent]:
        return [e for e in self._events if e.run_id == run_id and e.step_id == step_id]


# ===== Event Stream (для UI) =====

class EventStream:
    """Стриминг событий для canvas UI."""
    
    def __init__(self, store: EventStore):
        self.store = store
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
    
    async def subscribe(self, run_id: str) -> AsyncIterator[WorktrailEvent]:
        """Subscribe to events for a run."""
        queue = asyncio.Queue()
        
        if run_id not in self._subscribers:
            self._subscribers[run_id] = []
        self._subscribers[run_id].append(queue)
        
        try:
            while True:
                event = await queue.get()
                if event is None:  # Unsubscribe signal
                    break
                yield event
        finally:
            self._subscribers[run_id].remove(queue)
    
    async def publish(self, event: WorktrailEvent) -> None:
        """Publish event to subscribers."""
        await self.store.append(event)
        
        run_id = event.run_id
        if run_id in self._subscribers:
            for queue in self._subscribers[run_id]:
                await queue.put(event)
    
    async def get_history(self, run_id: str) -> List[WorktrailEvent]:
        """Get historical events for a run."""
        return await self.store.get_by_run(run_id)
    
    def unsubscribe_all(self, run_id: str) -> None:
        """Unsubscribe all listeners for a run."""
        if run_id in self._subscribers:
            for queue in self._subscribers[run_id]:
                queue.put_nowait(None)
            del self._subscribers[run_id]
```

### JSON формат события

```json
{
  "event_id": "run_xyz789:42",
  "event_type": "step.completed",
  "timestamp": "2025-01-16T15:30:00.123456",
  "tenant_id": "acme",
  "instance_id": "inst_abc123",
  "run_id": "run_xyz789",
  "segment_id": "content_pipeline",
  "step_id": "validator",
  "step_type": "llm_agent",
  "severity": "info",
  "correlation_id": null,
  "parent_event_id": "run_xyz789:41",
  "payload": {
    "input_tokens": 150,
    "output_tokens": 50,
    "model": "gpt-4"
  },
  "error": null,
  "duration_ms": 1250
}
```

---

## 📑 RFC #3: Segment JSON Contract (TASK-CAN-01)

### Назначение

Единый JSON-формат для хранения в КорпОС и отрисовки в UI.

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://llmteam.ai/schemas/segment/v1.json",
  "title": "Worktrail Segment",
  "description": "LLMTeam segment definition for canvas integration",
  "type": "object",
  "required": ["version", "segment_id", "name", "entrypoint", "steps"],
  "properties": {
    "version": {
      "type": "string",
      "const": "1.0",
      "description": "Schema version"
    },
    "segment_id": {
      "type": "string",
      "pattern": "^[a-z][a-z0-9_]*$",
      "description": "Unique segment identifier"
    },
    "name": {
      "type": "string",
      "description": "Human-readable name"
    },
    "description": {
      "type": "string",
      "description": "Segment description"
    },
    "entrypoint": {
      "type": "string",
      "description": "ID of the first step to execute"
    },
    "params": {
      "type": "object",
      "description": "Segment-level parameters",
      "properties": {
        "max_retries": {"type": "integer", "default": 3},
        "timeout_seconds": {"type": "number", "default": 300},
        "parallel_execution": {"type": "boolean", "default": false}
      }
    },
    "steps": {
      "type": "array",
      "items": {"$ref": "#/definitions/Step"},
      "minItems": 1
    },
    "edges": {
      "type": "array",
      "items": {"$ref": "#/definitions/Edge"},
      "default": []
    },
    "metadata": {
      "type": "object",
      "description": "Custom metadata for UI/storage"
    }
  },
  "definitions": {
    "Step": {
      "type": "object",
      "required": ["step_id", "type"],
      "properties": {
        "step_id": {
          "type": "string",
          "pattern": "^[a-z][a-z0-9_]*$"
        },
        "type": {
          "type": "string",
          "description": "Step type from Step Catalog"
        },
        "name": {
          "type": "string",
          "description": "Display name"
        },
        "config": {
          "type": "object",
          "description": "Step-specific configuration"
        },
        "ports": {
          "$ref": "#/definitions/Ports"
        },
        "position": {
          "$ref": "#/definitions/Position"
        },
        "ui": {
          "$ref": "#/definitions/UIMetadata"
        }
      }
    },
    "Edge": {
      "type": "object",
      "required": ["from", "to"],
      "properties": {
        "from": {"type": "string"},
        "from_port": {"type": "string", "default": "output"},
        "to": {"type": "string"},
        "to_port": {"type": "string", "default": "input"},
        "condition": {
          "type": "string",
          "description": "Optional condition expression"
        }
      }
    },
    "Ports": {
      "type": "object",
      "properties": {
        "input": {
          "type": "array",
          "items": {"type": "string"},
          "default": ["input"]
        },
        "output": {
          "type": "array",
          "items": {"type": "string"},
          "default": ["output"]
        }
      }
    },
    "Position": {
      "type": "object",
      "description": "Position on canvas",
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"}
      }
    },
    "UIMetadata": {
      "type": "object",
      "description": "UI-specific metadata",
      "properties": {
        "color": {"type": "string"},
        "icon": {"type": "string"},
        "collapsed": {"type": "boolean"}
      }
    }
  }
}
```

### Пример сегмента

```json
{
  "version": "1.0",
  "segment_id": "content_pipeline",
  "name": "Content Generation Pipeline",
  "description": "Generates and reviews content",
  "entrypoint": "validator",
  "params": {
    "max_retries": 3,
    "timeout_seconds": 600
  },
  "steps": [
    {
      "step_id": "validator",
      "type": "llm_agent",
      "name": "Input Validator",
      "config": {
        "llm_ref": "gpt4",
        "prompt_template_id": "validate_input_v1",
        "temperature": 0.1
      },
      "ports": {
        "input": ["data"],
        "output": ["validated", "errors"]
      },
      "position": {"x": 100, "y": 100}
    },
    {
      "step_id": "generator",
      "type": "llm_agent",
      "name": "Content Generator",
      "config": {
        "llm_ref": "gpt4",
        "prompt_template_id": "generate_content_v1",
        "temperature": 0.7,
        "max_tokens": 2000
      },
      "ports": {
        "input": ["topic", "style"],
        "output": ["content"]
      },
      "position": {"x": 300, "y": 100}
    },
    {
      "step_id": "reviewer",
      "type": "human_task",
      "name": "Human Review",
      "config": {
        "task_type": "approval",
        "assignee_ref": "content_reviewers",
        "timeout_hours": 24,
        "escalation_chain": ["team_lead", "manager"]
      },
      "ports": {
        "input": ["content"],
        "output": ["approved", "rejected", "modified"]
      },
      "position": {"x": 500, "y": 100}
    },
    {
      "step_id": "publisher",
      "type": "http_action",
      "name": "Publish Content",
      "config": {
        "client_ref": "cms_client",
        "method": "POST",
        "path": "/api/v1/articles",
        "retry_count": 3
      },
      "ports": {
        "input": ["content"],
        "output": ["result"]
      },
      "position": {"x": 700, "y": 100}
    }
  ],
  "edges": [
    {"from": "validator", "from_port": "validated", "to": "generator", "to_port": "topic"},
    {"from": "generator", "from_port": "content", "to": "reviewer", "to_port": "content"},
    {"from": "reviewer", "from_port": "approved", "to": "publisher", "to_port": "content"},
    {"from": "reviewer", "from_port": "modified", "to": "generator", "to_port": "topic"}
  ],
  "metadata": {
    "created_by": "user@acme.com",
    "created_at": "2025-01-16T10:00:00Z",
    "tags": ["content", "generation", "review"]
  }
}
```

### Модель данных Python

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class PortDefinition:
    """Определение порта шага."""
    name: str
    type: str = "any"  # "any", "string", "object", "array"
    required: bool = True
    description: str = ""


@dataclass
class StepPosition:
    """Позиция на canvas."""
    x: float
    y: float


@dataclass
class StepUIMetadata:
    """UI метаданные шага."""
    color: Optional[str] = None
    icon: Optional[str] = None
    collapsed: bool = False


@dataclass
class StepDefinition:
    """Определение шага в сегменте."""
    
    step_id: str
    type: str  # Ссылка на Step Catalog
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Ports
    input_ports: List[str] = field(default_factory=lambda: ["input"])
    output_ports: List[str] = field(default_factory=lambda: ["output"])
    
    # UI
    position: Optional[StepPosition] = None
    ui: Optional[StepUIMetadata] = None
    
    def to_dict(self) -> dict:
        result = {
            "step_id": self.step_id,
            "type": self.type,
            "config": self.config,
            "ports": {
                "input": self.input_ports,
                "output": self.output_ports,
            },
        }
        if self.name:
            result["name"] = self.name
        if self.position:
            result["position"] = {"x": self.position.x, "y": self.position.y}
        if self.ui:
            result["ui"] = {
                "color": self.ui.color,
                "icon": self.ui.icon,
                "collapsed": self.ui.collapsed,
            }
        return result


@dataclass
class EdgeDefinition:
    """Определение связи между шагами."""
    
    from_step: str
    to_step: str
    from_port: str = "output"
    to_port: str = "input"
    condition: Optional[str] = None  # Expression для условных переходов
    
    def to_dict(self) -> dict:
        result = {
            "from": self.from_step,
            "from_port": self.from_port,
            "to": self.to_step,
            "to_port": self.to_port,
        }
        if self.condition:
            result["condition"] = self.condition
        return result


@dataclass
class SegmentParams:
    """Параметры сегмента."""
    max_retries: int = 3
    timeout_seconds: float = 300
    parallel_execution: bool = False


@dataclass
class SegmentDefinition:
    """
    Определение сегмента (Worktrail Segment).
    
    Это основной JSON контракт для canvas.
    """
    
    segment_id: str
    name: str
    entrypoint: str
    steps: List[StepDefinition]
    
    # Optional
    description: str = ""
    version: str = "1.0"
    params: SegmentParams = field(default_factory=SegmentParams)
    edges: List[EdgeDefinition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize to dict (JSON-compatible)."""
        return {
            "version": self.version,
            "segment_id": self.segment_id,
            "name": self.name,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "params": {
                "max_retries": self.params.max_retries,
                "timeout_seconds": self.params.timeout_seconds,
                "parallel_execution": self.params.parallel_execution,
            },
            "steps": [s.to_dict() for s in self.steps],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SegmentDefinition":
        """Deserialize from dict."""
        steps = [
            StepDefinition(
                step_id=s["step_id"],
                type=s["type"],
                name=s.get("name", ""),
                config=s.get("config", {}),
                input_ports=s.get("ports", {}).get("input", ["input"]),
                output_ports=s.get("ports", {}).get("output", ["output"]),
                position=StepPosition(**s["position"]) if s.get("position") else None,
            )
            for s in data["steps"]
        ]
        
        edges = [
            EdgeDefinition(
                from_step=e["from"],
                to_step=e["to"],
                from_port=e.get("from_port", "output"),
                to_port=e.get("to_port", "input"),
                condition=e.get("condition"),
            )
            for e in data.get("edges", [])
        ]
        
        params_data = data.get("params", {})
        params = SegmentParams(
            max_retries=params_data.get("max_retries", 3),
            timeout_seconds=params_data.get("timeout_seconds", 300),
            parallel_execution=params_data.get("parallel_execution", False),
        )
        
        return cls(
            segment_id=data["segment_id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            entrypoint=data["entrypoint"],
            params=params,
            steps=steps,
            edges=edges,
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "SegmentDefinition":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def validate(self) -> List[str]:
        """Validate segment definition. Returns list of errors."""
        errors = []
        
        step_ids = {s.step_id for s in self.steps}
        
        # Check entrypoint exists
        if self.entrypoint not in step_ids:
            errors.append(f"Entrypoint '{self.entrypoint}' not found in steps")
        
        # Check edges reference valid steps
        for edge in self.edges:
            if edge.from_step not in step_ids:
                errors.append(f"Edge from '{edge.from_step}' references unknown step")
            if edge.to_step not in step_ids:
                errors.append(f"Edge to '{edge.to_step}' references unknown step")
        
        # Check for duplicate step IDs
        if len(step_ids) != len(self.steps):
            errors.append("Duplicate step IDs found")
        
        return errors
```

---

## 📑 RFC #4: Step Catalog API (TASK-CAN-02)

### Назначение

Каталог типов шагов с метаданными для UI и JSON Schema для конфигов.

### Модель данных

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class StepCategory(Enum):
    """Категории шагов для группировки в UI."""
    AI = "ai"
    DATA = "data"
    INTEGRATION = "integration"
    CONTROL = "control"
    HUMAN = "human"
    UTILITY = "utility"


@dataclass
class PortSpec:
    """Спецификация порта."""
    name: str
    type: str = "any"           # JSON Schema type
    description: str = ""
    required: bool = True
    default: Any = None
    schema: Optional[dict] = None  # Full JSON Schema for complex types


@dataclass
class StepTypeMetadata:
    """
    Метаданные типа шага для Step Catalog.
    
    Используется canvas для:
    - Построения палитры блоков
    - Генерации форм конфигурации
    - Валидации связей между шагами
    """
    
    # === Identity ===
    type_id: str                    # "llm_agent", "http_action"
    version: str                    # "1.0"
    
    # === Display ===
    display_name: str               # "LLM Agent"
    description: str                # "Executes LLM prompt"
    category: StepCategory          # StepCategory.AI
    icon: str = "robot"             # Icon name for UI
    color: str = "#4A90D9"          # Default color
    
    # === Configuration Schema ===
    config_schema: Dict[str, Any] = field(default_factory=dict)  # JSON Schema
    
    # === Ports ===
    input_ports: List[PortSpec] = field(default_factory=list)
    output_ports: List[PortSpec] = field(default_factory=list)
    
    # === Behavior ===
    supports_retry: bool = True
    supports_timeout: bool = True
    supports_parallel: bool = False
    is_async: bool = True
    
    # === Documentation ===
    docs_url: Optional[str] = None
    examples: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Export for UI consumption."""
        return {
            "type_id": self.type_id,
            "version": self.version,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "icon": self.icon,
            "color": self.color,
            "config_schema": self.config_schema,
            "input_ports": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.input_ports
            ],
            "output_ports": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                }
                for p in self.output_ports
            ],
            "supports_retry": self.supports_retry,
            "supports_timeout": self.supports_timeout,
            "supports_parallel": self.supports_parallel,
            "is_async": self.is_async,
            "docs_url": self.docs_url,
            "examples": self.examples,
        }


class StepCatalog:
    """
    Каталог типов шагов.
    
    Singleton, доступен через StepCatalog.instance().
    """
    
    _instance: Optional["StepCatalog"] = None
    
    def __init__(self):
        self._types: Dict[str, StepTypeMetadata] = {}
        self._handlers: Dict[str, Callable] = {}
        self._version = "1.0"
    
    @classmethod
    def instance(cls) -> "StepCatalog":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtin_types()
        return cls._instance
    
    def register(
        self,
        metadata: StepTypeMetadata,
        handler: Callable = None,
    ) -> None:
        """
        Register step type.
        
        Args:
            metadata: Step type metadata
            handler: Optional handler function/class
        """
        self._types[metadata.type_id] = metadata
        if handler:
            self._handlers[metadata.type_id] = handler
    
    def get(self, type_id: str) -> Optional[StepTypeMetadata]:
        """Get step type metadata."""
        return self._types.get(type_id)
    
    def get_handler(self, type_id: str) -> Optional[Callable]:
        """Get step handler."""
        return self._handlers.get(type_id)
    
    def list_all(self) -> List[StepTypeMetadata]:
        """List all registered step types."""
        return list(self._types.values())
    
    def list_by_category(self, category: StepCategory) -> List[StepTypeMetadata]:
        """List step types by category."""
        return [t for t in self._types.values() if t.category == category]
    
    def export_for_ui(self) -> dict:
        """
        Export catalog for canvas UI.
        
        Returns dict suitable for JSON serialization.
        """
        return {
            "version": self._version,
            "categories": [c.value for c in StepCategory],
            "types": {
                type_id: meta.to_dict()
                for type_id, meta in self._types.items()
            },
        }
    
    def validate_step_config(self, type_id: str, config: dict) -> List[str]:
        """Validate step config against schema."""
        metadata = self.get(type_id)
        if not metadata:
            return [f"Unknown step type: {type_id}"]
        
        # TODO: Implement JSON Schema validation
        errors = []
        return errors
    
    def _register_builtin_types(self) -> None:
        """Register built-in step types."""
        
        # LLM Agent
        self.register(StepTypeMetadata(
            type_id="llm_agent",
            version="1.0",
            display_name="LLM Agent",
            description="Execute LLM prompt with optional tools",
            category=StepCategory.AI,
            icon="robot",
            color="#4A90D9",
            config_schema={
                "type": "object",
                "properties": {
                    "llm_ref": {
                        "type": "string",
                        "description": "Reference to LLM provider",
                    },
                    "prompt_template_id": {
                        "type": "string",
                        "description": "Prompt template ID",
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 2,
                        "default": 0.7,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1000,
                    },
                },
                "required": ["llm_ref"],
            },
            input_ports=[
                PortSpec("input", "object", "Input data"),
            ],
            output_ports=[
                PortSpec("output", "string", "LLM response"),
                PortSpec("error", "object", "Error if failed"),
            ],
        ))
        
        # HTTP Action
        self.register(StepTypeMetadata(
            type_id="http_action",
            version="1.0",
            display_name="HTTP Request",
            description="Make HTTP request to external API",
            category=StepCategory.INTEGRATION,
            icon="globe",
            color="#50C878",
            config_schema={
                "type": "object",
                "properties": {
                    "client_ref": {
                        "type": "string",
                        "description": "Reference to HTTP client",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        "default": "POST",
                    },
                    "path": {
                        "type": "string",
                        "description": "Request path",
                    },
                    "headers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "retry_count": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                    },
                },
                "required": ["client_ref", "path"],
            },
            input_ports=[
                PortSpec("body", "object", "Request body"),
            ],
            output_ports=[
                PortSpec("response", "object", "Response data"),
                PortSpec("status", "integer", "HTTP status code"),
            ],
        ))
        
        # Human Task
        self.register(StepTypeMetadata(
            type_id="human_task",
            version="1.0",
            display_name="Human Task",
            description="Request human input or approval",
            category=StepCategory.HUMAN,
            icon="user",
            color="#FF6B6B",
            config_schema={
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["approval", "input", "review", "choice"],
                        "default": "approval",
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title",
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description",
                    },
                    "assignee_ref": {
                        "type": "string",
                        "description": "Reference to assignee/group",
                    },
                    "timeout_hours": {
                        "type": "number",
                        "default": 24,
                    },
                    "escalation_chain": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["task_type"],
            },
            input_ports=[
                PortSpec("data", "object", "Data for human review"),
            ],
            output_ports=[
                PortSpec("approved", "object", "Output if approved"),
                PortSpec("rejected", "object", "Output if rejected"),
                PortSpec("modified", "object", "Output if modified"),
            ],
            supports_parallel=False,
        ))
        
        # Condition (branching)
        self.register(StepTypeMetadata(
            type_id="condition",
            version="1.0",
            display_name="Condition",
            description="Branch based on condition",
            category=StepCategory.CONTROL,
            icon="git-branch",
            color="#9B59B6",
            config_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Condition expression (Python-like)",
                    },
                },
                "required": ["expression"],
            },
            input_ports=[
                PortSpec("input", "any", "Data to evaluate"),
            ],
            output_ports=[
                PortSpec("true", "any", "Output if true"),
                PortSpec("false", "any", "Output if false"),
            ],
        ))
        
        # Parallel Split
        self.register(StepTypeMetadata(
            type_id="parallel_split",
            version="1.0",
            display_name="Parallel Split",
            description="Execute multiple branches in parallel",
            category=StepCategory.CONTROL,
            icon="git-fork",
            color="#9B59B6",
            config_schema={
                "type": "object",
                "properties": {
                    "branches": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 2,
                    },
                },
            },
            input_ports=[
                PortSpec("input", "any", "Data to distribute"),
            ],
            output_ports=[
                PortSpec("branch_1", "any", "Branch 1 output"),
                PortSpec("branch_2", "any", "Branch 2 output"),
            ],
            supports_parallel=True,
        ))
        
        # Parallel Join
        self.register(StepTypeMetadata(
            type_id="parallel_join",
            version="1.0",
            display_name="Parallel Join",
            description="Wait for all parallel branches",
            category=StepCategory.CONTROL,
            icon="git-merge",
            color="#9B59B6",
            config_schema={
                "type": "object",
                "properties": {
                    "merge_strategy": {
                        "type": "string",
                        "enum": ["all", "any", "first"],
                        "default": "all",
                    },
                },
            },
            input_ports=[
                PortSpec("branch_1", "any", "Branch 1 input"),
                PortSpec("branch_2", "any", "Branch 2 input"),
            ],
            output_ports=[
                PortSpec("output", "array", "Merged results"),
            ],
        ))
        
        # Data Transform
        self.register(StepTypeMetadata(
            type_id="transform",
            version="1.0",
            display_name="Transform",
            description="Transform data using expression",
            category=StepCategory.DATA,
            icon="shuffle",
            color="#F39C12",
            config_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Transform expression (JSONPath-like)",
                    },
                    "mapping": {
                        "type": "object",
                        "description": "Field mapping",
                    },
                },
            },
            input_ports=[
                PortSpec("input", "any", "Input data"),
            ],
            output_ports=[
                PortSpec("output", "any", "Transformed data"),
            ],
        ))
```

---

## 📑 RFC #5: Segment Runner (TASK-SEG-01)

### Назначение

Единая точка запуска сегмента с поддержкой cancel, timeout, retry hooks.

### Модель данных

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio


class SegmentStatus(Enum):
    """Статус выполнения сегмента."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class SegmentResult:
    """Результат выполнения сегмента."""
    
    run_id: str
    segment_id: str
    status: SegmentStatus
    
    # Output
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Error (if failed)
    error: Optional[ErrorInfo] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    
    # Steps info
    steps_completed: int = 0
    steps_total: int = 0
    current_step: Optional[str] = None
    
    # Events
    events: List[WorktrailEvent] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "current_step": self.current_step,
        }


@dataclass
class RunConfig:
    """Конфигурация запуска."""
    
    timeout: Optional[timedelta] = None
    max_retries: int = 3
    retry_delay: timedelta = timedelta(seconds=1)
    
    # Callbacks
    on_step_start: Optional[Callable] = None
    on_step_complete: Optional[Callable] = None
    on_step_error: Optional[Callable] = None
    on_cancel: Optional[Callable] = None
    
    # Persistence
    snapshot_interval: int = 0  # 0 = disabled, N = every N steps


class SegmentRunner:
    """
    Единая точка запуска сегмента.
    
    Используется КорпОС для выполнения сегментов как под-workflows.
    """
    
    def __init__(
        self,
        catalog: StepCatalog = None,
        event_stream: EventStream = None,
        snapshot_manager: SnapshotManager = None,
    ):
        self.catalog = catalog or StepCatalog.instance()
        self.event_stream = event_stream
        self.snapshot_manager = snapshot_manager
        
        self._running: Dict[str, asyncio.Task] = {}
        self._cancelled: set = set()
    
    async def run(
        self,
        segment: SegmentDefinition,
        runtime: RuntimeContext,
        input_data: Dict[str, Any],
        *,
        config: RunConfig = None,
    ) -> SegmentResult:
        """
        Execute segment.
        
        Args:
            segment: Segment definition (from JSON)
            runtime: Runtime context with resources
            input_data: Input data for entrypoint
            config: Run configuration
            
        Returns:
            SegmentResult with output or error
        """
        config = config or RunConfig()
        run_id = runtime.run_id
        
        # Create emitter
        emitter = EventEmitter(runtime)
        
        # Initialize result
        result = SegmentResult(
            run_id=run_id,
            segment_id=segment.segment_id,
            status=SegmentStatus.RUNNING,
            started_at=datetime.now(),
            steps_total=len(segment.steps),
        )
        
        # Emit start event
        emitter.segment_started({"input": input_data})
        
        try:
            # Create task
            task = asyncio.create_task(
                self._execute_segment(segment, runtime, input_data, emitter, result, config)
            )
            self._running[run_id] = task
            
            # Apply timeout
            if config.timeout:
                output = await asyncio.wait_for(task, config.timeout.total_seconds())
            else:
                output = await task
            
            # Success
            result.status = SegmentStatus.COMPLETED
            result.output = output
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - result.started_at).total_seconds() * 1000)
            
            emitter.segment_completed(result.duration_ms, {"output": output})
            
        except asyncio.CancelledError:
            result.status = SegmentStatus.CANCELLED
            result.completed_at = datetime.now()
            
            if config.on_cancel:
                await config.on_cancel(result)
            
        except asyncio.TimeoutError:
            result.status = SegmentStatus.TIMEOUT
            result.completed_at = datetime.now()
            result.error = ErrorInfo(
                error_type="TimeoutError",
                error_message=f"Segment timed out after {config.timeout}",
                recoverable=True,
            )
            emitter.segment_failed(result.error)
            
        except Exception as e:
            result.status = SegmentStatus.FAILED
            result.completed_at = datetime.now()
            result.error = ErrorInfo(
                error_type=type(e).__name__,
                error_message=str(e),
                recoverable=False,
            )
            emitter.segment_failed(result.error)
            
        finally:
            self._running.pop(run_id, None)
            self._cancelled.discard(run_id)
        
        return result
    
    async def cancel(self, run_id: str) -> bool:
        """
        Cancel running segment.
        
        Returns True if cancelled, False if not found.
        """
        task = self._running.get(run_id)
        if not task:
            return False
        
        self._cancelled.add(run_id)
        task.cancel()
        return True
    
    async def get_status(self, run_id: str) -> Optional[SegmentStatus]:
        """Get status of a run."""
        if run_id in self._running:
            if run_id in self._cancelled:
                return SegmentStatus.CANCELLED
            return SegmentStatus.RUNNING
        return None
    
    async def pause(self, run_id: str) -> Optional[str]:
        """
        Pause segment and create snapshot.
        
        Returns snapshot_id if successful.
        """
        # TODO: Implement with SnapshotManager
        pass
    
    async def resume(self, snapshot_id: str, runtime: RuntimeContext) -> SegmentResult:
        """
        Resume segment from snapshot.
        """
        # TODO: Implement with SnapshotManager
        pass
    
    async def _execute_segment(
        self,
        segment: SegmentDefinition,
        runtime: RuntimeContext,
        input_data: dict,
        emitter: EventEmitter,
        result: SegmentResult,
        config: RunConfig,
    ) -> dict:
        """Execute segment steps."""
        
        # Build execution graph
        step_map = {s.step_id: s for s in segment.steps}
        edge_map = self._build_edge_map(segment.edges)
        
        # State
        step_outputs: Dict[str, Any] = {}
        current_step_id = segment.entrypoint
        
        while current_step_id:
            # Check cancellation
            if runtime.run_id in self._cancelled:
                raise asyncio.CancelledError()
            
            step_def = step_map[current_step_id]
            result.current_step = current_step_id
            
            # Create step context
            step_ctx = runtime.child_context(current_step_id)
            
            # Get handler
            handler = self.catalog.get_handler(step_def.type)
            if not handler:
                raise ValueError(f"No handler for step type: {step_def.type}")
            
            # Gather input from edges
            step_input = self._gather_step_input(
                current_step_id,
                edge_map,
                step_outputs,
                input_data if current_step_id == segment.entrypoint else None,
            )
            
            # Emit step started
            emitter.step_started(current_step_id, step_def.type, {"input": step_input})
            step_start = datetime.now()
            
            # Execute with retry
            try:
                output = await self._execute_step_with_retry(
                    handler,
                    step_ctx,
                    step_def.config,
                    step_input,
                    config,
                )
                
                step_duration = int((datetime.now() - step_start).total_seconds() * 1000)
                step_outputs[current_step_id] = output
                result.steps_completed += 1
                
                emitter.step_completed(current_step_id, step_def.type, step_duration, {"output": output})
                
                if config.on_step_complete:
                    await config.on_step_complete(current_step_id, output)
                
            except Exception as e:
                error = ErrorInfo(
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                emitter.step_failed(current_step_id, step_def.type, error)
                
                if config.on_step_error:
                    await config.on_step_error(current_step_id, e)
                
                raise
            
            # Snapshot if configured
            if config.snapshot_interval > 0 and result.steps_completed % config.snapshot_interval == 0:
                if self.snapshot_manager:
                    await self.snapshot_manager.create_snapshot(...)
            
            # Determine next step
            current_step_id = self._get_next_step(
                current_step_id,
                edge_map,
                output,
            )
        
        # Return final output
        return step_outputs.get(segment.steps[-1].step_id, {})
    
    async def _execute_step_with_retry(
        self,
        handler: Callable,
        ctx: StepContext,
        config: dict,
        input_data: dict,
        run_config: RunConfig,
    ) -> Any:
        """Execute step with retry logic."""
        last_error = None
        
        for attempt in range(run_config.max_retries + 1):
            try:
                return await handler(ctx, config, input_data)
            except Exception as e:
                last_error = e
                if attempt < run_config.max_retries:
                    await asyncio.sleep(run_config.retry_delay.total_seconds())
        
        raise last_error
    
    def _build_edge_map(self, edges: List[EdgeDefinition]) -> Dict[str, List[EdgeDefinition]]:
        """Build map of outgoing edges for each step."""
        edge_map = {}
        for edge in edges:
            if edge.from_step not in edge_map:
                edge_map[edge.from_step] = []
            edge_map[edge.from_step].append(edge)
        return edge_map
    
    def _gather_step_input(
        self,
        step_id: str,
        edge_map: dict,
        step_outputs: dict,
        initial_input: dict = None,
    ) -> dict:
        """Gather input for step from incoming edges."""
        if initial_input:
            return initial_input
        
        # Find incoming edges
        inputs = {}
        for from_step, edges in edge_map.items():
            for edge in edges:
                if edge.to_step == step_id:
                    output = step_outputs.get(from_step, {})
                    inputs[edge.to_port] = output.get(edge.from_port, output)
        
        return inputs
    
    def _get_next_step(
        self,
        current_step: str,
        edge_map: dict,
        output: Any,
    ) -> Optional[str]:
        """Determine next step based on edges and output."""
        edges = edge_map.get(current_step, [])
        
        if not edges:
            return None
        
        # For now, just take first edge
        # TODO: Implement condition evaluation
        for edge in edges:
            if edge.condition:
                # Evaluate condition
                pass
            return edge.to_step
        
        return None
```

---

## 📁 Структура файлов v2.0.0

```
src/llmteam/
│
├── __init__.py                   # UPDATED: LLMTeam as main export
│
├── core/                         # NEW: Core classes (RFC #8)
│   ├── __init__.py
│   ├── team.py                  # LLMTeam (renamed from Pipeline)
│   ├── agent.py                 # Agent with ports
│   ├── group.py                 # Group
│   └── config.py                # TeamConfig, GroupConfig
│
├── ports/                        # NEW: Port Architecture (RFC #7)
│   ├── __init__.py
│   ├── models.py                # Port, PortSet, PortLevel, PortDirection
│   ├── presets.py               # PortPresets (agent, orchestrator, etc)
│   ├── connections.py           # PortConnection, routing
│   └── component.py             # PortedComponent base class
│
├── orchestration/                # UPDATED: Renamed orchestrators
│   ├── __init__.py
│   ├── team_orch.py             # TeamOrchestrator (was PipelineOrchestrator)
│   ├── group_orch.py            # GroupOrchestrator
│   ├── strategies.py            # OrchestrationStrategy, RuleBasedStrategy
│   └── decisions.py             # OrchestratorDecision
│
├── patterns/                     # NEW: Interaction Patterns (RFC #6)
│   ├── __init__.py
│   ├── critic_loop.py           # CriticLoop, CriticLoopConfig
│   ├── multi_critic.py          # MultiCriticLoop
│   ├── self_critic.py           # SelfCriticLoop
│   └── tournament.py            # TournamentLoop
│
├── runtime/                      # NEW: RuntimeContext (RFC #1)
│   ├── __init__.py
│   ├── context.py               # RuntimeContext, StepContext
│   ├── registries.py            # StoreRegistry, ClientRegistry, LLMRegistry
│   └── providers.py             # SecretsProvider protocols
│
├── events/                       # NEW: Worktrail Events (RFC #2)
│   ├── __init__.py
│   ├── models.py                # WorktrailEvent, EventType, ErrorInfo
│   ├── emitter.py               # EventEmitter
│   ├── stream.py                # EventStream
│   └── stores/
│       ├── memory.py            # MemoryEventStore
│       └── postgres.py          # PostgresEventStore
│
├── segment/                      # NEW: Segment JSON (RFC #3)
│   ├── __init__.py
│   ├── models.py                # SegmentDefinition, StepDefinition, EdgeDefinition
│   ├── parser.py                # JSON parsing/validation
│   └── schema.py                # JSON Schema definitions
│
├── catalog/                      # NEW: Step Catalog (RFC #4)
│   ├── __init__.py
│   ├── models.py                # StepTypeMetadata, PortSpec
│   ├── catalog.py               # StepCatalog
│   └── builtin/                 # Built-in step types
│       ├── llm_agent.py
│       ├── http_action.py
│       ├── human_task.py
│       ├── critic_loop.py       # CriticLoop as step type
│       └── control.py
│
├── runner/                       # NEW: Segment Runner (RFC #5)
│   ├── __init__.py
│   ├── runner.py                # SegmentRunner
│   ├── executor.py              # Step execution logic
│   └── scheduler.py             # Step scheduling
│
├── compat/                       # NEW: Backward Compatibility
│   ├── __init__.py
│   └── aliases.py               # Pipeline → LLMTeam aliases
│
├── tenancy/                      # FROM v1.7.0
├── audit/                        # FROM v1.7.0
├── context/                      # FROM v1.7.0
├── ratelimit/                    # FROM v1.7.0
├── licensing/                    # FROM v1.8.0
├── execution/                    # FROM v1.8.0
├── roles/                        # FROM v1.8.0 (process mining)
├── actions/                      # FROM v1.9.0
├── human/                        # FROM v1.9.0
└── persistence/                  # FROM v1.9.0
```

---

## 📅 План реализации

| Неделя | Задачи | RFC |
|--------|--------|-----|
| 1-2 | RuntimeContext, registries, injection | RFC #1 |
| 3 | WorktrailEvent, EventEmitter, EventStream | RFC #2 |
| 4 | SegmentDefinition, JSON Schema, parser | RFC #3 |
| 5-6 | StepCatalog, built-in types, UI export | RFC #4 |
| 7-8 | SegmentRunner, execution, cancel/timeout | RFC #5 |
| 9 | CriticLoop, MultiCriticLoop patterns | RFC #6 |
| 10 | Three-Level Port Architecture | RFC #7 |
| 11 | Pipeline → LLMTeam rename, migration | RFC #8 |
| 12 | Instance namespacing, Human tasks integration | P1 tasks |
| 13 | Integration documentation, testing | TASK-DOC-01 |

**Итого: ~13 недель**

---

## ✅ Критерии готовности v2.0.0

### P0 — Canvas Integration
- [ ] Segment JSON валидируется schema
- [ ] Step Catalog экспортируется для canvas UI
- [ ] RuntimeContext — единственный способ доступа к ресурсам
- [ ] Все события имеют обязательные поля
- [ ] SegmentRunner поддерживает cancel/timeout/pause/resume

### RFC #6 — Critic Loop
- [ ] CriticLoop работает с настраиваемыми условиями выхода
- [ ] MultiCriticLoop агрегирует feedback от нескольких критиков
- [ ] События critic_loop.* отправляются для UI

### RFC #7 — Port Architecture  
- [ ] Все компоненты имеют три уровня портов (workflow/agent/human)
- [ ] PortConnection поддерживает условный routing
- [ ] Canvas может отображать порты и соединения

### RFC #8 — LLMTeam Rename
- [ ] Pipeline переименован в LLMTeam
- [ ] Обратная совместимость через aliases
- [ ] Гайд по миграции документирован

### P1 — Production Ready
- [ ] Instance namespacing работает автоматически
- [ ] Human tasks интегрированы с events
- [ ] Документация для интеграции с КорпОС

---

## 📊 Сводка RFC v2.0.0

| # | RFC | Статус | Effort |
|---|-----|--------|--------|
| 1 | RuntimeContext Injection | 📋 Planned | 2 нед |
| 2 | Worktrail Events | 📋 Planned | 1 нед |
| 3 | Segment JSON Contract | 📋 Planned | 1 нед |
| 4 | Step Catalog API | 📋 Planned | 1.5 нед |
| 5 | Segment Runner | 📋 Planned | 1.5 нед |
| 6 | Critic Loop Pattern | 📋 Planned | 1 нед |
| 7 | Three-Level Port Architecture | 📋 Planned | 1 нед |
| 8 | Pipeline → LLMTeam Rename | 📋 Planned | 1 нед |

**Общий effort: ~13 недель**

---

## 🎯 Результат v2.0.0

```python
# === Новый API с LLMTeam ===

from llmteam import (
    # Core
    LLMTeam, Agent, Group,
    TeamOrchestrator, GroupOrchestrator,
    
    # Patterns
    CriticLoop, CriticLoopConfig,
    
    # Runtime
    RuntimeContext,
    
    # Segment
    SegmentDefinition, SegmentRunner,
    
    # Events
    EventStream,
)

# === Создаём агентов ===

writer = Agent(
    name="Writer",
    llm_ref="gpt4",
    system_prompt="You are a content writer...",
)

reviewer = Agent(
    name="Reviewer", 
    llm_ref="gpt4",
    system_prompt="You are a critical reviewer...",
)

# === Создаём команду ===

content_team = LLMTeam(
    name="Content Team",
    agents=[writer, reviewer],
    orchestrator=TeamOrchestrator(
        strategy=RuleBasedStrategy(),
        enable_human_escalation=True,  # human_out/human_in порты
    ),
)

# === Или используем CriticLoop ===

improvement_loop = CriticLoop(
    generator=writer,
    critic=reviewer,
    config=CriticLoopConfig(
        max_iterations=5,
        quality_threshold=0.85,
    ),
)

# === Runtime Context ===

runtime = RuntimeContext(
    tenant_id="acme",
    instance_id="inst_123",
    run_id="run_456",
    segment_id="content_creation",
    stores=stores,
    llms=llms,
    secrets=vault,
)

# === Запуск через код ===

result = await content_team.run(runtime, {
    "task": "Write a blog post about AI agents"
})

# === Или через JSON Segment (Canvas) ===

segment_json = '''
{
  "version": "1.0",
  "segment_id": "content_workflow",
  "entrypoint": "writer",
  "steps": [
    {"step_id": "writer", "type": "llm_agent", "config": {"llm_ref": "gpt4"}},
    {"step_id": "improve", "type": "critic_loop", "config": {
      "generator_ref": "writer",
      "critic_ref": "reviewer",
      "max_iterations": 5
    }},
    {"step_id": "approve", "type": "human_task", "config": {"task_type": "approval"}}
  ],
  "edges": [
    {"from": "writer", "to": "improve"},
    {"from": "improve", "to": "approve"}
  ]
}
'''

segment = SegmentDefinition.from_json(segment_json)
runner = SegmentRunner(event_stream=EventStream())

result = await runner.run(segment, runtime, {"topic": "AI agents"})

# === Canvas подписывается на события ===

async for event in event_stream.subscribe(runtime.run_id):
    # Обновляем UI
    if event.event_type == "step.completed":
        canvas.mark_step_complete(event.step_id)
    elif event.event_type == "human.task_created":
        canvas.show_human_task(event.payload)
    elif event.event_type == "critic_loop.iteration_completed":
        canvas.update_score(event.payload["score"])
```

### Полная иерархия v2.0.0

```
═══════════════════════════════════════════════════════════════════════════════

                              WORKFLOW (КорпОС)
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
                  workflow_in              workflow_out
                        │                         ▲
════════════════════════╪═════════════════════════╪════════════════════════════
                        │         GROUP           │
                        ▼      (Департамент)      │
               ┌────────────────────────────────────────┐
               │          GroupOrchestrator            │
               │                 │                     │
               │    ┌────────────┼────────────┐       │
               │    ▼            ▼            ▼       │
               │ LLMTeam     LLMTeam     LLMTeam      │
               │ (Team A)    (Team B)    (Team C)     │
               └────────────────────────────────────────┘
                        │                         │
════════════════════════╪═════════════════════════╪════════════════════════════
                        │       LLMTeam           │
                        ▼       (Команда)         │
               ┌────────────────────────────────────────┐
               │          TeamOrchestrator             │──────▶ human_out
               │                 │                     │             │
               │    ┌────────────┼────────────┐       │             ▼
               │    ▼            ▼            ▼       │        ┌─────────┐
               │  Agent       Agent       Agent       │        │  Human  │
               │ (Writer)   (Reviewer)  (Publisher)   │        │   Chat  │
               │    │            │            │       │        └────┬────┘
               │    └────────────┴────────────┘       │             │
               │         agent_in / agent_out         │◀────────────┘
               └────────────────────────────────────────┘        human_in

═══════════════════════════════════════════════════════════════════════════════

ПОРТЫ:
  workflow_in/out  — связь с внешним миром (КорпОС)
  agent_in/out     — связь между агентами внутри команды
  human_in/out     — диалог с человеком через чат

═══════════════════════════════════════════════════════════════════════════════
```

---

---

## 📑 RFC #7: Three-Level Port Architecture

### Назначение

Унифицированная архитектура портов для всех компонентов системы с тремя уровнями связей: workflow, agent, human.

### Архитектура

```
═══════════════════════════════════════════════════════════════════════════════

                              WORKFLOW (внешний мир)
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   workflow_in              workflow_out
                         │                         ▲
═════════════════════════╪═════════════════════════╪═══════════════════════════
                         │       LLMTeam           │
                         ▼                         │
                ┌──────────────────────────────────┴───┐
                │            ORCHESTRATOR              │
                │                                      │◀═══╗
    agent_in ──▶│  • routing logic                    │    ║
   (результаты  │  • state management                 │────╫───▶ agent_out
    от агентов) │  • decision making                  │    ║     (команды)
                │  • escalation                       │    ║
                └──────────────────────┬───────────────┘    ║
                                       │                    ║
                                       ▼                    ║
                                  human_out                 ║
                                       │                    ║
                                       ▼                    ║
                              ┌─────────────────┐          ║
                              │   HUMAN (чат)   │          ║
                              │                 │          ║
                              │  💬 Вопрос?     │          ║
                              │  ✅ Approve     │          ║
                              │  ❌ Reject      │          ║
                              │  ✏️ Modify      │          ║
                              └────────┬────────┘          ║
                                       │                    ║
                                       ▼                    ║
                                  human_in ════════════════╝

═══════════════════════════════════════════════════════════════════════════════
```

### Три уровня портов

| Уровень | Input Port | Output Port | Описание |
|---------|------------|-------------|----------|
| **Workflow** | `workflow_in` | `workflow_out` | Связь с внешним миром (КорпОС) |
| **Agent** | `agent_in` | `agent_out` | Связь между агентами внутри team |
| **Human** | `human_in` | `human_out` | Диалог с человеком через чат |

### Модель данных

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class PortLevel(Enum):
    """Уровень порта."""
    WORKFLOW = "workflow"    # Внешний мир
    AGENT = "agent"          # Между агентами
    HUMAN = "human"          # Человек


class PortDirection(Enum):
    """Направление порта."""
    INPUT = "input"
    OUTPUT = "output"


class HumanInteractionType(Enum):
    """Типы human interaction."""
    
    # Output (к человеку)
    CHAT_MESSAGE = "chat_message"       # Сообщение в чат
    APPROVAL_REQUEST = "approval"        # Запрос подтверждения
    CHOICE_REQUEST = "choice"           # Выбор из вариантов
    INPUT_REQUEST = "input"             # Запрос ввода данных
    NOTIFICATION = "notification"        # Уведомление (без ответа)
    
    # Input (от человека)  
    CHAT_RESPONSE = "chat_response"     # Ответ в чате
    APPROVAL_RESPONSE = "approved"      # Подтверждение
    REJECTION = "rejected"              # Отклонение
    MODIFICATION = "modified"           # Изменённые данные
    USER_INPUT = "user_input"           # Введённые данные


@dataclass
class Port:
    """Определение порта."""
    
    name: str
    level: PortLevel
    direction: PortDirection
    data_type: str = "any"              # JSON Schema type
    description: str = ""
    required: bool = True
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level.value,
            "direction": self.direction.value,
            "data_type": self.data_type,
            "description": self.description,
            "required": self.required,
        }


@dataclass
class PortSet:
    """Набор портов компонента."""
    
    # Workflow level
    workflow_in: List[Port] = field(default_factory=lambda: [
        Port("input", PortLevel.WORKFLOW, PortDirection.INPUT, 
             "object", "Input from external workflow")
    ])
    workflow_out: List[Port] = field(default_factory=lambda: [
        Port("output", PortLevel.WORKFLOW, PortDirection.OUTPUT,
             "any", "Output to external workflow"),
        Port("error", PortLevel.WORKFLOW, PortDirection.OUTPUT,
             "object", "Error output", required=False)
    ])
    
    # Agent level
    agent_in: List[Port] = field(default_factory=lambda: [
        Port("data", PortLevel.AGENT, PortDirection.INPUT,
             "any", "Data from other agents")
    ])
    agent_out: List[Port] = field(default_factory=lambda: [
        Port("result", PortLevel.AGENT, PortDirection.OUTPUT,
             "any", "Result to other agents")
    ])
    
    # Human level
    human_out: List[Port] = field(default_factory=list)
    human_in: List[Port] = field(default_factory=list)
    
    def has_human_ports(self) -> bool:
        return bool(self.human_out or self.human_in)
    
    def to_dict(self) -> dict:
        return {
            "workflow_in": [p.to_dict() for p in self.workflow_in],
            "workflow_out": [p.to_dict() for p in self.workflow_out],
            "agent_in": [p.to_dict() for p in self.agent_in],
            "agent_out": [p.to_dict() for p in self.agent_out],
            "human_out": [p.to_dict() for p in self.human_out],
            "human_in": [p.to_dict() for p in self.human_in],
        }


# === Preset Port Sets ===

class PortPresets:
    """Предустановленные наборы портов."""
    
    @staticmethod
    def agent() -> PortSet:
        """Порты для обычного агента (без human)."""
        return PortSet(
            human_out=[],
            human_in=[],
        )
    
    @staticmethod
    def orchestrator() -> PortSet:
        """Порты для оркестратора (с human)."""
        return PortSet(
            human_out=[
                Port("message", PortLevel.HUMAN, PortDirection.OUTPUT,
                     "object", "Message to human"),
                Port("approval_request", PortLevel.HUMAN, PortDirection.OUTPUT,
                     "object", "Approval request"),
                Port("choice_request", PortLevel.HUMAN, PortDirection.OUTPUT,
                     "object", "Choice request"),
            ],
            human_in=[
                Port("response", PortLevel.HUMAN, PortDirection.INPUT,
                     "object", "Response from human"),
                Port("approval", PortLevel.HUMAN, PortDirection.INPUT,
                     "object", "Approval decision"),
                Port("choice", PortLevel.HUMAN, PortDirection.INPUT,
                     "object", "Selected choice"),
            ],
        )
    
    @staticmethod
    def human_agent() -> PortSet:
        """Порты для агента с human interaction."""
        ports = PortPresets.agent()
        ports.human_out = [
            Port("question", PortLevel.HUMAN, PortDirection.OUTPUT,
                 "string", "Question to human"),
        ]
        ports.human_in = [
            Port("answer", PortLevel.HUMAN, PortDirection.INPUT,
                 "string", "Answer from human"),
        ]
        return ports


# === Port Connection ===

@dataclass
class PortConnection:
    """Соединение между портами."""
    
    from_component: str
    from_port: str
    from_level: PortLevel
    
    to_component: str
    to_port: str
    to_level: PortLevel
    
    condition: Optional[str] = None     # Условие для conditional routing
    transform: Optional[str] = None     # Трансформация данных
    
    def to_dict(self) -> dict:
        return {
            "from": {
                "component": self.from_component,
                "port": self.from_port,
                "level": self.from_level.value,
            },
            "to": {
                "component": self.to_component,
                "port": self.to_port,
                "level": self.to_level.value,
            },
            "condition": self.condition,
            "transform": self.transform,
        }


# === Component Base ===

class PortedComponent:
    """Базовый класс для компонентов с портами."""
    
    def __init__(self, ports: PortSet = None):
        self.ports = ports or PortSet()
        self._port_data: Dict[str, Any] = {}
    
    # === Workflow ports ===
    
    async def receive_workflow(self, port: str, data: Any) -> None:
        """Получить данные из workflow."""
        self._port_data[f"workflow_in:{port}"] = data
    
    async def send_workflow(self, port: str, data: Any) -> None:
        """Отправить данные в workflow."""
        self._port_data[f"workflow_out:{port}"] = data
        # Trigger output event
    
    # === Agent ports ===
    
    async def receive_agent(self, port: str, data: Any, from_agent: str) -> None:
        """Получить данные от другого агента."""
        key = f"agent_in:{port}:{from_agent}"
        self._port_data[key] = data
    
    async def send_agent(self, port: str, data: Any, to_agent: str) -> None:
        """Отправить данные другому агенту."""
        key = f"agent_out:{port}:{to_agent}"
        self._port_data[key] = data
        # Trigger routing
    
    # === Human ports ===
    
    async def send_human(self, port: str, data: Any) -> str:
        """Отправить сообщение человеку. Возвращает interaction_id."""
        # Create human interaction request
        pass
    
    async def receive_human(self, interaction_id: str) -> Any:
        """Получить ответ от человека."""
        # Wait for human response
        pass
```

### Кто имеет какие порты

| Компонент | workflow | agent | human |
|-----------|:--------:|:-----:|:-----:|
| **Agent** | ✅ | ✅ | ⚠️ опционально |
| **TeamOrchestrator** | ✅ | ✅ | ✅ |
| **LLMTeam** | ✅ | ✅ | ✅ (через Orch) |
| **GroupOrchestrator** | ✅ | ✅ | ✅ |
| **Group** | ✅ | ❌ | ✅ (через Orch) |

### JSON формат для Canvas

```json
{
  "component_id": "content_team",
  "type": "llmteam",
  "ports": {
    "workflow_in": [
      {"name": "task", "data_type": "object", "required": true}
    ],
    "workflow_out": [
      {"name": "result", "data_type": "object"},
      {"name": "error", "data_type": "object", "required": false}
    ],
    "agent_in": [
      {"name": "data", "data_type": "any"}
    ],
    "agent_out": [
      {"name": "result", "data_type": "any"}
    ],
    "human_out": [
      {"name": "approval_request", "data_type": "object"}
    ],
    "human_in": [
      {"name": "approval_response", "data_type": "object"}
    ]
  }
}
```

### Визуализация в Canvas UI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LLMTeam: Content Team                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WORKFLOW                                                                  │
│   ═══════════════════════════════════════════════════════════════════════  │
│   [workflow_in:task] ─────────────────────────▶ [workflow_out:result]      │
│                                                                             │
│   AGENTS                                                                    │
│   ═══════════════════════════════════════════════════════════════════════  │
│   [agent_in:data] ────┬────▶ Agent A ────┬────▶ [agent_out:result]         │
│                       │                  │                                  │
│                       ├────▶ Agent B ────┤                                  │
│                       │                  │                                  │
│                       └────▶ Agent C ────┘                                  │
│                                                                             │
│   HUMAN                                                                     │
│   ═══════════════════════════════════════════════════════════════════════  │
│   [human_out:request] ────▶ 💬 Chat ────▶ [human_in:response]              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📑 RFC #8: Pipeline → LLMTeam Rename

### Назначение

Переименование основного класса `Pipeline` в `LLMTeam` для соответствия названию библиотеки и бизнес-терминологии.

### Изменения именования

| Было (v1.9.0) | Стало (v2.0.0) | Описание |
|---------------|----------------|----------|
| `Pipeline` | `LLMTeam` | Основной класс команды агентов |
| `PipelineOrchestrator` | `TeamOrchestrator` | Оркестратор команды |
| `PipelineSnapshot` | `TeamSnapshot` | Снимок состояния команды |
| `PipelineExecutor` | `TeamExecutor` | Исполнитель команды |
| `PipelineConfig` | `TeamConfig` | Конфигурация команды |
| `PipelineResult` | `TeamResult` | Результат выполнения |
| `PipelineEvent` | `TeamEvent` | События команды |

### Бизнес-терминология

```
═══════════════════════════════════════════════════════════════════════════════

    ОРГАНИЗАЦИОННАЯ СТРУКТУРА              LLMTEAM ИЕРАРХИЯ
    ════════════════════════              ═══════════════════

         Компания                              System
            │                                     │
            ▼                                     ▼
         Отдел ─────────────────────────▶      Group
            │                                     │
            ▼                                     ▼
         Команда ───────────────────────▶    LLMTeam
            │                                     │
            ▼                                     ▼
        Сотрудник ──────────────────────▶     Agent

═══════════════════════════════════════════════════════════════════════════════

    Group          = Отдел / Департамент (несколько команд)
    LLMTeam        = Команда (несколько агентов)
    Agent          = Сотрудник (один исполнитель)
    
    GroupOrchestrator = Руководитель отдела
    TeamOrchestrator  = Тимлид / Руководитель команды

═══════════════════════════════════════════════════════════════════════════════
```

### Новые классы

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class TeamConfig:
    """Конфигурация LLMTeam."""
    
    name: str
    description: str = ""
    
    # Agents
    max_agents: int = 50
    allow_dynamic_agents: bool = False
    
    # Execution
    max_iterations: int = 100
    timeout_seconds: float = 300
    parallel_execution: bool = False
    
    # Human interaction
    enable_human_interaction: bool = True
    default_approval_timeout_hours: float = 24
    
    # Persistence
    enable_snapshots: bool = True
    snapshot_interval: int = 10  # Every N steps
    
    # Ports
    ports: PortSet = field(default_factory=PortSet)


@dataclass  
class TeamResult:
    """Результат выполнения LLMTeam."""
    
    team_id: str
    run_id: str
    status: str  # "completed", "failed", "cancelled", "timeout"
    
    # Output
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    
    # Stats
    agents_invoked: int = 0
    human_interactions: int = 0
    iterations: int = 0
    
    # Error
    error: Optional[Dict[str, Any]] = None
    
    # History
    events: List["TeamEvent"] = field(default_factory=list)


class LLMTeam(PortedComponent):
    """
    Команда AI агентов.
    
    Основной класс библиотеки LLMTeam.
    Объединяет несколько агентов под управлением оркестратора.
    """
    
    def __init__(
        self,
        name: str,
        agents: List["Agent"],
        orchestrator: "TeamOrchestrator" = None,
        config: TeamConfig = None,
    ):
        self.team_id = self._generate_id()
        self.name = name
        self.agents = {a.agent_id: a for a in agents}
        self.orchestrator = orchestrator or TeamOrchestrator()
        self.config = config or TeamConfig(name=name)
        
        # Initialize ports
        super().__init__(PortPresets.orchestrator())
        
        # State
        self._runs: Dict[str, TeamResult] = {}
    
    def _generate_id(self) -> str:
        import uuid
        return f"team_{uuid.uuid4().hex[:8]}"
    
    # === Agent Management ===
    
    def add_agent(self, agent: "Agent") -> None:
        """Добавить агента в команду."""
        self.agents[agent.agent_id] = agent
    
    def remove_agent(self, agent_id: str) -> None:
        """Удалить агента из команды."""
        del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional["Agent"]:
        """Получить агента по ID."""
        return self.agents.get(agent_id)
    
    # === Execution ===
    
    async def run(
        self,
        ctx: "RuntimeContext",
        input_data: Dict[str, Any],
    ) -> TeamResult:
        """
        Запустить команду.
        
        Args:
            ctx: Runtime context
            input_data: Входные данные (через workflow_in)
            
        Returns:
            TeamResult с результатом выполнения
        """
        run_id = ctx.run_id
        
        # Initialize result
        result = TeamResult(
            team_id=self.team_id,
            run_id=run_id,
            status="running",
            started_at=datetime.now(),
        )
        self._runs[run_id] = result
        
        try:
            # Receive workflow input
            await self.receive_workflow("input", input_data)
            
            # Run orchestrator
            output = await self.orchestrator.orchestrate(
                ctx=ctx,
                team=self,
                input_data=input_data,
            )
            
            # Send workflow output
            await self.send_workflow("output", output)
            
            # Finalize result
            result.status = "completed"
            result.output = output
            result.completed_at = datetime.now()
            result.duration_ms = int(
                (result.completed_at - result.started_at).total_seconds() * 1000
            )
            
        except Exception as e:
            result.status = "failed"
            result.error = {
                "type": type(e).__name__,
                "message": str(e),
            }
            result.completed_at = datetime.now()
            raise
        
        return result
    
    # === Human Interaction ===
    
    async def request_human_approval(
        self,
        ctx: "RuntimeContext",
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Запросить одобрение у человека."""
        interaction_id = await self.send_human("approval_request", request)
        return await self.receive_human(interaction_id)
    
    async def send_human_message(
        self,
        ctx: "RuntimeContext",
        message: str,
    ) -> Optional[str]:
        """Отправить сообщение человеку в чат."""
        interaction_id = await self.send_human("message", {"text": message})
        response = await self.receive_human(interaction_id)
        return response.get("text") if response else None
    
    # === Serialization ===
    
    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            "team_id": self.team_id,
            "name": self.name,
            "agents": [a.to_dict() for a in self.agents.values()],
            "config": {
                "max_agents": self.config.max_agents,
                "timeout_seconds": self.config.timeout_seconds,
                "parallel_execution": self.config.parallel_execution,
            },
            "ports": self.ports.to_dict(),
        }


class TeamOrchestrator(PortedComponent):
    """
    Оркестратор команды.
    
    Управляет выполнением агентов внутри LLMTeam.
    """
    
    def __init__(
        self,
        strategy: "OrchestrationStrategy" = None,
        enable_human_escalation: bool = True,
    ):
        super().__init__(PortPresets.orchestrator())
        self.strategy = strategy or RuleBasedStrategy()
        self.enable_human_escalation = enable_human_escalation
    
    async def orchestrate(
        self,
        ctx: "RuntimeContext",
        team: LLMTeam,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Оркестрировать выполнение команды.
        """
        current_data = input_data
        iteration = 0
        
        while iteration < team.config.max_iterations:
            iteration += 1
            
            # Decide next action
            decision = await self.strategy.decide(
                ctx=ctx,
                team=team,
                current_data=current_data,
                iteration=iteration,
            )
            
            if decision.action == "complete":
                return decision.output
            
            elif decision.action == "invoke_agent":
                # Send to agent via agent_out
                await self.send_agent("command", decision.agent_input, decision.agent_id)
                
                # Get agent result via agent_in
                agent = team.get_agent(decision.agent_id)
                result = await agent.run(ctx, decision.agent_input)
                await self.receive_agent("result", result, decision.agent_id)
                
                current_data = result
            
            elif decision.action == "request_human":
                # Send to human via human_out
                await self.send_human("request", decision.human_request)
                
                # Wait for response via human_in
                response = await self.receive_human(decision.interaction_id)
                current_data = response
            
            elif decision.action == "escalate":
                # Escalate to human
                if self.enable_human_escalation:
                    await self.send_human("escalation", {
                        "reason": decision.escalation_reason,
                        "context": current_data,
                    })
                    response = await self.receive_human(decision.interaction_id)
                    current_data = response
                else:
                    raise EscalationError(decision.escalation_reason)
        
        raise MaxIterationsError(f"Max iterations ({team.config.max_iterations}) reached")
```

### Обратная совместимость

```python
# === Deprecated aliases ===

import warnings
from functools import wraps


def deprecated_alias(new_class, old_name: str):
    """Create deprecated alias for a class."""
    
    class DeprecatedClass(new_class):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"{old_name} is deprecated, use {new_class.__name__} instead. "
                f"Will be removed in v3.0.0",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(*args, **kwargs)
    
    DeprecatedClass.__name__ = old_name
    DeprecatedClass.__qualname__ = old_name
    return DeprecatedClass


# Aliases for backward compatibility
Pipeline = deprecated_alias(LLMTeam, "Pipeline")
PipelineOrchestrator = deprecated_alias(TeamOrchestrator, "PipelineOrchestrator")
PipelineConfig = deprecated_alias(TeamConfig, "PipelineConfig")
PipelineResult = deprecated_alias(TeamResult, "PipelineResult")
```

### Миграция кода

```python
# === Было (v1.9.0) ===

from llmteam import Pipeline, PipelineOrchestrator

pipeline = Pipeline(
    agents=[agent_a, agent_b],
    orchestrator=PipelineOrchestrator(),
)
result = await pipeline.run(input_data)


# === Стало (v2.0.0) ===

from llmteam import LLMTeam, TeamOrchestrator

team = LLMTeam(
    name="My Team",
    agents=[agent_a, agent_b],
    orchestrator=TeamOrchestrator(),
)
result = await team.run(ctx, input_data)
```

### Гайд по миграции

```markdown
# Migration Guide: v1.9.0 → v2.0.0

## Class Renames

| v1.9.0 | v2.0.0 | Action |
|--------|--------|--------|
| `Pipeline` | `LLMTeam` | Find & Replace |
| `PipelineOrchestrator` | `TeamOrchestrator` | Find & Replace |
| `PipelineSnapshot` | `TeamSnapshot` | Find & Replace |
| `PipelineConfig` | `TeamConfig` | Find & Replace |
| `PipelineResult` | `TeamResult` | Find & Replace |

## API Changes

### Constructor

```python
# v1.9.0
Pipeline(agents=[...], orchestrator=...)

# v2.0.0
LLMTeam(name="...", agents=[...], orchestrator=...)
```

### Run method

```python
# v1.9.0
result = await pipeline.run(input_data)

# v2.0.0
result = await team.run(ctx, input_data)
```

## Automatic Migration Script

```bash
# Run migration script
python -m llmteam.migrate v1_to_v2 ./src/
```
```

### JSON формат для Canvas

```json
{
  "component_id": "content_team",
  "type": "llmteam",
  "name": "Content Creation Team",
  "config": {
    "max_agents": 10,
    "timeout_seconds": 300,
    "parallel_execution": false,
    "enable_human_interaction": true
  },
  "agents": [
    {"agent_id": "analyst", "type": "llm_agent"},
    {"agent_id": "writer", "type": "llm_agent"},
    {"agent_id": "reviewer", "type": "llm_agent"}
  ],
  "orchestrator": {
    "type": "team_orchestrator",
    "strategy": "rule_based"
  },
  "ports": {
    "workflow_in": [{"name": "task", "data_type": "object"}],
    "workflow_out": [{"name": "result", "data_type": "object"}],
    "agent_in": [{"name": "data", "data_type": "any"}],
    "agent_out": [{"name": "result", "data_type": "any"}],
    "human_out": [{"name": "request", "data_type": "object"}],
    "human_in": [{"name": "response", "data_type": "object"}]
  }
}
```

---

## 📑 RFC #6: Critic Loop Pattern (Рефлексивный цикл)

### Назначение

Встроенный паттерн для итеративного улучшения результата через взаимодействие агентов Generator (генератор) и Critic (критик).

### Паттерн

```
┌─────────────────────────────────────────────────────────┐
│                    CRITIC LOOP                          │
│                                                         │
│    ┌───────────┐         ┌───────────┐                 │
│    │ GENERATOR │────────▶│  CRITIC   │                 │
│    │  (Agent)  │◀────────│  (Agent)  │                 │
│    └───────────┘ feedback└───────────┘                 │
│         │                      │                        │
│         │ approved             │ max_iterations         │
│         ▼                      ▼                        │
│    ┌─────────────────────────────┐                     │
│    │         OUTPUT              │                     │
│    └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### Варианты завершения цикла

| Условие | Описание |
|---------|----------|
| `approved` | Критик одобрил результат |
| `max_iterations` | Достигнут лимит итераций |
| `quality_threshold` | Достигнут порог качества (score ≥ threshold) |
| `no_improvement` | Нет улучшения N итераций подряд |
| `timeout` | Превышено время выполнения |

### Модель данных

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Literal
from enum import Enum
from datetime import timedelta


class LoopExitCondition(Enum):
    """Условия выхода из цикла."""
    APPROVED = "approved"              # Критик одобрил
    MAX_ITERATIONS = "max_iterations"  # Лимит итераций
    QUALITY_MET = "quality_met"        # Достигнут порог качества
    NO_IMPROVEMENT = "no_improvement"  # Нет улучшения
    TIMEOUT = "timeout"                # Таймаут
    MANUAL_STOP = "manual_stop"        # Ручная остановка


@dataclass
class CriticFeedback:
    """Обратная связь от критика."""
    
    approved: bool                     # Одобрено или нет
    score: float                       # Оценка 0.0 - 1.0
    feedback: str                      # Текстовая обратная связь
    suggestions: List[str] = field(default_factory=list)  # Конкретные предложения
    aspects: Dict[str, float] = field(default_factory=dict)  # Оценки по аспектам
    
    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "score": self.score,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "aspects": self.aspects,
        }


@dataclass
class LoopIteration:
    """Информация об одной итерации цикла."""
    
    iteration: int
    generator_output: Any
    critic_feedback: CriticFeedback
    duration_ms: int
    
    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "generator_output": self.generator_output,
            "critic_feedback": self.critic_feedback.to_dict(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class CriticLoopResult:
    """Результат выполнения Critic Loop."""
    
    final_output: Any                  # Финальный результат
    exit_condition: LoopExitCondition  # Почему завершился
    iterations_count: int              # Сколько итераций
    total_duration_ms: int             # Общее время
    final_score: float                 # Финальная оценка
    iterations: List[LoopIteration] = field(default_factory=list)  # История
    
    def to_dict(self) -> dict:
        return {
            "final_output": self.final_output,
            "exit_condition": self.exit_condition.value,
            "iterations_count": self.iterations_count,
            "total_duration_ms": self.total_duration_ms,
            "final_score": self.final_score,
            "iterations": [i.to_dict() for i in self.iterations],
        }


@dataclass
class CriticLoopConfig:
    """Конфигурация Critic Loop."""
    
    # === Лимиты ===
    max_iterations: int = 5            # Максимум итераций
    timeout: Optional[timedelta] = None  # Общий таймаут
    
    # === Условия выхода ===
    quality_threshold: float = 0.8     # Порог качества для выхода
    no_improvement_limit: int = 2      # Выход если N итераций без улучшения
    min_improvement: float = 0.05      # Минимальное улучшение для продолжения
    
    # === Поведение ===
    include_history: bool = True       # Передавать историю генератору
    include_all_feedback: bool = False # Передавать всю историю feedback
    
    # === Callbacks ===
    on_iteration: Optional[Callable[[LoopIteration], None]] = None
    on_improvement: Optional[Callable[[float, float], None]] = None  # (old_score, new_score)


class CriticLoop:
    """
    Рефлексивный цикл Generator-Critic.
    
    Позволяет итеративно улучшать результат через
    взаимодействие двух агентов.
    """
    
    def __init__(
        self,
        generator: "Agent",            # Агент-генератор
        critic: "Agent",               # Агент-критик
        config: CriticLoopConfig = None,
    ):
        self.generator = generator
        self.critic = critic
        self.config = config or CriticLoopConfig()
    
    async def run(
        self,
        ctx: "StepContext",
        initial_input: Dict[str, Any],
    ) -> CriticLoopResult:
        """
        Запустить цикл улучшения.
        
        Args:
            ctx: Step context
            initial_input: Начальные данные для генератора
            
        Returns:
            CriticLoopResult с финальным результатом и историей
        """
        import time
        start_time = time.time()
        
        iterations: List[LoopIteration] = []
        current_input = initial_input
        best_output = None
        best_score = 0.0
        no_improvement_count = 0
        
        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            
            # === 1. Generator создаёт/улучшает результат ===
            generator_input = self._prepare_generator_input(
                current_input,
                iterations,
            )
            generator_output = await self.generator.run(ctx, generator_input)
            
            # === 2. Critic оценивает результат ===
            critic_input = self._prepare_critic_input(
                initial_input,
                generator_output,
                iterations,
            )
            critic_response = await self.critic.run(ctx, critic_input)
            feedback = self._parse_critic_feedback(critic_response)
            
            # === 3. Записываем итерацию ===
            iter_duration = int((time.time() - iter_start) * 1000)
            loop_iteration = LoopIteration(
                iteration=iteration,
                generator_output=generator_output,
                critic_feedback=feedback,
                duration_ms=iter_duration,
            )
            iterations.append(loop_iteration)
            
            # Callback
            if self.config.on_iteration:
                self.config.on_iteration(loop_iteration)
            
            # === 4. Проверяем улучшение ===
            if feedback.score > best_score:
                improvement = feedback.score - best_score
                
                if self.config.on_improvement:
                    self.config.on_improvement(best_score, feedback.score)
                
                if improvement >= self.config.min_improvement:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                best_score = feedback.score
                best_output = generator_output
            else:
                no_improvement_count += 1
            
            # === 5. Проверяем условия выхода ===
            
            # Критик одобрил
            if feedback.approved:
                return self._make_result(
                    generator_output,
                    LoopExitCondition.APPROVED,
                    iterations,
                    start_time,
                    feedback.score,
                )
            
            # Достигнут порог качества
            if feedback.score >= self.config.quality_threshold:
                return self._make_result(
                    generator_output,
                    LoopExitCondition.QUALITY_MET,
                    iterations,
                    start_time,
                    feedback.score,
                )
            
            # Нет улучшения
            if no_improvement_count >= self.config.no_improvement_limit:
                return self._make_result(
                    best_output,
                    LoopExitCondition.NO_IMPROVEMENT,
                    iterations,
                    start_time,
                    best_score,
                )
            
            # Таймаут
            if self.config.timeout:
                elapsed = time.time() - start_time
                if elapsed >= self.config.timeout.total_seconds():
                    return self._make_result(
                        best_output,
                        LoopExitCondition.TIMEOUT,
                        iterations,
                        start_time,
                        best_score,
                    )
            
            # === 6. Подготовка к следующей итерации ===
            current_input = {
                **initial_input,
                "previous_output": generator_output,
                "feedback": feedback.feedback,
                "suggestions": feedback.suggestions,
                "score": feedback.score,
            }
        
        # Достигнут лимит итераций
        return self._make_result(
            best_output,
            LoopExitCondition.MAX_ITERATIONS,
            iterations,
            start_time,
            best_score,
        )
    
    def _prepare_generator_input(
        self,
        current_input: dict,
        iterations: List[LoopIteration],
    ) -> dict:
        """Подготовить input для генератора."""
        result = {**current_input}
        
        if self.config.include_history and iterations:
            if self.config.include_all_feedback:
                result["history"] = [
                    {
                        "iteration": i.iteration,
                        "feedback": i.critic_feedback.feedback,
                        "score": i.critic_feedback.score,
                    }
                    for i in iterations
                ]
            else:
                # Только последний feedback
                last = iterations[-1]
                result["last_feedback"] = last.critic_feedback.feedback
                result["last_score"] = last.critic_feedback.score
        
        return result
    
    def _prepare_critic_input(
        self,
        original_input: dict,
        generator_output: Any,
        iterations: List[LoopIteration],
    ) -> dict:
        """Подготовить input для критика."""
        return {
            "original_request": original_input,
            "generated_output": generator_output,
            "iteration": len(iterations) + 1,
            "previous_scores": [i.critic_feedback.score for i in iterations],
        }
    
    def _parse_critic_feedback(self, critic_response: Any) -> CriticFeedback:
        """Парсить ответ критика в CriticFeedback."""
        # Если критик вернул структурированный ответ
        if isinstance(critic_response, dict):
            return CriticFeedback(
                approved=critic_response.get("approved", False),
                score=critic_response.get("score", 0.5),
                feedback=critic_response.get("feedback", ""),
                suggestions=critic_response.get("suggestions", []),
                aspects=critic_response.get("aspects", {}),
            )
        
        # Если строка — пытаемся распарсить
        # TODO: Более умный парсинг
        return CriticFeedback(
            approved=False,
            score=0.5,
            feedback=str(critic_response),
            suggestions=[],
        )
    
    def _make_result(
        self,
        output: Any,
        condition: LoopExitCondition,
        iterations: List[LoopIteration],
        start_time: float,
        final_score: float,
    ) -> CriticLoopResult:
        """Создать результат."""
        import time
        return CriticLoopResult(
            final_output=output,
            exit_condition=condition,
            iterations_count=len(iterations),
            total_duration_ms=int((time.time() - start_time) * 1000),
            final_score=final_score,
            iterations=iterations,
        )


# ===== Step Type для Catalog =====

CRITIC_LOOP_STEP_TYPE = StepTypeMetadata(
    type_id="critic_loop",
    version="1.0",
    display_name="Critic Loop",
    description="Iteratively improve output through Generator-Critic interaction",
    category=StepCategory.AI,
    icon="refresh-cw",
    color="#8B5CF6",
    config_schema={
        "type": "object",
        "properties": {
            "generator_ref": {
                "type": "string",
                "description": "Reference to generator agent",
            },
            "critic_ref": {
                "type": "string",
                "description": "Reference to critic agent",
            },
            "max_iterations": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
            "quality_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.8,
            },
            "no_improvement_limit": {
                "type": "integer",
                "minimum": 1,
                "default": 2,
            },
            "timeout_seconds": {
                "type": "number",
                "default": 300,
            },
        },
        "required": ["generator_ref", "critic_ref"],
    },
    input_ports=[
        PortSpec("input", "object", "Initial input for generator"),
    ],
    output_ports=[
        PortSpec("output", "any", "Final improved output"),
        PortSpec("score", "number", "Final quality score"),
        PortSpec("iterations", "integer", "Number of iterations"),
        PortSpec("history", "array", "Iteration history"),
    ],
)
```

### JSON формат для Canvas

```json
{
  "step_id": "improve_content",
  "type": "critic_loop",
  "name": "Iterative Content Improvement",
  "config": {
    "generator_ref": "content_writer",
    "critic_ref": "content_reviewer",
    "max_iterations": 5,
    "quality_threshold": 0.85,
    "no_improvement_limit": 2,
    "timeout_seconds": 300
  },
  "ports": {
    "input": ["input"],
    "output": ["output", "score", "iterations", "history"]
  }
}
```

### Пример использования

```python
from llmteam.patterns import CriticLoop, CriticLoopConfig
from llmteam.agents import LLMAgent

# === Создаём агентов ===

generator = LLMAgent(
    name="ContentWriter",
    system_prompt="""You are a content writer. 
    Write or improve content based on the request.
    If feedback is provided, address all suggestions.""",
    llm_ref="gpt4",
)

critic = LLMAgent(
    name="ContentReviewer", 
    system_prompt="""You are a critical content reviewer.
    Evaluate the content and provide:
    - approved: true/false
    - score: 0.0 to 1.0
    - feedback: detailed feedback
    - suggestions: list of specific improvements
    
    Be constructive but demanding. Only approve if score >= 0.85.""",
    llm_ref="gpt4",
    output_format="json",
)

# === Настраиваем цикл ===

loop = CriticLoop(
    generator=generator,
    critic=critic,
    config=CriticLoopConfig(
        max_iterations=5,
        quality_threshold=0.85,
        no_improvement_limit=2,
        timeout=timedelta(minutes=5),
        on_iteration=lambda i: print(f"Iteration {i.iteration}: score={i.critic_feedback.score}"),
    ),
)

# === Запускаем ===

result = await loop.run(ctx, {
    "task": "Write a blog post about AI agents",
    "style": "professional but engaging",
    "length": "500-700 words",
})

print(f"Exit: {result.exit_condition.value}")
print(f"Iterations: {result.iterations_count}")
print(f"Final score: {result.final_score}")
print(f"Output: {result.final_output}")
```

### Расширенные паттерны

```python
# === Multi-Critic (несколько критиков) ===

class MultiCriticLoop(CriticLoop):
    """Цикл с несколькими критиками."""
    
    def __init__(
        self,
        generator: Agent,
        critics: List[Agent],           # Несколько критиков
        aggregation: Literal["average", "min", "max", "unanimous"] = "average",
        **kwargs,
    ):
        self.critics = critics
        self.aggregation = aggregation
        super().__init__(generator, critics[0], **kwargs)
    
    async def _get_aggregated_feedback(self, ctx, critic_input) -> CriticFeedback:
        # Получаем feedback от всех критиков
        feedbacks = []
        for critic in self.critics:
            response = await critic.run(ctx, critic_input)
            feedbacks.append(self._parse_critic_feedback(response))
        
        # Агрегируем
        if self.aggregation == "average":
            score = sum(f.score for f in feedbacks) / len(feedbacks)
            approved = all(f.approved for f in feedbacks)
        elif self.aggregation == "min":
            score = min(f.score for f in feedbacks)
            approved = all(f.approved for f in feedbacks)
        # ...
        
        return CriticFeedback(
            approved=approved,
            score=score,
            feedback="\n\n".join(f"[{self.critics[i].name}]: {f.feedback}" 
                                  for i, f in enumerate(feedbacks)),
            suggestions=[s for f in feedbacks for s in f.suggestions],
        )


# === Self-Critic (агент критикует сам себя) ===

class SelfCriticLoop:
    """Агент критикует собственный результат."""
    
    def __init__(
        self,
        agent: Agent,
        critic_prompt: str = "Now critically review your own output...",
        **kwargs,
    ):
        self.agent = agent
        self.critic_prompt = critic_prompt


# === Tournament (соревнование вариантов) ===

class TournamentLoop:
    """Генерирует несколько вариантов, критик выбирает лучший."""
    
    def __init__(
        self,
        generators: List[Agent],        # Несколько генераторов
        judge: Agent,                   # Судья выбирает лучший
        rounds: int = 3,
    ):
        self.generators = generators
        self.judge = judge
        self.rounds = rounds
```

### События для UI

```python
# Специальные события для Critic Loop
class CriticLoopEventType(Enum):
    LOOP_STARTED = "critic_loop.started"
    ITERATION_STARTED = "critic_loop.iteration_started"
    GENERATOR_COMPLETED = "critic_loop.generator_completed"
    CRITIC_COMPLETED = "critic_loop.critic_completed"
    ITERATION_COMPLETED = "critic_loop.iteration_completed"
    IMPROVEMENT_DETECTED = "critic_loop.improvement"
    NO_IMPROVEMENT = "critic_loop.no_improvement"
    LOOP_COMPLETED = "critic_loop.completed"

# Canvas может показывать:
# - Текущую итерацию
# - График улучшения score
# - Feedback от критика
# - Причину завершения
```

---

## 📊 Обновлённый Step Catalog

| Step Type | Категория | Описание |
|-----------|-----------|----------|
| `llm_agent` | AI | Базовый LLM агент |
| `critic_loop` | AI | **NEW** Рефлексивный цикл Generator-Critic |
| `multi_critic_loop` | AI | **NEW** Цикл с несколькими критиками |
| `http_action` | Integration | HTTP запрос |
| `human_task` | Human | Задача для человека |
| `condition` | Control | Условное ветвление |
| `parallel_split` | Control | Параллельное выполнение |
| `parallel_join` | Control | Ожидание параллельных веток |
| `transform` | Data | Трансформация данных |

---

**Версия: 2.0.0**
**Кодовое имя: Canvas Integration**
**Зависимость: v1.9.0**
