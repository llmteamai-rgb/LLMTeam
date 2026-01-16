# ⚙️ Пакет изменений v1.9.0: Workflow Runtime

## 🎯 Цель релиза

Полноценный Enterprise Workflow Runtime:
- Интеграция с внешними системами
- Human-in-the-loop для критичных решений
- Персистентность для долгих workflow

**Зависимость:** Требует v1.7.0 + v1.8.0

---

## 📋 Состав пакета

| # | RFC | Файл | Effort |
|---|-----|------|--------|
| 1 | External Actions | `rfc-external-actions.md` | 2 нед |
| 2 | Human Interaction | `rfc-human-interaction-approval.md` | 2.5 нед |
| 3 | Persistence | `rfc-pipeline-persistence.md` | 2 нед |

**Общий effort: 6.5 недель**

---

## 🔗 Зависимости

```
v1.7.0 (Security)     v1.8.0 (Orchestration)
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  External Actions   │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Human Interaction  │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    Persistence      │
         └─────────────────────┘
```

---

## 📑 RFC #1: External Actions

### Назначение

Агенты вызывают внешние системы (API, webhooks, databases).

### Модель данных

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import asyncio
import aiohttp


class ActionType(Enum):
    """Тип внешнего действия."""
    WEBHOOK = "webhook"
    REST_API = "rest_api"
    GRPC = "grpc"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FUNCTION = "function"


class ActionStatus(Enum):
    """Статус выполнения."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ActionConfig:
    """Конфигурация действия."""
    
    name: str
    action_type: ActionType
    
    # Connection
    url: Optional[str] = None
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Timeouts
    timeout_seconds: float = 30.0
    
    # Retry (использует RateLimiter из v1.7.0)
    retry_count: int = 3
    
    # Validation
    request_schema: Optional[dict] = None
    response_schema: Optional[dict] = None
    
    # Security
    auth_type: str = ""  # "bearer", "basic", "api_key"
    auth_config: Dict[str, str] = field(default_factory=dict)


@dataclass
class ActionContext:
    """Контекст выполнения действия."""
    
    action_name: str
    run_id: str
    agent_name: str
    tenant_id: str  # Из v1.7.0 TenantContext
    
    # Input
    input_data: Dict[str, Any] = field(default_factory=dict)
    
    # State
    pipeline_state: Dict[str, Any] = field(default_factory=dict)
    agent_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = ""


@dataclass
class ActionResult:
    """Результат выполнения."""
    
    action_name: str
    status: ActionStatus
    
    # Response
    response_data: Any = None
    response_code: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_ms: int = 0
    
    # Errors
    error_message: str = ""
    error_type: str = ""
    retry_count: int = 0
    
    # Audit (для v1.7.0 AuditTrail)
    audit_metadata: Dict[str, Any] = field(default_factory=dict)


class ActionHandler:
    """Базовый handler для действий."""
    
    async def execute(self, context: ActionContext) -> ActionResult:
        raise NotImplementedError
    
    async def validate_request(self, context: ActionContext) -> bool:
        return True
    
    async def validate_response(self, result: ActionResult) -> bool:
        return True


class WebhookActionHandler(ActionHandler):
    """Handler для webhook вызовов."""
    
    def __init__(self, config: ActionConfig):
        self.config = config
    
    async def execute(self, context: ActionContext) -> ActionResult:
        started_at = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = self._prepare_headers(context)
                
                async with session.request(
                    method=self.config.method,
                    url=self.config.url,
                    json=context.input_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    response_data = await response.json()
                    
                    return ActionResult(
                        action_name=self.config.name,
                        status=ActionStatus.COMPLETED if response.ok else ActionStatus.FAILED,
                        response_data=response_data,
                        response_code=response.status,
                        started_at=started_at,
                        completed_at=datetime.now(),
                        duration_ms=int((datetime.now() - started_at).total_seconds() * 1000),
                    )
                    
        except asyncio.TimeoutError:
            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.TIMEOUT,
                error_message="Request timeout",
                started_at=started_at,
                completed_at=datetime.now(),
            )
            
        except Exception as e:
            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.FAILED,
                error_message=str(e),
                error_type=type(e).__name__,
                started_at=started_at,
                completed_at=datetime.now(),
            )
    
    def _prepare_headers(self, context: ActionContext) -> Dict[str, str]:
        headers = self.config.headers.copy()
        headers["Content-Type"] = "application/json"
        headers["X-Correlation-ID"] = context.correlation_id
        headers["X-Tenant-ID"] = context.tenant_id
        
        # Auth
        if self.config.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.config.auth_config.get('token', '')}"
        elif self.config.auth_type == "api_key":
            key_header = self.config.auth_config.get("header", "X-API-Key")
            headers[key_header] = self.config.auth_config.get("key", "")
        
        return headers


class FunctionActionHandler(ActionHandler):
    """Handler для вызова Python функций."""
    
    def __init__(self, config: ActionConfig, func: Callable):
        self.config = config
        self.func = func
    
    async def execute(self, context: ActionContext) -> ActionResult:
        started_at = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(context.input_data, context.pipeline_state)
            else:
                result = self.func(context.input_data, context.pipeline_state)
            
            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.COMPLETED,
                response_data=result,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_ms=int((datetime.now() - started_at).total_seconds() * 1000),
            )
            
        except Exception as e:
            return ActionResult(
                action_name=self.config.name,
                status=ActionStatus.FAILED,
                error_message=str(e),
                error_type=type(e).__name__,
                started_at=started_at,
                completed_at=datetime.now(),
            )


class ActionRegistry:
    """Реестр доступных действий."""
    
    def __init__(self):
        self._handlers: Dict[str, ActionHandler] = {}
        self._configs: Dict[str, ActionConfig] = {}
    
    def register(self, config: ActionConfig, handler: ActionHandler) -> None:
        self._configs[config.name] = config
        self._handlers[config.name] = handler
    
    def register_webhook(
        self,
        name: str,
        url: str,
        method: str = "POST",
        **kwargs,
    ) -> None:
        config = ActionConfig(
            name=name,
            action_type=ActionType.WEBHOOK,
            url=url,
            method=method,
            **kwargs,
        )
        self.register(config, WebhookActionHandler(config))
    
    def register_function(
        self,
        name: str,
        func: Callable,
        **kwargs,
    ) -> None:
        config = ActionConfig(
            name=name,
            action_type=ActionType.FUNCTION,
            **kwargs,
        )
        self.register(config, FunctionActionHandler(config, func))
    
    def get_handler(self, name: str) -> Optional[ActionHandler]:
        return self._handlers.get(name)
    
    def list_actions(self) -> List[str]:
        return list(self._handlers.keys())


class ActionExecutor:
    """
    Executor для внешних действий.
    
    Интегрируется с:
    - RateLimiter из v1.7.0
    - AuditTrail из v1.7.0
    - TenantContext из v1.7.0
    """
    
    def __init__(
        self,
        registry: ActionRegistry,
        rate_limiter: "RateLimitedExecutor" = None,  # v1.7.0
        audit_trail: "AuditTrail" = None,            # v1.7.0
    ):
        self.registry = registry
        self.rate_limiter = rate_limiter
        self.audit_trail = audit_trail
    
    async def execute(
        self,
        action_name: str,
        context: ActionContext,
    ) -> ActionResult:
        """Выполнить действие."""
        handler = self.registry.get_handler(action_name)
        if not handler:
            return ActionResult(
                action_name=action_name,
                status=ActionStatus.FAILED,
                error_message=f"Action '{action_name}' not found",
            )
        
        # Rate limiting
        if self.rate_limiter:
            result = await self.rate_limiter.execute(
                action_name,
                handler.execute,
                context,
            )
        else:
            result = await handler.execute(context)
        
        # Audit
        if self.audit_trail:
            await self.audit_trail.log(
                AuditEventType.ACTION_COMPLETED if result.status == ActionStatus.COMPLETED 
                else AuditEventType.ACTION_FAILED,
                actor_id=context.agent_name,
                resource_type="external_action",
                resource_id=action_name,
                metadata={
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                    "response_code": result.response_code,
                },
            )
        
        return result
```

### Использование

```python
from llmteam.actions import ActionRegistry, ActionExecutor, ActionContext

# Registry
registry = ActionRegistry()

# Webhook
registry.register_webhook(
    "check_inventory",
    url="https://erp.company.com/api/inventory",
    auth_type="bearer",
    auth_config={"token": "secret"},
)

# Function
registry.register_function(
    "calculate_price",
    func=lambda data, state: {"price": data["quantity"] * 100},
)

# Executor с интеграцией v1.7.0
executor = ActionExecutor(
    registry=registry,
    rate_limiter=rate_limiter,  # v1.7.0
    audit_trail=audit_trail,    # v1.7.0
)

# Execute
result = await executor.execute(
    "check_inventory",
    ActionContext(
        action_name="check_inventory",
        run_id="run_123",
        agent_name="inventory_agent",
        tenant_id="acme",
        input_data={"sku": "PROD-001"},
    ),
)
```

---

## 📑 RFC #2: Human Interaction

### Назначение

Human-in-the-loop для критичных решений: approval, chat, task assignment.

### Модель данных

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import asyncio


class InteractionType(Enum):
    """Тип взаимодействия."""
    APPROVAL = "approval"           # Да/Нет
    CHOICE = "choice"               # Выбор из вариантов
    INPUT = "input"                 # Ввод данных
    REVIEW = "review"               # Проверка и правка
    CHAT = "chat"                   # Диалог
    TASK = "task"                   # Выполнить задачу


class InteractionStatus(Enum):
    """Статус взаимодействия."""
    PENDING = "pending"
    NOTIFIED = "notified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


class InteractionPriority(Enum):
    """Приоритет."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InteractionRequest:
    """Запрос на взаимодействие с человеком."""
    
    request_id: str
    interaction_type: InteractionType
    
    # Context
    run_id: str
    pipeline_id: str
    agent_name: str
    step_name: str
    tenant_id: str
    
    # Content
    title: str
    description: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # For CHOICE type
    options: List[Dict[str, Any]] = field(default_factory=list)
    
    # For INPUT type
    input_schema: Optional[dict] = None
    
    # Assignment
    assignee_id: Optional[str] = None
    assignee_group: Optional[str] = None
    
    # Priority & Timing
    priority: InteractionPriority = InteractionPriority.NORMAL
    timeout: timedelta = timedelta(hours=24)
    deadline: Optional[datetime] = None
    
    # SLA
    sla_warning: Optional[timedelta] = None  # Предупреждение
    sla_breach: Optional[timedelta] = None   # Нарушение
    
    # Status
    status: InteractionStatus = InteractionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    
    # Escalation
    escalation_chain: List[str] = field(default_factory=list)
    current_escalation_level: int = 0


@dataclass
class InteractionResponse:
    """Ответ человека."""
    
    request_id: str
    responder_id: str
    
    # Decision
    approved: Optional[bool] = None       # For APPROVAL
    selected_option: Optional[str] = None # For CHOICE
    input_data: Dict[str, Any] = field(default_factory=dict)  # For INPUT
    review_changes: Dict[str, Any] = field(default_factory=dict)  # For REVIEW
    
    # Metadata
    response_time: datetime = field(default_factory=datetime.now)
    comment: str = ""
    
    # Audit
    reason: str = ""


@dataclass
class NotificationConfig:
    """Конфигурация уведомлений."""
    
    # Channels
    email_enabled: bool = True
    slack_enabled: bool = False
    teams_enabled: bool = False
    webhook_enabled: bool = False
    
    # Settings
    slack_channel: str = ""
    slack_webhook_url: str = ""
    teams_webhook_url: str = ""
    custom_webhook_url: str = ""
    
    # Templates
    email_template: str = ""
    slack_template: str = ""


class NotificationChannel:
    """Базовый канал уведомлений."""
    
    async def send(self, request: InteractionRequest, recipients: List[str]) -> bool:
        raise NotImplementedError


class SlackNotificationChannel(NotificationChannel):
    """Slack уведомления."""
    
    def __init__(self, webhook_url: str, channel: str = ""):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send(self, request: InteractionRequest, recipients: List[str]) -> bool:
        payload = {
            "channel": self.channel,
            "text": f"*{request.title}*\n{request.description}",
            "attachments": [
                {
                    "color": self._get_priority_color(request.priority),
                    "fields": [
                        {"title": "Pipeline", "value": request.pipeline_id, "short": True},
                        {"title": "Priority", "value": request.priority.value, "short": True},
                        {"title": "Deadline", "value": str(request.deadline), "short": True},
                    ],
                    "actions": self._build_actions(request),
                }
            ],
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload) as resp:
                return resp.ok
    
    def _get_priority_color(self, priority: InteractionPriority) -> str:
        colors = {
            InteractionPriority.LOW: "#36a64f",
            InteractionPriority.NORMAL: "#2196f3",
            InteractionPriority.HIGH: "#ff9800",
            InteractionPriority.CRITICAL: "#f44336",
        }
        return colors.get(priority, "#2196f3")
    
    def _build_actions(self, request: InteractionRequest) -> List[dict]:
        if request.interaction_type == InteractionType.APPROVAL:
            return [
                {"type": "button", "text": "Approve", "value": "approve", "style": "primary"},
                {"type": "button", "text": "Reject", "value": "reject", "style": "danger"},
            ]
        return []


class EmailNotificationChannel(NotificationChannel):
    """Email уведомления."""
    
    def __init__(self, smtp_config: dict):
        self.smtp_config = smtp_config
    
    async def send(self, request: InteractionRequest, recipients: List[str]) -> bool:
        # Email implementation
        pass


class HumanInteractionManager:
    """
    Менеджер взаимодействия с людьми.
    
    Интегрируется с:
    - AuditTrail из v1.7.0
    - TenantContext из v1.7.0
    """
    
    def __init__(
        self,
        store: "InteractionStore",
        notification_config: NotificationConfig = None,
        audit_trail: "AuditTrail" = None,
    ):
        self.store = store
        self.notification_config = notification_config or NotificationConfig()
        self.audit_trail = audit_trail
        
        self._channels: Dict[str, NotificationChannel] = {}
        self._pending_requests: Dict[str, asyncio.Event] = {}
        
        self._setup_channels()
    
    def _setup_channels(self) -> None:
        """Настроить каналы уведомлений."""
        if self.notification_config.slack_enabled:
            self._channels["slack"] = SlackNotificationChannel(
                self.notification_config.slack_webhook_url,
                self.notification_config.slack_channel,
            )
        
        # Add other channels...
    
    async def request_approval(
        self,
        title: str,
        description: str,
        *,
        run_id: str,
        pipeline_id: str,
        agent_name: str,
        assignee: str = None,
        priority: InteractionPriority = InteractionPriority.NORMAL,
        timeout: timedelta = timedelta(hours=24),
        context_data: dict = None,
    ) -> InteractionRequest:
        """Запросить approval."""
        request = InteractionRequest(
            request_id=generate_uuid(),
            interaction_type=InteractionType.APPROVAL,
            run_id=run_id,
            pipeline_id=pipeline_id,
            agent_name=agent_name,
            step_name="approval",
            tenant_id=current_tenant.get(),
            title=title,
            description=description,
            context_data=context_data or {},
            assignee_id=assignee,
            priority=priority,
            timeout=timeout,
            deadline=datetime.now() + timeout,
        )
        
        await self.store.save(request)
        await self._notify(request)
        
        # Audit
        if self.audit_trail:
            await self.audit_trail.log(
                AuditEventType.APPROVAL_REQUESTED,
                actor_id=agent_name,
                resource_type="approval_request",
                resource_id=request.request_id,
                metadata={"title": title, "assignee": assignee},
            )
        
        return request
    
    async def request_choice(
        self,
        title: str,
        description: str,
        options: List[Dict[str, Any]],
        **kwargs,
    ) -> InteractionRequest:
        """Запросить выбор из вариантов."""
        request = InteractionRequest(
            request_id=generate_uuid(),
            interaction_type=InteractionType.CHOICE,
            title=title,
            description=description,
            options=options,
            **kwargs,
        )
        
        await self.store.save(request)
        await self._notify(request)
        return request
    
    async def request_input(
        self,
        title: str,
        description: str,
        input_schema: dict,
        **kwargs,
    ) -> InteractionRequest:
        """Запросить ввод данных."""
        request = InteractionRequest(
            request_id=generate_uuid(),
            interaction_type=InteractionType.INPUT,
            title=title,
            description=description,
            input_schema=input_schema,
            **kwargs,
        )
        
        await self.store.save(request)
        await self._notify(request)
        return request
    
    async def wait_for_response(
        self,
        request_id: str,
        timeout: float = None,
    ) -> Optional[InteractionResponse]:
        """Ожидать ответ."""
        request = await self.store.get(request_id)
        if not request:
            return None
        
        timeout = timeout or request.timeout.total_seconds()
        
        # Create event for waiting
        event = asyncio.Event()
        self._pending_requests[request_id] = event
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return await self.store.get_response(request_id)
            
        except asyncio.TimeoutError:
            # Handle timeout
            request.status = InteractionStatus.TIMEOUT
            await self.store.update(request)
            
            # Try escalation
            await self._escalate(request)
            
            return None
            
        finally:
            self._pending_requests.pop(request_id, None)
    
    async def respond(
        self,
        request_id: str,
        responder_id: str,
        *,
        approved: bool = None,
        selected_option: str = None,
        input_data: dict = None,
        comment: str = "",
        reason: str = "",
    ) -> InteractionResponse:
        """Ответить на запрос."""
        request = await self.store.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")
        
        response = InteractionResponse(
            request_id=request_id,
            responder_id=responder_id,
            approved=approved,
            selected_option=selected_option,
            input_data=input_data or {},
            comment=comment,
            reason=reason,
        )
        
        # Update request status
        request.status = InteractionStatus.COMPLETED
        if approved is False:
            request.status = InteractionStatus.REJECTED
        
        await self.store.save_response(response)
        await self.store.update(request)
        
        # Notify waiting coroutine
        if request_id in self._pending_requests:
            self._pending_requests[request_id].set()
        
        # Audit
        if self.audit_trail:
            await self.audit_trail.log_approval(
                approved=approved if approved is not None else True,
                approver_id=responder_id,
                request_id=request_id,
                reason=reason,
            )
        
        return response
    
    async def _notify(self, request: InteractionRequest) -> None:
        """Отправить уведомления."""
        recipients = [request.assignee_id] if request.assignee_id else []
        
        for channel in self._channels.values():
            await channel.send(request, recipients)
        
        request.status = InteractionStatus.NOTIFIED
        await self.store.update(request)
    
    async def _escalate(self, request: InteractionRequest) -> None:
        """Эскалировать запрос."""
        if request.current_escalation_level >= len(request.escalation_chain):
            return
        
        request.current_escalation_level += 1
        next_assignee = request.escalation_chain[request.current_escalation_level - 1]
        request.assignee_id = next_assignee
        request.status = InteractionStatus.ESCALATED
        
        await self.store.update(request)
        await self._notify(request)
        
        if self.audit_trail:
            await self.audit_trail.log(
                AuditEventType.APPROVAL_TIMEOUT,
                resource_id=request.request_id,
                metadata={"escalated_to": next_assignee},
            )
```

### Использование

```python
from llmteam.human import HumanInteractionManager, NotificationConfig, InteractionPriority
from datetime import timedelta

# Setup
manager = HumanInteractionManager(
    store=PostgresInteractionStore(conn),
    notification_config=NotificationConfig(
        slack_enabled=True,
        slack_webhook_url="https://hooks.slack.com/...",
        slack_channel="#approvals",
    ),
    audit_trail=audit_trail,
)

# В агенте
request = await manager.request_approval(
    title="Large Payment Approval",
    description=f"Payment of ${amount} to {recipient}",
    run_id=run_id,
    pipeline_id=pipeline_id,
    agent_name="payment_agent",
    assignee="finance_team",
    priority=InteractionPriority.HIGH,
    timeout=timedelta(hours=4),
    context_data={"amount": amount, "recipient": recipient},
)

# Ожидание ответа (pipeline приостанавливается)
response = await manager.wait_for_response(request.request_id)

if response and response.approved:
    # Продолжить обработку
    pass
else:
    # Отмена или альтернативный путь
    pass
```

---

## 📑 RFC #3: Persistence

### Назначение

Snapshot для pause/resume долгих workflow.

### Модель данных

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import pickle
import hashlib


class SnapshotType(Enum):
    """Тип snapshot."""
    AUTO = "auto"           # Автоматический (по интервалу)
    MANUAL = "manual"       # Ручной
    CHECKPOINT = "checkpoint"  # На checkpoint
    PAUSE = "pause"         # При паузе
    ERROR = "error"         # При ошибке (для recovery)


class PipelinePhase(Enum):
    """Фаза pipeline."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    WAITING_ACTION = "waiting_action"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentSnapshot:
    """Snapshot состояния агента."""
    
    agent_name: str
    
    # State
    state: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Messages history
    messages: List[dict] = field(default_factory=list)
    
    # Execution
    completed_steps: List[str] = field(default_factory=list)
    current_step: str = ""
    
    # Metrics
    tokens_used: int = 0
    execution_time_ms: int = 0


@dataclass
class PipelineSnapshot:
    """Snapshot состояния pipeline."""
    
    snapshot_id: str
    snapshot_type: SnapshotType
    
    # Identity
    pipeline_id: str
    run_id: str
    tenant_id: str
    
    # Version (для совместимости)
    pipeline_version: str
    snapshot_version: str = "1.0"
    
    # State
    phase: PipelinePhase
    global_state: Dict[str, Any] = field(default_factory=dict)
    
    # Agents
    agent_snapshots: Dict[str, AgentSnapshot] = field(default_factory=dict)
    
    # Execution
    completed_steps: List[str] = field(default_factory=list)
    current_step: str = ""
    next_steps: List[str] = field(default_factory=list)
    
    # Pending
    pending_actions: List[str] = field(default_factory=list)
    pending_approvals: List[str] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    
    # Metrics
    total_tokens: int = 0
    total_actions: int = 0
    total_approvals: int = 0
    
    # Integrity
    checksum: str = ""
    
    def compute_checksum(self) -> str:
        """Вычислить checksum для проверки целостности."""
        data = {
            "pipeline_id": self.pipeline_id,
            "run_id": self.run_id,
            "phase": self.phase.value,
            "global_state": json.dumps(self.global_state, sort_keys=True),
            "completed_steps": self.completed_steps,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def verify(self) -> bool:
        """Проверить целостность."""
        return self.checksum == self.compute_checksum()


@dataclass
class RestoreResult:
    """Результат восстановления."""
    
    success: bool
    snapshot_id: str
    run_id: str
    
    # Restored state
    phase: PipelinePhase
    current_step: str
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    # What was skipped
    skipped_agents: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)


class SnapshotStore:
    """Абстрактный store для snapshots."""
    
    async def save(self, snapshot: PipelineSnapshot) -> None:
        raise NotImplementedError
    
    async def load(self, snapshot_id: str) -> Optional[PipelineSnapshot]:
        raise NotImplementedError
    
    async def load_latest(self, run_id: str) -> Optional[PipelineSnapshot]:
        raise NotImplementedError
    
    async def list(self, pipeline_id: str, limit: int = 10) -> List[PipelineSnapshot]:
        raise NotImplementedError
    
    async def delete(self, snapshot_id: str) -> None:
        raise NotImplementedError


class PostgresSnapshotStore(SnapshotStore):
    """PostgreSQL store для snapshots."""
    
    def __init__(self, conn_string: str):
        self.conn_string = conn_string
    
    async def save(self, snapshot: PipelineSnapshot) -> None:
        snapshot.checksum = snapshot.compute_checksum()
        
        query = """
            INSERT INTO pipeline_snapshots (
                snapshot_id, pipeline_id, run_id, tenant_id,
                phase, data, created_at, checksum
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        # ... implementation
    
    async def load(self, snapshot_id: str) -> Optional[PipelineSnapshot]:
        query = "SELECT * FROM pipeline_snapshots WHERE snapshot_id = $1"
        # ... implementation
        pass


class RedisSnapshotStore(SnapshotStore):
    """Redis store для быстрого доступа к последним snapshots."""
    
    def __init__(self, redis_url: str, ttl_hours: int = 24):
        self.redis_url = redis_url
        self.ttl_hours = ttl_hours
    
    async def save(self, snapshot: PipelineSnapshot) -> None:
        key = f"snapshot:{snapshot.run_id}:{snapshot.snapshot_id}"
        data = pickle.dumps(snapshot)
        # SET with TTL
        pass


class HybridSnapshotStore(SnapshotStore):
    """
    Гибридный store: Redis для hot, PostgreSQL для cold.
    """
    
    def __init__(self, redis: RedisSnapshotStore, postgres: PostgresSnapshotStore):
        self.redis = redis
        self.postgres = postgres
    
    async def save(self, snapshot: PipelineSnapshot) -> None:
        # Save to both
        await asyncio.gather(
            self.redis.save(snapshot),
            self.postgres.save(snapshot),
        )
    
    async def load(self, snapshot_id: str) -> Optional[PipelineSnapshot]:
        # Try Redis first
        snapshot = await self.redis.load(snapshot_id)
        if snapshot:
            return snapshot
        
        # Fallback to PostgreSQL
        return await self.postgres.load(snapshot_id)


class SnapshotManager:
    """
    Менеджер snapshots.
    
    Интегрируется с:
    - TenantContext из v1.7.0
    - AuditTrail из v1.7.0
    """
    
    def __init__(
        self,
        store: SnapshotStore,
        auto_snapshot_interval: int = 0,  # 0 = disabled
        audit_trail: "AuditTrail" = None,
    ):
        self.store = store
        self.auto_interval = auto_snapshot_interval
        self.audit_trail = audit_trail
        
        self._step_count = 0
    
    async def create_snapshot(
        self,
        pipeline: "Pipeline",
        snapshot_type: SnapshotType = SnapshotType.MANUAL,
    ) -> PipelineSnapshot:
        """Создать snapshot."""
        
        # Collect agent snapshots
        agent_snapshots = {}
        for name, agent in pipeline._agents.items():
            agent_snapshots[name] = AgentSnapshot(
                agent_name=name,
                state=agent.get_state(),
                context=agent.context.get_visible_context(
                    viewer_id="snapshot_manager",
                    viewer_role="system",
                ),
                messages=agent.get_messages(),
                completed_steps=agent.completed_steps,
                current_step=agent.current_step,
            )
        
        snapshot = PipelineSnapshot(
            snapshot_id=generate_uuid(),
            snapshot_type=snapshot_type,
            pipeline_id=pipeline.pipeline_id,
            run_id=pipeline.current_run_id,
            tenant_id=current_tenant.get(),
            pipeline_version=pipeline.version,
            phase=pipeline.phase,
            global_state=pipeline.state.copy(),
            agent_snapshots=agent_snapshots,
            completed_steps=pipeline.completed_steps,
            current_step=pipeline.current_step,
            next_steps=pipeline.get_next_steps(),
            pending_actions=list(pipeline.pending_actions),
            pending_approvals=list(pipeline.pending_approvals),
            started_at=pipeline.started_at,
        )
        
        await self.store.save(snapshot)
        
        # Audit
        if self.audit_trail:
            await self.audit_trail.log(
                AuditEventType.PIPELINE_PAUSED,
                resource_type="pipeline_snapshot",
                resource_id=snapshot.snapshot_id,
                metadata={"type": snapshot_type.value},
            )
        
        return snapshot
    
    async def restore_snapshot(
        self,
        snapshot_id: str,
        pipeline: "Pipeline",
    ) -> RestoreResult:
        """Восстановить из snapshot."""
        
        snapshot = await self.store.load(snapshot_id)
        if not snapshot:
            return RestoreResult(
                success=False,
                snapshot_id=snapshot_id,
                run_id="",
                phase=PipelinePhase.FAILED,
                current_step="",
                warnings=["Snapshot not found"],
            )
        
        # Verify integrity
        if not snapshot.verify():
            return RestoreResult(
                success=False,
                snapshot_id=snapshot_id,
                run_id=snapshot.run_id,
                phase=PipelinePhase.FAILED,
                current_step="",
                warnings=["Snapshot integrity check failed"],
            )
        
        warnings = []
        skipped_agents = []
        
        # Restore global state
        pipeline.state = snapshot.global_state.copy()
        pipeline.current_run_id = snapshot.run_id
        pipeline.phase = snapshot.phase
        pipeline.completed_steps = snapshot.completed_steps.copy()
        pipeline.current_step = snapshot.current_step
        
        # Restore agents
        for agent_name, agent_snapshot in snapshot.agent_snapshots.items():
            if agent_name not in pipeline._agents:
                skipped_agents.append(agent_name)
                warnings.append(f"Agent '{agent_name}' not found in pipeline")
                continue
            
            agent = pipeline._agents[agent_name]
            agent.restore_state(agent_snapshot.state)
            agent.restore_messages(agent_snapshot.messages)
            agent.completed_steps = agent_snapshot.completed_steps.copy()
            agent.current_step = agent_snapshot.current_step
        
        # Restore pending items
        pipeline.pending_actions = set(snapshot.pending_actions)
        pipeline.pending_approvals = set(snapshot.pending_approvals)
        
        # Audit
        if self.audit_trail:
            await self.audit_trail.log(
                AuditEventType.PIPELINE_RESUMED,
                resource_type="pipeline_snapshot",
                resource_id=snapshot_id,
            )
        
        return RestoreResult(
            success=True,
            snapshot_id=snapshot_id,
            run_id=snapshot.run_id,
            phase=snapshot.phase,
            current_step=snapshot.current_step,
            warnings=warnings,
            skipped_agents=skipped_agents,
        )
    
    async def on_step_complete(self, pipeline: "Pipeline") -> None:
        """Callback после завершения шага."""
        self._step_count += 1
        
        if self.auto_interval > 0 and self._step_count % self.auto_interval == 0:
            await self.create_snapshot(pipeline, SnapshotType.AUTO)
    
    async def pause(self, pipeline: "Pipeline") -> PipelineSnapshot:
        """Приостановить pipeline."""
        pipeline.phase = PipelinePhase.PAUSED
        pipeline.paused_at = datetime.now()
        return await self.create_snapshot(pipeline, SnapshotType.PAUSE)
    
    async def resume(self, snapshot_id: str, pipeline: "Pipeline") -> RestoreResult:
        """Возобновить pipeline."""
        result = await self.restore_snapshot(snapshot_id, pipeline)
        if result.success:
            pipeline.phase = PipelinePhase.RUNNING
        return result
```

### Использование

```python
from llmteam.persistence import SnapshotManager, HybridSnapshotStore

# Setup
snapshot_manager = SnapshotManager(
    store=HybridSnapshotStore(
        redis=RedisSnapshotStore("redis://localhost"),
        postgres=PostgresSnapshotStore(conn),
    ),
    auto_snapshot_interval=10,  # Каждые 10 шагов
    audit_trail=audit_trail,
)

# В pipeline
pipeline = create_pipeline("long_workflow", llm=llm)
pipeline.with_snapshot_manager(snapshot_manager)

# Pause
snapshot = await snapshot_manager.pause(pipeline)
print(f"Paused at: {snapshot.current_step}")

# ... позже ...

# Resume
result = await snapshot_manager.resume(snapshot.snapshot_id, pipeline)
if result.success:
    await pipeline.continue_run()
```

---

## 📅 План реализации

| Неделя | Задачи |
|--------|--------|
| 1 | External Actions: модели, ActionRegistry |
| 2 | External Actions: handlers, интеграция с v1.7.0 Rate Limiting |
| 3 | Human Interaction: модели, InteractionManager |
| 4 | Human Interaction: notifications (Slack, Email) |
| 5 | Human Interaction: wait/respond, escalation |
| 6 | Persistence: модели, SnapshotStore implementations |
| 7 | Persistence: SnapshotManager, pause/resume |
| +0.5 | Интеграционное тестирование, документация |

**Итого: ~7.5 недель**

---

## 📁 Структура файлов

```
src/llmteam/
├── actions/
│   ├── __init__.py
│   ├── models.py              # ActionConfig, ActionContext, ActionResult
│   ├── registry.py            # ActionRegistry
│   ├── executor.py            # ActionExecutor
│   └── handlers/
│       ├── webhook.py         # WebhookActionHandler
│       ├── function.py        # FunctionActionHandler
│       ├── database.py        # DatabaseActionHandler
│       └── grpc.py            # GrpcActionHandler
│
├── human/
│   ├── __init__.py
│   ├── models.py              # InteractionRequest, InteractionResponse
│   ├── manager.py             # HumanInteractionManager
│   ├── store.py               # InteractionStore
│   └── notifications/
│       ├── base.py            # NotificationChannel
│       ├── slack.py           # SlackNotificationChannel
│       ├── email.py           # EmailNotificationChannel
│       ├── teams.py           # TeamsNotificationChannel
│       └── webhook.py         # WebhookNotificationChannel
│
└── persistence/
    ├── __init__.py
    ├── models.py              # PipelineSnapshot, AgentSnapshot
    ├── manager.py             # SnapshotManager
    └── stores/
        ├── base.py            # SnapshotStore
        ├── postgres.py        # PostgresSnapshotStore
        ├── redis.py           # RedisSnapshotStore
        └── hybrid.py          # HybridSnapshotStore
```

---

## ✅ Критерии готовности

- [ ] Интеграция с v1.7.0 (Audit, Rate Limiting, Tenant)
- [ ] Интеграция с v1.8.0 (Orchestration, Process Mining)
- [ ] Все тесты проходят
- [ ] Slack/Email notifications работают
- [ ] Snapshot restore успешен после перезапуска
- [ ] Документация с примерами
- [ ] End-to-end тест полного workflow

---

## 🎯 Результат v1.9.0

```python
from llmteam import create_pipeline
from llmteam.actions import ActionRegistry, ActionExecutor
from llmteam.human import HumanInteractionManager, NotificationConfig
from llmteam.persistence import SnapshotManager, HybridSnapshotStore

# === Setup (с интеграцией v1.7.0 и v1.8.0) ===

# Actions
actions = ActionRegistry()
actions.register_webhook("check_inventory", url="https://erp.company.com/api/inventory")
actions.register_webhook("create_order", url="https://erp.company.com/api/orders")

action_executor = ActionExecutor(
    registry=actions,
    rate_limiter=rate_limiter,  # v1.7.0
    audit_trail=audit_trail,    # v1.7.0
)

# Human interaction
human_manager = HumanInteractionManager(
    store=PostgresInteractionStore(conn),
    notification_config=NotificationConfig(
        slack_enabled=True,
        slack_webhook_url="https://hooks.slack.com/...",
    ),
    audit_trail=audit_trail,
)

# Persistence
snapshot_manager = SnapshotManager(
    store=HybridSnapshotStore(redis, postgres),
    auto_snapshot_interval=10,
    audit_trail=audit_trail,
)

# === Pipeline ===

pipeline = (
    create_pipeline("order_processing", llm=llm)
    
    # v1.7.0 - Security
    .with_tenant_manager(tenant_manager)
    .with_audit(audit_trail)
    
    # v1.8.0 - Orchestration
    .with_orchestrator(orchestrator)
    .with_executor(executor)
    
    # v1.9.0 - Workflow
    .with_actions(action_executor)
    .with_human_interaction(human_manager)
    .with_persistence(snapshot_manager)
    
    # Agents
    .add_agent("validator", prompt="...")
    .add_agent("pricer", prompt="...", 
               actions=["check_inventory"])
    .add_agent("approver", prompt="...", 
               requires_approval=True,
               approval_config={"threshold": 1000})
    .add_agent("executor", prompt="...", 
               actions=["create_order"])
)

# === Run ===

async with tenant_manager.context("acme"):
    result = await pipeline.run_async({
        "order": {"items": [...], "total": 5000}
    })
    
    # Pipeline автоматически:
    # 1. Вызовет check_inventory (External Action)
    # 2. Запросит approval если total > 1000 (Human Interaction)
    # 3. Создаст snapshot при ожидании (Persistence)
    # 4. Продолжит после approval
    # 5. Вызовет create_order
```

---

## 📊 Сводка по всем релизам

| Версия | Название | Effort | Результат |
|--------|----------|--------|-----------|
| **v1.7.0** | Security Foundation | 5 нед | Enterprise-ready security |
| **v1.8.0** | Orchestration Intelligence | 7.5 нед | Smart orchestration + analytics |
| **v1.9.0** | Workflow Runtime | 7.5 нед | Full workflow capabilities |
| **ИТОГО** | — | **20 нед** | Enterprise AI Workflow Runtime |

---

**Версия: 1.9.0**
**Кодовое имя: Workflow Runtime**
**Зависимости: v1.7.0, v1.8.0**
