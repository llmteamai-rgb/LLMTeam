# 🟠 P1 — Важные замечания (для production)

**Версия:** 2.0.0  
**Дата:** 17 января 2025  
**Статус:** ⚠️ Важно для production-ready

---

## P1-1: Отсутствует Critic Loop (RFC #6)

**Серьёзность:** 🟠 ВАЖНАЯ  
**Влияние:** Неполная реализация v2.0.0

### Проблема

RFC #6 (Critic Loop Pattern) не реализован:
- Нет класса `CriticLoop`
- Нет `CriticLoopConfig`
- Нет step type `critic_loop` в каталоге
- Нет событий `critic_loop.*`

### Что нужно реализовать

```python
# llmteam/patterns/critic_loop.py

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum

class CriticVerdict(Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"

@dataclass
class CriticLoopConfig:
    """Configuration for critic loop pattern."""
    max_iterations: int = 5
    quality_threshold: float = 0.85
    timeout_per_iteration: float = 60.0
    stop_on_rejection: bool = True
    improvement_threshold: float = 0.05  # Min improvement to continue

@dataclass
class CriticFeedback:
    """Feedback from critic agent."""
    verdict: CriticVerdict
    score: float  # 0.0 - 1.0
    feedback: str
    suggestions: list[str] = field(default_factory=list)

@dataclass
class CriticLoopResult:
    """Result of critic loop execution."""
    final_output: Any
    iterations: int
    final_score: float
    history: list[dict]  # [{iteration, output, feedback}]
    converged: bool
    reason: str  # "quality_threshold", "max_iterations", "rejected", "no_improvement"

class CriticLoop:
    """
    Recursive improvement through Generator-Critic pattern.
    
    Example:
        loop = CriticLoop(
            generator=writer_agent,
            critic=reviewer_agent,
            config=CriticLoopConfig(
                max_iterations=5,
                quality_threshold=0.85,
            ),
        )
        result = await loop.run(ctx, {"task": "Write article about AI"})
    """
    
    def __init__(
        self,
        generator: Any,  # Agent or callable
        critic: Any,      # Agent or callable
        config: CriticLoopConfig = None,
    ):
        self.generator = generator
        self.critic = critic
        self.config = config or CriticLoopConfig()
    
    async def run(
        self,
        context: "RuntimeContext",
        input_data: dict[str, Any],
    ) -> CriticLoopResult:
        """Execute critic loop until convergence or max iterations."""
        history = []
        current_input = input_data
        previous_score = 0.0
        
        for iteration in range(self.config.max_iterations):
            # Generate
            output = await self._generate(context, current_input, history)
            
            # Critique
            feedback = await self._critique(context, output, history)
            
            history.append({
                "iteration": iteration + 1,
                "output": output,
                "feedback": feedback,
            })
            
            # Check convergence
            if feedback.verdict == CriticVerdict.APPROVED:
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=feedback.score,
                    history=history,
                    converged=True,
                    reason="quality_threshold",
                )
            
            if feedback.verdict == CriticVerdict.REJECTED and self.config.stop_on_rejection:
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=feedback.score,
                    history=history,
                    converged=False,
                    reason="rejected",
                )
            
            # Check improvement
            improvement = feedback.score - previous_score
            if iteration > 0 and improvement < self.config.improvement_threshold:
                return CriticLoopResult(
                    final_output=output,
                    iterations=iteration + 1,
                    final_score=feedback.score,
                    history=history,
                    converged=False,
                    reason="no_improvement",
                )
            
            previous_score = feedback.score
            current_input = self._prepare_revision_input(input_data, output, feedback)
        
        # Max iterations reached
        return CriticLoopResult(
            final_output=history[-1]["output"],
            iterations=self.config.max_iterations,
            final_score=history[-1]["feedback"].score,
            history=history,
            converged=False,
            reason="max_iterations",
        )
```

### Effort

2-3 дня

---

## P1-2: Отсутствует pause/resume в SegmentRunner

**Серьёзность:** 🟠 ВАЖНАЯ  
**Влияние:** Невозможно приостанавливать долгие workflows

### Проблема

`SegmentRunner` имеет только:
- `run()` — запустить
- `cancel()` — отменить
- `get_status()` — статус

Отсутствуют:
- `pause()` — приостановить с сохранением состояния
- `resume()` — возобновить из snapshot

### Решение

```python
# В runner.py

async def pause(self, run_id: str) -> Optional[str]:
    """
    Pause running segment and create snapshot.
    
    Returns:
        snapshot_id if paused successfully, None if run not found
    """
    if run_id not in self._running:
        return None
    
    status = self._status.get(run_id)
    if not status or status.status != SegmentStatus.RUNNING:
        return None
    
    # Mark as pausing
    status.status = SegmentStatus.PAUSED
    
    # Create snapshot
    snapshot = SegmentSnapshot(
        snapshot_id=generate_id("snap"),
        run_id=run_id,
        segment_id=status.segment_id,
        current_step=status.current_step,
        completed_steps=list(status.completed_steps),
        step_outputs=dict(status.step_outputs),
        context_data=status.context_data,
        created_at=datetime.now(),
    )
    
    # Save to persistence
    if self._snapshot_store:
        await self._snapshot_store.save(snapshot)
    
    self._snapshots[snapshot.snapshot_id] = snapshot
    
    return snapshot.snapshot_id

async def resume(
    self,
    snapshot_id: str,
    runtime: RuntimeContext,
) -> SegmentResult:
    """
    Resume segment from snapshot.
    
    Args:
        snapshot_id: ID of snapshot to resume from
        runtime: Runtime context (can be different from original)
        
    Returns:
        SegmentResult with resumed execution
    """
    # Load snapshot
    snapshot = self._snapshots.get(snapshot_id)
    if not snapshot and self._snapshot_store:
        snapshot = await self._snapshot_store.load(snapshot_id)
    
    if not snapshot:
        raise CanvasError(f"Snapshot {snapshot_id} not found")
    
    # Load segment
    segment = await self._load_segment(snapshot.segment_id)
    
    # Create new run
    run_id = generate_id("run")
    
    # Restore state
    result = SegmentResult(
        run_id=run_id,
        segment_id=snapshot.segment_id,
        status=SegmentStatus.RUNNING,
        started_at=datetime.now(),
        resumed_from=snapshot_id,
    )
    
    # Continue from current step
    return await self._execute_from_step(
        segment=segment,
        runtime=runtime,
        result=result,
        start_step=snapshot.current_step,
        step_outputs=snapshot.step_outputs,
    )
```

### Effort

1-2 дня

---

## P1-3: Утечка памяти в MemoryAuditStore

**Серьёзность:** 🟠 ВАЖНАЯ  
**Влияние:** OOM при длительной работе

### Проблема

Файл: `src/llmteam/audit/stores/memory.py`, строка 34:

```python
self._records: List[AuditRecord] = []  # Без лимита!
```

Записи накапливаются бесконечно.

### Решение

```python
from collections import deque

class MemoryAuditStore:
    def __init__(self, max_records: int = 100_000):
        self._records: deque[AuditRecord] = deque(maxlen=max_records)
        self._max_records = max_records
```

### Также проверить

- `MemoryTenantStore`
- `MemoryKeyValueStore`
- `RateLimiter._requests_*` (уже используют deque, но без maxlen)

### Effort

2-4 часа

---

## P1-4: Отсутствует REST API слой

**Серьёзность:** 🟠 ВАЖНАЯ  
**Влияние:** Невозможна интеграция с Canvas платформами

### Проблема

Для интеграции с N8N/Nodul/КорпОС нужен HTTP API.

### Решение

Создать `llmteam/api/` или отдельный пакет `llmteam-api`:

```python
# llmteam/api/app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="LLMTeam API", version="2.0.0")

class RunRequest(BaseModel):
    segment_id: str
    input_data: dict
    idempotency_key: str | None = None
    timeout: float | None = None

class RunResponse(BaseModel):
    run_id: str
    status: str
    
@app.post("/api/v1/segments/{segment_id}/runs")
async def run_segment(
    segment_id: str,
    request: RunRequest,
    background: BackgroundTasks,
) -> RunResponse:
    """Start segment execution."""
    # Idempotency check
    if request.idempotency_key:
        existing = await cache.get(request.idempotency_key)
        if existing:
            return RunResponse(run_id=existing, status="already_started")
    
    # Start run
    run_id = await runner.start(segment_id, request.input_data)
    
    if request.idempotency_key:
        await cache.set(request.idempotency_key, run_id, ttl=3600)
    
    return RunResponse(run_id=run_id, status="started")

@app.get("/api/v1/runs/{run_id}")
async def get_run_status(run_id: str):
    """Get run status."""
    status = await runner.get_status(run_id)
    if not status:
        raise HTTPException(404, "Run not found")
    return status

@app.post("/api/v1/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Cancel running segment."""
    success = await runner.cancel(run_id)
    if not success:
        raise HTTPException(404, "Run not found or already completed")
    return {"status": "cancelled"}

@app.get("/api/v1/catalog")
async def get_catalog():
    """Get step types catalog."""
    return catalog.export_catalog()
```

### Endpoints

| Method | Endpoint | Описание |
|--------|----------|----------|
| POST | `/api/v1/segments` | Создать/обновить сегмент |
| GET | `/api/v1/segments/{id}` | Получить сегмент |
| POST | `/api/v1/segments/{id}/runs` | Запустить |
| GET | `/api/v1/runs/{id}` | Статус |
| GET | `/api/v1/runs/{id}/events` | События (SSE) |
| POST | `/api/v1/runs/{id}/cancel` | Отменить |
| POST | `/api/v1/runs/{id}/pause` | Приостановить |
| POST | `/api/v1/runs/{id}/resume` | Возобновить |
| GET | `/api/v1/catalog` | Каталог типов |
| GET | `/api/v1/health` | Health check |

### Effort

3-5 дней

---

## P1-5: Отсутствует WebSocket для real-time событий

**Серьёзность:** 🟠 ВАЖНАЯ  
**Влияние:** Нет real-time обновлений в UI

### Проблема

Canvas платформы требуют real-time обновления статуса выполнения.

### Решение

```python
# llmteam/api/websocket.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set

class ConnectionManager:
    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}  # run_id -> websockets
    
    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self._connections:
            self._connections[run_id] = set()
        self._connections[run_id].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, run_id: str):
        self._connections.get(run_id, set()).discard(websocket)
    
    async def broadcast(self, run_id: str, event: dict):
        for ws in self._connections.get(run_id, set()):
            await ws.send_json(event)

manager = ConnectionManager()

@app.websocket("/api/v1/runs/{run_id}/ws")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await manager.connect(websocket, run_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket, run_id)

# В EventEmitter добавить broadcast
class WebSocketEventEmitter(EventEmitter):
    async def emit(self, event: WorktrailEvent):
        await super().emit(event)
        await manager.broadcast(event.run_id, event.to_dict())
```

### Effort

1-2 дня

---

## P1-6: Переименование Pipeline → LLMTeam (RFC #8)

**Серьёзность:** 🟠 ВАЖНАЯ  
**Влияние:** Несоответствие брендингу и документации

### Проблема

Согласно RFC #8, класс `Pipeline` должен быть переименован в `LLMTeam`:

```python
# Было (сейчас)
from llmteam.roles import PipelineOrchestrator

# Должно быть
from llmteam import LLMTeam
from llmteam.roles import TeamOrchestrator
```

### Решение

1. Создать новые классы с новыми именами
2. Добавить алиасы для обратной совместимости
3. Добавить deprecation warnings

```python
# llmteam/core/team.py
class LLMTeam:
    """
    A team of AI agents working together.
    
    Renamed from Pipeline in v2.0.0.
    """
    pass

# llmteam/compat.py
import warnings

class Pipeline(LLMTeam):
    """Deprecated. Use LLMTeam instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Pipeline is deprecated, use LLMTeam instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

### Effort

1-2 дня

---

## P1-7: Добавить Three-Level Ports (RFC #7)

**Серьёзность:** 🟡 СРЕДНЯЯ  
**Влияние:** Упрощённая архитектура портов

### Проблема

RFC #7 требует три уровня портов:
- `workflow_in/out` — связь с внешним миром (КорпОС)
- `agent_in/out` — связь между агентами
- `human_in/out` — диалог с человеком

Сейчас порты простые: input/output.

### Решение

```python
# llmteam/ports/models.py
from enum import Enum

class PortLevel(Enum):
    WORKFLOW = "workflow"  # External: КорпОС, webhooks
    AGENT = "agent"        # Internal: between agents
    HUMAN = "human"        # Human interaction

@dataclass
class TypedPort:
    name: str
    level: PortLevel
    data_type: str = "any"
    required: bool = True
    description: str = ""

@dataclass 
class StepPorts:
    workflow_in: list[TypedPort] = field(default_factory=list)
    workflow_out: list[TypedPort] = field(default_factory=list)
    agent_in: list[TypedPort] = field(default_factory=list)
    agent_out: list[TypedPort] = field(default_factory=list)
    human_in: list[TypedPort] = field(default_factory=list)
    human_out: list[TypedPort] = field(default_factory=list)
```

### Effort

2-3 дня

---

## 📊 Сводка P1

| ID | Задача | Effort | Приоритет |
|----|--------|--------|-----------|
| P1-1 | Critic Loop (RFC #6) | 2-3 дня | Высокий |
| P1-2 | pause/resume в Runner | 1-2 дня | Высокий |
| P1-3 | Лимиты в MemoryStores | 2-4 часа | Высокий |
| P1-4 | REST API слой | 3-5 дней | Высокий |
| P1-5 | WebSocket события | 1-2 дня | Средний |
| P1-6 | Pipeline → LLMTeam | 1-2 дня | Средний |
| P1-7 | Three-Level Ports | 2-3 дня | Низкий |

**Общий effort P1:** ~2-3 недели

---

## ✅ Definition of Done для P1

- [ ] `CriticLoop` класс реализован и протестирован
- [ ] `runner.pause()` и `runner.resume()` работают
- [ ] Все MemoryStore имеют `maxlen`
- [ ] REST API запускается: `uvicorn llmteam.api:app`
- [ ] WebSocket подключение работает
- [ ] `from llmteam import LLMTeam` работает
- [ ] Three-Level Ports в StepTypeMetadata
