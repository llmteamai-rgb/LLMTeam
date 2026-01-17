# 🟢 P2 — Улучшения (nice-to-have)

**Версия:** 2.0.0  
**Дата:** 17 января 2025  
**Статус:** 💡 Улучшения качества

---

## P2-1: Документация контрактов

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Упрощает интеграцию для внешних команд

### Проблема

Отсутствует документация API контрактов для внешних интеграторов.

### Решение

Создать документацию в `docs/contracts/`:

| Файл | Содержание |
|------|------------|
| `segment-json.md` | JSON Schema + примеры + валидация |
| `event-contract.md` | EventType enum + payload schemas |
| `runtime-context.md` | RuntimeContext API + injection |
| `step-catalog.md` | StepTypeMetadata + builtin types |
| `api-reference.md` | REST API endpoints |

**Пример `segment-json.md`:**

```markdown
# Segment JSON Contract

## Version
Current: 1.0

## Schema

\`\`\`json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["segment_id", "name", "entrypoint", "steps"],
  "properties": {
    "version": {"type": "string", "default": "1.0"},
    "segment_id": {"type": "string"},
    "name": {"type": "string"},
    "entrypoint": {"type": "string"},
    "steps": {
      "type": "array",
      "items": {"$ref": "#/definitions/StepDefinition"}
    }
  }
}
\`\`\`

## Example

\`\`\`json
{
  "version": "1.0",
  "segment_id": "article_writer",
  "name": "Article Writer Pipeline",
  "entrypoint": "research",
  "steps": [
    {
      "step_id": "research",
      "type": "llm_agent",
      "config": {
        "llm_ref": "gpt4",
        "system_prompt": "You are a researcher..."
      }
    }
  ]
}
\`\`\`
```

### Effort

2-3 дня

---

## P2-2: Генерация JSON Schema из Pydantic моделей

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Автоматическая синхронизация схем

### Проблема

JSON Schema в каталоге написаны вручную и могут рассинхронизироваться с кодом.

### Решение

Использовать Pydantic для автогенерации:

```python
# llmteam/canvas/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class LLMAgentConfig(BaseModel):
    """Configuration for llm_agent step type."""
    
    llm_ref: str = Field(..., description="Reference to LLM in registry")
    system_prompt: str = Field(..., description="System prompt for agent")
    user_prompt_template: Optional[str] = Field(
        None, 
        description="Template for user prompt with {input} placeholders"
    )
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    output_format: Optional[str] = Field(None, enum=["text", "json", "markdown"])

# Автогенерация схемы
schema = LLMAgentConfig.model_json_schema()
```

```python
# В catalog.py
def _build_builtin_types(self) -> dict[str, StepTypeMetadata]:
    return {
        "llm_agent": StepTypeMetadata(
            type_id="llm_agent",
            display_name="LLM Agent",
            config_schema=LLMAgentConfig.model_json_schema(),
            # ...
        ),
    }
```

### Effort

1 день

---

## P2-3: Валидация config по JSON Schema

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Раннее обнаружение ошибок конфигурации

### Проблема

Сейчас config шага не валидируется против JSON Schema при загрузке сегмента.

### Решение

```python
# llmteam/canvas/validation.py
from jsonschema import validate, ValidationError as JsonSchemaError

class SegmentValidator:
    def __init__(self, catalog: StepCatalog):
        self.catalog = catalog
    
    def validate_segment(self, segment: SegmentDefinition) -> list[str]:
        """Validate segment and return list of errors."""
        errors = []
        
        # Basic validation
        if not segment.steps:
            errors.append("Segment must have at least one step")
        
        # Validate each step
        step_ids = set()
        for step in segment.steps:
            # Duplicate ID
            if step.step_id in step_ids:
                errors.append(f"Duplicate step_id: {step.step_id}")
            step_ids.add(step.step_id)
            
            # Unknown type
            type_meta = self.catalog.get_type(step.type)
            if not type_meta:
                errors.append(f"Unknown step type: {step.type}")
                continue
            
            # Validate config against schema
            if type_meta.config_schema:
                try:
                    validate(step.config, type_meta.config_schema)
                except JsonSchemaError as e:
                    errors.append(f"Step {step.step_id}: {e.message}")
        
        # Validate edges
        for edge in segment.edges:
            if edge.from_step not in step_ids:
                errors.append(f"Edge references unknown step: {edge.from_step}")
            if edge.to_step not in step_ids:
                errors.append(f"Edge references unknown step: {edge.to_step}")
        
        # Validate entrypoint
        if segment.entrypoint not in step_ids:
            errors.append(f"Entrypoint references unknown step: {segment.entrypoint}")
        
        return errors
```

### Effort

4-8 часов

---

## P2-4: Prometheus метрики

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Observability для production

### Проблема

Нет метрик для мониторинга в Prometheus/Grafana.

### Решение

```python
# llmteam/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Counters
SEGMENTS_STARTED = Counter(
    'llmteam_segments_started_total',
    'Total segments started',
    ['segment_id', 'tenant_id']
)
SEGMENTS_COMPLETED = Counter(
    'llmteam_segments_completed_total',
    'Total segments completed',
    ['segment_id', 'tenant_id', 'status']
)
STEPS_EXECUTED = Counter(
    'llmteam_steps_executed_total',
    'Total steps executed',
    ['step_type', 'tenant_id']
)

# Histograms
SEGMENT_DURATION = Histogram(
    'llmteam_segment_duration_seconds',
    'Segment execution duration',
    ['segment_id'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)
STEP_DURATION = Histogram(
    'llmteam_step_duration_seconds',
    'Step execution duration',
    ['step_type'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

# Gauges
RUNNING_SEGMENTS = Gauge(
    'llmteam_running_segments',
    'Currently running segments',
    ['tenant_id']
)

# Integration with runner
class MetricsEventListener:
    async def on_segment_started(self, event):
        SEGMENTS_STARTED.labels(
            segment_id=event.segment_id,
            tenant_id=event.tenant_id,
        ).inc()
        RUNNING_SEGMENTS.labels(tenant_id=event.tenant_id).inc()
    
    async def on_segment_completed(self, event):
        SEGMENTS_COMPLETED.labels(
            segment_id=event.segment_id,
            tenant_id=event.tenant_id,
            status=event.status,
        ).inc()
        RUNNING_SEGMENTS.labels(tenant_id=event.tenant_id).dec()
```

### Effort

1-2 дня

---

## P2-5: Structured Logging

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Лучшая отладка и анализ

### Проблема

Нет стандартизированного логирования.

### Решение

```python
# llmteam/observability/logging.py
import structlog
from typing import Any

def configure_logging(
    level: str = "INFO",
    format: str = "json",  # "json" or "console"
    processors: list = None,
):
    """Configure structured logging for llmteam."""
    
    if format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

# Usage
logger = structlog.get_logger("llmteam.runner")

async def run_step(step):
    logger.info(
        "step_started",
        step_id=step.step_id,
        step_type=step.type,
        run_id=context.run_id,
    )
```

### Effort

4-8 часов

---

## P2-6: N8N Node Package

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Прямая интеграция с N8N

### Проблема

Для интеграции с N8N нужен npm пакет с нодами.

### Решение

Создать `@llmteam/n8n-nodes`:

```typescript
// packages/n8n-nodes/src/nodes/LLMTeamSegment.node.ts
import {
  IExecuteFunctions,
  INodeType,
  INodeTypeDescription,
  INodeExecutionData,
} from 'n8n-workflow';

export class LLMTeamSegment implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'LLMTeam Segment',
    name: 'llmteamSegment',
    icon: 'file:llmteam.svg',
    group: ['transform'],
    version: 1,
    description: 'Execute LLMTeam segment',
    defaults: { name: 'LLMTeam Segment' },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [
      { name: 'llmteamApi', required: true }
    ],
    properties: [
      {
        displayName: 'Segment ID',
        name: 'segmentId',
        type: 'string',
        required: true,
        default: '',
      },
      {
        displayName: 'Wait for Completion',
        name: 'waitForCompletion',
        type: 'boolean',
        default: true,
      },
      {
        displayName: 'Timeout (seconds)',
        name: 'timeout',
        type: 'number',
        default: 300,
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const credentials = await this.getCredentials('llmteamApi');
    const segmentId = this.getNodeParameter('segmentId', 0) as string;
    const waitForCompletion = this.getNodeParameter('waitForCompletion', 0);
    const timeout = this.getNodeParameter('timeout', 0) as number;
    
    const inputData = this.getInputData();
    
    // Start segment
    const response = await this.helpers.httpRequest({
      method: 'POST',
      url: `${credentials.apiUrl}/api/v1/segments/${segmentId}/runs`,
      body: { 
        input_data: inputData[0].json,
        timeout,
      },
      headers: { 
        Authorization: `Bearer ${credentials.apiKey}` 
      },
    });
    
    if (!waitForCompletion) {
      return [[{ json: response }]];
    }
    
    // Poll for completion
    const runId = response.run_id;
    const deadline = Date.now() + timeout * 1000;
    
    while (Date.now() < deadline) {
      const status = await this.helpers.httpRequest({
        method: 'GET',
        url: `${credentials.apiUrl}/api/v1/runs/${runId}`,
        headers: { Authorization: `Bearer ${credentials.apiKey}` },
      });
      
      if (status.status === 'completed') {
        return [[{ json: status.output }]];
      }
      if (status.status === 'failed') {
        throw new Error(`Segment failed: ${status.error}`);
      }
      
      await new Promise(r => setTimeout(r, 1000));
    }
    
    throw new Error('Segment execution timed out');
  }
}
```

### Effort

3-5 дней

---

## P2-7: CLI инструмент

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Удобство для разработчиков

### Проблема

Нет CLI для управления llmteam.

### Решение

```python
# llmteam/cli/main.py
import click
import asyncio
from llmteam.canvas import SegmentRunner, StepCatalog

@click.group()
def cli():
    """LLMTeam CLI - Enterprise AI Workflow Runtime"""
    pass

@cli.command()
@click.argument('segment_file')
@click.option('--input', '-i', type=click.File('r'), help='Input JSON file')
@click.option('--output', '-o', type=click.File('w'), default='-', help='Output file')
@click.option('--timeout', '-t', default=300, help='Timeout in seconds')
def run(segment_file, input, output, timeout):
    """Run a segment from file."""
    import json
    
    segment = json.load(open(segment_file))
    input_data = json.load(input) if input else {}
    
    result = asyncio.run(_run_segment(segment, input_data, timeout))
    
    json.dump(result, output, indent=2)
    click.echo(f"\nStatus: {result['status']}")

@cli.command()
def catalog():
    """List available step types."""
    cat = StepCatalog()
    
    click.echo("Available Step Types:\n")
    for type_id, meta in cat.list_types().items():
        click.echo(f"  {type_id}")
        click.echo(f"    {meta.description}")
        click.echo()

@cli.command()
@click.argument('segment_file')
def validate(segment_file):
    """Validate a segment file."""
    import json
    from llmteam.canvas.validation import SegmentValidator
    
    segment = json.load(open(segment_file))
    validator = SegmentValidator(StepCatalog())
    errors = validator.validate_segment(segment)
    
    if errors:
        click.echo("Validation errors:", err=True)
        for error in errors:
            click.echo(f"  ❌ {error}", err=True)
        raise SystemExit(1)
    
    click.echo("✅ Segment is valid")

@cli.command()
def version():
    """Show version."""
    import llmteam
    click.echo(f"llmteam {llmteam.__version__}")

if __name__ == '__main__':
    cli()
```

**pyproject.toml:**
```toml
[project.scripts]
llmteam = "llmteam.cli.main:cli"
```

### Effort

1-2 дня

---

## P2-8: Makefile для разработки

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Стандартизация команд разработки

### Решение

```makefile
# Makefile

.PHONY: install test lint format build clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/llmteam --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/llmteam/

format:
	black src/ tests/
	ruff check --fix src/ tests/

build: clean
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

release: clean lint test build
	twine upload dist/*
```

### Effort

1 час

---

## P2-9: Docker образ

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Упрощает деплой

### Решение

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Create non-root user
RUN useradd -m llmteam
USER llmteam

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "import llmteam; print('ok')"

# Default command (API server)
CMD ["uvicorn", "llmteam.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  llmteam:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLMTEAM_LICENSE_KEY=${LLMTEAM_LICENSE_KEY}
      - DATABASE_URL=${DATABASE_URL}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Effort

4-8 часов

---

## P2-10: GitHub Actions CI/CD

**Серьёзность:** 🟢 УЛУЧШЕНИЕ  
**Влияние:** Автоматизация тестирования и релизов

### Решение

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Lint
        run: |
          ruff check src/ tests/
          mypy src/llmteam/
      
      - name: Test
        run: pytest tests/ -v --cov=src/llmteam
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  release:
    needs: test
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build
        run: |
          pip install build
          python -m build
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Effort

2-4 часа

---

## 📊 Сводка P2

| ID | Задача | Effort | Приоритет |
|----|--------|--------|-----------|
| P2-1 | Документация контрактов | 2-3 дня | Средний |
| P2-2 | Pydantic JSON Schema | 1 день | Средний |
| P2-3 | Валидация config | 4-8 часов | Средний |
| P2-4 | Prometheus метрики | 1-2 дня | Низкий |
| P2-5 | Structured Logging | 4-8 часов | Низкий |
| P2-6 | N8N Node Package | 3-5 дней | Низкий |
| P2-7 | CLI инструмент | 1-2 дня | Низкий |
| P2-8 | Makefile | 1 час | Низкий |
| P2-9 | Docker образ | 4-8 часов | Низкий |
| P2-10 | GitHub Actions | 2-4 часа | Низкий |

**Общий effort P2:** ~2-3 недели

---

## 🎯 Рекомендуемый порядок

### Фаза 1: Релиз v2.0.0 (P0)
1. Очистка архива (P0-1, P0-2)
2. Обязательные файлы (P0-3, P0-4, P0-5)
3. Безопасность условий (P0-6)

### Фаза 2: Production-ready v2.1.0 (P1)
1. Critic Loop (P1-1)
2. pause/resume (P1-2)
3. Memory limits (P1-3)
4. REST API (P1-4)
5. WebSocket (P1-5)

### Фаза 3: Enterprise v2.2.0 (P2)
1. Документация (P2-1)
2. Prometheus (P2-4)
3. CLI (P2-7)
4. Docker (P2-9)
5. CI/CD (P2-10)

---

## ✅ Definition of Done для P2

- [ ] Документация в `docs/contracts/`
- [ ] JSON Schema генерируется из Pydantic
- [ ] `segment.validate()` работает
- [ ] Prometheus endpoint `/metrics`
- [ ] Structlog настроен
- [ ] N8N пакет опубликован
- [ ] `llmteam run segment.json` работает
- [ ] `make test` работает
- [ ] Docker образ собирается
- [ ] GitHub Actions зелёные
