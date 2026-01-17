# Testing Guide for llmteam

## Memory-Safe Testing

Тесты llmteam оптимизированы для работы в условиях ограниченной памяти.

### Проблема

При запуске всех тестов одновременно могут возникать проблемы с памятью из-за:
- Накопления данных в in-memory stores
- Множества async объектов
- Параллельного выполнения тестов

### Решение

Используйте один из безопасных способов запуска тестов:

## Способы запуска тестов

### 1. Рекомендуемый способ (через скрипт)

```bash
cd llmteam

# Последовательный запуск (самый безопасный)
python run_tests.py

# С ограниченной параллельностью (2 worker'а)
python run_tests.py --parallel 2

# Только конкретный модуль
python run_tests.py --module tenancy

# Только быстрые тесты
python run_tests.py --fast

# С coverage
python run_tests.py --coverage
```

### 2. Ручной запуск по модулям

```bash
cd llmteam

# Windows
set PYTHONPATH=src

# Linux/Mac
export PYTHONPATH=src

# Запускать модули по очереди
pytest tests/tenancy/ -v
pytest tests/audit/ -v
pytest tests/context/ -v
pytest tests/ratelimit/ -v
pytest tests/licensing/ -v
pytest tests/execution/ -v
pytest tests/roles/ -v
```

### 3. Контролируемый параллельный запуск

```bash
cd llmteam

# Windows
set PYTHONPATH=src && pytest tests/ -v -n 2 --dist loadgroup

# Linux/Mac
PYTHONPATH=src pytest tests/ -v -n 2 --dist loadgroup
```

**Важно:** не используйте больше 2-4 workers, это может вызвать проблемы с памятью.

### 4. Запуск с маркерами

```bash
# Только unit тесты (быстрые)
PYTHONPATH=src pytest tests/ -v -m unit

# Исключить медленные тесты
PYTHONPATH=src pytest tests/ -v -m "not slow"

# Исключить memory-intensive тесты
PYTHONPATH=src pytest tests/ -v -m "not memory_intensive"
```

## Автоматическая очистка

Все тесты теперь включают автоматическую очистку:

1. **conftest.py** - централизованные фикстуры с cleanup
2. **Autouse fixtures** - автоматическая очистка после каждого теста
3. **Garbage collection** - принудительный сбор мусора
4. **Async cleanup** - корректное закрытие async ресурсов

## Организация тестов

Тесты организованы по маркерам:

- `@pytest.mark.unit` - быстрые unit тесты
- `@pytest.mark.integration` - интеграционные тесты
- `@pytest.mark.slow` - медленные тесты
- `@pytest.mark.memory_intensive` - тесты с высоким потреблением памяти

## Coverage

```bash
# Sequential с coverage (безопасно)
python run_tests.py --coverage

# Или вручную
PYTHONPATH=src pytest tests/ -v --cov=llmteam --cov-report=html
# Отчет в htmlcov/index.html
```

## Troubleshooting

### Все еще возникают проблемы с памятью?

1. **Запускайте тесты последовательно**:
   ```bash
   python run_tests.py
   ```

2. **Запускайте по одному модулю**:
   ```bash
   python run_tests.py --module tenancy
   python run_tests.py --module audit
   # и т.д.
   ```

3. **Проверьте доступную память**:
   ```bash
   # Windows
   wmic OS get FreePhysicalMemory

   # Linux
   free -h
   ```

4. **Закройте другие приложения** перед запуском тестов

5. **Используйте fast режим** для быстрой проверки:
   ```bash
   python run_tests.py --fast
   ```

### Тест зависает?

- Каждый тест имеет timeout 30 секунд
- Используйте `--maxfail=1` чтобы остановиться на первой ошибке:
  ```bash
  PYTHONPATH=src pytest tests/ -v --maxfail=1
  ```

### Нужна отладка конкретного теста?

```bash
# Запустить один тест с полным выводом
PYTHONPATH=src pytest tests/tenancy/test_tenancy.py::TestTenantConfig::test_default_config -vv -s
```

## CI/CD Integration

Для CI/CD рекомендуется последовательный запуск:

```yaml
# .github/workflows/test.yml пример
- name: Run tests
  run: |
    cd llmteam
    python run_tests.py --coverage
```

## Структура тестов

```
tests/
├── conftest.py           # Общие фикстуры и cleanup
├── tenancy/
│   ├── __init__.py
│   └── test_tenancy.py
├── audit/
│   ├── __init__.py
│   └── test_audit.py
├── context/
│   ├── __init__.py
│   ├── test_context.py
│   └── test_hierarchical.py
└── ...
```

## Best Practices

1. **Всегда используйте фикстуры** из `conftest.py` вместо создания объектов напрямую
2. **Маркируйте тесты** соответствующими метками (`@pytest.mark.unit`, etc.)
3. **Не создавайте большие датасеты** в тестах
4. **Используйте `run_tests.py`** для обычного запуска
5. **Запускайте coverage периодически**, не при каждом запуске

## Полезные команды

```bash
# Список всех тестов
PYTHONPATH=src pytest tests/ --collect-only

# Список маркеров
PYTHONPATH=src pytest --markers

# Запуск с выводом print
PYTHONPATH=src pytest tests/ -v -s

# Остановка на первой ошибке
PYTHONPATH=src pytest tests/ -v -x

# Повтор упавших тестов
PYTHONPATH=src pytest tests/ -v --lf
```

## Требования

Минимальные требования для запуска всех тестов:
- **RAM**: 2GB+ свободной памяти
- **Python**: 3.10+
- **Dependencies**: `pip install -e ".[dev]"`

Для параллельного запуска:
- **RAM**: 4GB+ свободной памяти
- **CPU**: 2+ cores
