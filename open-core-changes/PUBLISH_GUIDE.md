# 🚀 Инструкция по публикации LLMTeam на PyPI

## Обзор

Этот документ содержит пошаговую инструкцию по публикации библиотеки LLMTeam на PyPI с Open Core моделью лицензирования.

---

## Предварительные требования

- [x] Аккаунт на GitHub (llmteamai-rgb)
- [x] Репозиторий создан (LLMTeam)
- [x] Лицензия Apache 2.0
- [ ] Аккаунт на PyPI
- [ ] 2FA включён на PyPI
- [ ] API Token создан

---

## Шаг 1: Применить Open Core изменения

### Вариант A: Автоматически (рекомендуется)

```bash
cd /путь/к/llmteam

# Скачать пакет изменений (или скопировать из outputs)
# Запустить скрипт
python apply_open_core.py ./src/llmteam
```

### Вариант B: Вручную

1. Скопировать `open-core-changes/licensing/` → `src/llmteam/licensing/`
2. Скопировать `open-core-changes/__init__.py` → `src/llmteam/__init__.py`
3. Применить патчи из `PATCHES.md` к каждому файлу

---

## Шаг 2: Обновить метаданные

Заменить `pyproject.toml` на подготовленную версию:

```bash
cp /путь/к/outputs/pyproject.toml ./pyproject.toml
```

Ключевые поля:
```toml
[project]
name = "llmteam"
version = "1.9.0"
description = "Hierarchical multi-agent orchestration..."
license = "Apache-2.0"
authors = [{ name = "KirilinVS", email = "LLMTeamai@gmail.com" }]

[project.urls]
Homepage = "https://llmteam.ai"
Repository = "https://github.com/llmteamai-rgb/LLMTeam"
```

---

## Шаг 3: Обновить документацию

```bash
cp /путь/к/outputs/README.md ./README.md
cp /путь/к/outputs/CHANGELOG.md ./CHANGELOG.md
cp /путь/к/outputs/LICENSE ./LICENSE
```

---

## Шаг 4: Проверить проект

```bash
# Проверить синтаксис
python -m py_compile src/llmteam/__init__.py

# Запустить тесты (если есть)
pytest tests/ -v

# Проверить импорты
python -c "import sys; sys.path.insert(0, 'src'); import llmteam; print(llmteam.__version__)"
```

---

## Шаг 5: Загрузить на GitHub

```bash
cd /путь/к/llmteam

# Добавить все изменения
git add .

# Коммит
git commit -m "v1.9.0: Open Core licensing model

- Add license tier system (Community/Professional/Enterprise)
- Protect enterprise features with decorators
- Update metadata for PyPI publication
- Add comprehensive documentation"

# Загрузить
git push origin main

# Создать тег версии
git tag v1.9.0
git push origin v1.9.0
```

---

## Шаг 6: Зарегистрироваться на PyPI

1. Перейти на https://pypi.org/account/register/
2. Создать аккаунт
3. Подтвердить email
4. **Включить 2FA** (обязательно для новых аккаунтов):
   - Settings → Account Security → Add 2FA
5. Создать API Token:
   - Settings → API tokens → Add API token
   - Scope: "Entire account" (для первой публикации)
   - **Сохранить токен!** Он показывается только один раз

---

## Шаг 7: Настроить credentials

### Вариант A: Файл ~/.pypirc

```ini
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

### Вариант B: Переменные окружения

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE
```

---

## Шаг 8: Собрать пакет

```bash
cd /путь/к/llmteam

# Установить инструменты
pip install build twine

# Очистить старые сборки
rm -rf dist/ build/ *.egg-info

# Собрать
python -m build
```

Результат:
```
dist/
├── llmteam-1.9.0-py3-none-any.whl
└── llmteam-1.9.0.tar.gz
```

---

## Шаг 9: Проверить пакет

```bash
# Проверить метаданные
twine check dist/*

# Должно показать:
# Checking dist/llmteam-1.9.0-py3-none-any.whl: PASSED
# Checking dist/llmteam-1.9.0.tar.gz: PASSED
```

---

## Шаг 10: Тест на TestPyPI (рекомендуется)

```bash
# Загрузить на TestPyPI
twine upload --repository testpypi dist/*

# Проверить установку
pip install --index-url https://test.pypi.org/simple/ llmteam

# Проверить работу
python -c "import llmteam; print(llmteam.__version__); llmteam.print_license_status()"
```

---

## Шаг 11: Публикация на PyPI

```bash
# Загрузить на PyPI
twine upload dist/*

# Или с явным указанием токена
twine upload -u __token__ -p pypi-YOUR-TOKEN dist/*
```

---

## Шаг 12: Проверить публикацию

### Проверить страницу на PyPI

Открыть: https://pypi.org/project/llmteam/

Должно быть:
- ✅ Название: llmteam
- ✅ Версия: 1.9.0
- ✅ Описание отображается
- ✅ Ссылки работают

### Проверить установку

```bash
# В чистом окружении
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# или: test_env\Scripts\activate  # Windows

pip install llmteam

python -c "
import llmteam

print(f'Version: {llmteam.__version__}')
print(f'Tier: {llmteam.get_tier()}')

# Проверить Community features
from llmteam import RateLimiter, CircuitBreaker
print('Community features: OK')

# Проверить что Professional заблокирован
try:
    from llmteam import ProcessMiningEngine
    engine = ProcessMiningEngine()
except llmteam.FeatureNotLicensedError:
    print('Professional features: LOCKED (correct!)')
"
```

---

## Шаг 13: Создать Release на GitHub

1. Перейти: https://github.com/llmteamai-rgb/LLMTeam/releases
2. "Create a new release"
3. Tag: v1.9.0
4. Title: "LLMTeam v1.9.0 - Open Core Release"
5. Description:
```markdown
## 🎉 First Public Release

LLMTeam is now available on PyPI!

### Installation
```bash
pip install llmteam
```

### What's New in v1.9.0
- External Actions (webhooks, functions)
- Human-in-the-loop with escalation
- Pipeline state persistence (pause/resume)

### License Tiers
- **Community** (free): Basic features
- **Professional** ($99/mo): Process Mining, PostgreSQL, Human-in-the-loop
- **Enterprise**: Multi-tenant, Audit Trail, SSO

### Links
- 📦 PyPI: https://pypi.org/project/llmteam/
- 📖 Docs: https://docs.llmteam.ai
- 🐛 Issues: https://github.com/llmteamai-rgb/LLMTeam/issues
```
6. Attach files: `llmteam-1.9.0-py3-none-any.whl`, `llmteam-1.9.0.tar.gz`
7. "Publish release"

---

## ✅ Готово!

Пакет опубликован и доступен:

```bash
pip install llmteam
```

```python
import llmteam

# Community features (бесплатно)
from llmteam import RateLimiter, CircuitBreaker

# Professional features (требуют лицензию)
llmteam.activate("LLMT-PRO-XXXX-20261231")
from llmteam import ProcessMiningEngine
```

---

## Следующие шаги

1. **Настроить сайт llmteam.ai** — страница покупки лицензий
2. **Создать docs.llmteam.ai** — документация
3. **Настроить CI/CD** — автоматическая публикация при тегах
4. **Продвижение** — Hacker News, Reddit, Twitter
