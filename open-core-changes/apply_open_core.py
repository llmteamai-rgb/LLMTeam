#!/usr/bin/env python3
"""
apply_open_core.py

Скрипт для автоматического применения Open Core патчей к библиотеке LLMTeam.

Использование:
    python apply_open_core.py /path/to/llmteam/src/llmteam

Что делает:
    1. Копирует обновлённый модуль licensing/
    2. Обновляет корневой __init__.py
    3. Добавляет декораторы к защищённым классам
"""

import os
import re
import sys
import shutil
from pathlib import Path


# Классы для защиты
ENTERPRISE_CLASSES = [
    ("tenancy/manager.py", "TenantManager"),
    ("tenancy/context.py", "TenantContext"),
    ("tenancy/stores/postgres.py", "PostgresTenantStore"),
    ("audit/trail.py", "AuditTrail"),
    ("audit/stores/postgres.py", "PostgresAuditStore"),
]

PROFESSIONAL_CLASSES = [
    ("roles/process_mining.py", "ProcessMiningEngine"),
    ("persistence/stores/postgres.py", "PostgresSnapshotStore"),
    ("human/manager.py", "HumanInteractionManager"),
    ("actions/executor.py", "ActionExecutor"),
    ("ratelimit/executor.py", "RateLimitedExecutor"),
]


def add_decorator_to_class(file_path: Path, class_name: str, decorator: str) -> bool:
    """
    Добавляет декоратор к классу в файле.
    
    Returns:
        True если изменения внесены, False если уже есть или ошибка
    """
    if not file_path.exists():
        print(f"  ⚠️  Файл не найден: {file_path}")
        return False
    
    content = file_path.read_text(encoding='utf-8')
    
    # Проверяем, не добавлен ли уже декоратор
    if f"@{decorator}" in content and class_name in content:
        print(f"  ✓  {class_name} уже защищён")
        return False
    
    # Ищем определение класса
    pattern = rf'^(class\s+{class_name}\s*[\(:])'
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        print(f"  ⚠️  Класс {class_name} не найден в {file_path}")
        return False
    
    # Добавляем импорт если нужно
    import_line = f"from llmteam.licensing import {decorator}\n"
    if import_line not in content:
        # Находим место для импорта (после других импортов)
        import_section_end = 0
        for m in re.finditer(r'^(from|import)\s+', content, re.MULTILINE):
            line_end = content.find('\n', m.end())
            if line_end > import_section_end:
                import_section_end = line_end + 1
        
        if import_section_end > 0:
            content = content[:import_section_end] + "\n" + import_line + content[import_section_end:]
        else:
            content = import_line + "\n" + content
    
    # Добавляем декоратор перед классом
    decorator_line = f"@{decorator}\n"
    content = re.sub(
        pattern,
        decorator_line + r'\1',
        content,
        count=1,
        flags=re.MULTILINE
    )
    
    # Сохраняем
    file_path.write_text(content, encoding='utf-8')
    print(f"  ✅ {class_name} защищён с @{decorator}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Использование: python apply_open_core.py /path/to/llmteam/src/llmteam")
        sys.exit(1)
    
    llmteam_path = Path(sys.argv[1])
    
    if not llmteam_path.exists():
        print(f"❌ Директория не найдена: {llmteam_path}")
        sys.exit(1)
    
    if not (llmteam_path / "__init__.py").exists():
        print(f"❌ Не похоже на пакет llmteam: {llmteam_path}")
        sys.exit(1)
    
    print(f"\n🔧 Применение Open Core патчей к {llmteam_path}\n")
    
    # Определяем путь к патчам
    script_dir = Path(__file__).parent
    
    # 1. Копируем модуль licensing
    print("📁 Шаг 1: Обновление модуля licensing/")
    licensing_src = script_dir / "licensing"
    licensing_dst = llmteam_path / "licensing"
    
    if licensing_src.exists():
        if licensing_dst.exists():
            shutil.rmtree(licensing_dst)
        shutil.copytree(licensing_src, licensing_dst)
        print("  ✅ licensing/ обновлён")
    else:
        print("  ⚠️  licensing/ не найден в патчах, пропускаем")
    
    # 2. Обновляем __init__.py
    print("\n📄 Шаг 2: Обновление __init__.py")
    init_src = script_dir / "__init__.py"
    init_dst = llmteam_path / "__init__.py"
    
    if init_src.exists():
        # Бэкап
        if init_dst.exists():
            backup = init_dst.with_suffix('.py.bak')
            shutil.copy(init_dst, backup)
            print(f"  📋 Бэкап: {backup}")
        
        shutil.copy(init_src, init_dst)
        print("  ✅ __init__.py обновлён")
    else:
        print("  ⚠️  __init__.py не найден в патчах, пропускаем")
    
    # 3. Добавляем декораторы Enterprise
    print("\n🔒 Шаг 3: Защита Enterprise классов")
    for file_rel, class_name in ENTERPRISE_CLASSES:
        file_path = llmteam_path / file_rel
        add_decorator_to_class(file_path, class_name, "enterprise_only")
    
    # 4. Добавляем декораторы Professional
    print("\n🔐 Шаг 4: Защита Professional классов")
    for file_rel, class_name in PROFESSIONAL_CLASSES:
        file_path = llmteam_path / file_rel
        add_decorator_to_class(file_path, class_name, "professional_only")
    
    print("\n" + "=" * 60)
    print("✅ Open Core патчи применены!")
    print("=" * 60)
    print("\nСледующие шаги:")
    print("  1. Проверьте изменения: git diff")
    print("  2. Запустите тесты: pytest")
    print("  3. Соберите пакет: python -m build")
    print("  4. Загрузите на PyPI: twine upload dist/*")
    print()


if __name__ == "__main__":
    main()
