"""
LLMTeam Playground - Streamlit App.

Interactive interface for testing LLMTeam library.

RFC-024: v6.1.0 improvements:
- Detailed Events Log with prompts, tokens, cost
- Configurator Settings with editable system prompt
- Visual Graph (Graphviz/Mermaid/Text)
- Improved History with timeline view
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# RFC-024: Constants and Templates
# =============================================================================

# Event type icons
EVENT_ICONS = {
    "user_input": "📝",
    "run_started": "🚀",
    "task_analysis_started": "🔄",
    "team_creation_started": "🔄",
    "agent_selected": "🎯",
    "agent_started": "🔄",
    "agent_completed": "✅",
    "agent_failed": "❌",
    "adaptive_decision": "🔀",
    "checkpoint_created": "💾",
    "cost_update": "💰",
    "tool_call": "🔧",
    "tool_result": "📦",
    "run_completed": "🏁",
    "run_failed": "❌",
    "solve_completed": "🏁",
    "solve_failed": "❌",
    "error": "❌",
}

# Run mode icons
RUN_MODE_ICONS = {
    "team": "🤖",
    "solve": "🎯",
    "interactive": "💬",
}

# Default configurator system prompt
DEFAULT_CONFIGURATOR_PROMPT = """You are an expert AI team configurator. Your task is to analyze
user requests and design optimal teams of AI agents.

For each task, you should:
1. Understand the goal and constraints
2. Identify required capabilities and roles
3. Design agents with clear responsibilities
4. Define the workflow (sequential, parallel, or hybrid)
5. Suggest decision points for adaptive routing if needed

Quality interpretation:
- 0-30: Draft/fast mode - use fastest models, minimal agents
- 30-50: Economy - balance speed and quality
- 50-70: Balanced - good quality, reasonable cost
- 70-90: Production - high quality, more thorough
- 90-100: Best - maximum quality, comprehensive approach
"""

# Configurator prompt templates
CONFIGURATOR_TEMPLATES = {
    "Default": DEFAULT_CONFIGURATOR_PROMPT,
    "Minimalist": """You are a minimal AI team configurator.
Rules:
- Maximum 3 agents
- Use only gpt-4o-mini model
- Simple linear workflow (A → B → C)
- No decision points unless absolutely necessary
- Focus on speed and cost efficiency""",
    "Enterprise": """You are an enterprise AI team configurator.
Rules:
- Include validation agents for all outputs
- Add checkpoints before critical operations
- Use compliance-aware prompts
- Include audit trail in agent outputs
- Add error handling and recovery agents
- Use gpt-4o or gpt-4-turbo for critical steps""",
    "Creative": """You are a creative AI team configurator.
Rules:
- Include creative roles: writer, artist, critic
- Allow higher temperature (0.8-1.0) for creative agents
- Add brainstorming and ideation steps
- Include a critic/reviewer agent
- Use iterative refinement workflows""",
    "Research": """You are a research-focused AI team configurator.
Rules:
- Include data gathering agents
- Add fact-checking and verification steps
- Use structured output formats
- Include source citation requirements
- Add analysis and synthesis agents
- Use lower temperature (0.2-0.4) for factual accuracy""",
}

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Config file for persistent settings
CONFIG_FILE = Path(__file__).parent / ".playground_config.json"


def load_config() -> dict:
    """Load config from file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def save_config(config: dict) -> None:
    """Save config to file."""
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
    except Exception:
        pass


def get_saved_api_key() -> str:
    """Get API key from config or environment."""
    config = load_config()
    return config.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from llmteam import LLMTeam
from llmteam.agents.orchestrator import OrchestratorMode
from llmteam.quality import QualityManager

# RFC-022/RFC-023: Interactive session and Configurator
from llmteam.team.interactive import InteractiveSession, SessionState as InteractiveState
from llmteam.configuration import Configurator


# === Page Config ===
st.set_page_config(
    page_title="LLMTeam Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# === Session State Init ===
def init_session_state():
    """Initialize session state variables."""
    if "agents" not in st.session_state:
        st.session_state.agents = []
    if "team" not in st.session_state:
        st.session_state.team = None
    if "run_history" not in st.session_state:
        st.session_state.run_history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = get_saved_api_key()
    # RFC-022: Routing mode
    if "routing_mode" not in st.session_state:
        st.session_state.routing_mode = "hybrid"
    # RFC-023: Interactive session
    if "interactive_session" not in st.session_state:
        st.session_state.interactive_session = None
    if "session_messages" not in st.session_state:
        st.session_state.session_messages = []
    # RFC-023: Decision points
    if "decision_points" not in st.session_state:
        st.session_state.decision_points = []
    # RFC-024: Configurator settings
    if "configurator_model" not in st.session_state:
        st.session_state.configurator_model = "gpt-4o"
    if "configurator_system_prompt" not in st.session_state:
        st.session_state.configurator_system_prompt = DEFAULT_CONFIGURATOR_PROMPT
    # RFC-024: Workflow edges
    if "edges" not in st.session_state:
        st.session_state.edges = []
    # Always set env var from session state
    if st.session_state.api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key


init_session_state()


# === Sidebar: Settings ===
def render_sidebar():
    """Render sidebar with settings."""
    st.sidebar.title("⚙️ Settings")

    # API Key
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="Your OpenAI API key",
    )
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        # Save to config file for persistence
        config = load_config()
        config["api_key"] = api_key
        save_config(config)

    st.sidebar.divider()

    # Team Settings
    st.sidebar.subheader("Team Settings")

    team_id = st.sidebar.text_input("Team ID", value="playground-team")

    quality = st.sidebar.slider(
        "Quality",
        min_value=0,
        max_value=100,
        value=50,
        help="0=draft, 50=balanced, 100=best quality",
    )

    quality_preset = st.sidebar.selectbox(
        "Or use preset",
        options=["(custom)", "draft", "economy", "balanced", "production", "best"],
        index=0,
    )

    if quality_preset != "(custom)":
        preset_values = {"draft": 20, "economy": 30, "balanced": 50, "production": 75, "best": 95}
        quality = preset_values[quality_preset]

    orchestration = st.sidebar.checkbox(
        "Enable Router Mode",
        value=True,
        help="Orchestrator selects which agent to run",
    )

    # RFC-022: Routing Mode
    st.sidebar.divider()
    st.sidebar.subheader("Routing Mode")
    routing_mode = st.sidebar.radio(
        "Routing Strategy",
        options=["hybrid", "sequential", "dynamic"],
        index=0,
        help="""
        - **hybrid** (recommended): Deterministic graph + AdaptiveStep for decisions
        - **sequential**: Fixed agent order, no dynamic routing
        - **dynamic**: LLM decides each step (flexible but expensive)
        """,
    )
    st.session_state.routing_mode = routing_mode

    if routing_mode == "hybrid":
        st.sidebar.info("Rules-first routing with LLM fallback")
    elif routing_mode == "dynamic":
        st.sidebar.warning("Full LLM routing - higher cost")

    st.sidebar.divider()

    max_cost = st.sidebar.number_input(
        "Max Cost per Run ($)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1,
        help="Budget limit for single run",
    )

    st.sidebar.divider()

    # Export/Import
    st.sidebar.subheader("Export/Import")

    if st.sidebar.button("📥 Export Config"):
        config = {
            "team_id": team_id,
            "quality": quality,
            "routing_mode": routing_mode,
            "orchestration": orchestration,
            "max_cost_per_run": max_cost,
            "agents": st.session_state.agents,
            "edges": st.session_state.edges,
            "decision_points": st.session_state.decision_points,
            # RFC-024: Configurator settings
            "configurator": {
                "model": st.session_state.configurator_model,
                "system_prompt": st.session_state.configurator_system_prompt,
            },
        }
        st.sidebar.download_button(
            "Download JSON",
            data=json.dumps(config, indent=2, ensure_ascii=False),
            file_name=f"{team_id}_config.json",
            mime="application/json",
        )

    uploaded = st.sidebar.file_uploader("Import Config", type=["json"])
    if uploaded:
        try:
            config = json.load(uploaded)
            st.session_state.agents = config.get("agents", [])
            st.session_state.edges = config.get("edges", [])
            st.session_state.decision_points = config.get("decision_points", [])
            st.session_state.routing_mode = config.get("routing_mode", "hybrid")
            # RFC-024: Load configurator settings
            if "configurator" in config:
                st.session_state.configurator_model = config["configurator"].get("model", "gpt-4o")
                st.session_state.configurator_system_prompt = config["configurator"].get(
                    "system_prompt", DEFAULT_CONFIGURATOR_PROMPT
                )
            st.sidebar.success("Config loaded!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    return {
        "team_id": team_id,
        "quality": quality,
        "routing_mode": routing_mode,
        "orchestration": orchestration,
        "max_cost_per_run": max_cost,
    }


# === Agent Builder ===
def render_agent_builder():
    """Render agent builder section."""
    st.header("🤖 Team Builder")

    # RFC-024: Sub-tabs for different modes
    sub_tabs = st.tabs(["🧠 Конфигуратор", "✋ Ручной режим", "⚙️ Настройки конфигуратора"])

    with sub_tabs[0]:
        render_configurator_mode()

    with sub_tabs[1]:
        render_manual_mode()

    with sub_tabs[2]:
        render_configurator_settings()


def render_configurator_mode():
    """Render configurator-based team builder."""
    st.subheader("🧠 Конфигуратор команды")
    st.markdown("Опишите задачу, и AI создаст оптимальную команду агентов.")

    # Check API key
    if not st.session_state.api_key:
        st.warning("⚠️ Введите OpenAI API Key в боковой панели")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        task_description = st.text_area(
            "Опишите вашу задачу",
            placeholder="Например: Создать команду для анализа новостей, извлечения ключевых фактов и написания краткого отчёта",
            height=120,
            key="task_description",
        )

        constraints = st.text_input(
            "Ограничения (опционально)",
            placeholder="Например: максимум 3 агента, использовать gpt-4o-mini",
            key="constraints",
        )

    with col2:
        st.markdown("**Примеры задач:**")
        st.markdown("""
        - Исследование темы и написание статьи
        - Анализ данных и визуализация
        - Перевод и локализация контента
        - Код-ревью и рефакторинг
        - Генерация маркетинговых текстов
        """)

    if st.button("🚀 Создать команду", type="primary", use_container_width=True):
        if not task_description:
            st.error("Опишите задачу")
            return

        with st.spinner("Анализирую задачу и создаю команду..."):
            asyncio.run(run_configurator(task_description, constraints))

    # Show current team
    st.divider()
    render_current_agents()


async def run_configurator(task: str, constraints: str = ""):
    """Run configurator to create team."""
    try:
        from llmteam.builder import DynamicTeamBuilder

        builder = DynamicTeamBuilder(
            model="gpt-4o-mini",
            quality=st.session_state.get("quality", 50),
        )

        # Add constraints to task if provided
        full_task = task
        if constraints:
            full_task = f"{task}\n\nОграничения: {constraints}"

        # Analyze task and get blueprint
        blueprint = await builder.analyze_task(full_task)

        # Convert blueprint agents to our format
        new_agents = []
        for agent in blueprint.agents:
            agent_config = {
                "type": "llm",
                "role": agent.role,
                "prompt": agent.prompt,
                "model": agent.model,
                "temperature": agent.temperature,
                "max_tokens": 1000,
            }
            new_agents.append(agent_config)

        st.session_state.agents = new_agents
        st.success(f"✅ Создана команда из {len(new_agents)} агентов!")

        # Show blueprint info
        st.info(f"**{blueprint.team_id}**: {blueprint.description}")

        st.rerun()

    except ImportError:
        st.error("DynamicTeamBuilder не доступен. Используйте ручной режим.")
    except Exception as e:
        st.error(f"Ошибка: {e}")


def render_manual_mode():
    """Render manual agent builder."""
    st.subheader("✋ Ручное создание агентов")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Добавить агента**")

        agent_type = st.selectbox(
            "Тип агента",
            options=["llm", "rag", "kag"],
            index=0,
        )

        role = st.text_input("Роль", placeholder="например: researcher, writer, reviewer")

        prompt = st.text_area(
            "Промпт",
            placeholder="Ты помощник, который выполняет задачу: {task}...",
            height=150,
        )

        model = st.selectbox(
            "Модель",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
        )

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=16000,
            value=1000,
            step=100,
        )

        if st.button("➕ Add Agent", type="primary"):
            if not role:
                st.error("Role is required")
            elif not prompt:
                st.error("Prompt is required")
            else:
                agent_config = {
                    "type": agent_type,
                    "role": role,
                    "prompt": prompt,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                # Note: tools are not included as they require ToolDefinition objects

                st.session_state.agents.append(agent_config)
                st.success(f"Added agent: {role}")
                st.rerun()

    with col2:
        render_current_agents()


def render_current_agents():
    """Render current agents list."""
    st.subheader("📋 Текущая команда")

    if not st.session_state.agents:
        st.info("Команда пуста. Создайте агентов.")
        return

    for i, agent in enumerate(st.session_state.agents):
        with st.expander(f"**{agent['role']}** ({agent['type']})", expanded=False):
            st.code(json.dumps(agent, indent=2, ensure_ascii=False), language="json")

            col_edit, col_del = st.columns(2)
            with col_del:
                if st.button(f"🗑️ Удалить", key=f"del_{i}"):
                    st.session_state.agents.pop(i)
                    st.rerun()

    # Clear all button
    if len(st.session_state.agents) > 1:
        if st.button("🗑️ Очистить всё", type="secondary"):
            st.session_state.agents = []
            st.rerun()


# === RFC-024: Configurator Settings ===
def render_configurator_settings():
    """Render configurator settings sub-tab."""
    st.subheader("⚙️ Настройки конфигуратора")
    st.markdown("Настройте модель и системный промпт для AI-конфигуратора.")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Model selector
        model = st.selectbox(
            "Модель конфигуратора",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-5-sonnet"],
            index=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-5-sonnet"].index(
                st.session_state.configurator_model
            ) if st.session_state.configurator_model in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-5-sonnet"] else 0,
            help="Модель для анализа задач и создания команд",
            key="cfg_model_select",
        )

        st.divider()

        # Template selector
        st.markdown("**Шаблоны промптов:**")
        template_name = st.selectbox(
            "Выберите шаблон",
            options=list(CONFIGURATOR_TEMPLATES.keys()),
            index=0,
            key="cfg_template_select",
        )

        if st.button("📥 Применить шаблон", key="cfg_apply_template"):
            st.session_state.configurator_system_prompt = CONFIGURATOR_TEMPLATES[template_name]
            st.success(f"Применён шаблон: {template_name}")
            st.rerun()

        st.divider()

        # Template descriptions
        st.markdown("**Описания шаблонов:**")
        st.markdown("""
        - **Default** — Стандартный универсальный промпт
        - **Minimalist** — Максимум 3 агента, gpt-4o-mini
        - **Enterprise** — Валидация, checkpoints, compliance
        - **Creative** — Writer, artist, critic роли
        - **Research** — Data gathering, fact-checking
        """)

    with col2:
        # System prompt editor
        prompt = st.text_area(
            "Системный промпт",
            value=st.session_state.configurator_system_prompt,
            height=400,
            key="cfg_system_prompt",
            help="Промпт, который определяет поведение конфигуратора",
        )

        col_save, col_reset = st.columns(2)

        with col_save:
            if st.button("💾 Сохранить", type="primary", key="cfg_save"):
                st.session_state.configurator_model = model
                st.session_state.configurator_system_prompt = prompt
                st.success("Настройки сохранены!")

        with col_reset:
            if st.button("🔄 Сбросить к дефолтному", key="cfg_reset"):
                st.session_state.configurator_model = "gpt-4o"
                st.session_state.configurator_system_prompt = DEFAULT_CONFIGURATOR_PROMPT
                st.success("Настройки сброшены")
                st.rerun()

    # Show current settings
    st.divider()
    st.markdown("**Текущие настройки:**")
    col_m, col_p = st.columns(2)
    with col_m:
        st.info(f"Модель: **{st.session_state.configurator_model}**")
    with col_p:
        prompt_preview = st.session_state.configurator_system_prompt[:100] + "..."
        st.info(f"Промпт: {prompt_preview}")


# =============================================================================
# RFC-024: Workflow Designer
# =============================================================================

def generate_dot_graph() -> str:
    """Generate DOT graph from agents, edges and decision points."""
    agents = st.session_state.agents
    edges = st.session_state.edges
    decision_points = st.session_state.decision_points

    lines = [
        'digraph workflow {',
        '    rankdir=LR;',
        '    node [fontname="Arial"];',
        '    edge [fontname="Arial", fontsize=10];',
        '',
        '    // Start and End nodes',
        '    START [shape=circle style=filled fillcolor="#90ee90" label="START"];',
        '    END [shape=circle style=filled fillcolor="#ff6b6b" label="END"];',
        '',
        '    // Agent nodes',
    ]

    # Add agent nodes
    for agent in agents:
        role = agent.get("role", "agent")
        model = agent.get("model", "")
        label = f"{role}\\n({model})" if model else role
        lines.append(f'    {role} [shape=box style="filled,rounded" fillcolor="#87ceeb" label="{label}"];')

    # Add decision point nodes
    if decision_points:
        lines.append('')
        lines.append('    // Decision point nodes')
        for dp in decision_points:
            dp_id = dp.get("decision_id", "dp")
            dp_type = dp.get("decision_type", "branch")
            lines.append(f'    {dp_id} [shape=diamond style=filled fillcolor="#ffd700" label="{dp_id}\\n({dp_type})"];')

    lines.append('')
    lines.append('    // Edges')

    # Add edges
    if edges:
        for edge in edges:
            from_node = edge.get("from", "START")
            to_node = edge.get("to", "END")
            condition = edge.get("condition", "")
            if condition:
                lines.append(f'    {from_node} -> {to_node} [label="{condition}"];')
            else:
                lines.append(f'    {from_node} -> {to_node};')
    elif agents:
        # Auto-generate linear flow if no edges defined
        lines.append('    // Auto-generated linear flow')
        if agents:
            lines.append(f'    START -> {agents[0]["role"]};')
            for i in range(len(agents) - 1):
                lines.append(f'    {agents[i]["role"]} -> {agents[i+1]["role"]};')
            lines.append(f'    {agents[-1]["role"]} -> END;')
        else:
            lines.append('    START -> END;')

    lines.append('}')
    return '\n'.join(lines)


def generate_mermaid_code() -> str:
    """Generate Mermaid flowchart code."""
    agents = st.session_state.agents
    edges = st.session_state.edges
    decision_points = st.session_state.decision_points

    lines = [
        'flowchart LR',
        '    START((START))',
        '    END((END))',
    ]

    # Add agent nodes
    for agent in agents:
        role = agent.get("role", "agent")
        model = agent.get("model", "")
        label = f"{role}<br/>{model}" if model else role
        lines.append(f'    {role}["{label}"]')

    # Add decision point nodes
    for dp in decision_points:
        dp_id = dp.get("decision_id", "dp")
        lines.append(f'    {dp_id}{{{{{dp_id}}}}}')

    # Add edges
    if edges:
        for edge in edges:
            from_node = edge.get("from", "START")
            to_node = edge.get("to", "END")
            condition = edge.get("condition", "")
            if condition:
                lines.append(f'    {from_node} -->|{condition}| {to_node}')
            else:
                lines.append(f'    {from_node} --> {to_node}')
    elif agents:
        # Auto-generate linear flow
        lines.append(f'    START --> {agents[0]["role"]}')
        for i in range(len(agents) - 1):
            lines.append(f'    {agents[i]["role"]} --> {agents[i+1]["role"]}')
        lines.append(f'    {agents[-1]["role"]} --> END')

    # Add styling
    lines.append('')
    lines.append('    style START fill:#90ee90')
    lines.append('    style END fill:#ff6b6b')
    for agent in agents:
        lines.append(f'    style {agent["role"]} fill:#87ceeb')
    for dp in decision_points:
        lines.append(f'    style {dp["decision_id"]} fill:#ffd700')

    return '\n'.join(lines)


def generate_text_graph() -> str:
    """Generate ASCII text representation of the workflow."""
    agents = st.session_state.agents
    edges = st.session_state.edges
    decision_points = st.session_state.decision_points

    if not agents:
        return "START → END\n\n(No agents defined)"

    lines = []
    lines.append("=" * 60)
    lines.append("WORKFLOW GRAPH")
    lines.append("=" * 60)
    lines.append("")

    # Show flow notation
    if edges:
        lines.append("Flow (from edges):")
        for edge in edges:
            cond = f" [{edge['condition']}]" if edge.get("condition") else ""
            lines.append(f"  {edge['from']} → {edge['to']}{cond}")
    else:
        # Linear flow
        flow_parts = ["START"] + [a["role"] for a in agents] + ["END"]
        lines.append("Flow (linear):")
        lines.append("  " + " → ".join(flow_parts))

    lines.append("")
    lines.append("-" * 60)
    lines.append("NODES:")
    lines.append("-" * 60)

    # Agents
    for agent in agents:
        lines.append(f"  [AGENT] {agent['role']}")
        lines.append(f"          Model: {agent.get('model', 'default')}")
        lines.append(f"          Type: {agent.get('type', 'llm')}")
        lines.append("")

    # Decision points
    if decision_points:
        lines.append("-" * 60)
        lines.append("DECISION POINTS:")
        lines.append("-" * 60)
        for dp in decision_points:
            lines.append(f"  [DECISION] {dp.get('decision_id', 'dp')}")
            lines.append(f"             Type: {dp.get('decision_type', 'branch')}")
            lines.append(f"             After: {dp.get('after_step', 'N/A')}")
            if dp.get("rules"):
                lines.append(f"             Rules: {len(dp['rules'])}")
            lines.append("")

    lines.append("=" * 60)
    return '\n'.join(lines)


def auto_generate_edges():
    """Auto-generate linear edges from agents."""
    agents = st.session_state.agents
    if not agents:
        return []

    edges = [{"from": "START", "to": agents[0]["role"]}]
    for i in range(len(agents) - 1):
        edges.append({"from": agents[i]["role"], "to": agents[i + 1]["role"]})
    edges.append({"from": agents[-1]["role"], "to": "END"})
    return edges


def render_workflow_designer():
    """Render Workflow Designer tab with visual graph."""
    st.header("🔀 Workflow Designer")

    if not st.session_state.agents:
        st.info("Добавьте агентов во вкладке 'Agents', чтобы создать workflow.")
        return

    # Sub-tabs
    sub_tabs = st.tabs(["📊 Visual Graph", "✏️ Edit Flow", "📋 Flow Summary"])

    with sub_tabs[0]:
        render_visual_graph()

    with sub_tabs[1]:
        render_edge_editor()

    with sub_tabs[2]:
        render_flow_summary()


def render_visual_graph():
    """Render visual graph sub-tab."""
    st.subheader("📊 Visual Graph")

    # Visualization mode selector
    viz_mode = st.radio(
        "Режим визуализации",
        options=["Graphviz", "Mermaid", "Text"],
        horizontal=True,
        help="Graphviz отображается интерактивно, Mermaid - код для копирования",
    )

    if viz_mode == "Graphviz":
        dot_code = generate_dot_graph()

        # Show graph
        try:
            st.graphviz_chart(dot_code, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка отображения графа: {e}")
            st.code(dot_code, language="dot")

        # Show DOT code in expander
        with st.expander("Показать DOT код"):
            st.code(dot_code, language="dot")

    elif viz_mode == "Mermaid":
        mermaid_code = generate_mermaid_code()

        st.markdown("**Mermaid код** (скопируйте в [Mermaid Live Editor](https://mermaid.live)):")
        st.code(mermaid_code, language="mermaid")

        st.markdown("[Открыть Mermaid Live Editor →](https://mermaid.live)")

    else:  # Text
        text_graph = generate_text_graph()
        st.code(text_graph, language="text")


def render_edge_editor():
    """Render edge editor sub-tab."""
    st.subheader("✏️ Edit Flow")

    agents = st.session_state.agents
    decision_points = st.session_state.decision_points

    # Available nodes
    nodes = ["START"] + [a["role"] for a in agents]
    for dp in decision_points:
        nodes.append(dp.get("decision_id", "dp"))
    nodes.append("END")

    st.markdown(f"**Доступные узлы:** {', '.join(nodes)}")

    # Current edges table
    st.markdown("**Текущие связи:**")

    if st.session_state.edges:
        for i, edge in enumerate(st.session_state.edges):
            col1, col2, col3, col4 = st.columns([2, 2, 3, 1])
            with col1:
                st.text(edge.get("from", "?"))
            with col2:
                st.text(edge.get("to", "?"))
            with col3:
                st.text(edge.get("condition", "-"))
            with col4:
                if st.button("❌", key=f"del_edge_{i}"):
                    st.session_state.edges.pop(i)
                    st.rerun()
    else:
        st.info("Нет связей. Добавьте или сгенерируйте автоматически.")

    st.divider()

    # Add new edge
    st.markdown("**Добавить связь:**")
    col_from, col_to, col_cond = st.columns(3)

    with col_from:
        from_node = st.selectbox("From", options=nodes, key="edge_from")
    with col_to:
        to_node = st.selectbox("To", options=nodes, key="edge_to")
    with col_cond:
        condition = st.text_input("Condition (опционально)", key="edge_condition",
                                   placeholder="output.quality >= 80")

    col_add, col_auto, col_clear = st.columns(3)

    with col_add:
        if st.button("➕ Добавить связь", type="primary"):
            new_edge = {"from": from_node, "to": to_node}
            if condition:
                new_edge["condition"] = condition
            st.session_state.edges.append(new_edge)
            st.rerun()

    with col_auto:
        if st.button("🔗 Линейный поток"):
            st.session_state.edges = auto_generate_edges()
            st.success("Сгенерирован линейный поток")
            st.rerun()

    with col_clear:
        if st.button("🔄 Очистить связи"):
            st.session_state.edges = []
            st.rerun()


def render_flow_summary():
    """Render flow summary sub-tab."""
    st.subheader("📋 Flow Summary")

    agents = st.session_state.agents
    edges = st.session_state.edges
    decision_points = st.session_state.decision_points
    routing_mode = st.session_state.routing_mode

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Agents", len(agents))
    with col2:
        st.metric("Edges", len(edges) if edges else "auto")
    with col3:
        st.metric("Decision Points", len(decision_points))
    with col4:
        st.metric("Routing Mode", routing_mode)

    st.divider()

    # Flow notation
    st.markdown("**Flow Notation:**")
    if edges:
        # Build flow string from edges
        flow_parts = []
        for edge in edges:
            cond = f"[{edge['condition']}]" if edge.get("condition") else ""
            flow_parts.append(f"{edge['from']} →{cond} {edge['to']}")
        st.code("\n".join(flow_parts))
    elif agents:
        # Linear notation
        flow = " → ".join(["START"] + [a["role"] for a in agents] + ["END"])
        st.code(flow)

    # Validation warnings
    st.markdown("**Validation:**")
    warnings = []

    if routing_mode == "hybrid" and not decision_points:
        warnings.append("⚠️ Hybrid mode без Decision Points — добавьте точки принятия решений")

    if routing_mode == "sequential" and decision_points:
        warnings.append("⚠️ Sequential mode с Decision Points — переключитесь на Hybrid")

    # Check for unreachable agents
    if edges:
        connected = set()
        for edge in edges:
            connected.add(edge.get("to"))
        for agent in agents:
            if agent["role"] not in connected and agent != agents[0]:
                warnings.append(f"⚠️ Agent '{agent['role']}' не достижим (нет входящих связей)")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("✅ Workflow валиден")

    # Export
    st.divider()
    st.markdown("**Export:**")

    workflow_def = {
        "agents": [{"role": a["role"], "model": a.get("model")} for a in agents],
        "edges": edges,
        "decision_points": decision_points,
        "routing_mode": routing_mode,
    }

    st.json(workflow_def)

    st.download_button(
        "📥 Export Workflow JSON",
        data=json.dumps(workflow_def, indent=2, ensure_ascii=False),
        file_name="workflow.json",
        mime="application/json",
    )


# =============================================================================
# RFC-024: Events Log Formatting
# =============================================================================

def format_event_for_display(event_info: dict, agents: list) -> str:
    """Format event info as markdown for timeline display."""
    event_type = event_info.get("type", "unknown").lower()
    icon = EVENT_ICONS.get(event_type, "📌")
    timestamp = event_info.get("timestamp", "")
    if timestamp:
        # Extract just the time part
        try:
            timestamp = timestamp.split("T")[1][:8]
        except Exception:
            pass

    lines = []

    # Header line
    header = f"**{icon} {event_type.upper()}**"
    if timestamp:
        header += f" `{timestamp}`"
    if event_info.get("agent_id"):
        header += f" — **{event_info['agent_id']}**"
    lines.append(header)

    # Details based on event type
    if event_type == "user_input":
        input_text = event_info.get("input", "")
        if input_text:
            lines.append(f"> {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
        if event_info.get("quality"):
            lines.append(f"Quality: {event_info['quality']}")

    elif event_type == "run_started":
        team_id = event_info.get("team_id", "unknown")
        agents_list = event_info.get("agents", [])
        lines.append(f"Team: **{team_id}** | Agents: {', '.join(agents_list) if agents_list else 'N/A'}")

    elif event_type == "agent_selected":
        reason = event_info.get("reason", "")
        if reason:
            lines.append(f"> {reason[:150]}{'...' if len(reason) > 150 else ''}")
        model = event_info.get("model", "")
        confidence = event_info.get("confidence")
        details = []
        if model:
            details.append(f"Model: {model}")
        if confidence:
            details.append(f"Confidence: {confidence}")
        if details:
            lines.append(" | ".join(details))
        # Show prompt preview if available
        prompt_preview = event_info.get("prompt_preview", "")
        if prompt_preview:
            lines.append(f"```\n{prompt_preview[:200]}{'...' if len(prompt_preview) > 200 else ''}\n```")

    elif event_type == "agent_completed":
        output = event_info.get("output", "")
        if output:
            lines.append(f"> {output[:150]}{'...' if len(output) > 150 else ''}")
        tokens = event_info.get("tokens")
        cost = event_info.get("cost")
        details = []
        if tokens:
            details.append(f"Tokens: {tokens}")
        if cost:
            details.append(f"Cost: ${cost:.4f}")
        if event_info.get("output_length"):
            details.append(f"Length: {event_info['output_length']}")
        if details:
            lines.append(" | ".join(details))

    elif event_type == "adaptive_decision":
        method = event_info.get("method", "unknown")
        target = event_info.get("target", "unknown")
        method_icon = "📏" if method == "rule" else "🧠"
        lines.append(f"{method_icon} **{method.upper()}** → **{target}**")
        if event_info.get("rule_matched"):
            lines.append(f"Rule: `{event_info['rule_matched']}`")
        if event_info.get("reasoning"):
            lines.append(f"> {event_info['reasoning'][:100]}")

    elif event_type == "cost_update":
        tokens = event_info.get("tokens", 0)
        cost = event_info.get("cost", 0)
        lines.append(f"Tokens: {tokens} | Cost: ${cost:.4f}" if cost else f"Tokens: {tokens}")

    elif event_type == "tool_call":
        tool = event_info.get("tool", "unknown")
        lines.append(f"Tool: **{tool}**")

    elif event_type in ["run_completed", "solve_completed"]:
        lines.append("✅ **Success**")
        if event_info.get("duration"):
            lines.append(f"Duration: {event_info['duration']:.2f}s")

    elif event_type in ["run_failed", "solve_failed", "error"]:
        error = event_info.get("error", "Unknown error")
        lines.append(f"❌ Error: {error[:200]}")

    return "\n".join(lines)


# === Team Runner ===
def render_team_runner(settings: Dict[str, Any]):
    """Render team execution section."""
    st.header("▶️ Run Team")

    if not st.session_state.agents:
        st.warning("Add at least one agent to run the team.")
        return

    if not st.session_state.api_key:
        st.warning("Set OpenAI API Key in the sidebar.")
        return

    # Input data
    st.subheader("Input Data")

    input_mode = st.radio(
        "Input Mode",
        options=["Simple Text", "JSON"],
        horizontal=True,
    )

    if input_mode == "Simple Text":
        user_input = st.text_area(
            "Enter your query",
            placeholder="What would you like the team to do?",
            height=100,
        )
        input_data = {"input": user_input, "query": user_input}
    else:
        json_input = st.text_area(
            "JSON Input",
            value='{\n  "query": ""\n}',
            height=150,
        )
        try:
            input_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            input_data = {}

    # Run options
    col1, col2, col3 = st.columns(3)

    with col1:
        run_quality = st.number_input(
            "Quality Override",
            min_value=0,
            max_value=100,
            value=settings["quality"],
            help="Override quality for this run",
        )

    with col2:
        importance = st.selectbox(
            "Importance",
            options=["normal", "low", "high", "critical"],
            index=0,
        )

    with col3:
        stream_mode = st.checkbox("Streaming Mode", value=True)

    # Run button
    if st.button("🚀 Run Team", type="primary", use_container_width=True):
        if not input_data or (input_mode == "Simple Text" and not user_input):
            st.error("Please provide input data")
            return

        run_team(settings, input_data, run_quality, importance, stream_mode)


def run_team(
    settings: Dict[str, Any],
    input_data: Dict[str, Any],
    quality: int,
    importance: str,
    stream: bool,
):
    """Execute the team."""

    # Create team
    team = LLMTeam(
        team_id=settings["team_id"],
        agents=st.session_state.agents,
        orchestration=settings["orchestration"],
        quality=settings["quality"],
        max_cost_per_run=settings["max_cost_per_run"],
    )

    st.session_state.team = team

    # Output container
    output_container = st.container()

    with output_container:
        st.subheader("Execution")

        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Events log
        events_container = st.expander("📋 Events Log", expanded=True)
        events_log = []

        # Result container
        result_container = st.empty()

        async def execute():
            start_time = datetime.now()
            total_tokens = 0
            total_cost = 0.0
            routing_decisions = []

            if stream:
                status_text.text("Starting stream...")

                with events_container:
                    # RFC-024: Use markdown for timeline view
                    event_placeholder = st.empty()
                    timeline_md = []

                    try:
                        i = 0
                        async for event in team.stream(input_data):
                            i += 1
                            progress_bar.progress(min(i * 10, 100))

                            # Safely get event type name
                            try:
                                event_type_name = event.type.name if hasattr(event.type, 'name') else str(event.type)
                            except Exception:
                                event_type_name = "UNKNOWN"

                            event_info = {
                                "type": event_type_name,
                                "agent_id": event.agent_id,
                                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                            }

                            # Use string comparison to avoid enum caching issues
                            event_type_str = event_type_name.lower()
                            icon = EVENT_ICONS.get(event_type_str, "📌")

                            if event_type_str == "run_started":
                                status_text.text(f"🚀 Run started: {event.data.get('team_id', 'unknown')}")
                                event_info["team_id"] = event.data.get("team_id")
                                event_info["agents"] = event.data.get("agents", [])

                            elif event_type_str == "user_input":
                                user_input_str = str(event.data.get("input", {}))
                                if len(user_input_str) > 100:
                                    user_input_str = user_input_str[:100] + "..."
                                display_str = user_input_str[:50] if len(user_input_str) > 50 else user_input_str
                                status_text.text(f"📝 User input: {display_str}")
                                event_info["input"] = user_input_str
                                event_info["quality"] = event.data.get("quality")
                                event_info["routing_mode"] = event.data.get("routing_mode", st.session_state.routing_mode)

                            elif event_type_str == "agent_selected":
                                reason = event.data.get("reason", "")
                                status_text.text(f"🎯 Orchestrator selected: {event.agent_id}")
                                event_info["selected_by"] = event.data.get("selected_by", "orchestrator")
                                event_info["reason"] = reason[:150] if reason else None
                                event_info["confidence"] = event.data.get("confidence")
                                event_info["model"] = event.data.get("model", "")
                                # RFC-024: Get agent prompt preview
                                agent_prompt = ""
                                for agent in st.session_state.agents:
                                    if agent.get("role") == event.agent_id:
                                        agent_prompt = agent.get("prompt", "")
                                        event_info["model"] = agent.get("model", "")
                                        break
                                if agent_prompt:
                                    event_info["prompt_preview"] = agent_prompt[:200]

                            elif event_type_str == "agent_started":
                                status_text.text(f"🔄 Agent: {event.agent_id}")
                                event_info["status"] = "started"

                            elif event_type_str == "agent_completed":
                                status_text.text(f"✅ Agent: {event.agent_id} completed")
                                output = str(event.data.get("output", ""))
                                event_info["output"] = output[:200]
                                event_info["output_length"] = len(output)
                                # RFC-024: Capture tokens and cost
                                event_info["tokens"] = event.data.get("tokens_used", event.data.get("tokens"))
                                event_info["cost"] = event.data.get("cost")
                                if event_info["tokens"]:
                                    total_tokens += event_info["tokens"]
                                if event_info["cost"]:
                                    total_cost += event_info["cost"]

                            elif event_type_str == "adaptive_decision":
                                method = event.data.get("method", "llm")
                                target = event.data.get("target", "unknown")
                                status_text.text(f"🔀 Decision: {method} → {target}")
                                event_info["method"] = method
                                event_info["target"] = target
                                event_info["decision_id"] = event.data.get("decision_id")
                                event_info["rule_matched"] = event.data.get("rule_matched")
                                event_info["reasoning"] = event.data.get("reasoning", "")[:100]
                                routing_decisions.append({
                                    "decision_id": event_info["decision_id"],
                                    "method": method,
                                    "target": target,
                                })

                            elif event_type_str == "tool_call":
                                status_text.text(f"🔧 Tool: {event.data.get('tool_name', 'unknown')}")
                                event_info["tool"] = event.data.get("tool_name")

                            elif event_type_str == "run_completed":
                                status_text.text("✅ Run completed")
                                event_info["success"] = True
                                event_info["duration"] = (datetime.now() - start_time).total_seconds()

                            elif event_type_str == "run_failed":
                                status_text.text("❌ Run failed")
                                event_info["error"] = str(event.data.get("error", ""))

                            elif event_type_str == "cost_update":
                                event_info["tokens"] = event.data.get("tokens")
                                event_info["cost"] = event.data.get("current_cost")
                                if event_info["tokens"]:
                                    total_tokens = event_info["tokens"]
                                if event_info["cost"]:
                                    total_cost = event_info["cost"]

                            elif event_type_str == "tool_result":
                                event_info["result"] = str(event.data.get("result", ""))[:100]

                            else:
                                # Handle any other event types gracefully
                                event_info["raw_data"] = str(event.data)[:200] if event.data else None

                            events_log.append(event_info)

                            # RFC-024: Format as timeline markdown
                            formatted = format_event_for_display(event_info, st.session_state.agents)
                            timeline_md.append(formatted)
                            timeline_md.append("---")
                            event_placeholder.markdown("\n\n".join(timeline_md))

                        progress_bar.progress(100)

                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        status_text.text(f"❌ Error: {e}")
                        events_log.append({
                            "type": "ERROR",
                            "error": str(e),
                            "traceback": error_details,
                        })
                        st.error(f"```\n{error_details}\n```")
                        return None
            else:
                status_text.text("Running...")
                progress_bar.progress(50)

                try:
                    result = await team.run(
                        input_data,
                        quality=quality,
                        importance=importance if importance != "normal" else None,
                    )
                    progress_bar.progress(100)

                    if result.success:
                        status_text.text("✅ Run completed")
                    else:
                        status_text.text(f"❌ Run failed: {result.error}")

                    return result

                except Exception as e:
                    status_text.text(f"❌ Error: {e}")
                    progress_bar.progress(100)
                    return None

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # RFC-024: Save to history with extended fields
            run_record = {
                "timestamp": start_time.isoformat(),
                "mode": "team",
                "duration_s": duration,
                "input": input_data,
                "quality": quality,
                "importance": importance,
                "routing_mode": st.session_state.routing_mode,
                "agents": [{"role": a["role"], "model": a.get("model", "")} for a in st.session_state.agents],
                "events": events_log,
                "success": True,  # If we got here, it was successful
                "routing_decisions": routing_decisions,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
            }
            st.session_state.run_history.append(run_record)

            return None

        # Run async
        result = asyncio.run(execute())

        # Display result
        if result:
            with result_container:
                st.subheader("Result")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", "✅ Success" if result.success else "❌ Failed")
                with col2:
                    st.metric("Agents Called", len(result.agents_called) if result.agents_called else 0)
                with col3:
                    st.metric("Iterations", result.iterations if hasattr(result, "iterations") else "-")

                if result.output:
                    st.subheader("Output")
                    if isinstance(result.output, dict):
                        st.json(result.output)
                    else:
                        st.write(result.output)

                if result.error:
                    st.error(f"Error: {result.error}")


# === History === (RFC-024: Improved timeline view)
def render_history():
    """Render run history with timeline view."""
    st.header("📜 Run History")

    if not st.session_state.run_history:
        st.info("No runs yet.")
        return

    # RFC-024: Summary metrics
    total_runs = len(st.session_state.run_history)
    successful_runs = sum(1 for r in st.session_state.run_history if r.get("success", True))
    total_cost = sum(r.get("total_cost", 0) for r in st.session_state.run_history)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        st.metric("Success Rate", f"{successful_runs}/{total_runs}")
    with col3:
        st.metric("Total Cost", f"${total_cost:.4f}")

    st.divider()

    # Run list
    for i, run in enumerate(reversed(st.session_state.run_history)):
        run_num = total_runs - i
        timestamp = run.get("timestamp", "")[:19]
        mode = run.get("mode", "team")
        success = run.get("success", True)
        mode_icon = RUN_MODE_ICONS.get(mode, "🤖")
        status_icon = "✅" if success else "❌"

        with st.expander(f"{status_icon} {mode_icon} Run #{run_num} — {timestamp}"):
            # Summary section
            st.markdown("### 📊 Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Mode:** {mode}")
                st.write(f"**Duration:** {run.get('duration_s', 0):.2f}s")
            with col2:
                st.write(f"**Routing:** {run.get('routing_mode', 'N/A')}")
                st.write(f"**Importance:** {run.get('importance', 'normal')}")
            with col3:
                cost = run.get("total_cost", 0)
                st.write(f"**Cost:** ${cost:.4f}" if cost else "**Cost:** N/A")
                st.write(f"**Events:** {len(run.get('events', []))}")
            with col4:
                st.write(f"**Status:** {'Success' if success else 'Failed'}")
                agents = run.get("agents", [])
                st.write(f"**Agents:** {len(agents)}")

            # User Input
            st.markdown("### 📝 User Input")
            input_data = run.get("input", {})
            input_text = input_data.get("query", input_data.get("input", str(input_data)))
            st.markdown(f"> {input_text[:300]}{'...' if len(str(input_text)) > 300 else ''}")

            # Agents
            if agents := run.get("agents"):
                st.markdown("### 🤖 Agents")
                agents_str = ", ".join([f"**{a['role']}** ({a.get('model', 'default')})" for a in agents])
                st.markdown(agents_str)

            # Events Timeline
            if events := run.get("events", []):
                st.markdown("### 📋 Events Timeline")

                for j, event in enumerate(events, 1):
                    event_type = event.get("type", "unknown").lower()
                    icon = EVENT_ICONS.get(event_type, "📌")
                    timestamp_str = event.get("timestamp", "")
                    if timestamp_str:
                        try:
                            timestamp_str = timestamp_str.split("T")[1][:8]
                        except Exception:
                            pass

                    # Event header
                    header = f"{j}. {icon} **{event_type.upper()}**"
                    if timestamp_str:
                        header += f" `{timestamp_str}`"
                    if event.get("agent_id"):
                        header += f" — **{event['agent_id']}**"
                    st.markdown(header)

                    # Event details based on type
                    if event_type == "agent_selected" and event.get("prompt_preview"):
                        with st.expander("View agent prompt"):
                            st.code(event["prompt_preview"], language="text")

                    if event_type == "agent_completed":
                        output = event.get("output", "")
                        if output:
                            st.markdown(f"> {output[:100]}{'...' if len(output) > 100 else ''}")
                        details = []
                        if event.get("tokens"):
                            details.append(f"Tokens: {event['tokens']}")
                        if event.get("cost"):
                            details.append(f"Cost: ${event['cost']:.4f}")
                        if details:
                            st.caption(" | ".join(details))

                    if event_type == "adaptive_decision":
                        method = event.get("method", "unknown")
                        target = event.get("target", "unknown")
                        method_icon = "📏" if method == "rule" else "🧠"
                        st.markdown(f"{method_icon} {method.upper()} → **{target}**")

            # Routing Decisions
            if routing := run.get("routing_decisions"):
                st.markdown("### 🔀 Routing Decisions")
                for dec in routing:
                    method = dec.get("method", "unknown")
                    target = dec.get("target", "unknown")
                    method_icon = "📏" if method == "rule" else "🧠"
                    st.markdown(f"- {method_icon} **{dec.get('decision_id', 'N/A')}**: {target} ({method})")

            # Raw JSON toggle
            with st.expander("🔍 Raw JSON"):
                st.json(run)

    st.divider()

    # Actions
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear History"):
            st.session_state.run_history = []
            st.rerun()
    with col_export:
        if st.button("📥 Export History"):
            st.download_button(
                "Download JSON",
                data=json.dumps(st.session_state.run_history, indent=2, ensure_ascii=False),
                file_name="run_history.json",
                mime="application/json",
            )


# === Quality Info ===
def get_preset_name(quality: int) -> str:
    """Get preset name from quality value."""
    if quality <= 20:
        return "draft"
    elif quality <= 30:
        return "economy"
    elif quality <= 60:
        return "balanced"
    elif quality <= 80:
        return "production"
    else:
        return "best"


def render_quality_info(settings: Dict[str, Any]):
    """Render quality information."""
    st.header("📊 Quality Info")

    manager = QualityManager(settings["quality"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Quality Level", settings["quality"])
        st.write(f"**Preset:** {get_preset_name(settings['quality'])}")

    with col2:
        params = manager.get_generation_params()
        st.metric("Temperature", f"{params['temperature']:.1f}")
        st.write(f"**Max Tokens:** {params['max_tokens']}")

    with col3:
        st.write("**Models by complexity:**")
        st.write(f"- Simple: {manager.get_model('simple')}")
        st.write(f"- Medium: {manager.get_model('medium')}")
        st.write(f"- Complex: {manager.get_model('complex')}")


# === RFC-022: Task Solver Tab ===
def render_task_solver(settings: Dict[str, Any]):
    """Render Task Solver tab - L1 API for one-call task solving."""
    st.header("Task Solver")
    st.markdown("""
    **L1 API**: Describe your task and LLMTeam will automatically create a team and solve it.
    """)

    if not st.session_state.api_key:
        st.warning("Set OpenAI API Key in the sidebar.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        task = st.text_area(
            "Describe your task",
            placeholder="Write an article about AI in medicine, 2000 words, professional tone...",
            height=150,
            key="task_solver_input",
        )

        constraints_text = st.text_input(
            "Constraints (optional)",
            placeholder="language: english, tone: professional, max length: 2000 words",
            key="task_solver_constraints",
        )

    with col2:
        st.markdown("**Settings:**")
        quality = st.slider("Quality", 0, 100, settings["quality"], key="solver_quality")
        max_cost = st.number_input("Max Cost ($)", 0.1, 10.0, 1.0, key="solver_max_cost")
        routing_mode = st.selectbox(
            "Routing Mode",
            ["hybrid", "sequential", "dynamic"],
            index=["hybrid", "sequential", "dynamic"].index(settings["routing_mode"]),
            key="solver_routing_mode",
        )

    if st.button("Solve Task", type="primary", use_container_width=True):
        if not task:
            st.error("Please describe your task")
            return

        # Parse constraints
        constraints = {}
        if constraints_text:
            for item in constraints_text.split(","):
                if ":" in item:
                    key, value = item.split(":", 1)
                    constraints[key.strip()] = value.strip()

        with st.spinner("Solving task..."):
            result_container = st.empty()

            async def solve():
                try:
                    result = await LLMTeam.solve(
                        task=task,
                        quality=quality,
                        constraints=constraints,
                        max_cost=max_cost,
                        routing_mode=routing_mode,
                    )
                    return result
                except Exception as e:
                    return {"error": str(e)}

            result = asyncio.run(solve())

            with result_container.container():
                st.subheader("Result")

                if isinstance(result, dict) and "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status = "Success" if result.success else "Failed"
                        st.metric("Status", status)
                    with col2:
                        cost = f"${result.cost:.4f}" if hasattr(result, "cost") and result.cost else "N/A"
                        st.metric("Cost", cost)
                    with col3:
                        duration = f"{result.duration:.1f}s" if hasattr(result, "duration") else "N/A"
                        st.metric("Duration", duration)

                    st.subheader("Output")
                    if result.output:
                        if isinstance(result.output, dict):
                            st.json(result.output)
                        else:
                            st.write(result.output)

                    # Show routing decisions if any
                    if hasattr(result, "routing_decisions") and result.routing_decisions:
                        with st.expander("Routing Decisions"):
                            for decision in result.routing_decisions:
                                method_icon = "rule" if decision.get("method") == "rule" else "brain"
                                st.markdown(f"- **{decision.get('decision_id', 'unknown')}**: {decision.get('target')} ({method_icon})")


# === RFC-023: Interactive Session Tab ===
def render_interactive_session(settings: Dict[str, Any]):
    """Render Interactive Session tab - L1/L2 API for Q&A."""
    st.header("Interactive Session")
    st.markdown("""
    **L1 Interactive**: Have a conversation to clarify your task before execution.
    """)

    if not st.session_state.api_key:
        st.warning("Set OpenAI API Key in the sidebar.")
        return

    # Session state display
    session = st.session_state.interactive_session

    col1, col2 = st.columns([3, 1])

    with col2:
        if session:
            st.markdown("**Session State:**")
            state_colors = {
                "idle": "gray",
                "gathering_info": "blue",
                "routing_config": "yellow",
                "proposing": "green",
                "ready": "green",
                "executing": "orange",
                "completed": "green",
                "failed": "red",
            }
            state = session.state.value
            st.markdown(f":{state_colors.get(state, 'gray')}[{state}]")

            st.markdown("**Quality:**")
            st.write(session.quality)

            st.markdown("**Routing:**")
            st.write(session.routing_mode)

            if st.button("Reset Session"):
                st.session_state.interactive_session = None
                st.session_state.session_messages = []
                st.rerun()

    with col1:
        # Chat interface
        chat_container = st.container()

        with chat_container:
            # Show message history
            for msg in st.session_state.session_messages:
                role = msg["role"]
                content = msg["content"]
                with st.chat_message(role):
                    st.markdown(content)

            # Show current question if any
            if session and session.question:
                with st.chat_message("assistant"):
                    st.markdown(session.question)
                    if session.question_options:
                        st.markdown("**Options:** " + ", ".join(session.question_options))

            # Show proposal if in proposing state
            if session and session.state == InteractiveState.PROPOSING and session.plan:
                with st.chat_message("assistant"):
                    st.markdown("**Proposed Team:**")
                    st.code(session.plan)

        # Input area
        if not session:
            # Start new session
            new_task = st.text_input("Describe your task to start a session", key="new_session_task")
            col_start, col_quality = st.columns([2, 1])
            with col_quality:
                session_quality = st.slider("Quality", 0, 100, settings["quality"], key="session_quality")

            if st.button("Start Session", type="primary"):
                if new_task:
                    with st.spinner("Starting session..."):
                        async def start():
                            s = InteractiveSession(
                                task=new_task,
                                quality=session_quality,
                                routing_mode=settings["routing_mode"],
                            )
                            await s._start()
                            return s

                        st.session_state.interactive_session = asyncio.run(start())
                        st.session_state.session_messages = [{"role": "user", "content": new_task}]
                        st.rerun()

        elif session.state in [InteractiveState.GATHERING_INFO, InteractiveState.ROUTING_CONFIG]:
            # Answer question
            answer = st.text_input("Your answer", key="session_answer")
            if st.button("Send"):
                if answer:
                    with st.spinner("Processing..."):
                        async def answer_q():
                            await session.answer(answer)

                        asyncio.run(answer_q())
                        st.session_state.session_messages.append({"role": "user", "content": answer})
                        st.rerun()

        elif session.state == InteractiveState.PROPOSING:
            # Adjust or confirm
            col_adj, col_conf = st.columns(2)
            with col_adj:
                adjustment = st.text_input("Adjustment (optional)", key="session_adjustment")
                if st.button("Adjust"):
                    if adjustment:
                        with st.spinner("Adjusting..."):
                            async def adjust():
                                await session.adjust(adjustment)

                            asyncio.run(adjust())
                            st.session_state.session_messages.append({"role": "user", "content": adjustment})
                            st.rerun()

            with col_conf:
                if st.button("Confirm & Execute", type="primary"):
                    with st.spinner("Executing..."):
                        async def execute():
                            result = await session.execute()
                            return result

                        result = asyncio.run(execute())

                        st.subheader("Result")
                        if result.success:
                            st.success("Execution completed!")
                            if result.output:
                                st.write(result.output)
                        else:
                            st.error(f"Failed: {result.error}")

        elif session.state == InteractiveState.COMPLETED:
            st.success("Session completed!")
            if session.result:
                st.write(session.result.output)


# === Main App ===
def main():
    """Main application entry point."""
    st.title("🤖 LLMTeam Playground v6.1.0")
    st.markdown("Interactive testing interface for LLMTeam library - RFC-022/RFC-023/RFC-024")

    # Sidebar
    settings = render_sidebar()

    # Main tabs - RFC-024 adds Workflow tab
    tabs = st.tabs([
        "🎯 Task Solver",
        "💬 Interactive",
        "🤖 Agents",
        "🔀 Workflow",
        "▶️ Run",
        "📜 History",
        "📊 Info"
    ])

    with tabs[0]:
        render_task_solver(settings)

    with tabs[1]:
        render_interactive_session(settings)

    with tabs[2]:
        render_agent_builder()

    with tabs[3]:
        render_workflow_designer()

    with tabs[4]:
        render_team_runner(settings)

    with tabs[5]:
        render_history()

    with tabs[6]:
        render_quality_info(settings)


if __name__ == "__main__":
    main()
