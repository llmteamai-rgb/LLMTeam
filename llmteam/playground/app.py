"""
LLMTeam Playground - Streamlit App.

Interactive interface for testing LLMTeam library.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from llmteam import LLMTeam
from llmteam.agents.orchestrator import OrchestratorMode
from llmteam.quality import QualityManager
from llmteam.events.streaming import StreamEventType


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
        st.session_state.api_key = os.environ.get("OPENAI_API_KEY", "")


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
            "orchestration": orchestration,
            "max_cost_per_run": max_cost,
            "agents": st.session_state.agents,
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
            st.sidebar.success("Config loaded!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    return {
        "team_id": team_id,
        "quality": quality,
        "orchestration": orchestration,
        "max_cost_per_run": max_cost,
    }


# === Agent Builder ===
def render_agent_builder():
    """Render agent builder section."""
    st.header("🤖 Agent Builder")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add New Agent")

        agent_type = st.selectbox(
            "Agent Type",
            options=["llm", "rag", "kag"],
            index=0,
        )

        role = st.text_input("Role", placeholder="e.g., researcher, writer, reviewer")

        prompt = st.text_area(
            "System Prompt",
            placeholder="You are a helpful assistant. Your task is to {task}...",
            height=150,
        )

        model = st.selectbox(
            "Model",
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

        # Tools selection
        available_tools = ["web_search", "http_fetch", "json_extract", "text_summarize", "code_eval"]
        tools = st.multiselect("Tools", options=available_tools)

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
                if tools:
                    agent_config["tools"] = tools

                st.session_state.agents.append(agent_config)
                st.success(f"Added agent: {role}")
                st.rerun()

    with col2:
        st.subheader("Current Agents")

        if not st.session_state.agents:
            st.info("No agents yet. Add one using the form.")
        else:
            for i, agent in enumerate(st.session_state.agents):
                with st.expander(f"**{agent['role']}** ({agent['type']})", expanded=False):
                    st.code(json.dumps(agent, indent=2), language="json")

                    col_edit, col_del = st.columns(2)
                    with col_del:
                        if st.button(f"🗑️ Delete", key=f"del_{i}"):
                            st.session_state.agents.pop(i)
                            st.rerun()


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

            if stream:
                status_text.text("Starting stream...")

                with events_container:
                    event_placeholder = st.empty()

                    try:
                        i = 0
                        async for event in team.stream(
                            input_data,
                            quality=quality,
                            importance=importance if importance != "normal" else None,
                        ):
                            i += 1
                            progress_bar.progress(min(i * 10, 100))

                            event_info = {
                                "type": event.type.name,
                                "agent_id": event.agent_id,
                                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                            }

                            if event.type == StreamEventType.AGENT_STARTED:
                                status_text.text(f"🔄 Agent: {event.agent_id}")
                                event_info["status"] = "started"
                            elif event.type == StreamEventType.AGENT_COMPLETED:
                                status_text.text(f"✅ Agent: {event.agent_id} completed")
                                event_info["output"] = str(event.data.get("output", ""))[:200]
                            elif event.type == StreamEventType.TOOL_CALL:
                                status_text.text(f"🔧 Tool: {event.data.get('tool_name', 'unknown')}")
                                event_info["tool"] = event.data.get("tool_name")
                            elif event.type == StreamEventType.RUN_COMPLETED:
                                status_text.text("✅ Run completed")
                                event_info["success"] = True
                            elif event.type == StreamEventType.RUN_FAILED:
                                status_text.text("❌ Run failed")
                                event_info["error"] = str(event.data.get("error", ""))

                            events_log.append(event_info)
                            event_placeholder.json(events_log)

                        progress_bar.progress(100)

                    except Exception as e:
                        status_text.text(f"❌ Error: {e}")
                        events_log.append({"type": "ERROR", "error": str(e)})
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

            # Save to history
            run_record = {
                "timestamp": start_time.isoformat(),
                "duration_s": duration,
                "input": input_data,
                "quality": quality,
                "importance": importance,
                "events": events_log,
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


# === History ===
def render_history():
    """Render run history."""
    st.header("📜 Run History")

    if not st.session_state.run_history:
        st.info("No runs yet.")
        return

    for i, run in enumerate(reversed(st.session_state.run_history)):
        with st.expander(f"Run {len(st.session_state.run_history) - i} - {run['timestamp'][:19]}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Duration:** {run['duration_s']:.2f}s")
                st.write(f"**Quality:** {run['quality']}")
            with col2:
                st.write(f"**Importance:** {run['importance']}")
                st.write(f"**Events:** {len(run['events'])}")

            st.json(run["input"])

    if st.button("🗑️ Clear History"):
        st.session_state.run_history = []
        st.rerun()


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


# === Main App ===
def main():
    """Main application entry point."""
    st.title("🎮 LLMTeam Playground")
    st.markdown("Interactive testing interface for LLMTeam library")

    # Sidebar
    settings = render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Agents", "▶️ Run", "📜 History", "📊 Quality"])

    with tab1:
        render_agent_builder()

    with tab2:
        render_team_runner(settings)

    with tab3:
        render_history()

    with tab4:
        render_quality_info(settings)


if __name__ == "__main__":
    main()
