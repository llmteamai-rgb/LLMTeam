# LLMTeam Playground

Interactive testing interface for the LLMTeam library.

## Installation

```bash
pip install llmteam-ai[playground]
# or
pip install streamlit
```

## Usage

### Run directly

```bash
cd llmteam
streamlit run playground/app.py
```

### Run as module

```bash
cd llmteam
python -m playground
```

### With PYTHONPATH (development)

```bash
cd llmteam
$env:PYTHONPATH="src"  # PowerShell
# or
export PYTHONPATH=src  # Bash

streamlit run playground/app.py
```

## Features

### 🤖 Agent Builder

- Create LLM/RAG/KAG agents
- Configure role, prompt, model, temperature
- Add tools (web_search, http_fetch, etc.)
- Edit and delete agents

### ▶️ Team Runner

- Run teams with custom input
- Simple text or JSON input mode
- Quality override per run
- Importance levels (low, normal, high, critical)
- Streaming mode with live events

### 📜 Run History

- View previous runs
- Input, events, duration
- Clear history

### 📊 Quality Info

- Current quality settings
- Model selection by complexity
- Generation parameters

### ⚙️ Settings (Sidebar)

- OpenAI API Key
- Team ID
- Quality slider (0-100)
- Quality presets (draft, economy, balanced, production, best)
- Router mode toggle
- Max cost per run
- Export/Import config

## Screenshots

```
┌─────────────────────────────────────────────────────────────────┐
│  🎮 LLMTeam Playground                                          │
│  Interactive testing interface for LLMTeam library              │
├─────────────────────────────────────────────────────────────────┤
│  [🤖 Agents] [▶️ Run] [📜 History] [📊 Quality]                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🤖 Agent Builder                                               │
│  ┌─────────────────────┬─────────────────────────┐              │
│  │ Add New Agent       │ Current Agents          │              │
│  │                     │                         │              │
│  │ Type: [llm ▼]       │ ▶ researcher (llm)      │              │
│  │ Role: [_________]   │ ▶ writer (llm)          │              │
│  │ Prompt:             │ ▶ reviewer (llm)        │              │
│  │ [_______________]   │                         │              │
│  │ Model: [gpt-4o-mini]│                         │              │
│  │ Temp: [====○====]   │                         │              │
│  │ Tools: [○ web_search│                         │              │
│  │        [○ http_fetch│                         │              │
│  │ [➕ Add Agent]      │                         │              │
│  └─────────────────────┴─────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Config Format

Export produces JSON:

```json
{
  "team_id": "playground-team",
  "quality": 70,
  "orchestration": true,
  "max_cost_per_run": 1.0,
  "agents": [
    {
      "type": "llm",
      "role": "researcher",
      "prompt": "You are a researcher...",
      "model": "gpt-4o-mini",
      "temperature": 0.7,
      "max_tokens": 1000,
      "tools": ["web_search"]
    }
  ]
}
```

## Environment Variables

- `OPENAI_API_KEY` - Pre-fill API key in sidebar
