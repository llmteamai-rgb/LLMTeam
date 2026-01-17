# FastAPI Server Example

REST API server for running LLMTeam segments.

## Features

- REST API for segment execution
- WebSocket for real-time events
- Segment validation endpoint
- Health check

## Installation

```bash
pip install llmteam-ai[api,providers]
```

## Run Server

```bash
export OPENAI_API_KEY=sk-your-key
uvicorn main:app --reload
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Validate Segment
```bash
POST /segments/validate
Content-Type: application/json

{
  "segment_id": "test",
  "name": "Test",
  "entrypoint": "start",
  "steps": [...]
}
```

### Run Segment
```bash
POST /segments/run
Content-Type: application/json

{
  "segment": {...},
  "input_data": {"query": "Hello"}
}
```

### WebSocket Events
```javascript
ws = new WebSocket("ws://localhost:8000/ws/events")
ws.onmessage = (event) => console.log(JSON.parse(event.data))
```

## Files

- `main.py` - FastAPI application
- `requirements.txt` - Python dependencies
