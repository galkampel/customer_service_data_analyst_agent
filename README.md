# Customer Service Data Analyst Agent

LangGraph-based customer service dataset analyst with:
- Task 1: router + tools + ReAct loop + reasoning trace
- Task 2: persistent episodic memory and user profile memory
- Task 3: FastMCP server for dataset-analysis tools
- Bonus B: query recommender with suggest/refine/confirm flow

## Quick Start

1. Sync environment and activate venv:

```bash
uv sync
source .venv/bin/activate
```

2. Configure environment variables in `.env`:

```bash
NEBIUS_API_KEY=<your_key>
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
NEBIUS_MAIN_MODEL=meta-llama/Llama-3.3-70B-Instruct
NEBIUS_ROUTER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
NEBIUS_PROFILE_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
NEBIUS_RECOMMENDER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
```

The main agent can use a stronger model, while router, profile extraction, and
query recommendation can use smaller/faster models because they produce short,
low-complexity outputs.

## CLI Usage

### Interactive mode

```bash
python main.py --session alice
```

Interactive commands:
- `profile` -> show stored profile for current session
- `sessions` -> list all persisted sessions
- `recommend` / `suggest` -> ask for a follow-up query recommendation
- `exit` / `quit` -> stop

### Single query mode

```bash
python main.py --session alice --query "Summarize the FEEDBACK category."
```

### List stored sessions

```bash
python main.py --list-sessions
```

## Query Recommender (Bonus B)

The recommender suggests one concrete next dataset query and pauses. It never
executes a recommendation until you explicitly confirm it.

Pending suggestions are stored per session under local `memory/` state, so the
flow works across separate single-query CLI invocations with the same
`--session`.

```bash
python main.py --session demo --query "What should I query next?"
python main.py --session demo --query "make it about refunds"
python main.py --session demo --query "yes, run it"
```

In interactive mode, you can also type `recommend` or `suggest` to start the
same flow. Reply `no` or `cancel` to clear a pending recommendation.

Validated flow:
- suggest: no dataset tools are called
- refine: updates the pending suggestion without execution
- confirm: executes the pending suggestion through the normal tool loop
- decline: clears the pending suggestion so a later `yes` cannot run it

## Memory Behavior (Task 2)

### Episodic memory
- Conversation checkpoints are stored in `memory/checkpoints.db`.
- Reusing the same `--session` restores thread state.
- Runtime memory files are local artifacts and should not be committed.

### Regression note
- Unstructured summarization guards are scoped to the current user turn so
	older checkpointed tool observations do not trigger premature finalization.

### User profile memory
- Profiles are stored at `memory/profiles/<session_id>.json`.
- Durable facts are extracted from latest user+assistant exchange.
- Ask: `What do you remember about me?` to read memory-backed profile facts.

## Validation Commands

Run a quick smoke check:

```bash
uv sync
source .venv/bin/activate
python -m py_compile agent.py memory.py main.py tools.py
python -m py_compile recommender.py llm_config.py
python main.py --session t2 --query "I prefer concise answers with examples."
python main.py --session t2 --query "What do you remember about me?"
python main.py --session rec-demo --query "What should I query next?"
python main.py --session rec-demo --query "make it about refund examples"
python main.py --session rec-demo --query "yes, run it"
python main.py --list-sessions
```

## MCP Server (Task 3)

FastMCP is a Python framework that exposes normal Python functions as MCP
tools over standard transports (here: `streamable-http`). It lets MCP
clients call your dataset-analysis functions in a structured, tool-first way.

### Start MCP server

```bash
uv sync
source .venv/bin/activate
python mcp_server.py
```

Server settings:
- host: `0.0.0.0`
- port: `8000`
- transport: `streamable-http`

### Exposed MCP tools

- `list_categories`
- `get_intent_distribution`
- `show_examples`
- `count_rows_for_intent`
- `summarize_category`
- `search_responses`

### Phase D smoke validation (local)

```bash
uv sync
source .venv/bin/activate
python -m py_compile mcp_server.py
python - <<'PY'
import mcp_server

print(mcp_server.list_categories().splitlines()[0])
print(mcp_server.get_intent_distribution(category="ACCOUNT").splitlines()[0])
print(mcp_server.show_examples(n=2, category="REFUND").splitlines()[0])
print(mcp_server.count_rows_for_intent("get_refund"))
print(mcp_server.search_responses("refund", n=2).splitlines()[0])
PY
```

### Client invocation example (Python MCP client)

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main() -> None:
	async with streamablehttp_client("http://localhost:8000/mcp") as (
		read_stream,
		write_stream,
		_,
	):
		async with ClientSession(read_stream, write_stream) as session:
			await session.initialize()
			result = await session.call_tool("list_categories", {})
			print(result.content[0].text)


asyncio.run(main())
```
