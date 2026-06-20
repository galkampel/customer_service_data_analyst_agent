# Customer Service Data Analyst Agent

LangGraph-based data analyst agent for the Bitext Customer Service dataset.

Implemented scope:
- Task 1: router + ReAct tool loop + CLI reasoning trace + max-iteration guard
- Task 2: persistent episodic memory + persistent user profile memory
- Task 3: FastMCP server exposing dataset analysis tools
- Streamlit chat UI with session switch, reasoning trace, and query recommendation flow

## 5-Minute Setup

This repository uses `pyproject.toml` with pinned/explicit dependency versions.

1. Install/sync dependencies and activate virtual environment:

```bash
uv sync
source .venv/bin/activate
```

2. Create `.env` in project root:

```bash
NEBIUS_API_KEY=<your_key>
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
NEBIUS_MAIN_MODEL=meta-llama/Llama-3.3-70B-Instruct
NEBIUS_ROUTER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
NEBIUS_PROFILE_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
NEBIUS_RECOMMENDER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
```

3. Quick syntax gate:

```bash
python -m py_compile agent.py memory.py main.py tools.py recommender.py llm_config.py mcp_server.py streamlit_app.py
```

## Model Choice and Rationale

All LLM calls use Nebius Token Factory models.

Role mapping:
- Main agent (`NEBIUS_MAIN_MODEL`): `meta-llama/Llama-3.3-70B-Instruct`
- Router (`NEBIUS_ROUTER_MODEL`): `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`
- Profile extractor (`NEBIUS_PROFILE_MODEL`): `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`
- Query recommender (`NEBIUS_RECOMMENDER_MODEL`): `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`
- Fallback (`NEBIUS_MODEL`): `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`

Why this split:
- Strong model on the main reasoning/tool-use loop improves answer quality.
- Smaller/faster models on routing/profile/recommendation reduce latency and cost
	for high-frequency, low-complexity subtasks.

## Architecture Overview

Core modules:
- `data_loader.py`: dataset load/normalize/cache
- `tools.py`: typed analysis tools with Pydantic schemas
- `agent.py`: LangGraph router + agent/tool loop + recommender branch
- `memory.py`: SQLite checkpoint memory + JSON profile/pending state
- `recommender.py`: suggest/refine/confirm/decline helpers
- `main.py`: CLI entrypoint
- `mcp_server.py`: FastMCP server exposing tool subset
- `streamlit_app.py`: chat UI wrapper with reasoning trace

Graph behavior:
- Router classifies each turn into `structured`, `unstructured`,
	`out_of_scope`, or `recommender`.
- Structured/unstructured route through ReAct tool loop.
- Out-of-scope returns a polite dataset-only refusal.
- Recommender suggests a query and executes only after user confirmation.

## Tool Definitions

Agent tools (from `tools.py`):
- `list_categories`: categories with row counts
- `list_intents(category)`: intents in one category
- `count_rows()`: row count on working set/full dataset
- `filter_by_intent(intent)`: set working set for downstream steps
- `filter_by_category(category)`: set working set for downstream steps
- `show_examples(n, category?, intent?)`: sample rows
- `search_responses(keyword, n, use_working_set)`: literal keyword search
- `get_intent_distribution(category?)`: counts + percentages
- `summarize_category(category, n_examples)`: compact category summary

Tools are exposed as `StructuredTool` objects with clear descriptions and
Pydantic input schemas to improve model tool selection.

## CLI Usage

Interactive mode:

```bash
python main.py --session alice
```

Interactive commands:
- `profile`: show stored profile for current session
- `sessions`: list stored sessions
- `recommend` / `suggest`: request a follow-up query suggestion
- `exit` / `quit`: stop

Single query mode:

```bash
python main.py --session alice --query "Summarize the FEEDBACK category."
```

List sessions:

```bash
python main.py --list-sessions
```

## Memory Behavior

Episodic memory:
- Conversation checkpoints: `memory/checkpoints.db`
- Same `--session` restores same conversation state across restarts

User profile memory:
- Profile files: `memory/profiles/<session_id>.json`
- Read with: `What do you remember about me?`

Recommendation pending state:
- Pending recommendation files:
	`memory/pending_recommendations/<session_id>.json`

## Query Recommendation Flow

The agent supports suggest -> refine -> confirm -> execute.

Example:

```bash
python main.py --session demo --query "What should I query next?"
python main.py --session demo --query "make it about refund examples"
python main.py --session demo --query "yes, run it"
```

Behavior guarantees:
- Suggestion/refinement does not auto-execute dataset tools.
- Decline clears pending state.
- A later `yes` does not execute stale cancelled suggestions.

## Streamlit Chat UI

Run:

```bash
streamlit run streamlit_app.py
```

Implemented UI features:
- Sidebar API key input
- Session input + existing-session switch
- Profile panel
- Chat transcript with reasoning trace expanders
- Recommendation convenience button
- Session transcript rehydration from checkpoints
- Error handling for graph init/query failures

Headless startup check:

```bash
streamlit run streamlit_app.py --server.headless true --server.port 8502
```

## MCP Server

Run server:

```bash
python mcp_server.py
```

Settings:
- Host: `0.0.0.0`
- Port: `8000`
- Transport: `streamable-http`
- Endpoint: `http://localhost:8000/mcp`

Exposed MCP tools:
- `list_categories`
- `get_intent_distribution`
- `show_examples`
- `count_rows_for_intent`
- `summarize_category`
- `search_responses`

Client connection example (call one tool):

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

## Smoke Test Bundle

```bash
source .venv/bin/activate

# Task 1
python main.py --session t1-structured --query "How many rows are in REFUND?"
python main.py --session t1-unstructured --query "Summarize the FEEDBACK category."
python main.py --session t1-oos --query "Who won the 2024 Champions League?"

# Task 2
python main.py --session t2-memory --query "I prefer concise answers with examples."
python main.py --session t2-memory --query "What do you remember about me?"

# Task 3
python -m py_compile mcp_server.py
python - <<'PY'
import mcp_server
print(mcp_server.list_categories().splitlines()[0])
print(mcp_server.get_intent_distribution(category="ACCOUNT").splitlines()[0])
PY

# UI startup
streamlit run streamlit_app.py --server.headless true --server.port 8502
```
