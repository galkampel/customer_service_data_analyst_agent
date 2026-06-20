# Customer Service Data Analyst Agent

LangGraph-based customer service dataset analyst with:
- Task 1: router + tools + ReAct loop + reasoning trace
- Task 2: persistent episodic memory and user profile memory

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
```

## CLI Usage

### Interactive mode

```bash
python main.py --session alice
```

Interactive commands:
- `profile` -> show stored profile for current session
- `sessions` -> list all persisted sessions
- `exit` / `quit` -> stop

### Single query mode

```bash
python main.py --session alice --query "Summarize the FEEDBACK category."
```

### List stored sessions

```bash
python main.py --list-sessions
```

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
python main.py --session t2 --query "I prefer concise answers with examples."
python main.py --session t2 --query "What do you remember about me?"
python main.py --list-sessions
```
