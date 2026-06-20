# Task 1 Instructions

This guide explains how to run and verify Task 1 (router + tools + ReAct loop + CLI reasoning trace + max-iteration fallback), and how to inspect the graph in LangGraph Studio.

## 1) Prerequisites

1. Activate your virtual environment:

```bash
source .venv/bin/activate
```

2. Ensure required packages are installed:

```bash
pip install -e .
```

3. Configure Nebius credentials and model variables in a local `.env` file
   at project root (Option B split-model setup):

```bash
cat > .env << 'EOF'
NEBIUS_API_KEY=<your_key>
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/

# Fallback model (used if role-specific vars are missing)
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast

# Option B role split
NEBIUS_MAIN_MODEL=meta-llama/Llama-3.3-70B-Instruct
NEBIUS_ROUTER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
EOF
```

The app auto-loads `.env` via `load_dotenv()`, so no `export` commands are required.

Notes:
- Main reasoning/tool-use calls go to NEBIUS_MAIN_MODEL.
- Query classification (router) goes to NEBIUS_ROUTER_MODEL.
- If role-specific variables are not set, the app falls back to NEBIUS_MODEL.

## 2) Run Task 1 in CLI

### Interactive mode

```bash
python main.py
```

What you should see:
- A prompt where you can type queries.
- Tool calls and observations printed before the final answer.
- This confirms reasoning transparency required by Task 1.

### Single-query mode

```bash
python main.py --query "How many refund requests did we get?"
```

### Verbose mode

```bash
python main.py --verbose
```

## 3) Recommended Task 1 Validation Queries

Run these in interactive mode and confirm behavior:

Structured:
- What categories exist in the dataset?
- How many refund requests did we get?
- Show me 5 examples from the SHIPPING category.
- What is the distribution of intents in the ACCOUNT category?

Unstructured:
- Summarize the FEEDBACK category.
- How do customer service representatives typically respond to cancellation requests?

Out-of-scope:
- Who won the 2024 Champions League?
- Write me a poem about customer service.

Expected behavior:
- Structured/unstructured queries use tools.
- Unstructured summaries should gather evidence quickly (typically one tool
   call) and then produce a final synthesis without looping.
- Out-of-scope queries are declined politely and should not be answered from general knowledge.

## 4) What to check for Task 1 grading

1. Router behavior
- Query is classified before tool selection.
- Out-of-scope is declined.

2. Tool quality
- Tools have clear names and descriptions.
- Inputs are validated by Pydantic schemas.
- Keyword search uses literal matching and can search current filtered working
   set for faster follow-up analysis.

3. Multi-step reasoning
- At least one chain like filter_by_intent -> count_rows appears in trace for counting intents.

4. CLI reasoning visibility
- Tool calls and tool observations are printed before final response.

5. Max-iteration fallback
- If the loop exceeds max iterations, a graceful fallback answer is returned.

## 5) LangGraph Studio Usage

Your graph is configured in langgraph.json under:
- customer_service_agent -> ./agent.py:graph

### Start Studio/dev server

From project root:

```bash
langgraph dev
```

If your shell cannot find the command, use:

```bash
python -m langgraph_cli dev
```

If you use uv tooling:

```bash
uv run langgraph dev
```

Open the URL printed in terminal (typically local web UI) and load the customer_service_agent graph.

### Studio workflow for Task 1

1. Start a new run/thread.
2. Send a structured query (for example: How many refund requests did we get?).
3. Inspect node execution order:
   - START -> router -> agent -> tools -> agent -> END
4. Open router output and confirm query_type classification.
5. Open agent/tool events and confirm:
   - selected tool
   - tool arguments
   - tool observation returned to agent
6. Repeat with out-of-scope query and confirm:
   - router classifies out_of_scope
   - out_of_scope response path is used
   - no irrelevant knowledge answer

### Studio troubleshooting

If graph does not initialize:
1. Ensure `.env` exists in project root and includes NEBIUS_API_KEY.
2. Ensure dependencies are installed in the active environment.
3. Re-run:

```bash
python -c "import agent; print('graph initialized:', agent.graph is not None)"
```

If output is False, environment variables are missing or model client initialization failed.

## 6) Quick smoke checks

Dataset + tools:

```bash
python -c "from data_loader import load_dataset; from tools import build_tools; print('rows=', len(load_dataset()), 'tools=', len(build_tools()))"
```

Expected (approx):
- rows = 26872
- tools = 9

## 7) Current Task 1 files

- agent.py
- tools.py
- data_loader.py
- llm_config.py
- main.py
- langgraph.json
- .env.example

These are the core files used to run and inspect Task 1.
