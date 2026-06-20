# Task 3 Detailed Implementation Guide: MCP Server (FastMCP)

## Goal
Implement Task 3 by adding an MCP server that exposes customer-service analysis tools in a safe, stateless way.

Task 3 acceptance from your plan:
- FastMCP server runs successfully.
- Exposes at least 3 tools.
- README contains MCP startup + client invocation example.

---

## What to build
Create a new root file:
- `mcp_server.py`

Expose these MCP tools at minimum:
1. `list_categories`
2. `get_intent_distribution`
3. `show_examples`

Recommended additional MCP tools:
1. `count_rows_for_intent` (stateless wrapper)
2. `summarize_category`
3. `search_responses`

Important design rule:
- Keep MCP tools stateless. Do not depend on in-process filter state across calls.

---

## Why stateless wrappers are required
Your current `tools.py` supports chainable state (`_STATE['filtered_df']`) for agent ReAct loops. This is fine for a single in-process agent turn, but MCP clients are often concurrent and request-by-request.

If MCP tools rely on shared mutable state, callers can interfere with each other.

Best practice for Task 3:
- Reuse pure read/query functions directly.
- For operations that currently rely on filter state, create MCP-specific stateless wrappers that compute results from full dataset each call.

---

## Phase A: Add MCP dependency and verify environment
1. Ensure `fastmcp` is in dependencies.
2. Run:

```bash
uv sync
source .venv/bin/activate
python -c "import fastmcp; print('fastmcp ok')"
```

If import fails, resolve dependencies before writing server code.

---

## Phase B: Design MCP surface

### Required public tool contracts
Define clear, stable schemas and return strings (or JSON-like dicts) suitable for MCP clients.

1. `list_categories()`
- Input: none
- Output: category list with counts
- Backing logic: `tools.list_categories`

2. `get_intent_distribution(category: str | None = None)`
- Input: optional category
- Output: intent counts + percentages
- Backing logic: `tools.get_intent_distribution`

3. `show_examples(n: int = 5, category: str | None = None, intent: str | None = None)`
- Input: bounded `n`, optional category/intent
- Output: examples text
- Backing logic: prefer stateless wrapper in MCP layer to avoid shared filter coupling

### Recommended extra contracts
4. `count_rows_for_intent(intent: str)`
- Stateless, one-shot helper for external clients.
- Internally compute direct count from full dataset.

5. `summarize_category(category: str, n_examples: int = 3)`
- Backing logic: `tools.summarize_category`

6. `search_responses(keyword: str, n: int = 5)`
- For MCP, call with `use_working_set=False` explicitly to guarantee stateless full-dataset behavior.

---

## Phase C: Implement `mcp_server.py`

Structure recommendation:
1. Imports:
- `FastMCP` from `fastmcp`
- Existing helpers from `tools.py`
- `load_dataset` for stateless wrappers when needed

2. App initialization:
- Give a clear server name and instructions for clients.

3. Tool registration:
- Use MCP decorators to expose each tool with concise description.
- Validate inputs at function boundary (bounds and empty strings).

4. Stateless wrappers:
- Do not call `filter_by_*` in MCP tools.
- Avoid `reset_filter()` as a control mechanism; compute directly from data per call.

5. Startup entrypoint:
- Run with streamable HTTP transport.
- Host `0.0.0.0`, port `8000` (as in plan).

---

## Suggested implementation pattern

### Keep these direct passthroughs
- `list_categories`
- `get_intent_distribution`
- `summarize_category`

### Prefer wrapper for these
- `show_examples`:
  - Build subset from full dataset using optional filters.
  - Sample deterministically for reproducibility.

- `count_rows_for_intent`:
  - `df[df['intent'] == intent.lower().strip()]` count.

- `search_responses`:
  - delegate to `search_responses(keyword, n, use_working_set=False)`.

---

## Input validation checklist
For every MCP tool:
1. Trim strings.
2. Reject empty required strings with clear error text.
3. Bound integer params (`n`, `n_examples`) to safe ranges.
4. Normalize category and intent casing consistently with existing tooling conventions.

---

## Error handling checklist
1. Catch exceptions per tool and return actionable error text.
2. Never leak stack traces to tool callers.
3. Include hint text for invalid category/intent values.
4. Keep responses deterministic and concise.

---

## Phase D: Local run and verification
Run server:

```bash
uv sync
source .venv/bin/activate
python mcp_server.py
```

Expected:
- Server starts on `0.0.0.0:8000`.
- No import/runtime exceptions.

Smoke tool checks (manual/client):
1. `list_categories` returns category rows.
2. `get_intent_distribution(category='ACCOUNT')` returns breakdown.
3. `show_examples(category='REFUND', n=2)` returns 2 rows.
4. `count_rows_for_intent(intent='get_refund')` returns non-zero.
5. `search_responses(keyword='refund', n=3)` returns matches.

---

## Phase E: README updates for grading
Add a dedicated MCP section to `README.md` with:

1. Start command
```bash
uv sync
source .venv/bin/activate
python mcp_server.py
```

2. Connection info
- transport: streamable-http
- endpoint host/port

3. Example Python client snippet
- Connect to MCP server
- Call at least one exposed tool

4. Tool list and short descriptions
- Include required 3 tools explicitly.

---

## Regression guardrails
After adding MCP server, re-run these checks:
1. Task 1 core scenarios still pass.
2. Task 2 memory still works (`--session`, `--list-sessions`, memory question).
3. MCP tools do not mutate or depend on shared filter state.

---

## Definition of done for Task 3
Task 3 is done when all are true:
1. `mcp_server.py` runs cleanly.
2. At least 3 MCP tools are exposed and callable.
3. Tool behavior is stateless and deterministic.
4. README includes startup + client usage example.
5. Task 1/2 behaviors remain intact.

---

## Suggested commit split
1. `feat(task3): add FastMCP server with stateless analysis tools`
- `mcp_server.py`
- any minimal helper changes required in `tools.py`

2. `docs(task3): add MCP startup and client usage`
- `README.md`
- optional task notes update
