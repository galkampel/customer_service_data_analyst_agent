# Streamlit UI Detailed Implementation Plan

## Goal
Build Bonus A as a production-ready root Streamlit app that wraps the existing
LangGraph agent, preserves reasoning transparency, and supports session-aware
memory and recommender flows.

Current status:
- Phase A implemented
- Phase B implemented
- Phase C implemented
- Phase D implemented
- Phase E implemented
- Phase F implemented
- Phase G implemented
- Phase H documentation updates implemented

Target file to create:
- `streamlit_app.py`

Primary success criteria:
1. Chat UI works end-to-end with the root `agent.py` and `run_query()`.
2. Tool calls and observations are visible in expandable sections.
3. Session switching works and reuses memory-backed behavior.
4. Recommender flow works from the UI (suggest, refine, confirm, decline)
   without auto-execution.

---

## Important current-code constraints

Before implementation, align with current root APIs only:

1. `run_query()` signature in root `agent.py` is:
   - `run_query(compiled_graph, query, verbose=True, session_id="default")`
2. `run_query()` already handles pending recommendation logic internally via
   persisted state in `memory/pending_recommendations/`.
3. Root memory utilities provide:
   - `load_user_profile(session_id)`
   - `format_profile_for_prompt(profile)`
   - `list_sessions()`
   - `get_checkpointer()`
4. Avoid relying on obsolete helper names from older code paths (for example,
   `format_profile_for_display` or older `run_query` signatures).

Design implication:
- Streamlit should not duplicate recommender state logic in `st.session_state`
  unless needed for temporary UI-only hints. Use the existing backend flow by
  sending user text through `run_query()`.

---

## Phase A - Foundation and dependencies

### Files
- `pyproject.toml`
- `README.md`

### Tasks
1. Ensure `streamlit` is present in dependencies.
2. Confirm project runs with:
   - `uv sync`
   - `source .venv/bin/activate`
   - `python -c "import streamlit; print(streamlit.__version__)"`
3. Add a short README section for launching the UI.

### Validation
1. `uv sync` completes.
2. `import streamlit` works in `.venv`.

---

## Phase B - App skeleton and session state model

### File
- `streamlit_app.py`

### Tasks
1. Add `st.set_page_config(...)`.
2. Initialize app-level session state keys:
   - `session_id: str` (default `default`)
   - `messages: list[dict]` for rendered chat
   - `graph` cached compiled graph object
   - `api_key_set: bool`
3. Build a `get_graph()` helper:
   - create checkpointer via `get_checkpointer()`
   - compile graph via `build_graph(checkpointer=...)`
   - cache in `st.session_state`
4. Add a top-level `main()` entrypoint and call it under
   `if __name__ == "__main__":`.

### Notes
Keep app import-safe; avoid heavy graph creation at module import time.

### Validation
Run:
- `streamlit run streamlit_app.py`

Expected:
- App opens without exceptions.

---

## Phase C - Sidebar controls (memory-focused)

### File
- `streamlit_app.py`

### Tasks
1. API key input:
   - password field in sidebar
   - update `os.environ["NEBIUS_API_KEY"]` if entered
2. Session ID controls:
   - editable text input for current session
   - load button / reactive switch
3. Session list:
   - use `list_sessions()`
   - optional dropdown to switch existing sessions
4. Profile panel:
   - load via `load_user_profile(session_id)`
   - render through `format_profile_for_prompt(profile)`

### Behavior
On session switch:
1. update `session_id`
2. clear UI chat messages
3. keep backend memory intact (do not delete checkpoints)
4. reset cached graph only if needed by your architecture

### Validation
Manual:
1. Create session A and ask a query.
2. Switch to session B and ask different query.
3. Switch back to A and verify profile/session context still differs.

---

## Phase D - Chat surface and reasoning trace

### File
- `streamlit_app.py`

### Tasks
1. Add chat transcript rendering:
   - user bubble
   - assistant bubble
2. Add reasoning renderer for each assistant response:
   - for each step in `steps` returned by `run_query()`
   - show tool calls and observations in expanders
3. Submit pipeline:
   - read `st.chat_input()`
   - append user message
   - call `run_query(compiled_graph=graph, query=user_text, session_id=...)`
   - append assistant answer + steps

### UX rule
Reasoning should be visible by default via expandable controls, preserving
assignment transparency requirements.

### Validation
Manual queries:
1. Structured: "How many refund requests did we get?"
2. Unstructured: "Summarize the FEEDBACK category."
3. Out-of-scope: "Who won the 2024 Champions League?"

Expected:
1. Structured/unstructured show tool traces.
2. Out-of-scope returns refusal without tool loops.

---

## Phase E - Recommender integration (use backend flow)

### File
- `streamlit_app.py`

### Recommended integration
Do not re-implement recommender state machine in UI.

Leverage existing backend behavior in `run_query()`:
1. User asks suggestion question.
2. Backend stores pending recommendation.
3. User sends refinement/confirm/decline text.
4. Backend resolves pending logic and executes only on confirmation.

### Optional convenience control
Add sidebar button:
- "Suggest a follow-up query"
- when clicked, submit synthetic user text:
  - "What should I query next?"
  through the same `run_query()` path.

### Validation
Manual flow in UI:
1. Ask: "What should I query next?"
2. Reply: "make it about refund examples"
3. Reply: "yes, run it"
4. Repeat with decline:
   - ask suggestion
   - reply "no thanks"
   - then reply "yes"

Expected:
1. Suggest/refine responses have no tool calls.
2. Confirmation runs normal tool loop.
3. Decline clears pending state; later "yes" does not run stale suggestion.

---

## Phase F - Session persistence behavior in UI

### File
- `streamlit_app.py`

### Problem to address
UI message list in Streamlit session state can be lost on refresh, while backend
memory persists. Decide and implement one of these patterns:

Option 1 (fast, acceptable):
- Keep only current-page chat rendering in Streamlit state.
- Rely on backend memory for actual agent behavior.

Option 2 (better UX):
- Rehydrate display chat from graph state checkpoints on session load.

Recommendation:
- Start with Option 1 for robustness and speed.
- Add Option 2 later if grading requires visible historical chat on reload.

Implementation note:
- Option 2 is now implemented in root `streamlit_app.py` via checkpoint
   rehydration (`graph.get_state(...)`) and message mapping to UI chat rows.

### Validation
1. Ask a query in session `ui-a`.
2. Reload browser tab.
3. Ask follow-up in same session.
4. Verify backend still uses prior context even if UI transcript resets.

---

## Phase G - Error handling and resilience

### File
- `streamlit_app.py`

### Tasks
1. Guard API key missing case with user-visible warning.
2. Wrap graph invocation in try/except and show friendly error box.
3. Truncate very long tool observations in UI to keep page responsive.
4. Keep deterministic empty-response fallback text:
   - "No answer produced, try rephrasing your question."

### Validation
1. Remove API key and verify warning state.
2. Trigger a failing request and verify graceful message without crash.

---

## Phase H - Documentation and grading readiness

### Files
- `README.md`
- optionally `PLAN.md`

### README additions
1. How to run Streamlit:
   - `uv sync`
   - `source .venv/bin/activate`
   - `streamlit run streamlit_app.py`
2. What UI includes:
   - session switch
   - profile panel
   - reasoning trace
   - recommender flow
3. Quick demo script for graders.

### Optional plan update
Mark Bonus A checklist as complete when verified.

---

## Verification checklist for Bonus A

Technical checks:
1. `python -m py_compile streamlit_app.py`
2. Streamlit app launches successfully.

Behavior checks:
1. Structured, unstructured, out-of-scope flows behave correctly.
2. Reasoning steps are displayed in UI.
3. Session switching changes memory context.
4. Recommender suggest/refine/confirm/decline flow works.

Latest executed verification results:
1. `python -m py_compile streamlit_app.py` passed.
2. Headless startup passed on port `8502`.
3. Live smoke passed for:
   - structured query
   - unstructured summary
   - out-of-scope refusal
   - recommender suggest -> refine -> confirm flow
4. Deterministic recommender/state tests passed with fallback mode
   (`NEBIUS_API_KEY=`).

Regression checks:
1. CLI still works:
   - `python main.py --session reg-cli --query "How many rows are in REFUND?"`
2. MCP server still starts:
   - `python mcp_server.py`

---

## Suggested commit split

1. `feat(ui): add Streamlit chat shell with session controls`
   - `streamlit_app.py`
   - minimal dependency updates if needed

2. `feat(ui): add reasoning trace and recommender UX flow`
   - `streamlit_app.py`

3. `docs(ui): document Streamlit setup and demo flow`
   - `README.md`
   - optional `PLAN.md` checklist update
