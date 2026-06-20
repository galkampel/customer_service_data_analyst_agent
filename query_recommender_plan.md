# Query Recommender Detailed Implementation Plan

## Goal
Add the Bonus B query recommender flow to the customer-service data analyst agent.

The recommender should help the user decide what to ask next, but it must not execute the suggested query until the user explicitly confirms. The final behavior should support this flow:

1. User asks for a recommendation.
2. Agent suggests one concrete dataset query.
3. User can refine, decline, or confirm the suggestion.
4. Agent only runs the suggested query after clear confirmation.

---

## Implementation status
The root implementation now supports:

- `structured`, `unstructured`, and `out_of_scope` routing in `agent.py`.
- `recommender` routing in `agent.py`.
- Persistent session/profile memory through `memory.py`.
- Persistent pending recommendations through `memory.py`.
- Role-based LLM selection through `llm_config.py`.
- CLI execution through `main.py`.
- MCP server tooling through `mcp_server.py`.
- Root-level recommendation helpers in `recommender.py`.
- Interactive `recommend` and `suggest` aliases in `main.py`.

The root code now contains the active recommender implementation.

Validated behavior:

- Suggestion requests produce a pending query and do not call dataset tools.
- Refinement updates the pending suggestion and keeps it pending.
- Confirmation executes the pending suggestion through the normal agent/tool loop.
- Declination clears pending state and prevents stale later confirmations.
- Rule checks avoid false positives such as `yesterday` or `knowledge`.

---

## Desired user experience

### Suggest
User:

```text
What should I query next?
```

Agent:

```text
Suggested query:
"Compare the top intents in REFUND and PAYMENT and show one example from each."

Would you like me to run this query? Reply yes to run it, no to cancel, or describe how to change it.
```

No tools should be called during this step.

### Refine
User:

```text
Make it about delivery instead.
```

Agent:

```text
Updated suggested query:
"Compare the top intents in DELIVERY and SHIPPING and show one example from each."

Would you like me to run this query? Reply yes to run it, no to cancel, or describe how to change it.
```

Still no dataset tools should be called.

### Confirm and execute
User:

```text
yes, run it
```

Agent executes the pending suggested query through the normal graph/tool path and returns the grounded answer with reasoning steps.

### Decline
User:

```text
no thanks
```

Agent cancels the pending suggestion and does not execute anything.

---

## Phase 0 - Add model configuration

### Files
- `llm_config.py`
- `.env.example`
- `README.md`

### Tasks
1. Add recommender role support to the shared LLM factory:
   - Role name: `recommender`.
   - Environment variable: `NEBIUS_RECOMMENDER_MODEL`.
   - Fallback: `NEBIUS_MODEL`.
2. Add `NEBIUS_RECOMMENDER_MODEL` to `.env.example`.
3. Document that the recommender can use a smaller/cheaper model because it generates short suggestions and refinements.

### Logic
Recommendation is a low-risk, short-output task. It should not consume the strongest model by default. Keeping it role-based also matches the existing architecture for router/profile/main models.

### Validation
Run:

```bash
uv sync
source .venv/bin/activate
python - <<'PY'
from llm_config import make_llm
llm = make_llm(role="recommender", temperature=0.2)
print(type(llm).__name__)
PY
```

Expected: no import/config error when environment variables are configured.

---

## Phase 1 - Create `recommender.py`

### Responsibilities
The module should be pure orchestration logic for query suggestions. It should not import or call dataset tools directly.

### Public functions
Implement these root-level functions:

```python
def generate_suggestion(
    messages: list[BaseMessage],
    user_profile: dict[str, Any],
) -> str: ...

def refine_suggestion(pending: str, refinement: str) -> str: ...

def is_confirmation(text: str) -> bool: ...

def is_declination(text: str) -> bool: ...

def format_suggestion_response(suggestion: str) -> str: ...
```

### Suggestion prompt requirements
The suggestion prompt should tell the LLM:

- The agent analyzes the Bitext Customer Service dataset.
- Valid categories are `ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, and `SUBSCRIPTION`.
- Suggest exactly one concrete next query.
- Build on recent conversation and user profile.
- Keep the suggestion executable by the existing agent tools.
- Return only the query text, with no explanation or markdown.

### Refinement prompt requirements
The refinement prompt should include:

- Current pending query.
- User refinement request.
- Instruction to return one revised executable query only.

### Confirmation detection
Use rule-based detection rather than an LLM call.

Confirm examples:

- `yes`
- `yeah`
- `sure`
- `ok`
- `go ahead`
- `run it`
- `do it`
- `sounds good`

Decline examples:

- `no`
- `no thanks`
- `cancel`
- `skip`
- `not now`
- `never mind`

Use word-boundary regex/tokenization so words like `yesterday`, `notebook`, or `knowledge` do not accidentally trigger yes/no logic.

### Fallback behavior
If recommender LLM setup fails or the API key is missing, return a deterministic fallback query:

```text
What is the distribution of intents in the REFUND category?
```

This keeps the CLI usable in limited environments.

### Validation
Run a syntax/import check:

```bash
source .venv/bin/activate
python -m py_compile recommender.py
python - <<'PY'
from recommender import is_confirmation, is_declination
assert is_confirmation("yes, run it")
assert is_confirmation("go ahead")
assert not is_confirmation("yesterday")
assert is_declination("no thanks")
assert is_declination("never mind")
assert not is_declination("knowledge")
print("recommender rules ok")
PY
```

---

## Phase 2 - Extend graph state and router

### Files
- `agent.py`

### Tasks
1. Extend `QueryType`:

```python
QueryType = Literal[
    "structured",
    "unstructured",
    "out_of_scope",
    "recommender",
]
```

2. Update the router prompt to include `recommender`.

3. Add router examples:

```text
"What should I query next?" -> recommender
"Suggest a useful follow-up analysis" -> recommender
"Recommend a question about this dataset" -> recommender
```

4. Update `normalize_query_type()` to accept `recommender`.

5. Extend `AgentState` with:

```python
pending_recommendation: str | None
```

6. Keep uncertain labels defaulting to `structured`, not `recommender`.

### Logic
The recommender should be a deliberate route only when the user asks for a suggestion. Ambiguous data questions should continue to the normal tool loop.

### Validation
Run a narrow parser test:

```bash
source .venv/bin/activate
python - <<'PY'
from agent import normalize_query_type
assert normalize_query_type("recommender") == "recommender"
assert normalize_query_type("recommender.") == "recommender"
assert normalize_query_type("unknown") == "structured"
print("query type parser ok")
PY
```

---

## Phase 3 - Add recommender graph node

### Files
- `agent.py`

### Tasks
1. Add a `recommender_node(state)` function.
2. In the node:
   - Load recent conversation from `state["messages"]`.
   - Load user profile from `state["user_profile"]`.
   - Call `generate_suggestion(...)`.
   - Return an `AIMessage` with `format_suggestion_response(suggestion)`.
   - Store `pending_recommendation` in graph state.
3. Add the node to `build_graph()`.
4. Update `after_router()` so:
   - `out_of_scope` routes to `out_of_scope`.
   - `recommender` routes to `recommender`.
   - everything else routes to `agent`.
5. Add an edge from `recommender` to `END`.

### Logic
The recommender node should not call dataset tools. It only proposes a query and pauses for user consent.

### Validation
Run a no-tool recommender smoke test with a configured API key, or rely on fallback if no key is available:

```bash
source .venv/bin/activate
python main.py --session rec-smoke --query "What should I query next?"
```

Expected:

- Output contains a suggested query.
- No tool call lines are printed.
- The answer asks whether to run/refine/cancel.

---

## Phase 4 - Handle pending recommendation turns

### Files
- `agent.py`
- `main.py`

### Problem
The graph checkpoint can persist messages, but the root `run_query()` currently creates a fresh input state each call. The implementation needs a reliable place to remember the pending query across turns.

### Recommended approach
Persist pending recommendation outside the graph checkpoint in a small JSON file under `memory/`, similar to the profile store.

Example helper names:

```python
load_pending_recommendation(session_id: str) -> str | None
save_pending_recommendation(session_id: str, suggestion: str | None) -> None
```

These helpers can live in `memory.py` or `recommender.py`. Prefer `memory.py` if you want all session persistence in one module.

### Turn handling in `run_query()`
At the very start of `run_query()`:

1. Load pending recommendation for `session_id`.
2. If pending exists:
   - If latest user text is confirmation:
     - Clear pending recommendation.
     - Replace the user query with the pending suggestion.
     - Run through the normal graph as `structured`/normal routing.
   - If latest user text is declination:
     - Clear pending recommendation.
     - Return a cancellation answer without tool calls.
   - Otherwise:
     - Treat user text as refinement.
     - Call `refine_suggestion(pending, query)`.
     - Save the refined suggestion.
     - Return formatted suggestion without tool calls.
3. If no pending exists:
   - Continue normal routing.

### Important guard
Do not update the durable user profile based only on a `yes`, `no`, or refinement turn unless the final executed query produces a meaningful assistant answer. Otherwise the profile extractor may store noisy facts such as "user said yes".

### Validation
Run this sequence:

```bash
source .venv/bin/activate
python main.py --session rec-flow --query "What should I query next?"
python main.py --session rec-flow --query "make it about delivery"
python main.py --session rec-flow --query "yes, run it"
```

Expected:

- First command suggests a query, no tools.
- Second command refines the pending query, no tools.
- Third command runs the refined query through the normal tool loop.

Then test decline:

```bash
python main.py --session rec-cancel --query "What should I query next?"
python main.py --session rec-cancel --query "no thanks"
python main.py --session rec-cancel --query "yes"
```

Expected:

- `no thanks` cancels the suggestion.
- The later `yes` should not execute the old suggestion.

---

## Phase 5 - CLI ergonomics

### Files
- `main.py`
- `README.md`

### Tasks
1. In interactive mode, no new command is strictly required. Users can type:

```text
What should I query next?
```

2. Optional command aliases:

```text
recommend
suggest
```

These can internally call `run_query(..., "What should I query next?")`.

3. When a pending suggestion exists, the CLI prompt can stay simple, but the assistant response must clearly tell the user they can confirm, cancel, or refine.

### Validation
Run:

```bash
source .venv/bin/activate
python main.py --session rec-interactive
```

Manual flow:

```text
What should I query next?
make it more focused on refunds
yes
```

Expected: the third turn executes the refined recommendation and prints normal reasoning trace.

---

## Phase 6 - Optional Streamlit integration

### Files
- `streamlit_app.py`

### Tasks
1. Add a sidebar or chat action button: `Suggest a follow-up query`.
2. Store pending suggestion in `st.session_state.pending_recommendation`.
3. In chat input handling:
   - Confirmation runs pending suggestion.
   - Declination clears pending suggestion.
   - Other text refines pending suggestion.
4. Show recommendation messages without reasoning trace.
5. Show executed recommendation answers with normal reasoning trace.

### Logic
The UI should mirror the CLI behavior. The recommender should feel like a small workflow, not a separate app mode.

### Validation
Run:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Manual checks:

- Click suggest.
- Refine suggestion.
- Confirm and verify a real dataset answer appears.
- Decline and verify no old pending query runs later.

---

## Phase 7 - Documentation updates

### Files
- `README.md`
- `PLAN.md` if you want to mark Bonus B complete

### README section should include
1. What the recommender does.
2. CLI examples:

```bash
python main.py --session demo --query "What should I query next?"
python main.py --session demo --query "make it about refunds"
python main.py --session demo --query "yes, run it"
```

3. Safety rule: recommendations are never auto-executed.
4. Model split note: `NEBIUS_RECOMMENDER_MODEL` can be a smaller model.

---

## Regression checklist

After implementation, run these checks:

```bash
source .venv/bin/activate
python -m py_compile agent.py recommender.py main.py memory.py llm_config.py
```

Task 1 still works:

```bash
python main.py --session reg-structured --query "How many rows are in REFUND?"
python main.py --session reg-unstructured --query "Summarize the FEEDBACK category."
python main.py --session reg-oos --query "Who won the Champions League?"
```

Task 2 memory still works:

```bash
python main.py --session reg-memory --query "I prefer concise answers with examples."
python main.py --session reg-memory --query "What do you remember about me?"
```

Recommender works:

```bash
python main.py --session reg-rec --query "What should I query next?"
python main.py --session reg-rec --query "make it about refund examples"
python main.py --session reg-rec --query "yes, run it"
```

MCP still starts:

```bash
python mcp_server.py
```

---

## Definition of done

The query recommender task is complete when:

1. [x] `recommender.py` exists and exposes suggestion, refinement, confirmation, declination, and formatting helpers.
2. [x] Router supports the `recommender` label.
3. [x] Recommender requests produce a suggestion without calling dataset tools.
4. [x] Pending suggestions persist per session.
5. [x] Refinement updates the pending suggestion without executing it.
6. [x] Declination clears the pending suggestion.
7. [x] Confirmation runs the pending suggestion through the normal agent/tool loop.
8. [x] The root CLI supports the full suggest/refine/confirm flow.
9. [x] README documents usage and the no-auto-execution rule.
10. [x] Task 1, Task 2, and Task 3 regression checks remain available in docs.

---

## Test results

Latest recommender validation covered:

1. Syntax/import checks for `agent.py`, `memory.py`, `main.py`, `recommender.py`, and `llm_config.py`.
2. Router parser checks for `recommender`, `recommender.`, and unknown fallback to `structured`.
3. Confirmation/declination checks, including false-positive guards for `yesterday` and `knowledge`.
4. Pending recommendation save/load/clear behavior.
5. Cross-turn `run_query()` behavior for suggest, refine, confirm, decline, and stale confirmation after cancellation.
6. Direct `recommender_node()` check confirming it returns a suggestion without tool calls.
7. Live CLI smoke with configured Nebius environment:

```bash
python main.py --session rec-live-smoke --query "What should I query next?"
python main.py --session rec-live-smoke --query "make it about refund examples"
python main.py --session rec-live-smoke --query "yes, run it"
```

The live confirmation step executed the refined recommendation through normal
tool calls and returned a grounded REFUND-category answer.

---

## Suggested commit split

1. `feat(recommender): add query suggestion and refinement helpers`
   - `recommender.py`
   - `llm_config.py`
   - `.env.example`

2. `feat(agent): route and persist query recommendations`
   - `agent.py`
   - `memory.py`
   - `main.py`

3. `docs(recommender): document suggest refine confirm flow`
   - `README.md`
   - `PLAN.md`
   - `query_recommender_plan.md`
