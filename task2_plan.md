# Task 2 Implementation Plan: Persistent Memory

## Objective
Implement Task 2 memory requirements for:
- Persistent episodic memory (conversation state across restarts)
- Persistent user profile memory (durable user facts)

The implementation must preserve Task 1 behavior (structured, unstructured, out-of-scope routing and tool execution).

---

## Scope and Deliverables
Create and integrate:
1. `memory.py` for profile + session memory helpers
2. Checkpointer integration in `agent.py`
3. Session-aware execution in `run_query(...)`
4. CLI session controls in `main.py`
5. README/task docs updates for memory usage and validation

Expected Task 2 outcomes:
- Reusing a session ID restores context
- User profile persists on disk and is used in future answers
- The agent can answer memory questions (for example: "What do you remember about me?")

---

## Architecture Decisions

### 1) Episodic memory (conversation state)
Use LangGraph checkpointer with SQLite.
- Store path: `memory/checkpoints.db`
- Thread key: `session_id` (passed in graph config)
- Behavior: same `session_id` resumes prior state

### 2) User profile memory (durable facts)
Use JSON files keyed by session.
- Store path: `memory/profiles/<session_id>.json`
- Update source: latest user+assistant exchange only
- Merge mode: conservative (append/update only strong durable facts)

### 3) Role-based model usage
Use the small model for profile extraction via role-aware LLM factory.
- Prefer role key: `profile`
- Fallback to `NEBIUS_MODEL` when role-specific model var is missing

---

## Phase-by-Phase Plan

## Phase 0: Dependencies and config
1. Ensure dependencies include SQLite checkpointer support.
2. Verify `.env.example` has memory-related model variables:
   - `NEBIUS_PROFILE_MODEL` (optional but recommended)
3. Ensure runtime creates memory directories automatically:
   - `memory/`
   - `memory/profiles/`

Acceptance:
- Import checks pass for checkpointer module.
- App runs without manual directory creation.

## Phase 1: Build `memory.py`
Implement these functions:
1. `get_checkpointer(db_path: str = "memory/checkpoints.db")`
2. `list_sessions(db_path: str = "memory/checkpoints.db") -> list[str]`
3. `load_user_profile(session_id: str) -> dict`
4. `save_user_profile(session_id: str, profile: dict) -> None`
5. `update_user_profile(profile: dict, user_text: str, assistant_text: str) -> dict`
6. `format_profile_for_prompt(profile: dict) -> str`

Implementation notes:
- Use atomic write for profile files (temp file + replace).
- Return empty/default profile when file is missing.
- Fail gracefully on malformed JSON (backup and reset).

Acceptance:
- Profile load/save roundtrip works.
- `list_sessions()` returns known thread IDs.

## Phase 2: Integrate episodic memory in `agent.py`
1. Compile graph with checkpointer from `memory.py`.
2. Update execution path to pass configurable thread ID:
   - `config={"configurable": {"thread_id": session_id}}`
3. Extend state contract with session fields if needed.

Acceptance:
- With same session ID, follow-up questions use prior context after restart.

## Phase 3: Integrate profile memory in `agent.py`
1. Load profile at query start.
2. Add compact profile context into agent system prompt.
3. After final answer, call `update_user_profile(...)` and save profile.
4. Add explicit handling for memory question intent:
   - If user asks "what do you remember about me", answer from stored profile.

Acceptance:
- Profile influences future responses.
- Memory question returns stable profile-based output.

## Phase 4: Update CLI in `main.py`
1. Add arguments:
   - `--session <id>`
   - `--list-sessions`
2. In interactive mode, show active session clearly.
3. Add commands:
   - `sessions` to print known sessions
   - `profile` to print current session profile
4. Ensure both interactive and single-query mode pass `session_id` into query runner.

Acceptance:
- CLI supports switching sessions and listing persisted sessions.

## Phase 5: Documentation updates
Update task docs and README with:
1. How to run with session IDs
2. How to list sessions
3. How profile memory works
4. Example validation script/commands

Acceptance:
- Another user can validate Task 2 from docs only.

---

## Data Contracts

### Profile schema (recommended)
Use a stable JSON shape:

```json
{
  "preferences": [],
  "frequent_categories": [],
  "frequent_intents": [],
  "constraints": [],
  "notes": [],
  "last_updated": "ISO-8601 timestamp"
}
```

Rules:
- Keep values normalized (trim, lowercase where appropriate).
- Do not store ephemeral one-off requests as durable preferences.
- Merge by deduplicating values.

---

## Prompting Rules for Profile Extraction
Use strict extraction rules:
1. Extract only durable user facts.
2. Ignore transient task wording.
3. Prefer explicit user statements over inferred facts.
4. Output machine-parseable JSON only.
5. If uncertain, return no update.

---

## Validation Plan

## Functional tests
1. Session restoration:
   - Run with `--session alice`
   - Ask two related questions
   - Restart app and ask follow-up
   - Verify continuity

2. Session isolation:
   - Compare `--session alice` vs `--session bob`
   - Verify no context bleed

3. Profile persistence:
   - State a durable preference
   - Restart app
   - Ask "What do you remember about me?"
   - Verify preference is returned

4. Out-of-scope regression:
   - Ask non-dataset question
   - Verify polite refusal still works

5. Structured/unstructured regression:
   - Re-run Task 1 checks
   - Verify no degradation

## Failure-path tests
1. Delete/corrupt profile file and verify graceful recovery.
2. Remove/lock checkpoint DB and verify readable error handling.

---

## Risks and Mitigations
1. Risk: profile overfitting from noisy turns
- Mitigation: conservative extraction and merge policy

2. Risk: memory causes prompt bloat
- Mitigation: compact profile formatter with strict length cap

3. Risk: session mismatch between graph and profile store
- Mitigation: single canonical `session_id` passed end-to-end

4. Risk: Task 1 regressions
- Mitigation: run 3-scenario regression suite after each phase

---

## Definition of Done (Task 2)
All must be true:
1. Same `session_id` restores conversation after restart.
2. Profile persists to disk and is updated after assistant responses.
3. Memory question is answered from profile store.
4. CLI supports `--session` and `--list-sessions`.
5. Task 1 behavior remains intact.
