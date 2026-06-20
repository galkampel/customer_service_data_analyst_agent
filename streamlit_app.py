"""Streamlit UI for the Customer Service Data Analyst Agent.

This file currently covers:
- Phase B: app bootstrap and graph caching
- Phase C: sidebar API/session/profile controls
- Phase D: chat interface with reasoning trace rendering
- Phase E: recommender integration via backend run_query flow
- Phase F: session transcript rehydration from checkpoints
- Phase G: error handling hardening
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent import build_graph, graph as default_graph, run_query
from memory import (
    format_profile_for_prompt,
    get_checkpointer,
    list_sessions,
    load_user_profile,
)


def init_session_state() -> None:
    """Initialize Streamlit session state keys used by the UI."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = load_user_profile(
            st.session_state.session_id
        )
    if "rehydrated_session_id" not in st.session_state:
        st.session_state.rehydrated_session_id = None
    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = bool(os.environ.get("NEBIUS_API_KEY"))


def get_graph() -> Any | None:
    """Return compiled graph, building and caching it on first access."""
    if st.session_state.graph is not None:
        return st.session_state.graph

    try:
        if default_graph is not None:
            st.session_state.graph = default_graph
            return st.session_state.graph

        checkpointer = get_checkpointer()
        st.session_state.graph = build_graph(checkpointer=checkpointer)
        return st.session_state.graph
    except EnvironmentError as exc:
        st.error(
            "Unable to initialize agent graph. Set NEBIUS_API_KEY and retry."
        )
        st.caption(f"Details: {exc}")
        return None
    except (RuntimeError, ValueError, TypeError) as exc:
        st.error("Failed to initialize agent graph.")
        st.caption(f"Details: {exc}")
        return None


def _chat_row_from_message(message: BaseMessage) -> dict[str, Any] | None:
    """Convert a LangChain message into a chat row for UI rendering."""
    content = str(message.content).strip()
    if not content:
        return None

    if isinstance(message, HumanMessage):
        return {"role": "user", "content": content, "steps": []}

    if isinstance(message, AIMessage):
        if message.tool_calls:
            return None
        return {"role": "assistant", "content": content, "steps": []}

    return None


def rehydrate_messages_from_checkpoint() -> None:
    """Load chat transcript for the active session from checkpoint state."""
    session_id = st.session_state.session_id
    if st.session_state.rehydrated_session_id == session_id:
        return

    graph = get_graph()
    if graph is None:
        return

    try:
        config = {"configurable": {"thread_id": session_id}}
        state = graph.get_state(config)
        values = state.values if state else {}
        messages = values.get("messages", []) if values else []

        rows: list[dict[str, Any]] = []
        for message in messages:
            row = _chat_row_from_message(message)
            if row is not None:
                rows.append(row)

        st.session_state.messages = rows
        st.session_state.rehydrated_session_id = session_id
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        st.warning("Could not rehydrate chat transcript for this session.")
        st.caption(f"Details: {exc}")


def switch_session(session_id: str) -> None:
    """Switch active session and refresh UI-local state."""
    cleaned = session_id.strip() or "default"
    if cleaned == st.session_state.session_id:
        return

    st.session_state.session_id = cleaned
    st.session_state.messages = []
    st.session_state.user_profile = load_user_profile(cleaned)
    st.session_state.rehydrated_session_id = None


def render_sidebar() -> None:
    """Render sidebar controls for API key, session, and profile."""
    with st.sidebar:
        st.title("Settings")

        api_key_value = st.text_input(
            "Nebius API Key",
            value=os.environ.get("NEBIUS_API_KEY", ""),
            type="password",
            help="Used by router/agent/profile/recommender models.",
        )
        if api_key_value:
            os.environ["NEBIUS_API_KEY"] = api_key_value
            st.session_state.api_key_set = True
        else:
            st.session_state.api_key_set = bool(
                os.environ.get("NEBIUS_API_KEY")
            )

        st.divider()
        st.subheader("Session")

        session_input = st.text_input(
            "Active session ID",
            value=st.session_state.session_id,
        )
        if session_input.strip() != st.session_state.session_id:
            switch_session(session_input)
            st.rerun()

        sessions = list_sessions()
        if sessions:
            selected = st.selectbox(
                "Switch to existing session",
                options=[""] + sessions,
                index=0,
            )
            if selected and selected != st.session_state.session_id:
                switch_session(selected)
                st.rerun()

        if st.button("Clear current chat display"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("User profile")
        profile_text = format_profile_for_prompt(
            st.session_state.user_profile
        )
        st.text(profile_text)

        st.divider()
        st.subheader("Query recommender")
        st.caption(
            "Uses the same backend flow as chat: suggest, refine, "
            "confirm, or decline."
        )
        if st.button("Suggest a follow-up query"):
            handle_user_input("What should I query next?")
            st.rerun()


def render_step(step: dict[str, str]) -> None:
    """Render one reasoning step in an expander."""
    step_type = step.get("type", "")
    name = step.get("name", "tool")
    content = step.get("content", "")

    if step_type == "tool_call":
        label = f"Tool call: {name}"
    elif step_type == "observation":
        label = f"Observation: {name}"
    else:
        label = "Reasoning step"

    with st.expander(label, expanded=False):
        if step_type == "tool_call":
            st.code(content or "{}", language="python")
        else:
            preview = (
                content[:3000] + "..."
                if isinstance(content, str) and len(content) > 3000
                else content
            )
            st.text(str(preview))


def render_chat_history() -> None:
    """Render conversation messages and assistant reasoning traces."""
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            if role == "assistant":
                for step in msg.get("steps", []):
                    render_step(step)
            st.markdown(msg.get("content", ""))


def handle_user_input(user_text: str) -> None:
    """Send one user turn through the backend and render results."""
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_text,
            "steps": [],
        }
    )

    graph = get_graph()
    if graph is None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I could not initialize the agent graph. "
                    "Set NEBIUS_API_KEY and try again."
                ),
                "steps": [],
            }
        )
        return

    try:
        with st.spinner("Thinking..."):
            answer, steps = run_query(
                compiled_graph=graph,
                query=user_text,
                verbose=False,
                session_id=st.session_state.session_id,
            )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I hit an internal error while processing that request. "
                    "Please try rephrasing it."
                ),
                "steps": [],
            }
        )
        st.error("Request failed")
        st.caption(f"Details: {exc}")
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
            or "(No answer produced. Try rephrasing your request.)",
            "steps": steps,
        }
    )
    st.session_state.user_profile = load_user_profile(
        st.session_state.session_id
    )


def render_main_view() -> None:
    """Render the main chat interface."""
    rehydrate_messages_from_checkpoint()

    st.title("Customer Service Data Analyst Agent")
    st.caption(
        f"Session: {st.session_state.session_id} "
        "- reasoning traces shown for assistant answers"
    )

    if st.session_state.api_key_set:
        st.success("NEBIUS_API_KEY detected")
    else:
        st.warning(
            "NEBIUS_API_KEY is not set. "
            "Configure it in your environment or .env file."
        )

    render_chat_history()

    user_text = st.chat_input("Ask about the customer service dataset...")
    if user_text:
        handle_user_input(user_text)
        st.rerun()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Customer Service Analyst",
        page_icon="CS",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    render_sidebar()
    render_main_view()


if __name__ == "__main__":
    main()
