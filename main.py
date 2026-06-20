"""Command-line interface for Task 1.

This CLI starts an interactive loop and prints reasoning traces
(tool calls and observations) before the final answer.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from agent import build_graph, graph as default_graph, run_query
from memory import (
    format_profile_for_prompt,
    list_sessions,
    load_user_profile,
)


def parse_args() -> argparse.Namespace:
    """Parse supported Task 1 CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent (Task 1)"
    )
    parser.add_argument("--query", help="Run a single query and exit")
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID for persistent memory (default: default)",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List stored session IDs and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def get_compiled_graph():
    """Return graph compiled with memory support when available."""
    return default_graph if default_graph is not None else build_graph()


def ensure_env() -> None:
    """Validate required environment variables before model calls."""
    if not os.environ.get("NEBIUS_API_KEY"):
        print(
            "Error: NEBIUS_API_KEY is not set.\n"
            "Set it in a local .env file (project root), for example:\n"
            "  NEBIUS_API_KEY=your_key",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run_interactive(session_id: str) -> None:
    """Run interactive Task 1 chat loop."""
    graph = get_compiled_graph()
    print("Customer Service Data Analyst Agent")
    print(f"Session: {session_id}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if user_input.lower() == "sessions":
            sessions = list_sessions()
            if sessions:
                print("\nStored sessions:")
                for sid in sessions:
                    print(f"- {sid}")
            else:
                print("\nNo stored sessions found.")
            continue
        if user_input.lower() == "profile":
            profile = load_user_profile(session_id)
            print(f"\n{format_profile_for_prompt(profile)}\n")
            continue
        if user_input.lower() in {"recommend", "suggest"}:
            user_input = "What should I query next?"

        print("\nThinking...")
        answer, _ = run_query(
            compiled_graph=graph,
            query=user_input,
            verbose=True,
            session_id=session_id,
        )
        if answer:
            print(f"\nAgent: {answer}\n")


def run_single(query: str, session_id: str) -> None:
    """Run one query non-interactively."""
    graph = get_compiled_graph()
    print(f"Session: {session_id}")
    print(f"Query: {query}\n")
    answer, _ = run_query(
        compiled_graph=graph,
        query=query,
        verbose=True,
        session_id=session_id,
    )
    print(f"\nAnswer: {answer}\n")


def main() -> None:
    """Task 1 CLI entrypoint."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ensure_env()
    if args.list_sessions:
        sessions = list_sessions()
        if sessions:
            print("Stored sessions:")
            for sid in sessions:
                print(f"- {sid}")
        else:
            print("No stored sessions found.")
        return

    if args.query:
        run_single(args.query, args.session)
    else:
        run_interactive(args.session)


if __name__ == "__main__":
    main()
