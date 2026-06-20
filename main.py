"""Command-line interface for Task 1.

This CLI starts an interactive loop and prints reasoning traces
(tool calls and observations) before the final answer.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from agent import build_graph, run_query


def parse_args() -> argparse.Namespace:
    """Parse supported Task 1 CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent (Task 1)"
    )
    parser.add_argument("--query", help="Run a single query and exit")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


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


def run_interactive() -> None:
    """Run interactive Task 1 chat loop."""
    graph = build_graph()
    print("Customer Service Data Analyst Agent")
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

        print("\nThinking...")
        answer, _ = run_query(
            compiled_graph=graph,
            query=user_input,
            verbose=True,
        )
        if answer:
            print(f"\nAgent: {answer}\n")


def run_single(query: str) -> None:
    """Run one query non-interactively."""
    graph = build_graph()
    print(f"Query: {query}\n")
    answer, _ = run_query(
        compiled_graph=graph,
        query=query,
        verbose=True,
    )
    print(f"\nAnswer: {answer}\n")


def main() -> None:
    """Task 1 CLI entrypoint."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ensure_env()
    if args.query:
        run_single(args.query)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
