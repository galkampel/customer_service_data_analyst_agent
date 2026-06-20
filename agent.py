"""Task 1 LangGraph ReAct agent for the customer-service dataset.

This graph implements:
- Dedicated query router (structured / unstructured / out_of_scope)
- ReAct tool loop (agent <-> tools)
- Max-iteration fallback for safe termination
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from data_loader import get_categories, load_dataset
from llm_config import make_llm
from tools import build_tools, reset_filter

MAX_ITERATIONS = 12
QueryType = Literal["structured", "unstructured", "out_of_scope"]

ROUTER_SYSTEM_PROMPT = """You classify user queries for a Customer Service
Data Analyst Agent.

The dataset contains customer support instructions, agent responses,
categories, and intents.

Return exactly ONE label from:
- structured: concrete, data-driven dataset query
- unstructured: open-ended dataset summary/explanation
- out_of_scope: unrelated to the dataset

Examples:
- 'How many refund requests did we get?' -> structured
- 'Show me 3 examples from SHIPPING' -> structured
- 'Summarize FEEDBACK' -> unstructured
- 'Who won the Champions League?' -> out_of_scope

Important rule: if uncertain, return structured.

Respond with ONLY: structured, unstructured, or out_of_scope.
"""

AGENT_SYSTEM_PROMPT = """You are a Customer Service Data Analyst Agent.

You analyze the Bitext customer service dataset.

Dataset facts:
- total rows: {row_count}
- categories ({category_count}): {categories}

Behavior requirements:
- For structured questions, call tools and produce exact data-driven answers.
- For unstructured questions, gather evidence with tools, then summarize.
- Keep final answers concise, accurate, and grounded in tool results.
- If a query is out of scope, decline politely.

Tool-use discipline:
- Never call the same tool with the same arguments twice.
- One or two tool calls are usually enough; after that, write the final answer.
- For summarization queries, a single summarize_category call is sufficient;
  then synthesize a short prose summary from its output.
"""

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with analysis of the Bitext Customer Service dataset. "
    "Please ask about categories, intents, examples, distributions, or "
    "summaries in that dataset."
)

MAX_ITERATION_MESSAGE = (
    "I reached the maximum reasoning steps before completing this answer. "
    "Please rephrase the request or split it into smaller questions."
)


class AgentState(TypedDict):
    """Graph state for Task 1 execution."""

    messages: Annotated[list[BaseMessage], add_messages]
    query_type: QueryType
    iteration_count: int


def normalize_query_type(raw_label: str) -> QueryType:
    """Convert an LLM router response into a valid query type."""
    # Router output is model-generated text, so parse defensively and default
    # to the safest in-scope path when the response is empty or unexpected.
    tokens = raw_label.strip().lower().split()
    if not tokens:  # Empty or whitespace-only response.
        return "structured"

    normalized = tokens[0].rstrip(".,;:")
    # Keep graph state constrained to QueryType values while handling noisy
    # punctuation or extra words from the router model response.
    match normalized:
        case "structured" | "unstructured" | "out_of_scope":
            return normalized
        case _:
            # On uncertain labels, prefer in-scope routing over refusal.
            return "structured"


def router_node(state: AgentState) -> dict[str, Any]:
    """Classify the latest user query before entering tool selection."""
    human_messages = [
        m for m in state["messages"] if isinstance(m, HumanMessage)
    ]
    if not human_messages:
        return {"query_type": "structured", "iteration_count": 0}

    # Only route based on the latest user message.
    user_text = human_messages[-1].content
    llm = make_llm(role="router", temperature=0.0)
    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]
    )
    label = normalize_query_type(str(response.content))
    return {"query_type": label, "iteration_count": 0}


FORCE_FINAL_ANSWER_INSTRUCTION = (
    "Stop calling tools. Use the existing tool observations to write a "
    "concise final answer now."
)

FORCE_UNSTRUCTURED_FINAL_INSTRUCTION = (
    "This is an unstructured summarization query and enough evidence has "
    "already been collected. Do not call tools again. Write a concise "
    "grounded summary now."
)


def _last_tool_call_repeats(messages: list[BaseMessage]) -> bool:
    """Detect whether the last two assistant tool calls are identical."""
    signatures: list[tuple[str, str]] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                signatures.append(
                    (call["name"], repr(call.get("args", {})))
                )
    return len(signatures) >= 2 and signatures[-1] == signatures[-2]


def _has_tool_observation(messages: list[BaseMessage]) -> bool:
    """Return True after at least one tool result has been observed."""
    return any(isinstance(msg, ToolMessage) for msg in messages)


def build_agent_node(tools: list[Any]):
    """Create the main agent node with bound tools."""
    llm = make_llm(role="main", temperature=0.0).bind_tools(tools)
    # Tool-less LLM used to force synthesis when the model loops on tools.
    plain_llm = make_llm(role="main", temperature=0.0)
    df = load_dataset()
    categories = ", ".join(get_categories())

    def _agent_node(state: AgentState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0)
        if iteration >= MAX_ITERATIONS:
            return {
                "messages": [AIMessage(content=MAX_ITERATION_MESSAGE)],
                "iteration_count": iteration + 1,
            }

        if state.get("query_type") == "out_of_scope":
            return {
                "messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)],
                "iteration_count": iteration + 1,
            }

        prompt = AGENT_SYSTEM_PROMPT.format(
            row_count=f"{len(df):,}",
            category_count=len(get_categories()),
            categories=categories,
        )
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=prompt)] + list(messages)

        # For unstructured summarization, one evidence-gathering tool result
        # is enough. Force synthesis immediately to avoid redundant tool loops.
        if (
            state.get("query_type") == "unstructured"
            and _has_tool_observation(list(messages))
        ):
            messages = list(messages) + [
                SystemMessage(content=FORCE_UNSTRUCTURED_FINAL_INSTRUCTION)
            ]
            response = plain_llm.invoke(messages)
            return {
                "messages": [response],
                "iteration_count": iteration + 1,
            }

        # If the model just repeated an identical tool call, force a final
        # answer with a tool-less LLM so the loop cannot continue.
        if _last_tool_call_repeats(list(messages)):
            messages = list(messages) + [
                SystemMessage(content=FORCE_FINAL_ANSWER_INSTRUCTION)
            ]
            response = plain_llm.invoke(messages)
            return {
                "messages": [response],
                "iteration_count": iteration + 1,
            }

        response = llm.invoke(messages)
        return {"messages": [response], "iteration_count": iteration + 1}

    return _agent_node


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route to tools when the latest assistant message includes tool calls."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def after_router(state: AgentState) -> Literal["agent", "out_of_scope"]:
    """Branch after router classification."""
    if state.get("query_type") == "out_of_scope":
        return "out_of_scope"
    return "agent"


def out_of_scope_node(state: AgentState) -> dict[str, Any]:
    """Return polite refusal for non-dataset questions."""
    _ = state
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}


def build_graph():
    """Compile and return the Task 1 graph."""
    tools = build_tools()
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("agent", build_agent_node(tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("out_of_scope", out_of_scope_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        after_router,
        {
            "agent": "agent",
            "out_of_scope": "out_of_scope",
        },
    )
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("out_of_scope", END)
    return workflow.compile()


def run_query(
    compiled_graph: Any,
    query: str,
    verbose: bool = True,
) -> tuple[str, list[dict[str, str]]]:
    """Execute one query and collect reasoning steps for CLI display."""
    reset_filter()

    state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query_type": "structured",
        "iteration_count": 0,
    }

    steps: list[dict[str, str]] = []
    answer = ""

    for event in compiled_graph.stream(state, stream_mode="values"):
        messages = event.get("messages", [])
        if not messages:
            continue
        last = messages[-1]

        if isinstance(last, AIMessage):
            if last.tool_calls:
                for call in last.tool_calls:
                    step = {
                        "type": "tool_call",
                        "name": call["name"],
                        "content": str(call.get("args", {})),
                    }
                    steps.append(step)
                    if verbose:
                        print(
                            f"\n[tool call] "
                            f"{step['name']} {step['content']}"
                        )
            elif last.content:
                answer = str(last.content)
                if verbose:
                    print(f"\n[final answer]\n{answer}")

        elif isinstance(last, ToolMessage):
            step = {
                "type": "observation",
                "name": last.name or "tool",
                "content": str(last.content),
            }
            steps.append(step)
            if verbose:
                preview = (
                    step["content"][:300] + "..."
                    if len(step["content"]) > 300
                    else step["content"]
                )
                print(f"\n[observation from {step['name']}]\n{preview}")

    return answer, steps


# Exposed for LangGraph CLI integrations.
# Keep module import safe when NEBIUS_API_KEY is not set.
try:
    graph = build_graph()
except EnvironmentError:
    graph = None
