"""AccountCare Support Agent — LangGraph routing over validated tool calls.

Scenario
--------
AccountCare's support agent answers subscriber questions about billing and
plans. General questions ("what payment methods do you accept?") should be
answered directly, with no tool call. Account-specific questions (status,
refund eligibility, escalation) must always go through a tool — the model
must never guess or invent account data.

Your task
---------
Complete only the three TODO functions in this file:

1. route_after_model
2. execute_tool_calls
3. build_graph

You must also write utils/../prompt.txt yourself (see prompt.txt in this
folder) — the system prompt that decides when the model answers directly
versus calls a tool, what it does when user_id is missing, and when it must
escalate.

The Groq-backed model builder, the three account tools, and the graph state
are supplied. During evaluation, pytest supplies a stub/mocked model, so no
real API call, key, or network access is used. Do not modify the supplied
tool implementations (utils/account_tools.py) or the state definition
(utils/state.py).
"""
from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from utils.account_tools import TOOLS, TOOL_REGISTRY
from utils.state import AgentState

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


def build_llm() -> ChatGroq:
    """Supplied. Builds the Groq-backed chat model used outside of tests."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is required to call Groq.")
    return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_MODEL), api_key=api_key)


def load_system_prompt() -> str:
    """Supplied. Reads the system prompt you write in prompt.txt."""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def agent_node(state: AgentState, llm_with_tools: Any) -> dict[str, Any]:
    """Supplied. Calls the model with your system prompt plus the running
    message history, and appends its reply to state."""
    system = SystemMessage(content=load_system_prompt())
    response = llm_with_tools.invoke([system, *state["messages"]])
    return {"messages": [response]}


def route_after_model(state: AgentState) -> str:
    """Decide whether the graph goes to the tools node or ends.

    TODO 1 contract:
    1. Look at the most recent message in state["messages"].
    2. If it is an AIMessage carrying one or more tool calls (a non-empty
       `tool_calls` attribute), return the string "tools".
    3. Otherwise, return the string "end".
    4. Do not execute any tool here, and do not mutate state.
    """
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def execute_tool_calls(tool_calls: list[dict[str, Any]]) -> list[ToolMessage]:
    """Execute every tool call requested in a single model turn.

    TODO 2 contract:
    1. For each entry in tool_calls (each a dict with at least "name",
       "args", and "id"), look up the handler in TOOL_REGISTRY by name.
    2. If the name is not present in TOOL_REGISTRY, do not raise — build a
       ToolMessage whose content is
       json.dumps({"ok": False, "error": "unknown_tool"}, sort_keys=True)
       and whose tool_call_id is tool_calls[i]["id"].
    3. Otherwise, call the handler with `**tool_calls[i]["args"]`.
       - On success, build a ToolMessage whose content is
         json.dumps({"ok": True, "output": <return value>}, sort_keys=True).
       - On any exception, catch it and build a ToolMessage whose content is
         json.dumps({"ok": False, "error": "tool_execution_failed"}, sort_keys=True).
         Never include the raw exception text or message in the output.
    4. Every ToolMessage must carry the matching tool_calls[i]["id"] as its
       tool_call_id and tool_calls[i]["name"] as its name.
    5. Return one ToolMessage per input tool call, in the same order as the
       input list.
    """
    results: list[ToolMessage] = []
    for call in tool_calls:
        handler = TOOL_REGISTRY.get(call["name"])
        if handler is None:
            payload = {"ok": False, "error": "unknown_tool"}
        else:
            try:
                payload = {"ok": True, "output": handler(**(call.get("args") or {}))}
                json.dumps(payload, sort_keys=True)
            except Exception:
                payload = {"ok": False, "error": "tool_execution_failed"}
        results.append(
            ToolMessage(
                content=json.dumps(payload, sort_keys=True),
                tool_call_id=call["id"],
                name=call["name"],
            )
        )
    return results


def tools_node(state: AgentState) -> dict[str, Any]:
    """Supplied. Extracts tool calls from the last message and dispatches
    them through execute_tool_calls."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    return {"messages": execute_tool_calls(tool_calls)}


def build_graph(llm_with_tools: Any) -> Any:
    """Assemble and compile the AccountCare agent graph.

    TODO 3 contract:
    1. Create a StateGraph over AgentState.
    2. Add a node named "agent" that calls
       agent_node(state, llm_with_tools).
    3. Add a node named "tools" that calls tools_node(state).
    4. Wire START -> "agent".
    5. Add a conditional edge out of "agent" using route_after_model as the
       routing function, mapping its "tools" return value to the "tools"
       node and its "end" return value to END.
    6. Wire "tools" -> "agent" (so the agent sees tool results and can
       either respond or call another tool).
    7. Compile and return the graph.
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_model, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_agent(user_message: str, llm_with_tools: Any = None) -> str:
    """Supplied entry point used by the tests and the app."""
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must not be blank.")

    llm = llm_with_tools if llm_with_tools is not None else build_llm().bind_tools(TOOLS)
    graph = build_graph(llm)
    result = graph.invoke({"messages": [{"role": "user", "content": user_message.strip()}]})
    final_message = result["messages"][-1]
    return str(getattr(final_message, "content", "") or "")


def main() -> None:
    queries = [
        "What payment methods do you accept?",
        "Can you check the status of account U-100?",
        "I want a refund for account U-200, I bought it 5 days ago.",
        "This is taking too long, get me a human for account U-100, I'm frustrated.",
    ]
    print("AccountCare Support Agent")
    print("=" * 40)
    for query in queries:
        print(f"\nQuery: {query}")
        print(f"Response: {run_agent(query)}")
        print("-" * 40)


if __name__ == "__main__":
    main()
