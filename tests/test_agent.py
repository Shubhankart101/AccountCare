"""Local tests using a deterministic stub model; no network or API key needed."""
import json
import os
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from main import build_graph, execute_tool_calls, route_after_model, run_agent  # noqa: E402


class StubModel:
    """Returns scripted AIMessages in order and records each invocation."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies[len(self.calls) - 1]


def tool_call(name, args, call_id="c1"):
    return {"name": name, "args": args, "id": call_id}


def payload(msg):
    return json.loads(msg.content)


# route_after_model

def test_route_tools_when_tool_calls_present():
    msg = AIMessage(content="", tool_calls=[tool_call("check_subscription_status", {"user_id": "U-100"})])
    assert route_after_model({"messages": [HumanMessage(content="hi"), msg]}) == "tools"


def test_route_end_for_plain_ai_message():
    assert route_after_model({"messages": [AIMessage(content="done")]}) == "end"


def test_route_end_for_non_ai_message():
    assert route_after_model({"messages": [HumanMessage(content="hi")]}) == "end"


# execute_tool_calls

def test_execute_success():
    [msg] = execute_tool_calls([tool_call("check_subscription_status", {"user_id": "U-100"}, "abc")])
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "abc"
    assert msg.name == "check_subscription_status"
    body = payload(msg)
    assert body["ok"] is True
    assert body["output"]["plan"] == "Pro"


def test_execute_unknown_tool_does_not_run_anything():
    [msg] = execute_tool_calls([tool_call("drop_database", {"x": 1}, "z")])
    assert payload(msg) == {"ok": False, "error": "unknown_tool"}
    assert msg.tool_call_id == "z"
    assert msg.name == "drop_database"


def test_execute_exception_is_sanitized(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setitem(main.TOOL_REGISTRY, "check_subscription_status", boom)
    [msg] = execute_tool_calls([tool_call("check_subscription_status", {"user_id": "U-1"})])
    assert payload(msg) == {"ok": False, "error": "tool_execution_failed"}
    assert "secret" not in msg.content


def test_execute_bad_arguments_are_sanitized():
    [msg] = execute_tool_calls([tool_call("check_subscription_status", {"wrong": "arg"})])
    assert payload(msg) == {"ok": False, "error": "tool_execution_failed"}


def test_execute_multiple_calls_keep_order_and_ids():
    calls = [
        tool_call("check_subscription_status", {"user_id": "U-200"}, "1"),
        tool_call("nope", {}, "2"),
        tool_call("calculate_refund_eligibility", {"user_id": "U-100", "days_since_purchase": 30}, "3"),
    ]
    msgs = execute_tool_calls(calls)
    assert [m.tool_call_id for m in msgs] == ["1", "2", "3"]
    assert [m.name for m in msgs] == ["check_subscription_status", "nope", "calculate_refund_eligibility"]
    assert payload(msgs[0])["output"]["status"] == "past_due"
    assert payload(msgs[1])["error"] == "unknown_tool"
    assert payload(msgs[2])["output"]["eligible"] is False


def test_execute_empty_list():
    assert execute_tool_calls([]) == []


# full graph

def test_direct_answer_makes_single_model_call():
    model = StubModel([AIMessage(content="We accept credit cards and UPI.")])
    assert run_agent("What payment methods do you accept?", model) == "We accept credit cards and UPI."
    assert len(model.calls) == 1


def test_tool_round_trip():
    model = StubModel([
        AIMessage(content="", tool_calls=[tool_call("check_subscription_status", {"user_id": "U-100"})]),
        AIMessage(content="U-100 is on the Pro plan and active."),
    ])
    assert run_agent("Status of U-100?", model) == "U-100 is on the Pro plan and active."
    assert len(model.calls) == 2
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert payload(tool_msgs[0])["output"]["plan"] == "Pro"


def test_chained_tool_calls_loop_back_to_agent():
    model = StubModel([
        AIMessage(content="", tool_calls=[tool_call("calculate_refund_eligibility", {"user_id": "U-200", "days_since_purchase": 40}, "a")]),
        AIMessage(content="", tool_calls=[tool_call("escalate_to_human", {"user_id": "U-200", "reason": "refund dispute"}, "b")]),
        AIMessage(content="Ticket created."),
    ])
    assert run_agent("Refund for U-200, 40 days, I dispute this", model) == "Ticket created."
    assert len(model.calls) == 3


def test_graph_survives_unknown_tool_and_failure():
    model = StubModel([
        AIMessage(content="", tool_calls=[tool_call("ghost", {}, "g"), tool_call("check_subscription_status", {}, "h")]),
        AIMessage(content="Sorry, I could not retrieve that."),
    ])
    graph = build_graph(model)
    result = graph.invoke({"messages": [{"role": "user", "content": "check"}]})
    assert result["messages"][-1].content == "Sorry, I could not retrieve that."
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [payload(m)["error"] for m in tool_msgs] == ["unknown_tool", "tool_execution_failed"]


def test_system_prompt_is_sent_first():
    model = StubModel([AIMessage(content="ok")])
    run_agent("hello", model)
    first = model.calls[0][0]
    assert first.type == "system"
    assert first.content == main.load_system_prompt()


def test_blank_message_rejected():
    with pytest.raises(ValueError):
        run_agent("   ", StubModel([]))


# prompt sanity

@pytest.mark.parametrize("term", [
    "check_subscription_status", "calculate_refund_eligibility", "escalate_to_human", "user_id",
])
def test_prompt_mentions_key_terms(term):
    assert term in main.load_system_prompt()
