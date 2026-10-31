"""Supplied AccountCare tools and registry. Do not modify.

Three tools, exposed both as LangChain @tool objects (for binding to the
model) and as a plain-function registry (for execute_tool_calls to invoke
without going through the LangChain tool wrapper).
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


ACCOUNTS: dict[str, dict[str, Any]] = {
    "U-100": {
        "plan": "Pro",
        "status": "active",
        "renewal_date": "2026-10-01",
        "purchase_date": "2026-08-20",
    },
    "U-200": {
        "plan": "Basic",
        "status": "past_due",
        "renewal_date": "2026-09-10",
        "purchase_date": "2026-07-01",
    },
}


def _check_subscription_status(user_id: str) -> dict[str, Any]:
    account = ACCOUNTS.get(user_id)
    if account is None:
        return {"found": False, "user_id": user_id}
    return {
        "found": True,
        "user_id": user_id,
        "plan": account["plan"],
        "status": account["status"],
        "renewal_date": account["renewal_date"],
    }


def _calculate_refund_eligibility(user_id: str, days_since_purchase: int) -> dict[str, Any]:
    account = ACCOUNTS.get(user_id)
    eligible = days_since_purchase <= 14
    amount = 0.0
    if eligible and account is not None:
        amount = 49.0 if account["plan"] == "Pro" else 19.0
    return {"user_id": user_id, "eligible": eligible, "refund_amount": amount}


def _escalate_to_human(user_id: str, reason: str) -> dict[str, Any]:
    return {
        "ticket_id": f"ESC-{user_id}",
        "user_id": user_id,
        "reason": reason,
        "escalated": True,
    }


@tool
def check_subscription_status(user_id: str) -> dict[str, Any]:
    """Look up the current plan, status, and renewal date for a subscriber's account. Requires the account's user_id."""
    return _check_subscription_status(user_id)


@tool
def calculate_refund_eligibility(user_id: str, days_since_purchase: int) -> dict[str, Any]:
    """Determine whether a subscriber qualifies for a refund, based on how many days have passed since purchase (refunds are only available within 14 days). Requires user_id and days_since_purchase."""
    return _calculate_refund_eligibility(user_id, days_since_purchase)


@tool
def escalate_to_human(user_id: str, reason: str) -> dict[str, Any]:
    """Create a ticket that routes the subscriber to a human support agent. Does not contact a human directly or resolve the issue itself. Requires user_id and a short reason."""
    return _escalate_to_human(user_id, reason)


TOOLS = [check_subscription_status, calculate_refund_eligibility, escalate_to_human]

TOOL_REGISTRY = {
    "check_subscription_status": _check_subscription_status,
    "calculate_refund_eligibility": _calculate_refund_eligibility,
    "escalate_to_human": _escalate_to_human,
}
