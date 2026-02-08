"""
Alert / Block Agent
===================

Pipeline position: **Node 4**

Responsibility:
    - Evaluate the composite risk score against configured thresholds
    - Execute the appropriate action: allow, flag, hold, or block
    - Notify relevant parties (fraud ops, customer, issuing bank)
    - Write `action_taken` into state

This is the enforcement point of the pipeline.  In production this agent
will call real downstream systems (payment switch, case management, SMS
gateway, etc.).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fraud_detection.config.settings import (
    RISK_THRESHOLD_BLOCK,
    RISK_THRESHOLD_FLAG,
    RISK_THRESHOLD_HOLD,
)
from fraud_detection.core.state import ActionResult, FraudDetectionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action executors  (placeholder implementations)
# ---------------------------------------------------------------------------

def _execute_block(txn_id: str, risk_score: float) -> ActionResult:
    """Block the transaction and notify all parties.

    # ── FUTURE: Payment Switch Integration ───────────────────────────
    # Call the payment switch API to decline the authorisation:
    #   payment_switch.decline(txn_id, reason="FRAUD_BLOCK")
    # Simultaneously open a case in the case management system:
    #   case_mgmt.create_case(txn_id, severity="critical")
    # ------------------------------------------------------------------
    """
    logger.warning(f"BLOCKING transaction {txn_id} (risk={risk_score})")
    return ActionResult(
        action="block",
        reason=f"Risk score {risk_score} >= block threshold {RISK_THRESHOLD_BLOCK}",
        notified_parties=["fraud_ops", "customer", "issuer", "compliance"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _execute_hold(txn_id: str, risk_score: float) -> ActionResult:
    """Hold the transaction for manual review.

    # ── FUTURE: Case Management Integration ──────────────────────────
    # Create a pending-review case:
    #   case_mgmt.create_case(txn_id, severity="high", status="pending_review")
    # Route to the appropriate analyst queue based on pattern type.
    # ------------------------------------------------------------------
    """
    logger.warning(f"HOLDING transaction {txn_id} for review (risk={risk_score})")
    return ActionResult(
        action="hold",
        reason=f"Risk score {risk_score} >= hold threshold {RISK_THRESHOLD_HOLD}",
        notified_parties=["fraud_ops", "customer"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _execute_flag(txn_id: str, risk_score: float) -> ActionResult:
    """Allow the transaction but flag it for monitoring."""
    logger.info(f"FLAGGING transaction {txn_id} (risk={risk_score})")
    return ActionResult(
        action="flag",
        reason=f"Risk score {risk_score} >= flag threshold {RISK_THRESHOLD_FLAG}",
        notified_parties=["fraud_ops"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _execute_allow(txn_id: str, risk_score: float) -> ActionResult:
    """Allow the transaction without any flags."""
    logger.info(f"ALLOWING transaction {txn_id} (risk={risk_score})")
    return ActionResult(
        action="allow",
        reason=f"Risk score {risk_score} below all thresholds",
        notified_parties=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def alert_block_agent(state: FraudDetectionState) -> dict:
    """
    LangGraph node: Alert / Block Agent.

    Reads
    -----
    - state["raw_transaction"]
    - state["risk_score"]

    Writes
    ------
    - action_taken : ActionResult

    # ── FUTURE: Reinforcement Learning Action Selection ──────────────
    # Instead of fixed threshold-based decisions, use an RL policy:
    #   rl_state = RLState(
    #       risk_score=risk_score,
    #       customer_value=customer_service.get_ltv(account_id),
    #       txn_context=txn,
    #   )
    #   action = rl_policy.select_action(rl_state)
    #
    # The RL policy learns to balance:
    #   - Fraud loss prevention  (reward for catching fraud)
    #   - Customer experience    (penalty for false blocks)
    #   - Regulatory compliance  (hard constraint: never allow known fraud)
    # ------------------------------------------------------------------

    # ── FUTURE: LLM-Generated Customer Communication ─────────────────
    # When blocking or holding, generate a personalised customer
    # notification using an LLM:
    #   message = llm.invoke(
    #       "Draft a polite SMS/email to inform the customer that "
    #       "transaction {txn_id} has been held for security review."
    #   )
    #   sms_gateway.send(customer_phone, message)
    # ------------------------------------------------------------------
    """
    logger.info("=== Alert / Block Agent: START ===")

    txn = state.get("raw_transaction", {})
    risk_score = state.get("risk_score", 0.0)
    txn_id = txn.get("transaction_id", "UNKNOWN")

    # Threshold-based action selection
    if risk_score >= RISK_THRESHOLD_BLOCK:
        action = _execute_block(txn_id, risk_score)
    elif risk_score >= RISK_THRESHOLD_HOLD:
        action = _execute_hold(txn_id, risk_score)
    elif risk_score >= RISK_THRESHOLD_FLAG:
        action = _execute_flag(txn_id, risk_score)
    else:
        action = _execute_allow(txn_id, risk_score)

    logger.info(
        f"Transaction {txn_id}: ACTION = {action['action'].upper()} | "
        f"reason = {action['reason']}"
    )
    if action["notified_parties"]:
        logger.info(f"  Notified: {', '.join(action['notified_parties'])}")

    logger.info("=== Alert / Block Agent: END ===\n")

    return {"action_taken": action}
