"""
Fraud Detection Pipeline Graph
===============================

Constructs the LangGraph ``StateGraph`` that wires all five agents into a
conditional pipeline with two routing decisions:

Routing decision 1 — after Monitoring (Check 1)
------------------------------------------------
    Zero anomaly flags  →  Fast Path  →  fast_path_defaults  →  Compliance
    Any anomaly flag    →  Full Pipeline  →  Pattern → Risk → Alert → …

Routing decision 2 — after Alert/Block (Check 4)
-------------------------------------------------
    action == "hold"  →  human_review  →  Compliance
    any other action  →  Compliance

Human-in-the-loop (Check 4 hold path)
--------------------------------------
The graph is compiled with ``interrupt_before=[NODE_HUMAN_REVIEW]``.
When a transaction scores 60-79 (hold), LangGraph pauses *before*
``human_review`` and saves state to the checkpointer.

A fraud analyst reviews the frozen state via the case management UI,
then resumes the graph (optionally updating ``action_taken``):

    compiled.invoke(None, config={"configurable": {"thread_id": run_id}})

The ``human_review`` node itself is a pass-through; it just logs the
analyst's decision so the compliance log is complete.

Full topology
-------------
    START
      └─► transaction_monitoring
            ├─[flags]──► pattern_detection
            │               └─► risk_assessment
            │                       └─► alert_block
            │                               ├─[hold]──► human_review ──► compliance_logging
            │                               └─[other]──────────────────► compliance_logging
            └─[clean]──► fast_path_defaults
                                └─► compliance_logging
                                        └─► END
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from fraud_detection.agents.alert_block import alert_block_agent
from fraud_detection.agents.compliance import compliance_logging_agent
from fraud_detection.agents.monitoring import transaction_monitoring_agent
from fraud_detection.agents.pattern_detection import pattern_detection_agent
from fraud_detection.agents.risk_assessment import risk_assessment_agent
from fraud_detection.core.state import (
    ActionResult,
    FraudDetectionState,
    RiskBreakdown,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node identifiers — centralised to avoid magic strings
# ---------------------------------------------------------------------------
NODE_MONITORING    = "transaction_monitoring"
NODE_PATTERN       = "pattern_detection"
NODE_RISK          = "risk_assessment"
NODE_ALERT         = "alert_block"
NODE_HUMAN_REVIEW  = "human_review"
NODE_FAST_DEFAULTS = "fast_path_defaults"
NODE_COMPLIANCE    = "compliance_logging"


# ---------------------------------------------------------------------------
# Fast-path defaults  (clean transactions that skip Checks 2-4)
# ---------------------------------------------------------------------------

def fast_path_defaults_node(state: FraudDetectionState) -> dict:
    """
    Set sensible defaults for transactions that passed Check 1 with zero flags.

    Without this node, risk_score and action_taken would be absent from state,
    producing incomplete audit entries in Check 5.
    """
    txn = state.get("raw_transaction", {})
    txn_id = txn.get("transaction_id", "UNKNOWN")
    logger.info(
        "Transaction %s: FAST PATH — zero anomaly flags, bypassing Checks 2-4",
        txn_id,
    )
    return {
        "risk_score": 0.0,
        "risk_breakdown": RiskBreakdown(
            anomaly_score=0.0,
            pattern_score=0.0,
            historical_score=0.0,
            velocity_score=0.0,
            model_score=0.0,
        ),
        "action_taken": ActionResult(
            action="allow",
            reason="Fast path: zero anomaly flags detected by monitoring agent",
            notified_parties=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop review node  (hold path — Check 4 → Check 5 bridge)
# ---------------------------------------------------------------------------

def human_review_node(state: FraudDetectionState) -> dict:
    """
    Pass-through node that acts as the pause point for held transactions.

    Execution is always interrupted *before* this node
    (``interrupt_before=[NODE_HUMAN_REVIEW]``).  A fraud analyst then:

      1. Inspects the frozen ``FraudDetectionState`` in the case UI
      2. Optionally updates ``action_taken`` via a state-update call
      3. Resumes the graph — this node runs, logs the outcome, and
         routes execution to Compliance (Check 5)

    The node itself does not change state; it exists so the graph has a
    named pause point with a clear audit log entry.
    """
    txn = state.get("raw_transaction", {})
    txn_id = txn.get("transaction_id", "UNKNOWN")
    action = state.get("action_taken", {})
    logger.info(
        "[HUMAN REVIEW] Transaction %s reviewed — final action: %s",
        txn_id,
        action.get("action", "unknown").upper(),
    )
    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_monitoring(state: FraudDetectionState) -> str:
    """Route clean transactions to the fast path; flagged ones to full pipeline."""
    if state.get("is_anomalous"):
        return NODE_PATTERN
    return NODE_FAST_DEFAULTS


def _route_after_alert(state: FraudDetectionState) -> str:
    """Route held transactions to human review; all others straight to compliance."""
    action = state.get("action_taken", {})
    if action.get("action") == "hold":
        return NODE_HUMAN_REVIEW
    return NODE_COMPLIANCE


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_fraud_detection_graph() -> StateGraph:
    """
    Construct (but do not compile) the fraud-detection pipeline graph.

    Returns the ``StateGraph`` so callers can inspect or further extend it
    before compiling.
    """
    graph = StateGraph(FraudDetectionState)

    # Register nodes
    graph.add_node(NODE_MONITORING,    transaction_monitoring_agent)
    graph.add_node(NODE_PATTERN,       pattern_detection_agent)
    graph.add_node(NODE_RISK,          risk_assessment_agent)
    graph.add_node(NODE_ALERT,         alert_block_agent)
    graph.add_node(NODE_HUMAN_REVIEW,  human_review_node)
    graph.add_node(NODE_FAST_DEFAULTS, fast_path_defaults_node)
    graph.add_node(NODE_COMPLIANCE,    compliance_logging_agent)

    # Entry point
    graph.add_edge(START, NODE_MONITORING)

    # Routing decision 1: zero flags → fast path, any flag → full pipeline
    graph.add_conditional_edges(
        NODE_MONITORING,
        _route_after_monitoring,
        {
            NODE_PATTERN:       NODE_PATTERN,
            NODE_FAST_DEFAULTS: NODE_FAST_DEFAULTS,
        },
    )

    # Fast path: set defaults then go straight to compliance
    graph.add_edge(NODE_FAST_DEFAULTS, NODE_COMPLIANCE)

    # Full pipeline: pattern → risk → alert
    graph.add_edge(NODE_PATTERN, NODE_RISK)
    graph.add_edge(NODE_RISK,    NODE_ALERT)

    # Routing decision 2: hold → human review, others → compliance
    graph.add_conditional_edges(
        NODE_ALERT,
        _route_after_alert,
        {
            NODE_HUMAN_REVIEW: NODE_HUMAN_REVIEW,
            NODE_COMPLIANCE:   NODE_COMPLIANCE,
        },
    )

    # Human review always flows into compliance after analyst resumes
    graph.add_edge(NODE_HUMAN_REVIEW, NODE_COMPLIANCE)

    graph.add_edge(NODE_COMPLIANCE, END)

    return graph


def compile_fraud_detection_graph(checkpointer=None):
    """
    Build and compile the fraud-detection pipeline into an executable graph.

    Parameters
    ----------
    checkpointer
        A LangGraph checkpointer instance.  Defaults to ``MemorySaver``
        (in-process, suitable for dev/test).  In production, swap for a
        ``PostgresSaver`` or ``RedisSaver`` so hold-state survives restarts.

    The graph is compiled with ``interrupt_before=[NODE_HUMAN_REVIEW]`` so
    transactions scoring 60-79 automatically pause for analyst review.

    Returns a ``CompiledGraph`` ready to be invoked:

        result = compiled.invoke(
            {"raw_transaction": {...}, "pipeline_run_id": "..."},
            config={"configurable": {"thread_id": run_id}},
        )
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = build_fraud_detection_graph()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[NODE_HUMAN_REVIEW],
    )

    logger.info(
        "Pipeline compiled | fast-path shortcut: enabled | "
        "human-in-the-loop: interrupt_before=%s",
        NODE_HUMAN_REVIEW,
    )
    return compiled
