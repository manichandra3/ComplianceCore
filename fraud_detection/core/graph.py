"""
Fraud Detection Pipeline Graph
===============================

Constructs the LangGraph `StateGraph` that wires agents into a
conditional pipeline:

    START -> Monitoring --(anomalous?)--> Pattern Detection -> Risk Assessment
                          \                                     -> Alert/Block -> Compliance -> END
                           \-(clean)---> Fast-Path Defaults ----/

Clean transactions (~95% of volume) are short-circuited after Monitoring:
they skip Pattern Detection, Risk Assessment, and Alert/Block, receiving
sensible defaults before going straight to Compliance logging.

Architecture decisions:
    - Conditional routing after Monitoring reduces P95 latency for
      legitimate transactions while preserving full analysis for
      suspicious ones.
    - Agents communicate exclusively through the shared state dict.
    - `Annotated[list, operator.add]` reducers on list fields ensure
      each agent *appends* to lists rather than overwriting.

Future extensions:
    - Add a human-in-the-loop breakpoint before the Alert/Block node
      for transactions in the "hold" risk band.
    - Add parallel branches for independent enrichment (e.g. device
      intelligence + geo lookup running concurrently).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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
# Node identifiers -- centralised to avoid magic strings
# ---------------------------------------------------------------------------
NODE_MONITORING = "transaction_monitoring"
NODE_PATTERN = "pattern_detection"
NODE_RISK = "risk_assessment"
NODE_ALERT = "alert_block"
NODE_COMPLIANCE = "compliance_logging"
NODE_FAST_PATH = "fast_path_defaults"


# ---------------------------------------------------------------------------
# Fast-path node -- provides defaults for clean transactions
# ---------------------------------------------------------------------------

def _fast_path_defaults(state: FraudDetectionState) -> dict:
    """Set default risk_score, risk_breakdown, and action_taken for clean
    transactions that bypass Pattern Detection, Risk Assessment, and
    Alert/Block.

    This ensures the Compliance node always has complete state to work with.
    """
    txn = state.get("raw_transaction", {})
    txn_id = txn.get("transaction_id", "UNKNOWN")

    logger.info(
        f"Fast-path: transaction {txn_id} is clean -- "
        f"setting defaults and skipping to compliance"
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
            reason="No anomalies detected -- fast-path clean transaction",
            notified_parties=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    }


# ---------------------------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------------------------

def _route_after_monitoring(state: FraudDetectionState) -> str:
    """Route clean transactions to the fast-path node, anomalous ones to
    the full analysis pipeline."""
    if state.get("is_anomalous"):
        return NODE_PATTERN
    return NODE_FAST_PATH


def build_fraud_detection_graph() -> StateGraph:
    """
    Construct (but do not compile) the fraud-detection pipeline graph.

    Returns the `StateGraph` so callers can inspect or extend it before
    compiling.

    Graph topology:
        START -> Monitoring --[anomalous]--> Pattern Detection
                            |                    -> Risk Assessment
                            |                    -> Alert/Block
                            |                    -> Compliance -> END
                            +-[clean]--------> Fast-Path Defaults
                                                 -> Compliance -> END

    # ── FUTURE: Human-in-the-Loop ────────────────────────────────────
    # Insert a LangGraph `interrupt` before the Alert/Block node so
    # that "hold" decisions can be routed to a human analyst:
    #
    #   from langgraph.checkpoint.memory import MemorySaver
    #   graph.add_node(NODE_ALERT, alert_block_agent,
    #                  interrupt_before=True)
    #
    # The analyst reviews the case in the UI, then resumes the graph
    # with an approved/rejected decision.
    # ------------------------------------------------------------------
    """
    graph = StateGraph(FraudDetectionState)

    # Register agent nodes
    graph.add_node(NODE_MONITORING, transaction_monitoring_agent)
    graph.add_node(NODE_PATTERN, pattern_detection_agent)
    graph.add_node(NODE_RISK, risk_assessment_agent)
    graph.add_node(NODE_ALERT, alert_block_agent)
    graph.add_node(NODE_COMPLIANCE, compliance_logging_agent)
    graph.add_node(NODE_FAST_PATH, _fast_path_defaults)

    # Wire the pipeline with conditional routing after Monitoring
    graph.add_edge(START, NODE_MONITORING)

    # Conditional edge: anomalous -> full pipeline, clean -> fast-path
    graph.add_conditional_edges(
        NODE_MONITORING,
        _route_after_monitoring,
        {NODE_PATTERN: NODE_PATTERN, NODE_FAST_PATH: NODE_FAST_PATH},
    )

    # Full analysis path
    graph.add_edge(NODE_PATTERN, NODE_RISK)
    graph.add_edge(NODE_RISK, NODE_ALERT)
    graph.add_edge(NODE_ALERT, NODE_COMPLIANCE)

    # Fast-path merges back into Compliance
    graph.add_edge(NODE_FAST_PATH, NODE_COMPLIANCE)

    graph.add_edge(NODE_COMPLIANCE, END)

    return graph


def compile_fraud_detection_graph():
    """
    Build and compile the fraud-detection pipeline into an executable graph.

    Returns a `CompiledGraph` ready to be invoked:
        result = compiled.invoke({"raw_transaction": {...}})

    # ── FUTURE: Checkpointing / Persistence ──────────────────────────
    # Add a checkpointer so pipeline state survives process restarts:
    #
    #   from langgraph.checkpoint.postgres import PostgresSaver
    #   checkpointer = PostgresSaver(conn_string="postgresql://...")
    #   compiled = graph.compile(checkpointer=checkpointer)
    #
    # This also enables the human-in-the-loop workflow (the graph can
    # be resumed from the last checkpoint after analyst review).
    # ------------------------------------------------------------------
    """
    graph = build_fraud_detection_graph()
    compiled = graph.compile()
    logger.info(
        "Pipeline compiled with conditional routing: "
        "clean transactions skip Pattern/Risk/Alert via fast-path"
    )
    return compiled
