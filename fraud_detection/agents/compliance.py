"""
Compliance Logging Agent  (Check 5 — The Reporter & Broadcaster)
================================================================

Pipeline position: **Node 5** (terminal node)

Responsibility:
    1. Audit trail   — versioned LangGraph state written to storage for
                       regulators; every decision is fully explainable.
    2. SAR generation — auto-draft Suspicious Activity Reports when
                        risk_score >= SAR_THRESHOLD.
    3. Kafka broadcast — if the final action is ALLOW, publish an event
                         to Kafka/SQS so the async feedback worker wakes up,
                         fetches the transaction, and recalculates μ / σ
                         in DynamoDB.  This is the learning feedback loop
                         that makes the system smarter after each approval.

Writes to state
---------------
    compliance_logs         list[ComplianceEntry]  (appended via reducer)
    kafka_event_published   bool
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fraud_detection.config.settings import SAR_THRESHOLD
from fraud_detection.core.state import ComplianceEntry, FraudDetectionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compliance report generators
# ---------------------------------------------------------------------------

def _generate_audit_entry(state: FraudDetectionState) -> ComplianceEntry:
    """Generate a standard audit log entry summarising the pipeline run."""
    txn = state.get("raw_transaction", {})
    action = state.get("action_taken", {})
    risk_score = state.get("risk_score", 0.0)
    flags = state.get("anomaly_flags", [])
    patterns = state.get("detected_patterns", [])
    run_id = state.get("pipeline_run_id", "N/A")

    flag_summary = ", ".join(f["rule_id"] for f in flags) if flags else "none"
    pattern_summary = ", ".join(p["pattern_id"] for p in patterns) if patterns else "none"

    return ComplianceEntry(
        log_id=f"AUDIT-{uuid.uuid4().hex[:12].upper()}",
        event_type="PIPELINE_AUDIT",
        summary=(
            f"Run {run_id} | "
            f"Transaction {txn.get('transaction_id', 'N/A')} | "
            f"Amount: ${txn.get('amount', 0):,.2f} | "
            f"Risk: {risk_score}/100 | "
            f"Action: {action.get('action', 'N/A').upper()} | "
            f"Flags: [{flag_summary}] | "
            f"Patterns: [{pattern_summary}]"
        ),
        regulatory_references=["BSA/AML", "31 CFR 1010", "PCI-DSS 3.2.1"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _generate_sar_draft(state: FraudDetectionState) -> ComplianceEntry | None:
    """Auto-draft a Suspicious Activity Report if risk exceeds SAR threshold."""
    risk_score = state.get("risk_score", 0.0)

    if risk_score < SAR_THRESHOLD:
        return None

    txn = state.get("raw_transaction", {})
    patterns = state.get("detected_patterns", [])
    run_id = state.get("pipeline_run_id", "N/A")

    pattern_details = "; ".join(
        f"{p['pattern_id']} ({p['pattern_type']}, conf={p['confidence']:.0%})"
        for p in patterns
    ) if patterns else "N/A"

    return ComplianceEntry(
        log_id=f"SAR-{uuid.uuid4().hex[:12].upper()}",
        event_type="SAR_GENERATED",
        summary=(
            f"[AUTO-DRAFT SAR] Run {run_id} | "
            f"Transaction {txn.get('transaction_id', 'N/A')} "
            f"from account {txn.get('account_id', 'N/A')} | "
            f"Amount: ${txn.get('amount', 0):,.2f} | "
            f"Risk Score: {risk_score}/100 | "
            f"Detected Patterns: {pattern_details} | "
            f"Action Taken: {state.get('action_taken', {}).get('action', 'N/A').upper()} | "
            f"NARRATIVE: [Placeholder -- LLM-generated narrative will appear here]"
        ),
        regulatory_references=[
            "BSA/AML",
            "31 CFR 1020.320",   # SAR filing requirement
            "FinCEN Form 111",   # SAR form
            "31 CFR 1010.310",   # Recordkeeping
        ],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _generate_action_log(state: FraudDetectionState) -> ComplianceEntry:
    """Log the specific action taken for regulatory recordkeeping."""
    action = state.get("action_taken", {})
    txn = state.get("raw_transaction", {})
    run_id = state.get("pipeline_run_id", "N/A")

    action_type = action.get("action", "unknown").upper()
    event_type_map = {
        "BLOCK": "TRANSACTION_BLOCKED",
        "HOLD": "TRANSACTION_HELD",
        "FLAG": "TRANSACTION_FLAGGED",
        "ALLOW": "TRANSACTION_ALLOWED",
    }

    return ComplianceEntry(
        log_id=f"ACTION-{uuid.uuid4().hex[:12].upper()}",
        event_type=event_type_map.get(action_type, "TRANSACTION_PROCESSED"),
        summary=(
            f"Run {run_id} | "
            f"Action '{action_type}' executed on transaction "
            f"{txn.get('transaction_id', 'N/A')} | "
            f"Reason: {action.get('reason', 'N/A')} | "
            f"Notified: {', '.join(action.get('notified_parties', []))}"
        ),
        regulatory_references=["BSA/AML", "12 CFR 21.11"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def compliance_logging_agent(state: FraudDetectionState) -> dict:
    """
    LangGraph node: Compliance Logging Agent (Check 5).

    Reads
    -----
    - ALL state fields (full pipeline context for audit)

    Writes
    ------
    - compliance_logs        : list[ComplianceEntry]  (appended via reducer)
    - kafka_event_published  : bool
    """
    logger.debug("=== Compliance Logging Agent (Check 5): START ===")

    txn = state.get("raw_transaction", {})
    txn_id = txn.get("transaction_id", "UNKNOWN")
    risk_score = state.get("risk_score", 0.0)
    action = state.get("action_taken", {})
    run_id = state.get("pipeline_run_id", "N/A")

    logs: list[ComplianceEntry] = []

    # 1. Always generate an audit trail entry (regulatory requirement)
    audit_entry = _generate_audit_entry(state)
    logs.append(audit_entry)
    logger.debug(f"  Generated audit entry: {audit_entry['log_id']}")

    # 2. Always log the action taken
    action_log = _generate_action_log(state)
    logs.append(action_log)
    logger.debug(f"  Generated action log: {action_log['log_id']}")

    # 3. Auto-draft SAR if risk is high enough
    sar = _generate_sar_draft(state)
    if sar:
        logs.append(sar)
        logger.warning(
            f"  AUTO-GENERATED SAR DRAFT: {sar['log_id']} "
            f"(risk_score={risk_score})"
        )
    else:
        logger.debug(
            f"  SAR not required (risk_score={risk_score} < threshold={SAR_THRESHOLD})"
        )

    # 4. Feedback loop: broadcast ALLOWED transactions to Kafka/SQS so the
    #    async worker recalculates μ and σ in DynamoDB, making the system
    #    smarter for this user's next transaction.
    kafka_published = False
    if action.get("action") == "allow":
        from fraud_detection.core.kafka_client import publish_allowed_transaction

        account_id = str(txn.get("account_id", ""))
        amount = float(txn.get("amount", 0.0))
        kafka_published = publish_allowed_transaction(
            pipeline_run_id=run_id,
            account_id=account_id,
            transaction_id=txn_id,
            amount=amount,
        )
        if kafka_published:
            logger.debug(
                "  Kafka/SQS event published for allowed transaction %s", txn_id
            )
        else:
            logger.warning(
                "  Kafka/SQS unavailable — feedback event logged only for %s", txn_id
            )

    logger.debug(f"Transaction {txn_id}: {len(logs)} compliance entries generated")
    logger.debug("=== Compliance Logging Agent (Check 5): END ===\n")

    return {
        "compliance_logs": logs,
        "kafka_event_published": kafka_published,
    }
