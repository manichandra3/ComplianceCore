"""
Compliance Logging Agent
========================

Pipeline position: **Node 5** (terminal node)

Responsibility:
    - Generate a complete audit trail for every transaction processed
    - Auto-draft Suspicious Activity Reports (SARs) when risk exceeds
      the SAR threshold
    - Produce structured compliance log entries referencing applicable
      regulations (BSA/AML, OFAC, PCI-DSS, etc.)
    - Write `compliance_logs` into state

This agent ensures the institution meets its regulatory obligations and
maintains a defensible record of every fraud-detection decision.
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
    """Auto-draft a Suspicious Activity Report if risk exceeds SAR threshold.

    # ── FUTURE: LLM-Generated SAR Narrative ──────────────────────────
    # FinCEN SARs require a detailed narrative section.  Use an LLM to
    # draft this automatically:
    #   from langchain_core.prompts import ChatPromptTemplate
    #   sar_prompt = ChatPromptTemplate.from_messages([
    #       ("system", "You are a BSA/AML compliance officer..."),
    #       ("human", "Draft a SAR narrative for: {transaction_summary}")
    #   ])
    #   narrative = (sar_prompt | llm | StrOutputParser()).invoke({
    #       "transaction_summary": build_summary(state)
    #   })
    #
    # The LLM-generated narrative will include:
    #   - Description of suspicious activity
    #   - How it was detected (rules, patterns, ML signals)
    #   - Involved parties and accounts
    #   - Estimated exposure amount
    #   - Recommended next steps
    # ------------------------------------------------------------------

    # ── FUTURE: Neo4j Relationship Context ───────────────────────────
    # Enrich the SAR with relationship data from Neo4j:
    #   MATCH (a:Account {id: $account_id})-[r*1..3]-(related)
    #   RETURN related, type(r), labels(related)
    # This provides the "associated accounts and entities" section of
    # the SAR filing.
    # ------------------------------------------------------------------
    """
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
    LangGraph node: Compliance Logging Agent.

    Reads
    -----
    - ALL state fields (full pipeline context for audit)

    Writes
    ------
    - compliance_logs : list[ComplianceEntry]  (appended via reducer)

    This is the terminal node.  Its output completes the pipeline state
    that is returned to the caller.
    """
    logger.info("=== Compliance Logging Agent: START ===")

    txn = state.get("raw_transaction", {})
    txn_id = txn.get("transaction_id", "UNKNOWN")
    risk_score = state.get("risk_score", 0.0)

    logs: list[ComplianceEntry] = []

    # 1. Always generate an audit trail entry
    audit_entry = _generate_audit_entry(state)
    logs.append(audit_entry)
    logger.info(f"  Generated audit entry: {audit_entry['log_id']}")

    # 2. Always log the action taken
    action_log = _generate_action_log(state)
    logs.append(action_log)
    logger.info(f"  Generated action log: {action_log['log_id']}")

    # 3. Auto-draft SAR if risk is high enough
    sar = _generate_sar_draft(state)
    if sar:
        logs.append(sar)
        logger.warning(
            f"  AUTO-GENERATED SAR DRAFT: {sar['log_id']} "
            f"(risk_score={risk_score})"
        )
    else:
        logger.info(f"  SAR not required (risk_score={risk_score} < threshold={SAR_THRESHOLD})")

    logger.info(f"Transaction {txn_id}: {len(logs)} compliance entries generated")
    logger.info("=== Compliance Logging Agent: END ===\n")

    return {"compliance_logs": logs}
