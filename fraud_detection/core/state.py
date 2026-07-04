"""
FraudDetectionState - Central state schema for the multi-agent pipeline.
========================================================================

This TypedDict is the single source of truth passed between every agent node
in the LangGraph workflow. Each agent reads what it needs and writes back its
results. LangGraph merges updates automatically via its reducer semantics.

Architecture note:
    LangGraph state channels use "last-writer-wins" by default for scalar
    fields.  For list fields (detected_patterns, anomaly_flags,
    compliance_logs) we use the `Annotated[list, operator.add]` pattern so
    every agent *appends* rather than overwrites, preserving the full
    audit trail across the pipeline.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict, NotRequired


# ---------------------------------------------------------------------------
# Sub-schemas -- keep nested dicts well-typed for downstream consumers
# ---------------------------------------------------------------------------

class TransactionData(TypedDict, total=False):
    """Raw transaction payload ingested from the payment gateway / core
    banking API.  `total=False` makes every key optional so the schema
    tolerates partial data during early pipeline stages."""

    transaction_id: str
    account_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_category: str
    timestamp: str                 # ISO-8601
    location: str
    channel: str                   # e.g. "online", "pos", "atm"
    ip_address: str
    device_fingerprint: str
    recipient_account: str
    transaction_type: str          # e.g. "purchase", "transfer", "withdrawal"
    metadata: dict[str, Any]       # catch-all for gateway-specific fields


class UserProfile(TypedDict, total=False):
    """Statistical baseline for a user fetched from DynamoDB.

    Populated by the Transaction Monitoring Agent on every request.
    Written by the async feedback worker after each allowed transaction
    (triggered via Kafka/SQS broadcast from the Compliance Agent).
    """

    account_id: str
    txn_count: int       # total historical transactions processed
    amount_mean: float   # running mean of transaction amounts (μ)
    amount_std: float    # running std dev of transaction amounts (σ)
    cohort: str          # "retail_low" | "retail_high" | "business" | "unknown"
    last_updated: str    # ISO-8601


class AnomalyFlag(TypedDict):
    """Single anomaly surfaced by the Transaction Monitoring Agent."""
    rule_id: str                   # e.g. "velocity_check", "geo_anomaly"
    description: str
    severity: str                  # "low" | "medium" | "high" | "critical"
    confidence: float              # 0.0 - 1.0


class DetectedPattern(TypedDict):
    """A fraud pattern identified by the Pattern Detection Agent."""
    pattern_id: str                # e.g. "structuring", "mule_network"
    pattern_type: str              # "sequence" | "network" | "behavioral"
    description: str
    involved_entities: list[str]   # account IDs, merchant IDs, etc.
    evidence: dict[str, Any]       # supporting data points
    confidence: float              # 0.0 - 1.0


class RiskBreakdown(TypedDict):
    """Granular risk components computed by the Risk Assessment Agent."""
    anomaly_score: float           # contribution from monitoring flags
    pattern_score: float           # contribution from detected patterns
    historical_score: float        # contribution from account history
    velocity_score: float          # contribution from transaction velocity
    model_score: float             # contribution from ML model prediction


class ActionResult(TypedDict):
    """Outcome produced by the Alert/Block Agent."""
    action: str                    # "allow" | "flag" | "hold" | "block"
    reason: str
    notified_parties: list[str]    # e.g. ["fraud_ops", "customer", "issuer"]
    timestamp: str                 # ISO-8601 when action was executed


class ComplianceEntry(TypedDict):
    """Single compliance/audit log entry."""
    log_id: str
    event_type: str                # e.g. "SAR_GENERATED", "CASE_OPENED"
    summary: str
    regulatory_references: list[str]  # e.g. ["BSA/AML", "31 CFR 1020.320"]
    timestamp: str


# ---------------------------------------------------------------------------
# Top-level pipeline state -- this is what LangGraph passes node-to-node
# ---------------------------------------------------------------------------

class FraudDetectionState(TypedDict):
    """
    Master state flowing through the LangGraph pipeline.

    Pipeline stages and their primary read/write fields:
        1. Monitoring   -> reads: raw_transaction          writes: anomaly_flags, is_anomalous
        2. Pattern Det. -> reads: raw_transaction, flags   writes: detected_patterns
        3. Risk Assess. -> reads: flags, patterns          writes: risk_score, risk_breakdown
        4. Alert/Block  -> reads: risk_score               writes: action_taken
        5. Compliance   -> reads: ALL                      writes: compliance_logs

    Fields using `Annotated[list, operator.add]` are *append-only* channels:
    every node that writes to them *adds* entries rather than replacing the
    list, which is exactly what we want for audit trails.
    """

    # ── Ingest ────────────────────────────────────────────────────────
    raw_transaction: TransactionData
    pipeline_run_id: str           # unique ID for this execution

    # ── Outputs populated during pipeline execution ───────────────────
    is_anomalous: NotRequired[bool]
    risk_score: NotRequired[float]              # 0 - 100 composite score
    risk_breakdown: NotRequired[RiskBreakdown]
    action_taken: NotRequired[ActionResult]

    # ── Monitoring Agent enrichment ───────────────────────────────────
    user_profile: NotRequired[UserProfile]      # fetched from DynamoDB
    is_new_user: NotRequired[bool]              # txn_count < cold-start threshold
    z_score: NotRequired[float | None]          # (amount - μ) / σ; None if new user
    velocity_count: NotRequired[int]            # txns in current Redis window

    # ── Compliance Agent feedback ─────────────────────────────────────
    kafka_event_published: NotRequired[bool]    # True if feedback event was broadcast

    # Append-only fields
    anomaly_flags: NotRequired[Annotated[list[AnomalyFlag], operator.add]]
    detected_patterns: NotRequired[Annotated[list[DetectedPattern], operator.add]]
    compliance_logs: NotRequired[Annotated[list[ComplianceEntry], operator.add]]
    processing_errors: NotRequired[Annotated[list[str], operator.add]]
