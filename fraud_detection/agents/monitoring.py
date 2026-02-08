"""
Transaction Monitoring Agent
============================

Pipeline position: **Node 1** (entry point)

Responsibility:
    - Ingest raw transaction data from the upstream payment gateway
    - Apply rule-based and statistical filters to eliminate noise
    - Flag anomalies (velocity spikes, geo-impossibilities, amount outliers)
    - Write `anomaly_flags` and `is_anomalous` into state

Future integrations marked with "# ── FUTURE" comments.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fraud_detection.config.settings import (
    AMOUNT_ANOMALY_STDDEV,
    GEO_VELOCITY_MAX_KMH,
    VELOCITY_MAX_COUNT,
    VELOCITY_WINDOW_SECONDS,
)
from fraud_detection.core.state import AnomalyFlag, FraudDetectionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper heuristics  (placeholder logic -- will be replaced by real models)
# ---------------------------------------------------------------------------

def _check_amount_anomaly(amount: float) -> AnomalyFlag | None:
    """Flag transactions that exceed a hard-coded high-value threshold.

    # ── FUTURE: Statistical model ─────────────────────────────────────
    # Replace this static check with a per-account Gaussian model:
    #   mean, std = account_stats_service.get(account_id)
    #   if amount > mean + AMOUNT_ANOMALY_STDDEV * std: flag
    # The model parameters will be stored in the feature store and
    # refreshed nightly via a batch pipeline.
    # ------------------------------------------------------------------
    """
    HIGH_VALUE_PLACEHOLDER = 5_000.0  # naive stand-in
    if amount > HIGH_VALUE_PLACEHOLDER:
        return AnomalyFlag(
            rule_id="high_value_txn",
            description=f"Transaction amount ${amount:,.2f} exceeds "
                        f"high-value threshold ${HIGH_VALUE_PLACEHOLDER:,.2f}",
            severity="high" if amount > 10_000 else "medium",
            confidence=min(amount / 20_000, 1.0),
        )
    return None


def _check_velocity(transaction: dict[str, Any]) -> AnomalyFlag | None:
    """Flag if the account has too many transactions in a short window.

    # ── FUTURE: Sliding-window counter via Redis / Flink ──────────────
    # In production, replace this stub with a real-time counter:
    #   count = redis.incr(f"velocity:{account_id}", ex=VELOCITY_WINDOW_SECONDS)
    #   if count > VELOCITY_MAX_COUNT: flag
    # This enables sub-millisecond velocity checks at scale.
    # ------------------------------------------------------------------
    """
    # Placeholder: always returns None (no velocity data yet)
    _ = VELOCITY_WINDOW_SECONDS, VELOCITY_MAX_COUNT  # suppress unused
    return None


def _check_geo_anomaly(transaction: dict[str, Any]) -> AnomalyFlag | None:
    """Detect impossible-travel scenarios.

    # ── FUTURE: Geo-distance service ─────────────────────────────────
    # Compare current txn location against the last known location:
    #   distance_km = haversine(prev_loc, curr_loc)
    #   time_hours  = (curr_ts - prev_ts).total_seconds() / 3600
    #   if distance_km / time_hours > GEO_VELOCITY_MAX_KMH: flag
    # Previous location will be fetched from Redis or the account
    # profile service.
    # ------------------------------------------------------------------
    """
    _ = GEO_VELOCITY_MAX_KMH  # suppress unused
    location = transaction.get("location", "")
    if location and "foreign" in location.lower():
        return AnomalyFlag(
            rule_id="geo_anomaly",
            description=f"Transaction from potentially risky location: {location}",
            severity="medium",
            confidence=0.6,
        )
    return None


def _check_channel_risk(transaction: dict[str, Any]) -> AnomalyFlag | None:
    """Flag transactions from high-risk channels or new devices."""
    device = str(transaction.get("device_fingerprint", ""))
    if device and device.startswith("NEW_"):
        return AnomalyFlag(
            rule_id="new_device",
            description=f"Transaction from previously unseen device: {device}",
            severity="medium",
            confidence=0.7,
        )
    return None


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def transaction_monitoring_agent(state: FraudDetectionState) -> dict:
    """
    LangGraph node: Transaction Monitoring Agent.

    Reads
    -----
    - state["raw_transaction"]

    Writes
    ------
    - anomaly_flags : list[AnomalyFlag]  (appended via reducer)
    - is_anomalous  : bool

    # ── FUTURE: LLM Integration ──────────────────────────────────────
    # After rule-based checks, send borderline transactions to an LLM
    # for contextual analysis:
    #   from langchain_core.prompts import ChatPromptTemplate
    #   prompt = ChatPromptTemplate.from_messages([...])
    #   chain = prompt | llm | output_parser
    #   llm_assessment = chain.invoke({"transaction": txn})
    # This catches sophisticated fraud that static rules miss.
    # ------------------------------------------------------------------
    """
    logger.info("=== Transaction Monitoring Agent: START ===")

    txn = state.get("raw_transaction", {})
    amount = txn.get("amount", 0.0)
    txn_id = txn.get("transaction_id", "UNKNOWN")

    logger.info(f"Processing transaction {txn_id} | amount=${amount:,.2f}")

    # Run all heuristic checks
    flags: list[AnomalyFlag] = []
    errors: list[str] = []

    for check_fn in [_check_amount_anomaly]:
        try:
            result = check_fn(amount)
            if result:
                flags.append(result)
        except Exception as exc:
            msg = f"Monitoring check {check_fn.__name__} failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    txn_dict: dict[str, Any] = dict(txn)  # cast TypedDict -> dict for helpers
    for check_fn in [_check_velocity, _check_geo_anomaly, _check_channel_risk]:
        try:
            result = check_fn(txn_dict)
            if result:
                flags.append(result)
        except Exception as exc:
            msg = f"Monitoring check {check_fn.__name__} failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    is_anomalous = len(flags) > 0

    logger.info(
        f"Transaction {txn_id}: {len(flags)} anomalies detected | "
        f"is_anomalous={is_anomalous}"
    )
    for flag in flags:
        logger.info(f"  [{flag['severity'].upper()}] {flag['rule_id']}: {flag['description']}")

    logger.info("=== Transaction Monitoring Agent: END ===\n")

    return {
        "anomaly_flags": flags,
        "is_anomalous": is_anomalous,
        "processing_errors": errors,
    }
