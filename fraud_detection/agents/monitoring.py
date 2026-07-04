"""
Transaction Monitoring Agent  (Check 1 — The Smart Gatekeeper)
==============================================================

Pipeline position: **Node 1** (entry point)

This agent is stateful: it fetches the user's memory from two external
stores before deciding how to classify the transaction.

Flow
----
1. Parallel DB fetch
   - Redis  : increment & read the sliding-window velocity counter
   - DynamoDB: fetch the user's statistical baseline (μ, σ, txn_count)

2. Cold-start check
   - txn_count < COLD_START_TXN_THRESHOLD → "new user" path
   - Flags are generated from cohort-based static thresholds
   - Pattern Detection (Check 2) acts as the primary safety net here

3. Dynamic check  (established users only)
   - z-score = (amount - μ) / σ
   - Flag if z > ZSCORE_ALERT_THRESHOLD  (personally anomalous)

4. Velocity check  (both paths)
   - Flag if Redis counter > VELOCITY_MAX_TRANSFERS in the window

5. Route decision
   - Zero flags → Fast Path straight to Check 5 (Compliance)
   - Any flag   → Full Pipeline (Check 2 → 3 → 4 → 5)

Writes to state
---------------
    anomaly_flags    list[AnomalyFlag]   appended via reducer
    is_anomalous     bool
    user_profile     UserProfile | None  fetched from DynamoDB
    is_new_user      bool
    z_score          float | None        None for new users
    velocity_count   int
    processing_errors list[str]          appended via reducer
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any

from fraud_detection.config.settings import (
    COHORT_AMOUNT_THRESHOLDS,
    COLD_START_TXN_THRESHOLD,
    GEO_VELOCITY_MAX_KMH,
    VELOCITY_MAX_TRANSFERS,
    VELOCITY_WINDOW_MINUTES,
    ZSCORE_ALERT_THRESHOLD,
)
from fraud_detection.core.state import AnomalyFlag, FraudDetectionState

logger = logging.getLogger(__name__)

_DB_FETCH_TIMEOUT: float = 3.0  # seconds; keeps P99 latency bounded


# ---------------------------------------------------------------------------
# Parallel DB fetch
# ---------------------------------------------------------------------------

def _fetch_user_data(
    account_id: str,
    window_seconds: int,
) -> tuple[dict[str, Any] | None, int, str | None]:
    """
    Fetch user profile (DynamoDB) and velocity counter (Redis) in parallel.

    Returns
    -------
    (user_profile, velocity_count, velocity_error)
    """
    from fraud_detection.core.dynamo_client import get_user_profile
    from fraud_detection.core.redis_client import increment_velocity

    user_profile: dict[str, Any] | None = None
    velocity_count: int = 0
    velocity_error: str | None = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        dynamo_future = pool.submit(get_user_profile, account_id)
        redis_future = pool.submit(increment_velocity, account_id, window_seconds)

        try:
            user_profile = dynamo_future.result(timeout=_DB_FETCH_TIMEOUT)
        except FutureTimeoutError:
            logger.warning("DynamoDB fetch timed out for account %s", account_id)
        except Exception as exc:
            logger.warning("DynamoDB fetch failed for account %s: %s", account_id, exc)

        try:
            velocity_count, velocity_error = redis_future.result(
                timeout=_DB_FETCH_TIMEOUT
            )
        except FutureTimeoutError:
            velocity_error = "Redis velocity fetch timed out"
            logger.warning(velocity_error)
        except Exception as exc:
            velocity_error = f"Redis velocity fetch failed: {exc}"
            logger.warning(velocity_error)

    return user_profile, velocity_count, velocity_error


# ---------------------------------------------------------------------------
# Anomaly checks
# ---------------------------------------------------------------------------

def _check_zscore(amount: float, mean: float, std: float) -> AnomalyFlag | None:
    """Flag if the transaction amount is personally anomalous (established users)."""
    if std < 1.0:
        # Std too small to compute a meaningful z-score (avoid division noise)
        return None
    z = (amount - mean) / std
    if z > ZSCORE_ALERT_THRESHOLD:
        return AnomalyFlag(
            rule_id="zscore_anomaly",
            description=(
                f"Amount ${amount:,.2f} is {z:.1f}σ above personal mean "
                f"(μ=${mean:,.2f}, σ=${std:,.2f})"
            ),
            severity="high" if z > ZSCORE_ALERT_THRESHOLD * 1.5 else "medium",
            confidence=min((z - ZSCORE_ALERT_THRESHOLD) / ZSCORE_ALERT_THRESHOLD, 1.0),
        )
    return None


def _check_cohort_threshold(amount: float, cohort: str) -> AnomalyFlag | None:
    """Flag if amount exceeds cohort static limit (new-user cold-start fallback)."""
    threshold = COHORT_AMOUNT_THRESHOLDS.get(cohort, COHORT_AMOUNT_THRESHOLDS["unknown"])
    if amount > threshold:
        return AnomalyFlag(
            rule_id="cohort_threshold_exceeded",
            description=(
                f"Amount ${amount:,.2f} exceeds cohort '{cohort}' "
                f"threshold ${threshold:,.2f} (new-user rule)"
            ),
            severity="high" if amount > threshold * 2 else "medium",
            confidence=min(amount / (threshold * 2), 1.0),
        )
    return None


def _check_velocity(velocity_count: int) -> AnomalyFlag | None:
    """Flag accounts that exceed the transfer velocity limit."""
    if velocity_count > VELOCITY_MAX_TRANSFERS:
        return AnomalyFlag(
            rule_id="velocity_exceeded",
            description=(
                f"Account made {velocity_count} transactions in the last "
                f"{VELOCITY_WINDOW_MINUTES} minutes "
                f"(limit: {VELOCITY_MAX_TRANSFERS})"
            ),
            severity="high" if velocity_count > VELOCITY_MAX_TRANSFERS * 2 else "medium",
            confidence=min(velocity_count / (VELOCITY_MAX_TRANSFERS * 2), 1.0),
        )
    return None


def _check_geo_anomaly(txn: dict[str, Any]) -> AnomalyFlag | None:
    """Detect impossible-travel / high-risk location signals."""
    location = txn.get("location", "")
    if location and "foreign" in location.lower():
        return AnomalyFlag(
            rule_id="geo_anomaly",
            description=f"Transaction from potentially risky location: {location}",
            severity="medium",
            confidence=0.6,
        )
    return None


def _check_channel_risk(txn: dict[str, Any]) -> AnomalyFlag | None:
    """Flag transactions from high-risk channels or previously unseen devices."""
    device = str(txn.get("device_fingerprint", ""))
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
    LangGraph node: Transaction Monitoring Agent (Check 1).

    Reads
    -----
    - state["raw_transaction"]

    Writes
    ------
    - anomaly_flags    : list[AnomalyFlag]
    - is_anomalous     : bool
    - user_profile     : UserProfile | None
    - is_new_user      : bool
    - z_score          : float | None
    - velocity_count   : int
    - processing_errors: list[str]
    """
    logger.info("=== Transaction Monitoring Agent (Check 1): START ===")

    txn = state.get("raw_transaction", {})
    txn_dict: dict[str, Any] = dict(txn)
    amount = float(txn_dict.get("amount", 0.0))
    account_id = str(txn_dict.get("account_id", "UNKNOWN"))
    txn_id = txn_dict.get("transaction_id", "UNKNOWN")
    window_seconds = VELOCITY_WINDOW_MINUTES * 60

    logger.info("Processing transaction %s | account=%s | amount=$%,.2f", txn_id, account_id, amount)

    flags: list[AnomalyFlag] = []
    errors: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Parallel DB fetch (Redis + DynamoDB)
    # ------------------------------------------------------------------
    user_profile, velocity_count, velocity_error = _fetch_user_data(
        account_id, window_seconds
    )
    if velocity_error:
        errors.append(velocity_error)

    # ------------------------------------------------------------------
    # Step 2: Cold-start determination
    # ------------------------------------------------------------------
    txn_count = user_profile.get("txn_count", 0) if user_profile else 0
    is_new_user = txn_count < COLD_START_TXN_THRESHOLD

    if is_new_user:
        logger.info(
            "Cold-start path: account %s has %d transactions (threshold=%d)",
            account_id, txn_count, COLD_START_TXN_THRESHOLD,
        )
    else:
        logger.info(
            "Established path: account %s has %d transactions",
            account_id, txn_count,
        )

    # ------------------------------------------------------------------
    # Step 3: Amount anomaly check (path-dependent)
    # ------------------------------------------------------------------
    z_score: float | None = None

    if is_new_user:
        # No reliable μ/σ → fall back to cohort static threshold
        cohort = user_profile.get("cohort", "unknown") if user_profile else "unknown"
        flag = _check_cohort_threshold(amount, cohort)
        if flag:
            flags.append(flag)
    else:
        # Established user: compute personal z-score
        mean = user_profile.get("amount_mean", 0.0)  # type: ignore[union-attr]
        std = user_profile.get("amount_std", 0.0)    # type: ignore[union-attr]
        try:
            z_score = (amount - mean) / std if std > 1.0 else None
        except ZeroDivisionError:
            z_score = None

        flag = _check_zscore(amount, mean, std)
        if flag:
            flags.append(flag)

    # ------------------------------------------------------------------
    # Step 4: Velocity check (both paths)
    # ------------------------------------------------------------------
    velocity_flag = _check_velocity(velocity_count)
    if velocity_flag:
        flags.append(velocity_flag)

    # ------------------------------------------------------------------
    # Step 5: Context checks (geo, channel)
    # ------------------------------------------------------------------
    for check_fn in (_check_geo_anomaly, _check_channel_risk):
        try:
            result = check_fn(txn_dict)
            if result:
                flags.append(result)
        except Exception as exc:
            msg = f"Monitoring check {check_fn.__name__} failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    # ------------------------------------------------------------------
    # Step 6: Route decision
    # ------------------------------------------------------------------
    is_anomalous = len(flags) > 0
    route = "full_pipeline" if is_anomalous else "fast_path"

    logger.info(
        "Transaction %s: %d flags | is_anomalous=%s | route=%s | velocity=%d",
        txn_id, len(flags), is_anomalous, route, velocity_count,
    )
    for flag in flags:
        logger.info(
            "  [%s] %s: %s",
            flag["severity"].upper(), flag["rule_id"], flag["description"],
        )

    logger.info("=== Transaction Monitoring Agent (Check 1): END ===\n")

    return {
        "anomaly_flags": flags,
        "is_anomalous": is_anomalous,
        "user_profile": user_profile,
        "is_new_user": is_new_user,
        "z_score": z_score,
        "velocity_count": velocity_count,
        "processing_errors": errors,
    }
