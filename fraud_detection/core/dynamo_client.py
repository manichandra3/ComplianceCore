"""
DynamoDB Client
===============

Read/write access to the ``fraud_user_profiles`` DynamoDB table, which
stores each account's statistical baseline (μ, σ, txn_count, cohort).

Table schema
------------
  Primary key:  account_id  (String)
  Attributes:
    txn_count     Number  — total transactions processed
    amount_mean   Number  — running mean of approved amounts  (μ)
    amount_m2     Number  — Welford M2 accumulator  (sum of squared deviations)
    amount_std    Number  — √(M2 / txn_count); stored for fast reads
    cohort        String  — "retail_low" | "retail_high" | "business" | "unknown"
    last_updated  String  — ISO-8601

The mean and std are maintained using Welford's online algorithm so we
never need to store the raw transaction history.

Falls back gracefully when DynamoDB is unreachable.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _get_table() -> Any | None:
    """Return the DynamoDB Table resource or None if unavailable."""
    try:
        import boto3
        from fraud_detection.config.settings import (
            DYNAMO_REGION,
            DYNAMO_TABLE_USER_PROFILES,
        )

        dynamodb = boto3.resource("dynamodb", region_name=DYNAMO_REGION)
        return dynamodb.Table(DYNAMO_TABLE_USER_PROFILES)

    except Exception as exc:
        logger.debug("DynamoDB unavailable: %s", exc)
        return None


def get_user_profile(account_id: str) -> dict[str, Any] | None:
    """
    Fetch the statistical profile for *account_id* from DynamoDB.

    Returns None when:
      - The account has no profile (new account)
      - DynamoDB is unreachable

    Returns
    -------
    dict with keys: account_id, txn_count, amount_mean, amount_std,
                    cohort, last_updated
    """
    table = _get_table()
    if table is None:
        return None

    try:
        response = table.get_item(Key={"account_id": account_id})
        item = response.get("Item")
        if not item:
            return None

        txn_count = int(item.get("txn_count", 0))
        m2 = float(item.get("amount_m2", 0.0))
        # Derive std from stored M2 (population std; use txn_count - 1 for sample)
        amount_std = math.sqrt(m2 / txn_count) if txn_count > 1 else 0.0

        return {
            "account_id": str(item.get("account_id", account_id)),
            "txn_count": txn_count,
            "amount_mean": float(item.get("amount_mean", 0.0)),
            "amount_std": amount_std,
            "cohort": str(item.get("cohort", "unknown")),
            "last_updated": str(item.get("last_updated", "")),
        }

    except Exception as exc:
        logger.warning("DynamoDB get_user_profile failed for %s: %s", account_id, exc)
        return None


def update_user_profile(account_id: str, new_amount: float) -> bool:
    """
    Recalculate and persist the user's statistical baseline using a new
    approved transaction amount.

    Uses Welford's online algorithm for numerically stable incremental
    updates without storing the full transaction history:

        n_new   = n + 1
        δ       = x - mean
        mean_new = mean + δ / n_new
        δ2      = x - mean_new
        M2_new  = M2 + δ * δ2
        std_new = √(M2_new / n_new)

    This function is called by the async feedback worker that consumes
    the Kafka/SQS event published by the Compliance Agent.

    Returns True on success, False on failure.
    """
    table = _get_table()
    if table is None:
        return False

    try:
        from decimal import Decimal

        # Read-modify-write (acceptable for async offline worker; not on hot path)
        current = get_user_profile(account_id)

        if current is None:
            # First transaction: create profile with initial values
            table.put_item(Item={
                "account_id": account_id,
                "txn_count": 1,
                "amount_mean": Decimal(str(round(new_amount, 6))),
                "amount_m2": Decimal("0"),
                "amount_std": Decimal("0"),
                "cohort": "unknown",
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
            return True

        n = current["txn_count"]
        mean = current["amount_mean"]
        # Retrieve raw M2 separately since get_user_profile derives std
        response = table.get_item(Key={"account_id": account_id})
        m2 = float(response["Item"].get("amount_m2", 0.0))

        n_new = n + 1
        delta = new_amount - mean
        mean_new = mean + delta / n_new
        delta2 = new_amount - mean_new
        m2_new = m2 + delta * delta2
        std_new = math.sqrt(m2_new / n_new) if n_new > 1 else 0.0

        table.update_item(
            Key={"account_id": account_id},
            UpdateExpression=(
                "SET txn_count = :n, amount_mean = :mean, "
                "amount_m2 = :m2, amount_std = :std, last_updated = :ts"
            ),
            ExpressionAttributeValues={
                ":n":    n_new,
                ":mean": Decimal(str(round(mean_new, 6))),
                ":m2":   Decimal(str(round(m2_new, 6))),
                ":std":  Decimal(str(round(std_new, 6))),
                ":ts":   datetime.now(timezone.utc).isoformat(),
            },
        )
        return True

    except Exception as exc:
        logger.warning("DynamoDB update_user_profile failed for %s: %s", account_id, exc)
        return False
