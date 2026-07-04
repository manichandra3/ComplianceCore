"""
Redis Client
============

Provides real-time velocity counter operations backed by Redis.
Used exclusively by the Transaction Monitoring Agent.

Every transaction increments the sliding-window counter for its account.
The counter key expires automatically after the window duration, giving
a true sliding-window without a background cleanup job.

Key schema
----------
    velocity:{account_id}:w{window_seconds}   -- INCR with TTL = window_seconds

Falls back gracefully when Redis is unreachable: returns (0, error_msg)
so the monitoring agent can degrade safely.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton — lazily initialised and reused across invocations
_client: Any | None = None


def _get_client() -> Any | None:
    """Lazily acquire a Redis connection; return None if unavailable."""
    global _client
    if _client is not None:
        return _client

    try:
        import redis
        from fraud_detection.config.settings import (
            REDIS_DB,
            REDIS_HOST,
            REDIS_PASSWORD,
            REDIS_PORT,
            REDIS_SOCKET_TIMEOUT,
        )

        conn = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD or None,
            socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            decode_responses=True,
        )
        conn.ping()
        _client = conn
        logger.debug("Redis connection established at %s:%s", REDIS_HOST, REDIS_PORT)
        return _client

    except Exception as exc:
        logger.debug("Redis unavailable: %s", exc)
        return None


def increment_velocity(account_id: str, window_seconds: int) -> tuple[int, str | None]:
    """
    Increment and return the velocity counter for *account_id*.

    The first INCR in a window also sets the TTL, so the window slides
    from the first transaction rather than a fixed clock boundary.

    Returns
    -------
    (count, error)
        ``count``  — current counter value after the increment (1 = first txn).
        ``error``  — non-None string when Redis is unreachable; count is 0.
    """
    conn = _get_client()
    if conn is None:
        return 0, "Redis unavailable — velocity check skipped"

    key = f"velocity:{account_id}:w{window_seconds}"
    try:
        pipe = conn.pipeline(transaction=False)
        pipe.incr(key)
        # nx=True: only set the TTL the first time (don't reset it on each txn)
        pipe.expire(key, window_seconds, nx=True)
        results = pipe.execute()
        count: int = results[0]
        return count, None

    except Exception as exc:
        msg = f"Redis velocity increment failed for {account_id}: {exc}"
        logger.warning(msg)
        # Reset singleton so next call retries the connection
        global _client
        _client = None
        return 0, msg


def get_velocity(account_id: str, window_seconds: int) -> tuple[int, str | None]:
    """
    Read (without incrementing) the current velocity count for *account_id*.

    Useful for read-only checks (e.g., risk assessment) that should not
    themselves affect the counter.

    Returns
    -------
    (count, error)
    """
    conn = _get_client()
    if conn is None:
        return 0, "Redis unavailable"

    key = f"velocity:{account_id}:w{window_seconds}"
    try:
        val = conn.get(key)
        return (int(val) if val else 0), None
    except Exception as exc:
        msg = f"Redis get velocity failed for {account_id}: {exc}"
        logger.warning(msg)
        return 0, msg
