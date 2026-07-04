"""
Kafka / SQS Event Publisher
============================

Publishes fraud-pipeline events to a message queue so that async workers
can consume them and update downstream state (e.g., recalculate μ / σ in
DynamoDB after an allowed transaction — the feedback loop that makes the
system smarter over time).

Transport priority
------------------
1. Kafka  (confluent-kafka or kafka-python; configured via KAFKA_BROKER)
2. SQS    (boto3; configured via SQS_QUEUE_URL)
3. Structured log  (guaranteed fallback so no events are silently lost)

Callers never need to know which transport was used.  The return value
signals success/failure so the Compliance Agent can set
``kafka_event_published`` in state for auditability.

Event payload schema
--------------------
{
    "schema_version": "1.0",
    "event_type":     "TRANSACTION_ALLOWED",
    "pipeline_run_id": "...",
    "account_id":     "...",
    "transaction_id": "...",
    "amount":         1234.56,
    "timestamp":      "2026-07-04T00:00:00Z"
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport implementations
# ---------------------------------------------------------------------------

def _try_kafka(payload: dict[str, Any]) -> bool:
    """Attempt to publish *payload* to Kafka. Returns True on success."""
    try:
        from fraud_detection.config.settings import (
            KAFKA_BROKER,
            KAFKA_PRODUCER_TIMEOUT,
            KAFKA_TOPIC_ALLOWED_TXN,
        )

        # Try confluent-kafka first, fall back to kafka-python
        try:
            from confluent_kafka import Producer  # type: ignore[import]

            producer = Producer({"bootstrap.servers": KAFKA_BROKER})
            producer.produce(
                KAFKA_TOPIC_ALLOWED_TXN,
                key=payload.get("account_id", "").encode(),
                value=json.dumps(payload).encode(),
            )
            producer.flush(timeout=KAFKA_PRODUCER_TIMEOUT)
            logger.debug(
                "Kafka event published [confluent]: topic=%s txn=%s",
                KAFKA_TOPIC_ALLOWED_TXN,
                payload.get("transaction_id"),
            )
            return True

        except ImportError:
            pass

        from kafka import KafkaProducer  # type: ignore[import]

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=lambda k: k.encode() if k else None,
            request_timeout_ms=int(KAFKA_PRODUCER_TIMEOUT * 1000),
        )
        future = producer.send(
            KAFKA_TOPIC_ALLOWED_TXN,
            key=payload.get("account_id", ""),
            value=payload,
        )
        future.get(timeout=KAFKA_PRODUCER_TIMEOUT)
        producer.close(timeout=KAFKA_PRODUCER_TIMEOUT)
        logger.debug(
            "Kafka event published [kafka-python]: topic=%s txn=%s",
            KAFKA_TOPIC_ALLOWED_TXN,
            payload.get("transaction_id"),
        )
        return True

    except Exception as exc:
        logger.debug("Kafka publish failed: %s", exc)
        return False


def _try_sqs(payload: dict[str, Any]) -> bool:
    """Attempt to publish *payload* to SQS. Returns True on success."""
    try:
        from fraud_detection.config.settings import DYNAMO_REGION, SQS_QUEUE_URL

        if not SQS_QUEUE_URL:
            return False

        import boto3

        sqs = boto3.client("sqs", region_name=DYNAMO_REGION)
        sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(payload),
            MessageAttributes={
                "event_type": {
                    "DataType": "String",
                    "StringValue": payload.get("event_type", "UNKNOWN"),
                }
            },
        )
        logger.debug(
            "SQS event published: queue=%s txn=%s",
            SQS_QUEUE_URL,
            payload.get("transaction_id"),
        )
        return True

    except Exception as exc:
        logger.debug("SQS publish failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def publish_allowed_transaction(
    pipeline_run_id: str,
    account_id: str,
    transaction_id: str,
    amount: float,
) -> bool:
    """
    Broadcast a ``TRANSACTION_ALLOWED`` event so the async feedback worker
    can update the user's statistical baseline in DynamoDB.

    Transport order: Kafka -> SQS -> structured log.

    Returns True if the event was delivered to a queue; False if only the
    log fallback was used (so the caller can flag it in pipeline state).
    """
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "event_type": "TRANSACTION_ALLOWED",
        "pipeline_run_id": pipeline_run_id,
        "account_id": account_id,
        "transaction_id": transaction_id,
        "amount": amount,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if _try_kafka(payload):
        return True

    if _try_sqs(payload):
        return True

    # Structured-log fallback — event is not lost; can be replayed from logs
    logger.warning(
        "No message queue available — event logged for manual replay: %s",
        json.dumps(payload),
    )
    return False
