"""
Configuration & thresholds for the Fraud Detection pipeline.
=============================================================

All magic numbers are centralised here so they can be tuned without touching
agent logic.  In production these would come from a config service / env vars.
"""

import os

# ---------------------------------------------------------------------------
# Risk score thresholds  (used by Alert/Block Agent)
# ---------------------------------------------------------------------------
RISK_THRESHOLD_BLOCK: float = 80.0     # >= this -> block the transaction
RISK_THRESHOLD_HOLD: float = 60.0      # >= this -> hold for manual review
RISK_THRESHOLD_FLAG: float = 40.0      # >= this -> flag but allow
# anything below FLAG threshold -> allow silently

# ---------------------------------------------------------------------------
# Monitoring Agent tunables
# ---------------------------------------------------------------------------
VELOCITY_WINDOW_SECONDS: int = 3600    # 1-hour sliding window
VELOCITY_MAX_COUNT: int = 10           # max txns in window before flag
AMOUNT_ANOMALY_STDDEV: float = 3.0     # flag if amount > mean + N*stddev
GEO_VELOCITY_MAX_KMH: float = 900.0   # impossible travel speed (km/h)

# ---------------------------------------------------------------------------
# Pattern Detection thresholds
# ---------------------------------------------------------------------------
STRUCTURING_THRESHOLD: float = 10_000.0   # BSA/AML structuring boundary
STRUCTURING_MIN_AMOUNT: float = 9000.0
STRUCTURING_HEURISTIC_RATIO: float = 0.8  # e.g., 0.8 means 80% of threshold
STRUCTURING_MIN_TXNS: int = 2
STRUCTURING_WINDOW_HOURS: int = 24
STRUCTURING_WINDOW_DAYS: int = 7
MULE_FAN_IN_MIN_SENDERS: int = 5
MULE_FAN_IN_WINDOW_DAYS: int = 1
MULE_HEURISTIC_MIN_AMOUNT: float = 3000.0
MIN_PATTERN_CONFIDENCE: float = 0.5       # ignore patterns below this

# ---------------------------------------------------------------------------
# Risk weight allocation  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_ANOMALY: float = 0.25
WEIGHT_PATTERN: float = 0.30
WEIGHT_HISTORICAL: float = 0.15
WEIGHT_VELOCITY: float = 0.15
WEIGHT_MODEL: float = 0.15

# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------
SAR_THRESHOLD: float = 70.0    # auto-generate SAR draft above this score

# ---------------------------------------------------------------------------
# Neo4j Graph Database
# ---------------------------------------------------------------------------
NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE: str = os.environ.get("NEO4J_DATABASE", "fraud_graph")
NEO4J_MAX_CONNECTION_POOL_SIZE: int = int(
    os.environ.get("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")
)
NEO4J_CONNECTION_TIMEOUT: float = float(
    os.environ.get("NEO4J_CONNECTION_TIMEOUT", "5.0")
)
#
# The graph DB is used by:
#   - Pattern Detection Agent: entity-relationship traversal to find
#     mule networks, shared device fingerprints, account clusters
#   - Risk Assessment Agent: pull historical risk sub-graph per entity

# ---------------------------------------------------------------------------
# Cold-start / new-user detection
# ---------------------------------------------------------------------------
COLD_START_TXN_THRESHOLD: int = 15   # accounts with fewer txns are "new"

# ---------------------------------------------------------------------------
# Velocity  (Redis-based sliding window)
# ---------------------------------------------------------------------------
VELOCITY_WINDOW_MINUTES: int = 10    # sliding window length in minutes
VELOCITY_MAX_TRANSFERS: int = 5      # flag after this many txns in the window

# ---------------------------------------------------------------------------
# Statistical anomaly (z-score)
# ---------------------------------------------------------------------------
ZSCORE_ALERT_THRESHOLD: float = 3.0  # flag when (amount - μ) / σ > this

# ---------------------------------------------------------------------------
# Cohort-based static thresholds  (fallback for new users with no μ/σ)
# ---------------------------------------------------------------------------
COHORT_AMOUNT_THRESHOLDS: dict[str, float] = {
    "retail_low":  2_000.0,
    "retail_high": 15_000.0,
    "business":    50_000.0,
    "unknown":     5_000.0,   # conservative default
}

# ---------------------------------------------------------------------------
# Redis  (velocity counter store)
# ---------------------------------------------------------------------------
REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.environ.get("REDIS_DB", "0"))
REDIS_PASSWORD: str = os.environ.get("REDIS_PASSWORD", "")
REDIS_SOCKET_TIMEOUT: float = 2.0

# ---------------------------------------------------------------------------
# DynamoDB  (user statistical baseline store)
# ---------------------------------------------------------------------------
DYNAMO_TABLE_USER_PROFILES: str = os.environ.get(
    "DYNAMO_TABLE_USER_PROFILES", "fraud_user_profiles"
)
DYNAMO_REGION: str = os.environ.get("AWS_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Kafka / SQS  (feedback loop event bus)
# ---------------------------------------------------------------------------
KAFKA_BROKER: str = os.environ.get("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC_ALLOWED_TXN: str = os.environ.get(
    "KAFKA_TOPIC_ALLOWED_TXN", "fraud.txn.allowed"
)
KAFKA_PRODUCER_TIMEOUT: float = 5.0
SQS_QUEUE_URL: str = os.environ.get("SQS_QUEUE_URL", "")

# ---------------------------------------------------------------------------
# Adaptive risk weights — NEW USER  (txn_count < COLD_START_TXN_THRESHOLD)
# No reliable personal history → lean on graph patterns and velocity signals
# ---------------------------------------------------------------------------
WEIGHT_ANOMALY_NEW_USER: float    = 0.15
WEIGHT_PATTERN_NEW_USER: float    = 0.40
WEIGHT_HISTORICAL_NEW_USER: float = 0.05
WEIGHT_VELOCITY_NEW_USER: float   = 0.25
WEIGHT_MODEL_NEW_USER: float      = 0.15

# ---------------------------------------------------------------------------
# Adaptive risk weights — ESTABLISHED USER  (txn_count >= threshold)
# Rich personal history → z-score anomaly is the dominant signal
# ---------------------------------------------------------------------------
WEIGHT_ANOMALY_ESTABLISHED: float    = 0.35
WEIGHT_PATTERN_ESTABLISHED: float    = 0.25
WEIGHT_HISTORICAL_ESTABLISHED: float = 0.15
WEIGHT_VELOCITY_ESTABLISHED: float   = 0.10
WEIGHT_MODEL_ESTABLISHED: float      = 0.15
