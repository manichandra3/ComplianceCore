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
STRUCTURING_WINDOW_HOURS: int = 24
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
# Future integration placeholders
# ---------------------------------------------------------------------------

# ── LLM Configuration ────────────────────────────────────────────────
# LLM_PROVIDER = "openai"              # or "anthropic", "azure_openai"
# LLM_MODEL = "gpt-4o"
# LLM_TEMPERATURE = 0.0                # deterministic for fraud decisions
# LLM_MAX_TOKENS = 2048
#
# The LLM will be used by:
#   - Pattern Detection Agent: to interpret novel/ambiguous patterns
#   - Risk Assessment Agent: to generate natural-language risk narratives
#   - Compliance Agent: to draft SAR narrative sections

# ── Neo4j Graph Database ─────────────────────────────────────────────
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

# ── Reinforcement Learning ───────────────────────────────────────────
# RL_MODEL_PATH = "models/rl_fraud_agent_v1.pt"
# RL_ACTION_SPACE = ["allow", "flag", "hold", "block"]
# RL_REWARD_DECAY = 0.99
#
# The RL module will be used by:
#   - Risk Assessment Agent: dynamic score adjustment
#   - Alert/Block Agent: learn optimal action selection from outcomes
