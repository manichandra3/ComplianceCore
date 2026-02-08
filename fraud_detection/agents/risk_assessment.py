"""
Risk Assessment Agent
=====================

Pipeline position: **Node 3**

Responsibility:
    - Aggregate findings from Monitoring and Pattern Detection agents
    - Compute a composite risk score (0 -- 100)
    - Produce a granular `risk_breakdown` for explainability
    - Write `risk_score` and `risk_breakdown` into state

This agent is the primary candidate for **Reinforcement Learning** integration:
an RL module will learn to adjust the composite score based on downstream
outcomes (confirmed fraud, false positives, customer friction costs).
"""

from __future__ import annotations

import logging
from typing import Any

from fraud_detection.config.settings import (
    WEIGHT_ANOMALY,
    WEIGHT_HISTORICAL,
    WEIGHT_MODEL,
    WEIGHT_PATTERN,
    WEIGHT_VELOCITY,
)
from fraud_detection.core.state import FraudDetectionState, RiskBreakdown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_SEVERITY_SCORE_MAP: dict[str, float] = {
    "critical": 95.0,
    "high": 75.0,
    "medium": 50.0,
    "low": 25.0,
}


def _score_anomalies(flags: list[dict[str, Any]]) -> float:
    """Convert anomaly flags into a 0-100 sub-score.

    Uses severity and confidence to produce a weighted average.
    """
    if not flags:
        return 0.0

    weighted_sum = sum(
        _SEVERITY_SCORE_MAP.get(f.get("severity", "low"), 25.0) * f.get("confidence", 0.5)
        for f in flags
    )
    # Normalise: max possible is 95 * 1.0 per flag, cap at 100
    raw = weighted_sum / max(len(flags), 1)
    return min(raw, 100.0)


def _score_patterns(patterns: list[dict[str, Any]]) -> float:
    """Convert detected patterns into a 0-100 sub-score.

    # ── FUTURE: Neo4j Enrichment ─────────────────────────────────────
    # Pull the pattern's historical hit-rate from Neo4j:
    #   MATCH (p:Pattern {id: $pattern_id})-[:CONFIRMED_FRAUD]->(c:Case)
    #   RETURN count(c) AS confirmed, p.total_hits AS total
    # Use confirmed/total as a Bayesian prior to weight the score.
    # ------------------------------------------------------------------
    """
    if not patterns:
        return 0.0

    # Weight by pattern type severity
    type_weights = {"network": 1.3, "sequence": 1.1, "behavioral": 1.0}
    weighted_sum = sum(
        80.0  # base pattern score
        * p.get("confidence", 0.5)
        * type_weights.get(p.get("pattern_type", "behavioral"), 1.0)
        for p in patterns
    )
    raw = weighted_sum / max(len(patterns), 1)
    return min(raw, 100.0)


def _score_historical(txn: dict[str, Any]) -> float:
    """Score based on account history.

    # ── FUTURE: Feature Store / Neo4j Lookup ─────────────────────────
    # Query the feature store for account-level risk indicators:
    #   - Previous fraud cases on this account
    #   - Account age (newer = riskier)
    #   - Average transaction amount (deviation = riskier)
    #   - Number of disputes in last 90 days
    # ------------------------------------------------------------------
    """
    # Placeholder: use amount as a rough proxy for historical risk.
    # Higher amounts relative to typical spending are riskier.
    amount = float(txn.get("amount", 0))
    if amount > 10_000:
        return 70.0
    elif amount > 5_000:
        return 50.0
    elif amount > 1_000:
        return 35.0
    return 15.0


def _score_velocity(txn: dict[str, Any]) -> float:
    """Score based on transaction velocity.

    # ── FUTURE: Real-Time Velocity Service ───────────────────────────
    # Pull current velocity metrics from Redis:
    #   count_1h = redis.get(f"velocity:1h:{account_id}")
    #   count_24h = redis.get(f"velocity:24h:{account_id}")
    #   amount_1h = redis.get(f"velocity:amount:1h:{account_id}")
    # Map these to a 0-100 score using a sigmoid function.
    # ------------------------------------------------------------------
    """
    # Placeholder: use channel as a proxy -- online channels have higher
    # velocity risk than POS/ATM.
    channel = str(txn.get("channel", ""))
    if channel == "online":
        return 45.0
    elif channel == "atm":
        return 30.0
    return 15.0


def _score_ml_model(txn: dict[str, Any], flags: list[dict[str, Any]], patterns: list[dict[str, Any]]) -> float:
    """Score from the ML fraud-detection model.

    # ── FUTURE: ML Model Inference ───────────────────────────────────
    # Call the model serving endpoint:
    #   features = feature_engineering.extract(txn, flags, patterns)
    #   prediction = model_client.predict(features)
    #   return prediction.fraud_probability * 100
    #
    # The model will be a gradient-boosted tree (XGBoost/LightGBM)
    # trained on historical transaction data with fraud labels.
    # ------------------------------------------------------------------

    # ── FUTURE: Reinforcement Learning Adjustment ────────────────────
    # After the base ML score, apply an RL adjustment:
    #   rl_state = RLState(txn=txn, flags=flags, patterns=patterns,
    #                       base_score=ml_score)
    #   rl_action = rl_agent.act(rl_state)
    #   adjusted_score = ml_score + rl_action.score_delta
    #
    # The RL agent will be trained using:
    #   - Reward = +1 for correctly caught fraud
    #   - Reward = -0.5 for false positive (customer friction)
    #   - Reward = -2 for missed fraud (loss)
    #   - Discount factor: RL_REWARD_DECAY from settings
    # ------------------------------------------------------------------
    """
    # Placeholder: combine signal count as a naive proxy for model output.
    # More flags + patterns = higher model confidence in fraud.
    signal_count = len(flags) + len(patterns)
    if signal_count >= 4:
        return 85.0
    elif signal_count >= 3:
        return 65.0
    elif signal_count >= 2:
        return 50.0
    elif signal_count >= 1:
        return 35.0
    return 10.0


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def risk_assessment_agent(state: FraudDetectionState) -> dict:
    """
    LangGraph node: Risk Assessment Agent.

    Reads
    -----
    - state["raw_transaction"]
    - state["anomaly_flags"]
    - state["detected_patterns"]

    Writes
    ------
    - risk_score     : float  (0-100)
    - risk_breakdown : RiskBreakdown

    # ── FUTURE: LLM Risk Narrative ───────────────────────────────────
    # After computing the numeric score, generate a human-readable risk
    # narrative using an LLM:
    #   narrative = llm.invoke(
    #       "Summarise the fraud risk for this transaction given these "
    #       "signals: {flags}, {patterns}, {scores}"
    #   )
    # This narrative will be included in the compliance report and
    # surfaced to fraud analysts in the case management UI.
    # ------------------------------------------------------------------
    """
    logger.info("=== Risk Assessment Agent: START ===")

    txn = state.get("raw_transaction", {})
    flags = state.get("anomaly_flags", [])
    patterns = state.get("detected_patterns", [])
    txn_id = txn.get("transaction_id", "UNKNOWN")

    # Cast TypedDicts to plain dicts for helper compatibility
    txn_dict: dict[str, Any] = dict(txn)
    flags_list: list[dict[str, Any]] = [dict(f) for f in flags]
    patterns_list: list[dict[str, Any]] = [dict(p) for p in patterns]

    # Compute component scores
    anomaly_score = _score_anomalies(flags_list)
    pattern_score = _score_patterns(patterns_list)
    historical_score = _score_historical(txn_dict)
    velocity_score = _score_velocity(txn_dict)
    model_score = _score_ml_model(txn_dict, flags_list, patterns_list)

    # Weighted composite
    risk_score = (
        WEIGHT_ANOMALY * anomaly_score
        + WEIGHT_PATTERN * pattern_score
        + WEIGHT_HISTORICAL * historical_score
        + WEIGHT_VELOCITY * velocity_score
        + WEIGHT_MODEL * model_score
    )
    risk_score = round(min(max(risk_score, 0.0), 100.0), 2)

    breakdown = RiskBreakdown(
        anomaly_score=round(anomaly_score, 2),
        pattern_score=round(pattern_score, 2),
        historical_score=round(historical_score, 2),
        velocity_score=round(velocity_score, 2),
        model_score=round(model_score, 2),
    )

    logger.info(f"Transaction {txn_id}: RISK SCORE = {risk_score}/100")
    logger.info(f"  Breakdown: anomaly={anomaly_score:.1f} pattern={pattern_score:.1f} "
                f"historical={historical_score:.1f} velocity={velocity_score:.1f} "
                f"model={model_score:.1f}")
    logger.info(f"  Weights:   anomaly={WEIGHT_ANOMALY} pattern={WEIGHT_PATTERN} "
                f"historical={WEIGHT_HISTORICAL} velocity={WEIGHT_VELOCITY} "
                f"model={WEIGHT_MODEL}")

    logger.info("=== Risk Assessment Agent: END ===\n")

    return {
        "risk_score": risk_score,
        "risk_breakdown": breakdown,
    }
