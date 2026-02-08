"""Tests for the Risk Assessment Agent (Node 3)."""

from __future__ import annotations

from fraud_detection.agents.risk_assessment import risk_assessment_agent
from fraud_detection.core.state import (
    AnomalyFlag,
    DetectedPattern,
    FraudDetectionState,
    TransactionData,
)


def _make_state(
    amount: float = 100.0,
    channel: str = "pos",
    flags: list | None = None,
    patterns: list | None = None,
) -> FraudDetectionState:
    txn = TransactionData(
        transaction_id="TEST-TXN-001",
        account_id="ACCT-TEST",
        amount=amount,
        currency="USD",
        merchant_id="MERCH-TEST",
        merchant_category="retail",
        timestamp="2026-01-01T00:00:00Z",
        location="New York, US",
        channel=channel,
        ip_address="1.2.3.4",
        device_fingerprint="DEV-KNOWN-ABC",
        recipient_account="",
        transaction_type="purchase",
        metadata={},
    )
    return FraudDetectionState(
        raw_transaction=txn,
        anomaly_flags=flags or [],
        detected_patterns=patterns or [],
    )


def _make_flag(rule_id: str = "test_rule", severity: str = "medium",
               confidence: float = 0.7) -> AnomalyFlag:
    return AnomalyFlag(
        rule_id=rule_id,
        description="Test anomaly",
        severity=severity,
        confidence=confidence,
    )


def _make_pattern(pattern_id: str = "test_pattern", pattern_type: str = "behavioral",
                  confidence: float = 0.6) -> DetectedPattern:
    return DetectedPattern(
        pattern_id=pattern_id,
        pattern_type=pattern_type,
        description="Test pattern",
        involved_entities=["ACCT-TEST"],
        evidence={},
        confidence=confidence,
    )


# ── Output structure ─────────────────────────────────────────────────


class TestOutputStructure:
    def test_returns_risk_score(self):
        result = risk_assessment_agent(_make_state())
        assert "risk_score" in result
        assert isinstance(result["risk_score"], float)

    def test_returns_risk_breakdown(self):
        result = risk_assessment_agent(_make_state())
        breakdown = result["risk_breakdown"]
        assert "anomaly_score" in breakdown
        assert "pattern_score" in breakdown
        assert "historical_score" in breakdown
        assert "velocity_score" in breakdown
        assert "model_score" in breakdown

    def test_risk_score_bounded_0_100(self):
        result = risk_assessment_agent(_make_state())
        assert 0.0 <= result["risk_score"] <= 100.0


# ── Clean transaction scoring ────────────────────────────────────────


class TestCleanTransactionScoring:
    def test_low_risk_for_small_clean_purchase(self):
        result = risk_assessment_agent(_make_state(amount=42.50))
        assert result["risk_score"] < 40.0  # should be well below FLAG threshold

    def test_anomaly_score_zero_when_no_flags(self):
        result = risk_assessment_agent(_make_state())
        assert result["risk_breakdown"]["anomaly_score"] == 0.0

    def test_pattern_score_zero_when_no_patterns(self):
        result = risk_assessment_agent(_make_state())
        assert result["risk_breakdown"]["pattern_score"] == 0.0


# ── Risk escalation with signals ─────────────────────────────────────


class TestRiskEscalation:
    def test_risk_increases_with_anomaly_flags(self):
        clean = risk_assessment_agent(_make_state())
        flagged = risk_assessment_agent(_make_state(
            flags=[_make_flag(severity="high", confidence=0.9)]
        ))
        assert flagged["risk_score"] > clean["risk_score"]

    def test_risk_increases_with_patterns(self):
        clean = risk_assessment_agent(_make_state())
        patterned = risk_assessment_agent(_make_state(
            patterns=[_make_pattern(pattern_type="network", confidence=0.8)]
        ))
        assert patterned["risk_score"] > clean["risk_score"]

    def test_risk_increases_with_both(self):
        flags_only = risk_assessment_agent(_make_state(
            flags=[_make_flag()]
        ))
        both = risk_assessment_agent(_make_state(
            flags=[_make_flag()],
            patterns=[_make_pattern()],
        ))
        assert both["risk_score"] > flags_only["risk_score"]

    def test_high_amount_increases_historical_score(self):
        low = risk_assessment_agent(_make_state(amount=100.0))
        high = risk_assessment_agent(_make_state(amount=15_000.0))
        assert high["risk_breakdown"]["historical_score"] > low["risk_breakdown"]["historical_score"]

    def test_online_channel_higher_velocity_than_pos(self):
        pos = risk_assessment_agent(_make_state(channel="pos"))
        online = risk_assessment_agent(_make_state(channel="online"))
        assert online["risk_breakdown"]["velocity_score"] > pos["risk_breakdown"]["velocity_score"]


# ── Weight composition ───────────────────────────────────────────────


class TestWeightComposition:
    def test_composite_is_weighted_sum(self):
        """Verify the composite score equals the weighted sum of components."""
        from fraud_detection.config.settings import (
            WEIGHT_ANOMALY, WEIGHT_PATTERN, WEIGHT_HISTORICAL,
            WEIGHT_VELOCITY, WEIGHT_MODEL,
        )

        result = risk_assessment_agent(_make_state(
            amount=9_000.0,
            channel="online",
            flags=[_make_flag(severity="high", confidence=0.8)],
            patterns=[_make_pattern(pattern_type="sequence", confidence=0.7)],
        ))
        bd = result["risk_breakdown"]
        expected = round(min(max(
            WEIGHT_ANOMALY * bd["anomaly_score"]
            + WEIGHT_PATTERN * bd["pattern_score"]
            + WEIGHT_HISTORICAL * bd["historical_score"]
            + WEIGHT_VELOCITY * bd["velocity_score"]
            + WEIGHT_MODEL * bd["model_score"],
            0.0), 100.0), 2)

        assert result["risk_score"] == expected
