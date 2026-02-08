"""Tests for the Alert/Block Agent (Node 4)."""

from __future__ import annotations

from fraud_detection.agents.alert_block import alert_block_agent
from fraud_detection.config.settings import (
    RISK_THRESHOLD_BLOCK,
    RISK_THRESHOLD_FLAG,
    RISK_THRESHOLD_HOLD,
)
from fraud_detection.core.state import FraudDetectionState, TransactionData


def _make_state(risk_score: float = 0.0) -> FraudDetectionState:
    txn = TransactionData(
        transaction_id="TEST-TXN-001",
        account_id="ACCT-TEST",
        amount=100.0,
        currency="USD",
        merchant_id="MERCH-TEST",
        merchant_category="retail",
        timestamp="2026-01-01T00:00:00Z",
        location="New York, US",
        channel="pos",
        ip_address="1.2.3.4",
        device_fingerprint="DEV-KNOWN-ABC",
        recipient_account="",
        transaction_type="purchase",
        metadata={},
    )
    return FraudDetectionState(
        raw_transaction=txn,
        risk_score=risk_score,
    )


# ── Action selection ─────────────────────────────────────────────────


class TestActionSelection:
    def test_allow_below_flag_threshold(self):
        result = alert_block_agent(_make_state(risk_score=10.0))
        assert result["action_taken"]["action"] == "allow"

    def test_flag_at_flag_threshold(self):
        result = alert_block_agent(_make_state(risk_score=RISK_THRESHOLD_FLAG))
        assert result["action_taken"]["action"] == "flag"

    def test_flag_between_flag_and_hold(self):
        score = (RISK_THRESHOLD_FLAG + RISK_THRESHOLD_HOLD) / 2
        result = alert_block_agent(_make_state(risk_score=score))
        assert result["action_taken"]["action"] == "flag"

    def test_hold_at_hold_threshold(self):
        result = alert_block_agent(_make_state(risk_score=RISK_THRESHOLD_HOLD))
        assert result["action_taken"]["action"] == "hold"

    def test_hold_between_hold_and_block(self):
        score = (RISK_THRESHOLD_HOLD + RISK_THRESHOLD_BLOCK) / 2
        result = alert_block_agent(_make_state(risk_score=score))
        assert result["action_taken"]["action"] == "hold"

    def test_block_at_block_threshold(self):
        result = alert_block_agent(_make_state(risk_score=RISK_THRESHOLD_BLOCK))
        assert result["action_taken"]["action"] == "block"

    def test_block_above_block_threshold(self):
        result = alert_block_agent(_make_state(risk_score=95.0))
        assert result["action_taken"]["action"] == "block"


# ── Output structure ─────────────────────────────────────────────────


class TestOutputStructure:
    def test_action_result_has_required_fields(self):
        result = alert_block_agent(_make_state(risk_score=50.0))
        action = result["action_taken"]
        assert "action" in action
        assert "reason" in action
        assert "notified_parties" in action
        assert "timestamp" in action

    def test_block_notifies_all_parties(self):
        result = alert_block_agent(_make_state(risk_score=90.0))
        parties = result["action_taken"]["notified_parties"]
        assert "fraud_ops" in parties
        assert "customer" in parties
        assert "issuer" in parties
        assert "compliance" in parties

    def test_allow_notifies_no_one(self):
        result = alert_block_agent(_make_state(risk_score=5.0))
        assert result["action_taken"]["notified_parties"] == []

    def test_flag_notifies_fraud_ops(self):
        result = alert_block_agent(_make_state(risk_score=45.0))
        assert "fraud_ops" in result["action_taken"]["notified_parties"]

    def test_reason_mentions_risk_score(self):
        result = alert_block_agent(_make_state(risk_score=85.0))
        assert "85.0" in result["action_taken"]["reason"]
