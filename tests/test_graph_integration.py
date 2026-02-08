"""Integration tests for the LangGraph pipeline with conditional routing."""

from __future__ import annotations

import uuid

from fraud_detection.core.graph import compile_fraud_detection_graph
from fraud_detection.core.state import FraudDetectionState, TransactionData


def _make_initial_state(
    amount: float = 42.50,
    location: str = "New York, US",
    device: str = "DEV-KNOWN-ABC",
    channel: str = "pos",
    transaction_type: str = "purchase",
    recipient_account: str = "",
) -> FraudDetectionState:
    txn = TransactionData(
        transaction_id=f"TEST-{uuid.uuid4().hex[:8].upper()}",
        account_id="ACCT-TEST",
        amount=amount,
        currency="USD",
        merchant_id="MERCH-TEST",
        merchant_category="retail",
        timestamp="2026-01-01T00:00:00Z",
        location=location,
        channel=channel,
        ip_address="1.2.3.4",
        device_fingerprint=device,
        recipient_account=recipient_account,
        transaction_type=transaction_type,
        metadata={},
    )
    return FraudDetectionState(
        raw_transaction=txn,
        pipeline_run_id=f"RUN-TEST-{uuid.uuid4().hex[:8].upper()}",
    )


# ── Conditional routing ──────────────────────────────────────────────


class TestConditionalRouting:
    """Verify that clean transactions take the fast-path and
    suspicious ones go through the full pipeline."""

    def setup_method(self):
        self.pipeline = compile_fraud_detection_graph()

    def test_clean_transaction_fast_path(self):
        """Small local purchase should bypass Pattern/Risk/Alert."""
        state = _make_initial_state(amount=42.50)
        result = self.pipeline.invoke(state)

        assert result["risk_score"] == 0.0
        assert result["action_taken"]["action"] == "allow"
        assert "fast-path" in result["action_taken"]["reason"].lower()
        assert result["detected_patterns"] == []
        assert result["anomaly_flags"] == []

    def test_suspicious_transaction_full_pipeline(self):
        """High-value foreign transfer should go through full analysis."""
        state = _make_initial_state(
            amount=15_750.00,
            location="Foreign - Lagos, NG",
            device="NEW_DEV-SUSPICIOUS",
            channel="online",
            transaction_type="transfer",
            recipient_account="MULE-ACCT-999",
        )
        result = self.pipeline.invoke(state)

        assert result["risk_score"] > 0.0
        assert result["is_anomalous"] is True
        assert len(result["anomaly_flags"]) > 0
        # Full pipeline should have patterns detected
        assert len(result["detected_patterns"]) > 0
        # Action should be based on computed risk, not fast-path defaults
        assert "fast-path" not in result["action_taken"]["reason"].lower()


# ── Pipeline completeness ────────────────────────────────────────────


class TestPipelineCompleteness:
    """Verify that both paths produce complete state."""

    def setup_method(self):
        self.pipeline = compile_fraud_detection_graph()

    def _assert_complete_state(self, result: dict):
        """Check that all required state fields are populated."""
        assert "raw_transaction" in result
        assert "anomaly_flags" in result
        assert "is_anomalous" in result
        assert "risk_score" in result
        assert "risk_breakdown" in result
        assert "action_taken" in result
        assert "compliance_logs" in result
        assert len(result["compliance_logs"]) >= 2  # at least audit + action

    def test_clean_path_produces_complete_state(self):
        state = _make_initial_state(amount=18.75)
        result = self.pipeline.invoke(state)
        self._assert_complete_state(result)

    def test_full_path_produces_complete_state(self):
        state = _make_initial_state(
            amount=9_400.00,
            channel="atm",
        )
        result = self.pipeline.invoke(state)
        self._assert_complete_state(result)

    def test_compliance_logs_contain_run_id(self):
        state = _make_initial_state()
        result = self.pipeline.invoke(state)
        run_id = result.get("pipeline_run_id", "")
        audit = next(log for log in result["compliance_logs"]
                     if log["event_type"] == "PIPELINE_AUDIT")
        assert run_id in audit["summary"]


# ── Risk breakdown structure ─────────────────────────────────────────


class TestRiskBreakdown:
    def setup_method(self):
        self.pipeline = compile_fraud_detection_graph()

    def test_fast_path_breakdown_all_zeros(self):
        state = _make_initial_state(amount=5.00)
        result = self.pipeline.invoke(state)
        bd = result["risk_breakdown"]
        assert bd["anomaly_score"] == 0.0
        assert bd["pattern_score"] == 0.0
        assert bd["historical_score"] == 0.0
        assert bd["velocity_score"] == 0.0
        assert bd["model_score"] == 0.0

    def test_full_path_breakdown_has_nonzero_components(self):
        state = _make_initial_state(
            amount=9_950.00,
            location="Foreign - Cayman Islands",
            device="NEW_DEV-XYZ",
            channel="online",
            transaction_type="transfer",
            recipient_account="DEST-ACCT-001",
        )
        result = self.pipeline.invoke(state)
        bd = result["risk_breakdown"]
        # At least anomaly and historical should be nonzero
        assert bd["anomaly_score"] > 0.0
        assert bd["historical_score"] > 0.0
