"""Tests for the Compliance Logging Agent (Node 5)."""

from __future__ import annotations

from fraud_detection.agents.compliance import compliance_logging_agent
from fraud_detection.config.settings import SAR_THRESHOLD
from fraud_detection.core.state import (
    ActionResult,
    AnomalyFlag,
    DetectedPattern,
    FraudDetectionState,
    RiskBreakdown,
    TransactionData,
)


def _make_state(
    risk_score: float = 10.0,
    action: str = "allow",
    pipeline_run_id: str = "RUN-TEST-001",
    flags: list | None = None,
    patterns: list | None = None,
) -> FraudDetectionState:
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
        anomaly_flags=flags or [],
        detected_patterns=patterns or [],
        risk_score=risk_score,
        risk_breakdown=RiskBreakdown(
            anomaly_score=0.0,
            pattern_score=0.0,
            historical_score=0.0,
            velocity_score=0.0,
            model_score=0.0,
        ),
        action_taken=ActionResult(
            action=action,
            reason="Test reason",
            notified_parties=[],
            timestamp="2026-01-01T00:00:00Z",
        ),
        pipeline_run_id=pipeline_run_id,
    )


# ── Audit trail ──────────────────────────────────────────────────────


class TestAuditTrail:
    def test_always_generates_audit_entry(self):
        result = compliance_logging_agent(_make_state())
        logs = result["compliance_logs"]
        event_types = [log["event_type"] for log in logs]
        assert "PIPELINE_AUDIT" in event_types

    def test_always_generates_action_log(self):
        result = compliance_logging_agent(_make_state())
        logs = result["compliance_logs"]
        event_types = [log["event_type"] for log in logs]
        assert any("TRANSACTION_" in et for et in event_types)

    def test_minimum_two_entries(self):
        """Every transaction should produce at least audit + action entries."""
        result = compliance_logging_agent(_make_state())
        assert len(result["compliance_logs"]) >= 2


# ── Pipeline run ID in logs ──────────────────────────────────────────


class TestPipelineRunId:
    def test_audit_entry_contains_run_id(self):
        result = compliance_logging_agent(_make_state(pipeline_run_id="RUN-ABC123"))
        audit = next(log for log in result["compliance_logs"]
                     if log["event_type"] == "PIPELINE_AUDIT")
        assert "RUN-ABC123" in audit["summary"]

    def test_action_log_contains_run_id(self):
        result = compliance_logging_agent(_make_state(pipeline_run_id="RUN-XYZ789"))
        action_log = next(log for log in result["compliance_logs"]
                         if "TRANSACTION_" in log["event_type"])
        assert "RUN-XYZ789" in action_log["summary"]


# ── SAR generation ───────────────────────────────────────────────────


class TestSARGeneration:
    def test_no_sar_below_threshold(self):
        result = compliance_logging_agent(_make_state(risk_score=SAR_THRESHOLD - 1))
        event_types = [log["event_type"] for log in result["compliance_logs"]]
        assert "SAR_GENERATED" not in event_types

    def test_sar_at_threshold(self):
        result = compliance_logging_agent(_make_state(risk_score=SAR_THRESHOLD))
        event_types = [log["event_type"] for log in result["compliance_logs"]]
        assert "SAR_GENERATED" in event_types

    def test_sar_above_threshold(self):
        result = compliance_logging_agent(_make_state(risk_score=90.0))
        event_types = [log["event_type"] for log in result["compliance_logs"]]
        assert "SAR_GENERATED" in event_types

    def test_sar_produces_three_entries(self):
        """Audit + action + SAR = 3 entries."""
        result = compliance_logging_agent(_make_state(risk_score=85.0))
        assert len(result["compliance_logs"]) == 3

    def test_sar_references_bsa_aml(self):
        result = compliance_logging_agent(_make_state(risk_score=85.0))
        sar = next(log for log in result["compliance_logs"]
                   if log["event_type"] == "SAR_GENERATED")
        assert "BSA/AML" in sar["regulatory_references"]
        assert "31 CFR 1020.320" in sar["regulatory_references"]

    def test_sar_contains_run_id(self):
        result = compliance_logging_agent(
            _make_state(risk_score=85.0, pipeline_run_id="RUN-SAR-TEST")
        )
        sar = next(log for log in result["compliance_logs"]
                   if log["event_type"] == "SAR_GENERATED")
        assert "RUN-SAR-TEST" in sar["summary"]


# ── Action-type mapping ──────────────────────────────────────────────


class TestActionTypeMapping:
    def test_block_maps_to_transaction_blocked(self):
        result = compliance_logging_agent(_make_state(action="block"))
        action_log = next(log for log in result["compliance_logs"]
                         if log["event_type"] == "TRANSACTION_BLOCKED")
        assert action_log is not None

    def test_hold_maps_to_transaction_held(self):
        result = compliance_logging_agent(_make_state(action="hold"))
        action_log = next(log for log in result["compliance_logs"]
                         if log["event_type"] == "TRANSACTION_HELD")
        assert action_log is not None

    def test_flag_maps_to_transaction_flagged(self):
        result = compliance_logging_agent(_make_state(action="flag"))
        action_log = next(log for log in result["compliance_logs"]
                         if log["event_type"] == "TRANSACTION_FLAGGED")
        assert action_log is not None

    def test_allow_maps_to_transaction_allowed(self):
        result = compliance_logging_agent(_make_state(action="allow"))
        action_log = next(log for log in result["compliance_logs"]
                         if log["event_type"] == "TRANSACTION_ALLOWED")
        assert action_log is not None


# ── Log ID format ────────────────────────────────────────────────────


class TestLogIdFormat:
    def test_audit_log_id_starts_with_audit(self):
        result = compliance_logging_agent(_make_state())
        audit = next(log for log in result["compliance_logs"]
                     if log["event_type"] == "PIPELINE_AUDIT")
        assert audit["log_id"].startswith("AUDIT-")

    def test_action_log_id_starts_with_action(self):
        result = compliance_logging_agent(_make_state())
        action_log = next(log for log in result["compliance_logs"]
                         if "TRANSACTION_" in log["event_type"])
        assert action_log["log_id"].startswith("ACTION-")

    def test_sar_log_id_starts_with_sar(self):
        result = compliance_logging_agent(_make_state(risk_score=85.0))
        sar = next(log for log in result["compliance_logs"]
                   if log["event_type"] == "SAR_GENERATED")
        assert sar["log_id"].startswith("SAR-")
