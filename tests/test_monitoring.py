"""Tests for the Transaction Monitoring Agent (Node 1)."""

from __future__ import annotations

from fraud_detection.agents.monitoring import transaction_monitoring_agent
from fraud_detection.core.state import FraudDetectionState, TransactionData


def _make_state(
    amount: float = 100.0,
    location: str = "New York, US",
    device: str = "DEV-KNOWN-ABC",
    channel: str = "pos",
    **txn_overrides,
) -> FraudDetectionState:
    """Build a minimal FraudDetectionState for testing."""
    txn = TransactionData(
        transaction_id="TEST-TXN-001",
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
        recipient_account="",
        transaction_type="purchase",
        metadata={},
        **txn_overrides,
    )
    return FraudDetectionState(raw_transaction=txn)


# ── Clean transaction: no flags ──────────────────────────────────────


class TestCleanTransaction:
    def test_no_flags_for_small_local_purchase(self):
        state = _make_state(amount=42.50)
        result = transaction_monitoring_agent(state)

        assert result["is_anomalous"] is False
        assert result["anomaly_flags"] == []
        assert result["processing_errors"] == []

    def test_returns_expected_keys(self):
        result = transaction_monitoring_agent(_make_state())
        assert "anomaly_flags" in result
        assert "is_anomalous" in result
        assert "processing_errors" in result


# ── High-value threshold ─────────────────────────────────────────────


class TestAmountAnomaly:
    def test_flags_high_value_transaction(self):
        state = _make_state(amount=6_000.00)
        result = transaction_monitoring_agent(state)

        assert result["is_anomalous"] is True
        flags = result["anomaly_flags"]
        rule_ids = [f["rule_id"] for f in flags]
        assert "high_value_txn" in rule_ids

    def test_severity_high_above_10k(self):
        state = _make_state(amount=15_000.00)
        result = transaction_monitoring_agent(state)

        hv_flag = next(f for f in result["anomaly_flags"] if f["rule_id"] == "high_value_txn")
        assert hv_flag["severity"] == "high"

    def test_severity_medium_between_5k_and_10k(self):
        state = _make_state(amount=7_000.00)
        result = transaction_monitoring_agent(state)

        hv_flag = next(f for f in result["anomaly_flags"] if f["rule_id"] == "high_value_txn")
        assert hv_flag["severity"] == "medium"

    def test_no_flag_below_threshold(self):
        state = _make_state(amount=4_999.00)
        result = transaction_monitoring_agent(state)

        rule_ids = [f["rule_id"] for f in result["anomaly_flags"]]
        assert "high_value_txn" not in rule_ids

    def test_confidence_caps_at_1(self):
        state = _make_state(amount=50_000.00)
        result = transaction_monitoring_agent(state)

        hv_flag = next(f for f in result["anomaly_flags"] if f["rule_id"] == "high_value_txn")
        assert hv_flag["confidence"] <= 1.0


# ── Geo anomaly ──────────────────────────────────────────────────────


class TestGeoAnomaly:
    def test_flags_foreign_location(self):
        state = _make_state(location="Foreign - Lagos, NG")
        result = transaction_monitoring_agent(state)

        rule_ids = [f["rule_id"] for f in result["anomaly_flags"]]
        assert "geo_anomaly" in rule_ids

    def test_no_flag_domestic_location(self):
        state = _make_state(location="New York, US")
        result = transaction_monitoring_agent(state)

        rule_ids = [f["rule_id"] for f in result["anomaly_flags"]]
        assert "geo_anomaly" not in rule_ids


# ── New device detection ─────────────────────────────────────────────


class TestNewDevice:
    def test_flags_new_device(self):
        state = _make_state(device="NEW_DEV-XYZ123")
        result = transaction_monitoring_agent(state)

        rule_ids = [f["rule_id"] for f in result["anomaly_flags"]]
        assert "new_device" in rule_ids

    def test_no_flag_known_device(self):
        state = _make_state(device="DEV-KNOWN-ABC123")
        result = transaction_monitoring_agent(state)

        rule_ids = [f["rule_id"] for f in result["anomaly_flags"]]
        assert "new_device" not in rule_ids


# ── Multiple flags ───────────────────────────────────────────────────


class TestMultipleFlags:
    def test_multiple_anomalies_all_flagged(self):
        """A high-value transfer from a foreign location on a new device."""
        state = _make_state(
            amount=15_000.00,
            location="Foreign - Lagos, NG",
            device="NEW_DEV-SUSPICIOUS",
        )
        result = transaction_monitoring_agent(state)

        assert result["is_anomalous"] is True
        rule_ids = {f["rule_id"] for f in result["anomaly_flags"]}
        assert rule_ids == {"high_value_txn", "geo_anomaly", "new_device"}
