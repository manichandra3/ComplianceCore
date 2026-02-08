"""Tests for the Pattern Detection Agent (Node 2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fraud_detection.agents.pattern_detection import (
    PatternDetectionAgent,
    pattern_detection_agent,
    _neo4j_results_to_patterns,
)
from fraud_detection.core.state import FraudDetectionState, TransactionData


def _make_state(
    amount: float = 100.0,
    transaction_type: str = "purchase",
    recipient_account: str = "",
    channel: str = "pos",
    **txn_overrides,
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
        recipient_account=recipient_account,
        transaction_type=transaction_type,
        metadata={},
        **txn_overrides,
    )
    return FraudDetectionState(
        raw_transaction=txn,
        anomaly_flags=[],
        is_anomalous=False,
    )


# ── Structuring detection ────────────────────────────────────────────


class TestStructuringDetection:
    def test_flags_amount_just_below_ctr_threshold(self):
        """$9,400 is 94% of $10K -- should trigger structuring detection."""
        state = _make_state(amount=9_400.00)
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "structuring_suspect" in pattern_ids

    def test_flags_amount_at_80_percent(self):
        """$8,000 is exactly 80% -- boundary case, should trigger."""
        state = _make_state(amount=8_000.00)
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "structuring_suspect" in pattern_ids

    def test_no_flag_below_80_percent(self):
        """$7,999 is below the 80% threshold."""
        state = _make_state(amount=7_999.00)
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "structuring_suspect" not in pattern_ids

    def test_no_flag_at_or_above_threshold(self):
        """$10,000 exactly is at/above CTR threshold -- not structuring."""
        state = _make_state(amount=10_000.00)
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "structuring_suspect" not in pattern_ids

    def test_structuring_pattern_type_is_sequence(self):
        state = _make_state(amount=9_500.00)
        result = pattern_detection_agent(state)

        pattern = next(p for p in result["detected_patterns"]
                       if p["pattern_id"] == "structuring_suspect")
        assert pattern["pattern_type"] == "sequence"


# ── Mule network detection ───────────────────────────────────────────


class TestMuleNetworkDetection:
    def test_flags_large_transfer_with_recipient(self):
        state = _make_state(
            amount=5_000.00,
            transaction_type="transfer",
            recipient_account="MULE-ACCT-999",
        )
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "mule_network_candidate" in pattern_ids

    def test_no_flag_for_small_transfer(self):
        state = _make_state(
            amount=2_000.00,
            transaction_type="transfer",
            recipient_account="MULE-ACCT-999",
        )
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "mule_network_candidate" not in pattern_ids

    def test_no_flag_for_purchase(self):
        """Purchases (not transfers) should not trigger mule detection."""
        state = _make_state(
            amount=5_000.00,
            transaction_type="purchase",
            recipient_account="",
        )
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "mule_network_candidate" not in pattern_ids

    def test_no_flag_without_recipient(self):
        state = _make_state(
            amount=5_000.00,
            transaction_type="transfer",
            recipient_account="",
        )
        result = pattern_detection_agent(state)

        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "mule_network_candidate" not in pattern_ids

    def test_mule_pattern_type_is_network(self):
        state = _make_state(
            amount=5_000.00,
            transaction_type="transfer",
            recipient_account="MULE-ACCT-999",
        )
        result = pattern_detection_agent(state)

        pattern = next(p for p in result["detected_patterns"]
                       if p["pattern_id"] == "mule_network_candidate")
        assert pattern["pattern_type"] == "network"


# ── Clean transaction ────────────────────────────────────────────────


class TestCleanTransaction:
    def test_no_patterns_for_normal_purchase(self):
        state = _make_state(amount=42.50)
        result = pattern_detection_agent(state)

        assert result["detected_patterns"] == []

    def test_returns_processing_errors_key(self):
        result = pattern_detection_agent(_make_state())
        assert "processing_errors" in result
        assert result["processing_errors"] == []


# ══════════════════════════════════════════════════════════════════════
# PatternDetectionAgent (Neo4j-backed) tests — mocked driver
# ══════════════════════════════════════════════════════════════════════


def _mock_client(responses: dict[str, list[dict]]) -> MagicMock:
    """Build a mock Neo4jClient that returns canned responses per query.

    Parameters
    ----------
    responses
        Mapping from a *substring* of the Cypher query to the list of
        record-dicts that ``execute_query`` should return.
    """
    client = MagicMock()

    def _execute_query(query: str, params: dict | None = None):
        for key, records in responses.items():
            if key in query:
                return records
        return []

    client.execute_query.side_effect = _execute_query
    return client


class TestPatternDetectionAgentMuleQuery:
    """Query A — Mule Account (Fan-In Pattern)."""

    def test_detects_mule_when_positive(self):
        client = _mock_client({"is_mule": [{"is_mule": True}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=5000.0,
            recipient_account_id="ACCT-2",
        )
        assert "mule_network" in result["patterns_detected"]

    def test_no_mule_when_negative(self):
        client = _mock_client({"is_mule": [{"is_mule": False}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=5000.0,
            recipient_account_id="ACCT-2",
        )
        assert "mule_network" not in result["patterns_detected"]

    def test_skips_mule_query_without_recipient(self):
        client = _mock_client({})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=5000.0,
            recipient_account_id="",
        )
        assert "mule_network" not in result["patterns_detected"]
        # Should not have called execute_query for mule at all
        for call in client.execute_query.call_args_list:
            assert "is_mule" not in call[0][0]


class TestPatternDetectionAgentStructuringQuery:
    """Query B — Structuring (Smurfing)."""

    def test_detects_structuring_when_positive(self):
        client = _mock_client({"is_structuring": [{"is_structuring": True}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
        )
        assert "structuring" in result["patterns_detected"]

    def test_no_structuring_when_negative(self):
        client = _mock_client({"is_structuring": [{"is_structuring": False}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
        )
        assert "structuring" not in result["patterns_detected"]


class TestPatternDetectionAgentIdentityRingQuery:
    """Query C — Identity Ring (Shared Device with Fraudster)."""

    def test_detects_linked_to_fraudster(self):
        client = _mock_client({"linked_to_fraudster": [{"linked_to_fraudster": True}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=1000.0,
            device_fingerprint="DEV-SUSPICIOUS",
        )
        assert "linked_to_fraudster" in result["patterns_detected"]

    def test_no_fraudster_when_negative(self):
        client = _mock_client({"linked_to_fraudster": [{"linked_to_fraudster": False}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=1000.0,
            device_fingerprint="DEV-CLEAN",
        )
        assert "linked_to_fraudster" not in result["patterns_detected"]

    def test_skips_identity_ring_without_device_fingerprint(self):
        client = _mock_client({})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=1000.0,
            device_fingerprint="",
        )
        assert "linked_to_fraudster" not in result["patterns_detected"]


class TestPatternDetectionAgentCombined:
    """Tests covering multiple patterns and risk scoring."""

    def test_all_patterns_detected(self):
        client = _mock_client({
            "is_mule": [{"is_mule": True}],
            "is_structuring": [{"is_structuring": True}],
            "linked_to_fraudster": [{"linked_to_fraudster": True}],
        })
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
            recipient_account_id="ACCT-2",
            device_fingerprint="DEV-BAD",
        )
        assert set(result["patterns_detected"]) == {
            "mule_network", "structuring", "linked_to_fraudster"
        }

    def test_no_patterns_when_all_negative(self):
        client = _mock_client({
            "is_mule": [{"is_mule": False}],
            "is_structuring": [{"is_structuring": False}],
            "linked_to_fraudster": [{"linked_to_fraudster": False}],
        })
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
            recipient_account_id="ACCT-2",
            device_fingerprint="DEV-OK",
        )
        assert result["patterns_detected"] == []
        assert result["risk_contribution"] == 0

    def test_risk_contribution_single_mule(self):
        client = _mock_client({"is_mule": [{"is_mule": True}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=5000.0,
            recipient_account_id="ACCT-2",
        )
        assert result["risk_contribution"] == 40  # mule_network weight

    def test_risk_contribution_single_structuring(self):
        client = _mock_client({"is_structuring": [{"is_structuring": True}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
        )
        assert result["risk_contribution"] == 30  # structuring weight

    def test_risk_contribution_single_fraudster(self):
        client = _mock_client({"linked_to_fraudster": [{"linked_to_fraudster": True}]})
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=1000.0,
            device_fingerprint="DEV-BAD",
        )
        assert result["risk_contribution"] == 50  # linked_to_fraudster weight

    def test_risk_contribution_capped_at_max(self):
        """All 3 patterns = 40+30+50 = 120, should be capped at 95."""
        client = _mock_client({
            "is_mule": [{"is_mule": True}],
            "is_structuring": [{"is_structuring": True}],
            "linked_to_fraudster": [{"linked_to_fraudster": True}],
        })
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
            recipient_account_id="ACCT-2",
            device_fingerprint="DEV-BAD",
        )
        assert result["risk_contribution"] == 95  # capped at _MAX_RISK_CONTRIBUTION


class TestPatternDetectionAgentErrorHandling:
    """Verify resilience when Neo4j queries fail."""

    def test_query_failure_appends_to_errors(self):
        client = MagicMock()
        client.execute_query.side_effect = RuntimeError("Connection refused")
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
            recipient_account_id="ACCT-2",
            device_fingerprint="DEV-X",
        )
        # No patterns detected, but errors reported
        assert result["patterns_detected"] == []
        assert result["risk_contribution"] == 0
        assert len(result["errors"]) == 3  # all 3 queries fail

    def test_partial_failure_returns_successful_patterns(self):
        """If one query fails but others succeed, return what we can."""
        call_count = 0

        def _side_effect(query, params=None):
            nonlocal call_count
            call_count += 1
            if "is_mule" in query:
                return [{"is_mule": True}]
            if "is_structuring" in query:
                raise RuntimeError("timeout")
            if "linked_to_fraudster" in query:
                return [{"linked_to_fraudster": True}]
            return []

        client = MagicMock()
        client.execute_query.side_effect = _side_effect
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
            recipient_account_id="ACCT-2",
            device_fingerprint="DEV-X",
        )
        assert "mule_network" in result["patterns_detected"]
        assert "linked_to_fraudster" in result["patterns_detected"]
        assert "structuring" not in result["patterns_detected"]
        assert len(result["errors"]) == 1

    def test_empty_result_set_returns_no_pattern(self):
        """When Neo4j returns empty records for a query, treat as no match."""
        client = _mock_client({})  # empty responses for everything
        agent = PatternDetectionAgent(client)
        result = agent.detect_patterns(
            transaction_id="TXN-1",
            sender_account_id="ACCT-1",
            amount=9500.0,
            recipient_account_id="ACCT-2",
            device_fingerprint="DEV-X",
        )
        assert result["patterns_detected"] == []
        assert result["risk_contribution"] == 0
        assert result["errors"] == []


class TestNeo4jResultsToPatterns:
    """Test the conversion from PatternDetectionAgent output to DetectedPattern."""

    def test_mule_network_conversion(self):
        neo4j_result = {"patterns_detected": ["mule_network"], "risk_contribution": 40}
        txn = {"account_id": "ACCT-1", "recipient_account": "ACCT-2"}
        patterns = _neo4j_results_to_patterns(neo4j_result, txn)
        assert len(patterns) == 1
        assert patterns[0]["pattern_id"] == "neo4j_mule_fan_in"
        assert patterns[0]["pattern_type"] == "network"
        assert patterns[0]["confidence"] == 0.85

    def test_structuring_conversion(self):
        neo4j_result = {"patterns_detected": ["structuring"], "risk_contribution": 30}
        txn = {"account_id": "ACCT-1", "amount": 9500.0}
        patterns = _neo4j_results_to_patterns(neo4j_result, txn)
        assert len(patterns) == 1
        assert patterns[0]["pattern_id"] == "neo4j_structuring"
        assert patterns[0]["pattern_type"] == "sequence"
        assert patterns[0]["confidence"] == 0.90

    def test_identity_ring_conversion(self):
        neo4j_result = {"patterns_detected": ["linked_to_fraudster"], "risk_contribution": 50}
        txn = {"account_id": "ACCT-1", "device_fingerprint": "DEV-BAD"}
        patterns = _neo4j_results_to_patterns(neo4j_result, txn)
        assert len(patterns) == 1
        assert patterns[0]["pattern_id"] == "neo4j_identity_ring"
        assert patterns[0]["pattern_type"] == "network"
        assert patterns[0]["confidence"] == 0.95

    def test_empty_patterns_returns_empty_list(self):
        neo4j_result = {"patterns_detected": [], "risk_contribution": 0}
        patterns = _neo4j_results_to_patterns(neo4j_result, {})
        assert patterns == []

    def test_all_patterns_conversion(self):
        neo4j_result = {
            "patterns_detected": ["mule_network", "structuring", "linked_to_fraudster"],
            "risk_contribution": 95,
        }
        txn = {"account_id": "ACCT-1", "recipient_account": "ACCT-2",
               "device_fingerprint": "DEV-X", "amount": 9500.0}
        patterns = _neo4j_results_to_patterns(neo4j_result, txn)
        assert len(patterns) == 3
        pattern_ids = {p["pattern_id"] for p in patterns}
        assert pattern_ids == {"neo4j_mule_fan_in", "neo4j_structuring", "neo4j_identity_ring"}


class TestNodeFunctionGracefulDegradation:
    """Verify the LangGraph node function falls back to heuristics when Neo4j is down."""

    @patch("fraud_detection.agents.pattern_detection._try_get_neo4j_client")
    def test_uses_heuristics_when_neo4j_unavailable(self, mock_get_client):
        """When _try_get_neo4j_client returns None, heuristic detectors run."""
        mock_get_client.return_value = None
        state = _make_state(amount=9_400.00)
        result = pattern_detection_agent(state)

        # Should use heuristic structuring detector
        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "structuring_suspect" in pattern_ids

    @patch("fraud_detection.agents.pattern_detection._try_get_neo4j_client")
    def test_uses_neo4j_when_available(self, mock_get_client):
        """When Neo4j client is available, use graph-based detection."""
        mock_client = _mock_client({
            "is_mule": [{"is_mule": True}],
            "is_structuring": [{"is_structuring": False}],
        })
        mock_get_client.return_value = mock_client

        state = _make_state(
            amount=5_000.00,
            transaction_type="transfer",
            recipient_account="ACCT-DEST",
        )
        result = pattern_detection_agent(state)

        # Should use Neo4j-based patterns, not heuristic
        pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
        assert "neo4j_mule_fan_in" in pattern_ids
        # Heuristic patterns should NOT appear since Neo4j was used
        assert "mule_network_candidate" not in pattern_ids
        assert "structuring_suspect" not in pattern_ids

    @patch("fraud_detection.agents.pattern_detection._try_get_neo4j_client")
    def test_falls_back_on_neo4j_exception(self, mock_get_client):
        """If Neo4j client is returned but detect_patterns raises, fall back."""
        mock_client = MagicMock()
        mock_client.execute_query.side_effect = RuntimeError("Driver crashed")
        mock_client.verify_connectivity.return_value = True
        mock_get_client.return_value = mock_client

        # Use the mock_client but make PatternDetectionAgent itself raise
        # during instantiation by patching at the class level
        with patch(
            "fraud_detection.agents.pattern_detection.PatternDetectionAgent",
            side_effect=RuntimeError("init failed"),
        ):
            state = _make_state(amount=9_400.00)
            result = pattern_detection_agent(state)

            # Should have fallen back to heuristics
            pattern_ids = [p["pattern_id"] for p in result["detected_patterns"]]
            assert "structuring_suspect" in pattern_ids
            # Error should be logged
            assert len(result["processing_errors"]) >= 1
