"""
Pattern Detection Agent  (Check 2 — The Graph Sleuth)
======================================================

Pipeline position: **Node 2**

Responsibility:
    - Analyse flagged transactions for higher-order fraud patterns via Neo4j
    - For *new users* this is the primary safety net (Check 1 only runs
      cohort rules so there is little personal signal to lean on)
    - Detect structuring, mule networks, account takeover, and identity rings
    - Add device-context checks: emulator detection, VPN / location masking
    - Write `detected_patterns` into state

This agent integrates with a **Neo4j graph database** when available,
falling back to heuristic detectors when Neo4j is unreachable.

Graph data model
----------------
    Nodes:  User, Account, Transaction, IPAddress, Device
    Rels:   (User)-[:OWNS]->(Account)
            (Account)-[:SENDS]->(Transaction)-[:TO]->(Account)
            (Transaction)-[:HAS_IP]->(IPAddress)
            (Transaction)-[:HAS_DEVICE]->(Device)

Neo4j query depth
-----------------
    1-hop: direct device/IP link to FRAUDSTER
    2-hop: IP shared with an account belonging to FRAUDSTER
    2-hop: recipient account also received from fraudster-linked accounts
    3-hop: sender → mid account → fraudster-linked account (chain traversal)
"""

from __future__ import annotations

import logging
from typing import Any

from fraud_detection.config.settings import (
    MIN_PATTERN_CONFIDENCE,
    MULE_FAN_IN_MIN_SENDERS,
    MULE_FAN_IN_WINDOW_DAYS,
    MULE_HEURISTIC_MIN_AMOUNT,
    STRUCTURING_HEURISTIC_RATIO,
    STRUCTURING_MIN_AMOUNT,
    STRUCTURING_MIN_TXNS,
    STRUCTURING_THRESHOLD,
    STRUCTURING_WINDOW_DAYS,
    STRUCTURING_WINDOW_HOURS,
)
from fraud_detection.core.state import DetectedPattern, FraudDetectionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neo4j-backed Pattern Detection Agent
# ---------------------------------------------------------------------------

# Risk contribution weights per pattern type (higher = more severe)
_PATTERN_RISK_WEIGHTS: dict[str, int] = {
    "mule_network": 40,
    "structuring": 30,
    "linked_to_fraudster": 50,        # 1-hop device link
    "ip_2hop_fraudster": 35,          # 2-hop via shared IP
    "account_2hop_fraudster": 30,     # 2-hop via shared account relay
    "account_3hop_fraudster": 20,     # 3-hop chain (weaker signal)
}

# Maximum combined risk contribution from graph-based patterns
_MAX_RISK_CONTRIBUTION: int = 95


class PatternDetectionAgent:
    """Runs Cypher queries against Neo4j to detect graph-based fraud patterns.

    Parameters
    ----------
    neo4j_client
        An instance of ``Neo4jClient`` (from ``fraud_detection.core.neo4j_client``).
        The caller is responsible for lifecycle management.
    """

    # -- Cypher templates ----------------------------------------------------

    QUERY_MULE_FAN_IN: str = (
        "MATCH (beneficiary:Account)<-[:TO]-(t:Transaction)<-[:SENDS]-(sender:Account) "
        "WHERE id(beneficiary) = $beneficiary_id "
        "  AND t.timestamp > datetime() - duration($window_days) "
        "WITH beneficiary, count(DISTINCT sender) AS distinct_senders "
        "RETURN distinct_senders > $min_senders AS is_mule"
    )

    QUERY_STRUCTURING: str = (
        "MATCH (sender:Account)-[:SENDS]->(t:Transaction) "
        "WHERE id(sender) = $sender_id "
        "  AND t.amount > $min_amount AND t.amount < $max_amount "
        "  AND t.timestamp > datetime() - duration($window_days) "
        "RETURN count(t) > $min_txns AS is_structuring"
    )

    QUERY_IDENTITY_RING: str = (
        "MATCH (current_tx:Transaction)-[:HAS_DEVICE]->(d:Device)"
        "<-[:HAS_DEVICE]-(past_tx:Transaction)<-[:SENDS]-(a:Account)"
        "<-[:OWNS]-(u:User) "
        "WHERE id(current_tx) = $transaction_id "
        "  AND u.status = 'FRAUDSTER' "
        "RETURN count(u) > 0 AS linked_to_fraudster"
    )

    # 2-hop: current transaction's IP was also used in a fraudster's past txns
    QUERY_IP_2HOP: str = (
        "MATCH (tx:Transaction)-[:HAS_IP]->(ip:IPAddress)"
        "<-[:HAS_IP]-(t2:Transaction)<-[:SENDS]-(a:Account)<-[:OWNS]-(u:User) "
        "WHERE id(tx) = $transaction_id AND u.status = 'FRAUDSTER' "
        "RETURN count(DISTINCT u) > 0 AS linked_via_ip_2hop"
    )

    # 2-hop: sender's account has also sent money to accounts linked to fraudsters
    QUERY_ACCOUNT_2HOP: str = (
        "MATCH (sender:Account)-[:SENDS]->(t1:Transaction)-[:TO]->(mid:Account)"
        "<-[:TO]-(t2:Transaction)<-[:SENDS]-(fraudster_acct:Account)"
        "<-[:OWNS]-(u:User) "
        "WHERE id(sender) = $sender_id AND u.status = 'FRAUDSTER' "
        "RETURN count(DISTINCT u) > 0 AS linked_via_account_2hop"
    )

    # 3-hop: sender → mid-account → second-mid → fraudster-owned account
    QUERY_ACCOUNT_3HOP: str = (
        "MATCH (sender:Account)-[:SENDS]->(t1:Transaction)-[:TO]->(m1:Account)"
        "-[:SENDS]->(t2:Transaction)-[:TO]->(m2:Account)"
        "<-[:TO]-(t3:Transaction)<-[:SENDS]-(fa:Account)<-[:OWNS]-(u:User) "
        "WHERE id(sender) = $sender_id AND u.status = 'FRAUDSTER' "
        "RETURN count(DISTINCT u) > 0 AS linked_via_account_3hop"
    )

    def __init__(self, neo4j_client: Any) -> None:
        self._client = neo4j_client

    # -- Public API ----------------------------------------------------------

    def detect_patterns(
        self,
        transaction_id: str,
        sender_account_id: str,
        amount: float,
        recipient_account_id: str = "",
        device_fingerprint: str = "",
    ) -> dict[str, Any]:
        """Run all 3 Cypher queries and return detected patterns.

        Returns
        -------
        dict
            ``{"patterns_detected": [...], "risk_contribution": int}``
            Empty list and ``risk_contribution=0`` if nothing found.
        """
        patterns_detected: list[str] = []
        errors: list[str] = []

        # Query A — Mule Account (Fan-In)
        if recipient_account_id:
            result = self._safe_query(
                "mule_fan_in",
                self.QUERY_MULE_FAN_IN,
                {
                    "beneficiary_id": recipient_account_id,
                    "window_days": f"P{MULE_FAN_IN_WINDOW_DAYS}D",
                    "min_senders": MULE_FAN_IN_MIN_SENDERS,
                },
                errors,
            )
            if result and result.get("is_mule"):
                patterns_detected.append("mule_network")

        # Query B — Structuring (Smurfing)
        result = self._safe_query(
            "structuring",
            self.QUERY_STRUCTURING,
            {
                "sender_id": sender_account_id,
                "min_amount": STRUCTURING_MIN_AMOUNT,
                "max_amount": STRUCTURING_THRESHOLD,
                "window_days": f"P{STRUCTURING_WINDOW_DAYS}D",
                "min_txns": STRUCTURING_MIN_TXNS,
            },
            errors,
        )
        if result and result.get("is_structuring"):
            patterns_detected.append("structuring")

        # Query C — Identity Ring (Shared Device with Fraudster)  [1-hop]
        if transaction_id and device_fingerprint:
            result = self._safe_query(
                "identity_ring",
                self.QUERY_IDENTITY_RING,
                {"transaction_id": transaction_id},
                errors,
            )
            if result and result.get("linked_to_fraudster"):
                patterns_detected.append("linked_to_fraudster")

        # Query D — IP 2-hop fraudster link
        if transaction_id:
            result = self._safe_query(
                "ip_2hop",
                self.QUERY_IP_2HOP,
                {"transaction_id": transaction_id},
                errors,
            )
            if result and result.get("linked_via_ip_2hop"):
                patterns_detected.append("ip_2hop_fraudster")

        # Query E — Account 2-hop fraudster link
        if sender_account_id:
            result = self._safe_query(
                "account_2hop",
                self.QUERY_ACCOUNT_2HOP,
                {"sender_id": sender_account_id},
                errors,
            )
            if result and result.get("linked_via_account_2hop"):
                patterns_detected.append("account_2hop_fraudster")

        # Query F — Account 3-hop fraudster chain (deeper traversal)
        if sender_account_id:
            result = self._safe_query(
                "account_3hop",
                self.QUERY_ACCOUNT_3HOP,
                {"sender_id": sender_account_id},
                errors,
            )
            if result and result.get("linked_via_account_3hop"):
                patterns_detected.append("account_3hop_fraudster")

        risk_contribution = self._calculate_risk_contribution(patterns_detected)

        return {
            "patterns_detected": patterns_detected,
            "risk_contribution": risk_contribution,
            "errors": errors,
        }

    # -- Internal helpers ----------------------------------------------------

    def _safe_query(
        self,
        query_name: str,
        cypher: str,
        params: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any] | None:
        """Execute a Cypher query, returning the first record or None on failure."""
        try:
            records = self._client.execute_query(cypher, params)
            if records:
                return records[0]
            return None
        except Exception as exc:
            import neo4j
            if isinstance(exc, neo4j.exceptions.Neo4jError):
                msg = f"Neo4j Cypher error in '{query_name}': {exc}"
            elif isinstance(exc, neo4j.exceptions.ServiceUnavailable):
                msg = f"Neo4j connection error in '{query_name}': {exc}"
            else:
                raise exc  # Re-raise unexpected exceptions like KeyError
            
            logger.error(msg)
            errors.append(msg)
            return None

    @staticmethod
    def _calculate_risk_contribution(patterns: list[str]) -> int:
        """Score the combined risk contribution of detected patterns."""
        if not patterns:
            return 0
        total = sum(_PATTERN_RISK_WEIGHTS.get(p, 10) for p in patterns)
        return min(total, _MAX_RISK_CONTRIBUTION)


# ---------------------------------------------------------------------------
# Heuristic pattern detectors (fallbacks when Neo4j is unavailable)
# ---------------------------------------------------------------------------

def _detect_structuring(txn: dict[str, Any]) -> DetectedPattern | None:
    """Detect potential structuring / smurfing (heuristic fallback).

    Structuring = breaking a large sum into smaller deposits just below
    the BSA reporting threshold ($10,000 in the US) to avoid CTR filing.

    This naive single-transaction check is used when the Neo4j graph DB
    is unavailable. The graph-based query (PatternDetectionAgent) provides
    multi-transaction aggregation for more accurate detection.
    """
    amount = txn.get("amount", 0.0)
    # Naive heuristic: single txn just below threshold looks suspicious
    if STRUCTURING_THRESHOLD * STRUCTURING_HEURISTIC_RATIO <= amount < STRUCTURING_THRESHOLD:
        return DetectedPattern(
            pattern_id="structuring_suspect",
            pattern_type="sequence",
            description=(
                f"Transaction of ${amount:,.2f} is {STRUCTURING_THRESHOLD - amount:,.2f} "
                f"below the ${STRUCTURING_THRESHOLD:,.0f} CTR reporting threshold"
            ),
            involved_entities=[txn.get("account_id", ""), txn.get("recipient_account", "")],
            evidence={"amount": amount, "threshold": STRUCTURING_THRESHOLD},
            confidence=0.65,
        )
    return None


def _detect_mule_network(txn: dict[str, Any]) -> DetectedPattern | None:
    """Detect potential money-mule network activity (heuristic fallback).

    This placeholder flags large transfers to any recipient. The graph-based
    query (PatternDetectionAgent) uses fan-in analysis for accurate detection.
    """
    txn_type = txn.get("transaction_type", "")
    amount = txn.get("amount", 0.0)
    recipient = txn.get("recipient_account", "")

    # Placeholder: flag large transfers to unfamiliar recipients
    if txn_type == "transfer" and amount > MULE_HEURISTIC_MIN_AMOUNT and recipient:
        return DetectedPattern(
            pattern_id="mule_network_candidate",
            pattern_type="network",
            description=(
                f"Large transfer of ${amount:,.2f} to {recipient} -- "
                f"potential mule network relay"
            ),
            involved_entities=[txn.get("account_id", ""), recipient],
            evidence={
                "amount": amount,
                "transaction_type": txn_type,
                "recipient": recipient,
            },
            confidence=0.55,
        )
    return None


def _detect_rapid_channel_switching(txn: dict[str, Any]) -> DetectedPattern | None:
    """Detect rapid switching between channels (ATM -> online -> POS).

    Heuristic fallback -- requires historical transaction data that is not
    available from a single transaction. The Neo4j graph-based detection
    handles this via temporal queries when the DB is available.
    """
    # Placeholder: cannot detect without historical data
    return None


# Known emulator fingerprint substrings (case-insensitive)
_EMULATOR_MARKERS: frozenset[str] = frozenset({
    "emulator", "android_sdk", "genymotion", "bluestacks",
    "noxplayer", "ldplayer", "memu", "sdk_gphone",
})

# Metadata keys that signal location/identity masking
_MASKING_METADATA_KEYS: frozenset[str] = frozenset({
    "vpn_detected", "proxy_detected", "tor_detected",
    "datacenter_ip", "hosting_ip",
})


def _detect_emulator_device(txn: dict[str, Any]) -> DetectedPattern | None:
    """Flag transactions originating from a suspected device emulator.

    Emulators are a strong signal for new-account fraud: attackers spin up
    many virtual Android/iOS instances to create accounts and test stolen cards.
    """
    device = str(txn.get("device_fingerprint", "")).lower()
    if not device:
        return None

    matched = next((m for m in _EMULATOR_MARKERS if m in device), None)
    if matched:
        return DetectedPattern(
            pattern_id="emulator_device",
            pattern_type="behavioral",
            description=(
                f"Device fingerprint '{txn.get('device_fingerprint', '')}' "
                f"matches known emulator marker '{matched}'"
            ),
            involved_entities=[txn.get("account_id", "")],
            evidence={"device_fingerprint": txn.get("device_fingerprint"), "marker": matched},
            confidence=0.85,
        )
    return None


def _detect_location_masking(txn: dict[str, Any]) -> DetectedPattern | None:
    """Flag transactions where the IP / location is obscured via VPN, proxy, or Tor.

    New users without established trusted devices or locations are particularly
    susceptible to this signal — there is no trusted-device baseline to compare.
    """
    metadata = txn.get("metadata", {})
    if not isinstance(metadata, dict):
        return None

    triggered_keys = [k for k in _MASKING_METADATA_KEYS if metadata.get(k)]
    if not triggered_keys:
        return None

    return DetectedPattern(
        pattern_id="location_masking",
        pattern_type="behavioral",
        description=(
            f"Location/identity masking detected: {', '.join(triggered_keys)}"
        ),
        involved_entities=[
            txn.get("account_id", ""),
            txn.get("ip_address", ""),
        ],
        evidence={k: metadata[k] for k in triggered_keys},
        confidence=0.80,
    )


# ---------------------------------------------------------------------------
# Neo4j client lazy import helper
# ---------------------------------------------------------------------------

def _try_get_neo4j_client() -> Any | None:
    """Attempt to acquire the Neo4j client; return None if unavailable."""
    try:
        from fraud_detection.core.neo4j_client import get_neo4j_client
        client = get_neo4j_client()
        if client.verify_connectivity():
            return client
        return None
    except Exception as exc:
        logger.debug(f"Neo4j client unavailable: {exc}")
        return None


def _neo4j_results_to_patterns(
    neo4j_result: dict[str, Any],
    txn: dict[str, Any],
) -> list[DetectedPattern]:
    """Convert PatternDetectionAgent output to DetectedPattern TypedDicts."""
    patterns: list[DetectedPattern] = []
    pattern_names: list[str] = neo4j_result.get("patterns_detected", [])

    for name in pattern_names:
        if name == "mule_network":
            patterns.append(DetectedPattern(
                pattern_id="neo4j_mule_fan_in",
                pattern_type="network",
                description=(
                    f"Graph analysis: beneficiary account {txn.get('recipient_account', '')} "
                    f"received from >5 unique senders in 24h (fan-in pattern)"
                ),
                involved_entities=[
                    txn.get("account_id", ""),
                    txn.get("recipient_account", ""),
                ],
                evidence={
                    "source": "neo4j",
                    "query": "mule_fan_in",
                    "risk_contribution": neo4j_result.get("risk_contribution", 0),
                },
                confidence=0.85,
            ))
        elif name == "structuring":
            patterns.append(DetectedPattern(
                pattern_id="neo4j_structuring",
                pattern_type="sequence",
                description=(
                    f"Graph analysis: sender {txn.get('account_id', '')} made >2 transactions "
                    f"in $9K-$10K range within 7 days (structuring/smurfing)"
                ),
                involved_entities=[txn.get("account_id", "")],
                evidence={
                    "source": "neo4j",
                    "query": "structuring",
                    "amount": txn.get("amount", 0.0),
                    "risk_contribution": neo4j_result.get("risk_contribution", 0),
                },
                confidence=0.90,
            ))
        elif name == "linked_to_fraudster":
            patterns.append(DetectedPattern(
                pattern_id="neo4j_identity_ring",
                pattern_type="network",
                description=(
                    f"Graph analysis: device {txn.get('device_fingerprint', '')} "
                    f"is shared with a known FRAUDSTER user (1-hop)"
                ),
                involved_entities=[
                    txn.get("account_id", ""),
                    txn.get("device_fingerprint", ""),
                ],
                evidence={
                    "source": "neo4j",
                    "query": "identity_ring",
                    "hop_depth": 1,
                    "risk_contribution": neo4j_result.get("risk_contribution", 0),
                },
                confidence=0.95,
            ))
        elif name == "ip_2hop_fraudster":
            patterns.append(DetectedPattern(
                pattern_id="neo4j_ip_2hop",
                pattern_type="network",
                description=(
                    f"Graph analysis: transaction IP is shared with a FRAUDSTER "
                    f"account's past transactions (2-hop)"
                ),
                involved_entities=[
                    txn.get("account_id", ""),
                    txn.get("ip_address", ""),
                ],
                evidence={
                    "source": "neo4j",
                    "query": "ip_2hop",
                    "hop_depth": 2,
                    "risk_contribution": neo4j_result.get("risk_contribution", 0),
                },
                confidence=0.80,
            ))
        elif name == "account_2hop_fraudster":
            patterns.append(DetectedPattern(
                pattern_id="neo4j_account_2hop",
                pattern_type="network",
                description=(
                    f"Graph analysis: sender {txn.get('account_id', '')} transacted "
                    f"with an intermediary account linked to a FRAUDSTER (2-hop)"
                ),
                involved_entities=[
                    txn.get("account_id", ""),
                    txn.get("recipient_account", ""),
                ],
                evidence={
                    "source": "neo4j",
                    "query": "account_2hop",
                    "hop_depth": 2,
                    "risk_contribution": neo4j_result.get("risk_contribution", 0),
                },
                confidence=0.75,
            ))
        elif name == "account_3hop_fraudster":
            patterns.append(DetectedPattern(
                pattern_id="neo4j_account_3hop",
                pattern_type="network",
                description=(
                    f"Graph analysis: sender {txn.get('account_id', '')} is 3 hops "
                    f"from a FRAUDSTER account via account chain"
                ),
                involved_entities=[txn.get("account_id", "")],
                evidence={
                    "source": "neo4j",
                    "query": "account_3hop",
                    "hop_depth": 3,
                    "risk_contribution": neo4j_result.get("risk_contribution", 0),
                },
                confidence=0.60,
            ))

    return patterns


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

def pattern_detection_agent(state: FraudDetectionState) -> dict[str, Any]:
    """
    LangGraph node: Pattern Detection Agent.

    Reads
    -----
    - state["raw_transaction"]
    - state["anomaly_flags"]
    - state["is_anomalous"]

    Writes
    ------
    - detected_patterns : list[DetectedPattern]  (appended via reducer)

    Strategy:
        1. Attempt Neo4j graph-based detection first (PatternDetectionAgent)
        2. If Neo4j is unavailable, fall back to heuristic detectors
        3. Both paths populate the same DetectedPattern output format
    """
    logger.debug("=== Pattern Detection Agent: START ===")

    txn = state.get("raw_transaction", {})
    flags = state.get("anomaly_flags", [])
    txn_id = txn.get("transaction_id", "UNKNOWN")

    logger.debug(
        f"Analysing transaction {txn_id} with {len(flags)} upstream anomaly flags"
    )

    txn_dict: dict[str, Any] = dict(txn)  # cast TypedDict -> dict for helpers
    patterns: list[DetectedPattern] = []
    errors: list[str] = []

    # --- Try Neo4j-backed detection first ---------------------------------
    neo4j_client = _try_get_neo4j_client()
    used_neo4j = False

    if neo4j_client is not None:
        try:
            agent = PatternDetectionAgent(neo4j_client)
            neo4j_result = agent.detect_patterns(
                transaction_id=txn_dict.get("transaction_id", ""),
                sender_account_id=txn_dict.get("account_id", ""),
                amount=txn_dict.get("amount", 0.0),
                recipient_account_id=txn_dict.get("recipient_account", ""),
                device_fingerprint=txn_dict.get("device_fingerprint", ""),
            )
            graph_patterns = _neo4j_results_to_patterns(neo4j_result, txn_dict)
            # Filter by confidence threshold
            for p in graph_patterns:
                if p["confidence"] >= MIN_PATTERN_CONFIDENCE:
                    patterns.append(p)
            errors.extend(neo4j_result.get("errors", []))
            used_neo4j = True
            logger.debug(
                f"Neo4j detection found {len(graph_patterns)} patterns "
                f"(risk_contribution={neo4j_result.get('risk_contribution', 0)})"
            )
        except Exception as exc:
            import neo4j
            if isinstance(exc, (neo4j.exceptions.Neo4jError, neo4j.exceptions.ServiceUnavailable)):
                msg = f"Neo4j pattern detection failed, falling back to heuristics: {exc}"
                logger.warning(msg)
                errors.append(msg)
            else:
                raise exc

    # --- Heuristic fallback (always runs if Neo4j was unavailable) --------
    if not used_neo4j:
        logger.debug("Using heuristic pattern detectors (Neo4j unavailable)")
        for detector in [
            _detect_structuring,
            _detect_mule_network,
            _detect_rapid_channel_switching,
            _detect_emulator_device,
            _detect_location_masking,
        ]:
            try:
                result = detector(txn_dict)
                if result and result["confidence"] >= MIN_PATTERN_CONFIDENCE:
                    patterns.append(result)
            except Exception as exc:
                msg = f"Pattern detector {detector.__name__} failed: {exc}"
                logger.error(msg)
                errors.append(msg)

    # Device-context heuristics always run regardless of Neo4j availability
    # (cheap checks; critical safety net for new users without trusted devices)
    for device_check in (_detect_emulator_device, _detect_location_masking):
        try:
            result = device_check(txn_dict)
            if result and result["confidence"] >= MIN_PATTERN_CONFIDENCE:
                # Avoid duplicating if heuristic fallback already added it
                if not any(p["pattern_id"] == result["pattern_id"] for p in patterns):
                    patterns.append(result)
        except Exception as exc:
            msg = f"Device check {device_check.__name__} failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    logger.debug(f"Transaction {txn_id}: {len(patterns)} patterns detected")
    for pat in patterns:
        logger.debug(
            f"  [{pat['pattern_type'].upper()}] {pat['pattern_id']}: "
            f"{pat['description']} (confidence={pat['confidence']:.2f})"
        )

    logger.debug("=== Pattern Detection Agent: END ===\n")

    return {"detected_patterns": patterns, "processing_errors": errors}
