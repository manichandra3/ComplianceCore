#!/usr/bin/env python3
"""
FraudDetective - Pipeline Execution Script
============================================

Runs the multi-agent fraud detection pipeline against transaction data
loaded from CSV or Parquet files.

Usage:
    source .venv/bin/activate

    # Load from CSV (default: data/sample_transactions.csv)
    python run_pipeline.py

    # Load from a specific file
    python run_pipeline.py data/sample_transactions.csv
    python run_pipeline.py data/transactions.parquet

    # Fall back to hardcoded dummy data (for quick testing)
    python run_pipeline.py --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fraud_detection.core.graph import compile_fraud_detection_graph
from fraud_detection.core.loader import load_transactions
from fraud_detection.core.state import FraudDetectionState, TransactionData

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("FraudDetective")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CSV_PATH = Path(__file__).parent / "data" / "sample_transactions.csv"


# ---------------------------------------------------------------------------
# Hardcoded demo data (kept as a --demo fallback for quick testing)
# ---------------------------------------------------------------------------

def _get_demo_transactions() -> list[TransactionData]:
    """Return 4 hardcoded transactions covering the risk spectrum."""
    return [
        TransactionData(
            transaction_id="TXN-001-CLEAN",
            account_id="ACCT-8821",
            amount=42.50,
            currency="USD",
            merchant_id="MERCH-COFFEE-01",
            merchant_category="food_and_drink",
            timestamp=datetime.now(timezone.utc).isoformat(),
            location="New York, US",
            channel="pos",
            ip_address="",
            device_fingerprint="DEV-KNOWN-A1B2C3",
            recipient_account="",
            transaction_type="purchase",
            metadata={"pos_terminal": "T-4421"},
        ),
        TransactionData(
            transaction_id="TXN-002-SUS-TRANSFER",
            account_id="ACCT-3310",
            amount=15_750.00,
            currency="USD",
            merchant_id="",
            merchant_category="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            location="Foreign - Lagos, NG",
            channel="online",
            ip_address="102.89.45.12",
            device_fingerprint="NEW_DEV-X9F3K2",
            recipient_account="MULE-ACCT-7799",
            transaction_type="transfer",
            metadata={"browser": "Tor Browser 12.0", "vpn_detected": True},
        ),
        TransactionData(
            transaction_id="TXN-003-STRUCTURING",
            account_id="ACCT-5567",
            amount=9_400.00,
            currency="USD",
            merchant_id="",
            merchant_category="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            location="Miami, US",
            channel="atm",
            ip_address="",
            device_fingerprint="DEV-KNOWN-D4E5F6",
            recipient_account="",
            transaction_type="deposit",
            metadata={"atm_id": "ATM-MIA-0042"},
        ),
        TransactionData(
            transaction_id="TXN-004-NEW-DEVICE",
            account_id="ACCT-1122",
            amount=890.00,
            currency="USD",
            merchant_id="MERCH-ELECTRONICS-99",
            merchant_category="electronics",
            timestamp=datetime.now(timezone.utc).isoformat(),
            location="Chicago, US",
            channel="online",
            ip_address="73.162.45.201",
            device_fingerprint="NEW_DEV-Q7W8E9",
            recipient_account="",
            transaction_type="purchase",
            metadata={"first_time_merchant": True},
        ),
    ]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_separator(char: str = "=", width: int = 80) -> None:
    print(f"\n{char * width}")


def _print_result_summary(txn_id: str, result: dict) -> None:
    """Print a concise summary of the pipeline result for one transaction."""
    _print_separator()
    print(f"RESULT SUMMARY: {txn_id}")
    _print_separator("-")

    # Risk
    risk_score = result.get("risk_score", 0.0)
    print(f"  Risk Score:  {risk_score}/100")

    breakdown = result.get("risk_breakdown", {})
    if breakdown:
        print(f"  Breakdown:   anomaly={breakdown.get('anomaly_score', 0):.1f}  "
              f"pattern={breakdown.get('pattern_score', 0):.1f}  "
              f"historical={breakdown.get('historical_score', 0):.1f}  "
              f"velocity={breakdown.get('velocity_score', 0):.1f}  "
              f"model={breakdown.get('model_score', 0):.1f}")

    # Action
    action = result.get("action_taken", {})
    print(f"  Action:      {action.get('action', 'N/A').upper()}")
    print(f"  Reason:      {action.get('reason', 'N/A')}")
    notified = action.get("notified_parties", [])
    print(f"  Notified:    {', '.join(notified) if notified else 'none'}")

    # Anomalies
    flags = result.get("anomaly_flags", [])
    print(f"  Anomalies:   {len(flags)} flag(s)")
    for f in flags:
        print(f"    - [{f.get('severity', '?').upper()}] {f.get('rule_id')}: "
              f"{f.get('description')}")

    # Patterns
    patterns = result.get("detected_patterns", [])
    print(f"  Patterns:    {len(patterns)} detected")
    for p in patterns:
        print(f"    - [{p.get('pattern_type', '?').upper()}] {p.get('pattern_id')}: "
              f"{p.get('description')}")

    # Compliance
    logs = result.get("compliance_logs", [])
    print(f"  Compliance:  {len(logs)} log entries")
    for log_entry in logs:
        print(f"    - [{log_entry.get('event_type')}] {log_entry.get('log_id')}")
        print(f"      {log_entry.get('summary', '')[:120]}...")

    _print_separator()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudDetective - Multi-Agent Fraud Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py                              # load default CSV\n"
            "  python run_pipeline.py data/custom_feed.csv         # load specific CSV\n"
            "  python run_pipeline.py data/transactions.parquet    # load Parquet\n"
            "  python run_pipeline.py --demo                       # use hardcoded data\n"
        ),
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to a CSV or Parquet transaction file.  "
             f"Defaults to {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use hardcoded dummy transactions instead of loading from a file",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "pq"],
        default=None,
        help="Explicit file format (auto-detected from extension by default)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── Load transactions ─────────────────────────────────────────────
    if args.demo:
        logger.info("Using hardcoded demo transactions (--demo mode)")
        transactions = _get_demo_transactions()
        source_label = "demo (hardcoded)"
    else:
        file_path = Path(args.file) if args.file else DEFAULT_CSV_PATH
        logger.info(f"Loading transactions from: {file_path}")
        transactions = load_transactions(file_path, fmt=args.format)
        source_label = str(file_path)

    if not transactions:
        logger.error("No valid transactions to process.  Exiting.")
        sys.exit(1)

    # ── Compile pipeline ──────────────────────────────────────────────
    logger.info("Compiling FraudDetective pipeline graph...")
    pipeline = compile_fraud_detection_graph()

    logger.info(
        f"Pipeline compiled.  Processing {len(transactions)} transactions "
        f"from [{source_label}].\n"
    )

    # ── Process each transaction ──────────────────────────────────────
    results: list[dict] = []
    failed: list[tuple[str, str]] = []  # (txn_id, error_message)

    for i, txn in enumerate(transactions, start=1):
        txn_id = txn.get("transaction_id", f"TXN-{i}")
        _print_separator("#")
        print(f"  PROCESSING TRANSACTION {i}/{len(transactions)}: {txn_id}")
        print(f"  Amount: ${txn.get('amount', 0):,.2f} | "
              f"Type: {txn.get('transaction_type', '?')} | "
              f"Channel: {txn.get('channel', '?')}")
        _print_separator("#")

        # Build initial state
        initial_state: FraudDetectionState = {
            "raw_transaction": txn,
            "pipeline_run_id": f"RUN-{uuid.uuid4().hex[:8].upper()}",
        }

        # Execute the full pipeline with error resilience
        try:
            result = pipeline.invoke(initial_state)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                f"Pipeline FAILED for transaction {txn_id}: {error_msg}"
            )
            failed.append((txn_id, error_msg))
            # Build a minimal result so batch summary can still report it
            results.append({
                "raw_transaction": txn,
                "risk_score": 0.0,
                "action_taken": {"action": "error", "reason": error_msg,
                                 "notified_parties": [], "timestamp": ""},
                "anomaly_flags": [],
                "detected_patterns": [],
                "compliance_logs": [],
                "processing_errors": [error_msg],
            })
            continue

        results.append(result)

        # Print summary
        _print_result_summary(txn_id, result)

    # ── Batch summary table ───────────────────────────────────────────
    _print_separator("*", 80)
    print(f"  BATCH SUMMARY  (source: {source_label})")
    _print_separator("*", 80)
    print(f"  {'Transaction':<30} {'Amount':>12} {'Risk':>8} {'Action':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*10}")
    for r in results:
        txn = r.get("raw_transaction", {})
        print(f"  {txn.get('transaction_id', '?'):<30} "
              f"${txn.get('amount', 0):>10,.2f} "
              f"{r.get('risk_score', 0):>7.1f} "
              f"{r.get('action_taken', {}).get('action', '?').upper():>10}")

    # Report failures
    if failed:
        _print_separator("!", 80)
        print(f"  FAILURES: {len(failed)}/{len(transactions)} transactions failed")
        _print_separator("!", 80)
        for txn_id, error_msg in failed:
            print(f"  {txn_id}: {error_msg}")

    _print_separator("*", 80)

    # ── FUTURE: Stream to Kafka / Event Bus ──────────────────────────
    # In production, results would be published to a Kafka topic:
    #   for result in results:
    #       kafka_producer.send("fraud.pipeline.results",
    #                           value=json.dumps(result))
    # Downstream consumers (dashboards, case management, BI) subscribe
    # to this topic for real-time updates.
    # ------------------------------------------------------------------

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
