"""
Batch Transaction Loader
=========================

Loads transaction data from CSV or Parquet files and converts each row
into a `TransactionData` TypedDict ready for the fraud-detection pipeline.

Supported formats:
    - CSV  (.csv)  -- via Python's built-in `csv` module
    - Parquet (.parquet, .pq) -- via `pyarrow.parquet`

Usage:
    from fraud_detection.core.loader import load_transactions

    # Auto-detects format from extension
    transactions = load_transactions("data/sample_transactions.csv")
    transactions = load_transactions("data/sample_transactions.parquet")

    # Explicit format override
    transactions = load_transactions("data/feed.dat", fmt="csv")
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from fraud_detection.core.state import TransactionData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mapping -- normalises external column names to our schema
# ---------------------------------------------------------------------------

# Maps common alternative column names found in bank feeds / vendor exports
# to the canonical field names in TransactionData.
_COLUMN_ALIASES: dict[str, str] = {
    # Canonical -> itself (identity, for completeness)
    "transaction_id": "transaction_id",
    "account_id": "account_id",
    "amount": "amount",
    "currency": "currency",
    "merchant_id": "merchant_id",
    "merchant_category": "merchant_category",
    "timestamp": "timestamp",
    "location": "location",
    "channel": "channel",
    "ip_address": "ip_address",
    "device_fingerprint": "device_fingerprint",
    "recipient_account": "recipient_account",
    "transaction_type": "transaction_type",
    # Common aliases
    "txn_id": "transaction_id",
    "trans_id": "transaction_id",
    "acct_id": "account_id",
    "acct": "account_id",
    "amt": "amount",
    "ccy": "currency",
    "merch_id": "merchant_id",
    "mcc": "merchant_category",
    "ts": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "loc": "location",
    "city": "location",
    "chan": "channel",
    "ip": "ip_address",
    "device_fp": "device_fingerprint",
    "device_id": "device_fingerprint",
    "recipient": "recipient_account",
    "dest_account": "recipient_account",
    "txn_type": "transaction_type",
    "type": "transaction_type",
}

# Fields in TransactionData that are expected (used for validation)
_REQUIRED_FIELDS = {"transaction_id", "account_id", "amount"}
_NUMERIC_FIELDS = {"amount"}

# All valid TransactionData field names
_VALID_FIELDS = {
    "transaction_id", "account_id", "amount", "currency", "merchant_id",
    "merchant_category", "timestamp", "location", "channel", "ip_address",
    "device_fingerprint", "recipient_account", "transaction_type", "metadata",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_column_name(raw: str) -> str:
    """Map a raw column header to its canonical TransactionData field name."""
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return _COLUMN_ALIASES.get(cleaned, cleaned)


def _coerce_row(raw_row: dict[str, str]) -> TransactionData:
    """Convert a raw string-valued dict (from CSV) into a typed TransactionData.

    - Normalises column names via aliases
    - Coerces `amount` to float
    - Puts unrecognised columns into `metadata`
    - Replaces missing values with empty strings
    """
    normalised: dict[str, Any] = {}
    overflow: dict[str, Any] = {}

    for raw_key, raw_val in raw_row.items():
        canonical = _normalise_column_name(raw_key)
        val = raw_val.strip() if isinstance(raw_val, str) else raw_val

        if canonical in _VALID_FIELDS:
            normalised[canonical] = val
        else:
            # Column doesn't map to any known field -> stash in metadata
            overflow[canonical] = val

    # Type coercions
    if "amount" in normalised:
        try:
            normalised["amount"] = float(normalised["amount"])
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid amount '{normalised['amount']}' in transaction "
                f"{normalised.get('transaction_id', '?')} -- defaulting to 0.0"
            )
            normalised["amount"] = 0.0

    # Merge overflow into metadata
    existing_meta = normalised.get("metadata", {})
    if isinstance(existing_meta, str):
        existing_meta = {}
    normalised["metadata"] = {**existing_meta, **overflow}

    # Fill missing fields with defaults
    for field in _VALID_FIELDS:
        if field not in normalised:
            normalised[field] = {} if field == "metadata" else ""

    return TransactionData(**{k: v for k, v in normalised.items() if k in _VALID_FIELDS})  # type: ignore[misc]


def _validate_transactions(transactions: list[TransactionData]) -> list[TransactionData]:
    """Validate and filter loaded transactions.  Log warnings for issues."""
    valid: list[TransactionData] = []

    for i, txn in enumerate(transactions):
        issues: list[str] = []

        for field in _REQUIRED_FIELDS:
            val = txn.get(field)  # type: ignore[arg-type]
            if val is None or (isinstance(val, str) and not val.strip()):
                issues.append(f"missing required field '{field}'")

        amount = txn.get("amount", 0.0)
        if isinstance(amount, (int, float)) and amount < 0:
            issues.append(f"negative amount: {amount}")

        if issues:
            txn_id = txn.get("transaction_id", f"row-{i}")
            logger.warning(f"Transaction {txn_id} has issues: {'; '.join(issues)} -- skipping")
            continue

        valid.append(txn)

    skipped = len(transactions) - len(valid)
    if skipped:
        logger.warning(f"Skipped {skipped}/{len(transactions)} transactions due to validation errors")

    return valid


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[TransactionData]:
    """Load transactions from a CSV file."""
    logger.info(f"Loading CSV: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    logger.info(f"  Read {len(raw_rows)} rows from {path.name}")
    transactions = [_coerce_row(row) for row in raw_rows]
    return _validate_transactions(transactions)


# ---------------------------------------------------------------------------
# Parquet loader
# ---------------------------------------------------------------------------

def _load_parquet(path: Path) -> list[TransactionData]:
    """Load transactions from a Parquet file.

    Requires `pyarrow` to be installed.
    """
    logger.info(f"Loading Parquet: {path}")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support.  "
            "Install it with: pip install pyarrow"
        )

    table = pq.read_table(path)
    # Convert to list of dicts using columnar batch conversion (fast)
    col_data = {col: table.column(col).to_pylist() for col in table.column_names}
    raw_rows: list[dict[str, str]] = [
        {col: str(col_data[col][i]) for col in table.column_names}
        for i in range(table.num_rows)
    ]

    logger.info(f"  Read {len(raw_rows)} rows from {path.name}")
    transactions = [_coerce_row(row) for row in raw_rows]
    return _validate_transactions(transactions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FORMAT_LOADERS = {
    "csv": _load_csv,
    "parquet": _load_parquet,
    "pq": _load_parquet,
}


def load_transactions(
    path: str | Path,
    fmt: str | None = None,
) -> list[TransactionData]:
    """
    Load transactions from a file.

    Parameters
    ----------
    path : str or Path
        Path to the transaction data file.
    fmt : str, optional
        Explicit format override ("csv", "parquet", "pq").
        If not provided, the format is inferred from the file extension.

    Returns
    -------
    list[TransactionData]
        Validated transactions ready to feed into the pipeline.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format cannot be determined or is unsupported.
    """
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"Transaction file not found: {filepath}")

    # Determine format
    if fmt is None:
        ext = filepath.suffix.lstrip(".").lower()
        fmt = ext

    if fmt not in _FORMAT_LOADERS:
        supported = ", ".join(sorted(_FORMAT_LOADERS.keys()))
        raise ValueError(
            f"Unsupported format '{fmt}' for file {filepath.name}.  "
            f"Supported formats: {supported}"
        )

    loader = _FORMAT_LOADERS[fmt]
    transactions = loader(filepath)

    logger.info(f"Loaded {len(transactions)} valid transactions from {filepath.name}")
    return transactions
