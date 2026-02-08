# FraudDetective (ComplianceCore)

Multi-agent fraud detection and compliance pipeline built on LangGraph. The system ingests transaction feeds, runs monitoring and pattern analysis, computes a composite risk score, enforces an action (allow/flag/hold/block), and generates compliance logs and SAR drafts.

## What This Project Does
- Runs a LangGraph pipeline that routes clean transactions through a fast path and sends suspicious ones through full analysis.
- Produces an explainable risk score with component breakdowns.
- Records a compliance audit trail and auto-drafts SAR entries above threshold.
- Supports CSV/Parquet ingestion and optional Neo4j graph enrichment.

## Architecture At A Glance
Pipeline stages:
1. Monitoring -> anomaly flags, is_anomalous
2. Pattern Detection -> detected_patterns (Neo4j-backed when available)
3. Risk Assessment -> risk_score, risk_breakdown
4. Alert/Block -> action_taken
5. Compliance -> compliance_logs (+ SAR drafts when required)

Conditional routing:
- Clean transactions skip Pattern/Risk/Alert and go to Compliance with fast-path defaults.
- Suspicious transactions flow through the full pipeline.

## Key Modules
- `fraud_detection/core/graph.py`: LangGraph pipeline wiring and routing.
- `fraud_detection/core/state.py`: Typed state schema for the pipeline.
- `fraud_detection/core/loader.py`: CSV/Parquet ingestion and validation.
- `fraud_detection/agents/monitoring.py`: Rule-based anomaly detection.
- `fraud_detection/agents/pattern_detection.py`: Neo4j-backed patterns + heuristics.
- `fraud_detection/agents/risk_assessment.py`: Composite scoring and breakdown.
- `fraud_detection/agents/alert_block.py`: Action decisioning.
- `fraud_detection/agents/compliance.py`: Audit/SAR logging.

## Requirements
- Python >= 3.12
- Dependencies are in `pyproject.toml`

Optional integrations:
- Neo4j (graph enrichment for pattern detection)
- Future LLM + RL modules (placeholders are in code/config)

## Quick Start
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Run the pipeline (default CSV):

```bash
python run_pipeline.py
```

3. Run with a specific file:

```bash
python run_pipeline.py data/sample_transactions.csv
python run_pipeline.py data/transactions.parquet
```

4. Demo mode (hardcoded transactions):

```bash
python run_pipeline.py --demo
```

## Configuration
Tune thresholds and weights in `fraud_detection/config/settings.py`:
- Risk thresholds (block/hold/flag)
- Pattern confidence thresholds
- Risk component weights
- SAR generation threshold

Neo4j connection settings are also defined there and can be overridden by env vars:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Neo4j (Optional)
When Neo4j is available and reachable, pattern detection uses graph queries for:
- Mule network fan-in
- Structuring patterns
- Shared-device identity rings

To seed demo graph data (requires Neo4j running):

```bash
python seed_graph_data.py
```

## Tests
```bash
pytest
```

## Data Format
The loader expects a transaction feed with at least:
- `transaction_id`
- `account_id`
- `amount`

Supported columns (aliases are accepted and normalized):
`transaction_id`, `account_id`, `amount`, `currency`, `merchant_id`,
`merchant_category`, `timestamp`, `location`, `channel`, `ip_address`,
`device_fingerprint`, `recipient_account`, `transaction_type`, `metadata`

See `data/sample_transactions.csv` for an example.

## Output
The pipeline emits a per-transaction summary and a batch summary table.
Each transaction result includes:
- `risk_score` and `risk_breakdown`
- `action_taken` with reason and notified parties
- `anomaly_flags` and `detected_patterns`
- `compliance_logs` (audit + action + optional SAR draft)

## Upcoming Improvements
- Replace placeholder heuristics with production models and feature store lookups.
- Add real-time velocity counters and geo-distance checks.
- Implement Neo4j enrichment for historical risk and pattern priors.
- Add ML model inference and optional LLM-generated risk narratives.
- Add RL-based score adjustments and action policy learning.
- Add human-in-the-loop checkpoints for hold decisions.
- Stream pipeline results to Kafka or event bus for downstream consumers.
- Expand integrations (case management, payment switch, notification gateway).

## License
MIT
