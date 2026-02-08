"""
Neo4j Client - Connection management for the Fraud Detection pipeline.
======================================================================

Provides a thin wrapper around the official ``neo4j`` Python driver with:

    - Singleton-style lazy initialisation (one driver per process)
    - Graceful connection verification on first use
    - Context-manager protocol for clean shutdown
    - Configurable via ``fraud_detection.config.settings``

Usage::

    from fraud_detection.core.neo4j_client import get_neo4j_client

    client = get_neo4j_client()
    with client.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS cnt")
        print(result.single()["cnt"])

The client is designed to be imported by any agent that needs graph
queries (Pattern Detection, Risk Assessment, Compliance).
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from fraud_detection.config.settings import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    NEO4J_MAX_CONNECTION_POOL_SIZE,
    NEO4J_CONNECTION_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_driver: Driver | None = None


class Neo4jClient:
    """Manages a single Neo4j driver instance with health-check support.

    Parameters
    ----------
    uri : str
        Bolt URI for the Neo4j instance (e.g. ``bolt://localhost:7687``).
    user : str
        Authentication username.
    password : str
        Authentication password.
    database : str
        Target database name.
    max_connection_pool_size : int
        Maximum number of connections in the pool.
    connection_timeout : float
        Seconds to wait when establishing a new connection.
    """

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
        max_connection_pool_size: int = NEO4J_MAX_CONNECTION_POOL_SIZE,
        connection_timeout: float = NEO4J_CONNECTION_TIMEOUT,
    ) -> None:
        self._uri = uri
        self._database = database
        self._driver: Driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_connection_pool_size,
            connection_timeout=connection_timeout,
        )
        logger.info(f"Neo4j driver created for {uri} (database={database})")

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- public API ---------------------------------------------------------

    def verify_connectivity(self) -> bool:
        """Return True if the driver can reach the Neo4j server."""
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified")
            return True
        except (ServiceUnavailable, AuthError) as exc:
            logger.error(f"Neo4j connectivity check failed: {exc}")
            return False

    def session(self, **kwargs: Any) -> Session:
        """Open a new session against the configured database.

        Any extra ``kwargs`` are forwarded to ``driver.session()``.
        """
        return self._driver.session(database=self._database, **kwargs)

    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query and return results as a list of dicts.

        This is a convenience wrapper for simple single-shot queries.
        For transactions that need write access or multi-statement
        atomicity, use ``self.session()`` directly.
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def close(self) -> None:
        """Shut down the underlying driver and release all connections."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed")

    @property
    def driver(self) -> Driver:
        """Direct access to the underlying ``neo4j.Driver``."""
        return self._driver


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

def get_neo4j_client() -> Neo4jClient:
    """Return (and lazily create) the process-wide Neo4j client."""
    global _driver
    if _driver is None:
        _driver = Neo4jClient()  # type: ignore[assignment]
    return _driver  # type: ignore[return-value]


def close_neo4j_client() -> None:
    """Explicitly close the module-level Neo4j client."""
    global _driver
    if _driver is not None:
        _driver.close()  # type: ignore[union-attr]
        _driver = None
