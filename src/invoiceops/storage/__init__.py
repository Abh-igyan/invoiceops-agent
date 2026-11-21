"""Dual-mode storage adapters (Local SQLite/NumPy for P1-P3; BigQuery for P4/v5)."""

from typing import Any

from invoiceops.storage.local_store import LocalSqliteStore, RetrievedChunk

try:
    from invoiceops.storage.bigquery_store import BigQueryVectorStore
except ImportError:
    BigQueryVectorStore: Any = None  # type: ignore[no-redef, assignment]

__all__ = ["BigQueryVectorStore", "LocalSqliteStore", "RetrievedChunk"]
