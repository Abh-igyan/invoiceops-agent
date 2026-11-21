"""BigQuery Vector Store and Evaluation Run Store for Google Cloud (Phase P4/v5).

Executes native SQL `VECTOR_SEARCH` with cosine distance over 768/3072-dim embeddings,
supporting pre-filtered active clause retrieval, subquery parameterization,
and evaluation metric persistence into BigQuery `eval_runs`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from google.cloud import bigquery
from google.oauth2.credentials import Credentials

from invoiceops.schemas import CaseEvalRecord
from invoiceops.storage.local_store import RetrievedChunk

_DEFAULT_PROJECT_ID = "project-11d8a312-cb03-44f0-baa"

logger = logging.getLogger("invoiceops.storage.bigquery")


def _get_bigquery_credentials() -> Any:
    """Resolve credentials via Application Default Credentials or local gcloud token fallback."""
    import google.auth

    try:
        creds, _ = google.auth.default()
        return creds
    except Exception:
        # Fallback to local gcloud CLI token when running locally
        try:
            token = subprocess.check_output(
                "gcloud auth print-access-token", shell=True, text=True
            ).strip()
            return Credentials(token=token)
        except Exception as exc:
            logger.warning("Failed to obtain gcloud credentials: %s", exc)
            return None


class BigQueryVectorStore:
    """Production vector store using native Google Cloud BigQuery VECTOR_SEARCH."""

    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str = "invoiceops",
        policy_table: str = "policy_clauses",
        eval_runs_table: str = "eval_runs",
        location: str = "US",
        client: bigquery.Client | None = None,
    ) -> None:
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", _DEFAULT_PROJECT_ID)
        self.dataset_id = dataset_id
        self.policy_table = policy_table
        self.eval_runs_table = eval_runs_table
        self.location = location

        if client is not None:
            self._client = client
        else:
            creds = _get_bigquery_credentials()
            self._client = bigquery.Client(
                project=self.project_id,
                credentials=creds,
                location=self.location,
            )

    @property
    def client(self) -> bigquery.Client:
        return self._client

    def search_vector(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        active_only: bool = True,
    ) -> list[RetrievedChunk]:
        """Perform native BigQuery VECTOR_SEARCH with pre-filtering on status."""
        table_path = f"`{self.project_id}.{self.dataset_id}.{self.policy_table}`"
        status_filter = "WHERE status = 'active'" if active_only else ""

        sql = f"""
        SELECT
          base.clause_id,
          base.doc_id,
          base.text,
          base.status,
          base.topic,
          base.version,
          distance
        FROM VECTOR_SEARCH(
          (
            SELECT clause_id, doc_id, text, status, topic, version, embedding
            FROM {table_path}
            {status_filter}
          ),
          'embedding',
          (SELECT @query_embedding AS embedding),
          top_k => {int(top_k)},
          distance_type => 'COSINE'
        )
        ORDER BY distance ASC, base.clause_id ASC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding)
            ]
        )

        rows = list(self._client.query_and_wait(sql, job_config=job_config))
        chunks: list[RetrievedChunk] = []
        for r in rows:
            # Cosine distance to similarity: similarity = 1 - distance
            score = 1.0 - float(r.distance) if r.distance is not None else 0.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=r.clause_id,
                    doc_id=r.doc_id,
                    score=round(score, 6),
                    clause_ids=[r.clause_id],
                    text=r.text,
                    status=r.status,
                    topic=r.topic or "",
                    version=r.version,
                )
            )
        return chunks

    def search_hybrid_v2(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        active_only: bool = True,
        vector_weight: float = 0.50,
        lexical_weight: float = 0.25,
        rerank_weight: float = 0.25,
    ) -> list[RetrievedChunk]:
        """Perform BigQuery VECTOR_SEARCH with active clause pre-filter for v5."""
        return self.search_vector(
            query_embedding=query_vector,
            top_k=top_k,
            active_only=active_only,
        )

    def get_active_clause_ids(self) -> set[str]:
        """Return all active clause IDs from BigQuery."""
        table_path = f"`{self.project_id}.{self.dataset_id}.{self.policy_table}`"
        sql = f"SELECT clause_id FROM {table_path} WHERE status = 'active'"
        rows = list(self._client.query_and_wait(sql))
        return {r.clause_id for r in rows}

    def get_clause_text(self, clause_id: str) -> str | None:
        """Return formatted clause text for a given clause_id."""
        table_path = f"`{self.project_id}.{self.dataset_id}.{self.policy_table}`"
        sql = f"SELECT text FROM {table_path} WHERE clause_id = @clause_id LIMIT 1"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("clause_id", "STRING", clause_id)]
        )
        rows = list(self._client.query_and_wait(sql, job_config=job_config))
        if rows:
            return rows[0].text
        return None

    def record_eval_run(self, record: CaseEvalRecord) -> None:
        """Insert single evaluation case record into BigQuery eval_runs table."""
        table_ref = self._client.dataset(self.dataset_id).table(self.eval_runs_table)
        row = [
            {
                "run_id": record.run_id,
                "experiment_version": record.experiment_version,
                "config_version": record.config_version,
                "git_sha": record.git_sha,
                "split_name": record.split_name,
                "rep_index": record.rep_index,
                "case_id": record.case_id,
                "failure_mode": record.failure_mode,
                "is_prompt_injection": record.is_prompt_injection,
                "ground_truth_decision": record.ground_truth_decision,
                "predicted_decision": record.predicted_decision,
                "is_correct": record.is_correct,
                "weighted_error_cost": record.weighted_error_cost,
                "citation_valid": record.citation_valid,
                "retrieval_recall_at_k": record.retrieval_recall_at_k,
            }
        ]
        errors = self._client.insert_rows_json(table_ref, row)
        if errors:
            logger.error("Failed to insert eval_run row into BigQuery: %s", errors)
