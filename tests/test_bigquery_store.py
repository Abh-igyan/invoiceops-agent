"""Unit and integration tests for BigQueryVectorStore (Phase P4 / v5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("google.cloud.bigquery")

from invoiceops.schemas import CaseEvalRecord
from invoiceops.storage.bigquery_store import BigQueryVectorStore


class TestBigQueryVectorStoreMocked:
    """Test BigQueryVectorStore logic with a mocked BigQuery client."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def store(self, mock_client: MagicMock) -> BigQueryVectorStore:
        return BigQueryVectorStore(
            project_id="test-project",
            dataset_id="test_dataset",
            policy_table="policy_clauses",
            eval_runs_table="eval_runs",
            location="US",
            client=mock_client,
        )

    def test_search_vector_executes_prefiltered_vector_search(
        self,
        store: BigQueryVectorStore,
        mock_client: MagicMock,
    ) -> None:
        mock_row1 = MagicMock()
        mock_row1.clause_id = "POL-01#1"
        mock_row1.doc_id = "POL-01"
        mock_row1.text = "All invoices require a purchase order."
        mock_row1.status = "active"
        mock_row1.topic = "missing_po"
        mock_row1.version = "v2.0"
        mock_row1.distance = 0.05

        mock_client.query_and_wait.return_value = [mock_row1]

        query_vec = [0.1] * 768
        hits = store.search_vector(query_embedding=query_vec, top_k=5, active_only=True)

        assert len(hits) == 1
        assert hits[0].chunk_id == "POL-01#1"
        assert hits[0].score == 0.95
        assert hits[0].status == "active"
        assert hits[0].topic == "missing_po"

        # Verify SQL passed to query_and_wait contains pre-filter subquery
        call_args = mock_client.query_and_wait.call_args
        sql = call_args[0][0]
        assert "WHERE status = 'active'" in sql
        assert "VECTOR_SEARCH" in sql
        assert "ORDER BY distance ASC" in sql

    def test_get_active_clause_ids(
        self,
        store: BigQueryVectorStore,
        mock_client: MagicMock,
    ) -> None:
        row1, row2 = MagicMock(), MagicMock()
        row1.clause_id = "POL-01#1"
        row2.clause_id = "POL-02#1"
        mock_client.query_and_wait.return_value = [row1, row2]

        active_ids = store.get_active_clause_ids()
        assert active_ids == {"POL-01#1", "POL-02#1"}

    def test_record_eval_run_inserts_row(
        self,
        store: BigQueryVectorStore,
        mock_client: MagicMock,
    ) -> None:
        mock_dataset = MagicMock()
        mock_table = MagicMock()
        mock_client.dataset.return_value = mock_dataset
        mock_dataset.table.return_value = mock_table
        mock_client.insert_rows_json.return_value = []

        record = CaseEvalRecord(
            run_id="run-123",
            experiment_version="v5",
            config_version="p1.0.0",
            git_sha="abcdef1",
            split_name="frozen_160",
            rep_index=1,
            case_id="INV-4001",
            failure_mode="clean_match",
            is_prompt_injection=False,
            ground_truth_decision="approve",
            predicted_decision="approve",
            predicted_reason_codes=["THREE_WAY_MATCH_PASSED"],
            cited_ids=["POL-01#1", "PO-1001", "GRN-2001"],
            retrieved_clause_ids=["POL-01#1"],
            tools_called=["inspect_purchase_order", "match_invoice_to_po_and_grn"],
            is_correct=True,
            weighted_error_cost=0.0,
            citation_valid=True,
            retrieval_recall_at_k=1.0,
            retrieval_mrr=1.0,
            injection_resisted=True,
            failure_category="none",
            latency_ms=1200.0,
            input_tokens=1500,
            output_tokens=300,
            estimated_cost_usd=0.002,
            llm_calls=4,
            tool_calls=6,
            erp_retries=0,
            resolved_model="gemini-3.5-flash-lite",
            explanation="Approved",
            timestamp_utc="2026-11-21T08:00:00Z",
        )

        store.record_eval_run(record)
        mock_client.insert_rows_json.assert_called_once()
        inserted_rows = mock_client.insert_rows_json.call_args[0][1]
        assert len(inserted_rows) == 1
        assert inserted_rows[0]["case_id"] == "INV-4001"
        assert inserted_rows[0]["experiment_version"] == "v5"
