"""Local SQLite + NumPy vector index and `eval_runs` store (mirrors BigQuery schema)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from invoiceops.config import AppConfig, load_runtime_secrets
from invoiceops.schemas import CaseEvalRecord

CLAUSE_ID_PATTERN = re.compile(r"\b(POL-\d{2}(?:-V\d+)?#\d+)\b")


@dataclass(frozen=True)
class RetrievedChunk:
    """Single vector-search hit returned by naive policy retrieval."""

    chunk_id: str
    doc_id: str
    score: float
    clause_ids: list[str]
    text: str


def build_naive_chunks(
    policies_dir: Path,
    chunk_chars: int = 420,
    overlap_chars: int = 60,
) -> list[dict[str, Any]]:
    """Split raw policy Markdown files into naive fixed-character sliding windows (`v0`).

    Deliberately ignores document `status` (active vs superseded) and splits by character
    count rather than semantic clause boundaries to establish an honest `v0` baseline.
    """
    chunks: list[dict[str, Any]] = []
    step = max(1, chunk_chars - overlap_chars)

    for md_path in sorted(policies_dir.glob("*.md")):
        doc_id = md_path.stem
        raw = md_path.read_text(encoding="utf-8")
        # Track active clause heading at every character offset so mid-clause windows
        # still carry ground-truth clause attribution for recall@k evaluation.
        active_clause_by_pos: list[str | None] = [None] * len(raw)
        current_clause: str | None = None
        pos = 0
        for line in raw.splitlines(keepends=True):
            found = CLAUSE_ID_PATTERN.findall(line)
            if found:
                current_clause = found[0]
            for offset in range(len(line)):
                if pos + offset < len(active_clause_by_pos):
                    active_clause_by_pos[pos + offset] = current_clause
            pos += len(line)

        idx = 0
        chunk_num = 0
        while idx < len(raw):
            window = raw[idx : idx + chunk_chars].strip()
            if window:
                explicit_ids = CLAUSE_ID_PATTERN.findall(window)
                inherited = active_clause_by_pos[idx] if idx < len(active_clause_by_pos) else None
                clause_ids: list[str] = []
                for cid in ([inherited] if inherited else []) + explicit_ids:
                    if cid and cid not in clause_ids:
                        clause_ids.append(cid)
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}::naive_{chunk_num:02d}",
                        "doc_id": doc_id,
                        "clause_ids": clause_ids,
                        "text": window,
                        "sha256": hashlib.sha256(window.encode("utf-8")).hexdigest(),
                    }
                )
                chunk_num += 1
            if idx + chunk_chars >= len(raw):
                break
            idx += step

    return chunks


class GeminiEmbedder:
    """Thin wrapper around `google-genai` `embed_content` using configured embedding model."""

    def __init__(self, config: AppConfig, dotenv_path: Path | None = None) -> None:
        self.model_id = config.models.embedding
        secrets = load_runtime_secrets(dotenv_path)
        api_key = secrets.gemini_api_key.get_secret_value() if secrets.gemini_api_key else None
        self._client = genai.Client(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in bounded batches using explicit `types.Content` items.

        Note: `gemini-embedding-2` is multimodal and treats a bare `list[str]` as a single
        multi-part document (returning 1 vector), whereas `list[types.Content]` produces
        one vector per input string consistently across both `gemini-embedding-2` and `001`.
        """
        vectors: list[list[float]] = []
        batch_size = 40
        for start in range(0, len(texts), batch_size):
            batch = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in texts[start : start + batch_size]
            ]
            resp = self._client.models.embed_content(model=self.model_id, contents=batch)
            if not resp.embeddings:
                msg = "Embedding API returned empty embeddings list"
                raise RuntimeError(msg)
            for emb in resp.embeddings:
                if not emb.values:
                    msg = "Embedding item missing vector values"
                    raise RuntimeError(msg)
                vectors.append(list(emb.values))
        return vectors


class LocalSqliteStore:
    """SQLite storage for cached policy embeddings and `eval_runs` telemetry rows."""

    def __init__(self, db_path: Path, jsonl_path: Path | None = None) -> None:
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_chunks_v0 (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    clause_ids_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT NOT NULL,
                    experiment_version TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    git_sha TEXT NOT NULL,
                    split_name TEXT NOT NULL,
                    rep_index INTEGER NOT NULL,
                    case_id TEXT NOT NULL,
                    failure_mode TEXT NOT NULL,
                    is_prompt_injection INTEGER NOT NULL,
                    ground_truth_decision TEXT NOT NULL,
                    predicted_decision TEXT NOT NULL,
                    predicted_reason_codes_json TEXT NOT NULL,
                    cited_ids_json TEXT NOT NULL,
                    retrieved_clause_ids_json TEXT NOT NULL,
                    is_correct INTEGER NOT NULL,
                    weighted_error_cost REAL NOT NULL,
                    citation_valid INTEGER NOT NULL,
                    retrieval_recall_at_k REAL NOT NULL,
                    retrieval_mrr REAL NOT NULL,
                    injection_resisted INTEGER,
                    failure_category TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    llm_calls INTEGER NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    erp_retries INTEGER NOT NULL,
                    resolved_model TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def index_naive_chunks(
        self,
        chunks: list[dict[str, Any]],
        embed_fn: Callable[[list[str]], list[list[float]]],
    ) -> int:
        """Index naive policy chunks, reusing cached vectors when SHA-256 matches."""
        with self._connect() as conn:
            existing = {
                row[0]: (row[1], row[2])
                for row in conn.execute(
                    "SELECT chunk_id, sha256, embedding_json FROM policy_chunks_v0"
                ).fetchall()
            }

        to_embed: list[dict[str, Any]] = []
        ready_rows: list[tuple[str, str, str, str, str, str]] = []

        for ch in chunks:
            cached = existing.get(ch["chunk_id"])
            if cached and cached[0] == ch["sha256"]:
                ready_rows.append(
                    (
                        ch["chunk_id"],
                        ch["doc_id"],
                        json.dumps(ch["clause_ids"]),
                        ch["text"],
                        ch["sha256"],
                        cached[1],
                    )
                )
            else:
                to_embed.append(ch)

        if to_embed:
            new_vectors = embed_fn([c["text"] for c in to_embed])
            for ch, vec in zip(to_embed, new_vectors, strict=True):
                ready_rows.append(
                    (
                        ch["chunk_id"],
                        ch["doc_id"],
                        json.dumps(ch["clause_ids"]),
                        ch["text"],
                        ch["sha256"],
                        json.dumps(vec),
                    )
                )

        with self._connect() as conn:
            conn.execute("DELETE FROM policy_chunks_v0")
            conn.executemany(
                """
                INSERT INTO policy_chunks_v0
                (chunk_id, doc_id, clause_ids_json, text, sha256, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ready_rows,
            )
            conn.commit()

        return len(to_embed)

    def search_naive(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Perform exact cosine similarity search over `policy_chunks_v0`."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, clause_ids_json, text, embedding_json "
                "FROM policy_chunks_v0"
            ).fetchall()

        if not rows:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q = q / q_norm

        mat = np.asarray([json.loads(r[4]) for r in rows], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        sims = (mat / norms) @ q

        top_indices = np.argsort(-sims)[:top_k]
        results: list[RetrievedChunk] = []
        for idx in top_indices:
            row = rows[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=row[0],
                    doc_id=row[1],
                    score=round(float(sims[int(idx)]), 4),
                    clause_ids=json.loads(row[2]),
                    text=row[3],
                )
            )
        return results

    def record_eval_case(self, record: CaseEvalRecord) -> None:
        """Persist one evaluated case row to SQLite `eval_runs` and JSONL log."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    record.run_id,
                    record.experiment_version,
                    record.config_version,
                    record.git_sha,
                    record.split_name,
                    record.rep_index,
                    record.case_id,
                    record.failure_mode,
                    int(record.is_prompt_injection),
                    record.ground_truth_decision,
                    record.predicted_decision,
                    json.dumps(record.predicted_reason_codes),
                    json.dumps(record.cited_ids),
                    json.dumps(record.retrieved_clause_ids),
                    int(record.is_correct),
                    record.weighted_error_cost,
                    int(record.citation_valid),
                    record.retrieval_recall_at_k,
                    record.retrieval_mrr,
                    None if record.injection_resisted is None else int(record.injection_resisted),
                    record.failure_category,
                    record.latency_ms,
                    record.input_tokens,
                    record.output_tokens,
                    record.estimated_cost_usd,
                    record.llm_calls,
                    record.tool_calls,
                    record.erp_retries,
                    record.resolved_model,
                    record.explanation,
                    record.timestamp_utc,
                ),
            )
            conn.commit()

        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
