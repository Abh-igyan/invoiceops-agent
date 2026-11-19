# ADR-004: Semantic Clause Chunking, Active-Version Metadata Filtering, and Hybrid BM25 + Vector Reranking

- **Status:** Accepted
- **Date:** 2026-11-18
- **Phase:** P3 (`v2` Retrieval Experiment — `EXP-002`)

---

## 1. Context

In Phase P2 (`EXP-001`), decomposing `v0` into `v1` (`extraction_agent` $\rightarrow$ deterministic `compute_three_way_match` $\rightarrow$ `policy_rag_agent` $\rightarrow$ `orchestrator_agent`) raised decision accuracy from `61.11%` to `83.33%` while deliberately holding the policy index constant (`v0` naive 420-character sliding windows with 60-character overlap and unfiltered cosine similarity).

Inspecting `v1` traces (`results/eval_runs.jsonl`) and running a controlled retrieval ablation across `golden_12` ($n=12$), `dev` ($n=240$), and frozen `test` ($n=160$) revealed three structural defects in naive fixed-character vector retrieval:
1. **Mid-Clause Truncation:** Fixed 420-character windows split multi-condition rules mid-sentence (for example separating a clause ID header `POL-02-V2#2` from its `> 2.0% to 10.0% -> hold` threshold action), capping `dev_240` `Recall@5` at `90.00%` and `MRR` at `0.6812`.
2. **Superseded Policy Version Contamination:** Unfiltered vector similarity scored deprecated `v1.0` trap policies (`POL-02-V1#1` 5% price tolerance, `POL-05-V1#1` $25,000 approval limit, `POL-09-V1#1` 90-day timeliness window) nearly identically to active `v2.0` policies. Switching to semantic clause chunks *without* metadata filtering (`v2a`) actually **increased** superseded-clause contamination from `47.50%` to `50.00%` on `dev_240` because complete `V1` clauses have cleaner embeddings than truncated windows.
3. **Dense-Only Lexical Blindness on Exact Codes & Thresholds:** Pure cosine similarity (`v2b`) achieved `100.00%` `Recall@5` once filtered to `status = 'active'`, but ranked secondary clauses ahead of the primary governing clause on multi-constraint cases (`MRR = 0.8750` on `golden_12`, `0.8958` on `dev_240`).

---

## 2. Decisions

### 2.1 Atomic Semantic Clause Chunking (`build_semantic_clause_chunks`)
Instead of arbitrary character windows over raw Markdown, `v2` indexes each atomic `PolicyClause` (`25` total clauses across `15` versioned AP documents: `19` active, `6` superseded) as a self-contained unit in `policy_chunks_v2` (mirroring the BigQuery `policy_clauses` schema for Phase P4). Each chunk embeds structured provenance inline (`[POL-XX#N | status=ACTIVE | version=v2.0 | effective=YYYY-MM-DD | topic=...]`) alongside dedicated SQL columns (`clause_id`, `doc_id`, `version`, `effective_from`, `status`, `topic`).

### 2.2 Hard Metadata Pre-Filter (`status = 'active'`)
All production policy searches enforce `WHERE status = 'active'` at the storage layer (`LocalSqliteStore.search_hybrid_v2` in P3; BigQuery `VECTOR_SEARCH` pre-filter in P4). Superseded clauses remain stored in the table for historical auditability and ablation testing, but are excluded from active agent context—reducing superseded trap contamination to **`0.00%`** across all 400 dataset cases.

### 2.3 Linear Hybrid Scoring: Dense Cosine + Okapi BM25 + Domain Topic Reranker
`search_hybrid_v2` combines three normalized signals in `[0.0, 1.0]` over active clause candidates:
$$\text{score}(q, c) = 0.50 \cdot \cos(\mathbf{e}_q, \mathbf{e}_c) + 0.25 \cdot \widehat{\text{BM25}}(q, c) + 0.25 \cdot \text{TopicRerank}(q, c)$$
- **Dense Vector (`0.50`):** `gemini-embedding-2` cosine similarity captures paraphrased semantic intent.
- **Sparse Lexical (`0.25`):** Normalized Okapi BM25 (`k1=1.5, b=0.75`) matches exact numeric thresholds (`2%`, `10%`, `60`, `$500`, `$10,000`), field names (`received_date`, `expected_tax_rate`), and clause IDs.
- **Lightweight Domain Reranker (`0.25`) + Deterministic Query Expansion:** `build_hybrid_query_expansion` appends structured anomaly codes from `compute_three_way_match` (plus `security` / `POL-12#1` tags when `scan_untrusted_invoice_fields` flags adversarial supplier notes), and `_compute_topic_rerank_scores` boosts clauses whose `topic` or `clause_id` aligns with the detected anomalies—lifting `MRR` to **`1.0000`** across `golden_12`, `dev_240`, and `test_160` with zero extra LLM reranking calls.

---

## 3. Options Considered & Trade-offs

| Option | Pros | Cons / Why Rejected or Accepted |
|---|---|---|
| **Chosen: Atomic Clause Chunks + `status='active'` Filter + Weighted Dense/BM25/Topic Rerank** | `100%` Recall@5, `1.000` MRR, `0%` superseded contamination; zero extra LLM latency or cost; translates directly to BigQuery `VECTOR_SEARCH` + SQL metadata/lexical scoring in P4 | Requires structured clause metadata (`status`, `version`, `topic`) at ingestion time—appropriate for governed enterprise SOPs/policies |
| **Alternative 1: Unfiltered Semantic Chunking + Prompt Warning (`v2a`)** | Simple ingestion; relies on Orchestrator reading `[SUPERSEDED]` header | Empirically contaminated `50.0%` of `dev_240` top-5 contexts with stale `V1` rules, wasting context window slots and risking stale-threshold hallucinations |
| **Alternative 2: LLM Cross-Encoder Reranker (`gemini-3.5-flash-lite` listwise rerank)** | Flexible on unstructured prose | Adds +1 LLM call (`~4–6s` Free-Tier pacing latency and `~1,200` input tokens/case) for zero accuracy gain over deterministic BM25 + topic reranking on a structured policy corpus |
| **Alternative 3: Vertex AI Ranking API / Discovery Engine** | Managed cross-encoder | Explicitly out of scope per Section 2 ("Google services to use — and ONLY these": Gemini, ADK, BigQuery, Cloud Run, OTel, Secret Manager, Terraform) |

---

## 4. Consequences

1. **Controlled Isolation:** Holding models (`gemini-3.5-flash-lite`), prompts, sequential ERP calls, and the 3-agent topology identical to `v1` proved that fixing retrieval alone raised end-to-end decision accuracy from `83.33%` to **`97.22%`** and citation validity from `83.34%` to **`97.22%`**.
2. **Remaining Bottlenecks Handed Off to `v3` and `v4`:**
   - Sequential ERP calls and full-document JSON payloads keep `v2` input tokens around `7,142/case` (`$0.00354/case`), motivating `v3` (`EXP-003`) tiered model routing, prompt trimming, and parallel async ERP lookups.
   - The sole remaining failure across 36 `v2` evaluations (`INV-4219` in Run 1, `SCHEMA_VALIDATION_FAILED` from a raw newline split inside `"MISSING_OR_INVALID\n_PO"`) motivates `v4` (`EXP-005`) JSON control-character sanitization, bounded schema retries, and the Critic/Verifier agent.
