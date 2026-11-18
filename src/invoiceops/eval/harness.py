"""End-to-end evaluation harness (`make eval`) for `invoiceops-agent`."""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from invoiceops.agents.v0_baseline import V0BaselineEvaluatorAgent
from invoiceops.agents.v1_multi_agent import V1MultiAgentEvaluator
from invoiceops.config import DEFAULT_CONFIG_PATH, AppConfig, load_config
from invoiceops.eval.adk_trajectory import export_and_score_adk_golden_subset
from invoiceops.eval.html_report import generate_html_report
from invoiceops.eval.judge import ExplanationJudge
from invoiceops.eval.metrics import (
    classify_failure_taxonomy,
    compute_retrieval_metrics,
    compute_weighted_error_cost,
    summarize_multi_run,
    summarize_single_run,
    validate_citations,
)
from invoiceops.mcp_erp.client import (
    CircuitBreaker,
    ResilientErpMcpClient,
    running_local_erp_server,
)
from invoiceops.mcp_erp.flaky import FlakyInjector, FlakySettings
from invoiceops.mcp_erp.server import ErpDatabase
from invoiceops.schemas import CaseEvalRecord, PolicyClause, SyntheticCase
from invoiceops.storage.local_store import GeminiEmbedder, LocalSqliteStore, build_naive_chunks
from invoiceops.telemetry.tracing import InvoiceOpsTelemetry

REPO_ROOT = Path(__file__).resolve().parents[3]


def _get_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def estimate_eval_shadow_cost_usd(
    config: AppConfig,
    model_id: str,
    n_cases: int,
    runs: int,
    avg_input_tokens: int = 2600,
    avg_output_tokens: int = 280,
) -> float:
    """Pre-flight shadow Paid-Tier USD cost estimate before running an evaluation."""
    per_case = config.estimate_call_cost_usd(
        model_id=model_id,
        input_tokens=avg_input_tokens,
        output_tokens=avg_output_tokens,
    )
    return round(per_case * n_cases * runs, 4)


async def run_evaluation(
    version: str = "v0",
    split_name: str | None = None,
    runs: int | None = None,
    repo_root: Path = REPO_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Execute complete 6-layer evaluation for `version` and persist JSON + SQLite + HTML."""
    cfg = load_config(config_path)
    chosen_split = split_name or cfg.evaluation.default_dev_subset
    num_runs = runs if runs is not None else cfg.evaluation.runs_per_experiment

    generated_dir = repo_root / "data" / "generated"
    splits_dir = repo_root / "data" / "splits"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cases_path = generated_dir / "cases_manifest.json"
    erp_seed_path = generated_dir / "erp_seed.json"
    policies_manifest_path = generated_dir / "policies_manifest.json"
    split_path = splits_dir / "frozen_split_v1.json"
    cal_path = splits_dir / "judge_calibration_40.json"

    if not cases_path.exists() or not split_path.exists():
        msg = "Dataset missing. Run `make data` (or `python scripts/tasks.py data`) first."
        raise FileNotFoundError(msg)

    all_cases = {
        c["case_id"]: SyntheticCase.model_validate(c)
        for c in json.loads(cases_path.read_text(encoding="utf-8"))
    }
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    split_key_map = {
        "dev": "dev_case_ids",
        "test": "test_case_ids",
        "dev_stratified_36": "dev_stratified_36_ids",
        "dev_stratified_24": "dev_stratified_24_ids",
        "golden_12": "golden_12_ids",
    }
    if chosen_split not in split_key_map:
        msg = f"Unknown split '{chosen_split}'. Valid: {list(split_key_map)}"
        raise ValueError(msg)

    target_ids: list[str] = split_data[split_key_map[chosen_split]]
    target_cases = [all_cases[cid] for cid in target_ids]

    est_usd = estimate_eval_shadow_cost_usd(
        config=cfg,
        model_id=cfg.models.v0_baseline,
        n_cases=len(target_cases),
        runs=num_runs,
    )
    if est_usd > cfg.budget_guardrails.max_single_eval_estimated_usd:
        msg = (
            f"Estimated shadow cost ${est_usd:.2f} exceeds guardrail "
            f"${cfg.budget_guardrails.max_single_eval_estimated_usd:.2f}; aborting for approval."
        )
        raise RuntimeError(msg)

    print(
        f"[eval] Version={version} | Split={chosen_split} "
        f"({len(target_cases)} cases x {num_runs} runs) | "
        f"Live Spend=$0.00 (Free Tier) | Shadow Paid Estimate=${est_usd:.4f}",
        flush=True,
    )

    clauses = [
        PolicyClause.model_validate(x)
        for x in json.loads(policies_manifest_path.read_text(encoding="utf-8"))
    ]
    active_clause_ids = {c.clause_id for c in clauses if c.status == "active"}

    erp_db = ErpDatabase.from_seed_file(erp_seed_path)
    valid_erp_ids: set[str] = set(erp_db.vendors.keys()) | set(erp_db.purchase_orders.keys())
    for grn_list in erp_db.grns_by_po.values():
        for grn in grn_list:
            valid_erp_ids.add(grn["grn_id"])
    for hist_list in erp_db.history_by_vendor.values():
        for hist in hist_list:
            valid_erp_ids.add(hist["history_id"])

    dotenv_path = repo_root / ".env"
    embedder = GeminiEmbedder(config=cfg, dotenv_path=dotenv_path)
    store = LocalSqliteStore(
        db_path=repo_root / cfg.storage.local_sqlite_path,
        jsonl_path=repo_root / cfg.storage.local_eval_runs_jsonl,
    )

    naive_chunks = build_naive_chunks(
        policies_dir=generated_dir / "policies",
        chunk_chars=cfg.storage.naive_chunk_chars,
        overlap_chars=cfg.storage.naive_chunk_overlap_chars,
    )
    newly_embedded = store.index_naive_chunks(naive_chunks, embed_fn=embedder.embed_texts)
    print(
        f"[eval] Policy index ready: {len(naive_chunks)} naive chunks "
        f"({newly_embedded} newly embedded).",
        flush=True,
    )

    # Cache query embeddings across repeated runs to save latency and avoid redundant calls
    query_vec_cache: dict[str, list[float]] = {}

    def cached_embed_query(query: str) -> list[float]:
        norm_q = query.strip()
        if norm_q not in query_vec_cache:
            query_vec_cache[norm_q] = embedder.embed_texts([norm_q])[0]
        return query_vec_cache[norm_q]

    git_sha = _get_git_sha(repo_root)
    batch_run_id = (
        f"{version}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    )

    jsonl_file = repo_root / cfg.storage.local_eval_runs_jsonl
    resumed_cache: dict[tuple[int, str], CaseEvalRecord] = {}
    if jsonl_file.exists():
        for raw_line in jsonl_file.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            prev = CaseEvalRecord.model_validate_json(raw_line)
            if (
                prev.experiment_version == version
                and prev.config_version == cfg.config_version
                and prev.split_name == chosen_split
            ):
                resumed_cache[(prev.rep_index, prev.case_id)] = prev

    run_summaries: list[dict[str, Any]] = []
    first_run_traces: dict[str, dict[str, Any]] = {}
    first_run_records: list[CaseEvalRecord] = []

    telemetry = InvoiceOpsTelemetry(
        config=cfg,
        jsonl_trace_path=results_dir / "traces" / f"{version}_otel_spans.jsonl",
        keep_in_memory=True,
    )

    def _make_agent(client: ResilientErpMcpClient) -> Any:
        if version == "v0":
            return V0BaselineEvaluatorAgent(
                config=cfg,
                store=store,
                embed_query_fn=cached_embed_query,
                erp_client=client,
            )
        if version == "v1":
            return V1MultiAgentEvaluator(
                config=cfg,
                store=store,
                embed_query_fn=cached_embed_query,
                erp_client=client,
                telemetry=telemetry,
            )
        msg = f"Unsupported experiment version '{version}'. Supported: ['v0', 'v1']"
        raise ValueError(msg)

    flaky_clean = FlakyInjector(FlakySettings(enabled=False))
    async with running_local_erp_server(db=erp_db, flaky=flaky_clean) as mcp_url:
        erp_client = ResilientErpMcpClient(
            mcp_url=mcp_url,
            timeout_seconds=cfg.mock_erp.client_timeout_seconds,
            max_retries=cfg.mock_erp.max_retries,
            circuit_breaker=CircuitBreaker(
                failure_threshold=cfg.mock_erp.circuit_breaker_failure_threshold,
                recovery_seconds=cfg.mock_erp.circuit_breaker_recovery_seconds,
            ),
        )
        agent_runner = _make_agent(erp_client)

        for rep in range(1, num_runs + 1):
            print(f"[eval] Starting run {rep}/{num_runs} ...", flush=True)
            rep_records: list[CaseEvalRecord] = []
            for idx, case in enumerate(target_cases, start=1):
                cached_rec = resumed_cache.get((rep, case.case_id))
                if cached_rec is not None:
                    rep_records.append(cached_rec)
                    if rep == 1:
                        fallback_tools = cached_rec.tools_called or [
                            "lookup_vendor",
                            "lookup_purchase_order",
                            "lookup_goods_receipts",
                            "search_policy",
                        ]
                        first_run_traces[case.case_id] = {
                            "tools_called": fallback_tools,
                            "predicted_decision": cached_rec.predicted_decision,
                        }
                        first_run_records.append(cached_rec)
                    continue

                trace = await agent_runner.evaluate_invoice(
                    case_id=case.case_id,
                    vendor_id=case.invoice.vendor_id,
                    pdf_path=repo_root / case.pdf_path,
                )
                all_valid_erp = valid_erp_ids | {
                    h["hold_id"] for h in erp_db.holds_by_idempotency_key.values()
                }
                cited_ids = [ev.id for ev in trace.output.evidence]
                is_correct = trace.output.decision == case.ground_truth_decision
                w_err = compute_weighted_error_cost(
                    ground_truth=case.ground_truth_decision,
                    predicted=trace.output.decision,
                    weights=cfg.evaluation.error_cost_weights,
                )
                cit_ok = validate_citations(
                    evidence=trace.output.evidence,
                    active_clause_ids=active_clause_ids,
                    valid_erp_ids=all_valid_erp,
                )
                rec_k, mrr = compute_retrieval_metrics(
                    retrieved_clause_ids=trace.retrieved_clause_ids,
                    relevant_clause_ids=case.relevant_clause_ids,
                    k=cfg.storage.top_k_clauses,
                    ranked_lists=trace.retrieved_ranked_lists,
                )
                inj_resisted: bool | None = None
                if case.is_prompt_injection:
                    inj_resisted = trace.output.decision != "approve"

                fail_cat = classify_failure_taxonomy(
                    case=case,
                    predicted_decision=trace.output.decision,
                    predicted_reason_codes=trace.output.reason_codes,
                    cited_ids=cited_ids,
                    retrieved_clause_ids=trace.retrieved_clause_ids,
                    tools_called=trace.tools_called,
                    degraded_due_to_erp_error=trace.degraded_due_to_erp_error,
                )

                record = CaseEvalRecord(
                    run_id=batch_run_id,
                    experiment_version=version,
                    config_version=cfg.config_version,
                    git_sha=git_sha,
                    split_name=chosen_split,
                    rep_index=rep,
                    case_id=case.case_id,
                    failure_mode=case.failure_mode,
                    is_prompt_injection=case.is_prompt_injection,
                    ground_truth_decision=case.ground_truth_decision,
                    predicted_decision=trace.output.decision,
                    predicted_reason_codes=trace.output.reason_codes,
                    cited_ids=cited_ids,
                    retrieved_clause_ids=trace.retrieved_clause_ids,
                    tools_called=trace.tools_called,
                    is_correct=is_correct,
                    weighted_error_cost=w_err,
                    citation_valid=cit_ok,
                    retrieval_recall_at_k=rec_k,
                    retrieval_mrr=mrr,
                    injection_resisted=inj_resisted,
                    failure_category=fail_cat,
                    latency_ms=trace.latency_ms,
                    input_tokens=trace.input_tokens,
                    output_tokens=trace.output_tokens,
                    estimated_cost_usd=trace.estimated_cost_usd,
                    llm_calls=trace.llm_calls,
                    tool_calls=trace.tool_calls,
                    erp_retries=trace.erp_retries,
                    resolved_model=trace.resolved_model,
                    explanation=trace.explanation,
                    timestamp_utc=datetime.now(UTC).isoformat(),
                )
                store.record_eval_case(record)
                rep_records.append(record)

                if rep == 1:
                    first_run_traces[case.case_id] = {
                        "tools_called": trace.tools_called,
                        "predicted_decision": trace.output.decision,
                    }
                    first_run_records.append(record)

                if idx % 4 == 0 or idx == len(target_cases):
                    acc_so_far = sum(1 for r in rep_records if r.is_correct) / len(rep_records)
                    print(
                        f"  [run {rep}] {idx}/{len(target_cases)} cases | "
                        f"running acc={acc_so_far:.1%}",
                        flush=True,
                    )

                if cfg.evaluation.inter_case_delay_seconds > 0:
                    await asyncio.sleep(cfg.evaluation.inter_case_delay_seconds)

            run_summaries.append(summarize_single_run(rep_records))

    multi_run_summary = summarize_multi_run(run_summaries)

    # Layer 2: ADK Golden-12 Trajectory Evaluation
    golden_ids = [cid for cid in split_data["golden_12_ids"] if cid in first_run_traces]
    if not golden_ids:
        golden_ids = list(first_run_traces.keys())[:12]
    golden_cases = [all_cases[cid] for cid in golden_ids]
    adk_traj_summary = export_and_score_adk_golden_subset(
        golden_cases=golden_cases,
        actual_traces_by_case_id=first_run_traces,
        output_evalset_path=results_dir / f"adk_{version}_golden_12.evalset.json",
    )
    adk_traj_summary["evalset_file"] = f"results/adk_{version}_golden_12.evalset.json"

    existing_summary: dict[str, Any] = {}
    summary_file = results_dir / f"{version}_summary.json"
    if summary_file.exists():
        loaded_prev = json.loads(summary_file.read_text(encoding="utf-8"))
        if loaded_prev.get("config_version") == cfg.config_version:
            existing_summary = loaded_prev

    # Layer 4b: Flaky-ERP Robustness Sub-Benchmark on 4 representative golden cases
    if "flaky_erp_robustness_golden_12" in existing_summary:
        flaky_summary = existing_summary["flaky_erp_robustness_golden_12"]
        print(
            "[eval] Reusing cached flaky-ERP robustness benchmark for this config_version.",
            flush=True,
        )
    else:
        flaky_sample = golden_cases[:4]
        print(
            f"[eval] Running flaky-ERP robustness check on {len(flaky_sample)} cases ...",
            flush=True,
        )
        flaky_injector = FlakyInjector(
            FlakySettings(
                enabled=True,
                timeout_rate=cfg.mock_erp.flaky_timeout_rate,
                rate_limit_429_rate=cfg.mock_erp.flaky_rate_limit_429_rate,
                stale_read_rate=cfg.mock_erp.flaky_stale_read_rate,
                timeout_sleep_seconds=1.6,
                seed=cfg.evaluation.random_seed,
            )
        )
        flaky_completed = 0
        flaky_correct = 0
        flaky_total_retries = 0
        flaky_false_approvals = 0
        async with running_local_erp_server(db=erp_db, flaky=flaky_injector) as flaky_url:
            flaky_client = ResilientErpMcpClient(
                mcp_url=flaky_url,
                timeout_seconds=1.2,
                max_retries=cfg.mock_erp.max_retries,
                base_backoff_seconds=0.1,
                circuit_breaker=CircuitBreaker(
                    failure_threshold=cfg.mock_erp.circuit_breaker_failure_threshold,
                    recovery_seconds=cfg.mock_erp.circuit_breaker_recovery_seconds,
                ),
            )
            flaky_agent = _make_agent(flaky_client)
            for g_case in flaky_sample:
                f_trace = await flaky_agent.evaluate_invoice(
                    case_id=g_case.case_id,
                    vendor_id=g_case.invoice.vendor_id,
                    pdf_path=repo_root / g_case.pdf_path,
                )
                flaky_total_retries += f_trace.erp_retries
                if not f_trace.degraded_due_to_erp_error:
                    flaky_completed += 1
                if f_trace.output.decision == g_case.ground_truth_decision:
                    flaky_correct += 1
                elif f_trace.output.decision == "approve":
                    flaky_false_approvals += 1
                if cfg.evaluation.inter_case_delay_seconds > 0:
                    await asyncio.sleep(cfg.evaluation.inter_case_delay_seconds)

        flaky_summary = {
            "n_cases": len(flaky_sample),
            "task_completion_rate": round(flaky_completed / max(1, len(flaky_sample)), 4),
            "decision_accuracy_under_flaky_erp": round(
                flaky_correct / max(1, len(flaky_sample)), 4
            ),
            "unsafe_false_approvals_under_outage": flaky_false_approvals,
            "total_client_retries_triggered": flaky_total_retries,
            "mean_retries_per_case": round(flaky_total_retries / max(1, len(flaky_sample)), 2),
            "server_fault_stats": flaky_injector.stats(),
        }

    # Layer 5: Calibrated LLM-as-Judge
    if "llm_judge" in existing_summary:
        judge_summary = existing_summary["llm_judge"]
        print(
            "[eval] Reusing cached 40-item LLM-judge calibration for this config_version.",
            flush=True,
        )
    else:
        print(
            "[eval] Running LLM-as-judge calibration (40 rubric items) + explanation scoring ...",
            flush=True,
        )
        judge = ExplanationJudge(config=cfg, dotenv_path=dotenv_path)
        calibration_report = judge.calibrate(cal_path)
        explanation_pass_rate: float | None = None
        if calibration_report["reliable_for_reporting"] and first_run_records:
            sample_inputs = [
                {
                    "id": r.case_id,
                    "decision": r.predicted_decision,
                    "candidate_explanation": r.explanation,
                }
                for r in first_run_records
            ]
            scored = judge.score_batch(sample_inputs)
            explanation_pass_rate = round(sum(scored.values()) / max(1, len(first_run_records)), 4)

        judge_summary = {
            **calibration_report,
            "agent_explanation_pass_rate": explanation_pass_rate,
        }

    final_artifact = {
        "run_id": batch_run_id,
        "experiment_version": version,
        "config_version": cfg.config_version,
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "split_name": chosen_split,
        "primary_model": cfg.models.v0_baseline,
        "embedding_model": cfg.models.embedding,
        "multi_run": multi_run_summary,
        "adk_trajectory_golden_12": adk_traj_summary,
        "flaky_erp_robustness_golden_12": flaky_summary,
        "llm_judge": judge_summary,
    }

    if telemetry.jsonl_exporter.spans:
        first_trace_id = telemetry.jsonl_exporter.spans[0]["trace_id"]
        sample_spans = [
            s for s in telemetry.jsonl_exporter.spans if s["trace_id"] == first_trace_id
        ]
        sample_path = results_dir / f"{version}_trace_sample.json"
        sample_path.write_text(json.dumps(sample_spans, indent=2), encoding="utf-8")

    summary_file.write_text(json.dumps(final_artifact, indent=2), encoding="utf-8")
    report_html = generate_html_report(
        results_dir=results_dir, output_html=results_dir / "report.html"
    )
    print(f"[eval] Saved summary -> {summary_file.relative_to(repo_root)}", flush=True)
    print(f"[eval] Updated HTML report -> {report_html.relative_to(repo_root)}", flush=True)
    return final_artifact
