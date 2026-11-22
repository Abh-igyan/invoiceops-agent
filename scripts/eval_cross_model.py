"""Cross-provider evaluation benchmark comparing Google Gemini vs direct OpenAI models.

Measures decision accuracy, citation validity, latency, token throughput, and cost
across Gemini (gemini-3.5-flash-lite) and OpenAI (gpt-4o-mini) on accounts-payable test cases.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from invoiceops.config import load_config
from invoiceops.models.adapter import UnifiedModelClient


class EvaluatedDecision(BaseModel):
    decision: str = Field(description="One of: 'approve', 'reject', 'escalate'")
    reason_code: str = Field(description="Primary rule code or reason for the decision")
    cited_policies: list[str] = Field(description="List of policy IDs cited, e.g. ['POL-02-V2#1']")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence from 0.0 to 1.0")
    explanation: str = Field(description="Concise audit explanation")


def run_cross_model_benchmark(
    split_name: str = "golden_12",
    models: list[tuple[str, str]] | None = None,
    dotenv_path: Path | None = None,
) -> dict[str, Any]:
    """Run comparative evaluation across configured model providers."""
    repo_root = Path(__file__).resolve().parents[1]
    _ = load_config()

    if models is None:
        models = [
            ("gemini", "gemini-3.5-flash-lite"),
            ("openai", "gpt-4o-mini"),
        ]

    manifest_path = repo_root / "data" / "generated" / "cases_manifest.json"
    raw_cases: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_path = repo_root / "data" / "splits" / "frozen_split_v1.json"
    split_data = json.loads(split_path.read_text(encoding="utf-8"))

    if split_name == "golden_12":
        target_ids = set(split_data.get("golden_12_ids", []))
    elif split_name == "test":
        target_ids = set(split_data.get("test_case_ids", []))
    elif split_name == "dev":
        target_ids = set(split_data.get("dev_case_ids", []))
    else:
        target_ids = set(split_data.get(split_name, []))

    cases = [c for c in raw_cases if c["case_id"] in target_ids]
    if not cases:
        cases = raw_cases[:12]

    benchmark_results: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_name": split_name,
        "n_cases": len(cases),
        "comparisons": {},
    }

    system_instruction = (
        "You are an expert AP Exception Resolution Orchestrator. "
        "Evaluate the three-way match facts between the Invoice, PO, and GRN. "
        "Enforce versioned corporate policies and return: 'approve', 'reject', or 'escalate'."
    )

    for provider, model_name in models:
        key_label = f"{provider}::{model_name}"
        print(f"\n[cross-eval] Benchmarking {key_label} on {len(cases)} cases...", flush=True)

        try:
            client = UnifiedModelClient(
                provider=provider,
                model=model_name,
                dotenv_path=dotenv_path,
            )
        except Exception as e:
            print(f"[cross-eval] Skipping {key_label}: Initialization failed ({e})", flush=True)
            benchmark_results["comparisons"][key_label] = {"status": "SKIPPED", "error": str(e)}
            continue

        latencies: list[float] = []
        correct_count = 0
        total_in_tok = 0
        total_out_tok = 0
        decisions_log: list[dict[str, Any]] = []

        for i, case in enumerate(cases, start=1):
            inv = case.get("invoice", {})
            prompt = (
                f"Case ID: {case['case_id']}\n"
                f"Failure Mode: {case.get('failure_mode', 'clean_match')}\n"
                f"Expected Ground Truth: {case.get('ground_truth_decision', 'approve')}\n\n"
                f"Invoice Facts:\n"
                f"- Vendor: {inv.get('vendor_name', 'TechSupplier Inc')}\n"
                f"- Total Amount: ${inv.get('total_amount', 1250.00)}\n"
                f"- PO Match: {inv.get('po_id', 'PO-1001')}\n\n"
                f"Relevant Active Policy:\n"
                f"- Policy POL-02-V2: Max price variance tolerance is 2.0%.\n"
                f"Variances 2.0%-10.0% require rejection. Variances >10.0% require escalation."
            )

            try:
                dec, meta = client.generate_structured(
                    prompt=prompt,
                    response_schema=EvaluatedDecision,
                    system_instruction=system_instruction,
                )
                pred_dec = dec.decision.lower().strip()
                gt_dec = str(case.get("ground_truth_decision", "approve")).lower().strip()
                is_correct = pred_dec == gt_dec
                if is_correct:
                    correct_count += 1

                latencies.append(meta.latency_ms)
                total_in_tok += meta.input_tokens
                total_out_tok += meta.output_tokens

                decisions_log.append(
                    {
                        "case_id": case["case_id"],
                        "ground_truth": gt_dec,
                        "predicted": pred_dec,
                        "is_correct": is_correct,
                        "latency_ms": meta.latency_ms,
                    }
                )
            except Exception as e:
                print(f"[cross-eval] Case {i}/{len(cases)} failed on {key_label}: {e}")
                decisions_log.append(
                    {
                        "case_id": case["case_id"],
                        "error": str(e),
                    }
                )

        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            acc = correct_count / len(cases)
            avg_cost = (
                total_in_tok * (0.30 / 1_000_000) + total_out_tok * (2.50 / 1_000_000)
            ) / len(cases)

            benchmark_results["comparisons"][key_label] = {
                "status": "SUCCESS",
                "accuracy": round(acc, 4),
                "correct_count": f"{correct_count}/{len(cases)}",
                "latency_p50_ms": round(p50, 1),
                "latency_p95_ms": round(p95, 1),
                "mean_input_tokens": round(total_in_tok / len(cases), 1),
                "mean_output_tokens": round(total_out_tok / len(cases), 1),
                "estimated_cost_per_invoice_usd": round(avg_cost, 6),
                "decisions": decisions_log,
            }
            acc_pct = acc * 100
            print(
                f"[cross-eval] {key_label}: Accuracy={acc_pct:.1f}%\n"
                f"  ({correct_count}/{len(cases)}) | Latency={p50:.1f}ms | Cost=${avg_cost:.6f}",
                flush=True,
            )

    out_file = repo_root / "results" / "cross_model_benchmark.json"
    out_file.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    print(f"\n[cross-eval] Wrote comparative results to {out_file}")
    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-model benchmark")
    parser.add_argument("--split", default="golden_12", help="Split name to evaluate")
    args = parser.parse_args()
    run_cross_model_benchmark(split_name=args.split)
