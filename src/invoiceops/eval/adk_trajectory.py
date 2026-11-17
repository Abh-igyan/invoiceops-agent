"""Layer 2 Trajectory Evaluation using Google ADK `EvalSet` and `TrajectoryEvaluator`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from google.adk.evaluation.eval_case import EvalCase, IntermediateData, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, ToolTrajectoryCriterion
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.genai import types as genai_types

from invoiceops.schemas import SyntheticCase


def expected_tools_for_case(case: SyntheticCase) -> list[str]:
    """Return canonical minimal verification tool sequence for a golden invoice case."""
    tools = ["lookup_vendor"]
    if case.invoice.po_id is not None:
        tools.append("lookup_purchase_order")
        if case.failure_mode in {
            "clean_match",
            "price_variance_within_tolerance",
            "quantity_mismatch",
            "late_invoice_vs_terms",
            "ambiguous_escalation",
            "stale_policy_version",
        }:
            tools.append("lookup_goods_receipts")
    if case.failure_mode == "duplicate_invoice":
        tools.append("check_invoice_history")
    tools.append("search_policy")
    if case.ground_truth_decision == "hold":
        tools.append("place_payment_hold")
    return tools


def export_and_score_adk_golden_subset(
    golden_cases: list[SyntheticCase],
    actual_traces_by_case_id: dict[str, dict[str, Any]],
    output_evalset_path: Path,
) -> dict[str, Any]:
    """Write a valid ADK `.evalset.json` and score `tool_trajectory_avg_score` + response match.

    Demonstrates both ADK's brittle default `EXACT` match (which fails when optional extra
    checks or varied free-text search queries occur) and `ANY_ORDER` (`ignore_args=True`),
    which measures whether required verification tools were invoked.
    """
    expected_invocations: list[Invocation] = []
    actual_invocations: list[Invocation] = []
    eval_cases: list[EvalCase] = []
    decision_matches = 0

    for case in golden_cases:
        exp_tool_names = expected_tools_for_case(case)
        exp_calls = [genai_types.FunctionCall(name=name, args={}) for name in exp_tool_names]
        exp_text = case.ground_truth_decision

        exp_inv = Invocation(
            invocation_id=f"exp-{case.case_id}",
            user_content=genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_text(
                        text=f"Resolve {case.case_id} for {case.invoice.vendor_id}"
                    )
                ],
            ),
            final_response=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text=exp_text)],
            ),
            intermediate_data=IntermediateData(tool_uses=exp_calls),
        )
        expected_invocations.append(exp_inv)
        eval_cases.append(EvalCase(eval_id=case.case_id, conversation=[exp_inv]))

        trace = actual_traces_by_case_id.get(case.case_id, {})
        act_tool_names: list[str] = trace.get("tools_called", [])
        act_decision: str = trace.get("predicted_decision", "escalate_to_human")
        if act_decision == exp_text:
            decision_matches += 1

        act_calls = [genai_types.FunctionCall(name=name, args={}) for name in act_tool_names]
        act_inv = Invocation(
            invocation_id=f"act-{case.case_id}",
            user_content=exp_inv.user_content,
            final_response=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text=act_decision)],
            ),
            intermediate_data=IntermediateData(tool_uses=act_calls),
        )
        actual_invocations.append(act_inv)

    eval_set = EvalSet(
        eval_set_id="invoiceops_golden_12",
        name="InvoiceOps Golden 12 Trajectory Set",
        description="12-case stratified golden subset (1 per failure mode) for ADK trajectory eval",
        eval_cases=eval_cases,
    )
    output_evalset_path.parent.mkdir(parents=True, exist_ok=True)
    output_evalset_path.write_text(eval_set.model_dump_json(indent=2), encoding="utf-8")

    # Validate round-trip deserialization against ADK's official EvalSet model
    EvalSet.model_validate_json(output_evalset_path.read_text(encoding="utf-8"))

    any_order_metric = EvalMetric(
        metric_name="tool_trajectory_avg_score",
        threshold=0.75,
        criterion=ToolTrajectoryCriterion(
            threshold=0.75,
            match_type=ToolTrajectoryCriterion.MatchType.ANY_ORDER,
            ignore_args=True,
        ),
    )
    exact_metric = EvalMetric(
        metric_name="tool_trajectory_avg_score",
        threshold=0.75,
        criterion=ToolTrajectoryCriterion(
            threshold=0.75,
            match_type=ToolTrajectoryCriterion.MatchType.EXACT,
            ignore_args=True,
        ),
    )

    any_order_eval = TrajectoryEvaluator(eval_metric=any_order_metric)
    exact_eval = TrajectoryEvaluator(eval_metric=exact_metric)

    any_res = any_order_eval.evaluate_invocations(
        actual_invocations=actual_invocations,
        expected_invocations=expected_invocations,
    )
    exact_res = exact_eval.evaluate_invocations(
        actual_invocations=actual_invocations,
        expected_invocations=expected_invocations,
    )

    n = max(1, len(golden_cases))
    return {
        "evalset_file": output_evalset_path.as_posix(),
        "golden_cases": len(golden_cases),
        "tool_trajectory_avg_score_any_order": round(float(any_res.overall_score or 0.0), 4),
        "tool_trajectory_avg_score_exact": round(float(exact_res.overall_score or 0.0), 4),
        "response_match_score": round(decision_matches / n, 4),
    }
