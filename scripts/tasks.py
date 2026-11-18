"""Cross-platform task dispatcher mirroring Makefile targets (Linux/macOS/Windows)."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    """Execute a subprocess command and return its exit code."""
    print(f"+ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Parse CLI subcommand and run the requested project workflow."""
    parser = argparse.ArgumentParser(description="invoiceops-agent task runner")
    parser.add_argument(
        "target",
        choices=[
            "lint",
            "format",
            "test",
            "check",
            "data",
            "eval",
            "traces",
            "erp",
            "deploy",
            "destroy",
        ],
        help="Workflow target to execute",
    )
    parser.add_argument("--version", default="v0", help="Agent version to evaluate (default: v0)")
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split: dev_stratified_36 | dev | test | golden_12",
    )
    parser.add_argument("--runs", type=int, default=None, help="Number of repetitions (default: 3)")
    args = parser.parse_args(argv)

    py = sys.executable
    if args.target == "lint":
        return _run([py, "-m", "ruff", "check", "."]) or _run(
            [py, "-m", "ruff", "format", "--check", "."]
        )
    if args.target == "format":
        return _run([py, "-m", "ruff", "format", "."]) or _run(
            [py, "-m", "ruff", "check", "--fix", "."]
        )
    if args.target == "test":
        return _run([py, "-m", "pytest"])
    if args.target == "check":
        return main(["lint"]) or main(["test"])

    if args.target == "data":
        from invoiceops.data_gen.generator import generate_dataset

        summary = generate_dataset()
        print(json.dumps(summary, indent=2))
        return 0

    if args.target == "eval":
        from invoiceops.eval.harness import run_evaluation

        result = asyncio.run(
            run_evaluation(
                version=args.version,
                split_name=args.split,
                runs=args.runs,
            )
        )
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "version": result["experiment_version"],
                    "split": result["split_name"],
                    "metrics": result["multi_run"]["metrics"],
                    "adk_trajectory": result["adk_trajectory_golden_12"],
                    "flaky_erp": result["flaky_erp_robustness_golden_12"],
                    "llm_judge": result["llm_judge"],
                },
                indent=2,
            )
        )
        return 0

    if args.target == "traces":
        from invoiceops.data_gen.generator import REPO_ROOT

        sample_file = REPO_ROOT / "results" / f"{args.version}_trace_sample.json"
        if not sample_file.exists():
            print(f"No trace sample found at {sample_file}. Run `eval --version {args.version}`.")
            return 1
        spans = json.loads(sample_file.read_text(encoding="utf-8"))
        for sp in sorted(spans, key=lambda s: s["start_time_ns"]):
            indent = "  " if sp["parent_span_id"] else ""
            print(
                f"{indent}[{sp['name']}] status={sp['status']} "
                f"dur={sp['duration_ms']:.1f}ms attrs={sp['attributes']}"
            )
        return 0

    if args.target == "erp":
        import uvicorn

        from invoiceops.config import load_config
        from invoiceops.data_gen.generator import REPO_ROOT
        from invoiceops.mcp_erp.flaky import FlakyInjector, FlakySettings
        from invoiceops.mcp_erp.server import ErpDatabase, create_erp_app

        cfg = load_config()
        db = ErpDatabase.from_seed_file(REPO_ROOT / "data" / "generated" / "erp_seed.json")
        flaky = FlakyInjector(FlakySettings(enabled=cfg.mock_erp.flaky_mode))
        uvicorn.run(create_erp_app(db=db, flaky=flaky), host="127.0.0.1", port=8001)
        return 0

    phase_map = {
        "deploy": "Phase P4 (Terraform + Cloud Run two-service deployment)",
        "destroy": "Phase P4 (Terraform teardown of all billable Google Cloud resources)",
    }
    print(f"[P1 stub] `{args.target}` is scheduled for {phase_map[args.target]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
