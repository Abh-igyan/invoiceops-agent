"""Cross-platform task dispatcher mirroring Makefile targets (Linux/macOS/Windows)."""

from __future__ import annotations

import argparse
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
        choices=["lint", "format", "test", "check", "data", "eval", "deploy", "destroy"],
        help="Workflow target to execute",
    )
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

    phase_map = {
        "data": "Phase P1 (synthetic invoice/PO/GRN + versioned policy dataset generator)",
        "eval": "Phase P1 (frozen-split evaluation harness + cost/latency/taxonomy logger)",
        "deploy": "Phase P4 (Terraform + Cloud Run two-service deployment)",
        "destroy": "Phase P4 (Terraform teardown of all billable Google Cloud resources)",
    }
    print(f"[P0 stub] `{args.target}` is scheduled for {phase_map[args.target]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
