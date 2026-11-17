"""Static HTML report generator across experiment versions (`results/report.html`)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _fmt_metric(
    metrics: dict[str, Any],
    key: str,
    pct: bool = False,
    digits: int = 3,
) -> str:
    entry = metrics.get(key)
    if not entry:
        return "—"
    mean = float(entry["mean"])
    ci = float(entry["ci95_half_width"])
    if pct:
        return f"{mean * 100:.1f}% ± {ci * 100:.1f}%"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def generate_html_report(results_dir: Path, output_html: Path) -> Path:
    """Read `results/*_summary.json` files and render a self-contained static HTML dashboard."""
    summaries: list[dict[str, Any]] = []
    for json_path in sorted(results_dir.glob("*_summary.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        summaries.append(data)

    rows_html: list[str] = []
    taxonomy_rows: list[str] = []

    for item in summaries:
        ver = html.escape(str(item.get("experiment_version", "unknown")))
        split = html.escape(str(item.get("split_name", "")))
        m = item.get("multi_run", {}).get("metrics", {})

        rows_html.append(
            f"<tr>"
            f"<td><strong>{ver}</strong></td>"
            f"<td><code>{split}</code></td>"
            f"<td>{_fmt_metric(m, 'decision_accuracy', pct=True)}</td>"
            f"<td>{_fmt_metric(m, 'cost_weighted_error', digits=3)}</td>"
            f"<td>{_fmt_metric(m, 'citation_validity_rate', pct=True)}</td>"
            f"<td>{_fmt_metric(m, 'retrieval_recall_at_k', pct=True)}</td>"
            f"<td>{_fmt_metric(m, 'injection_resistance_rate', pct=True)}</td>"
            f"<td>{_fmt_metric(m, 'latency_p50_ms', digits=0)} / "
            f"{_fmt_metric(m, 'latency_p95_ms', digits=0)}</td>"
            f"<td>${_fmt_metric(m, 'mean_cost_per_case_usd', digits=5)}</td>"
            f"</tr>"
        )

        tax = item.get("multi_run", {}).get("mean_failure_taxonomy_counts", {})
        tax_str = ", ".join(f"{html.escape(k)}: {v}" for k, v in sorted(tax.items())) or "None"
        taxonomy_rows.append(f"<tr><td><strong>{ver}</strong></td><td>{tax_str}</td></tr>")

    table_body = "\n".join(rows_html) or "<tr><td colspan='9'>No summary files found.</td></tr>"
    tax_body = "\n".join(taxonomy_rows) or "<tr><td colspan='2'>No taxonomy data.</td></tr>"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>invoiceops-agent — Evaluation Report</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      margin: 2rem auto; max-width: 1180px; color: #111827; line-height: 1.5; padding: 0 1rem;
    }}
    h1, h2 {{ color: #1e3a8a; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.2rem 0 2rem; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.6rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f9fafb; }}
    code {{ background: #e5e7eb; padding: 0.1rem 0.35rem; border-radius: 4px; }}
    .meta {{ color: #4b5563; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>invoiceops-agent — Controlled Experiment Report</h1>
  <p class="meta">
    Generated from raw JSON in <code>results/</code> (mean ± 95% t-interval, temperature=0.0).
  </p>

  <h2>1. Outcome, Retrieval, Safety, Latency &amp; Shadow Production Cost</h2>
  <table>
    <thead>
      <tr>
        <th>Version</th>
        <th>Split</th>
        <th>Accuracy</th>
        <th>Cost-Weighted Error</th>
        <th>Citation Validity</th>
        <th>Retrieval Recall@5</th>
        <th>Injection Resistance</th>
        <th>Latency p50 / p95 (ms)</th>
        <th>Shadow Cost / Case (USD)</th>
      </tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>

  <h2>2. Mean Failure Taxonomy Breakdown (Failed Cases per Run)</h2>
  <table>
    <thead>
      <tr>
        <th>Version</th>
        <th>Failure Categories (mean count per run)</th>
      </tr>
    </thead>
    <tbody>
      {tax_body}
    </tbody>
  </table>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(doc, encoding="utf-8")
    return output_html
