"""
io.py
=====

Persistence and report export helpers for :class:`BacktestReport`.

Exposes
-------
* ``report_to_dict`` / ``report_from_dict`` — round-trippable JSON-friendly data.
* ``save_json`` / ``load_json`` — disk persistence.
* ``save_csv`` — one equity-curve CSV per portfolio plus ``summary.csv`` with
  a metrics matrix.
* ``save_html`` — self-contained HTML report (charts embedded as base64 PNG).

Design notes
------------
* No template engine or extra dependency is required.
* JSON uses ``None`` in place of ``NaN`` / ``Inf`` because JSON has no native
  representation for those values. The loader restores them to ``float('nan')``.
* Export functions receive a report and write files; they do not mutate engine
  state.
  This keeps :class:`BacktestReport` itself a thin dataclass.
"""

from __future__ import annotations

import base64
import dataclasses
import io as _stdio
import json
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from app.backtesting.backtest_engine import (
    BacktestMetrics,
    BacktestReport,
    BacktestResult,
    PortfolioSpec,
)

SCHEMA_VERSION = 1


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers for JSON-safe numeric values
# ═══════════════════════════════════════════════════════════════════════════

def _clean(x: Any) -> Any:
    """Recursively replace NaN and Inf with None for JSON output."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _restore_float(v: Any) -> float:
    return float("nan") if v is None else float(v)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataclass / dict conversion
# ═══════════════════════════════════════════════════════════════════════════

def _spec_to_dict(spec: PortfolioSpec) -> Dict[str, Any]:
    return {"name": spec.name, "weights": dict(spec.weights)}


def _spec_from_dict(d: Dict[str, Any]) -> PortfolioSpec:
    return PortfolioSpec(name=d["name"], weights={k: float(v) for k, v in d["weights"].items()})


def _metrics_to_dict(m: BacktestMetrics) -> Dict[str, Any]:
    return {f.name: getattr(m, f.name) for f in dataclasses.fields(m)}


def _metrics_from_dict(d: Dict[str, Any]) -> BacktestMetrics:
    field_names = {f.name for f in dataclasses.fields(BacktestMetrics)}
    kwargs: Dict[str, Any] = {}
    for name in field_names:
        if name not in d:
            continue
        if name == "avg_n_holdings":
            kwargs[name] = int(d[name]) if d[name] is not None else 0
        else:
            kwargs[name] = _restore_float(d[name])
    return BacktestMetrics(**kwargs)


def _series_to_dict(s: pd.Series) -> Dict[str, List[Any]]:
    idx = [t.isoformat() if isinstance(t, pd.Timestamp) else str(t) for t in s.index]
    vals = [None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else float(v)
            for v in s.values]
    return {"index": idx, "values": vals}


def _series_from_dict(d: Dict[str, List[Any]]) -> pd.Series:
    idx = pd.to_datetime(d["index"])
    vals = [_restore_float(v) for v in d["values"]]
    return pd.Series(vals, index=idx, name="portfolio_value")


def _result_to_dict(r: BacktestResult) -> Dict[str, Any]:
    return {
        "spec": _spec_to_dict(r.spec),
        "metrics": _metrics_to_dict(r.metrics),
        "portfolio_values": _series_to_dict(r.portfolio_values),
        # Store the full benchmark list while keeping the legacy singular
        # benchmark field readable by older payloads.
        "benchmark": _result_to_dict(r.benchmark) if r.benchmark is not None else None,
        "benchmarks": [_result_to_dict(b) for b in r.benchmarks] if r.benchmarks else [],
    }


def _result_from_dict(d: Dict[str, Any]) -> BacktestResult:
    benchmarks_raw = d.get("benchmarks")
    if benchmarks_raw:
        benchmarks = [_result_from_dict(b) for b in benchmarks_raw]
        primary = benchmarks[0]
    elif d.get("benchmark"):
        # Older payloads stored only the singular benchmark field.
        primary = _result_from_dict(d["benchmark"])
        benchmarks = [primary]
    else:
        primary = None
        benchmarks = []
    return BacktestResult(
        spec=_spec_from_dict(d["spec"]),
        metrics=_metrics_from_dict(d["metrics"]),
        portfolio_values=_series_from_dict(d["portfolio_values"]),
        benchmark=primary,
        benchmarks=benchmarks,
    )


def report_to_dict(report: BacktestReport) -> Dict[str, Any]:
    """Convert a :class:`BacktestReport` into JSON-friendly data."""
    return _clean({
        "schema_version": SCHEMA_VERSION,
        "start_date": report.start_date,
        "end_date": report.end_date,
        "initial_capital": float(report.initial_capital),
        "results": [_result_to_dict(r) for r in report.results],
        "benchmark": _result_to_dict(report.benchmark) if report.benchmark else None,
    })


def report_from_dict(d: Dict[str, Any]) -> BacktestReport:
    """Reconstruct a full report from :func:`report_to_dict` output."""
    sv = d.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={sv}; this build expects {SCHEMA_VERSION}"
        )
    return BacktestReport(
        results=[_result_from_dict(r) for r in d["results"]],
        benchmark=_result_from_dict(d["benchmark"]) if d.get("benchmark") else None,
        start_date=str(d["start_date"]),
        end_date=str(d["end_date"]),
        initial_capital=float(d["initial_capital"]),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  JSON persistence
# ═══════════════════════════════════════════════════════════════════════════

def save_json(report: BacktestReport, path: str) -> None:
    """Write ``report`` to ``path`` as indented UTF-8 JSON."""
    payload = report_to_dict(report)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_json(path: str) -> BacktestReport:
    """Load a report previously written by :func:`save_json`."""
    with open(path, "r", encoding="utf-8") as fh:
        return report_from_dict(json.load(fh))


# ═══════════════════════════════════════════════════════════════════════════
#  CSV export
# ═══════════════════════════════════════════════════════════════════════════

_METRIC_ORDER = [
    "total_return", "cagr", "annualised_volatility", "max_drawdown",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "information_ratio", "tracking_error",
    "var_95", "cvar_95", "ulcer_index", "downside_deviation",
    "best_period_return", "worst_period_return", "win_rate",
    "turnover", "avg_n_holdings",
    "start_value", "end_value",
]


def _safe_filename(s: str) -> str:
    out = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)
    return out.strip("_") or "portfolio"


def save_csv(report: BacktestReport, directory: str) -> Dict[str, str]:
    """Write portfolio equity CSVs and a metrics summary CSV.

    Returns a mapping ``{filename: full_path}`` of every file written.
    """
    os.makedirs(directory, exist_ok=True)
    written: Dict[str, str] = {}

    # Write each strategy curve with its primary benchmark beside it.
    for r in report.results:
        cols = {r.spec.name: r.portfolio_values}
        if r.benchmark is not None:
            cols[r.benchmark.spec.name] = r.benchmark.portfolio_values
        df = pd.DataFrame(cols)
        fname = f"equity_{_safe_filename(r.spec.name)}.csv"
        full = os.path.join(directory, fname)
        df.to_csv(full, index_label="date")
        written[fname] = full

    # Write one summary table for strategies and primary benchmarks.
    rows: List[Dict[str, Any]] = []
    for r in report.results:
        row = {"portfolio": r.spec.name, "kind": "strategy"}
        for k in _METRIC_ORDER:
            row[k] = getattr(r.metrics, k)
        rows.append(row)
        if r.benchmark is not None:
            brow = {"portfolio": r.benchmark.spec.name, "kind": "benchmark"}
            for k in _METRIC_ORDER:
                brow[k] = getattr(r.benchmark.metrics, k)
            rows.append(brow)
    summary_path = os.path.join(directory, "summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    written["summary.csv"] = summary_path

    return written


# ═══════════════════════════════════════════════════════════════════════════
#  HTML report with embedded PNG charts
# ═══════════════════════════════════════════════════════════════════════════

def _png_b64(fig) -> str:
    buf = _stdio.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _equity_chart_b64(report: BacktestReport):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r in report.results:
        ax.plot(r.portfolio_values.index, r.portfolio_values.values, label=r.spec.name, linewidth=1.5)
    # Plot one benchmark only so the equity chart stays readable.
    if report.results and report.results[0].benchmark is not None:
        b = report.results[0].benchmark
        ax.plot(b.portfolio_values.index, b.portfolio_values.values,
                label=b.spec.name, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Portfolio value over time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out = _png_b64(fig)
    plt.close(fig)
    return out


def _drawdown_chart_b64(report: BacktestReport):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for r in report.results:
        v = r.portfolio_values
        dd = (v / v.cummax() - 1.0) * 100.0
        ax.plot(dd.index, dd.values, label=r.spec.name, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown, %")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out = _png_b64(fig)
    plt.close(fig)
    return out


def _metrics_bar_b64(report: BacktestReport, metric: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.spec.name for r in report.results]
    values = [getattr(r.metrics, metric) for r in report.results]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.bar(names, values)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.autofmt_xdate(rotation=20)
    out = _png_b64(fig)
    plt.close(fig)
    return out


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if isinstance(v, float):
        return f"{v * 100:.2f} %" if pct else f"{v:.4f}"
    return str(v)


_PCT_METRICS = {
    "total_return", "cagr", "annualised_volatility", "max_drawdown",
    "var_95", "cvar_95", "downside_deviation",
    "best_period_return", "worst_period_return", "win_rate", "turnover",
}


def _metrics_table_html(report: BacktestReport) -> str:
    headers = ["Portfolio"] + _METRIC_ORDER
    th = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = []
    for r in report.results:
        cells = [f"<td><b>{r.spec.name}</b></td>"]
        for k in _METRIC_ORDER:
            cells.append(f"<td>{_fmt(getattr(r.metrics, k), pct=(k in _PCT_METRICS))}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
        if r.benchmark is not None:
            bcells = [f'<td style="color:#888"><i>{r.benchmark.spec.name}</i></td>']
            for k in _METRIC_ORDER:
                bcells.append(
                    f'<td style="color:#888">{_fmt(getattr(r.benchmark.metrics, k), pct=(k in _PCT_METRICS))}</td>'
                )
            rows_html.append("<tr>" + "".join(bcells) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Report — {start} → {end}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
          margin: 24px; color: #222; max-width: 1200px; }}
  h1 {{ color: #1ABC9C; margin-bottom: 4px; }}
  .meta {{ color: #555; font-size: 13px; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; font-size: 12px; margin: 12px 0 24px; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; white-space: nowrap; }}
  th {{ background: #1C2833; color: white; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  img {{ max-width: 100%; margin: 8px 0 24px; border: 1px solid #eee; border-radius: 4px; }}
  h2 {{ color: #1C2833; border-bottom: 2px solid #1ABC9C; padding-bottom: 4px; margin-top: 32px; }}
  .footer {{ color: #999; font-size: 11px; margin-top: 40px; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<div class="meta">
  <b>Period:</b> {start} → {end} &nbsp;|&nbsp;
  <b>Initial capital:</b> ${capital:,.0f} &nbsp;|&nbsp;
  <b>Portfolios tested:</b> {n}
</div>

<h2>Summary metrics</h2>
{table}

<h2>Equity curves</h2>
<img src="data:image/png;base64,{equity_png}" alt="Equity curves">

<h2>Drawdown</h2>
<img src="data:image/png;base64,{dd_png}" alt="Drawdown">

<h2>Sharpe ratio comparison</h2>
<img src="data:image/png;base64,{sharpe_png}" alt="Sharpe">

<h2>CAGR comparison</h2>
<img src="data:image/png;base64,{cagr_png}" alt="CAGR">

<div class="footer">Generated by InvestPortfolio Optimizer — Backtest Engine v{schema}.</div>
</body>
</html>
"""


def save_html(report: BacktestReport, path: str) -> None:
    """Write a self-contained HTML report with embedded charts."""
    html = _HTML_TEMPLATE.format(
        start=report.start_date,
        end=report.end_date,
        capital=report.initial_capital,
        n=len(report.results),
        table=_metrics_table_html(report),
        equity_png=_equity_chart_b64(report),
        dd_png=_drawdown_chart_b64(report),
        sharpe_png=_metrics_bar_b64(report, "sharpe_ratio", "Sharpe ratio"),
        cagr_png=_metrics_bar_b64(report, "cagr", "CAGR"),
        schema=SCHEMA_VERSION,
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
