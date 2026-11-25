"""HTML and Markdown reports of a model check."""
from __future__ import annotations

import base64
import html
import json
import os

import numpy as np

from . import plots

BADGE = {"pass": ("#e7f6ea", "#006c10", "PASS"), "caution": ("#fff4d6", "#7a5200", "CAUTION"),
         "fail": ("#fde8e8", "#a11a1b", "FAIL"), "info": ("#eef3fb", "#1b4f99", "FINDING")}

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#191919;margin:0;background:#fff}
main{max-width:920px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:24px;margin:0 0 6px} h2{font-size:17px;margin:34px 0 10px;border-bottom:1px solid #e3e3e3;padding-bottom:6px}
p{line-height:1.5} .lead{background:#f6f8fb;border-left:4px solid #257de0;padding:12px 14px;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 14px} th{text-align:left;color:#666;font-weight:600}
th,td{padding:6px 8px;border-bottom:1px solid #eee;vertical-align:top} td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 7px;border-radius:10px;white-space:nowrap}
.wrap{overflow-x:auto} img{max-width:100%;height:auto} small,.muted{color:#666} code{background:#f3f3f3;padding:1px 4px;border-radius:3px}
"""


def _table(df, digits=3, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append(f'<td class="num">{v:.{digits}f}</td>' if np.isfinite(v) else '<td class="num">&infin;</td>')
            elif isinstance(v, (int, np.integer)):
                cells.append(f'<td class="num">{v}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _img(png, alt):
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def to_html(fit, report, title="EAG model check"):
    """Return a self-contained HTML report (figures embedded)."""
    t = report.tables
    free = fit.free
    parts = [f"<h1>{html.escape(title)}</h1>",
             f'<p class="muted">{fit.data.n_students} students, {fit.data.n_components} components, '
             f'{fit.data.n_occasions} occasion(s), {fit.data.n_raters} rater(s). '
             f'Estimated in {fit.seconds:.1f} s with eagmodel.</p>',
             f'<p class="lead">{html.escape(report.headline)}</p>', "<h2>Checks</h2>"]
    rows = []
    for c in report.checks:
        bg, fg, lab = BADGE[c["verdict"]]
        rows.append(f"<tr><td>{html.escape(c['layer'])}</td><td>{html.escape(c['check'])}</td>"
                    f"<td>{html.escape(c['result'])}</td><td class='muted'>{html.escape(c['criterion'])}</td>"
                    f"<td><span class='badge' style='background:{bg};color:{fg}'>{lab}</span></td></tr>")
    parts.append('<div class="wrap"><table><thead><tr><th>Layer</th><th>Check</th><th>Result</th><th>Criterion</th>'
                 f'<th>Verdict</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    parts.append("<h2>Aggregation function</h2>")
    parts.append(f"<p>Competence is modelled as C = (&Sigma; w<sub>k</sub> &pi;<sub>k</sub><sup>&rho;</sup>)<sup>1/&rho;</sup>. "
                 f"The estimate is &rho; = {free.rho:.2f}, that is, a compensation index &kappa; = {free.kappa:.2f} "
                 f"(1 = sum score, 0.5 = geometric mean, towards 0 = weakest component decides). "
                 f"Holistic ratings follow {free.intercept:.1f} + {free.slope:.1f} &times; C with residual SD {free.sigma:.1f}.</p>")
    if report.profile is not None:
        parts.append(_img(plots.profile_figure(report.profile, free.kappa, free.kappa_ci), "Profile likelihood of kappa"))
    parts.append(_img(plots.form_figure(t["functional_form"]), "Residuals by profile unevenness"))
    parts.append(_img(plots.weights_figure(t["weights"]), "Component weights"))
    parts.append("<h2>Model comparison</h2>" + _table(t["models"], 2))
    parts.append("<h2>Components: reliability and growth</h2>" + _table(t["components"], 3))
    if fit.has_growth:
        parts.append(_img(plots.trajectory_figure(fit, t["students"]), "Mean mastery by design cycle"))
    parts.append("<h2>Indicators</h2>" + _table(t["items"].drop(columns=["category_counts"]), 2))
    parts.append("<h2>Raters</h2>" + _table(t["raters"], 2))
    parts.append("<h2>Students (first 25 rows; the full table is in students.csv)</h2>" + _table(t["students"], 2, 25))
    parts.append('<p class="muted">Verdicts apply conventional rules of thumb that are printed beside each check. '
                 'They support, and do not replace, substantive judgement about the rubric and the ratings.</p>')
    return f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"


def to_markdown(fit, report):
    lines = ["# EAG model check", "", report.headline, "", "| Layer | Check | Result | Verdict |", "|---|---|---|---|"]
    for c in report.checks:
        lines.append(f"| {c['layer']} | {c['check']} | {c['result']} | {c['verdict']} |")
    return "\n".join(lines) + "\n"


def summary_dict(fit, report):
    free = fit.free
    return dict(
        n_students=fit.data.n_students, components=list(fit.data.components),
        kappa=free.kappa, kappa_ci=list(free.kappa_ci), rho=free.rho, rho_ci=list(free.rho_ci),
        weights=dict(zip(fit.data.components, map(float, free.weights))),
        weights_se=dict(zip(fit.data.components, map(float, free.weights_se))),
        link=dict(intercept=free.intercept, slope=free.slope, residual_sd=free.sigma),
        rater_severity=dict(zip(fit.data.raters, map(float, free.severity))),
        lr_test_sum_score=fit.lr_linear, headline=report.headline, verdicts=report.verdict_counts(),
        checks=report.checks)


def write_report(fit, report, folder, title="EAG model check"):
    """Write report.html, report.md, summary.json and the result tables into a folder."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(to_html(fit, report, title))
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(to_markdown(fit, report))
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary_dict(fit, report), fh, indent=2, default=float)
    t = report.tables
    t["students"].to_csv(os.path.join(folder, "students.csv"), index=False)
    t["items"].to_csv(os.path.join(folder, "indicators.csv"), index=False)
    t["components"].to_csv(os.path.join(folder, "components.csv"), index=False)
    t["models"].to_csv(os.path.join(folder, "models.csv"), index=False)
    t["weights"].to_csv(os.path.join(folder, "weights.csv"), index=False)
    return os.path.join(folder, "report.html")
