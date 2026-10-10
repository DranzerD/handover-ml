"""HTML report for one select -> execute -> compare run."""

from __future__ import annotations

from pathlib import Path

from cellselect.reporting import charts
from cellselect.reporting import html as H


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def build_run_report(payload: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    sel = payload["selection"]
    exe = payload["execution"]
    rt = payload["runtime"]
    fd = payload["fault_detection"]
    gt = payload["ground_truth_revealed_after_execution"]
    vis = payload["scenario"]["visible"]
    failed_full = set(exe["full"]["failed"])
    selected_results = {r["test_id"]: r for r in exe["selected_results"]}
    detecting = {t for ts in gt["detecting_tests"].values() for t in ts}

    risk_png = charts.plot_selection_risk(sel["decisions"], failed_full, out / "risk_vs_failures.png")
    comp_png = charts.plot_strategy_comparison(
        payload["baseline_comparison"], out / "strategy_comparison.png"
    )
    body = [
        '<div class="note"><b>Simulated validation run.</b> Faults are injected into a behavioural cellular '
        "simulator; predicted risk (model output, before execution) is shown separately from confirmed "
        "failures (pytest results, after execution).</div>"
    ]
    body.append(
        H.tiles(
            [
                ("Tests selected", f"{sel['n_selected']} / {sel['n_total']}"),
                ("Runtime reduction (nominal, e2e)", _pct(rt["e2e_runtime_reduction_nominal"])),
                ("Fault-detection retention", _pct(fd["fault_retention"])),
                ("Faults detected / missed", f"{fd['faults_detected_selected']} / {fd['faults_missed']}"),
                ("Coverage retained", _pct(fd["coverage_retained"])),
                ("Selection overhead", f"{rt['selection_overhead_s'] * 1000:.0f} ms"),
            ]
        )
    )
    body.append("<h2>Change under test (visible to the selector)</h2>")
    env = vis["env"]
    body.append(
        H.table(
            [
                "change",
                "type",
                "area",
                "changed components",
                "lines",
                "device",
                "lab interference dB",
                "branch",
            ],
            [
                [
                    vis["change_id"],
                    vis["change_type"],
                    vis["area"],
                    ", ".join(vis["changed_components"]),
                    vis["lines_changed"],
                    env["device"],
                    env["lab_interference_db"],
                    env["firmware_branch"],
                ]
            ],
        )
    )
    if payload["warnings"] or payload["fallbacks"]:
        body.append(
            "<h2>Warnings and fallback decisions</h2><ul class='warnings'>"
            + "".join(f"<li>WARNING: {H.esc(w)}</li>" for w in payload["warnings"])
            + "".join(f"<li>FALLBACK: {H.esc(f)}</li>" for f in payload["fallbacks"])
            + "</ul>"
        )
    else:
        body.append("<h2>Warnings and fallback decisions</h2><p class='muted'>None.</p>")

    body.append("<h2>Runtime</h2>")
    body.append(
        H.table(
            ["", "full suite", "selected suite"],
            [
                ["nominal lab time (s)", rt["full_nominal_s"], rt["selected_nominal_s"]],
                [
                    "measured pytest wall-clock (s)",
                    rt["measured_wall_full_s"],
                    rt["measured_wall_selected_s"],
                ],
                ["tests executed", exe["full"]["n_run"], exe["selected"]["n_run"]],
                ["confirmed failures", exe["full"]["n_failed"], exe["selected"]["n_failed"]],
            ],
        )
    )
    body.append(
        f"<p class='muted'>Selection overhead {rt['selection_overhead_s']:.3f}s is included in the end-to-end "
        f"reduction. Lab-time emulation scale: {exe['time_scale']}. Determinism check (selected outcomes identical "
        f"to the same tests in the full run): {'OK' if exe['determinism_check']['ok'] else 'MISMATCH ' + str(exe['determinism_check']['mismatched_outcomes'])}.</p>"
    )

    body.append("<h2>Comparison against baseline strategies (same scenario, same budget)</h2>")
    rows = [
        [
            c["strategy"],
            c["n_tests_selected"],
            c["selected_fraction"],
            c["selection_overhead_s"] * 1000,
            c["coverage_retained"],
            c["faults_detected_selected"],
            c["faults_missed"],
            c["fault_retention"],
            c["time_to_first_detection_s"],
            "yes" if c["executed_via_pytest"] else "scored vs. full run",
        ]
        for c in payload["baseline_comparison"]
    ]
    body.append(
        H.table(
            [
                "strategy",
                "tests",
                "lab-time fraction",
                "overhead ms",
                "coverage",
                "faults detected",
                "faults missed",
                "retention",
                "time to 1st detection s",
                "executed",
            ],
            rows,
        )
    )
    body.append(
        '<div class="grid2">'
        + H.img(comp_png, "strategy comparison")
        + H.img(risk_png, "risk vs failures")
        + "</div>"
    )

    body.append("<h2>Ground truth (revealed after execution)</h2>")
    gt_rows = [
        [
            f,
            gt["fault_descriptions"].get(f, ""),
            ", ".join(gt["detecting_tests"].get(f, [])) or "— (no test detects it)",
            ", ".join(t for t in gt["detecting_tests"].get(f, []) if t in sel["selected"]) or "missed",
        ]
        for f in gt["active_faults"]
    ]
    body.append(
        H.table(["fault", "description", "detected by (full suite)", "detected by selection"], gt_rows)
        if gt_rows
        else "<p class='muted'>No faults were active in this scenario (clean change).</p>"
    )
    if gt["noise_failures"]:
        body.append(
            f"<p class='muted'>Noise failures (fail without any fault; environment-sensitive): {', '.join(gt['noise_failures'])}</p>"
        )

    body.append("<h2>Per-test decisions</h2>")
    trows = []
    for d in sel["decisions"]:
        t = d["test_id"]
        if d["selected"]:
            r = selected_results.get(t, {})
            res = (
                "<span class='pill fail'>FAILED</span>"
                if r.get("outcome") != "passed"
                else "<span class='pill pass'>passed</span>"
            )
            if r.get("message"):
                res += f"<br><span class='muted'>{H.esc(r['message'])}</span>"
        else:
            res = "<span class='muted'>not run</span>" + (
                " <span class='pill warn'>would fail</span>" if t in failed_full else ""
            )
        p = "—" if d["predicted_p"] is None else f"{d['predicted_p']:.3f}"
        trows.append(
            [
                d["rank"] or "",
                t,
                d["category"],
                int(d["duration_s"]),
                p,
                "✔ selected" if d["selected"] else "excluded",
                res,
                "fault" if t in detecting else "",
                H.esc(" | ".join(d["reasons"])),
                ", ".join(d["tags"]),
            ]
        )
    body.append(
        H.table(
            [
                "rank",
                "test",
                "category",
                "dur s",
                "predicted p(fail)",
                "decision",
                "actual result",
                "fault-revealing",
                "reasons",
                "coverage tags",
            ],
            trows,
            raw_cols={6, 8},
        )
    )
    return H.write(
        out / "run_report.html",
        H.page(
            f"CellSelect run report — change {vis['change_id']}",
            f"strategy={payload['strategy']} · budget {sel['budget_s']:.0f}s of {sel['full_duration_s']:.0f}s",
            "".join(body),
        ),
    )
