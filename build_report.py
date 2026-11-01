"""Turn the results files into the readiness report and the numbers for the governance memo.

Every number comes from config/ or results/; none is typed by hand.
Writes results/readiness_report.md and results/readiness_report.html.
"""
import csv
import html
import re

from common import CONFIG, RESULTS, load_json, now


def read_csv(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    """Number as text, the way it is stored (no rounding beyond what the results file holds)."""
    return f"{x:g}" if isinstance(x, float) else str(x)


def at_eps(rows, eps):
    return next((r for r in rows if float(r["epsilon"]) == eps), None)


# Report blocks, rendered to both Markdown and HTML -------------------------------------------
def to_markdown(blocks):
    out = []
    for kind, *a in blocks:
        if kind == "h1":
            out += [f"# {a[0]}", ""]
        elif kind == "h2":
            out += [f"## {a[0]}", ""]
        elif kind == "p":
            out += [a[0], ""]
        elif kind == "ul":
            out += [f"- {item}" for item in a[0]] + [""]
        elif kind == "img":
            out += [f"![{a[0]}]({a[1]})", ""]
        elif kind == "table":
            heads, rows = a
            out += ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
            out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]
    return "\n".join(out)


def inline_html(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\bPASS\b", '<span class="pass">PASS</span>', text)
    return re.sub(r"\bFAIL\b", '<span class="fail">FAIL</span>', text)


def to_html(blocks, title):
    body = []
    for kind, *a in blocks:
        if kind in ("h1", "h2"):
            body.append(f"<{kind}>{html.escape(a[0])}</{kind}>")
        elif kind == "p":
            cls = ' class="verdict"' if a[0].startswith("**Recommendation") else ""
            body.append(f"<p{cls}>{inline_html(a[0])}</p>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{inline_html(i)}</li>" for i in a[0]) + "</ul>")
        elif kind == "img":
            body.append(f'<figure><img src="{html.escape(a[1])}" alt="{html.escape(a[0])}"></figure>')
        elif kind == "table":
            heads, rows = a
            cell = lambda c: (f'<td class="pass">PASS</td>' if c == "PASS" else
                              f'<td class="fail">FAIL</td>' if c == "FAIL" else f"<td>{html.escape(str(c))}</td>")
            body.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{html.escape(h)}</th>" for h in heads)
                        + "</tr></thead><tbody>" + "".join("<tr>" + "".join(cell(c) for c in r) + "</tr>" for r in rows)
                        + "</tbody></table></div>")
    css = """
:root{--bg:#fff;--fg:#14212a;--muted:#5b6b75;--line:#c4cfd4;--head:#eef2f4;--pass:#1a7f37;--passbg:#dcf3e3;--fail:#b42318;--failbg:#fde2de}
@media (prefers-color-scheme:dark){:root{--bg:#12181c;--fg:#e4eaee;--muted:#9aa8b0;--line:#34424a;--head:#1c262c;--pass:#5fd17f;--passbg:#163222;--fail:#ff8a7a;--failbg:#3a1c18}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 "Segoe UI",Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px}
h1{font-size:1.6em;margin-bottom:.2em}h2{font-size:1.2em;margin-top:1.8em;border-bottom:1px solid var(--line);padding-bottom:4px}
.scroll{overflow-x:auto}table{border-collapse:collapse;margin:8px 0 16px;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:4px 9px;text-align:left}th{background:var(--head);font-weight:600}
.pass{color:var(--pass);background:var(--passbg);font-weight:700}.fail{color:var(--fail);background:var(--failbg);font-weight:700}
span.pass,span.fail{padding:0 4px;border-radius:3px}
.verdict{font-size:1.15em;padding:10px 14px;border:1px solid var(--line);border-radius:6px;background:var(--head)}
figure{margin:12px 0}img{max-width:100%;height:auto;background:#fff}li{margin:4px 0}
"""
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{css}</style></head><body>\n" + "\n".join(body) + "\n</body></html>\n")


def main():
    print("Reading config/ and results/ ...")
    acc, frozen = load_json(CONFIG / "acceptance.json"), load_json(CONFIG / "frozen.json")
    tm = load_json(RESULTS / "test_metrics.json")
    sweep, shifts, eff, types = (read_csv(n) for n in ("fgsm_sweep.csv", "natural_shift.csv", "efficiency.csv",
                                                        "recall_by_defect_type.csv"))
    opened = load_json(RESULTS / "test_opened.json")["openings"]
    times_opened = "once" if len(opened) == 1 else f"{len(opened)} times"
    c32, c8 = tm["clean"]["FP32"], tm["clean"]["INT8"]
    n_good, n_bad = tm["n_test_good"], tm["n_test_defect"]
    wc = sorted((r for r in sweep if r["mode"] == "worst_case"), key=lambda r: float(r["epsilon"]))
    ev = sorted((r for r in sweep if r["mode"] == "evasion"), key=lambda r: float(r["epsilon"]))
    pg = sorted((r for r in sweep if r["mode"].startswith("pgd")), key=lambda r: float(r["epsilon"]))
    eff_by = {r["model"]: r for r in eff}

    # Pre-set criteria, exactly as written in config/acceptance.json ----------------------------
    e_chk, floor, shift_floor = acc["robustness_check_eps"], acc["robustness_floor_auroc"], acc["natural_shift_floor_auroc"]
    fgsm_chk, pgd_chk = at_eps(wc, e_chk), at_eps(pg, e_chk)
    attacked = [(float(fgsm_chk["auroc_FP32"]), "FGSM")] if fgsm_chk else []
    attacked += [(float(pgd_chk["auroc_FP32"]), f"PGD-{frozen['pgd']['steps']}")] if pgd_chk else []
    attacked_auroc, attacked_by = min(attacked)
    drop = round(c32["auroc"] - c8["auroc"], 4)
    worst_shift, worst_model = min(((s, m) for s in shifts for m in ("FP32", "INT8")),
                                   key=lambda sm: float(sm[0][f"auroc_{sm[1]}"]))
    worst_shift_auroc = float(worst_shift[f"auroc_{worst_model}"])

    criteria = []
    for name, c in (("FP32", c32), ("INT8", c8)):
        criteria.append((f"Clean test AUROC ({name})", f">= {fnum(acc['min_clean_test_auroc'])}", fnum(c["auroc"]),
                         c["auroc"] >= acc["min_clean_test_auroc"]))
    for name, c in (("FP32", c32), ("INT8", c8)):
        criteria.append((f"Test recall at frozen threshold ({name})", f">= {fnum(acc['min_test_recall_at_threshold'])}",
                         fnum(c["recall"]), c["recall"] >= acc["min_test_recall_at_threshold"]))
    criteria.append(("INT8 AUROC drop versus FP32", f"<= {fnum(acc['max_int8_auroc_drop'])}", fnum(drop),
                     drop <= acc["max_int8_auroc_drop"]))
    for name in ("FP32", "INT8"):
        p95 = float(eff_by[name]["p95_ms"])
        criteria.append((f"p95 latency, {name} ONNX end to end (ms)", f"<= {fnum(acc['max_p95_latency_ms'])}",
                         eff_by[name]["p95_ms"], p95 <= acc["max_p95_latency_ms"]))
    criteria.append((f"Attacked AUROC at eps {fnum(e_chk)} ({e_chk * 255:.2f}/255), lower of FGSM and PGD worst case "
                     f"(lower: {attacked_by})", f">= {fnum(floor)}", fnum(attacked_auroc), attacked_auroc >= floor))
    criteria.append((f"Worst natural-shift AUROC ({worst_shift['condition']}, {worst_model})", f">= {fnum(shift_floor)}",
                     fnum(worst_shift_auroc), worst_shift_auroc >= shift_floor))
    go = all(ok for *_, ok in criteria)
    failed = [c[0] for c in criteria if not c[3]]

    # Findings beyond the pre-set criteria, derived from the numbers -----------------------------
    findings = []
    pgd_fp32_recalls = [float(r["recall_FP32"]) for r in pg]
    pgd_fa = [int(r["false_alarms_FP32"]) for r in pg]
    pgd_int8 = min(pg, key=lambda r: float(r["recall_INT8_transfer"])) if pg else None
    if pg and min(pgd_fp32_recalls) >= acc["min_test_recall_at_threshold"] and max(pgd_fa) > 0:
        eps_list = " and ".join(f"{r['epsilon']} ({float(r['epsilon']) * 255:.2f}/255)" for r in pg)
        text = (f"(a) Under PGD worst case at eps {eps_list}, FP32 recall stays at "
                f"{' and '.join(r['recall_FP32'] for r in pg)} while false alarms reach "
                f"{' and '.join(r['false_alarms_FP32'] for r in pg)} of {n_good} good parts. "
                f"The failure mode is false alarms on good parts, not missed defects: AUROC collapses to "
                f"{' and '.join(r['auroc_FP32'] for r in pg)} because good parts are pushed above defective ones.")
        int8_missed = round((1 - float(pgd_int8["recall_INT8_transfer"])) * n_bad)
        if int8_missed > 0:
            text += (f" One caveat: on the INT8 model the same attacked images give recall "
                     f"{pgd_int8['recall_INT8_transfer']} at eps {pgd_int8['epsilon']}, that is {int8_missed} of "
                     f"{n_bad} defects missed.")
        findings.append(text)
    elif pg:
        findings.append(f"(a) Under PGD worst case the lowest FP32 recall is {fnum(min(pgd_fp32_recalls))}: "
                        "the attack does cause missed defects.")
    flagged = [s for s in shifts if float(s["auroc_FP32"]) >= shift_floor and int(s["false_alarms_FP32"]) == n_good]
    if flagged:
        findings.append(
            "(b) Under " + ", ".join(f"{s['condition']} ({s['description']})" for s in flagged)
            + f" the AUROC criterion passes (FP32 AUROC "
            + ", ".join(s["auroc_FP32"] for s in flagged) + f"; floor {fnum(shift_floor)}), but the frozen threshold "
            "flags every good part (" + ", ".join(f"{s['false_alarms_FP32']} of {n_good}" for s in flagged)
            + " false alarms; INT8 " + ", ".join(f"{s['false_alarms_INT8']} of {n_good}" for s in flagged)
            + "). The scores still rank defects above good parts, but the whole score scale moves, so AUROC was not "
            "a sufficient criterion for operational robustness.")
    pgd_modes = sorted({r["mode"] for r in pg})
    if not any("evasion" in m for m in pgd_modes):
        findings.append(f"(c) PGD was run only as {', '.join(pgd_modes) or 'nothing'}; PGD in evasion mode "
                        "(pushing only defective parts towards 'normal') was not tested. One-step FGSM evasion left "
                        f"recall at {', '.join(sorted({r['recall_FP32'] for r in ev}))} at every epsilon, but whether "
                        "a multi-step attack can hide defects is unmeasured.")

    # Numbers for the governance memo ----------------------------------------------------------
    first_fail = next((r for r in wc if float(r["auroc_FP32"]) < floor), None)
    e = eff_by
    memo = [
        f"Recommendation: {'GO' if go else 'NO-GO'}. " + (f"{len(criteria)} of {len(criteria)} pre-set criteria passed."
            if go else f"{len(criteria) - len(failed)} of {len(criteria)} pre-set criteria passed; failed: "
            + "; ".join(failed) + "."),
        f"On the sealed test set ({n_good} good, {n_bad} defective parts, opened {times_opened}), the detector "
        f"reached AUROC {fnum(c32['auroc'])} (FP32) and {fnum(c8['auroc'])} (INT8), with recall {fnum(c32['recall'])} "
        f"and {fnum(c8['recall'])}, {c32['fn']} and {c8['fn']} missed defects, and {c32['fp']} and {c8['fp']} "
        "false alarms at the thresholds frozen on validation data.",
        f"Under one-step FGSM worst case at eps {fnum(e_chk)} ({e_chk * 255:.2f}/255), AUROC falls from "
        f"{fnum(c32['auroc'])} to {fgsm_chk['auroc_FP32']}, with {fgsm_chk['false_alarms_FP32']} of {n_good} good parts "
        f"flagged; FGSM evasion at the same strength leaves recall at {at_eps(ev, e_chk)['recall_FP32']}.",
        (f"FGSM worst case first takes AUROC below the {fnum(floor)} floor at eps {first_fail['epsilon']} "
         f"({first_fail['epsilon_255']}/255)." if first_fail else
         f"FGSM worst case keeps AUROC above the {fnum(floor)} floor at every epsilon tested, up to "
         f"{wc[-1]['epsilon']} ({wc[-1]['epsilon_255']}/255)."),
    ]
    if pgd_chk:
        memo.append(f"A 10-step PGD worst-case attack at eps {fnum(e_chk)} drives AUROC to {pgd_chk['auroc_FP32']} "
                    f"(INT8 on the same images: {pgd_chk['auroc_INT8_transfer']}): {pgd_chk['false_alarms_FP32']} of "
                    f"{n_good} good parts are flagged while recall stays at {pgd_chk['recall_FP32']}.")
    memo.append(f"Attack images built on FP32 transfer to INT8: worst-case FGSM at eps {fnum(e_chk)} gives INT8 AUROC "
                f"{fgsm_chk['auroc_INT8_transfer']}.")
    if flagged:
        memo.append(f"Natural image changes keep AUROC at or above {fnum(min(float(s['auroc_FP32']) for s in shifts))} "
                    f"(FP32), but under {', '.join(s['condition'] for s in flagged)} the frozen threshold flags "
                    f"all {n_good} good parts.")
    memo.append(f"INT8 shrinks the backbone from {e['FP32']['file_MB']} MB to {e['INT8']['file_MB']} MB and runs at "
                f"p50 {e['INT8']['p50_ms']} ms versus {e['FP32']['p50_ms']} ms ({e['INT8']['speedup_p50']}x), p95 "
                f"{e['INT8']['p95_ms']} ms, on an {e['INT8']['cpu']} limited to {e['INT8']['threads']} threads.")

    # Assemble ----------------------------------------------------------------------------------
    title = "Readiness report: MedVision bottle anomaly detector"
    b = [("h1", title),
         ("p", f"Generated {now()} from results evaluated {tm['evaluated_at']}. Thresholds frozen "
               f"{frozen['frozen_at']}. Test set opened {times_opened}, first at {opened[0]['at']}."),
         ("p", f"**Recommendation: {'GO' if go else 'NO-GO'}.** " + ("Every pre-set criterion passed." if go else
               f"{len(failed)} pre-set criterion failed. The criteria in config/acceptance.json were written before "
               "any model ran and are not changed after the fact." if len(failed) == 1 else
               f"{len(failed)} pre-set criteria failed. The criteria in config/acceptance.json were written before "
               "any model ran and are not changed after the fact.")),
         ("h2", "Pre-set criteria"),
         ("table", ["criterion", "required", "measured", "result"],
          [[n, r, m, "PASS" if ok else "FAIL"] for n, r, m, ok in criteria]),
         ("h2", f"Clean performance (sealed test set: {n_good} good, {n_bad} defective)"),
         ("table", ["model", "threshold", "AUROC", "AP", "precision", "recall", "F1", "TP", "FP (false alarms)",
                    "FN (missed defects)", "TN"],
          [[k, fnum(tm["thresholds"][k]), *(fnum(v[c]) for c in ("auroc", "ap", "precision", "recall", "f1",
                                                                     "tp", "fp", "fn", "tn"))]
           for k, v in tm["clean"].items()]),
         ("p", "Recall by defect type:"),
         ("table", ["defect type", "n", "FP32 recall", "INT8 recall"],
          [[r["defect_type"], r["n"], r["recall_FP32"], r["recall_INT8"]] for r in types]),
         ("h2", "FGSM sweep (one step, attack built on FP32 with full model access)"),
         ("table", ["epsilon", "out of 255", "worst case AUROC", "worst case AP", "worst case recall",
                    "worst case false alarms", "evasion AUROC", "evasion AP", "evasion recall", "evasion false alarms",
                    "INT8 AUROC (same worst-case images)", "INT8 recall (same images)"],
          [[w["epsilon"], w["epsilon_255"], w["auroc_FP32"], w["ap_FP32"], w["recall_FP32"], w["false_alarms_FP32"],
            v["auroc_FP32"], v["ap_FP32"], v["recall_FP32"], v["false_alarms_FP32"], w["auroc_INT8_transfer"],
            w["recall_INT8_transfer"]] for w, v in zip(wc, ev)]),
         ("h2", f"PGD ({frozen['pgd']['steps']} steps of {frozen['pgd']['step_size']}, worst case)"),
         ("table", ["epsilon", "out of 255", "AUROC", "AP", "recall", "false alarms",
                    "INT8 AUROC (same images)", "INT8 recall (same images)"],
          [[r["epsilon"], r["epsilon_255"], r["auroc_FP32"], r["ap_FP32"], r["recall_FP32"], r["false_alarms_FP32"],
            r["auroc_INT8_transfer"], r["recall_INT8_transfer"]] for r in pg]),
         ("p", f"Naive FGSM that pushes every image towards 'anomalous' at eps 0.03: AUROC "
               f"{fnum(tm['naive_fgsm_eps_0.03']['auroc'])}, AP {fnum(tm['naive_fgsm_eps_0.03']['ap'])}, recall "
               f"{fnum(tm['naive_fgsm_eps_0.03']['recall'])}, {tm['naive_fgsm_eps_0.03']['fp']} false alarms."),
         ("img", "AUROC against attack strength", "fgsm_curve.png"),
         ("img", "One defective part before and after the attack", "fgsm_example.png"),
         ("h2", "Natural shift (no attacker)"),
         ("table", ["condition", "what", "AUROC FP32", "recall FP32", "false alarms FP32",
                    "AUROC INT8", "recall INT8", "false alarms INT8"],
          [[s["condition"], s["description"], s["auroc_FP32"], s["recall_FP32"], f"{s['false_alarms_FP32']} of {n_good}",
            s["auroc_INT8"], s["recall_INT8"], f"{s['false_alarms_INT8']} of {n_good}"] for s in shifts]),
         ("h2", f"Efficiency ({e['INT8']['cpu']}, {e['INT8']['threads']} threads, one image, "
                f"{e['INT8']['warmups']} warm-ups, {e['INT8']['runs']} timed runs)"),
         ("table", ["model", "clean AUROC", "p50 ms", "p95 ms", "p99 ms", "speed-up (p50)", "backbone MB",
                    "memory bank vectors", "memory bank MB"],
          [[r["model"], fnum(tm["clean"][r["model"]]["auroc"]), r["p50_ms"], r["p95_ms"], r["p99_ms"],
            r["speedup_p50"], r["file_MB"], r["bank_vectors"], r["bank_MB"]] for r in eff]),
         ("h2", "Findings beyond the pre-set criteria"),
         ("ul", findings),
         ("h2", "Numbers for the governance memo"),
         ("ul", memo)]

    (RESULTS / "readiness_report.md").write_text(to_markdown(b), encoding="utf-8")
    (RESULTS / "readiness_report.html").write_text(to_html(b, title), encoding="utf-8")
    print(f"Recommendation: {'GO' if go else 'NO-GO'}")
    for n, r, m, ok in criteria:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}: {m} (required {r})")
    print("Wrote results/readiness_report.md and results/readiness_report.html")


if __name__ == "__main__":
    main()
