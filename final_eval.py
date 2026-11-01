"""Open the sealed test set ONCE and measure everything the governance memo needs.

  1. clean performance, FP32 (PyTorch) and INT8 (ONNX), at the frozen thresholds
  2. FGSM sweep on the FP32 model (worst case and evasion), 10-step PGD, transfer to INT8
  3. natural shift: darker, blur, noise, JPEG
  4. latency and size, ONNX FP32 versus ONNX INT8, end to end, 2 threads
Refuses to run a second time unless called with --rerun "reason"; every opening is logged
in results/test_opened.json.
"""
import csv
import hashlib
import io
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score

from audit_leakage import run_checks
from common import (CONFIG, MODELS, RESULTS, SPLITS, THREADS, load_batch, load_json, now, rows_for, save_json,
                    setup)
from fit import onnx_scores, ort_session
from patchcore import Backbone, PatchCore, fgsm, pgd

LOCK = RESULTS / "test_opened.json"


def metrics(scores, labels, thr):
    pred = scores >= thr
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"auroc": round(float(roc_auc_score(labels, scores)), 4),
            "ap": round(float(average_precision_score(labels, scores)), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def shift(x01, kind, seed=0):
    """The four frozen natural shifts."""
    if kind == "darker":
        return (x01 * 0.7).clamp(0, 1)
    if kind == "noise":
        g = torch.Generator().manual_seed(seed)
        return (x01 + 0.04 * torch.randn(x01.shape, generator=g)).clamp(0, 1)
    out = []
    for img in x01:
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        if kind == "blur":
            pil = pil.filter(ImageFilter.GaussianBlur(2))
        elif kind == "jpeg":
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=30)
            buf.seek(0)
            pil = Image.open(buf).convert("RGB")
        else:
            raise ValueError(kind)
        out.append(torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255).permute(2, 0, 1))
    return torch.stack(out)


def latency(sess, bank, x1, warm=10, runs=100):
    """End-to-end time for one image: normalise, ONNX backbone, nearest-neighbour scoring."""
    with torch.no_grad():
        for _ in range(warm):
            onnx_scores(sess, bank, x1)
        t = []
        for _ in range(runs):
            s = time.perf_counter()
            onnx_scores(sess, bank, x1)
            t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return {"p50_ms": round(float(np.percentile(t, 50)), 1), "p95_ms": round(float(np.percentile(t, 95)), 1),
            "p99_ms": round(float(np.percentile(t, 99)), 1), "warmups": warm, "runs": runs}


def cpu_name():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        import platform
        return platform.processor() or platform.machine()


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def evaluate(rows, out, frozen, acc, latency_runs=100):
    """Score the given rows with every frozen measurement and write the results into folder `out`."""
    setup()
    t0 = time.time()
    x = load_batch([r["path"] for r in rows])
    y = np.array([r["label"] for r in rows])
    types = [r["defect_type"] for r in rows]
    thr32, thr8 = frozen["models"]["FP32"]["threshold"], frozen["models"]["INT8"]["threshold"]
    fp32 = PatchCore(Backbone(), torch.load(MODELS / "bank_fp32.pt")["bank"])
    bank8 = torch.load(MODELS / "bank_int8.pt")["bank"]
    s32, s8 = ort_session(MODELS / "backbone_fp32.onnx"), ort_session(MODELS / "backbone_int8.onnx")

    def int8(imgs):
        with torch.no_grad():
            return onnx_scores(s8, bank8, imgs).numpy()

    print(f"Loaded {int((y == 0).sum())} good and {int((y == 1).sum())} defective images. "
          f"Frozen thresholds: FP32 {thr32:.4f}, INT8 {thr8:.4f}")

    # 1. Clean -----------------------------------------------------------------------
    print("\nPart 1/4: clean images")
    clean32 = fp32.score_batched(x).numpy()
    clean8 = int8(x)
    clean = {"FP32": metrics(clean32, y, thr32), "INT8": metrics(clean8, y, thr8)}
    for k, v in clean.items():
        print(f"  Clean {k}: AUROC {v['auroc']:.3f}  AP {v['ap']:.3f}  precision {v['precision']:.2f}  "
              f"recall {v['recall']:.2f}  F1 {v['f1']:.2f}  missed defects {v['fn']}  false alarms {v['fp']}")
    per_type = []
    for t in sorted(set(types) - {"good"}):
        m = np.array([tt == t for tt in types])
        per_type.append({"defect_type": t, "n": int(m.sum()),
                         "recall_FP32": round(float((clean32[m] >= thr32).mean()), 4),
                         "recall_INT8": round(float((clean8[m] >= thr8).mean()), 4)})
        print(f"  Recall on {t:13s} ({int(m.sum())} images): FP32 {per_type[-1]['recall_FP32']:.2f}, "
              f"INT8 {per_type[-1]['recall_INT8']:.2f}")

    # 2. Attacks ---------------------------------------------------------------------------
    print("\nPart 2/4: adversarial attacks on the FP32 model")
    yt = torch.tensor(y)
    attack_rows, worst = [], {}
    for eps in frozen["epsilons"]:
        for mode in frozen["attack_modes"]:
            xa = fgsm(fp32, x, yt, eps, mode)
            m = metrics(fp32.score_batched(xa).numpy(), y, thr32)
            row = {"epsilon": eps, "epsilon_255": round(eps * 255, 2), "mode": mode, "auroc_FP32": m["auroc"],
                   "ap_FP32": m["ap"], "recall_FP32": m["recall"], "false_alarms_FP32": m["fp"],
                   "auroc_INT8_transfer": "", "recall_INT8_transfer": ""}
            line = (f"  FGSM {mode:10s} eps {eps:.3f} ({eps * 255:4.1f}/255): AUROC {m['auroc']:.3f}  "
                    f"AP {m['ap']:.3f}  recall {m['recall']:.2f}  false alarms {m['fp']}")
            if mode == "worst_case":
                m8 = metrics(int8(xa), y, thr8)
                row["auroc_INT8_transfer"], row["recall_INT8_transfer"] = m8["auroc"], m8["recall"]
                worst[eps] = xa
                line += f"  | INT8 on same images: AUROC {m8['auroc']:.3f}  recall {m8['recall']:.2f}"
            attack_rows.append(row)
            print(line)
    pg = frozen["pgd"]
    for eps in pg["epsilons"]:
        xa = pgd(fp32, x, yt, eps, pg["steps"])
        m = metrics(fp32.score_batched(xa).numpy(), y, thr32)
        m8 = metrics(int8(xa), y, thr8)
        attack_rows.append({"epsilon": eps, "epsilon_255": round(eps * 255, 2), "mode": f"pgd{pg['steps']}_worst_case",
                            "auroc_FP32": m["auroc"], "ap_FP32": m["ap"], "recall_FP32": m["recall"],
                            "false_alarms_FP32": m["fp"], "auroc_INT8_transfer": m8["auroc"],
                            "recall_INT8_transfer": m8["recall"]})
        print(f"  PGD-{pg['steps']} worst case eps {eps:.3f} ({eps * 255:4.1f}/255): AUROC {m['auroc']:.3f}  "
              f"AP {m['ap']:.3f}  recall {m['recall']:.2f}  false alarms {m['fp']}  "
              f"| INT8 on same images: AUROC {m8['auroc']:.3f}  recall {m8['recall']:.2f}")
    naive = metrics(fp32.score_batched(fgsm(fp32, x, yt, 0.03, "naive_up")).numpy(), y, thr32)
    print(f"  Naive FGSM (every image pushed up) eps 0.030: AUROC {naive['auroc']:.3f}  "
          f"recall {naive['recall']:.2f}  false alarms {naive['fp']}")

    # 3. Natural shift ------------------------------------------------------------------------
    print("\nPart 3/4: natural image changes")
    shift_rows = []
    for kind, desc in frozen["shifts"].items():
        xs = shift(x, kind)
        m32 = metrics(fp32.score_batched(xs).numpy(), y, thr32)
        m8 = metrics(int8(xs), y, thr8)
        shift_rows.append({"condition": kind, "description": desc,
                           "auroc_FP32": m32["auroc"], "ap_FP32": m32["ap"], "recall_FP32": m32["recall"],
                           "false_alarms_FP32": m32["fp"], "auroc_INT8": m8["auroc"], "ap_INT8": m8["ap"],
                           "recall_INT8": m8["recall"], "false_alarms_INT8": m8["fp"]})
        print(f"  {kind:7s} ({desc}): AUROC FP32 {m32['auroc']:.3f} / INT8 {m8['auroc']:.3f}  "
              f"recall {m32['recall']:.2f} / {m8['recall']:.2f}  false alarms {m32['fp']} / {m8['fp']}")

    # 4. Latency and size ------------------------------------------------------------------------
    print(f"\nPart 4/4: speed on one image, {THREADS} threads")
    cpu = cpu_name()
    eff = []
    for name, sess, bank, f in (("FP32", s32, fp32.bank, "backbone_fp32.onnx"),
                                ("INT8", s8, bank8, "backbone_int8.onnx")):
        lat = latency(sess, bank, x[:1], runs=latency_runs)
        eff.append({"model": name, "cpu": cpu, "threads": THREADS,
                    "file_MB": round((MODELS / f).stat().st_size / 1e6, 2),
                    "bank_vectors": len(bank), "bank_MB": round(bank.numel() * bank.element_size() / 1e6, 2), **lat})
        print(f"  {name}: p50 {lat['p50_ms']} ms, p95 {lat['p95_ms']} ms, p99 {lat['p99_ms']} ms, "
              f"model file {eff[-1]['file_MB']} MB, memory bank {eff[-1]['bank_MB']} MB")
    eff[0]["speedup_p50"] = 1.0
    eff[1]["speedup_p50"] = round(eff[0]["p50_ms"] / eff[1]["p50_ms"], 2)
    print(f"  CPU: {cpu}. INT8 is {eff[1]['speedup_p50']}x the speed of FP32 (median)")

    # Save -------------------------------------------------------------------------------------
    write_csv(attack_rows, out / "fgsm_sweep.csv")
    write_csv(shift_rows, out / "natural_shift.csv")
    write_csv(eff, out / "efficiency.csv")
    write_csv(per_type, out / "recall_by_defect_type.csv")
    save_json({"evaluated_at": now(), "cpu": cpu, "threads": THREADS,
               "n_test_good": int((y == 0).sum()), "n_test_defect": int((y == 1).sum()),
               "thresholds": {"FP32": thr32, "INT8": thr8}, "clean": clean,
               "naive_fgsm_eps_0.03": {k: naive[k] for k in ("auroc", "ap", "recall", "fp")}},
              out / "test_metrics.json")

    # AUROC against epsilon.
    wc = [r for r in attack_rows if r["mode"] == "worst_case"]
    ev = [r for r in attack_rows if r["mode"] == "evasion"]
    pgr = [r for r in attack_rows if r["mode"].startswith("pgd")]
    floor = acc["robustness_floor_auroc"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([r["epsilon_255"] for r in wc], [r["auroc_FP32"] for r in wc], "o-", label="FP32, FGSM worst case")
    ax.plot([r["epsilon_255"] for r in ev], [r["auroc_FP32"] for r in ev], "s--", label="FP32, FGSM evasion (defects only)")
    ax.plot([r["epsilon_255"] for r in wc], [r["auroc_INT8_transfer"] for r in wc], "^:",
            label="INT8, same worst-case images")
    ax.plot([r["epsilon_255"] for r in pgr], [r["auroc_FP32"] for r in pgr], "kx", ms=9, mew=2,
            label=f"FP32, PGD-{pg['steps']} worst case")
    ax.axhline(floor, color="grey", lw=1)
    ax.text(0, floor + 0.015, f"acceptance floor {floor}", ha="left", va="bottom", color="grey", fontsize=8)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Attack strength epsilon (pixel steps out of 255)")
    ax.set_ylabel("AUROC on sealed test set")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Detection quality under adversarial attack")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fgsm_curve.png", dpi=150)
    plt.close(fig)

    # One defective image before and after the strongest worst-case FGSM.
    eps_show = max(worst)
    bad_idx = np.where(y == 1)[0]
    adv = fp32.score_batched(worst[eps_show][bad_idx]).numpy()
    crossed = [k for k, j in enumerate(bad_idx) if clean32[j] >= thr32 and adv[k] < thr32]
    k = crossed[0] if crossed else int(np.argmax(clean32[bad_idx] - adv))
    i, adv_score = int(bad_idx[k]), float(adv[k])
    pert = worst[eps_show][i] - x[i]
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.9))
    axs[0].imshow(x[i].permute(1, 2, 0))
    axs[0].set_title(f"Defective part ({types[i]})\nscore {clean32[i]:.2f}", fontsize=10)
    axs[1].imshow((pert / (2 * eps_show) + 0.5).clamp(0, 1).permute(1, 2, 0))
    axs[1].set_title(f"Perturbation, magnified\n(each pixel moves at most {eps_show * 255:.1f}/255)", fontsize=10)
    axs[2].imshow(worst[eps_show][i].permute(1, 2, 0))
    verdict = "PASSES as good" if adv_score < thr32 else "still rejected"
    axs[2].set_title(f"Attacked: score {adv_score:.2f}\n{verdict} (threshold {thr32:.2f})", fontsize=10)
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fgsm_example.png", dpi=150)
    plt.close(fig)
    print(f"\nExample image: {len(crossed)} defective images were pushed under the threshold at "
          f"eps {eps_show} ({eps_show * 255:.1f}/255); showing one {'that was' if crossed else 'with the largest score drop'}")
    print(f"Done in {time.time() - t0:.0f}s.")


def main():
    rerun = "--rerun" in sys.argv
    reason = sys.argv[sys.argv.index("--rerun") + 1] if rerun and sys.argv.index("--rerun") + 1 < len(sys.argv) else ""
    if rerun and not reason.strip():
        raise SystemExit('Give a reason: python final_eval.py --rerun "what you fixed"')
    if LOCK.exists() and not rerun:
        raise SystemExit("The test set was already opened (results/test_opened.json). "
                         'Rerun only to fix a bug: python final_eval.py --rerun "what you fixed"')

    print("Running the leakage audit before opening the test set...")
    checks = run_checks(allow_opened=rerun)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("AUDIT FAILED: the test set stays sealed. Fix the failures and run audit_leakage.py.")
    print("AUDIT PASSED")

    frozen = load_json(CONFIG / "frozen.json")
    acc = load_json(CONFIG / "acceptance.json")
    seal = load_json(SPLITS / "test_seal.json")
    test = rows_for("test")
    fp = hashlib.sha256("".join(sorted(r["sha256"] for r in test)).encode()).hexdigest()
    if fp != seal["test_set_fingerprint"] or len(test) != seal["n_test_images"]:
        raise SystemExit("The test split no longer matches splits/test_seal.json. Stopping.")
    print(f"Test fingerprint matches the seal ({fp[:16]}...)")

    log = load_json(LOCK) if LOCK.exists() else {"openings": []}
    log["openings"].append({"at": now(), "reason": reason if rerun else "first and only planned opening",
                            "frozen_config_written_at": frozen["frozen_at"], "test_set_fingerprint": fp})
    save_json(log, LOCK)
    print(f"Opening the test set (opening number {len(log['openings'])}, logged in results/test_opened.json)\n")
    evaluate(test, RESULTS, frozen, acc)
    print("Results saved in results/. Next: python build_report.py")


if __name__ == "__main__":
    main()
