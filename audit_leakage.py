"""Leakage audit: run after fit.py and before final_eval.py. Writes results/leakage_audit.md.

Test files are only hashed here (read as bytes), never loaded as images or scored.
"""
import hashlib
import re
import sys

import common
from common import CONFIG, RESULTS, ROOT, SPLITS, load_json, now, read_manifest, sha256

def run_checks(allow_opened=False):
    """Run every check, print one line each, and return a list of (name, ok, detail).

    allow_opened: final_eval.py --rerun passes True, since the test set has then been opened before.
    """
    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")

    rows = read_manifest()

    # 1. Every file is unchanged since the manifest was written.
    changed = [r["path"] for r in rows if sha256(ROOT / r["path"]) != r["sha256"]]
    check("Manifest hashes match the files on disk", not changed,
          f"{len(rows)} files re-hashed, {len(changed)} changed")

    # 2. The same image content never appears in two splits.
    seen = {}
    for r in rows:
        seen.setdefault(r["sha256"], set()).add(r["split"])
    overlap = sum(1 for s in seen.values() if len(s) > 1)
    check("No image appears in more than one split", overlap == 0, f"{overlap} overlapping images")

    # 3. Training data holds good parts only.
    bad_train = sum(1 for r in rows if r["split"] == "train" and r["label"] != 0)
    check("Training split contains good parts only", bad_train == 0, f"{bad_train} defective images in train")

    # 4. Test set unchanged since it was sealed, and still unopened.
    seal = load_json(SPLITS / "test_seal.json")
    test_hashes = sorted(r["sha256"] for r in rows if r["split"] == "test")
    fp = hashlib.sha256("".join(test_hashes).encode()).hexdigest()
    check("Test set matches its seal", fp == seal["test_set_fingerprint"] and len(test_hashes) == seal["n_test_images"],
          f"{len(test_hashes)} images, fingerprint {fp[:16]}..., sealed {seal['sealed_at']}")
    opened = (RESULTS / "test_opened.json").exists()
    if allow_opened and opened:
        check("Test set not yet opened", True, "already opened; allowed because this is a logged --rerun")
    else:
        check("Test set not yet opened", not opened,
              "results/test_opened.json absent" if not opened else "results/test_opened.json exists")

    # 5. Preprocessing statistics come from the pretrained backbone, not from our data.
    check("Normalisation uses published ImageNet constants",
          common.IMAGENET_MEAN == (0.485, 0.456, 0.406) and common.IMAGENET_STD == (0.229, 0.224, 0.225),
          f"mean {common.IMAGENET_MEAN}, std {common.IMAGENET_STD}")

    # 6. The model-building code never asks for the test split.
    src = (ROOT / "fit.py").read_text(encoding="utf-8")
    uses_test = bool(re.search(r"rows_for\(\s*[\"']test[\"']", src))
    check("fit.py never loads the test split", not uses_test, 'static check of fit.py for rows_for("test")')

    # 7. Thresholds and settings were frozen from validation data.
    frozen = load_json(CONFIG / "frozen.json")
    needed = ("threshold", "val_auroc", "val_ap", "val_precision", "val_recall", "val_f1")
    missing = [f"{name}.{k}" for name, m in frozen["models"].items() for k in needed if k not in m]
    check("Every frozen model has validation results", frozen["models"] and not missing,
          f"models {', '.join(frozen['models'])}; frozen {frozen['frozen_at']}"
          + (f"; missing {', '.join(missing)}" if missing else ""))

    # 8. INT8 calibration uses training images, both as recorded and in the code.
    calib_code = bool(re.search(r"Calib\(\s*x_train\b", src))
    check("INT8 calibration images come from the training split",
          "training" in frozen["calibration_source"] and calib_code,
          f"{frozen['calibration_source']}; fit.py calibrates on x_train: {calib_code}")

    # 9. Pass marks were written before any thresholds existed.
    accept, froz = CONFIG / "acceptance.json", CONFIG / "frozen.json"
    check("Acceptance criteria older than frozen thresholds",
          accept.stat().st_mtime < froz.stat().st_mtime, "file modification times")
    return checks


def main():
    print("Running the leakage audit...")
    checks = run_checks()
    passed = all(ok for _, ok, _ in checks)
    lines = ["# Leakage audit", "", f"Run {now()}. Result: **{'PASS' if passed else 'FAIL'}**", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    (RESULTS / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved results/leakage_audit.md")
    if passed:
        print("AUDIT PASSED: the test set may now be opened once by final_eval.py")
    else:
        print("AUDIT FAILED: fix the failures before opening the test set")
        sys.exit(1)


if __name__ == "__main__":
    main()
