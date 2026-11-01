"""Build the leakage-free splits and seal the test set before any model code runs.

train : good images only (the memory bank is built from these)
val   : good + defective images, used for every decision (threshold, settings)
test  : good + defective images, sealed now and opened exactly once by final_eval.py
"""
import csv
import hashlib
import random
from collections import Counter

from common import DATA, ROOT, SEED, SPLITS, now, save_json, sha256

HOLDOUT_GOOD = 40     # train/good images moved out so val and test have enough good examples
VAL_SHARE = 0.4       # share of each test/ folder that goes to validation


def main():
    if not DATA.exists():
        raise SystemExit(f"Data folder not found: {DATA}")
    rng = random.Random(SEED)
    rows = []

    print("Shuffling training images and holding some out...")
    train_good = sorted((DATA / "train" / "good").glob("*.png"))
    rng.shuffle(train_good)
    extra = train_good[:HOLDOUT_GOOD]
    half = HOLDOUT_GOOD // 2
    for p in train_good[HOLDOUT_GOOD:]:
        rows.append((p, "train", 0, "good", "train/good"))
    for p in extra[:half]:
        rows.append((p, "val", 0, "good", "train/good (held out)"))
    for p in extra[half:]:
        rows.append((p, "test", 0, "good", "train/good (held out)"))

    print("Splitting each test folder 40/60 into validation and test...")
    for type_dir in sorted(d for d in (DATA / "test").iterdir() if d.is_dir()):
        files = sorted(type_dir.glob("*.png"))
        rng.shuffle(files)
        n_val = round(len(files) * VAL_SHARE)
        label = 0 if type_dir.name == "good" else 1
        for i, p in enumerate(files):
            rows.append((p, "val" if i < n_val else "test", label, type_dir.name, f"test/{type_dir.name}"))

    print(f"Fingerprinting {len(rows)} images...")
    out = []
    for p, split, label, kind, source in rows:
        out.append({"path": p.relative_to(ROOT).as_posix(), "split": split, "label": label,
                    "defect_type": kind, "source_folder": source, "sha256": sha256(p)})

    # Leakage guard: the same image content may not appear in two splits.
    by_hash = {}
    for r in out:
        by_hash.setdefault(r["sha256"], set()).add(r["split"])
    clash = [h for h, s in by_hash.items() if len(s) > 1]
    assert not clash, f"{len(clash)} identical images appear in more than one split"
    print("Check passed: no image appears in more than one split.")

    with open(SPLITS / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    test_hashes = sorted(r["sha256"] for r in out if r["split"] == "test")
    seal = hashlib.sha256("".join(test_hashes).encode()).hexdigest()
    save_json({"sealed_at": now(), "n_test_images": len(test_hashes), "test_set_fingerprint": seal,
               "rule": "Test images are scored once, by final_eval.py."},
              SPLITS / "test_seal.json")

    counts = Counter((r["split"], r["defect_type"]) for r in out)
    test_good = counts[("test", "good")]
    test_bad = sum(n for (s, k), n in counts.items() if s == "test" and k != "good")
    lines = ["# Curation log", "", f"Generated {now()} by make_splits.py (seed {SEED}).", "",
             f"Source: {DATA.relative_to(ROOT).as_posix()}", "",
             "| split | type | images |", "|---|---|---|"]
    for (split, kind), n in sorted(counts.items()):
        lines.append(f"| {split} | {kind} | {n} |")
    lines += ["", "## Decisions",
              f"- Holdout: {HOLDOUT_GOOD} train/good images moved out of training ({half} to val, {half} to test) "
              "because the dataset's own test folder has too few good images to estimate false alarms.",
              f"- Stratified split: each test folder is split {int(VAL_SHARE * 100)}/{int((1 - VAL_SHARE) * 100)} "
              "into val/test per folder (rounded), so every defect type appears in both.",
              "- Good-only training: the detector learns what normal looks like from good images only.",
              "- ImageNet normalisation: resize 256, centre crop 224, published ImageNet mean/std from the "
              "backbone's pretraining. No statistic is computed from our images.",
              "- No exclusions: every image was kept. Exact duplicates across splits are checked by SHA-256 (none found).",
              "- Licence: MVTec AD is released under CC BY-NC-SA 4.0 (non-commercial); used here for coursework only.",
              "- Source: MVTec AD bottle category, downloaded from the Kaggle mirror ipythonx/mvtec-ad on 26 October 2026 "
              "because MVTec's download server was unavailable. Licence CC BY-NC-SA 4.0, coursework use only.",
              f"- Class balance: the test split is defect-heavy ({test_good} good, {test_bad} defective), so results "
              "are reported as AUROC and AP, plus precision and recall at a threshold fixed on validation, "
              "never as plain accuracy."]
    (SPLITS / "curation_log.md").write_text("\n".join(lines) + "\n")

    for split in ("train", "val", "test"):
        good = counts[(split, "good")]
        bad = sum(n for (s, k), n in counts.items() if s == split and k != "good")
        print(f"{split:5s}: {good:3d} good, {bad:3d} defective")
    print(f"Test set sealed: {len(test_hashes)} images, fingerprint {seal[:16]}...")


if __name__ == "__main__":
    main()
