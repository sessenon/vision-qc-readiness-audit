"""Build the FP32 and INT8 detectors and freeze every decision using validation data only.

This script never loads the test split.

Outputs
  models/bank_fp32.pt, models/bank_int8.pt       memory banks
  models/backbone_fp32.onnx, backbone_int8.onnx  feature extractors for deployment
  config/frozen.json                             thresholds and settings, fixed before the test set opens
"""
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (CONFIG, INT8_WEIGHTS, MODELS, REDUCE_RANGE, THREADS, load_batch, load_json, normalize, now,
                    rows_for, save_json, setup)
from patchcore import Backbone, PatchCore, embed, greedy_coreset, nearest_distances

CORESET = 0.01
N_CALIB = 50


def ort_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_patches(sess, x01, batch=8):
    out = []
    for i in range(0, len(x01), batch):
        f2, f3 = sess.run(None, {"image": normalize(x01[i:i + batch]).numpy()})
        out.append(embed(torch.from_numpy(f2), torch.from_numpy(f3)))
    return torch.cat(out)


def onnx_scores(sess, bank, x01):
    return nearest_distances(onnx_patches(sess, x01), bank).max(1).values


class Calib(CalibrationDataReader):
    """Feeds calibration images to the quantizer one at a time."""

    def __init__(self, x01):
        self.items = iter([{"image": normalize(x01[i:i + 1]).numpy()} for i in range(len(x01))])

    def get_next(self):
        return next(self.items, None)


THRESHOLD_RULE = ("Among thresholds whose validation recall meets val_recall_target_for_threshold, choose the one "
                  "with the fewest validation false alarms; break ties by the largest margin, placing the threshold "
                  "midway between the highest-scoring good image below it and the lowest-scoring defect at or "
                  "above it. A score at or above the threshold means 'defect'.")
THRESHOLD_RULE_CHANGED = ("26 Oct 2026, before the test set was opened: the previous rule missed one validation "
                          "defect with no reduction in false alarms.")


def pick_threshold(scores, labels, recall_target):
    """Threshold that meets the recall target with the fewest false alarms, then the widest gap (see THRESHOLD_RULE)."""
    good, bad = scores[labels == 0], scores[labels == 1]
    best = None
    # Each candidate gap sits between a good score g (or nothing, for "below every good image")
    # and the lowest defect score d above it, with no other good image in between.
    for g in [None, *np.unique(good)]:
        above = bad if g is None else bad[bad > g]
        if len(above) == 0:
            continue
        d = above.min()
        if g is None and (good < d).any():
            continue                                         # a real good image sits below d; covered by that g
        if g is not None and (good > g).any() and good[good > g].min() < d:
            continue                                         # another good image sits in the gap
        thr = float(d) if g is None else float((g + d) / 2)
        margin = 0.0 if g is None else float(d - g)
        pred = scores >= thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        rec = tp / len(bad)
        if rec < recall_target:
            continue
        key = (fp, -margin)
        if best is None or key < best[0]:
            best = (key, thr, tp, fp, margin)
    _, thr, tp, fp, margin = best
    prec, rec = tp / max(tp + fp, 1), tp / len(bad)
    return thr, {"val_precision": round(prec, 4), "val_recall": round(rec, 4),
                 "val_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "val_false_alarms": fp,
                 "val_margin": round(margin, 6)}


def main():
    setup()
    t0 = time.time()
    acc = load_json(CONFIG / "acceptance.json")
    train, val = rows_for("train"), rows_for("val")

    print("Step 1/6: loading training and validation images (the test set stays sealed)...")
    x_train = load_batch([r["path"] for r in train])
    x_val = load_batch([r["path"] for r in val])
    y_val = np.array([r["label"] for r in val])
    print(f"  {len(x_train)} training images (all good), {len(x_val)} validation images "
          f"({int((y_val == 0).sum())} good, {int(y_val.sum())} defective)")

    print("Step 2/6: building the FP32 detector's memory of what good bottles look like...")
    backbone = Backbone()
    fp32 = PatchCore(backbone).fit(x_train, CORESET)
    torch.save({"bank": fp32.bank, "n_train_patches": fp32.n_train_patches}, MODELS / "bank_fp32.pt")
    print(f"  kept {len(fp32.bank)} of {fp32.n_train_patches} training patches ({time.time() - t0:.0f}s so far)")

    print("Step 3/6: exporting the feature extractor to ONNX and compressing it to INT8...")
    fp32_onnx, pre_onnx, int8_onnx = (MODELS / n for n in
                                      ("backbone_fp32.onnx", "backbone_pre.onnx", "backbone_int8.onnx"))
    torch.onnx.export(backbone, normalize(x_train[:1]), str(fp32_onnx), input_names=["image"],
                      output_names=["f2", "f3"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "batch"}, "f2": {0: "batch"}, "f3": {0: "batch"}})
    quant_pre_process(str(fp32_onnx), str(pre_onnx))
    quantize_static(str(pre_onnx), str(int8_onnx), Calib(x_train[:N_CALIB]),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8,
                    weight_type=getattr(QuantType, INT8_WEIGHTS), reduce_range=REDUCE_RANGE,
                    calibrate_method=CalibrationMethod.MinMax, op_types_to_quantize=["Conv"])
    pre_onnx.unlink()
    print(f"  INT8 weights {INT8_WEIGHTS}, reduce_range {REDUCE_RANGE}, "
          f"calibrated on the first {N_CALIB} training images")
    s32, s8 = ort_session(fp32_onnx), ort_session(int8_onnx)

    print("Step 4/6: checking the ONNX FP32 model gives the same scores as PyTorch...")
    with torch.no_grad():
        check = (onnx_scores(s32, fp32.bank, x_val[:4]) - fp32.score(x_val[:4])).abs().max().item()
    print(f"  largest score difference on 4 validation images: {check:.5f}")

    print("Step 5/6: building the INT8 detector's own memory bank from INT8 features...")
    feats8 = onnx_patches(s8, x_train).reshape(-1, 384)
    bank8 = greedy_coreset(feats8, CORESET)
    torch.save({"bank": bank8, "n_train_patches": len(feats8)}, MODELS / "bank_int8.pt")
    print(f"  kept {len(bank8)} of {len(feats8)} patches ({time.time() - t0:.0f}s so far)")

    print("Step 6/6: choosing alarm thresholds on validation data and freezing them...")
    sc32 = fp32.score_batched(x_val).numpy()
    with torch.no_grad():
        sc8 = onnx_scores(s8, bank8, x_val).numpy()
    target = acc["val_recall_target_for_threshold"]
    frozen = {"frozen_at": now(), "coreset_ratio": CORESET, "calibration_images": N_CALIB,
              "int8_weights": INT8_WEIGHTS, "reduce_range": REDUCE_RANGE,
              "calibration_source": f"first {N_CALIB} training images (good parts only)",
              "val_recall_target": target,
              "threshold_rule": THRESHOLD_RULE,
              "threshold_rule_changed": THRESHOLD_RULE_CHANGED,
              "epsilons": [0, 0.002, 0.005, 0.01, 0.02, 0.03],
              "attack_modes": ["worst_case", "evasion"],
              "pgd": {"epsilons": [0.01, 0.03], "steps": 10, "step_size": "eps/4", "mode": "worst_case"},
              "shifts": {"darker": "pixels x 0.7", "blur": "Gaussian blur radius 2",
                         "noise": "Gaussian noise sd 0.04", "jpeg": "JPEG quality 30"},
              "models": {}}
    for name, sc in (("FP32", sc32), ("INT8", sc8)):
        thr, stats = pick_threshold(sc, y_val, target)
        frozen["models"][name] = {"threshold": round(thr, 6), "val_auroc": round(roc_auc_score(y_val, sc), 4),
                                  "val_ap": round(average_precision_score(y_val, sc), 4), **stats}
        m = frozen["models"][name]
        print(f"  {name}: val AUROC {m['val_auroc']:.3f}, AP {m['val_ap']:.3f}, threshold {thr:.4f}, "
              f"recall {m['val_recall']:.2f}, precision {m['val_precision']:.2f}, "
              f"false alarms {m['val_false_alarms']}, margin {m['val_margin']:.4f}")

    gap = frozen["models"]["FP32"]["val_auroc"] - frozen["models"]["INT8"]["val_auroc"]
    if gap > 0.02:
        print(f"WARNING: INT8 validation AUROC is {gap:.3f} below FP32 (more than 0.02).")
    save_json(frozen, CONFIG / "frozen.json")
    print(f"Saved config/frozen.json. Total time {time.time() - t0:.0f}s. The test set has not been touched.")


if __name__ == "__main__":
    main()
