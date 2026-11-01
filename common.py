"""Shared settings and helpers for the Week 7 capstone (MedVision anomaly detection)."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
CATEGORY = "bottle"
DATA = ROOT / "data" / CATEGORY
SPLITS = ROOT / "splits"
MODELS = ROOT / "models"
CONFIG = ROOT / "config"
RESULTS = ROOT / "results"
for d in (SPLITS, MODELS, CONFIG, RESULTS):
    d.mkdir(exist_ok=True)

SEED = 0
THREADS = 2

# INT8 weight type for ONNX Runtime quantization. Signed weights are ONNX Runtime's default.
# This i7-10875H has AVX2 but no VNNI, where 8-bit integer multiply-adds can saturate, so
# REDUCE_RANGE quantizes weights to 7 bits, as ONNX Runtime's docs recommend for such chips.
INT8_WEIGHTS = "QInt8"
REDUCE_RANGE = True

RESIZE, CROP = 256, 224
# Normalisation constants are the published values from the backbone's ImageNet pretraining,
# not statistics of our images, so no split can leak into preprocessing.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def setup():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(THREADS)


def load_image(path):
    """File -> float tensor 3x224x224 with pixel values in [0, 1] (not yet normalised)."""
    img = Image.open(path).convert("RGB").resize((RESIZE, RESIZE), Image.BILINEAR)
    off = (RESIZE - CROP) // 2
    img = img.crop((off, off, off + CROP, off + CROP))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def load_batch(paths):
    return torch.stack([load_image(p) for p in paths])


def normalize(x01):
    mean = torch.tensor(IMAGENET_MEAN, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


def read_manifest():
    """Rows of splits/manifest.csv as dicts; label is 0 for good, 1 for defect."""
    with open(SPLITS / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def rows_for(split, rows=None):
    rows = rows if rows is not None else read_manifest()
    return [r for r in rows if r["split"] == split]
