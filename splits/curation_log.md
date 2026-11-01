# Curation log

Generated 2026-10-26T16:30:29+00:00 by make_splits.py (seed 0).

Source: data/bottle

| split | type | images |
|---|---|---|
| test | broken_large | 12 |
| test | broken_small | 13 |
| test | contamination | 13 |
| test | good | 32 |
| train | good | 169 |
| val | broken_large | 8 |
| val | broken_small | 9 |
| val | contamination | 8 |
| val | good | 28 |

## Decisions
- Holdout: 40 train/good images moved out of training (20 to val, 20 to test) because the dataset's own test folder has too few good images to estimate false alarms.
- Stratified split: each test folder is split 40/60 into val/test per folder (rounded), so every defect type appears in both.
- Good-only training: the detector learns what normal looks like from good images only.
- ImageNet normalisation: resize 256, centre crop 224, published ImageNet mean/std from the backbone's pretraining. No statistic is computed from our images.
- No exclusions: every image was kept. Exact duplicates across splits are checked by SHA-256 (none found).
- Licence: MVTec AD is released under CC BY-NC-SA 4.0 (non-commercial); used here for coursework only.
- Source: MVTec AD bottle category, downloaded from the Kaggle mirror ipythonx/mvtec-ad on 26 October 2026 because MVTec's download server was unavailable. Licence CC BY-NC-SA 4.0, coursework use only.
- Class balance: the test split is defect-heavy (32 good, 38 defective), so results are reported as AUROC and AP, plus precision and recall at a threshold fixed on validation, never as plain accuracy.
