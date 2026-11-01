# Leakage audit

Run 2026-10-26T17:55:12+00:00. Result: **PASS**

| check | result | detail |
|---|---|---|
| Manifest hashes match the files on disk | PASS | 292 files re-hashed, 0 changed |
| No image appears in more than one split | PASS | 0 overlapping images |
| Training split contains good parts only | PASS | 0 defective images in train |
| Test set matches its seal | PASS | 70 images, fingerprint 2b91c8e8be1c3df7..., sealed 2026-10-26T16:30:29+00:00 |
| Test set not yet opened | PASS | results/test_opened.json absent |
| Normalisation uses published ImageNet constants | PASS | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| fit.py never loads the test split | PASS | static check of fit.py for rows_for("test") |
| Every frozen model has validation results | PASS | models FP32, INT8; frozen 2026-10-26T17:53:39+00:00 |
| INT8 calibration images come from the training split | PASS | first 50 training images (good parts only); fit.py calibrates on x_train: True |
| Acceptance criteria older than frozen thresholds | PASS | file modification times |
