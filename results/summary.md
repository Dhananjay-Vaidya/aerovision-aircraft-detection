# Final results (clean splits R1–R3; original labels)

Percentages for mAP, precision, recall, mPC, rPC, coverage, risk, accuracy and AURC. τ is a confidence threshold chosen on clean val (maximum coverage with val risk ≤ r*). Accepted FA / 100 img = accepted detections that match no GT box, per 100 images. – = not available.

Speed caveat: the laptop GPU (RTX 3050 Laptop) varies by up to ~50 % between runs of the same protocol, and the rounds of different models were not all timed in one session. The three YOLO26-s models have the same architecture (9.48 M params, 20.8 GFLOPs fused), so speed differences between them are measurement noise, not model differences.

| Metric | YOLOv8-s | YOLO26-s | YOLO26-s + plain FT | YOLO26-s + degradation FT |
|---|---:|---:|---:|---:|
| **Test (no corruption)** | | | | |
| mAP@50, clean test | 73.8 | 79.0 | 79.2 | 80.1 |
| mAP@50:95, clean test | 56.9 | 62.1 | 62.5 | 63.3 |
| Precision, clean test | 76.4 | 82.3 | 82.8 | 86.9 |
| Recall, clean test | 66.9 | 72.2 | 72.7 | 71.9 |
| Images, clean test | 1071 | 1071 | 1071 | 1071 |
| mAP@50, full test | 74.7 | 79.1 | 79.3 | 80.3 |
| mAP@50:95, full test | 58.7 | 63.3 | 64.0 | 64.8 |
| Precision, full test | 77.2 | 82.8 | 83.1 | 86.2 |
| Recall, full test | 69.4 | 74.0 | 73.8 | 73.9 |
| Images, full test | 1286 | 1286 | 1286 | 1286 |
| Params (M, fused) | 11.14 | 9.48 | 9.48 | 9.48 |
| GFLOPs (fused) | 28.5 | 20.8 | 20.8 | 20.8 |
| **Aircraft-C (clean test file names)** | | | | |
| Clean mAP@50 | 73.8 | 79.0 | 79.2 | 80.1 |
| mPC seen (mAP@50) | 42.4 | 46.0 | 45.6 | 73.0 |
| mPC held-out (mAP@50) | 52.6 | 58.6 | 57.7 | 68.5 |
| mPC all (mAP@50) | 44.7 | 48.8 | 48.3 | 72.0 |
| rPC = mPC all / clean (mAP@50) | 60.5 | 61.7 | 61.0 | 89.9 |
| Clean mAP@50:95 | 56.9 | 62.1 | 62.5 | 63.3 |
| mPC seen (mAP@50:95) | 30.2 | 33.1 | 33.0 | 55.1 |
| mPC held-out (mAP@50:95) | 38.4 | 43.1 | 42.8 | 51.1 |
| mPC all (mAP@50:95) | 32.0 | 35.3 | 35.1 | 54.2 |
| rPC = mPC all / clean (mAP@50:95) | 56.3 | 56.9 | 56.2 | 85.6 |
| mPC Defocus blur (mAP@50) | 48.9 | 54.8 | 56.2 | 71.9 |
| mPC Motion blur (mAP@50) | 33.7 | 39.9 | 40.5 | 67.8 |
| mPC Fog (mAP@50) | 67.3 | 75.1 | 74.6 | 79.3 |
| mPC Low light (mAP@50) | 39.7 | 41.6 | 41.6 | 73.8 |
| mPC Gaussian noise (mAP@50) | 22.9 | 21.6 | 20.2 | 68.3 |
| mPC JPEG (mAP@50) | 51.2 | 56.2 | 55.7 | 75.9 |
| mPC Downscale (mAP@50) | 33.1 | 32.5 | 30.4 | 73.9 |
| mPC Contrast † (mAP@50) | 60.6 | 66.7 | 64.9 | 76.2 |
| mPC Rain † (mAP@50) | 44.7 | 50.5 | 50.5 | 60.9 |
| **Abstention, risk on all detections** | | | | |
| AURC, val | 11.1 | 10.6 | 7.9 | 7.8 |
| AURC, clean test | 9.7 | 7.2 | 6.1 | 6.2 |
| AURC, Aircraft-C | 18.6 | 14.2 | 14.6 | 9.0 |
| τ (val, r*=5%) | 0.928 | 0.987 | 0.955 | 0.927 |
| Coverage, val (r*=5%) | 11.7 | 0.2 | 6.8 | 13.8 |
| Selective risk, val (r*=5%) | 5.1 | 0.0 | 5.4 | 5.2 |
| Affiliation acc., val (r*=5%) | 94.9 | 100.0 | 94.6 | 94.8 |
| Accepted FA / 100 img, val (r*=5%) | 0.05 | 0.00 | 0.00 | 0.09 |
| Coverage, clean test (r*=5%) | 13.0 | 0.3 | 8.1 | 15.3 |
| Selective risk, clean test (r*=5%) | 2.8 | 0.0 | 0.0 | 2.2 |
| Affiliation acc., clean test (r*=5%) | 97.2 | 100.0 | 100.0 | 97.8 |
| Accepted FA / 100 img, clean test (r*=5%) | 0.00 | 0.00 | 0.00 | 0.00 |
| Coverage, Aircraft-C (r*=5%) | 7.1 | 0.1 | 4.2 | 10.9 |
| Selective risk, Aircraft-C (r*=5%) | 4.9 | 7.1 | 2.9 | 3.2 |
| Affiliation acc., Aircraft-C (r*=5%) | 95.1 | 92.9 | 97.1 | 96.8 |
| Accepted FA / 100 img, Aircraft-C (r*=5%) | 0.07 | 0.01 | 0.02 | 0.00 |
| τ (val, r*=10%) | 0.764 | 0.686 | 0.619 | 0.622 |
| Coverage, val (r*=10%) | 33.4 | 42.8 | 49.4 | 49.2 |
| Selective risk, val (r*=10%) | 10.1 | 10.0 | 10.0 | 10.0 |
| Affiliation acc., val (r*=10%) | 89.9 | 90.0 | 90.0 | 90.0 |
| Accepted FA / 100 img, val (r*=10%) | 0.45 | 1.04 | 1.40 | 1.40 |
| Coverage, clean test (r*=10%) | 30.8 | 43.0 | 48.8 | 47.1 |
| Selective risk, clean test (r*=10%) | 7.2 | 8.5 | 10.0 | 9.3 |
| Affiliation acc., clean test (r*=10%) | 92.8 | 91.5 | 90.0 | 90.7 |
| Accepted FA / 100 img, clean test (r*=10%) | 0.37 | 0.93 | 1.59 | 1.68 |
| Coverage, Aircraft-C (r*=10%) | 22.4 | 30.1 | 36.0 | 42.1 |
| Selective risk, Aircraft-C (r*=10%) | 14.5 | 13.4 | 16.6 | 11.1 |
| Affiliation acc., Aircraft-C (r*=10%) | 85.5 | 86.6 | 83.4 | 88.9 |
| Accepted FA / 100 img, Aircraft-C (r*=10%) | 0.86 | 1.06 | 2.01 | 1.56 |
| **Abstention, risk on matched detections** | | | | |
| AURC, val | 7.8 | 8.9 | 6.2 | 6.0 |
| AURC, clean test | 5.7 | 5.2 | 4.5 | 4.4 |
| AURC, Aircraft-C | 11.8 | 9.5 | 9.9 | 6.2 |
| τ (val, r*=5%) | 0.925 | 0.987 | 0.955 | 0.924 |
| Coverage, val (r*=5%) | 14.6 | 0.3 | 7.6 | 16.3 |
| Selective risk, val (r*=5%) | 5.1 | 0.0 | 5.4 | 5.2 |
| Affiliation acc., val (r*=5%) | 94.9 | 100.0 | 94.6 | 94.8 |
| Accepted FA / 100 img, val (r*=5%) | 0.05 | 0.00 | 0.00 | 0.09 |
| Coverage, clean test (r*=5%) | 16.8 | 0.3 | 9.0 | 17.9 |
| Selective risk, clean test (r*=5%) | 2.6 | 0.0 | 0.0 | 2.1 |
| Affiliation acc., clean test (r*=5%) | 97.4 | 100.0 | 100.0 | 97.9 |
| Accepted FA / 100 img, clean test (r*=5%) | 0.00 | 0.00 | 0.00 | 0.00 |
| Coverage, Aircraft-C (r*=5%) | 9.6 | 0.2 | 5.0 | 13.6 |
| Selective risk, Aircraft-C (r*=5%) | 4.7 | 1.9 | 2.4 | 3.3 |
| Affiliation acc., Aircraft-C (r*=5%) | 95.3 | 98.1 | 97.6 | 96.7 |
| Accepted FA / 100 img, Aircraft-C (r*=5%) | 0.08 | 0.01 | 0.02 | 0.00 |
| τ (val, r*=10%) | 0.683 | 0.589 | 0.442 | 0.475 |
| Coverage, val (r*=10%) | 47.2 | 54.1 | 60.2 | 59.1 |
| Selective risk, val (r*=10%) | 10.0 | 10.0 | 10.0 | 10.0 |
| Affiliation acc., val (r*=10%) | 90.0 | 90.0 | 90.0 | 90.0 |
| Accepted FA / 100 img, val (r*=10%) | 0.86 | 2.07 | 4.50 | 2.97 |
| Coverage, clean test (r*=10%) | 45.3 | 53.2 | 58.9 | 56.6 |
| Selective risk, clean test (r*=10%) | 8.9 | 8.7 | 10.5 | 8.6 |
| Affiliation acc., clean test (r*=10%) | 91.1 | 91.3 | 89.5 | 91.4 |
| Accepted FA / 100 img, clean test (r*=10%) | 1.40 | 1.68 | 4.11 | 4.01 |
| Coverage, Aircraft-C (r*=10%) | 34.5 | 41.2 | 49.8 | 54.7 |
| Selective risk, Aircraft-C (r*=10%) | 14.1 | 13.1 | 16.3 | 11.3 |
| Affiliation acc., Aircraft-C (r*=10%) | 85.9 | 86.9 | 83.7 | 88.7 |
| Accepted FA / 100 img, Aircraft-C (r*=10%) | 1.63 | 1.93 | 4.80 | 3.85 |
| **Speed (RTX 3050 Laptop, FP16, batch 1, 640 px; 50 warm-up + 300 timed; median over repeated runs, see results/speed_rounds.csv)** | | | | |
| Forward ms mean | 8.3 | 13.3 | 14.9 | 8.5 |
| Forward ms p95 | 14.0 | 25.0 | 24.0 | 13.6 |
| Forward FPS (mean) | 120.9 | 74.9 | 67.0 | 117.2 |
| Forward FPS (at p95 latency) | 71.4 | 40.1 | 41.7 | 73.4 |
| End-to-end ms mean | 13.6 | 21.5 | 19.8 | 11.7 |
| End-to-end ms p95 | 20.9 | 31.2 | 29.8 | 18.9 |
| End-to-end FPS (mean) | 73.5 | 46.4 | 50.5 | 85.6 |
| End-to-end FPS (at p95 latency) | 47.9 | 32.1 | 33.6 | 53.0 |
