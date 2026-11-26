| Metric | YOLOv8-s | YOLO26-s |
|---|---|---|
| Params (M) | 11.15 | 9.98 |
| GFLOPs | 28.7 | 22.9 |
| Full test mAP@50 | 74.7 | 79.1 |
| Clean test mAP@50 | 71.2 | 75.9 |
| Clean test mAP@50:95 | 54.4 | 59.2 |
| mPC seen (mAP@50) | 41.0 | 44.2 |
| mPC held-out (mAP@50) | 51.0 | 56.5 |
| mPC all (mAP@50) | 43.2 | 46.9 |
| rPC all (mAP@50) | 60.7 | 61.8 |
| mPC all (mAP@50:95) | 30.7 | 33.6 |
| rPC all (mAP@50:95) | 56.4 | 56.8 |
| **Abstention, risk on all detections** (false alarm = error) | | |
| τ (clean val, r*=5%) | – | 0.987 |
| Coverage, clean val | 0.0 | 0.2 |
| Selective risk, clean val | 0.0 | 0.0 |
| AURC, clean val | 0.148 | 0.126 |
| Accepted false alarms / 100 images, clean val | 0.00 | 0.00 |
| Coverage, clean test | 0.0 | 0.3 |
| Selective risk, clean test | 0.0 | 0.0 |
| AURC, clean test | 0.119 | 0.090 |
| Accepted false alarms / 100 images, clean test | 0.00 | 0.00 |
| Coverage, Aircraft-C | 0.0 | 0.1 |
| Selective risk, Aircraft-C | 0.0 | 7.1 |
| AURC, Aircraft-C | 0.202 | 0.156 |
| Accepted false alarms / 100 images, Aircraft-C | 0.00 | 0.01 |
| **Abstention, risk on matched detections** (affiliation risk on detected aircraft) | | |
| τ (clean val, r*=5%) | 0.919 | 0.987 |
| Coverage, clean val | 15.9 | 0.2 |
| Selective risk, clean val | 5.1 | 0.0 |
| AURC, clean val | 0.077 | 0.086 |
| Accepted false alarms / 100 images, clean val | 1.01 | 0.00 |
| Coverage, clean test | 17.7 | 0.3 |
| Selective risk, clean test | 3.1 | 0.0 |
| AURC, clean test | 0.058 | 0.052 |
| Accepted false alarms / 100 images, clean test | 0.55 | 0.00 |
| Coverage, Aircraft-C | 10.2 | 0.2 |
| Selective risk, Aircraft-C | 5.4 | 1.9 |
| AURC, Aircraft-C | 0.119 | 0.095 |
| Accepted false alarms / 100 images, Aircraft-C | 0.31 | 0.01 |
| **Speed** (FP16, batch 1) | | |
| FP16 forward (ms) | 6.64 | 13.31 |
| FP16 end-to-end (ms) | 15.39 | 18.84 |
| FPS end-to-end | 65.0 | 53.1 |

### Confident affiliation errors (conf ≥ 0.9, clean splits; diagnose_abstain.py)

| Model | Split | Dets ≥ conf (definite) | Errors | (a1) FA, more dets than GT | (a2) FA, other [in images with stacked GT] | (b) wrong affiliation [GT other_aircraft] |
|---|---|---|---|---|---|---|
| YOLOv8-s | val | 1028 (658) | 68 | 1 | 27 [24] | 40 [24] |
| YOLOv8-s | test | 548 (349) | 20 | 0 | 8 [4] | 12 [7] |
| YOLO26-s | val | 1105 (737) | 71 | 0 | 23 [16] | 48 [32] |
| YOLO26-s | test | 597 (396) | 18 | 0 | 5 [1] | 13 [10] |

Most frequent (b) pairs:

- YOLOv8-s (val + test), predicted → GT: Rafale → other_aircraft (7), F-16 → other_aircraft (6), Rafale → F-16 (5), C-17 → Y-20 (4), J-20 → other_aircraft (4), C-17 → other_aircraft (4), Y-20 → other_aircraft (2), Rafale → JF-17 (2)
- YOLO26-s (val + test), predicted → GT: Rafale → other_aircraft (11), Rafale → F-16 (8), C-17 → other_aircraft (7), J-20 → other_aircraft (6), J-10 → other_aircraft (5), F-16 → MiG-29 (3), JF-17 → other_aircraft (3), F-16 → other_aircraft (3)
