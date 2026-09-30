# Frozen Zero-Shot Comparison

Laya is `convaiinnovations/laya` (`typed-decisions`). Jev is pinned `jev-1.13.0` via the TypeSafe API. Kev is `jaredpalmer/kev-0.8b` served locally over `/v1/systemone`. Majority is an a priori class prior and does not peek at evaluation labels. Compatibility, query, and brand/category use dual-judge field consensus; disagreements are withheld. ESCI is a balanced 8,000-pair human US test slice (2,000 per E/S/C/I).

| Field | majority acc | majority macro-F1 | majority ECE | laya acc | laya macro-F1 | laya ECE | jev acc | jev macro-F1 | jev ECE | kev acc | kev macro-F1 | kev ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Compatibility | 70.9% | 0.207 | 0.291 | 20.8% | 0.151 | 0.177 | 74.0% | 0.569 | 0.068 | 15.8% | 0.133 | 0.053 |
| Query goal | 42.4% | 0.099 | 0.576 | 47.9% | 0.328 | 0.348 | 79.3% | 0.762 | 0.032 | 27.6% | 0.306 | 0.070 |
| Query object | 38.3% | 0.111 | 0.617 | 30.2% | 0.215 | 0.263 | 79.6% | 0.703 | 0.033 | 37.8% | 0.310 | 0.196 |
| Query specificity | 31.9% | 0.081 | 0.681 | 41.0% | 0.299 | 0.331 | 80.6% | 0.771 | 0.056 | 19.4% | 0.257 | 0.063 |
| Query commerce scope | 51.3% | 0.226 | 0.487 | 53.5% | 0.438 | 0.512 | 93.1% | 0.884 | 0.081 | 28.3% | 0.285 | 0.139 |
| Brand intent | 52.3% | 0.172 | 0.477 | 41.6% | 0.164 | 0.367 | 94.6% | 0.688 | 0.071 | 36.2% | 0.262 | 0.174 |
| Brand choice | 56.3% | 0.080 | 0.437 | 45.4% | 0.393 | 0.429 | 96.4% | 0.948 | 0.035 | 89.8% | 0.852 | 0.501 |
| Category choice | 41.4% | 0.065 | 0.586 | 31.9% | 0.185 | 0.288 | 91.3% | 0.921 | 0.011 | 72.9% | 0.751 | 0.417 |
| Multi-target | 100.0% | 0.500 | 0.001 | 1.2% | 0.012 | 0.553 | 99.0% | 0.543 | 0.112 | 80.7% | 0.449 | 0.219 |
| ESCI (human slice) | 25.0% | 0.100 | 0.750 | 25.5% | 0.184 | 0.217 | 60.4% | 0.603 | 0.105 | 30.9% | 0.226 | 0.114 |

## Serving

- **majority compatibility:** p50 0.0 ms; nan items/s
- **majority query_segmentation:** p50 0.0 ms; nan items/s
- **majority brand_category:** p50 0.0 ms; nan items/s
- **majority esci:** p50 0.0 ms; nan items/s
- **laya compatibility:** p50 35.9 ms; 26.6 items/s; VRAM reserved 2724 MB
- **laya query_segmentation:** p50 95.4 ms; 10.4 items/s; VRAM reserved 3182 MB
- **laya brand_category:** p50 178.9 ms; 5.6 items/s; VRAM reserved 3730 MB
- **laya esci:** p50 20.5 ms; 47.7 items/s; VRAM reserved 2734 MB
- **jev compatibility:** p50 1122.8 ms; 0.9 items/s; $0.0609
- **jev query_segmentation:** p50 1192.6 ms; 0.8 items/s; $0.2030
- **jev brand_category:** p50 1183.6 ms; 0.8 items/s; $0.1273
- **jev esci:** p50 1160.4 ms; 0.8 items/s; $0.1694
- **kev compatibility:** p50 954.7 ms; 1.0 items/s; VRAM reserved 5416 MB
- **kev query_segmentation:** p50 1836.4 ms; 0.5 items/s; VRAM reserved 7874 MB
- **kev brand_category:** p50 1103.4 ms; 0.8 items/s; VRAM reserved 7896 MB
- **kev esci:** p50 867.6 ms; 1.1 items/s; VRAM reserved 7906 MB

## Reading the numbers

- Majority wins accuracy on imbalanced tasks because `incompatible` and `multi_target=false` dominate the consensus labels.
- Laya is fast locally but is not a strong frozen zero-shot classifier on these commerce schemas.
- Jev is the hosted System One reference. Compare it to Laya and Kev on the same frozen questions and compact states, not to supervised controls.
- Kev-0.8B is the first open TypeSafe-compatible checkpoint in this drop. It is much weaker than hosted Jev on compatibility, query axes, and ESCI, but brand and category choice are already usable.
- JevLite and larger Kev checkpoints have adapters and a Colab notebook in this package, but those full-slice runs are not part of this published result drop.
