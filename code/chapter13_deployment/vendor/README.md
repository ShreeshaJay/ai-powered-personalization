# Vendored published code

These files are byte-identical copies of the book's published companion code
for Chapters 5 to 8, taken from the public repository
<https://github.com/ShreeshaJay/ai-powered-personalization> at the commit
recorded in `MANIFEST.json` (`source_commit`). The services serve these
models and functions as published; nothing here is edited.

`MANIFEST.json` also lists every file with its path in the public repository,
its SHA-256 and its size.

| Vendored here | Used for |
|---|---|
| `chapter5_retrieval/supervised_two_tower.py` | two-tower model, training loop, `SupervisedRetriever` (reference in the tests) |
| `chapter6_ranking/models/historical_features.py` | `HistoricalFeatureBuilder`: the features in the online store |
| `chapter6_ranking/models/xgboost_ranker.py`, `feature_pipeline.py`, `feature_encoder.py` | lightweight ranker; derived time features |
| `chapter7_advanced_ranking/models/mmoe.py`, `train_mmoe.py` | heavy ranker (MMoE, `MMoEFeatureProcessor`, training and evaluation loop) |
| `chapter8_value_functions/value_function.py`, `mmr_diversity.py`, `business_rules.py` | value function, MMR diversity, artist pacing |

The remaining files (`deepfm.py`, `dcn.py`, `dlrm.py`, `polars_pipeline.py`,
the package `__init__.py` files, `model_interface.py`, `pipeline.py`) are
imported by the files above or by their packages and are copied so the
imports resolve unchanged. `common/published.py` loads the modules under
aliases (`ch6_models`, `ch7_models`, `ch8_value_functions`) without running
the package `__init__.py` files, so the services that do not need PyTorch
never import it.

Check the copies:

```bash
python scripts/check_vendor.py                                       # hashes still match MANIFEST.json
python scripts/check_vendor.py --repo ../ai-powered-personalization  # and match the published commit
```

Refresh after the published code changes, then rebuild the artifact bundles
and rerun the tests:

```bash
python scripts/vendor_published_code.py --repo ../ai-powered-personalization --ref <commit>
python -m pipeline_build.build_artifacts --profile synthetic
pytest
```

Every bundle records the SHA-256 of each vendored file it was built with, and
each service compares those hashes with the vendored files it runs before it
reports ready. A bundle built before a refresh therefore stops with a message
naming the changed files, instead of serving features computed by code other
than the code the models were trained with.
