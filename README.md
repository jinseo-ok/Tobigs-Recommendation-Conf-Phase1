# Trivago Hotel Recommender

Team members
=================

- Cho. SY
- Bae. YN
- Oh. JS
- Lee. HJ
- Lee. JB

Best single model
=================

Below there is a process to generate best single model.

Current process (full)
----------------------

1. cd data/
2. cd ../src/recsys/data_prep
3. ./run_data_prep.sh
4. C++ feature generation
```
cd ../../../cpp
make
./build/price # extracts price features
./build/scores # extracts incremental features for each impression (|-separated format)
./build/extractor # extracts comparison features for |-separated features (extracted in scores)
python select_features.py
```
5. cd ../src/recsys/data_generator/
6. cd data_generator; python generate_data_parallel_all.py; cd .. (pypy is good)
7. python split_events_sorted_trans.py
8. python vectorize_datasets.py
9. python model_val.py (validate model)
10. python model_submit.py (make test predictions)
11. python make_blend.py (prepare submission file)


Blend
=====

settings | mrr | coef
--- | --- | ----
{"dataset_path_matrix": "data/69542a/Xcsr.h5", "dataset_path_meta": "data/69542a/meta.h5", "model_class": "LGBMRanker", "model_params": {"boosting_type": "dart", "learning_rate": 0.2, "num_leaves": 64, "min_child_samples": 5, "n_estimators": 5000, "drop_rate": 0.015, "feature_fraction": 0.7, "bagging_fraction": 0.8, "n_jobs": -2}} | 0.69068 | 0.51223
