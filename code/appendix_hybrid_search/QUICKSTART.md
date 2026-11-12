# Quick start

Work from this directory. Use a virtual environment named `venv`. On Windows the interpreter is `venv\Scripts\python`. On macOS and Linux it is `venv/bin/python`.

If the machine has no CUDA build of PyTorch, install a CPU wheel from the PyTorch CPU index first. Then install the pinned libraries without replacing an existing torch build:

```bash
python -m pip install -r requirements.txt --upgrade-strategy only-if-needed
```

Put `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet` in one directory. The dataset licence is Apache-2.0. Point the scripts at that directory with `--data-dir` or with the environment variable `ESCI_DATA_DIR`.

Run the tests, then the teaching slice. The slice times in `README.md` are teaching measurements on an 8-product fixture.

```bash
python -m unittest discover -s tests -v
python run_cpu_slice.py --output-dir outputs/cpu_slice --device cpu
```

On a Colab L4 runtime that already provides torch `2.14.1+cu130`, keep that build and run the full catalog. The GPU hours in `README.md` are planning estimates.

```bash
pip install -r requirements.txt --upgrade-strategy only-if-needed
python run_pipeline.py --stages s1,s2 --data-dir /content/esci --output-dir outputs --device auto --prefix-source tasksource/esci
python run_pipeline.py --stages s5 --data-dir /content/esci --output-dir outputs --device auto
```

Add `--sync-cmd` when you want a command to run after each stage. Use the placeholders `{stage}`, `{step}`, and `{output}`. Optional stages are `s3` for the cross-encoder and `s4` for the fine-tune.

Exercise 1 writes the 1,000-query slice:

```bash
python exercises/ex1_slice.py --data-dir /path/to/esci --output outputs/splits/exercise_slice.json
```

Exercise 2 checks Qdrant local mode. With no matrices supplied, it uses a small random collection:

```bash
python exercises/ex2_parity.py --output outputs/qdrant_parity.json
```

Naver SPLADE vectors are CC BY-NC-SA 4.0 and stay an evaluation cache. Do not redistribute them.
