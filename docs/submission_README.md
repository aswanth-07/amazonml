# Business Entity Resolution: reproduction

This folder regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the challenge's training and test files. It
needs nothing beyond what is in this folder and the data: no network access,
no external lookup and no pretrained weights.

## Environment

Python 3.10 (the version tested), CPU only. The run takes a few minutes on a laptop.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout

```
<DATA_DIR>/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## Run end to end

```bash
python src/main.py run --data-dir <DATA_DIR> --out-dir output
```

This single command:

1. normalises all three sources of train and test
2. generates candidates (TF-IDF nearest neighbours, key blocks, a capped union)
3. builds pair features
4. trains LightGBM with 5-fold grouped cross-fitting and tunes the decision
   threshold for macro F0.5 on out-of-fold predictions
5. refits on all training pairs
6. predicts test, then writes and validates both TSVs

Metrics, the resolved config and error tables go to `runs/<timestamp>-run/`.
The trained model goes to `artifacts/model.joblib`.

Other commands:

```bash
python src/main.py cv --data-dir <DATA_DIR>          # out-of-fold evaluation on train only
python src/main.py audit --data-dir <DATA_DIR>       # data contract and assumption checks
python src/main.py validate --test-dir <DATA_DIR>/test
python src/main.py predict --model artifacts/model.joblib --data-dir <DATA_DIR> --out-dir output
```

Seeds are fixed (config `seed: 42`) and LightGBM runs in deterministic mode,
so repeated runs give identical outputs.

## Source map

| Path | Stage |
|---|---|
| `src/ber/normalize/` | name, address and country normalisation |
| `src/ber/blocking/` | candidate generation; its capped output is `candidate_pairs.tsv` |
| `src/ber/features.py` | pair features |
| `src/ber/matcher.py` | LightGBM with grouped cross-fitting |
| `src/ber/decision.py` | exclusive assignment and the tuned operating point |
| `src/ber/metrics.py` | the leaderboard metric (macro F0.5) and blocking diagnostics |
| `src/ber/pipeline.py` | orchestration |
| `src/ber/configs/default.yaml` | every tunable, with its default |
