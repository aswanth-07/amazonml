# Business Entity Resolution: reproduction

This folder regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the challenge's training and test files. It
needs nothing beyond what is in this folder and the data: no network access,
no external lookup, no pretrained weights, no GPU.

## Environment

Python 3.10 (the version tested) on 64-bit Windows or Linux.

| Resource | Needed |
|---|---|
| Memory | 32 GB (about 20 GB free during the run) |
| Cores | 16 or more recommended |
| Wall time | about 2 hours on a 24-thread laptop CPU |

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
python src/main.py run --config src/ber/configs/full.yaml --data-dir <DATA_DIR> --out-dir output
```

That one command runs these stages:

1. **Normalise** all six source files in parallel. Tables are cached in
   `artifacts/cache/`.
2. **Block train.** Hashed TF-IDF name and address views. Per-country
   inverted-index search on each record's rarest tokens, taking name, address
   and combined top-m. Every generated pair is re-scored with the full
   cosines, then each entity's best are selected, plus a reverse search.
3. **Stage 1.** Pair features on the candidates of 250,000 sampled entities,
   then LightGBM with 4 folds grouped by entity. The other training entities
   are scored by the fold average.
4. **Stage 2.** A stacked LightGBM on stage-1 probabilities in their entity
   and record context.
5. **Decide.** Exclusive assignment, with the threshold tuned for macro F0.5.
6. **Block and score test** the same way, then write both TSVs and validate
   them.

Metrics go to `runs/<timestamp>-run/`: blocking recall, out-of-fold and
holdout F0.5, per-country breakdown, feature importance, and error samples.
The trained models go to `artifacts/scale_model.joblib`.

To re-predict test with a saved model:

```bash
python src/main.py predict --model artifacts/scale_model.joblib --data-dir <DATA_DIR> --out-dir output
```

Seeds are fixed and LightGBM runs in deterministic mode.

## Source map

| Path | Stage |
|---|---|
| `src/ber/normalize/`, `src/ber/table.py` | name, address and country normalisation into compact tables |
| `src/ber/vectorize.py` | hashed TF-IDF views, prefix filter |
| `src/ber/blocking/multiview.py` | per-country inverted-index search (name, address, combined) |
| `src/ber/scale.py` | candidate generation and selection, both stages, tuning, prediction |
| `src/ber/pairfeat.py` | vectorised pair and group-context features |
| `src/ber/decision.py`, `src/ber/metrics.py` | exclusive decision, macro F0.5 |
| `src/ber/submission.py` | output writer and validator |
| `src/ber/configs/full.yaml` | the configuration of the submitted run |
