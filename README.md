# Business Entity Resolution (Amazon ML Challenge)

Match business records from three noisy sources. For every Source 1 entity,
find all Source 2 and Source 3 records that describe the same real-world
business. Scoring is macro F0.5 per Source 1 entity, singletons included.

The full brief, condensed, with the file where each rule is enforced:
[docs/problem_statement.md](docs/problem_statement.md). The methodology draft,
which becomes the submitted `Documentation_template.md`:
[docs/Documentation_template.md](docs/Documentation_template.md).

```
raw TSVs → parallel normalise → multi-view inverted-index blocking → full re-score + select → pair features
         → LightGBM (stage 1) → stacked LightGBM (stage 2) → exclusive, F0.5-tuned decision → output/*.tsv
                                        └─ candidate_pairs.tsv                              └─ matching_results.tsv
```

Built for the real data: 2.2M Source 1 × 10.3M Source 2/3 records per split,
on a laptop CPU with about 20 GB of free memory. The design and measurements
are in [docs/Documentation_template.md](docs/Documentation_template.md).

## Quick start

Python 3.10. Windows PowerShell is shown; on Linux or macOS use
`source .venv/bin/activate`.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
pytest -q                                     # about 110 tests, under 10 s
```

To see the whole pipeline work before the real data arrives, generate the
synthetic dataset (same format; France only in test) and run it:

```powershell
ber synth --out data/synthetic
ber run --data-dir data/synthetic --out-dir output/synthetic
```

## When the dataset arrives

1. Unzip the organisers' `student_resource/` into the repository root; it is
   git-ignored. Copy or link its `dataset/` folder to `./dataset`, so the files
   are at `dataset/train/train_source1.tsv` and so on. `ber validate` picks up
   `student_resource/utils/validate_submission.py` automatically.
2. **Audit the data and the pipeline's assumptions:**
   ```powershell
   ber audit --json runs/audit.json
   ```
   Read the recommendations at the end. They check three defaults:

   | Default | Holds when |
   |---|---|
   | `decision.exclusive=true` | no S2/S3 id belongs to two S1 entities |
   | `blocking.partition_by_country=true` | true matches share a country label |
   | `data.quoting=none` | no field starts with a quote |

3. **Evaluate on train only:**
   `ber cv --config src/ber/configs/full.yaml --data-dir student_resource/dataset`.
   It prints blocking recall and the oracle F0.5 (the ceiling blocking
   allows), then stage-1 and stage-2 F0.5 on the sampled entities (out of
   fold) and on the held-out entities. False-positive and false-negative
   samples are written to `runs/<timestamp>-cv/`.
4. **Produce the submission:**
   `ber run --config src/ber/configs/full.yaml --data-dir student_resource/dataset --out-dir output`.
   This takes about 2 hours. It does everything `cv` does, then blocks and
   scores test, writes `output/`, and validates it with the local validator.
   Normalised tables are cached, so later runs skip that step. Upload
   `output/matching_results.tsv` to the portal.
5. **Package:** replace `docs/Documentation_template.md` with the official
   template filled from the draft, then run
   `ber package --team <team_name>` to get `<team_name>_submission.zip`.

## Commands

| Command | What it does |
|---|---|
| `ber audit` | data-contract checks, noise statistics, assumption checks with recommendations |
| `ber cv [--config F] [--set k=v]` | out-of-fold evaluation on train, recorded under `runs/` |
| `ber run` | full pipeline: train, tune, predict test, write and validate `output/` |
| `ber predict --model artifacts/scale_model.joblib` | re-predict test from a saved model |
| `ber validate` | local re-implementation of every published rule, plus the official validator if present |
| `ber score --pred P --truth T` | macro F0.5 of any matching file against a truth file |
| `ber package --team NAME` | builds the submission zip in the required layout; refuses invalid outputs |
| `ber synth` | synthetic dataset in the challenge format |
| `ber fetch-model NAME` | one-time download of a licence-checked encoder (optional blocker) |

`python -m ber ...` and `python src/main.py ...` work without the console script.

## Configuration

Every tunable is in [src/ber/configs/default.yaml](src/ber/configs/default.yaml),
which mirrors the typed dataclasses in [src/ber/config.py](src/ber/config.py).
Unknown keys are errors. Override any value with a file or dotted keys:

```powershell
ber cv --config configs/fast.yaml
ber cv --set blocking.max_candidates=80 --set decision.strategy=expected_f --tag cap80-ef
ber cv --set blocking.tfidf.0.top_k=50
```

Each run directory stores the resolved config, `metrics.json` (overall,
per-country, per-cardinality and blocking metrics, plus the git revision), the
decision curve, feature importance and error tables. Comparing experiments
means comparing these folders.

## Layout

```
src/ber/
  schema.py, io.py         data contract; strict TSV reading, LF/no-quote writing
  normalize/               text cleaning, names (legal forms, DBA, skeleton), addresses (postal, landmarks)
  blocking/                TF-IDF top-k, key blocks, optional embeddings, capped union
  features.py              76 pair features including competition features
  matcher.py               LightGBM with grouped cross-fitting
  decision.py              exclusive assignment, threshold or expected-F rule, tuning
  metrics.py               exact leaderboard metric and blocking diagnostics
  pipeline.py              orchestration and run recording
  submission.py            output writer, validator, zip packager
  audit.py, synth.py       data audit; synthetic generator
  pretrained.py            licence/size registry for optional encoders
  cli.py
configs/                   example experiment configs
docs/                      problem statement, methodology draft, submission README
tests/                     unit and end-to-end tests, including fair-play guards
```

Git-ignored: `dataset/`, `student_resource/`, `data/`, `output/`, `runs/`,
`artifacts/`, `models/`, `*.zip`.

## Rules the code enforces

- **No external data lookup.** `tests/test_fair_play.py` allows the pipeline
  to import only the standard library (minus its networking modules) and the
  pinned dependencies. Any other import fails the build. The exceptions are
  the optional, offline encoder (`sentence_transformers`) and the setup-time
  weight download in `pretrained.py`.
- **Model licence and size.** The matcher is LightGBM (MIT). Any pretrained
  encoder must be in `ber.pretrained.REGISTRY` (MIT or Apache-2.0, ≤ 8B
  parameters) and is loaded offline.
- **Country is an open set.** There is no country one-hot and no filter to
  {US, India}; a test guards this. France is covered end to end on synthetic
  data.
- **Output format.** Files are written with LF endings and no quoting, one row
  per S1 entity, and matches are a subset of candidates by construction.

## Optional: dense-embedding blocker (GPU)

This blocker is off by default. It is worth trying if blocking recall is the
bottleneck.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs CUDA 12.8 wheels
pip install -e ".[embed]"
ber fetch-model intfloat/multilingual-e5-small
ber cv --config configs/embeddings.yaml
```

## Development

```powershell
ruff check src tests; ruff format src tests
pytest -q                       # everything
pytest -q -m "not slow"         # skip the end-to-end runs
```

CI runs lint and tests on Ubuntu and Windows
([.github/workflows/ci.yml](.github/workflows/ci.yml)).
