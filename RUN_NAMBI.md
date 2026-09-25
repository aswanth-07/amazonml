# Nambi: run v6 end to end (one command)

Everything is on branch `nikhil_exp`. You only need the dataset (the folder holding
`train/` and `test/`) and a laptop with an NVIDIA GPU. The script checks your machine
itself (GPU, free GPU memory, bfloat16 support, CPU threads, RAM, disk) and sizes every
batch, thread count and sample from what it finds. Nothing is hard-coded to one laptop.

## 1. Get the code

```bash
git clone https://github.com/aswanth-07/amazonml && cd amazonml
git switch nikhil_exp
```

(Already cloned: `git fetch origin && git switch nikhil_exp && git pull`.)

## 2. Make a Python 3.10-3.13 environment

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Linux:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

## 3. Run

```bash
python scripts/run_all.py --data-dir "PATH/TO/student_resource/dataset"
```

That is all. The first step installs everything, including PyTorch built for your
driver's CUDA version (from `nvidia-smi`). Then it measures the machine and prints the
profile it chose, and runs these steps:

| # | Step | What | Where | Rough time (16 GB GPU, 32 GB RAM) |
|---|---|---|---|---|
| 1 | setup | pip installs (pinned) + CUDA PyTorch | | 10 min |
| 2 | hardware | measure, write `runs/v6/profile.json` | | 1 min |
| 3 | models | download multilingual-e5-small (MIT) | | 2 min |
| 4 | embed | fine-tune bi-encoder, encode 24M texts, kNN | GPU | 1-1.5 h |
| 5 | runA | blocking, features, stage 1+2 | CPU | 3-4 h |
| 6 | ce | pairwise cross-encoder | GPU | 45 min |
| 7 | evalpairs | held-out pairs for the gates | CPU | 5 min |
| 8 | runB | + cross-encoder (about v4) | CPU+GPU | 1.5 h |
| 9 | pseudo | French pseudo-labels (runs A and B agree) | CPU | 5 min |
| 10 | cefr | France-adapted cross-encoder, gated on US/India | GPU | 45 min |
| 11 | large | (only with `--large`) bge-reranker-v2-m3 France-adapted | GPU | +3 h |
| 12-13 | groups, pgroups | C1 / C7 training groups from the raw data | CPU | 40 min |
| 14-15 | lw, lwp | C1 and C7 listwise rerankers | GPU | 1.5-2 h |
| 16 | lweval | C1/C7 vs stage 1 and the cross-encoder | GPU | 15 min |
| 17 | runC | v6: France CE + C1 + C7 + source fingerprints | CPU+GPU | 2-2.5 h |
| 18 | search | every stage-2 family on the holdout, with and without C1/C7 | CPU | 1-2 h |
| 19 | final | the best combination (scores cached) | CPU | 45 min |
| 20 | report | `runs/v6/RESULTS.md` | | 1 min |

About 14-17 hours in all. Keep the laptop plugged in, with sleep off, while it runs.

## 4. What you get

- `runs/v6/RESULTS.md`: every holdout macro F0.5 (runs A, B, C, final), the stage-2
  search with and without the listwise columns, C1/C7 evaluations, the cross-encoder
  gate reports, and which file to submit.
- `output-final/matching_results.tsv` and `candidate_pairs.tsv`, already validated:
  the v6 submission. Fallbacks: `output-C/`, `output-B/` (about v4), `output-A/`.
- **Do not upload to the portal without telling the lead**; paste the RESULTS.md
  numbers into the handbook, section 2.

## 5. If something stops

- **Rerun the same command**: finished steps are skipped (markers in `runs/v6/done/`).
- **Out of GPU memory**: handled; the step reruns with half the batch (up to 4 times).
- **Redo from a step**: `python scripts/run_all.py --data-dir ... --from lw`.
- **One step only**: `--only hardware` (for example, to see the chosen profile first).
- **Logs**: `runs/v6/logs/<step>.log`; the console prints a progress line every minute.
- **Disk**: the caches need about 120-150 GB free; the script warns when there is less.
- **RAM under 24 GB**: it still runs with smaller chunks and a smaller training sample,
  but close other apps.

## 6. Have more time or a big GPU?

- `--large` adds the France-adapted bge-reranker-v2-m3 (568M, Apache-2.0) when the GPU has
  at least 14 GB free: the strongest cross-encoder in the lead's v5.
- `--drop 0` trains without the orphan simulation, for comparison with v3/v4-style runs.

What each model is and why: `docs/nikhil_exp_plan.md`. All the ideas:
`docs/ideas_worth_trying.md`.
