#!/usr/bin/env bash
# v4 on a rented Linux GPU box: the whole pipeline plus the listwise reranker.
# Plan and reasoning: docs/v4_listwise.md.
#
# Box: 1 GPU (H100/A100 80 GB best; 24 GB works for the small models), >= 32 vCPU,
# >= 128 GB RAM, >= 300 GB disk. Run from the repository root, one step at a time:
#
#   bash scripts/gpu_runbook.sh setup     # venv, deps, tests, base encoder          ~20 min
#   bash scripts/gpu_runbook.sh embed     # fine-tune bi-encoder, encode, kNN (GPU)   ~40 min H100
#   bash scripts/gpu_runbook.sh shift     # train/test shift report (CPU)             ~10 min
#   bash scripts/gpu_runbook.sh runA      # blocking, features, stage 1+2, lw groups  3-5 h cold (CPU)
#   bash scripts/gpu_runbook.sh ce        # pairwise cross-encoder (GPU; can start once
#                                         #   artifacts/cache/train_blocking_*.joblib exists)  ~40 min
#   bash scripts/gpu_runbook.sh groups    # listwise training groups from the raw data (CPU,
#                                         #   no run needed; encoder share)            ~15-25 min
#   bash scripts/gpu_runbook.sh lw        # listwise reranker (GPU)                   ~20-40 min small
#   bash scripts/gpu_runbook.sh lweval    # quick comparison: p1 vs CE vs listwise    ~10 min
#   bash scripts/gpu_runbook.sh pgroups   # C7 pool-side groups from the raw data (CPU) ~20-30 min
#   bash scripts/gpu_runbook.sh lwp       # C7 pool-side listwise (GPU)               ~30-60 min small
#   bash scripts/gpu_runbook.sh lwpeval   # C7 owner accuracy vs p1 proxy and CE      ~10 min
#   bash scripts/gpu_runbook.sh runB      # stage 2 with CE + listwise, saves stage-2 inputs  ~1.5-2 h
#   bash scripts/gpu_runbook.sh search    # every stage-2 model combination (CPU)     ~1-3 h
#   bash scripts/gpu_runbook.sh final EXCLUDE=... [MONO=true]   # the submission   ~1 h (scores cached)
#
# Settings (environment variables):
#   DATA     dataset folder with train/ and test/            (student_resource/dataset)
#   DROP     orphan simulation share; 0.186 if `shift` says orphans    (0.0)
#   SHARE    listwise share of non-sample eligible entities  (0.5)
#   SAMPLE   stage-1/2 training entities                     (500000)
#   LW_INIT  encoder the listwise model starts from          (models/ce-small-v2)
#   LW_MODEL where the listwise model goes                   (models/lw-small)
#   LWP_MODEL where the pool-side listwise model goes        (models/lwp-small)
#   CE_MODEL pairwise cross-encoder                          (models/ce-small-v2)
#   GROUPS   listwise training groups (artifacts/lw_groups_enc.npz from `groups`, else
#            run A's lw_groups_train.npz)
set -euo pipefail

STEP=${1:-help}
shift || true
for kv in "$@"; do export "${kv?}"; done  # allow `final EXCLUDE=lw_grp MONO=true`

DATA=${DATA:-student_resource/dataset}
PY=${PY:-python}
DROP=${DROP:-0.0}
SHARE=${SHARE:-0.5}
SAMPLE=${SAMPLE:-500000}
LW_INIT=${LW_INIT:-models/ce-small-v2}
LW_MODEL=${LW_MODEL:-models/lw-small}
LWP_MODEL=${LWP_MODEL:-models/lwp-small}
CE_MODEL=${CE_MODEL:-models/ce-small-v2}
NCPU=$(nproc)
export PYTHONPATH=src
# the same settings for every run, so stage 1 is reusable between them
COMMON=(--config src/ber/configs/full.yaml --data-dir "$DATA"
        --set scale.emb_dir=artifacts/emb --set "scale.train_sample=$SAMPLE"
        --set "scale.lw_share=$SHARE" --set "scale.drop_s1_share=$DROP"
        --set "scale.ce_python=$PY" --set "scale.n_threads=$((NCPU - 4))"
        --set "scale.n_jobs=$((NCPU / 2))")

latest_run() { ls -td runs/*-"$1" 2>/dev/null | head -1; }
groups_file() {
  if [ -n "${GROUPS:-}" ]; then echo "$GROUPS"
  elif [ -f artifacts/lw_groups_enc.npz ]; then echo artifacts/lw_groups_enc.npz
  else echo "$(latest_run A)/lw_groups_train.npz"; fi
}

case "$STEP" in
setup)
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements-dev.txt
  # pick the wheel index that matches the driver (nvidia-smi: CUDA >= 12.8 -> cu128, >= 13.0 -> cu130)
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install "transformers==5.5.4" "tokenizers==0.22.2" "safetensors==0.7.0" "huggingface_hub==1.11.0"
  pip install -e . --no-deps
  pytest -q -m "not slow"
  python -m ber.cli fetch-model intfloat/multilingual-e5-small --dest models/multilingual-e5-small
  nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2; echo "cpus: $NCPU"
  test -f "$DATA/train/train_source1.tsv" || echo "!! put the dataset at $DATA (train/ and test/)"
  echo "next: source .venv/bin/activate && bash scripts/gpu_runbook.sh embed"
  ;;
embed)
  # skip the fine-tune if the lead's models/e5-small-ft is copied in: use `encode` instead of `all`
  cmd=all; [ -f models/e5-small-ft/config.json ] && cmd=encode
  $PY -m ber.embed $cmd --data-dir "$DATA" --base models/multilingual-e5-small \
      --model models/e5-small-ft --emb-dir artifacts/emb --epochs 0.5 --batch 512 --k 50 --k-reverse 3
  cat artifacts/emb/report.json
  ;;
shift)
  mkdir -p runs
  $PY scripts/shift_diagnostics.py --data-dir "$DATA" --emb-dir artifacts/emb \
      --out runs/shift_diagnostics.json ${RUN:+--run "$RUN"}
  echo "read the verdict lines at the end; orphan-like -> rerun runA with DROP=0.186"
  ;;
runA)
  # no cross-encoder yet: this run makes the caches, stage 1, lw_groups_train.npz, and a
  # submittable v2-level output-A (a fallback)
  $PY -m ber.cli run "${COMMON[@]}" --out-dir output-A --tag A
  RUN_A=$(latest_run A); echo "run A: $RUN_A"; ls -la "$RUN_A"/lw_groups_train.npz
  ;;
ce)
  # starts as soon as run A has written the train blocking cache (its first ~hour)
  BLOCK=$(ls -t artifacts/cache/train_blocking_*.joblib | head -1)
  $PY -m ber.crossenc train --data-dir "$DATA" --init models/e5-small-ft --blocking "$BLOCK" \
      --model "$CE_MODEL" --knn artifacts/emb/train_knn.npz --max-pairs 4000000 --batch 256
  ;;
groups)
  # encoder-share entities: the matcher's holdout stays exactly as in every other run
  $PY scripts/make_lw_groups.py --data-dir "$DATA" --out artifacts/lw_groups_enc.npz \
      --train "${LW_TRAIN:-400000}" --valid 30000
  ;;
lw)
  [ -f "$LW_INIT/config.json" ] || { echo "!! $LW_INIT missing (run 'ce' first or set LW_INIT)"; exit 1; }
  $PY -m ber.listwise train --data-dir "$DATA" --groups "$(groups_file)" \
      --init "$LW_INIT" --model "$LW_MODEL" --batch "${BATCH:-32}" --epochs "${EPOCHS:-1}" \
      --lr "${LR:-3e-5}" ${GRAD_CKPT:+--grad-ckpt}
  cat "$LW_MODEL/listwise.json"
  ;;
lweval)
  $PY -m ber.listwise eval --data-dir "$DATA" --groups "$(groups_file)" \
      --model "$LW_MODEL" --ce "$CE_MODEL" --max-entities 60000 | tee "runs/lweval_$(basename "$LW_MODEL").json"
  echo "gate: the blend of lw with the CE above the CE alone (macro_f05_proxy) -> keep it"
  ;;
pgroups)
  $PY scripts/make_lw_groups.py --data-dir "$DATA" --out artifacts/lwp_groups_enc.npz --side pool \
      --train "${LWP_TRAIN:-800000}" --valid 30000
  ;;
lwp)
  [ -f "$LW_INIT/config.json" ] || { echo "!! $LW_INIT missing (run 'ce' first or set LW_INIT)"; exit 1; }
  $PY -m ber.listwise train --data-dir "$DATA" --groups artifacts/lwp_groups_enc.npz \
      --init "$LW_INIT" --model "$LWP_MODEL" --batch "${BATCH:-32}" --epochs "${EPOCHS:-1}" \
      --lr "${LR:-3e-5}" ${GRAD_CKPT:+--grad-ckpt}
  cat "$LWP_MODEL/listwise.json"
  ;;
lwpeval)
  $PY -m ber.listwise eval --data-dir "$DATA" --groups artifacts/lwp_groups_enc.npz \
      --model "$LWP_MODEL" --ce "$CE_MODEL" --max-entities 60000 | tee "runs/lwpeval_$(basename "$LWP_MODEL").json"
  echo "gate: lwp (or its blend with the CE) owner_accuracy above the CE alone -> keep it"
  ;;
runB)
  RUN_A=$(latest_run A)
  MODELS=${MODELS:-$CE_MODEL,$LW_MODEL,$LWP_MODEL}
  $PY -m ber.cli run "${COMMON[@]}" --out-dir output-B --tag B \
      --set "scale.stage1_model=$RUN_A/scale_model.joblib" --set "scale.ce_model=$MODELS" \
      --set scale.save_stage2=true --set scale.ce_train_all=true --set scale.ce_pool_context=true
  RUN_B=$(latest_run B)
  $PY - "$RUN_A" "$RUN_B" <<'EOF'
import json, sys
for run in sys.argv[1:]:
    m = json.load(open(f"{run}/metrics.json"))
    d = m.get("decision", {})
    print(run, "stage1", m["stage1"].get("holdout"), "stage2", m.get("stage2", {}).get("holdout"),
          "decision", {k: v.get("holdout") for k, v in d.items() if isinstance(v, dict) and "holdout" in v})
EOF
  ;;
search)
  RUN_B=$(latest_run B)
  # screening on 250k sample entities keeps each combination to a few minutes
  $PY scripts/stage2_search.py --run "$RUN_B" --monotone --bags 3 --threads "$NCPU" \
      --max-sample "${MAX_SAMPLE:-250000}"
  ;;
final)
  RUN_A=$(latest_run A)
  MODELS=${MODELS:-$CE_MODEL,$LW_MODEL,$LWP_MODEL}
  $PY -m ber.cli run "${COMMON[@]}" --out-dir output-final --tag final \
      --set "scale.stage1_model=$RUN_A/scale_model.joblib" --set "scale.ce_model=$MODELS" \
      --set "scale.stage2_exclude=${EXCLUDE:-}" --set "scale.stage2_monotone=${MONO:-false}"       --set scale.ce_train_all=true --set scale.ce_pool_context=true
  $PY -m ber.cli validate --matching output-final/matching_results.tsv \
      --candidates output-final/candidate_pairs.tsv --test-dir "$DATA/test"
  echo "submit output-final/matching_results.tsv only if its holdout beats the best so far"
  ;;
*)
  sed -n '2,32p' "$0"
  ;;
esac
