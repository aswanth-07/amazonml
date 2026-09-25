#!/usr/bin/env bash
# v5 overnight. Usage: bash runs/v5_overnight.sh <drop_share: 0 or 0.186>
#   GPU: France self-trained cross-encoder (from ce-base, v3's confident French test decisions
#        as pseudo-labels + 700k original pairs), then checked on held-out US/India pairs.
#   CPU (in parallel): v5 = v3 + k_emb 20 + name-cluster features + 3 cross-encoders
#        (ce-best, ce-base, ce-fr) with pool-side context, every suitor scored in training,
#        700k training entities, optional orphan simulation.
cd "/c/Lenovo_data/amazon ml chal"
DROP=${1:-0}
V3=runs/20260925-161528-v3
DONOR=$V3
if [ "$DROP" != "0" ]; then DONOR=$(ls -d runs/*-v4 | tail -1); fi
echo "[$(date +%H:%M)] v5 overnight: drop=$DROP donor=$DONOR"

( export PYTHONPATH=src PYTHONUNBUFFERED=1
  py -3.13 -m ber.crossenc train --init models/ce-base \
    --blocking artifacts/cache/train_blocking_4ae347354a8f.joblib --knn artifacts/emb/train_knn.npz \
    --n-neg 4 --n-neg-knn 3 --max-pairs 700000 --batch 256 --lr 2e-5 \
    --pseudo artifacts/pseudo_france.npz --pseudo-split test --model models/ce-fr
  echo "[$(date +%H:%M)] ce-fr EXIT $?"
  py -3.13 -m ber.crossenc eval --model models/ce-base,models/ce-fr --split train --pairs artifacts/ce_eval_pairs.npz 2>&1 | grep "^models/"
) > runs/ce-fr.log 2>&1 &

PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -m ber.cli run --config src/ber/configs/full.yaml \
  --data-dir student_resource/dataset --out-dir output-v5 --tag v5 \
  --set scale.emb_dir=artifacts/emb --set scale.k_emb=20 \
  --set scale.ce_model=models/ce-best,models/ce-base,models/ce-fr \
  --set scale.ce_pool_context=true --set scale.ce_train_all=true --set scale.ce_wait=14400 \
  --set scale.train_sample=700000 --set scale.drop_s1_share=$DROP \
  --set scale.feature_donor_run=$DONOR
echo "[$(date +%H:%M)] v5 EXIT $?"
