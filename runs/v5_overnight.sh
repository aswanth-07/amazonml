#!/usr/bin/env bash
# v5 overnight. Usage: bash runs/v5_overnight.sh <drop_share: 0 or 0.186>
#   GPU (sequential):
#     1. ce-small-fr: e5-small cross-encoder (ce-best) continued on 1.2M original pairs plus
#        1.2M French test pairs that v3 and v4 both decided confidently.
#     2. ce-bge-fr: bge-reranker-v2-m3 (568M, Apache-2.0, multilingual reranker) fine-tuned on
#        1.0M original pairs plus 600k of those French pairs; micro-batches of 64 with frozen
#        word embeddings fit it in 16 GB (measured 250 pairs/s training, 1,095 pairs/s scoring).
#     3. scripts/ce_gate.py promotes each only if its held-out US/India log-loss is within
#        tolerance of the incumbent (ce-best for the small one, ce-base for bge); otherwise
#        the incumbent is copied in its place, so the run always gets a model.
#   CPU (in parallel): v5 = v3 + k_emb 20 + name-cluster features + the two gated cross-encoders
#        with pool-side context; bge scores only the pairs the small one is unsure of
#        (ce_band 0.001-0.999); every suitor scored in training; 700k training entities;
#        optional orphan simulation.
cd "/c/Lenovo_data/amazon ml chal"
DROP=${1:-0}
V3=runs/20260925-161528-v3
DONOR=$V3
if [ "$DROP" != "0" ]; then DONOR=$(ls -d runs/*-v4 | tail -1); fi
echo "[$(date +%H:%M)] v5 overnight: drop=$DROP donor=$DONOR"
rm -rf models/ce-small-fr models/ce-bge-fr models/ce-small-fr.tmp models/ce-bge-fr.tmp

( export PYTHONPATH=src PYTHONUNBUFFERED=1
  NEG="--blocking artifacts/cache/train_blocking_4ae347354a8f.joblib --knn artifacts/emb/train_knn.npz --n-neg 4 --n-neg-knn 3"
  py -3.13 -m ber.crossenc train --init models/ce-best $NEG --max-pairs 1200000 --batch 256 --lr 3e-5 \
    --pseudo artifacts/pseudo_france.npz --pseudo-split test --model models/ce-small-fr.tmp
  echo "[$(date +%H:%M)] ce-small-fr EXIT $?"
  py -3.13 scripts/ce_gate.py --eval artifacts/ce_eval_pairs.npz --france artifacts/france_ce_sample.npz \
    --tolerance 0.05 --report runs/ce_gate_small.json \
    --candidate models/ce-small-fr.tmp=models/ce-best=models/ce-small-fr
  echo "[$(date +%H:%M)] gate small EXIT $?"
  py -3.13 -m ber.crossenc train --init models/bge-reranker-v2-m3 $NEG --max-pairs 1000000 --batch 128 \
    --micro-batch 64 --freeze-embeddings --lr 1.5e-5 \
    --pseudo artifacts/pseudo_france_600k.npz --pseudo-split test --model models/ce-bge-fr.tmp
  echo "[$(date +%H:%M)] ce-bge-fr EXIT $?"
  py -3.13 scripts/ce_gate.py --eval artifacts/ce_eval_pairs.npz --france artifacts/france_ce_sample.npz \
    --tolerance 0.10 --report runs/ce_gate_bge.json \
    --candidate models/ce-bge-fr.tmp=models/ce-base=models/ce-bge-fr
  echo "[$(date +%H:%M)] gate bge EXIT $?"
  # if a gate itself failed, the run must still get a model: the incumbents
  for pair in ce-small-fr:ce-best ce-bge-fr:ce-base; do
    final=models/${pair%%:*}; src=models/${pair##*:}
    if [ ! -f $final/crossenc.json ]; then
      rm -rf $final $final.copying && cp -r $src $final.copying && mv $final.copying $final
      echo "[$(date +%H:%M)] $final: gate gave nothing, using $src"
    fi
  done
) > runs/ce-fr.log 2>&1 &

PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -m ber.cli run --config src/ber/configs/full.yaml \
  --data-dir student_resource/dataset --out-dir output-v5 --tag v5 \
  --set scale.emb_dir=artifacts/emb --set scale.k_emb=20 \
  --set scale.ce_model=models/ce-small-fr,models/ce-bge-fr --set scale.ce_band=0.001,0.999 \
  --set scale.ce_pool_context=true --set scale.ce_train_all=true --set scale.ce_wait=18000 \
  --set scale.train_sample=700000 --set scale.drop_s1_share=$DROP \
  --set scale.feature_donor_run=$DONOR
echo "[$(date +%H:%M)] v5 EXIT $?"
