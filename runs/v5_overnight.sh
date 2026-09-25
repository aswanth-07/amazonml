#!/usr/bin/env bash
# v5 overnight. Usage: bash runs/v5_overnight.sh <drop_share: 0 or 0.186>
#   GPU: France self-training of both cross-encoders. Each continues from its US/India
#        checkpoint on 1.2M original pairs plus 1.2M French test pairs that v3 and v4 both
#        decided confidently (artifacts/pseudo_france.npz: 450k matches, 750k hard
#        non-matches). scripts/ce_gate.py then promotes each adapted model only if its
#        held-out US/India log-loss stays within 5% of the source's (else the source is
#        copied in its place), and reports how undecided each is on French pairs.
#   CPU (in parallel): v5 = v3 + k_emb 20 + name-cluster features + the two gated
#        cross-encoders with pool-side context, every suitor scored in training,
#        700k training entities, optional orphan simulation.
cd "/c/Lenovo_data/amazon ml chal"
DROP=${1:-0}
V3=runs/20260925-161528-v3
DONOR=$V3
if [ "$DROP" != "0" ]; then DONOR=$(ls -d runs/*-v4 | tail -1); fi
echo "[$(date +%H:%M)] v5 overnight: drop=$DROP donor=$DONOR"
rm -rf models/ce-fr models/ce-small-fr models/ce-fr.tmp models/ce-small-fr.tmp

( export PYTHONPATH=src PYTHONUNBUFFERED=1
  COMMON="--blocking artifacts/cache/train_blocking_4ae347354a8f.joblib --knn artifacts/emb/train_knn.npz \
    --n-neg 4 --n-neg-knn 3 --max-pairs 1200000 --batch 256 \
    --pseudo artifacts/pseudo_france.npz --pseudo-split test"
  py -3.13 -m ber.crossenc train --init models/ce-base $COMMON --lr 2e-5 --model models/ce-fr.tmp
  echo "[$(date +%H:%M)] ce-fr EXIT $?"
  py -3.13 -m ber.crossenc train --init models/ce-best $COMMON --lr 3e-5 --model models/ce-small-fr.tmp
  echo "[$(date +%H:%M)] ce-small-fr EXIT $?"
  py -3.13 scripts/ce_gate.py --eval artifacts/ce_eval_pairs.npz --france artifacts/france_ce_sample.npz \
    --candidate models/ce-fr.tmp=models/ce-base=models/ce-fr \
    --candidate models/ce-small-fr.tmp=models/ce-best=models/ce-small-fr
  echo "[$(date +%H:%M)] gate EXIT $?"
  # if the gate itself failed, the run must still get a model: the sources, as in v3
  for pair in ce-fr:ce-base ce-small-fr:ce-best; do
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
  --set scale.ce_model=models/ce-small-fr,models/ce-fr \
  --set scale.ce_pool_context=true --set scale.ce_train_all=true --set scale.ce_wait=14400 \
  --set scale.train_sample=700000 --set scale.drop_s1_share=$DROP \
  --set scale.feature_donor_run=$DONOR
echo "[$(date +%H:%M)] v5 EXIT $?"
