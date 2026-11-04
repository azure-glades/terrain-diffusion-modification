#!/usr/bin/env bash
# Trains the four students: l1_gan, l2_gan (stochastic, adversarial) and l3, l4 (deterministic + texture losses).
# Safe to re-run: every run uses --resume, so it continues from runs/<name>/state.pt if it exists.
#
#   bash train_all.sh                    # two parallel tracks:  l1_gan -> l3   and   l2_gan -> l4
#   PARALLEL=0 bash train_all.sh         # one job at a time (slower, but gentlest on a 6 GB GPU / few CPU cores)
#   ONLY="l2_gan l4" bash train_all.sh   # just some of them
#   DRY=1 bash train_all.sh              # print the commands only
#   STEPS_L1=60000 bash train_all.sh     # override step counts (STEPS_L1/L2/L3/L4)
#
# Logs: runs/logs/<name>.log      Watch: tail -f runs/logs/l1_gan.log
set -uo pipefail

TRAIN=${TRAIN:-data/train2}
VAL=${VAL:-data/val2}
PARALLEL=${PARALLEL:-1}
ONLY=${ONLY:-"l4"}
DRY=${DRY:-0}
STEPS_L1=${STEPS_L1:-30000}
STEPS_L2=${STEPS_L2:-40000}
STEPS_L3=${STEPS_L3:-30000}
STEPS_L4=${STEPS_L4:-30000}
PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less fragmentation on small GPUs
mkdir -p runs/logs

want() { [[ " $ONLY " == *" $1 "* ]]; }

# name | args
run() {
  local name=$1; shift
  local cmd=($PY train_level.py --train "$TRAIN" --val "$VAL" --out "runs/$name" --resume --no-bf16 "$@")
  echo "[train_all] $name: ${cmd[*]}"
  if [[ $DRY == 1 ]]; then return 0; fi
  "${cmd[@]}" >> "runs/logs/$name.log" 2>&1
  local rc=$?
  echo "[train_all] $name finished with exit code $rc (log: runs/logs/$name.log)"
  return $rc
}

# GAN students: noise inputs, relaxed L1/gradient anchor, adversarial loss, numbered snapshots to choose from afterwards.
l1_gan() { want l1_gan && run l1_gan --level 1 --crop 2048 --bs 4 --depth 4 --steps "$STEPS_L1" \
             --n-noise 2 --spec-w 0.1 --slope-w 0.2 --grad-w 0.2 --adv-w 0.2 --adv-start 3000 --snapshot-every 5000; }
l2_gan() { want l2_gan && run l2_gan --level 2 --crop 1024 --bs 8 --depth 4 --steps "$STEPS_L2" \
             --n-noise 2 --spec-w 0.1 --slope-w 0.2 --grad-w 0.2 --adv-w 0.2 --adv-start 5000 --snapshot-every 5000; }
# Deterministic students: more data (crops of the 2048 patches) and the texture losses.
# L4 works at full finest resolution, so it uses 256 px crops: with fp32 the 512 px crops need > 6 GB at batch 8.
l3()     { want l3     && run l3     --level 3 --crop 512 --bs 8 --steps "$STEPS_L3" --spec-w 0.1 --slope-w 0.2 --val-n 16; }
l4()     { want l4     && run l4     --crop 256 --bs 8 --steps "$STEPS_L4" --spec-w 0.1 --slope-w 0.2 --val-n 16; }

if [[ $PARALLEL == 1 ]]; then
  ( l1_gan; l3 ) &  pidA=$!
  ( l2_gan; l4 ) &  pidB=$!
  wait $pidA; rcA=$?
  wait $pidB; rcB=$?
  rc=$(( rcA > rcB ? rcA : rcB ))
else
  rc=0
  for f in l1_gan l2_gan l3 l4; do $f || rc=$?; done
fi
echo "[train_all] done (exit $rc). Evaluate with cascade_eval.py / show_levels.py using last.pt or a snap_*.pt for the GAN levels."
exit $rcxit $rc
