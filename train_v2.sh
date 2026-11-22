#!/usr/bin/env bash
# Three-stage generator (replaces the four per-level students and the procedural level 0):
#
#   teacher coarse map (7.68 km)  --struct-->  level 0 (480 m)  --d1-->  level 2 (120 m)  --d2-->  level 4 (30 m)
#
#   struct : structure model, conditional flow matching (train_structure.py)
#   d1     : merged 4x detail stage, noise inputs + adversarial loss, trained on the teacher's level 0
#   d2     : merged 4x detail stage, noise inputs + adversarial loss, trained on d1's OUTPUT (--chain), so what it sees
#            in training is what it gets in the explorer
#
# Prerequisite (once per dataset; only runs the teacher's coarse stage):
#   python make_pyramid_data.py add-coarse --repo terrain-diffusion --out data/train2      # ... train3, val2, val3
#
#   bash train_v2.sh                       # struct and d1 in parallel, then d2
#   PARALLEL=0 bash train_v2.sh            # one at a time (gentlest on a 6 GB GPU)
#   ONLY="d1 d2" bash train_v2.sh          # just some of them (d2 needs runs/d1/last.pt)
#   DRY=1 bash train_v2.sh                 # print the commands only
# Every run uses --resume: re-running continues from runs/<name>/state.pt.
set -uo pipefail

TRAIN=${TRAIN:-data/train2,data/train3}
VAL=${VAL:-data/val2,data/val3}
PARALLEL=${PARALLEL:-1}
ONLY=${ONLY:-"struct d1 d2"}
DRY=${DRY:-0}
STEPS_STRUCT=${STEPS_STRUCT:-60000}
STEPS_D1=${STEPS_D1:-40000}
STEPS_D2=${STEPS_D2:-40000}
PY=${PY:-python}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p runs/logs

want() { [[ " $ONLY " == *" $1 "* ]]; }

run() {   # name script args...
  local name=$1 script=$2; shift 2
  local cmd=($PY "$script" --train "$TRAIN" --val "$VAL" --out "runs/$name" --resume "$@")
  echo "[train_v2] $name: ${cmd[*]}"
  if [[ $DRY == 1 ]]; then return 0; fi
  "${cmd[@]}" 2>&1 | tee -a "runs/logs/$name.log" | sed -u "s/^/[$name] /"
  local rc=${PIPESTATUS[0]}
  echo "[train_v2] $name finished with exit code $rc (log: runs/logs/$name.log)"
  return $rc
}

GAN="--n-noise 2 --spec-w 0.1 --slope-w 0.2 --adv-w 0.2 --adv-start 3000 --snapshot-every 5000 --no-bf16"

struct() { want struct || return 0; run struct train_structure.py --steps "$STEPS_STRUCT" --bs 16; }
# d1: 480 m -> 120 m. Whole 2048 px patches (512 px at level 2); small pixel-wise anchor as in the old GAN students.
d1() { want d1 || return 0; run d1 train_level.py --from-level 0 --level 2 --crop 2048 --bs 4 --depth 4 --base 32 \
         --steps "$STEPS_D1" --grad-w 0.2 $GAN; }
# d2: 120 m -> 30 m on d1's output. 512 px crops at the finest level (128 px at level 2, 32 px at level 0).
d2() { want d2 || return 0
       if [[ $DRY != 1 && ! -f runs/d1/last.pt ]]; then echo "[train_v2] d2 needs runs/d1/last.pt (train d1 first)"; return 1; fi
       run d2 train_level.py --from-level 2 --level 4 --chain runs/d1/last.pt --crop 512 --bs 4 --depth 4 --base 32 \
         --steps "$STEPS_D2" --val-n 16 $GAN; }

if [[ $PARALLEL == 1 ]]; then
  ( struct ) & pidA=$!
  ( d1 && d2 ); rcB=$?
  wait $pidA; rcA=$?
  rc=$(( rcA > rcB ? rcA : rcB ))
else
  rc=0
  for f in struct d1 d2; do $f || rc=$?; done
fi
echo "[train_v2] done (exit $rc). Look at the result:"
echo "  python show_structure.py --data data/val3 --ckpt runs/struct/best.pt --seam"
echo "  python cascade_eval.py --data data/val2 --students runs/d1/last.pt,runs/d2/last.pt"
echo "  python main.py --student runs/d1/last.pt,runs/d2/last.pt --structure runs/struct/best.pt --repo terrain-diffusion"
exit $rc
