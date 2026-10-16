#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_TAG="${RUN_TAG:-}"

VARIANT="${VARIANT:-}"

DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

AGGREGATOR="${AGGREGATOR:-vlad}"
: "${PATCH_TOKENS:=1}"
: "${PCA_DIM:=384}"
: "${OVERLAP:=0.75}"
: "${QE_K:=2}"
: "${IMAGE_SIZE:=0}"
: "${MAX_IMAGE_SIZE:=0}"
: "${INK_THRESHOLD:=0.04}"
: "${ESVM:=0}"
: "${EMBED_WORKERS:=4}"

_SUFFIX="${RUN_TAG:+_${RUN_TAG}}"
CKPT_DIR="${CKPT_DIR:-checkpoints/scenario1/ssl${_SUFFIX}}"
RUN_DIR="${RUN_DIR:-runs/scenario1/ssl${_SUFFIX}}"
OUT_DIR="${OUT_DIR:-results/scenario1${_SUFFIX}}"
CONFIG="${CONFIG:-configs/stage2_student.yaml}"
TRAINER="${TRAINER:-distill}"
if [ "$TRAINER" != distill ] && [ "$TRAINER" != teacher ]; then
  echo "[ERROR] TRAINER must be distill or teacher (got: $TRAINER)"
  exit 1
fi
EVAL_CONFIG="${EVAL_CONFIG:-$CONFIG}"
_BEST=last_distill.pth
[ "$TRAINER" = teacher ] && _BEST=last_teacher.pth
BACKBONE="${BACKBONE:-${CKPT_DIR}/${_BEST}}"

DESCRIPTOR_CACHE="${DESCRIPTOR_CACHE-${OUT_DIR}/desc_cache}"

source scenario_common.sh
mkdir -p "$OUT_DIR"

_OUT_TAG="${AGGREGATOR}"
case "${PATCH_TOKENS:-}" in 1|true|yes|on) _OUT_TAG="${_OUT_TAG}_patch" ;; esac
_OUT_TAG="${_OUT_TAG}${VARIANT:+_${VARIANT}}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/LOO_${_OUT_TAG}.csv}"

if [ "$AGGREGATOR" = vlad ] && [ -z "${VLAD_CLUSTERS:-}" ]; then
  EVAL_FLAGS+=(--vlad_clusters 100)
fi

if [ "${DO_EVAL:-1}" = 1 ] && [ "${DO_TRAIN:-1}" != 1 ] && [ ! -f "$BACKBONE" ]; then
  echo "[ERROR] Backbone checkpoint not found: $BACKBONE"
  echo "        Available checkpoint directories:"
  ls -1d checkpoints/*/ 2>/dev/null | sed 's/^/          /' || true
  echo "        Fix: set RUN_TAG (or BACKBONE=…) to match the training run."
  exit 1
fi

if [ "${DO_TRAIN:-1}" = 1 ]; then
  if [ "$TRAINER" = distill ]; then
    echo "=== Training: Stage 2, distillation ==="
    echo "    checkpoints -> $CKPT_DIR    run -> $RUN_DIR"
    run_py -u -X faulthandler distill_train.py --config "$CONFIG" \
      --run_dir "$RUN_DIR" --checkpoint_dir "$CKPT_DIR" \
      2>&1 | tee "$OUT_DIR/scenario1_train.log"
  else
    echo "=== Training: Stage 1, teacher (reference trainer) ==="
    echo "    checkpoints -> $CKPT_DIR    run -> $RUN_DIR"
    run_py -u -X faulthandler teacher_train.py --config "$CONFIG" \
      --run_dir "$RUN_DIR" --checkpoint_dir "$CKPT_DIR" \
      2>&1 | tee "$OUT_DIR/scenario1_train.log"
  fi
fi

if [ "${DO_EVAL:-1}" = 1 ]; then
  echo "=== Evaluation: leave-one-out (aggregator=$AGGREGATOR) ==="
  echo "    backbone -> $BACKBONE    results -> $OUT_DIR"
  run_py -u -X faulthandler leave_one_out.py --config "$EVAL_CONFIG" --model_path "$BACKBONE" \
    --method "$AGGREGATOR" --output_csv "$OUT_CSV" "${EVAL_FLAGS[@]}" \
    2>&1 | tee "$OUT_DIR/scenario1_eval_${_OUT_TAG}.log"
fi
