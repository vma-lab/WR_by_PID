#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU="${GPU:?set GPU}"

FAILED=""
run_row () {  # run one step; on failure record it and continue
  "$@" || { echo "[FAILED] $*" >&2; FAILED="${FAILED}  $*"$'\n'; }
}
finish () {  # summary at the end; exit non-zero if any step failed
  if [ -n "$FAILED" ]; then
    printf '[SUMMARY] failed steps:\n%s' "$FAILED" >&2
    exit 1
  fi
  echo "[SUMMARY] all steps completed"
}

# Table 6 (Sec. 5.5), CVL. Cropped handwriting fields; the foreground gate
# is disabled for every model. All rows are fully frozen: nothing of CVL is
# used for training or fitting. INK_THRESHOLD=0 applies to the whole run,
# so the codebook, PCA and exemplar-SVM negatives are fitted on the HisIR19
# validation pages under the same disabled gate (not reused from the gated
# HisIR19 runs).

REF=checkpoints/reference/reference_vit_s16.pth
SCFG=configs/eval_vit_s16.yaml
ECFG=configs/eval_student.yaml
S=checkpoints/scenario1/ssl_student/last_distill.pth
FIT=data/wi_comp_19_validation

cvl_row () {  # $1 tag, then extra env
  local tag="$1"; shift
  for var in std esvm_nopca; do
    extra=""; [ $var = esvm_nopca ] && extra="PCA_DIM=0 ESVM=1"
    run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG="$tag" VARIANT=$var "$@" \
      TEST_DIR=data/cvl_eval CODEBOOK_DIR="$FIT" PCA_FIT_DIR="$FIT" VAL_DIR="$FIT" \
      IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 INK_THRESHOLD=0 $extra \
      bash run_scenario1.sh
  done
}

cvl_row cvl_reference CONFIG=$SCFG EVAL_CONFIG=$SCFG BACKBONE=$REF BINARISATION=input
cvl_row cvl_student   TRAINER=distill CONFIG=configs/stage2_student.yaml \
  EVAL_CONFIG=$ECFG BACKBONE=$S DISTILL_PROJ=auto

finish
