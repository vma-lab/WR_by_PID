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

# Full configuration (Sec. 5.4 and Table 4): exemplar-SVM re-encoding
# without PCA, then the same query expansion. Descriptors are reused from
# the base-configuration runs of Tables 1 to 3 (same RUN_TAG) where cached.

REF=checkpoints/reference/reference_vit_s16.pth
SCFG=configs/eval_vit_s16.yaml
ECFG=configs/eval_student.yaml
T=checkpoints/scenario1/ssl_teacher/last_teacher.pth
FULL="PCA_DIM=0 ESVM=1"

# HisIR19, reference checkpoint and improved teacher
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=reference VARIANT=esvm_nopca CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 \
  CODEBOOK_OVERLAP=0 $FULL bash run_scenario1.sh

run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=teacher_eval VARIANT=esvm_nopca CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$T BINARISATION=input \
  IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 $FULL bash run_scenario1.sh

# HisIR19, final student
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=student_eval VARIANT=esvm_nopca TRAINER=distill \
  CONFIG=configs/stage2_student.yaml EVAL_CONFIG=$ECFG \
  BACKBONE=checkpoints/scenario1/ssl_student/last_distill.pth DISTILL_PROJ=auto \
  IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 $FULL bash run_scenario1.sh

# Historical-WI, reference checkpoint (provided binarised pages, inverted)
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=reference_histwi VARIANT=esvm_nopca CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 $FULL \
  TEST_DIR=data/wi_icdar_17_test_bin_inv CODEBOOK_DIR=data/wi_icdar_17_train_bin_inv \
  PCA_FIT_DIR=data/wi_icdar_17_train_bin_inv VAL_DIR=data/wi_icdar_17_train_bin_inv \
  bash run_scenario1.sh

# Historical-WI, final student
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=student_eval_histwi VARIANT=esvm_nopca TRAINER=distill \
  CONFIG=configs/stage2_student.yaml EVAL_CONFIG=$ECFG \
  BACKBONE=checkpoints/scenario1/ssl_student/last_distill.pth DISTILL_PROJ=auto \
  IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 $FULL \
  TEST_DIR=data/wi_icdar_17_test_colour CODEBOOK_DIR=data/wi_icdar_17_train_colour \
  PCA_FIT_DIR=data/wi_icdar_17_train_colour VAL_DIR=data/wi_icdar_17_train_colour \
  bash run_scenario1.sh

# representation ablation at full width, full configuration (Sec. 5.3)
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=student_eval_fullwidth VARIANT=esvm_nopca TRAINER=distill \
  CONFIG=configs/stage2_student.yaml EVAL_CONFIG=$ECFG \
  BACKBONE=checkpoints/scenario1/ssl_student/last_distill.pth \
  IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 $FULL bash run_scenario1.sh

finish
