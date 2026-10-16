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

# Table 2 (Sec. 5.2). Stage 1 training and the teacher rows (base
# configuration).

# training (once each; about 2 days per run)
run_row env GPU=$GPU RUN_TAG=teacher TRAINER=teacher \
  CONFIG=configs/stage1_teacher.yaml DO_EVAL=0 bash run_scenario1.sh
run_row env GPU=$GPU RUN_TAG=teacher_reproduction TRAINER=teacher \
  CONFIG=configs/stage1_teacher_reproduction.yaml DO_EVAL=0 bash run_scenario1.sh

# sampler-only control (page-grouped batches, contrastive weight zero)
run_row env GPU=$GPU RUN_TAG=teacher_control TRAINER=teacher \
  CONFIG=configs/stage1_teacher_control.yaml DO_EVAL=0 bash run_scenario1.sh

SCFG=configs/eval_vit_s16.yaml
T=checkpoints/scenario1/ssl_teacher/last_teacher.pth

# improved teacher, HisIR19 and Historical-WI
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=teacher_eval VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$T BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 \
  CODEBOOK_OVERLAP=0 bash run_scenario1.sh
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=teacher_eval_histwi VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$T BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 \
  TEST_DIR=data/wi_icdar_17_test_bin_inv CODEBOOK_DIR=data/wi_icdar_17_train_bin_inv \
  PCA_FIT_DIR=data/wi_icdar_17_train_bin_inv VAL_DIR=data/wi_icdar_17_train_bin_inv \
  bash run_scenario1.sh

# sampler-only control row
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=teacher_control_eval VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=checkpoints/scenario1/ssl_teacher_control/last_teacher.pth BINARISATION=input \
  IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 bash run_scenario1.sh

# the reproduction rows are in reproduce/table1_baseline.sh

finish
