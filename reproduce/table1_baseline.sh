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

# Table 1 (Sec. 3). The evaluation ladder on the reference checkpoint,
# the base configuration, and the from-scratch reproduction. The ladder
# rows print the exact-PCA variant (VARIANT=std, PCA_EXACT=1); the
# randomised-PCA variants (VARIANT=rpca) of r0 and r2 are the solver check
# of Appendix A, and the r2 rpca row is the pairing partner of the
# codebook-fit step.

REF=checkpoints/reference/reference_vit_s16.pth
SCFG=configs/eval_vit_s16.yaml
EXACT="TOKEN_L2=0 VLAD_PRE_L2=0 VLAD_RESIDUAL_NORM=0 INTRA_NORM=0 INK_THRESHOLD=0.0390625"
FIT="CODEBOOK_MAX_TOKENS=0 CODEBOOK_MAX_PER_IMAGE=0 CODEBOOK_ALGORITHM=kmeans CODEBOOK_KMEANS_N_INIT=1"

# r0: the publication's own evaluation
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=table1_r0 VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=0 \
  CODEBOOK_OVERLAP=0 PCA_EXACT=1 $EXACT $FIT bash run_scenario1.sh

run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=table1_r0 VARIANT=rpca CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=0 \
  CODEBOOK_OVERLAP=0 PCA_EXACT=0 $EXACT $FIT bash run_scenario1.sh

# r1: all windows kept, as in the released reference code
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=table1_r1 VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 \
  CODEBOOK_OVERLAP=0 PCA_EXACT=1 $EXACT $FIT bash run_scenario1.sh

# r2: plus our VLAD normalisation
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=table1_r2 VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 \
  CODEBOOK_OVERLAP=0 PCA_EXACT=1 $FIT bash run_scenario1.sh

run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=table1_r2 VARIANT=rpca CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 \
  CODEBOOK_OVERLAP=0 PCA_EXACT=0 $FIT bash run_scenario1.sh

# base configuration (plus our codebook fit), HisIR19
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=reference VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 \
  CODEBOOK_OVERLAP=0 bash run_scenario1.sh

# base configuration, Historical-WI (provided binarised pages, inverted)
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=reference_histwi VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 \
  TEST_DIR=data/wi_icdar_17_test_bin_inv CODEBOOK_DIR=data/wi_icdar_17_train_bin_inv \
  PCA_FIT_DIR=data/wi_icdar_17_train_bin_inv VAL_DIR=data/wi_icdar_17_train_bin_inv \
  bash run_scenario1.sh

# from-scratch reproduction (train with reproduce/table2_teacher.sh first)
REPRO=checkpoints/scenario1/ssl_teacher_reproduction/last_teacher.pth
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=reproduction VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REPRO BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 \
  CODEBOOK_OVERLAP=0 bash run_scenario1.sh
run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=reproduction_histwi VARIANT=std CONFIG=$SCFG EVAL_CONFIG=$SCFG \
  BACKBONE=$REPRO BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 \
  TEST_DIR=data/wi_icdar_17_test_bin_inv CODEBOOK_DIR=data/wi_icdar_17_train_bin_inv \
  PCA_FIT_DIR=data/wi_icdar_17_train_bin_inv VAL_DIR=data/wi_icdar_17_train_bin_inv \
  bash run_scenario1.sh

finish

# paired steps, for example r1 against r0:
# python tools/paired_bootstrap.py \
#   results/scenario1_table1_r1/LOO_vlad_patch_std_per_query.csv \
#   results/scenario1_table1_r0/LOO_vlad_patch_std_per_query.csv
