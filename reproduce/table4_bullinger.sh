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

# Table 4, BullingerDB block (Sec. 5.4): the two-stage recipe trained on
# the corpus' training split (Stage 1, Stage 2, then the two evaluation
# rows with the temporal nDCG under the corpus' cosine protocol).

ECFG=configs/eval_student.yaml

run_row env GPU=$GPU RUN_TAG=bullinger_teacher TRAINER=teacher \
  CONFIG=configs/stage1_bullinger_teacher.yaml DO_EVAL=0 bash run_scenario1.sh
run_row env GPU=$GPU RUN_TAG=bullinger_student TRAINER=distill \
  CONFIG=configs/stage2_bullinger_student.yaml DO_EVAL=0 bash run_scenario1.sh

BS=checkpoints/scenario1/ssl_bullinger_student/last_distill.pth
for var in std esvm_nopca; do
  extra=""; [ $var = esvm_nopca ] && extra="PCA_DIM=0 ESVM=1"
  run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=bullinger_trained VARIANT=$var TRAINER=distill \
    CONFIG=configs/stage2_bullinger_student.yaml EVAL_CONFIG=$ECFG BACKBONE=$BS \
    DISTILL_PROJ=auto TEST_DIR=data/bullinger_wr_test_colour05 \
    CODEBOOK_DIR=data/bullinger_wr_train_colour05 PCA_FIT_DIR=data/bullinger_wr_train_colour05 \
    VAL_DIR=data/bullinger_wr_train_colour05 \
    IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 $extra bash run_scenario1.sh
  # right after each variant: the nDCG reads that variant's descriptor cache
  run_row python tools/bullinger_ndcg.py --tag bullinger_trained --variant $var
done

finish
