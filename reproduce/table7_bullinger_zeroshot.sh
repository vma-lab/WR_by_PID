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

# Table 7 (Sec. 5.5), BullingerDB zero-shot. Nothing of BullingerDB is used for
# training or fitting; codebook, PCA and exemplar-SVM negatives come
# from the HisIR19 validation set. The trained-on-target rows of
# Table 4 (Sec. 5.4) are reproduced by table4_bullinger.sh.

ECFG=configs/eval_student.yaml
S=checkpoints/scenario1/ssl_student/last_distill.pth

for var in std esvm_nopca; do
  extra=""; [ $var = esvm_nopca ] && extra="PCA_DIM=0 ESVM=1"
  run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=bullinger_zeroshot VARIANT=$var TRAINER=distill \
    CONFIG=configs/stage2_student.yaml EVAL_CONFIG=$ECFG BACKBONE=$S DISTILL_PROJ=auto \
    TEST_DIR=data/bullinger_wr_test_colour05 CODEBOOK_DIR=data/wi_comp_19_validation \
    PCA_FIT_DIR=data/wi_comp_19_validation VAL_DIR=data/wi_comp_19_validation \
    IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 $extra bash run_scenario1.sh
  # right after each variant: the nDCG reads that variant's descriptor cache
  run_row python tools/bullinger_ndcg.py --tag bullinger_zeroshot --variant $var
done

finish
