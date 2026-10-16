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

# Appendix A. The publication's minimum-stroke window rule, measured on
# the reference checkpoint on HisIR19, in the base and the full
# configuration. SIMPLE_WINDOW=0 with configs/eval_vit_s16.yaml keeps a
# window when the ink pixels make up at least tiling.binary_ink_threshold
# (0.025) of the bounding box of the window's ink.

REF=checkpoints/reference/reference_vit_s16.pth
SCFG=configs/eval_vit_s16.yaml

for var in std esvm_nopca; do
  extra=""; [ $var = esvm_nopca ] && extra="PCA_DIM=0 ESVM=1"
  run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG=reference_rule VARIANT=$var CONFIG=$SCFG EVAL_CONFIG=$SCFG \
    BACKBONE=$REF BINARISATION=input IMAGE_SIZE=-1 SIMPLE_WINDOW=0 \
    CODEBOOK_OVERLAP=0 $extra bash run_scenario1.sh
done

finish

# python tools/paired_bootstrap.py \
#   results/scenario1_reference_rule/LOO_vlad_patch_std_per_query.csv \
#   results/scenario1_reference/LOO_vlad_patch_std_per_query.csv
