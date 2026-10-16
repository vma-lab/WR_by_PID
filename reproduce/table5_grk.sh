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

# Table 5 (Sec. 5.5), GRK-Papyri. Leave-one-out over the 50 photographs;
# codebook fitted without labels on the corpus itself, no PCA (50 pages
# cannot support a 384-dimensional fit), no exemplar-SVM, query expansion
# kept. Two cells: the standard foreground gate as token selector, and
# the same pipeline with an external binary mask as the token selector
# (data/grk_papyri_mask, see data/README.md; network inputs unchanged).
# Pages are large at native resolution, so the embed worker count is kept
# low.

REF=checkpoints/reference/reference_vit_s16.pth
SCFG=configs/eval_vit_s16.yaml
ECFG=configs/eval_student.yaml
S=checkpoints/scenario1/ssl_student/last_distill.pth
D=data/grk_papyri

grk_row () {  # $1 tag, then extra env
  local tag="$1"; shift
  run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG="$tag" VARIANT=std \
    TEST_DIR=$D CODEBOOK_DIR=$D PCA_FIT_DIR=$D VAL_DIR=$D \
    PCA_DIM=0 CODEBOOK_OVERLAP=0 SIMPLE_WINDOW=1 IMAGE_SIZE=-1 \
    EMBED_WORKERS=2 "$@" bash run_scenario1.sh
}

grk_row grk_reference      CONFIG=$SCFG EVAL_CONFIG=$SCFG BACKBONE=$REF BINARISATION=input
grk_row grk_student        TRAINER=distill CONFIG=configs/stage2_student.yaml \
                           EVAL_CONFIG=$ECFG BACKBONE=$S DISTILL_PROJ=auto
grk_row grk_reference_mask CONFIG=$SCFG EVAL_CONFIG=$SCFG BACKBONE=$REF BINARISATION=input \
                           INK_MASK_DIR=data/grk_papyri_mask
grk_row grk_student_mask   TRAINER=distill CONFIG=configs/stage2_student.yaml \
                           EVAL_CONFIG=$ECFG BACKBONE=$S DISTILL_PROJ=auto \
                           INK_MASK_DIR=data/grk_papyri_mask

finish
