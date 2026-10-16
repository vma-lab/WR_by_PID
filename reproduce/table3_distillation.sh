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

# Table 3 (Sec. 5.3). Stage 2 training, the student rows, and the
# controls (base configuration; HisIR19 unless stated).

# training (5 epochs each)
for tag in student student_baseline student_baseline_reproduction; do
  case $tag in
    student) cfg=configs/stage2_student.yaml ;;
    student_baseline) cfg=configs/stage2_student_baseline.yaml ;;
    student_baseline_reproduction) cfg=configs/stage2_student_baseline_reproduction.yaml ;;
  esac
  run_row env GPU=$GPU RUN_TAG=$tag TRAINER=distill CONFIG=$cfg DO_EVAL=0 bash run_scenario1.sh
done

ECFG=configs/eval_student.yaml

student_row () {  # $1 run tag  $2 checkpoint  $3 extra env...
  local tag="$1" bb="$2"; shift 2
  run_row env GPU=$GPU DO_TRAIN=0 RUN_TAG="$tag" VARIANT=std TRAINER=distill \
    CONFIG=configs/stage2_student.yaml EVAL_CONFIG=$ECFG BACKBONE="$bb" \
    IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 "$@" bash run_scenario1.sh
}

# student rows (projected tokens, through W)
student_row student_eval checkpoints/scenario1/ssl_student/last_distill.pth DISTILL_PROJ=auto
student_row baseline_eval checkpoints/scenario1/ssl_student_baseline/last_distill.pth DISTILL_PROJ=auto
student_row baseline_reproduction_eval \
  checkpoints/scenario1/ssl_student_baseline_reproduction/last_distill.pth DISTILL_PROJ=auto

# undistilled initialisation control
student_row dinov3_control checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m.pth

# representation ablation: the same final checkpoint at full width
student_row student_eval_fullwidth checkpoints/scenario1/ssl_student/last_distill.pth

# final student on Historical-WI (raw colour pages)
student_row student_eval_histwi checkpoints/scenario1/ssl_student/last_distill.pth \
  DISTILL_PROJ=auto TEST_DIR=data/wi_icdar_17_test_colour \
  CODEBOOK_DIR=data/wi_icdar_17_train_colour PCA_FIT_DIR=data/wi_icdar_17_train_colour \
  VAL_DIR=data/wi_icdar_17_train_colour

finish

# paired tests, for example the final student against the reference:
# python tools/paired_bootstrap.py \
#   results/scenario1_student_eval/LOO_vlad_patch_std_per_query.csv \
#   results/scenario1_reference/LOO_vlad_patch_std_per_query.csv
