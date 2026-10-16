#!/usr/bin/env bash
set -euo pipefail

if [ -n "${CONDA_PREFIX:-}" ]; then
  export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
  _LDP="$CONDA_PREFIX/lib"
else
  _LDP=""
fi

run_py() { LD_LIBRARY_PATH="${_LDP:+${_LDP}:}${LD_LIBRARY_PATH:-}" python "$@"; }

if [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
echo "[scenario] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset: all GPUs visible>}"

TEST_DIR="${TEST_DIR:-data/wi_comp_19_test}"
VAL_DIR="${VAL_DIR:-data/wi_comp_19_validation}"

if [[ "${TEST_DIR}" != *wi_comp_19_test* && -z "${CODEBOOK_DIR:-}${PCA_FIT_DIR:-}" ]]; then
  echo "[scenario] WARNING: TEST_DIR=${TEST_DIR} is not the HisIR19 test set but"
  echo "[scenario]          codebook/PCA/ESVM fits still default to ${VAL_DIR}"
  echo "[scenario]          (frozen-pipeline zero-shot cell). Set CODEBOOK_DIR/"
  echo "[scenario]          PCA_FIT_DIR/ESVM_NEG_DIR explicitly to adapt - and use"
  echo "[scenario]          a FRESH RUN_TAG so the desc_cache of a HisIR19 run is"
  echo "[scenario]          not overwritten."
fi

EVAL_FLAGS=(
  --input_dir      "$TEST_DIR"
  --codebook_dir   "${CODEBOOK_DIR:-$VAL_DIR}"
  --pca_dim        "${PCA_DIM:-512}"
  --pca_fit_dir    "${PCA_FIT_DIR:-$VAL_DIR}"
  --image_size     "${IMAGE_SIZE:-0}"
  --max_image_size "${MAX_IMAGE_SIZE:-4096}"
  --eval_overlap   "${OVERLAP:-0.5}"
  --binarisation   "${BINARISATION:-filter}"
)

case "${ESVM:-1}" in 1|true|yes|on)
  EVAL_FLAGS+=(--esvm --esvm_neg_dir "${ESVM_NEG_DIR:-$VAL_DIR}"
               --esvm_num_neg "${ESVM_NUM_NEG:-0}"
               --esvm_n_jobs "$(nproc 2>/dev/null || echo 8)")
  [ -n "${ESVM_C:-}" ] && EVAL_FLAGS+=(--esvm_C "${ESVM_C}") ;;
esac

if [ -n "${INK_THRESHOLD:-}" ]; then
  EVAL_FLAGS+=(--ink_threshold "${INK_THRESHOLD}")
fi
if [ -n "${INK_MASK_DIR:-}" ]; then
  EVAL_FLAGS+=(--ink_mask_dir "${INK_MASK_DIR}")
fi

case "${SIMPLE_WINDOW:-1}" in 1|true|yes|on) EVAL_FLAGS+=(--simple_window) ;; esac

[ -n "${DISTILL_PROJ:-}" ] && EVAL_FLAGS+=(--distill_proj "${DISTILL_PROJ}")
[ -n "${LABELS_CSV:-}" ] && EVAL_FLAGS+=(--labels_csv "${LABELS_CSV}")

if [ -n "${EMBED_WORKERS:-}" ]; then
  EVAL_FLAGS+=(--embed_workers "${EMBED_WORKERS}")
fi

case "${TOKEN_L2:-1}" in 0|false|no|off) EVAL_FLAGS+=(--token_l2 0) ;; esac
case "${VLAD_PRE_L2:-1}" in 0|false|no|off) EVAL_FLAGS+=(--vlad_pre_l2 0) ;; esac
case "${VLAD_RESIDUAL_NORM:-1}" in 0|false|no|off) EVAL_FLAGS+=(--vlad_residual_norm 0) ;; esac
case "${PCA_EXACT:-0}" in 1|true|yes|on) EVAL_FLAGS+=(--pca_exact) ;; esac
[ -n "${CODEBOOK_OVERLAP:-}" ] && EVAL_FLAGS+=(--codebook_overlap "${CODEBOOK_OVERLAP}")
[ -n "${CODEBOOK_MAX_TOKENS:-}" ] && EVAL_FLAGS+=(--codebook_max_tokens "${CODEBOOK_MAX_TOKENS}")
[ -n "${CODEBOOK_MAX_PER_IMAGE:-}" ] && EVAL_FLAGS+=(--codebook_max_per_image "${CODEBOOK_MAX_PER_IMAGE}")
[ -n "${CODEBOOK_ALGORITHM:-}" ] && EVAL_FLAGS+=(--codebook_algorithm "${CODEBOOK_ALGORITHM}")
[ -n "${CODEBOOK_KMEANS_N_INIT:-}" ] && EVAL_FLAGS+=(--codebook_kmeans_n_init "${CODEBOOK_KMEANS_N_INIT}")

if [ -n "${MAX_TOKENS_PER_PAGE:-}" ]; then
  EVAL_FLAGS+=(--max_tokens_per_page "${MAX_TOKENS_PER_PAGE}")
fi

case "${INTRA_NORM:-1}" in 0|false|no|off)
  EVAL_FLAGS+=(--disable_intra_norm) ;;
esac

if [ -n "${QE_K:-}" ]; then
  EVAL_FLAGS+=(--qe_k "${QE_K}")
  [ -n "${QE_ALPHA:-}" ] && EVAL_FLAGS+=(--qe_alpha "${QE_ALPHA}")
  [ -n "${QE_ITERS:-}" ] && EVAL_FLAGS+=(--qe_iters "${QE_ITERS}")
fi

if [ -n "${DESCRIPTOR_CACHE:-}" ]; then
  EVAL_FLAGS+=(--descriptor_cache "${DESCRIPTOR_CACHE}")
fi

case "${PATCH_TOKENS:-}" in
  1|true|yes|on)   EVAL_FLAGS+=(--use_patch_tokens) ;;
  0|false|no|off)  EVAL_FLAGS+=(--no-use_patch_tokens) ;;
esac

if [ -n "${VLAD_CLUSTERS:-}" ]; then
  EVAL_FLAGS+=(--vlad_clusters "${VLAD_CLUSTERS}")
fi

if [ -n "${SALAD_SUPPORT:-}" ]; then
  EVAL_FLAGS+=(--salad_support_points "${SALAD_SUPPORT}")
fi

case "${SALAD_DUSTBIN:-}" in
  1|true|yes|on)   EVAL_FLAGS+=(--salad_dustbin) ;;
  0|false|no|off)  EVAL_FLAGS+=(--no-salad_dustbin) ;;
esac
