#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_URL="${REPO_URL:-https://github.com/Traven16/SSL_ViT_WR.git}"
SHA=176313767bbf650876cf0cd5080c0310755e9c81

if [ ! -d reference ]; then
  git clone "$REPO_URL" reference
  git -C reference checkout "$SHA"
fi
if [ "$(git -C reference rev-parse HEAD)" != "$SHA" ]; then
  echo "third_party/reference is not at commit $SHA; remove it and re-run" >&2
  exit 1
fi
if [ ! -d reference_patched ]; then
  cp -r reference reference_patched
  python ../tools/patch_raven_supcon.py reference_patched
fi
echo "reference code ready: reference/ (unmodified), reference_patched/ (supcon)"
