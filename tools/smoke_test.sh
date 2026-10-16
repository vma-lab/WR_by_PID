#!/usr/bin/env bash
# Release smoke test (CPU only; needs neither data nor weights):
#   1. shell syntax of every script and byte-compilation of every module;
#   2. the evaluation entry point outside a conda environment;
#   3. the paired bootstrap on shipped per-query results (expected values);
#   4. the Stage-1 command for every configs/stage1_*.yaml and the
#      validation-table patch of the reference trainer;
#   5. imports of the main modules and checks of every shipped config.
#
#   bash tools/smoke_test.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail () { echo "[smoke] FAIL: $*" >&2; exit 1; }

echo "[smoke] 1/5 shell syntax and byte-compilation"
for f in *.sh reproduce/*.sh third_party/*.sh tools/*.sh; do
  bash -n "$f" || fail "bash -n $f"
done
python - <<'PY'
import os
skip = {"data", "checkpoints", "runs", "results", "reference",
        "reference_patched", "dinov3", "__pycache__", ".git", ".venv",
        "venv", "env"}
n = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in skip]
    for name in sorted(files):
        if name.endswith(".py"):
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                compile(f.read(), path, "exec")
            n += 1
print(f"[smoke]   {n} Python files compile")
PY

echo "[smoke] 2/5 run_scenario1.sh without conda"
env -u CONDA_PREFIX DO_TRAIN=0 DO_EVAL=0 RUN_TAG=smoke OUT_DIR="$TMP/out" \
  bash run_scenario1.sh > "$TMP/noconda.log" 2>&1 \
  || { cat "$TMP/noconda.log"; fail "run_scenario1.sh outside conda"; }

echo "[smoke] 3/5 paired bootstrap on shipped results"
pair () {  # pair A B "flags" expected-substring...
  local a="$1" b="$2" flags="$3" out; shift 3
  # shellcheck disable=SC2086
  out="$(python tools/paired_bootstrap.py "results/$a.csv" "results/$b.csv" $flags)" \
    || fail "paired_bootstrap $a $b $flags"
  for s in "$@"; do
    grep -qF -- "$s" <<<"$out" || { echo "$out"; fail "$a - $b $flags: expected '$s'"; }
  done
  echo "[smoke]   $a - $b ${flags:+($flags) }ok"
}
pair hisir19_student_base hisir19_reference_base "" \
     "+1.009 pt" "[+0.845, +1.172]" "p = <1e-4" "1469/816"
pair histwi_student_base histwi_reference_base "" "+0.257 pt" "p = 0.0485"
pair histwi_student_base histwi_teacher_base "" "+0.037 pt" "p = 0.7569"
pair histwi_student_full histwi_reference_full "" "+0.127 pt" "p = 0.2667"
pair hisir19_student_base hisir19_reference_base "--cluster writer" \
     "from 2568 writers" "+1.009 pt" "[+0.736, +1.284]"
pair histwi_student_base histwi_reference_base "--cluster writer" \
     "from 720 writers" "+0.257 pt"
if python tools/paired_bootstrap.py results/hisir19_student_base.csv \
     results/histwi_student_base.csv > "$TMP/mismatch.log" 2>&1; then
  fail "paired_bootstrap accepted two different query sets"
fi
grep -qF "not a valid pair" "$TMP/mismatch.log" || fail "mismatch message"

echo "[smoke] 4/5 Stage-1 command and reference-trainer patch"
python - "$TMP" <<'PY'
import glob
import shutil
import sys
import types
from pathlib import Path

tmp = Path(sys.argv[1])
sys.path.insert(0, ".")
import teacher_train as tt
from patch_raven_dynval import apply_dynval_patch

repo = tt.REPO
for path in sorted(glob.glob("configs/stage1_*.yaml")):
    am = tt._load_cfg(path)["attmask"]
    cmd, cwd = tt._build_attmask_cmd(am, tmp / "out")
    data = Path(cmd[cmd.index("--data_path") + 1])
    assert data.is_absolute() and data == (repo / am["data_path"]).resolve(), data
    assert cwd == (repo / am["clone"]).resolve() / "attmask", cwd
    assert am["clone"] in ("third_party/reference",
                           "third_party/reference_patched"), am["clone"]
    state = "present" if data.is_dir() else "not prepared yet"
    print(f"[smoke]   {path}: --data_path {data.relative_to(repo)} ({state})")
(tmp / "empty").mkdir()
try:
    tt._check_data({"data_path": str(tmp / "empty")})
    raise AssertionError("an empty training directory was accepted")
except SystemExit:
    pass

# The validation-table patch on a minimal copy of the patched function:
# 1,182 rows on Historical-WI (and below), the page count above it.
src = ("import torch\n"
       "def eval_writerretrieval(model, dataloader, log_as=''):\n"
       "    page_descriptors = torch.zeros((1182, 384)).cuda()\n"
       "    return page_descriptors\n")
(tmp / "clone" / "attmask").mkdir(parents=True)
(tmp / "clone" / "attmask" / "main_attmask.py").write_text(src)
assert apply_dynval_patch(tmp / "clone") == 1
assert apply_dynval_patch(tmp / "clone") == 0


class _Rows:
    def __init__(self, shape):
        self.shape = shape

    def cuda(self):
        return self


ns = {}
code = (tmp / "clone" / "attmask" / "main_attmask.py").read_text()
saved = sys.modules.get("torch")
sys.modules["torch"] = types.SimpleNamespace(zeros=_Rows)
try:
    exec(compile(code, "main_attmask.py", "exec"), ns)
finally:
    if saved is not None:
        sys.modules["torch"] = saved
    else:
        del sys.modules["torch"]
for pages, rows in ((1182, 1182), (394, 1182), (4396, 4396)):
    dl = types.SimpleNamespace(dataset=types.SimpleNamespace(
        docs={i: None for i in range(pages)}))
    assert ns["eval_writerretrieval"](None, dl).shape == (rows, 384), pages
print("[smoke]   validation-table patch: 1,182 rows kept, larger corpora sized")

for name in ("reference", "reference_patched"):
    real = Path("third_party") / name / "attmask" / "main_attmask.py"
    if not real.exists():
        print(f"[smoke]   third_party/{name}: not set up (skipped)")
        continue
    dst = tmp / name / "attmask"
    dst.mkdir(parents=True)
    shutil.copy2(real, dst / "main_attmask.py")
    apply_dynval_patch(tmp / name)
    assert apply_dynval_patch(tmp / name) == 0
    print(f"[smoke]   third_party/{name}: patch anchors verified")
PY

echo "[smoke] 5/5 module imports and config checks"
python - <<'PY'
import glob
import importlib
import os
from pathlib import Path

from omegaconf import OmegaConf

mods = ["leave_one_out", "distill_train", "teacher_train", "model_factory",
        "data_loader", "vlad_codebook_utils", "unfold_transform",
        "inference_utils", "checkpoint_utils", "ink_aug", "runtime_env"]
for m in mods:
    importlib.import_module(m)
print(f"[smoke]   {len(mods)} modules import")


def need(cfg, path, keys):
    for k in keys:
        if OmegaConf.select(cfg, k) is None:
            raise SystemExit(f"[smoke] FAIL: {path} lacks {k}")


def need_file(path, f):
    if not os.path.isfile(f):
        raise SystemExit(f"[smoke] FAIL: {path} names a missing file {f}")


for path in sorted(glob.glob("configs/*.yaml")):
    cfg = OmegaConf.load(path)
    OmegaConf.to_container(cfg, resolve=True)
    name = Path(path).name
    if name.startswith("stage1_"):
        need(cfg, path, ["attmask.clone", "attmask.data_path", "attmask.epochs",
                         "attmask.lr", "probe.val_dir", "probe.eval_config"])
        need_file(path, cfg.probe.eval_config)
    else:
        need(cfg, path, ["model.name", "model.input_norm.mean",
                         "model.input_norm.std", "dataset.image_size",
                         "dataset.tile_size", "tiling.patch_size"])
    if name.startswith("stage2_"):
        need(cfg, path, ["teacher.model", "teacher.checkpoint",
                         "training.epochs", "training.batch_size",
                         "training.lr", "training.checkpoint_dir",
                         "training.val_probe.eval_config", "dataset.root_dir"])
        need_file(path, cfg.training.val_probe.eval_config)
    print(f"[smoke]   {path}: ok")
PY

echo "smoke test passed"
