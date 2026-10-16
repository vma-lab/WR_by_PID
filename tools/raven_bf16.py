#!/usr/bin/env python3
"""Add an optional bfloat16 autocast mode (--use_bf16) to the reference
AttMask trainer; inactive unless requested.

Two idempotent edits to attmask/main_attmask.py, both no-ops when
--use_bf16 is false, so the patched trainer behaves exactly like the
unpatched one unless bfloat16 is requested:
  1. add `--use_bf16` (bool_flag, default False);
  2. autocast dtype = bfloat16 when --use_bf16, else the original float16
     path.
bfloat16 needs no GradScaler: with --use_fp16 false the trainer's
fp16_scaler is None and its plain backward path (loss.backward();
optimizer.step()) is used.

teacher_train.py applies the edits and passes `--use_fp16 false
--use_bf16 true` when a config sets `attmask.bf16: true` (or the BF16
environment variable is set). Among the shipped configs only
configs/stage1_bullinger_teacher.yaml does so; the other Stage-1 configs
train in float16 (--use_fp16 true). Verify: the patched file py_compiles.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

MARK = "[bf16 patch]"

_ARG_ANCHOR = (
    "    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help=\"\"\"Whether or not"
)
_ARG_NEW = (
    "    parser.add_argument('--use_bf16', type=utils.bool_flag, default=False,  # %s\n"
    "        help='Autocast in bfloat16 (fp32 range, no GradScaler). Run with "
    "--use_fp16 false.')\n"
    "    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help=\"\"\"Whether or not"
) % MARK

_AC_ANCHOR = "        with torch.cuda.amp.autocast(fp16_scaler is not None):\n"
_AC_NEW = (
    "        with torch.cuda.amp.autocast(  # %s\n"
    "                enabled=(fp16_scaler is not None) or args.use_bf16,\n"
    "                dtype=torch.bfloat16 if args.use_bf16 else torch.float16):\n"
) % MARK

def _edit(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise SystemExit(
            f"[bf16] ANCHOR NOT FOUND in {path.name}:\n{old!r}\n"
            "The reference code differs from the audited revision - re-check before patching.")
    if text.count(old) != 1:
        raise SystemExit(f"[bf16] anchor not unique in {path.name}: {old!r}")
    path.write_text(text.replace(old, new))
    return True

def apply_bf16_patch(clone) -> int:
    """Add gated bf16 support to <clone>/attmask/main_attmask.py. Idempotent;
    returns the number of edits applied (0 = already patched)."""
    mp = Path(clone).expanduser() / "attmask" / "main_attmask.py"
    if not mp.exists():
        raise SystemExit(f"[bf16] {mp} not found - wrong clone path?")
    n = _edit(mp, _ARG_ANCHOR, _ARG_NEW) + _edit(mp, _AC_ANCHOR, _AC_NEW)
    py_compile.compile(str(mp), doraise=True)
    return n

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: raven_bf16.py <clone_root>")
    created = apply_bf16_patch(sys.argv[1])
    print(f"[bf16] {created} edit(s) applied (0 = already patched); main_attmask.py compiles.")
