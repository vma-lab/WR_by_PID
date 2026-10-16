#!/usr/bin/env python3
"""Import shims the reference SSL_ViT_WR/attmask trainer needs to run.

These are not training-logic changes: they add files the reference
repository does not ship, so that `main_attmask.py` can be imported at
all. The gradient loop and every training instruction stay unchanged:

  * icdar2017loader.py - the trainer's loader.py does `from icdar2017loader
    import icdar2017patcher`, but the repository only ships
    `docloader.py`. The shim is an exact-copy alias (same class).
  * evaluation/ stubs - main_attmask.py imports eval_knn / unsup_cls at
    module load for an online-kNN path that is never enabled (--eval off;
    the kNN loader is commented out in the reference source).
    NotImplementedError stubs satisfy the import without adding behaviour.

teacher_train.py applies them to any clone before launch, and
patch_raven_supcon.py applies them too, so a fresh clone needs no manual
step.

Idempotent - returns the number of files newly created (0 = already present).
"""

from __future__ import annotations

import shutil
from pathlib import Path

_EVAL_STUBS = [
    ("evaluation/__init__.py", ""),
    ("evaluation/unsupervised/__init__.py", ""),
    ("evaluation/eval_knn.py",
     "def extract_features(*a, **k):\n    raise NotImplementedError\n"
     "def knn_classifier(*a, **k):\n    raise NotImplementedError\n"),
    ("evaluation/unsupervised/unsup_cls.py",
     "def eval_pred(*a, **k):\n    raise NotImplementedError\n"),
]

def apply_base_fixes(clone) -> int:
    """Ensure `clone`/attmask can be imported. `clone` is the repo root
    (the dir that contains attmask/main_attmask.py). Returns files created."""
    am = Path(clone).expanduser() / "attmask"
    if not (am / "main_attmask.py").exists():
        raise SystemExit(
            f"[base-fix] {am}/main_attmask.py not found - wrong clone path?")
    n = 0

    if not (am / "icdar2017loader.py").exists():
        shutil.copy2(am / "docloader.py", am / "icdar2017loader.py")
        n += 1

    (am / "evaluation" / "unsupervised").mkdir(parents=True, exist_ok=True)
    for rel, body in _EVAL_STUBS:
        p = am / rel
        if not p.exists():
            p.write_text(body)
            n += 1
    return n

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: raven_base_fixes.py <clone_root>")
    created = apply_base_fixes(sys.argv[1])
    print(f"[base-fix] {created} file(s) created (0 = already present).")
