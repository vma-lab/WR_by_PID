#!/usr/bin/env python3
"""Size the reference trainer's in-training validation table from its data.

`eval_writerretrieval` in attmask/main_attmask.py preallocates
`page_descriptors = torch.zeros((1182, 384)).cuda()` and indexes it by page
id. 1,182 is the number of Historical-WI training pages, the corpus the
validation set is built from in every HisIR19 run. On a larger training
corpus (BullingerDB: 4,396 pages) the first in-training validation stops
with an IndexError.

The edit sizes the table from the validation dataset's page registry
(`dataloader.dataset.docs`), with 1,182 as the floor. On Historical-WI the
table therefore keeps its original shape and the run is unchanged; no
training instruction is touched. teacher_train.py applies it to every clone
before launch, next to the import shims of raven_base_fixes.py.

    python tools/patch_raven_dynval.py third_party/reference_patched

Anchor-verified and idempotent: returns 0 when the clone already carries
the edit, and stops if the anchor line is not found exactly once.
"""

from __future__ import annotations

from pathlib import Path

MARK = "[dynval patch]"

_ANCHOR = "    page_descriptors = torch.zeros((1182, 384)).cuda()\n"
_NEW_LINE = "    page_descriptors = torch.zeros((max(_n_pages, 1182), 384)).cuda()\n"
_NEW = (
    "    # %s size the table from the validation dataset; 1,182 (the\n"
    "    # Historical-WI training page count) stays the floor.\n"
    "    _n_pages = len(getattr(getattr(dataloader, 'dataset', None), 'docs', []) or [])\n"
    + _NEW_LINE
) % MARK


def apply_dynval_patch(clone) -> int:
    """Patch <clone>/attmask/main_attmask.py. Idempotent; returns the number
    of edits applied (0 = already patched)."""
    mp = Path(clone).expanduser() / "attmask" / "main_attmask.py"
    if not mp.exists():
        raise SystemExit(f"[dynval] {mp} not found - wrong clone path?")
    text = mp.read_text()
    if _NEW_LINE in text:
        return 0
    if text.count(_ANCHOR) != 1:
        raise SystemExit(
            f"[dynval] anchor found {text.count(_ANCHOR)} times in {mp.name} "
            f"(expected once):\n{_ANCHOR!r}\n"
            "The reference code differs from the pinned revision - re-check before patching.")
    new = text.replace(_ANCHOR, _NEW, 1)
    compile(new, str(mp), "exec")
    mp.write_text(new)
    return 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_raven_dynval.py <clone_root>")
    n = apply_dynval_patch(sys.argv[1])
    print(f"[dynval] {n} edit(s) applied (0 = already patched).")
