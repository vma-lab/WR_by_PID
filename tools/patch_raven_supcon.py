#!/usr/bin/env python3
"""Patch a copy of the reference AttMask trainer with the page-level
supervised-contrastive (page-SupCon) auxiliary loss of Stage 1, with
spatially disjoint positives.

WHAT IT ADDS (all gated on --page_supcon_weight > 0 or
--page_supcon_sampler_only; at the defaults the patched trainer's training
math is byte-equivalent to the unpatched one):
  * page-grouped batches: each batch = pages_per_batch x windows_per_page
    windows, positives selected per page with pairwise window IoU <=
    max_positive_iou (0.0 = strictly non-overlapping). At the reference
    trainer's --overlap 0.875, adjacent training windows share ~87% of
    their pixels, so random same-page positives would be near-duplicates
    and page-SupCon would reward shared pixels rather than the writer's
    hand; positives are therefore disjoint by default.
  * a SupCon term over the online network's backbone CLS features of the
    global crops, keyed on page id, added to the iBOT/AttMask loss with
    weight --page_supcon_weight. Everything else (masking, DINO/iBOT heads,
    schedules) is untouched.

USAGE: normally run once by `bash third_party/setup_reference.sh`, which
copies the pinned clone to third_party/reference_patched and calls

    python tools/patch_raven_supcon.py third_party/reference_patched

The patcher is idempotent and also applies the import shims of
raven_base_fixes.py. Verify: the script py_compiles every patched file.

New CLI args added to main_attmask.py:
  --page_supcon_weight (0.0=off)  --supcon_temperature (0.1)
  --pages_per_batch (16)  --windows_per_page (8)   [16*8 = the reference
      batch of 128]
  --max_positive_iou (0.0)
  --page_supcon_sampler_only (false) - the sampler-only control: installs
      the page-grouped disjoint batch sampler with the auxiliary loss off
      (weight 0), separating the batch-composition change from the SupCon
      gradient (configs/stage1_teacher_control.yaml).

NB: re-running an updated patcher over an already-patched clone creates a
mixed-version clone (duplicate argparse block). Rebuild from scratch
instead: rm -rf third_party/reference_patched && bash
third_party/setup_reference.sh.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raven_base_fixes import apply_base_fixes

MARK = "[supcon patch]"

PAGE_SUPCON_SRC = '''\
"""Page-SupCon auxiliary machinery for the AttMask trainer. [supcon patch]
Batch sampler is torch-free (duck-typed batch_sampler: iterable of index
lists + __len__ + set_epoch); the loss uses torch."""

import random

import torch
import torch.nn.functional as F


def _iou(a, b, ws):
    """IoU of two ws-sized square windows given (y, x) corners."""
    iy = max(0, ws - abs(a[0] - b[0]))
    ix = max(0, ws - abs(a[1] - b[1]))
    inter = iy * ix
    return inter / float(2 * ws * ws - inter)


class PageGroupBatchSampler:
    """Batches of pages_per_batch pages x windows_per_page windows, windows
    within a page greedily chosen with pairwise IoU <= max_positive_iou
    (when a page cannot supply enough disjoint windows, the batch is topped
    up with the page's remaining windows in their shuffled draw order, so
    the batch size stays constant)."""

    def __init__(self, patcher, pages_per_batch, windows_per_page,
                 max_positive_iou=0.0, seed=0):
        self.P = int(pages_per_batch)
        self.G = int(windows_per_page)
        self.max_iou = float(max_positive_iou)
        self.seed = int(seed)
        self.epoch = 0
        self.ws = int(patcher.window_size)
        self.by_page = {}
        for idx, (fname, coord) in enumerate(patcher.patch_coords):
            pid = patcher.docs[fname]["page_id"]
            self.by_page.setdefault(pid, []).append((idx, coord))
        self.pages = sorted(self.by_page)
        self.n_batches = max(1, len(patcher.patch_coords) // (self.P * self.G))
        short = sum(1 for p in self.pages if len(self.by_page[p]) < self.G)
        print(f"[supcon] sampler: {len(self.pages)} pages, "
              f"{self.n_batches} batches of {self.P}x{self.G}, "
              f"max_iou={self.max_iou} ({short} pages have <{self.G} windows)")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _pick(self, rng, cands):
        """Greedy IoU-constrained pick of up to G windows from one page."""
        cands = cands[:]
        rng.shuffle(cands)
        kept, rest = [], []
        for idx, coord in cands:
            if len(kept) == self.G:
                break
            if all(_iou(coord, kc, self.ws) <= self.max_iou for _, kc in kept):
                kept.append((idx, coord))
            else:
                rest.append((idx, coord))
        for item in rest:                      # top-up: batch size must hold
            if len(kept) == self.G:
                break
            kept.append(item)
        while len(kept) < self.G:              # top-up to the fixed batch width
            kept_ids = {i for i, _ in kept}
            fresh = [c for c in cands if c[0] not in kept_ids]
            # a page with < G distinct windows must repeat one; draw an unused
            # window first so we don't create a same-window (max-sim) positive
            # pair while other windows of the page go unused.
            kept.append(rng.choice(fresh) if fresh else rng.choice(cands))
        return [idx for idx, _ in kept]

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        order = self.pages[:]
        rng.shuffle(order)
        pos = 0
        for _ in range(self.n_batches):
            batch = []
            for _ in range(self.P):
                if pos >= len(order):
                    rng.shuffle(order)
                    pos = 0
                page = order[pos]
                pos += 1
                batch.extend(self._pick(rng, self.by_page[page]))
            yield batch


def page_supcon_loss(feats, labels, temperature=0.1):
    """Supervised-contrastive loss over L2-normalised features, positives =
    same label (page id). Anchors with no positive are skipped."""
    f = F.normalize(feats.float(), dim=1)
    sim = (f @ f.t()) / float(temperature)
    n = sim.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    pos = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if not bool(valid.any()):
        return feats.sum() * 0.0
    loss = -(log_prob.masked_fill(~pos, 0.0).sum(1)[valid]
             / pos_cnt[valid].float())
    return loss.mean()
'''


def _edit(path: Path, old: str, new: str, *, must: bool = True) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        if must:
            raise SystemExit(f"[patch] ANCHOR NOT FOUND in {path.name}:\n{old!r}\n"
                             "The reference code differs from the audited revision - "
                             "re-check before patching.")
        return False
    if text.count(old) != 1:
        raise SystemExit(f"[patch] anchor not unique in {path.name}: {old!r}")
    path.write_text(text.replace(old, new))
    return True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    clone = Path(sys.argv[1]).expanduser()
    am = clone / "attmask"
    if not (am / "main_attmask.py").exists():
        raise SystemExit(f"[patch] {am}/main_attmask.py not found - wrong path?")

    created = apply_base_fixes(clone)
    if created:
        print(f"[patch] base fix: {created} import-shim file(s) created")

    (am / "page_supcon.py").write_text(PAGE_SUPCON_SRC)
    print("[patch] wrote page_supcon.py")

    n = 0

    n += _edit(am / "loader.py",
               "        return output, writer\n",
               "        return output, writer, page  # %s\n" % MARK)

    n += _edit(am / "utils.py",
               "        # Run the head forward on the concatenated features.\n"
               "        output_ = self.head(output)\n",
               "        # Run the head forward on the concatenated features.\n"
               "        # %s stash backbone CLS for the page-SupCon aux loss\n"
               "        self.last_cls_feat = output[:, 0] if output.dim() == 3 else output\n"
               "        output_ = self.head(output)\n" % MARK)

    mp = am / "main_attmask.py"

    n += _edit(mp,
               "from docloader import icdar2017patcher\n",
               "from docloader import icdar2017patcher\n"
               "from page_supcon import PageGroupBatchSampler, page_supcon_loss  # %s\n" % MARK)

    n += _edit(mp,
               "    parser.add_argument('--batch_size_per_gpu', default=128, type=int,\n",
               "    # %s\n"
               "    parser.add_argument('--page_supcon_weight', default=0.0, type=float,\n"
               "        help='Weight of the page-SupCon auxiliary loss (0 = off).')\n"
               "    parser.add_argument('--supcon_temperature', default=0.1, type=float)\n"
               "    parser.add_argument('--pages_per_batch', default=16, type=int)\n"
               "    parser.add_argument('--windows_per_page', default=8, type=int)\n"
               "    parser.add_argument('--max_positive_iou', default=0.0, type=float,\n"
               "        help='Max pairwise IoU between same-page positives (0 = disjoint).')\n"
               "    parser.add_argument('--page_supcon_sampler_only', default=False,\n"
               "        type=utils.bool_flag,\n"
               "        help='Control run: page-grouped batches WITHOUT the aux loss '\n"
               "             '(weight stays 0) - isolates batch composition.')\n"
               "    parser.add_argument('--batch_size_per_gpu', default=128, type=int,\n" % MARK)

    n += _edit(mp,
               "    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)\n",
               "    # %s page-grouped batches when the aux loss is on (or the\n"
               "    # sampler-only mechanistic control is requested)\n"
               "    if args.page_supcon_weight > 0 or args.page_supcon_sampler_only:\n"
               "        assert args.pages_per_batch * args.windows_per_page == args.batch_size_per_gpu, \\\n"
               "            'pages_per_batch * windows_per_page must equal batch_size_per_gpu'\n"
               "        assert utils.get_world_size() == 1, \\\n"
               "            'page_supcon batch_sampler is single-GPU only (no rank sharding); '\\\n"
               "            'launch the trainer with --nproc_per_node=1'\n"
               "        batch_sampler_train = PageGroupBatchSampler(\n"
               "            dataset_train.dataset, args.pages_per_batch, args.windows_per_page,\n"
               "            max_positive_iou=args.max_positive_iou, seed=args.seed)\n"
               "        sampler_train = None\n"
               "    else:\n"
               "        batch_sampler_train = None\n"
               "        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)\n" % MARK)

    n += _edit(mp,
               "    data_loader_train = torch.utils.data.DataLoader(\n"
               "        dataset_train,\n"
               "        sampler=sampler_train,\n"
               "        batch_size=args.batch_size_per_gpu,\n"
               "        num_workers=args.num_workers,\n"
               "        pin_memory=True,\n"
               "        drop_last=True\n"
               "    )\n",
               "    if batch_sampler_train is not None:  # %s\n"
               "        data_loader_train = torch.utils.data.DataLoader(\n"
               "            dataset_train,\n"
               "            batch_sampler=batch_sampler_train,\n"
               "            num_workers=args.num_workers,\n"
               "            pin_memory=True,\n"
               "        )\n"
               "    else:\n"
               "        data_loader_train = torch.utils.data.DataLoader(\n"
               "            dataset_train,\n"
               "            sampler=sampler_train,\n"
               "            batch_size=args.batch_size_per_gpu,\n"
               "            num_workers=args.num_workers,\n"
               "            pin_memory=True,\n"
               "            drop_last=True\n"
               "        )\n" % MARK)

    n += _edit(mp,
               "        data_loader_train.sampler.set_epoch(epoch)\n",
               "        for _s in (getattr(data_loader_train, 'batch_sampler', None),  # %s\n"
               "                   data_loader_train.sampler):\n"
               "            if _s is not None and hasattr(_s, 'set_epoch'):\n"
               "                _s.set_epoch(epoch)\n" % MARK)

    n += _edit(mp,
               "    for it, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):\n",
               "    for it, (images, labels, pages) in enumerate(metric_logger.log_every(data_loader, 10, header)):  # %s\n" % MARK)

    n += _edit(mp,
               "            student_output = student(images[:args.global_crops_number], mask=masks)#, keep_ids=ids)\n",
               "            student_output = student(images[:args.global_crops_number], mask=masks)#, keep_ids=ids)\n"
               "            supcon_feats = student.module.last_cls_feat if args.page_supcon_weight > 0 else None  # %s\n" % MARK)

    n += _edit(mp,
               "            loss = all_loss.pop('loss')\n",
               "            loss = all_loss.pop('loss')\n"
               "            if args.page_supcon_weight > 0 and supcon_feats is not None:  # %s\n"
               "                sc_lab = pages.cuda(non_blocking=True).repeat(args.global_crops_number)\n"
               "                sc = page_supcon_loss(supcon_feats, sc_lab,\n"
               "                                      temperature=args.supcon_temperature)\n"
               "                loss = loss + args.page_supcon_weight * sc\n"
               "                all_loss['page_supcon'] = sc.detach()\n" % MARK)

    for f in ["main_attmask.py", "loader.py", "utils.py", "page_supcon.py"]:
        py_compile.compile(str(am / f), doraise=True)
    print(f"[patch] OK: {n} new edit(s) applied (0 = already patched); "
          "all patched files py_compile clean.")
    print("[patch] Train with the aux loss via teacher_train.py extra_args "
          "(configs/stage1_teacher.yaml) - weight 0.0 keeps the "
          "training math unchanged.")


if __name__ == "__main__":
    main()
