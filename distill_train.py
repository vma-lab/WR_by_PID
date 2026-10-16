#!/usr/bin/env python3
"""Cross-input (privileged-binarisation) distillation trainer (Stage 2).

A FROZEN ViT-S/16 teacher trained on binarised windows (Stage 1, white ink
on black) teaches a natural-image-pretrained DINOv3 student, fed the
unbinarised page, to produce the teacher's features. At test time the
student's input is the page without binarisation.

Per training tile (one window, two renderings):
    teacher input : hard ink mask, white ink on black, {0,1}, 3-channel, NO
                    normalisation (the reference trainer's ToTensor-only
                    regime); Sauvola on the tile by default, or the provided
                    binarised page when teacher.view_dir is set (BullingerDB);
    student input : grayscale (luma) 3-channel, ImageNet-normalised;
    loss          : mean per-token cosine distance between the aligned
                    patch-token grids (+ a CLS term), plus, when
                    aux.rkd.weight > 0, the relational term: the mean squared
                    difference of the per-tile Gram matrices of the
                    l2-normalised patch tokens of student and teacher. ViT-S/16 student matches
                    the teacher grid (14x14 @224) and dim (384) exactly; a
                    linear projection is inserted automatically when the
                    student dim differs (e.g. ViT-B/16 -> 768->384).

No EMA teacher, no centering -> the DINO uniform-collapse mode cannot occur.

Checkpoints are saved as ``{'backbone': state_dict, ...}`` - directly loadable
by the standard evaluation pipeline (leave_one_out via BACKBONE=...).

Usage:
    python distill_train.py --config configs/stage2_student.yaml
    python distill_train.py --config ... --max_steps 50     # smoke run
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ink_aug import sauvola_ink_mask
from model_factory import create_backbone
from runtime_env import resolve_device


class DistillTileDataset(Dataset):
    """Wraps a FlattenedTileDataset; emits (student_view, teacher_view)."""

    def __init__(self, tile_ds, *, mean, std, sauvola_window=51, sauvola_k=0.2,
                 sauvola_morph=3, view_dir: str | None = None):
        self.tile_ds = tile_ds
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.window = int(sauvola_window)
        self.k = float(sauvola_k)
        self.morph = int(sauvola_morph)
        self._bin_paths = None
        if view_dir:
            import re
            from PIL import Image, ImageOps
            _pat = re.compile(r'^(\d+)-(?:\d+-)?(IMG(?:_MAX)?_\d+)$')
            def _key(stem):
                m = _pat.match(stem)
                return f"{m.group(1)}-{m.group(2)}" if m else stem.replace('-', '_')
            exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
            self._bin_paths = {}
            for p in sorted(Path(view_dir).rglob('*')):
                if p.suffix.lower() in exts:
                    k = _key(p.stem)
                    if k in self._bin_paths:
                        raise RuntimeError(f"[distill] view_dir key collision: {k}")
                    self._bin_paths[k] = p
            missing, mismatched = [], []
            self._bin_by_page = {}
            for path, _ in tile_ds.dataset.samples:
                cp = Path(path)
                w = cp.parent.name
                cands = [cp.stem, f"{w}_{cp.stem}"]
                if w[:1] in ('p', 'P') and w[1:].isdigit():
                    cands.append(f"{w[1:]}_{cp.stem}")
                bp = next((self._bin_paths[k] for k in map(_key, cands)
                           if k in self._bin_paths), None)
                if bp is None:
                    missing.append(cp.stem)
                    continue
                if ImageOps.exif_transpose(Image.open(bp)).size != \
                   ImageOps.exif_transpose(Image.open(cp)).size:
                    mismatched.append(cp.stem)
                    continue
                self._bin_by_page[str(path)] = bp
            if missing or mismatched:
                raise RuntimeError(
                    f"[distill] teacher view_dir={view_dir}: "
                    f"{len(missing)} pages missing (e.g. {missing[:3]}), "
                    f"{len(mismatched)} size-mismatched (e.g. {mismatched[:3]}) "
                    f"- colour/bin pages must pair by stem at identical pixel size")
            g = np.asarray(ImageOps.exif_transpose(
                Image.open(next(iter(self._bin_paths.values())))).convert('L'))
            self._bin_ink_dark = bool(g.mean() > 127)
            self._bin_cache = {}
            self._view_loader = getattr(tile_ds.dataset, 'loader', None)
            if not hasattr(self._view_loader, 'geometry'):
                raise RuntimeError(
                    "[distill] teacher view_dir requires the dataset loader "
                    "to expose .geometry() (CV2Loader) - cannot map provided "
                    "bins into the tile coordinate frame otherwise")
            gm = self._view_loader.geometry(*g.shape[:2])
            print(f"[distill] teacher view: PROVIDED binarised pages from "
                  f"{view_dir} ({len(tile_ds.dataset.samples)} pages paired, "
                  f"sizes verified, ink={'dark' if self._bin_ink_dark else 'light'}); "
                  f"frame-mapped via loader geometry (sample page: mode="
                  f"{gm['mode']}, scale={gm['scale']:.4f}, "
                  f"pad=({gm['pad_top']},{gm['pad_left']}))")

    def _bin_ink(self, path):
        """Provided-bin ink mask IN THE LOADER FRAME (resized + padded
        exactly like the colour page the tile records index), cached."""
        arr = self._bin_cache.get(path)
        if arr is None:
            import cv2
            from PIL import Image, ImageOps
            g = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('L'))
            ink = (g < 128) if self._bin_ink_dark else (g >= 128)
            gm = self._view_loader.geometry(*ink.shape)
            if (gm['new_h'], gm['new_w']) != ink.shape:
                ink = cv2.resize(ink.astype(np.uint8) * 255,
                                 (gm['new_w'], gm['new_h']),
                                 interpolation=cv2.INTER_AREA) >= 128
            arr = np.zeros((gm['out_h'], gm['out_w']), dtype=bool)
            arr[gm['pad_top']:gm['pad_top'] + gm['new_h'],
                gm['pad_left']:gm['pad_left'] + gm['new_w']] = ink
            if len(self._bin_cache) >= 16:
                self._bin_cache.pop(next(iter(self._bin_cache)))
            self._bin_cache[path] = arr
        return arr

    def __len__(self):
        return len(self.tile_ds)

    def __getitem__(self, index):
        item = self.tile_ds[index]
        tile = item[0] if isinstance(item, (tuple, list)) else item

        gray = TF.rgb_to_grayscale(tile, num_output_channels=1)

        if self._bin_paths is not None:
            img_idx, top, left = self.tile_ds.tile_records[index]
            path, _ = self.tile_ds.dataset.samples[img_idx]
            full = self._bin_ink(str(self._bin_by_page[str(path)]))
            pw, ph = self.tile_ds._load_image(img_idx).size
            if full.shape != (ph, pw):
                raise RuntimeError(
                    f"[distill] view frame mismatch on {Path(path).stem}: "
                    f"mapped bin {full.shape} vs loaded page {(ph, pw)} - "
                    f"loader geometry out of sync")
            h, w = gray.shape[-2], gray.shape[-1]
            sub = full[top:top + h, left:left + w]
            if sub.shape != (h, w):
                tmp = np.zeros((h, w), dtype=bool)
                tmp[:sub.shape[0], :sub.shape[1]] = sub
                sub = tmp
            ink = sub
        else:
            g8 = (gray[0].clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
            ink = sauvola_ink_mask(g8, window=self.window, k=self.k,
                                   morph_open_kernel=self.morph)
        t_view = torch.from_numpy(ink.astype(np.float32))
        t_view = t_view.unsqueeze(0).repeat(3, 1, 1)

        s_view = gray.repeat(3, 1, 1)
        s_view = (s_view - self.mean) / self.std
        return s_view, t_view


def _load_state_into(wrapper: nn.Module, ckpt_path: str, device, *, tag: str,
                     min_frac: float = 0.9) -> None:
    """Load {'backbone': sd} / raw-sd checkpoints into a ViTWrapper-style
    wrapper with loud coverage reporting (strict=False would otherwise fail
    silently)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("backbone", state) if isinstance(state, dict) else state
    if not any(k.startswith("backbone.") for k in sd):
        sd = {f"backbone.{k}": v for k, v in sd.items()}
    result = wrapper.load_state_dict(sd, strict=False)
    model_keys = set(wrapper.state_dict().keys())
    matched = len(model_keys) - len(result.missing_keys)
    frac = matched / max(len(model_keys), 1)
    print(f"[distill] {tag}: loaded {matched}/{len(model_keys)} tensors "
          f"({frac:.1%}) from {ckpt_path}")
    if frac < min_frac:
        raise SystemExit(f"[distill] {tag}: <{min_frac:.0%} keys matched - "
                         "wrong checkpoint/architecture, aborting.")


def _token_cos_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cosine) between corresponding tokens of [B,N,D] tensors."""
    d = 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1)
    return d.mean()


def _build_param_groups(student: nn.Module, base_lr: float, layer_decay: float,
                        weight_decay: float, extra: list[nn.Parameter]):
    """Layer-wise LR decay over 'blocks.N.' names (deeper = higher LR)."""
    ids = set()
    for name, _ in student.named_parameters():
        parts = name.split("blocks.")
        if len(parts) > 1:
            ids.add(int(parts[1].split(".")[0]))
    n_blocks = (max(ids) + 1) if ids else 0
    groups: dict[int, list] = {}
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if "blocks." in name:
            layer = int(name.split("blocks.")[1].split(".")[0]) + 1
        elif any(s in name for s in ("patch_embed", "pos_embed", "cls_token",
                                     "mask_token", "register", "storage_tokens")):
            layer = 0
        else:
            layer = n_blocks + 1
        groups.setdefault(layer, []).append(p)
    out = []
    for layer, params in groups.items():
        scale = layer_decay ** (n_blocks + 1 - layer)
        out.append({"params": params, "lr": base_lr * scale,
                    "weight_decay": weight_decay})
    if extra:
        out.append({"params": extra, "lr": base_lr, "weight_decay": 0.0})
    return out


def _val_probe_loo(ckpt_path, cfg, run_dir, epoch: int) -> float:
    """REAL validation probe: ink-gated VLAD leave-one-out mAP on the val set,
    computed by running the actual ``leave_one_out.py`` in a subprocess on a
    saved checkpoint - the SAME code path as the full evaluation.

    The cell is deliberately light and transductive (codebook on val itself,
    no PCA, reduced resolution): a RELATIVE monitoring signal, not a test
    estimate. Config (training.val_probe): mode "loo_vlad",
    eval_config (REQUIRED, e.g. configs/eval_student.yaml), image_size
    (default 1024), overlap (0.5), ink_threshold (0.04), vlad_clusters (100),
    embed_workers (4), every (probe every N epochs, default 1).
    """
    import subprocess
    p = cfg.training.get("val_probe", {}) or {}
    eval_cfg = p.get("eval_config")
    val_dir = str(cfg.training.get("val_dir", "") or "")
    if not eval_cfg or not val_dir:
        print("[distill][warn] loo_vlad probe needs val_probe.eval_config and "
              "training.val_dir; val_map=nan.")
        return float("nan")
    out_csv = Path(run_dir) / f"probe_loo_ep{epoch:02d}.csv"
    log_path = Path(run_dir) / f"probe_loo_ep{epoch:02d}.log"
    cmd = [sys.executable, "-u", "leave_one_out.py",
           "--config", str(eval_cfg), "--model_path", str(ckpt_path),
           "--method", "vlad", "--vlad_clusters", str(int(p.get("vlad_clusters", 100))),
           "--input_dir", val_dir, "--codebook_dir", val_dir,
           "--pca_dim", "0",
           "--binarisation", "filter",
           "--ink_threshold", str(float(p.get("ink_threshold", 0.04))),
           "--eval_overlap", str(float(p.get("overlap", 0.5))),
           "--use_patch_tokens",
           "--image_size", str(int(p.get("image_size", 1024))),
           "--max_image_size", "0",
           "--embed_workers", str(int(p.get("embed_workers", 4))),
           "--output_csv", str(out_csv)]
    t0 = time.time()
    timeout_s = float(p.get("probe_timeout_min", 180)) * 60.0
    try:
        with open(log_path, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=str(Path(__file__).resolve().parent),
                                 timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[distill][warn] loo_vlad probe TIMED OUT after "
              f"{timeout_s/60:.0f} min (see {log_path}); val_map=nan.")
        return float("nan")
    if rc != 0 or not out_csv.exists():
        print(f"[distill][warn] loo_vlad probe failed (rc={rc}, see {log_path}); "
              "val_map=nan.")
        return float("nan")
    try:
        last = out_csv.read_text().strip().splitlines()[-1]
        val_map = float(last.replace("\r", "").split(",")[-1])
    except Exception as exc:
        print(f"[distill][warn] loo_vlad probe parse failed ({exc!r}); val_map=nan.")
        return float("nan")
    print(f"[distill] loo_vlad probe ep{epoch}: val_map={val_map:.4f} "
          f"({time.time() - t0:.0f}s)")
    return val_map


def main():
    ap = argparse.ArgumentParser(description="Cross-input distillation trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="smoke run: stop after N optimisation steps")
    ap.add_argument("--run_dir", default=None,
                    help="log/CSV dir (default: training.run_dir, else "
                         "runs/<checkpoint_dir name>)")
    ap.add_argument("--checkpoint_dir", default=None,
                    help="checkpoint dir (default: training.checkpoint_dir)")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)

    device = resolve_device(cfg.training.get("device", "cuda"), tag="[distill]")
    torch.manual_seed(int(cfg.training.get("seed", 42)))
    torch.backends.cudnn.benchmark = True

    mean = list(cfg.model.input_norm.mean)
    std = list(cfg.model.input_norm.std)

    tcfg = cfg.teacher
    teacher = create_backbone(str(tcfg.model), pretrained=False, device=device)
    _load_state_into(teacher, str(tcfg.checkpoint), device, tag="teacher")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = create_backbone(str(cfg.model.name),
                              pretrained=bool(cfg.model.get("pretrained", True)),
                              device=device)
    if cfg.model.get("checkpoint"):
        _load_state_into(student, str(cfg.model.checkpoint), device,
                         tag="student-init")
    student.train()

    s_dim = int(getattr(student, "embed_dim", 0) or 0)
    t_dim = int(getattr(teacher, "embed_dim", 0) or 384)
    proj = None
    extra_params: list[nn.Parameter] = []
    if s_dim and s_dim != t_dim:
        proj = nn.Linear(s_dim, t_dim).to(device)
        extra_params = list(proj.parameters())
        print(f"[distill] projection head {s_dim} -> {t_dim} (the map W)")

    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from data_loader import CV2Loader, FlattenedTileDataset
    from unfold_transform import UnfoldTransform

    t = cfg.tiling
    loader = CV2Loader(int(cfg.dataset.image_size), int(cfg.dataset.tile_size),
                       tile_overlap=float(t.overlap),
                       max_image_size=int(getattr(cfg.dataset, "max_image_size", 0) or 0))
    folder = ImageFolder(root=str(cfg.dataset.root_dir), loader=loader)
    tile_fn = transforms.Compose([
        transforms.ToTensor(),
        UnfoldTransform(int(cfg.dataset.tile_size), float(t.overlap),
                        float(t.max_padding_pct) if t.max_padding_pct is not None else None,
                        float(t.min_stroke_pct) if t.min_stroke_pct is not None else None,
                        patch_size=int(t.patch_size)),
    ])
    tile_ds = FlattenedTileDataset(folder, tile_fn, tile_transform=None,
                                   min_stroke_pct=t.min_stroke_pct)
    print(f"[distill] {len(tile_ds)} training tiles from {cfg.dataset.root_dir}")

    scfg = cfg.teacher.get("sauvola", {})
    _vdir = cfg.teacher.get("view_dir", None)
    ds = DistillTileDataset(tile_ds, mean=mean, std=std,
                            sauvola_window=int(scfg.get("window", 51)),
                            sauvola_k=float(scfg.get("k", 0.2)),
                            sauvola_morph=int(scfg.get("morph_open_kernel", 3)),
                            view_dir=(str(Path(str(_vdir)).expanduser())
                                      if _vdir else None))

    aux_cfg = OmegaConf.select(cfg, "aux") or {}
    rkd_w = float((aux_cfg.get("rkd", {}) or {}).get("weight", 0.0))
    if rkd_w > 0:
        print(f"[distill] aux losses: rkd={rkd_w}")

    probe_cfg = cfg.training.get("val_probe", {}) or {}
    probe_mode = str(probe_cfg.get("mode", "loo_vlad")).lower()
    if probe_mode != "loo_vlad":
        raise SystemExit(f"[distill] unknown training.val_probe.mode "
                         f"'{probe_mode}' (use 'loo_vlad').")
    probe_every = max(1, int(probe_cfg.get("every", 1)))

    n_workers = int(cfg.training.get("num_workers", 8))
    dl = DataLoader(ds, batch_size=int(cfg.training.batch_size), shuffle=True,
                    num_workers=n_workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=n_workers > 0)

    base_lr = float(cfg.training.lr)
    groups = _build_param_groups(student, base_lr,
                                 float(cfg.training.get("layer_decay", 0.75)),
                                 float(cfg.training.get("weight_decay", 0.04)),
                                 extra_params)
    opt = torch.optim.AdamW(groups)
    epochs = int(cfg.training.epochs)
    steps_per_epoch = len(dl)
    warmup = int(cfg.training.get("warmup_epochs", 1)) * steps_per_epoch
    total = max(epochs * steps_per_epoch, 1)

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    w_patch = float(cfg.teacher.get("loss_weights", {}).get("patch", 1.0))
    w_cls = float(cfg.teacher.get("loss_weights", {}).get("cls", 0.5))

    out_dir = Path(args.checkpoint_dir or cfg.training.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir or cfg.training.get("run_dir", "")
                   or Path("runs") / out_dir.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_snapshot.yaml")
    csv_path = run_dir / "distill_log.csv"
    print(f"[distill] run dir {run_dir}  checkpoints {out_dir}")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "patch_loss", "cls_loss",
                                "lr", "val_map", "grad_norm", "secs"])

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir))
    except Exception as exc:
        print(f"[distill] TensorBoard unavailable ({exc!r}); CSV-only logging.")

    step = 0
    val_dir = cfg.training.get("val_dir", "")
    nonfinite = 0
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]

    for epoch in range(epochs):
        t0 = time.time()
        run_l = run_p = run_c = run_g = 0.0
        run_rkd = 0.0
        n_opt = 0
        for sv, tv in dl:
            scale = lr_scale(step)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
            sv = sv.to(device, non_blocking=True)
            tv = tv.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_cls, t_pat = teacher(tv, return_patch_tokens=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_cls, s_pat = student(sv, return_patch_tokens=True)
                s_pat_raw = s_pat
                if proj is not None:
                    s_pat, s_cls = proj(s_pat), proj(s_cls)
                if s_pat.shape[1] != t_pat.shape[1]:
                    raise SystemExit(f"[distill] token-grid mismatch: student "
                                     f"{s_pat.shape} vs teacher {t_pat.shape}")
                lp = _token_cos_loss(s_pat, t_pat)
                lc = _token_cos_loss(s_cls.unsqueeze(1), t_cls.unsqueeze(1))
                loss = w_patch * lp + w_cls * lc
                l_rkd = None
                if rkd_w > 0:
                    fs = F.normalize(s_pat_raw.float(), dim=-1)
                    ft = F.normalize(t_pat.float(), dim=-1)
                    l_rkd = F.mse_loss(fs @ fs.transpose(1, 2),
                                       ft @ ft.transpose(1, 2))
                    loss = loss + rkd_w * l_rkd
            if not torch.isfinite(loss):
                nonfinite += 1
                print(f"[distill][warn] non-finite loss at step {step} "
                      f"({nonfinite}/10)")
                if nonfinite >= 10:
                    raise SystemExit("[distill] 10 non-finite steps - aborting.")
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + extra_params, 3.0)
            opt.step()
            run_l += float(loss)
            run_p += float(lp)
            run_c += float(lc)
            run_g += float(gnorm)
            if l_rkd is not None:
                run_rkd += float(l_rkd)
            n_opt += 1
            step += 1
            if step % 100 == 0 or step == 10:
                peak_lr = max(g['lr'] for g in opt.param_groups)
                aux_str = ""
                if rkd_w > 0:
                    aux_str += f" rkd {run_rkd / max(n_opt, 1):.4f}"
                print(f"[distill] ep {epoch} step {step}/{total} "
                      f"loss {run_l / max(n_opt, 1):.4f}{aux_str} "
                      f"lr {peak_lr:.2e}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        n = max(n_opt, 1)
        val_map = float("nan")
        secs = time.time() - t0
        peak_lr = max(g["lr"] for g in opt.param_groups)
        print(f"[distill] EPOCH {epoch}: loss={run_l / n:.4f} "
              f"patch={run_p / n:.4f} cls={run_c / n:.4f} "
              f"val_map={val_map:.4f} grad={run_g / n:.2f} ({secs:.0f}s)",
              flush=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, run_l / n, run_p / n, run_c / n,
                                    peak_lr, val_map, run_g / n, int(secs)])
        if writer is not None:
            writer.add_scalar("loss/total", run_l / n, epoch)
            writer.add_scalar("loss/patch", run_p / n, epoch)
            writer.add_scalar("loss/cls", run_c / n, epoch)
            writer.add_scalar("align/patch_cos", 1.0 - run_p / n, epoch)
            writer.add_scalar("align/cls_cos", 1.0 - run_c / n, epoch)
            writer.add_scalar("lr", peak_lr, epoch)
            writer.add_scalar("grad_norm", run_g / n, epoch)
            if rkd_w > 0:
                writer.add_scalar("loss/aux_rkd", run_rkd / n, epoch)
            if not math.isnan(val_map):
                writer.add_scalar("probe/val_map", val_map, epoch)

        state = {"backbone": student.state_dict(),
                 "distill_meta": {"teacher": str(tcfg.checkpoint),
                                  "epoch": epoch, "val_map": val_map}}
        if proj is not None:
            state["distill_proj"] = proj.state_dict()
        torch.save(state, out_dir / "last_distill.pth")
        if val_dir and not args.max_steps and epoch % probe_every == 0:
            val_map = _val_probe_loo(out_dir / "last_distill.pth", cfg,
                                     run_dir, epoch)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([f"{epoch}-loo", "", "", "", "",
                                        val_map, "", ""])
            if writer is not None and not math.isnan(val_map):
                writer.add_scalar("probe/val_loo_map", val_map, epoch)
        if args.max_steps and step >= args.max_steps:
            print("[distill] max_steps smoke run complete.")
            break

    if writer is not None:
        writer.close()
    print(f"[distill] DONE. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
