#!/usr/bin/env python3
"""Convert an AttMask/iBOT/DINO training checkpoint to the evaluation format.

The reference SSL_ViT_WR/attmask trainer saves ``{'student': sd, 'teacher': sd, 'optimizer': ..., ...}``
where each sd holds a DDP-wrapped MultiCropWrapper:
``module.backbone.<vit params>`` + ``module.head.<dino head>``.
The ViT parameter names (cls_token, pos_embed, patch_embed.proj.*,
blocks.N.*, norm.*) are timm-compatible, so conversion is prefix surgery:

  1. pick 'student' (the online network, evaluated throughout the paper;
     the default) or 'teacher' (the momentum encoder);
  2. strip 'module.' (DDP) and keep only 'backbone.*' (drop the DINO head);
  3. drop iBOT-only params with no timm equivalent (masked_embed / mask_token);
  4. re-prefix with 'backbone.' to match PatchTokenViT.state_dict()
     (our extvit wrapper stores the timm ViT under .backbone);
  5. save {'backbone': mapped} - the exact shape checkpoint_utils.
     load_backbone() consumes.

Verification matters: our loader uses strict=False, which would SILENTLY load
nothing on a key mismatch. --verify builds ``extvit:vit_small_patch16_224``
via model_factory.create_backbone and hard-fails unless >=90% of the model's
keys are matched by the converted dict.

Usage (from the repository root):
    python tools/convert_raven_ckpt.py /path/to/checkpoint.pth \
        checkpoints/reference/reference_vit_s16.pth --key student --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_DROP_SUFFIXES = ("masked_embed", "mask_token")


def extract_backbone_sd(ckpt: dict, key: str) -> dict:
    """Pull the ViT state dict out of a training checkpoint."""
    if key in ckpt and isinstance(ckpt[key], dict):
        sd = ckpt[key]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    else:
        raise SystemExit(
            f"[convert] checkpoint has no '{key}' entry and is not a flat "
            f"state dict; top-level keys: {sorted(ckpt.keys())[:20]}"
        )

    out = {}
    dropped_head, dropped_misc = 0, 0
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("head."):
            dropped_head += 1
            continue
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if any(k.endswith(s) or k == s for s in _DROP_SUFFIXES):
            dropped_misc += 1
            continue
        out[k] = v
    print(f"[convert] kept {len(out)} tensors "
          f"(dropped {dropped_head} head.*, {dropped_misc} mask-token params)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="AttMask/DINO training checkpoint (.pth)")
    ap.add_argument("dst", help="output path for the converted eval checkpoint")
    ap.add_argument("--key", default="student", choices=["teacher", "student"],
                    help="which network to extract (default: student, the "
                         "online network evaluated in the paper; teacher = "
                         "the momentum encoder)")
    ap.add_argument("--verify", action="store_true",
                    help="build extvit:vit_small_patch16_224 and check key coverage")
    ap.add_argument("--arch", default="extvit:vit_small_patch16_224",
                    help="model key used for --verify (default: the "
                         "reference ViT-S/16)")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"[convert] unexpected checkpoint type: {type(ckpt)}")
    print(f"[convert] loaded {args.src}; top-level keys: {sorted(ckpt.keys())[:12]}")

    vit_sd = extract_backbone_sd(ckpt, args.key)

    pe = vit_sd.get("pos_embed")
    if pe is not None:
        print(f"[convert] pos_embed shape: {tuple(pe.shape)} "
              f"(ViT-S/16 @224 expects (1, 197, 384))")
    cls = vit_sd.get("cls_token")
    if cls is not None:
        print(f"[convert] cls_token shape: {tuple(cls.shape)}")
    n_blocks = len({k.split(".")[1] for k in vit_sd if k.startswith("blocks.")})
    print(f"[convert] transformer blocks: {n_blocks} (ViT-S expects 12)")

    mapped = {f"backbone.{k}": v for k, v in vit_sd.items()}

    if args.verify:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from model_factory import create_backbone
        backbone = create_backbone(args.arch, pretrained=False,
                                   device=torch.device("cpu"))
        result = backbone.load_state_dict(mapped, strict=False)
        model_keys = set(backbone.state_dict().keys())
        matched = model_keys - set(result.missing_keys)
        frac = len(matched) / max(len(model_keys), 1)
        print(f"[verify] model keys: {len(model_keys)}  matched: {len(matched)} "
              f"({frac:.1%})  missing: {len(result.missing_keys)}  "
              f"unexpected: {len(result.unexpected_keys)}")
        if result.missing_keys:
            print(f"[verify] first missing: {result.missing_keys[:8]}")
        if result.unexpected_keys:
            print(f"[verify] first unexpected: {result.unexpected_keys[:8]}")
        if frac < 0.9:
            raise SystemExit(
                "[verify] FAILED: <90% of model keys matched - the converted "
                "dict would mostly no-op under the eval's strict=False load. "
                "Inspect the prefixes above before using this checkpoint."
            )
        print("[verify] OK - checkpoint will load into the eval pipeline.")

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": mapped}, args.dst)
    print(f"[convert] wrote {args.dst}  (format: {{'backbone': <state dict>}})")


if __name__ == "__main__":
    main()
