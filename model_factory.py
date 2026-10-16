import warnings
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_ink_fraction(
    images: torch.Tensor,
    grid_h: int,
    grid_w: int,
    mean,
    std,
    *,
    window: int = 51,
    k: float = 0.2,
    morph_open: int = 3,
) -> Optional[torch.Tensor]:
    """Per-patch ink fraction of a NORMALISED input batch, aligned with the ViT
    patch grid.

    This reproduces the dataset binariser (``data_loader._apply_binarisation``)
    on-device, so the ink-count foreground gate (Raven et al., Eq. 2) can be
    driven from the exact tile the ViT sees (converted to grey for the
    threshold) - no separate binary input, no dataset/augmentation plumbing,
    and perfect patch alignment.

    Steps: de-normalise -> grayscale 0-255 -> Sauvola threshold via box
    filters -> ink mask (ink = dark) -> optional morphological opening
    (despeckle) -> average-pool to ``[grid_h, grid_w]`` -> flatten row-major
    to ``[B, grid_h*grid_w]`` in ``[0, 1]`` (the ink fraction of each patch).

    Note on borders: the local mean is taken over a REPLICATE-padded window
    (the form all reported results use), whereas the reference C++
    implementations use the clipped-window mean.  The two differ only in the
    ``window//2`` rim of each tile.

    Returns ``None`` on any failure; with the ink gate requested, the
    evaluation and codebook callers then stop with an error instead of
    running without the gate.
    """
    try:
        B, C, H, W = images.shape
        device = images.device
        m = torch.as_tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        s = torch.as_tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        raw = images.to(torch.float32) * s + m
        gray = (raw.mean(dim=1, keepdim=True).clamp(0.0, 1.0)) * 255.0

        win = int(window) | 1
        pad = win // 2
        gpad = F.pad(gray, (pad, pad, pad, pad), mode="replicate")
        mean_loc = F.avg_pool2d(gpad, win, stride=1)
        sq = F.avg_pool2d(gpad * gpad, win, stride=1)
        std_loc = (sq - mean_loc * mean_loc).clamp_min(0.0).sqrt()
        thr = mean_loc * (1.0 + float(k) * (std_loc / 128.0 - 1.0))
        ink = (gray < thr).to(torch.float32)

        mk = int(morph_open or 0)
        if mk > 1:
            p = mk // 2
            eroded = -F.max_pool2d(F.pad(-ink, (p, p, p, p), value=0.0), mk, stride=1)
            ink = F.max_pool2d(F.pad(eroded, (p, p, p, p), value=0.0), mk, stride=1)

        ph, pw = max(1, H // int(grid_h)), max(1, W // int(grid_w))
        frac = F.avg_pool2d(ink, (ph, pw), stride=(ph, pw))
        return frac[:, 0, : int(grid_h), : int(grid_w)].reshape(B, int(grid_h) * int(grid_w))
    except Exception:
        return None


class ViTWrapper(nn.Module):
    """Base wrapper of the ViT backbones (DINOv3 student, timm teacher).

    Holds the on-device ink-count gate hook (``ink_foreground``; configured by
    ``inference_utils.configure_ink_foreground``), forwards unknown attributes
    to the wrapped model and accepts state dicts with or without the
    ``backbone.`` prefix.  Subclasses implement ``forward``.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.ink_foreground = False
        self._ink_fraction_per_patch: Optional[torch.Tensor] = None
        self._input_mean = None
        self._input_std = None
        self._ink_window = 51
        self._ink_k = 0.2
        self._ink_morph_open = 3
        self.embed_dim = getattr(backbone, "embed_dim", None)
        if self.embed_dim is not None:
            self.num_features = self.embed_dim

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            backbone = self._modules.get("backbone")
            if backbone is not None and hasattr(backbone, name):
                return getattr(backbone, name)
            raise

    def load_state_dict(self, state_dict, strict=True):
        if any(not k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {
                (f"backbone.{k}" if not k.startswith("backbone.") else k): v
                for k, v in state_dict.items()
            }
        result = super().load_state_dict(state_dict, strict=strict)
        return result


class DINOv3ViT(ViTWrapper):
    """Wrapper for DINOv3 (RoPE) backbones, the Stage-2 student.

    Delegates to the backbone's own ``forward_features(x)`` and returns a
    ``(cls, patches)`` pair plus the ink-count gate hook, the interface the
    distillation and evaluation code expects.
    """

    def forward(
        self,
        images: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> torch.Tensor:
        base = self.backbone
        feats = base.forward_features(images, masks=None)
        if not isinstance(feats, dict) or "x_norm_clstoken" not in feats:
            raise RuntimeError(
                "DINOv3 forward_features did not return the expected dict with "
                "'x_norm_clstoken'/'x_norm_patchtokens'; check the installed "
                "dinov3 hub version."
            )
        cls = feats["x_norm_clstoken"]
        patches = feats.get("x_norm_patchtokens")

        self._ink_fraction_per_patch = None
        if (
            self.ink_foreground
            and self._input_mean is not None
            and self._input_std is not None
            and patches is not None
        ):
            grid = int(round(patches.shape[1] ** 0.5))
            self._ink_fraction_per_patch = patch_ink_fraction(
                images, grid, grid, self._input_mean, self._input_std,
                window=self._ink_window, k=self._ink_k, morph_open=self._ink_morph_open,
            )

        head = getattr(base, "head", None)
        cls_out = head(cls) if head is not None else cls
        if return_patch_tokens:
            return cls_out, patches
        return cls_out


class PatchTokenViT(ViTWrapper):
    """Inference-only wrapper for an *external* (e.g. timm) plain-ViT backbone.

    This is the integration point for third-party SSL ViTs that run through
    the *same* leave-one-out / aggregation pipeline as the DINOv3 student -
    here the reference writer-retrieval ViT-S/16 (Raven et al., ICDAR 2024)
    and the Stage-1 teachers trained with its recipe - so evaluating such a
    backbone is just an eval command, not a separate codebase.

    It subclasses :class:`ViTWrapper` to inherit the ink-count gate hook
    (the ``ink_foreground`` Sauvola gate that ``configure_ink_foreground`` /
    ``embed_tiles`` rely on).  The forward path delegates to the backbone's
    own ``forward_features`` (the timm ViT contract) and slices CLS + patch
    tokens from the returned ``[B, T, D]`` sequence.  The wrapper evaluates a
    trained checkpoint; it is not used for training.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__(backbone)
        self._num_prefix = int(getattr(backbone, "num_prefix_tokens", 1) or 1)

    def forward(
        self,
        images: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> torch.Tensor:
        base = self.backbone
        feats = base.forward_features(images)
        if isinstance(feats, dict):
            for k in ("x_norm_patchtokens", "last_hidden_state", "x", "tokens"):
                if isinstance(feats.get(k), torch.Tensor):
                    feats = feats[k]
                    break
            else:
                raise RuntimeError(
                    "external ViT forward_features returned a dict without a "
                    "recognised token tensor; inspect the checkpoint's architecture."
                )
        if feats.ndim != 3:
            raise RuntimeError(
                f"external ViT forward_features must return [B, T, D]; got {tuple(feats.shape)}"
            )

        num_prefix = self._num_prefix
        cls = feats[:, 0]
        patches = feats[:, num_prefix:]

        self._ink_fraction_per_patch = None
        if (
            self.ink_foreground
            and self._input_mean is not None
            and self._input_std is not None
            and patches.shape[1] > 0
        ):
            grid = int(round(patches.shape[1] ** 0.5))
            self._ink_fraction_per_patch = patch_ink_fraction(
                images, grid, grid, self._input_mean, self._input_std,
                window=self._ink_window, k=self._ink_k, morph_open=self._ink_morph_open,
            )

        if return_patch_tokens:
            return cls, patches
        return cls

    def load_state_dict(self, state_dict, strict=False):
        """Robustly map an external SSL checkpoint onto the wrapped ViT.

        Third-party writer-retrieval checkpoints come in many shapes: a bare
        timm state-dict, a DINO-style ``{'teacher': ..., 'student': ...}`` dict
        (the online network, ``'student'``, is taken when both are present),
        a Lightning ``{'state_dict': ...}`` blob, ``nn.DataParallel`` ``module.``
        prefixes, ``backbone.``/``encoder.``/``0.`` nesting, etc.  We unwrap the
        most common containers, strip those prefixes, drop projection/loss heads
        that are not part of the ViT trunk, re-prefix with ``backbone.`` (our
        wrapper stores the ViT at ``self.backbone``), and load non-strictly with
        a shape check.  An explicit ``[extvit]`` report makes a bad mapping
        impossible to miss - validate it the first time you load real weights.
        """
        sd = state_dict
        if isinstance(sd, dict):
            for key in ("student", "teacher", "model_state_dict", "state_dict",
                        "model", "backbone", "net", "encoder"):
                inner = sd.get(key)
                if isinstance(inner, dict) and any(
                    isinstance(v, torch.Tensor) for v in inner.values()
                ):
                    sd = inner
                    break

        def _strip(k: str) -> str:
            changed = True
            while changed:
                changed = False
                for pre in ("module.", "backbone.", "encoder.", "model.", "0."):
                    if k.startswith(pre):
                        k = k[len(pre):]
                        changed = True
            return k

        target = dict(self.state_dict())
        cleaned = {}
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            nk = _strip(k)
            if any(nk.startswith(p) for p in (
                "head.", "dino_head.", "ibot_head.", "projection", "projector",
                "predictor", "prototypes",
            )):
                continue
            cleaned[f"backbone.{nk}"] = v

        matched = {
            k: v for k, v in cleaned.items()
            if k in target and tuple(target[k].shape) == tuple(v.shape)
        }
        result = super(ViTWrapper, self).load_state_dict(matched, strict=False)

        n_bb = sum(1 for k in target if k.startswith("backbone."))
        n_ck = len(cleaned)
        print(
            f"[extvit] loaded {len(matched)}/{n_bb} backbone tensors "
            f"(checkpoint offered {n_ck} trunk tensors)",
            flush=True,
        )
        for crit in ("backbone.pos_embed", "backbone.cls_token",
                     "backbone.patch_embed.proj.weight"):
            if crit in target and crit not in matched:
                warnings.warn(
                    f"[extvit] did NOT load '{crit}' - shape/name mismatch with the "
                    f"checkpoint (offered shape "
                    f"{tuple(cleaned[crit].shape) if crit in cleaned else 'absent'}, "
                    f"model expects {tuple(target[crit].shape)}). The eval will run "
                    f"on a partly-random backbone; fix the remap before trusting "
                    f"numbers.",
                    RuntimeWarning,
                )
        if n_bb and len(matched) < 0.7 * n_bb:
            warnings.warn(
                f"[extvit] only {len(matched)}/{n_bb} backbone tensors matched - "
                f"the checkpoint key layout likely differs from what this remap "
                f"expects. Print the checkpoint keys and adjust the prefix stripping.",
                RuntimeWarning,
            )
        return result


def _load_dinov3(model_key: str, pretrained: bool) -> nn.Module:
    """Load a DINOv3 backbone, tolerant of API/weight gating and hub extras.

    The dinov3 repo's ``hubconf.py`` imports its segmentation/eval extras at
    module load (``torchmetrics`` etc.), so ``torch.hub.load`` fails on a
    plain training env even though a backbone build needs none of that.  The
    primary path therefore imports ``dinov3.hub.backbones`` DIRECTLY from the
    cached repo (verified stdlib-only at import) and calls the entrypoint:

      1) locate the repo: $DINOV3_REPO_DIR or the torch.hub cache
         (``<hub_dir>/facebookresearch_dinov3_main``);
      2) if ``pretrained`` and the licence-gated weights exist at the local
         convention ``checkpoints/dinov3/<key>_pretrain_lvd1689m.pth``, pass
         them via ``weights=`` (strict load inside dinov3);
      3) otherwise build the bare architecture - weights are then overlaid
         from ``model.checkpoint`` by the caller (ViTWrapper.load_state_dict
         prefix-remaps a raw dinov3 state dict onto the wrapper);
      4) fall back to ``torch.hub.load`` (works when the optional hubconf
         deps are installed), else raise an actionable error.
    """
    import importlib
    import os
    import sys
    import urllib.error

    repo = 'facebookresearch/dinov3'
    hub_dir = torch.hub.get_dir()
    local_repo = os.environ.get(
        'DINOV3_REPO_DIR', os.path.join(hub_dir, 'facebookresearch_dinov3_main')
    )
    local_weights = os.path.join(
        'checkpoints', 'dinov3', f'{model_key}_pretrain_lvd1689m.pth'
    )

    direct_err: Optional[Exception] = None
    if os.path.isdir(local_repo):
        try:
            if local_repo not in sys.path:
                sys.path.insert(0, local_repo)
            backbones = importlib.import_module('dinov3.hub.backbones')
            entry = getattr(backbones, model_key, None)
            if entry is None:
                raise AttributeError(
                    f"dinov3.hub.backbones has no entrypoint '{model_key}'"
                )
            if pretrained and os.path.isfile(local_weights):
                print(
                    f"[model_factory] DINOv3 '{model_key}' from {local_repo} "
                    f"(hubconf bypassed); weights <- {local_weights}",
                    flush=True,
                )
                return entry(pretrained=True, weights=local_weights)
            model = entry(pretrained=False)
            if pretrained:
                warnings.warn(
                    f"DINOv3 '{model_key}' built WITHOUT weights (gated download; "
                    f"no local file at {local_weights}). Ensure model.checkpoint "
                    "points at the downloaded .pth and check the loaded-weights "
                    "log line - otherwise you are training from random init.",
                    RuntimeWarning,
                )
            return model
        except Exception as exc:
            direct_err = exc
            warnings.warn(
                f"Direct dinov3.hub.backbones import failed ({exc!r}); "
                "falling back to torch.hub.load.",
                RuntimeWarning,
            )

    try:
        return torch.hub.load(repo, model_key, pretrained=pretrained, verbose=False)
    except TypeError:
        return torch.hub.load(repo, model_key, verbose=False)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            ModuleNotFoundError) as exc:
        if os.path.isdir(local_repo) and not isinstance(exc, ModuleNotFoundError):
            warnings.warn(
                f"torch.hub could not fetch DINOv3 weights ({exc}); building the "
                f"architecture from {local_repo} WITHOUT weights - ensure "
                "model.checkpoint points at the locally downloaded .pth.",
                RuntimeWarning,
            )
            try:
                return torch.hub.load(local_repo, model_key, pretrained=False,
                                      verbose=False, source='local')
            except TypeError:
                return torch.hub.load(local_repo, model_key, verbose=False, source='local')
            except Exception as local_exc:
                exc = local_exc
        raise RuntimeError(
            "Could not load DINOv3. hubconf.py additionally imports the repo's "
            "segmentation/eval extras - a ModuleNotFoundError like 'torchmetrics' "
            "means those extras are missing (either rely on the direct-import path "
            "above, set DINOV3_REPO_DIR to the cloned repo, or `pip install "
            "torchmetrics` to appease hubconf). If the repo itself is absent, clone/"
            "cache 'facebookresearch/dinov3' and download the weights, then set "
            "model.checkpoint to the local .pth. "
            f"Direct-import error: {direct_err!r}; hub error: {exc!r}"
        ) from exc


def create_backbone(model_key: str, pretrained: bool,
                    device: torch.device) -> torch.nn.Module:
    """Create a vision backbone: DINOv3 (``dinov3_*``, the Stage-2 student) or
    an external timm ViT (``extvit:<timm name>``, the Stage-1 teachers and the
    reference checkpoint)."""
    key = model_key.lower()

    if 'dinov3' in key:
        backbone = _load_dinov3(model_key, pretrained)
        return DINOv3ViT(backbone).to(device)
    if key.startswith('extvit:'):
        timm_name = model_key.split(':', 1)[1]
        try:
            inner = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            warnings.warn(
                f"timm failed to build external ViT '{timm_name}' ({e}); "
                "retrying without pretrained weights"
            )
            inner = timm.create_model(timm_name, pretrained=False, num_classes=0)
        return PatchTokenViT(inner).to(device)
    raise ValueError(
        f"Unknown model '{model_key}' (use dinov3_* or extvit:<timm name>)")
