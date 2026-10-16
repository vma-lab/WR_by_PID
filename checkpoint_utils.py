"""Shared checkpoint loading utilities.

Provides ``load_backbone``, the checkpoint loader of ``leave_one_out.py``;
the trainers' validation probes run leave_one_out.py, so every evaluation
uses the same loading logic.
"""

import logging

import torch

from model_factory import create_backbone

logger = logging.getLogger(__name__)


def load_backbone(cfg, device, ckpt_path):
    """Create the backbone and load its weights from *ckpt_path*.

    Parameters
    ----------
    cfg : OmegaConf
        Full experiment config (must contain ``model.name`` and
        ``model.pretrained``).
    device : torch.device
        Target device for model parameters.
    ckpt_path : str | pathlib.Path
        Path to a ``.pth`` checkpoint file.

    Returns
    -------
    backbone : nn.Module
        Backbone in eval mode, on *device*.
    """
    if not ckpt_path:
        raise ValueError("ckpt_path must point to a trained checkpoint")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = create_backbone(cfg.model.name, cfg.model.pretrained, device)

    def _load_reported(module, sd, what):
        """strict=False load with a LOUD coverage report.

        A key-mismatched checkpoint (wrong model.name in the eval config, un-
        prefixed keys, format drift) would otherwise silently evaluate near-
        random weights as a plausible-but-garbage mAP - the hazard that
        tools/convert_raven_ckpt.py --verify and distill_train.py also guard
        against.  Coverage is measured AFTER the load from the returned
        ``missing_keys`` so that custom ``load_state_dict`` overrides (extvit's
        prefix remapping, ViTWrapper's auto-prefixing) are honoured rather than
        second-guessed.  Zero matched tensors is always an error; <90% warns.
        """
        n_total = len(module.state_dict())
        res = module.load_state_dict(sd, strict=False)
        n_missing = len(getattr(res, "missing_keys", []) or [])
        matched = max(n_total - n_missing, 0)
        logger.info("[ckpt] %s: matched %d/%d model tensors from %s",
                    what, matched, n_total, ckpt_path)
        if n_total and matched == 0:
            sample = sorted(sd.keys())[:5] if isinstance(sd, dict) else []
            raise RuntimeError(
                f"Checkpoint {ckpt_path} matched 0/{n_total} {what} tensors - "
                f"wrong architecture/format for model '{cfg.model.name}' "
                f"(strict=False would silently evaluate init weights). "
                f"Sample ckpt keys: {sample}"
            )
        if n_total and matched / n_total < 0.9:
            logger.warning(
                "[ckpt] %s: only %.0f%% of model tensors matched (%d/%d) - "
                "verify this is intentional (partial load).",
                what, 100.0 * matched / n_total, matched, n_total,
            )

    if isinstance(state, dict) and "backbone" in state:
        _load_reported(backbone, state["backbone"], "backbone")
    elif isinstance(state, dict) and "model_state_dict" in state:
        _load_reported(backbone, state["model_state_dict"], "backbone")
    else:
        _load_reported(backbone, state, "backbone")

    return backbone.to(device).eval()
