"""Device selection shared by the Stage-2 trainer and the evaluation.

:func:`resolve_device` honours ``training.device: cuda:N`` FOR REAL (validate
the index, ``torch.cuda.set_device``), understands the runner's ``GPU=<n>`` /
``CUDA_VISIBLE_DEVICES`` pin, and shouts when a restriction was probably
inherited from the calling shell (otherwise two concurrent runs can land on
one GPU).
"""

from __future__ import annotations

import os

import torch


def resolve_device(
    requested_device: str,
    *,
    tag: str = "",
) -> torch.device:
    """Resolve+pin the compute device from a config ``training.device`` string."""
    requested_device = str(requested_device)
    if torch.cuda.is_available():
        device_str = requested_device
    else:
        device_str = "cpu"
        if "cuda" in requested_device:
            reason = "CUDA was requested but is not available; running on CPU."
            print(f"{tag}[WARN] {reason}", flush=True)
    device = torch.device(device_str)

    if device.type == "cuda":
        _n_vis = torch.cuda.device_count()
        _idx = device.index if device.index is not None else 0
        if _idx >= _n_vis:
            _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            if _cvd is not None:
                if os.environ.get("GPU"):
                    print(
                        f"{tag}[INFO] device '{device_str}' + GPU="
                        f"{os.environ['GPU']} pin: running on physical GPU(s) "
                        f"{_cvd} (visible cuda:0).",
                        flush=True,
                    )
                else:
                    print(
                        f"{tag}[WARN] ======================================================\n"
                        f"{tag}[WARN] device '{device_str}' requests a GPU outside\n"
                        f"{tag}[WARN] CUDA_VISIBLE_DEVICES={_cvd} ({_n_vis} visible) - and no\n"
                        f"{tag}[WARN] GPU=<n> was passed, so this restriction was likely\n"
                        f"{tag}[WARN] INHERITED from the calling shell. This run\n"
                        f"{tag}[WARN] will execute on physical GPU(s) {_cvd}, which may be\n"
                        f"{tag}[WARN] occupied. If unintended: relaunch with GPU=<n>, or\n"
                        f"{tag}[WARN] `unset CUDA_VISIBLE_DEVICES` in the launching shell.\n"
                        f"{tag}[WARN] ======================================================",
                        flush=True,
                    )
                device = torch.device("cuda", 0)
            else:
                raise RuntimeError(
                    f"device='{device_str}' but only {_n_vis} CUDA device(s) "
                    f"are visible. Fix training.device in the config or launch "
                    f"with GPU=<n> (runners) / CUDA_VISIBLE_DEVICES=<n>."
                )
        torch.cuda.set_device(device)

    print(f"{tag}[INFO] Using device: {device}", flush=True)
    if device.type == "cuda":
        print(
            f"{tag}[INFO] CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset - ALL GPUs visible>')}",
            flush=True,
        )
        torch.cuda.empty_cache()
        try:
            cuda_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_idx)
            total_gb = props.total_memory / (1024 ** 3)
            print(
                f"{tag}[INFO] CUDA device {cuda_idx}: {props.name} "
                f"(compute {props.major}.{props.minor}, {total_gb:.1f} GB VRAM)",
                flush=True,
            )
            print(
                f"{tag}[INFO] torch {torch.__version__}, CUDA runtime "
                f"{torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}",
                flush=True,
            )
        except Exception:
            pass
    return device
