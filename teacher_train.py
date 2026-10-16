#!/usr/bin/env python
"""Teacher-from-scratch training, harness-integrated (TRAINER=teacher).

Runs the reference AttMask trainer (SSL_ViT_WR/attmask/main_attmask.py,
fetched by third_party/setup_reference.sh) as a supervised subprocess and
gives it the same interface as the Stage-2 trainer (distill_train.py):

  * honours the --config/--run_dir/--checkpoint_dir contract of
    run_scenario1.sh (TRAINER=teacher);
  * per-epoch progress CSV in run_dir (teacher_log.csv: epoch, train_loss,
    lr, val mAP per key) - same consume-and-plot layout as distill_log.csv;
  * REAL in-training validation every N epochs: converts the checkpoint
    (tools/convert_raven_ckpt.py) and runs the actual leave_one_out.py on
    the validation set (VLAD-100, binary input, pca 0 - RELATIVE numbers,
    same probe philosophy as distill's loo_vlad);
  * writes last_teacher.pth (bare-ViT eval format, directly usable as
    BACKBONE=), the reported checkpoint;
  * collapse abort: if the student val mAP is still below probe.abort_below
    after probe.abort_after epochs, the training subprocess is killed and
    the run exits non-zero - a dead run costs ~1 epoch, not 2 days;
  * loss-pin abort (probe.pin_abort, default ON): kills the run within
    minutes when the trainer's own `cls:` window-median goes FLAT (spread
    < 1e-3 over 200 consecutive log lines ≈ 2,000 iters) - the signature of
    entropy collapse (student softmax constant/uniform, pinning the value
    at exactly ln(256)=5.5452 for thousands of iterations). Unlike the
    val-LOO abort this needs no calibration AND catches mid-training pins
    the probe floor can never see (a pinned backbone stops learning but its
    probe plateaus at pre-pin quality, above any sane floor);
  * restart-safe: supervisor state (abort counters + log offset) persists
    in run_dir/supervisor_state.json; a relaunch with the trainer's
    checkpoint.pth still present RESUMES it (appending to teacher_log.csv)
    instead of re-consuming stale log.txt lines;
  * bounded, honest probes: each probe round converts BOTH keys from ONE
    checkpoint snapshot (the live checkpoint.pth can advance an epoch while
    the first ~45-min probe runs), probe subprocesses have a hard timeout
    (probe.probe_timeout_min, default 180), and probe.max_probe_failures
    (default 5) consecutive select-key failures abort the run rather than
    leaving it silently unsupervised with frozen abort counters.

The gradient loop is the reference authors' script, unmodified: the reproduction must
reproduce the published teacher recipe exactly - any port of the loop into
this repo would confound "recipe works" with "port is faithful". This
supervisor adds monitoring around the loop without touching a single
training instruction. Before launch it applies, idempotently, the import
shims of tools/raven_base_fixes.py and the validation-table sizing of
tools/patch_raven_dynval.py (neither changes training). Relative config
paths (clone, data_path) are taken from the release root. Variant runs
change only config knobs passed straight to the same script.

Usage (normally via run_scenario1.sh - see the config headers):
  GPU=0 DO_EVAL=0 RUN_TAG=teacher TRAINER=teacher \
    CONFIG=configs/stage1_teacher.yaml bash run_scenario1.sh
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "tools"))
from raven_base_fixes import apply_base_fixes
from raven_bf16 import apply_bf16_patch
from patch_raven_dynval import apply_dynval_patch


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _use_bf16(am: dict) -> bool:
    """bf16 is opt-in via the config (`attmask.bf16: true`) or a BF16 env
    override (so any config can be run in bf16 without editing it)."""
    return _truthy(am.get("bf16", "")) or _truthy(os.environ.get("BF16", ""))


def _load_cfg(path: str) -> dict:
    with open(os.path.expanduser(str(path)), "r") as f:
        return yaml.safe_load(f)


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(str(p)))


def _abs(p: str) -> Path:
    """A config path as an absolute path; relative paths are taken from the
    release root (the trainer runs in its clone, not here)."""
    q = _expand(p)
    return q if q.is_absolute() else (REPO / q).resolve()


def _check_data(am: dict) -> Path:
    """Stop before launch if the training directory holds no pages (the
    reference loader would otherwise read an empty folder silently)."""
    dp = _abs(am["data_path"])
    if not dp.is_dir() or not any(f.is_file() for f in dp.iterdir()):
        raise SystemExit(f"[teacher] no training pages in {dp}")
    return dp


def _build_attmask_cmd(am: dict, out_dir: Path) -> tuple[list[str], Path]:
    """The reference training command; every published knob
    is a config default here so a variant file only states what it changes."""
    cwd = _abs(am["clone"]) / "attmask"
    port = os.environ.get("MASTER_PORT") or am.get("master_port", 29500)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=1", f"--master_port={port}",
        "main_attmask.py",
        "--arch", str(am.get("arch", "vit_small")),
        "--patch_size", str(am.get("patch_size", 16)),
        "--epochs", str(am["epochs"]),
        "--warmup_epochs", str(am.get("warmup_epochs", 2)),
        "--freeze_last_layer", str(am.get("freeze_last_layer", 1)),
        "--lr", str(am["lr"]),
        "--overlap", str(am.get("overlap", 0.875)),
        "--batch_size_per_gpu", str(am.get("batch_size", 128)),
        "--global_crops_number", "2", "--global_crops_scale", "0.4", "1.0",
        "--local_crops_number", "8", "--local_crops_scale", "0.05", "0.4",
        "--augmentation", str(am.get("augmentation", "morph")),
        "--pred_shape", "attmask_high", "--pred_ratio", "0.3",
        "--masking_prob", "0.5", "--show_max", "0.1",
        "--data_path", str(_abs(am["data_path"])),
        "--output_dir", str(out_dir),
    ]
    cmd += [str(x) for x in (am.get("extra_args") or [])]
    if _use_bf16(am):
        cmd += ["--use_fp16", "false", "--use_bf16", "true"]
    return cmd, cwd


def _probe_key(ck: Path, key: str, epoch: int, pr: dict, run_dir: Path):
    """Convert one key out of the raw AttMask checkpoint and measure real
    val LOO mAP. Returns (mAP, converted_path) or (None, None) on failure.
    The caller deletes the converted file after the probe."""
    conv = run_dir / f"probe_ep{epoch:02d}_{key}.pth"
    try:
        r = subprocess.run(
            [sys.executable, "tools/convert_raven_ckpt.py", str(ck), str(conv),
             "--key", key, "--verify"],
            cwd=REPO, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"[teacher] convert({key}) TIMED OUT ep{epoch}")
        return None, None
    if r.returncode != 0:
        print(f"[teacher] convert({key}) FAILED ep{epoch}:\n{r.stderr[-800:]}")
        return None, None

    out_csv = run_dir / f"probe_loo_ep{epoch:02d}_{key}.csv"
    log_path = run_dir / f"probe_loo_ep{epoch:02d}_{key}.log"
    cmd = [sys.executable, "-u", "leave_one_out.py",
           "--config", str(_expand(pr.get("eval_config",
                                  "configs/eval_vit_s16.yaml"))),
           "--model_path", str(conv),
           "--method", "vlad", "--vlad_clusters", str(pr.get("vlad_clusters", 100)),
           "--input_dir", str(_expand(pr["val_dir"])),
           "--codebook_dir", str(_expand(pr["val_dir"])),
           "--pca_dim", "0",
           "--codebook_overlap", str(pr.get("codebook_overlap", 0)),
           "--binarisation", str(pr.get("binarisation", "input")),
           "--ink_threshold", str(pr.get("ink_threshold", 0.04)),
           "--eval_overlap", str(pr.get("overlap", 0.5)),
           "--use_patch_tokens",
           "--image_size", str(pr.get("image_size", 1024)),
           "--max_image_size", "0",
           "--embed_workers", str(pr.get("embed_workers", 4)),
           "--output_csv", str(out_csv)]
    env = os.environ.copy()
    if pr.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(pr["gpu"])
    timeout_s = float(pr.get("probe_timeout_min", 180)) * 60
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=lf,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            print(f"[teacher] probe({key}) TIMED OUT ep{epoch} after "
                  f"{timeout_s / 60:.0f} min - killed; see {log_path}")
            conv.unlink(missing_ok=True)
            return None, None
    if rc != 0 or not out_csv.exists():
        print(f"[teacher] probe({key}) FAILED ep{epoch} (rc={rc}), see {log_path}")
        conv.unlink(missing_ok=True)
        return None, None
    try:
        last = out_csv.read_text().strip().splitlines()[-1]
        m = float(last.replace("\r", "").rsplit(",", 1)[-1])
    except (ValueError, IndexError) as exc:
        print(f"[teacher] probe({key}) unparseable CSV ep{epoch}: {exc}")
        conv.unlink(missing_ok=True)
        return None, None
    return m, conv


def _ckpt_epoch(ck: Path) -> int | None:
    try:
        import torch
        return int(torch.load(ck, map_location="cpu").get("epoch"))
    except Exception:
        return None


class _LossPinDetector:
    """Detect entropy collapse from the reference trainer's own iteration log.

    A healthy run's printed `cls:` window-median jitters batch-to-batch by
    ~0.05; a collapsed student (constant softmax output) pins it to ~1e-6:
    cls flat at ln(256)=5.545177
    (out_dim 256 ⇒ uniform) with spread ~1e-5 for 4,000+ consecutive iters.
    A spread < PIN_EPS across WINDOW consecutive log lines (10 iters apart)
    cannot happen in a live run, so this abort stays on for every run,
    including the reproduction: past the pin the trajectory it would record
    is a constant. Disable with probe.pin_abort: false."""

    WINDOW = 200
    PIN_EPS = 1e-3

    _ITER = re.compile(r"\bcls: (\d+\.\d+) \(")
    _OUT_DIM = re.compile(r"^out_dim: (\d+)", re.M)

    def __init__(self, log_path: Path):
        self.path = log_path
        self.pos = log_path.stat().st_size if log_path.exists() else 0
        self.vals = collections.deque(maxlen=self.WINDOW)
        self.out_dim = None

    def poll(self) -> float | None:
        """Consume newly appended log lines; return the pinned value when the
        collapse signature is present, else None."""
        try:
            with open(self.path, errors="replace") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return None
        if self.out_dim is None:
            m = self._OUT_DIM.search(chunk)
            if m:
                self.out_dim = int(m.group(1))
        for m in self._ITER.finditer(chunk):
            self.vals.append(float(m.group(1)))
        if (len(self.vals) == self.WINDOW
                and max(self.vals) - min(self.vals) < self.PIN_EPS):
            return sum(self.vals) / len(self.vals)
        return None

    def describe(self, pinned: float) -> str:
        if self.out_dim and abs(pinned - math.log(self.out_dim)) < 0.01:
            return (f" = ln({self.out_dim}) - student softmax UNIFORM over "
                    f"out_dim; the head emits a constant, the backbone gets "
                    f"~zero gradient")
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--checkpoint_dir", default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    am, pr = cfg["attmask"], cfg.get("probe", {})
    _check_data(am)
    tr = cfg.get("training", {})
    run_dir = _expand(args.run_dir or tr.get("run_dir", "runs/teacher_attmask")).resolve()
    ckpt_dir = _expand(args.checkpoint_dir
                       or tr.get("checkpoint_dir", "checkpoints/teacher_attmask")).resolve()
    out_dir = ckpt_dir / "attmask"
    for d in (run_dir, ckpt_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config_snapshot.yaml")

    every = int(pr.get("every", 2))
    keys = list(pr.get("keys", ["student", "teacher"]))
    select_key = str(pr.get("select_key", "student"))
    abort_below = float(pr.get("abort_below", 0.15))
    abort_after = int(pr.get("abort_after", 3))
    abort_persist = int(pr.get("abort_persist", 2))
    pin_abort = _truthy(pr.get("pin_abort", True))

    state_path = run_dir / "supervisor_state.json"
    resuming = state_path.exists() and (out_dir / "checkpoint.pth").exists()
    low_streak, fail_streak, seen = 0, 0, 0
    if resuming:
        try:
            st = json.loads(state_path.read_text())
            low_streak = int(st.get("low_streak", 0))
            fail_streak = int(st.get("fail_streak", 0))
            seen = int(st.get("log_lines_seen", 0))
            print(f"[teacher] RESUME: restored supervisor state "
                  f"({seen} log lines consumed)")
        except (ValueError, OSError) as exc:
            print(f"[teacher] WARNING: unreadable {state_path} ({exc}) - "
                  f"starting with fresh state")
            resuming = False
    else:
        state_path.unlink(missing_ok=True)

    csv_path = run_dir / "teacher_log.csv"
    if not (resuming and csv_path.exists()):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "lr"]
                                   + [f"val_map_{k}" for k in keys])

    def _save_state():
        try:
            state_path.write_text(json.dumps({
                "low_streak": low_streak,
                "fail_streak": fail_streak, "log_lines_seen": seen}))
        except OSError as exc:
            print(f"[teacher] WARNING: could not save state: {exc}")

    clone = _abs(am["clone"])
    created = apply_base_fixes(clone)
    if created:
        print(f"[teacher] applied {created} import-shim file(s) to {clone}")
    if apply_dynval_patch(clone):
        print(f"[teacher] patched {clone}: in-training validation table sized "
              f"from the training pages (floor 1,182)")
    if _use_bf16(am):
        nbf = apply_bf16_patch(clone)
        print(f"[teacher] bf16 mode ON ({'patched' if nbf else 'already patched'} "
              f"{clone}) - autocast=bfloat16, no GradScaler")

    cmd, cwd = _build_attmask_cmd(am, out_dir)
    total = int(am["epochs"])
    print(f"[teacher] config: {args.config}")
    print(f"[teacher] launching the reference trainer in {cwd}")
    print(f"[teacher]   {' '.join(cmd)}")
    print(f"[teacher] probe: every {every} ep, keys={keys}, select={select_key}; "
          f"abort if {select_key} mAP < {abort_below} after ep {abort_after}")

    child_log = open(out_dir / "attmask_train.log", "a")
    child = subprocess.Popen(cmd, cwd=cwd, stdout=child_log,
                             stderr=subprocess.STDOUT, start_new_session=True)

    def _kill_child(*_):
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGTERM, lambda *a: (_kill_child(), sys.exit(143)))
    signal.signal(signal.SIGINT, lambda *a: (_kill_child(), sys.exit(130)))

    log_txt = out_dir / "log.txt"
    pin = _LossPinDetector(out_dir / "attmask_train.log") if pin_abort else None
    last_beat = time.time()
    max_probe_failures = int(pr.get("max_probe_failures", 5))
    try:
        while True:
            done = child.poll() is not None
            lines = log_txt.read_text().splitlines() if log_txt.exists() else []
            for idx in range(seen, len(lines)):
                try:
                    js = json.loads(lines[idx])
                except json.JSONDecodeError:
                    continue
                ep = int(js.get("epoch", -1))
                loss, lr = js.get("train_loss", ""), js.get("train_lr", "")
                maps = {k: "" for k in keys}
                is_last = ep >= total - 1
                is_newest = idx == len(lines) - 1
                if is_newest and ((ep % every == every - 1) or is_last or done):
                    ck = out_dir / "checkpoint.pth"
                    snap = run_dir / "probe_snapshot.pth"
                    try:
                        shutil.copy2(ck, snap)
                    except OSError as exc:
                        print(f"[teacher] probe snapshot failed ep{ep}: {exc}")
                        snap = None
                        fail_streak += 1
                    if snap is not None:
                        sn_ep = _ckpt_epoch(snap)
                        if sn_ep is not None and sn_ep not in (ep, ep + 1):
                            print(f"[teacher] WARNING: snapshot epoch field "
                                  f"{sn_ep} vs probed log epoch {ep} - probe "
                                  f"labels may be shifted")
                        for k in keys:
                            m, conv = _probe_key(snap, k, ep, pr, run_dir)
                            if m is None:
                                if k == select_key:
                                    fail_streak += 1
                                    print(f"[teacher] WARNING: {select_key} "
                                          f"probe failed ({fail_streak} in a "
                                          f"row; abort at "
                                          f"{max_probe_failures})", flush=True)
                                continue
                            if k == select_key:
                                fail_streak = 0
                            maps[k] = m
                            if conv is not None:
                                conv.unlink(missing_ok=True)
                        snap.unlink(missing_ok=True)
                    if fail_streak >= max_probe_failures:
                        print(f"[teacher] *** {fail_streak} consecutive "
                              f"{select_key} probe failures - supervision is "
                              f"blind (abort rules frozen); "
                              f"killing the run. Fix the probe (see the "
                              f"probe_loo_*.log tails) and relaunch: the "
                              f"supervisor resumes from supervisor_state.json "
                              f"+ the trainer's checkpoint.pth. ***", flush=True)
                        _kill_child()
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            pass
                        sys.exit(5)
                    _save_state()
                    sel = maps.get(select_key)
                    if sel != "" and sel is not None:
                        if abort_below > 0 and sel < abort_below and ep >= abort_after:
                            low_streak += 1
                        else:
                            low_streak = 0
                        if abort_below > 0 and low_streak >= abort_persist:
                            print(f"[teacher] *** COLLAPSE: {select_key} val mAP "
                                  f"below {abort_below} for {low_streak} consecutive "
                                  f"probes (latest {sel:.4f} at ep {ep}) - killing "
                                  f"the run ***")
                            _kill_child()
                            try:
                                child.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                pass
                            sys.exit(3)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([ep, loss, lr]
                                           + [maps[k] for k in keys])
                probe_txt = " ".join(f"{k}={maps[k]:.4f}" for k in keys
                                     if maps[k] != "")
                print(f"[teacher] epoch {ep + 1}/{total} loss={loss} "
                      f"{('| val ' + probe_txt) if probe_txt else ''}",
                      flush=True)
                last_beat = time.time()
            if len(lines) != seen:
                seen = len(lines)
                _save_state()

            if pin is not None and not done:
                pinned = pin.poll()
                if pinned is not None:
                    print(f"[teacher] *** LOSS PINNED: cls window-median flat "
                          f"at {pinned:.6f}{pin.describe(pinned)} for "
                          f"{pin.WINDOW} consecutive log lines "
                          f"(~{pin.WINDOW * 10} iters) - entropy collapse; "
                          f"killing the run ***", flush=True)
                    _kill_child()
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
                    sys.exit(4)

            if done:
                break
            if time.time() - last_beat > 600:
                tail = ""
                try:
                    with open(out_dir / "attmask_train.log") as f:
                        tail = (f.readlines() or [""])[-1].strip()
                except OSError:
                    pass
                print(f"[teacher] training... {tail[-160:]}", flush=True)
                last_beat = time.time()
            time.sleep(30)
    finally:
        child_log.close()

    try:
        child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    if child.returncode not in (0, None):
        print(f"[teacher] the reference trainer exited rc={child.returncode} - see "
              f"{out_dir / 'attmask_train.log'}")
        sys.exit(child.returncode)

    import zipfile
    _cands = [out_dir / "checkpoint.pth"] + sorted(
        out_dir.glob("checkpoint0*.pth"), reverse=True)
    ck = next((c for c in _cands if c.exists() and zipfile.is_zipfile(str(c))),
              None)
    if ck is None:
        sys.exit("[teacher] no intact checkpoint to convert (all torn/absent)")
    if ck.name != "checkpoint.pth":
        print(f"[teacher] WARNING: checkpoint.pth is torn/absent - converting "
              f"the newest intact snapshot {ck.name} instead (an EARLIER "
              f"epoch; the true last epoch is lost).", flush=True)
    r = subprocess.run(
        [sys.executable, "tools/convert_raven_ckpt.py", str(ck),
         str(ckpt_dir / "last_teacher.pth"), "--key", select_key, "--verify"],
        cwd=REPO)
    if r.returncode != 0:
        sys.exit("[teacher] final conversion failed")
    print(f"[teacher] DONE. last (ep {_ckpt_epoch(ck)}) -> "
          f"{ckpt_dir}/last_teacher.pth")
    print(f"[teacher] curve: {csv_path}")


if __name__ == "__main__":
    main()
