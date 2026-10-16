# Reference training code

Stage 1 supervises the training loop of the reference system, which is
not redistributed here. Source: https://github.com/Traven16/SSL_ViT_WR,
commit 176313767bbf650876cf0cd5080c0310755e9c81. Set it up once:

    bash third_party/setup_reference.sh

This clones the reference repository at that commit into
third_party/reference (used by configs/stage1_teacher_reproduction.yaml)
and creates a patched copy at third_party/reference_patched
(tools/patch_raven_supcon.py adds the page-level supervised-contrastive
loss and the page-grouped batch sampler; used by
configs/stage1_teacher.yaml, configs/stage1_teacher_control.yaml and
configs/stage1_bullinger_teacher.yaml). The script stops if an existing
clone is at a different commit; REPO_URL overrides the clone address.

Before each launch, teacher_train.py applies to the clone it trains with
the import shims of tools/raven_base_fixes.py (files the repository
imports but does not ship) and tools/patch_raven_dynval.py, which sizes
the trainer's in-training validation table from the training pages
(unchanged on Historical-WI, needed for the larger BullingerDB training
set). For configs/stage1_bullinger_teacher.yaml it also applies the
optional bfloat16 mode of tools/raven_bf16.py. None of these edits
changes a training instruction of the reference loop, except that the
bfloat16 mode changes the mixed-precision type of the BullingerDB
teacher.

The reported teachers were trained with PyTorch 2.6.0 (CUDA 12.4). The
reference repository lists no dependency versions; requirements.txt
also installs kornia and petname, which its trainer imports.
