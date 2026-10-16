# Raw-input writer retrieval for historical documents by privileged-information distillation

Code and configurations for the paper "Raw-input writer retrieval for
historical documents by privileged-information distillation" (under
review).

The method has two training stages and one shared evaluation pipeline.
Stage 1 trains a ViT-S/16 teacher on binarised pages with the reference
system's self-supervised recipe plus a page-level supervised-contrastive
loss. Stage 2 distils the frozen teacher into a raw-input ViT-B/16
student with a token regression and a relational term. Evaluation uses
VLAD aggregation with either PCA whitening (base configuration) or
exemplar-SVM re-encoding (full configuration), followed by
kRNN (k-reciprocal nearest-neighbour) query expansion and cosine
ranking.

## Setup

    pip install -r requirements.txt

Tested with Python 3.11 and PyTorch 2.6.0 built for CUDA 12.4 (see
Determinism below). To check an installation (CPU only; no data or
weights needed), run

    bash tools/smoke_test.sh

1. Datasets: see data/README.md.
2. Reference training code (Stage 1 only): see third_party/README.md.
3. DINOv3 (Stage 2, and every evaluation of a student). The student
   backbone is built from the DINOv3 source code. Clone
   https://github.com/facebookresearch/dinov3 at the commit of the
   reported runs and point DINOV3_REPO_DIR at the clone:

       git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
       git -C third_party/dinov3 checkout 346f38fee679c56a6888f91c51670fae61d364e0
       export DINOV3_REPO_DIR=$PWD/third_party/dinov3

   model_factory.py imports the backbone from dinov3.hub.backbones in
   that directory, so the optional dependencies of the repository's
   torch.hub entry point are not needed. Without DINOV3_REPO_DIR it looks
   for the clone at <torch hub directory>/facebookresearch_dinov3_main,
   and as a last resort calls torch.hub.load('facebookresearch/dinov3'),
   which fetches the current default branch and needs those extra
   dependencies. The upstream README states PyTorch >= 2.7.1 for its own
   training and evaluation code; the backbone build used here ran on
   PyTorch 2.6.0.

   The ViT-B/16 weights (LVD-1689M) are distributed by their authors
   after an access request at
   https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/.
   Save the downloaded file dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
   as checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m.pth, the name
   the configurations and model_factory.py expect. The weights are needed
   only for Stage 2 training and for the undistilled control row;
   evaluating the released students needs only the source code.
4. The reference checkpoint evaluated in the paper is not part of this
   release; it was provided by the authors of the reference system on
   request. Convert it with

       python tools/convert_raven_ckpt.py <provided checkpoint> \
           checkpoints/reference/reference_vit_s16.pth --key student --verify

   --key student selects the online network, which every reference row
   evaluates (it is also the default); --key teacher would select the
   momentum encoder. Our measured per-query results for that checkpoint
   are included in results/ (see results/README.md).

## Training

Stage 1, teacher:

    GPU=0 RUN_TAG=teacher TRAINER=teacher \
      CONFIG=configs/stage1_teacher.yaml DO_EVAL=0 bash run_scenario1.sh

Stage 2, final student:

    GPU=0 RUN_TAG=student TRAINER=distill \
      CONFIG=configs/stage2_student.yaml DO_EVAL=0 bash run_scenario1.sh

Trained weights land in checkpoints/scenario1/ssl_<RUN_TAG>/. The
reported checkpoints are the final ones, last_teacher.pth and
last_distill.pth; the validation probes that both trainers run are
monitoring only and select no reported checkpoint.

Stage 1 runs the reference trainer as a subprocess of teacher_train.py,
which monitors it and converts its final checkpoint. The monitor stops a
run in three cases, none of which selects a checkpoint: when the online
network's probe mAP on the HisIR19 validation set (a reduced-geometry
probe) stays below probe.abort_below (0.45 in the shipped
configurations) for probe.abort_persist (2) consecutive probes from
epoch probe.abort_after (1) on (exit code 3); when the class loss in the
trainer's log stays flat, with a spread below 1e-3 over 200 consecutive
log lines (probe.pin_abort, on by default; exit code 4); or after
probe.max_probe_failures (5) consecutive failed probes (exit code 5).
probe.abort_below: 0 and probe.pin_abort: false disable the first two
rules. last_teacher.pth is converted from the trainer's final
checkpoint.pth; if that file is unreadable (an interrupted write), the
newest intact earlier snapshot (checkpoint0*.pth) is converted instead,
a WARNING naming that snapshot is printed, and the run still exits with
code 0, so check the training log for that warning.

The checkpoints used in the paper are available at <https://doi.org/10.25592/uhhfdm.21961>; to
evaluate without retraining, download each file and place it as
follows:

| download | place at |
|---|---|
| teacher.pth | checkpoints/scenario1/ssl_teacher/last_teacher.pth |
| teacher_reproduction.pth | checkpoints/scenario1/ssl_teacher_reproduction/last_teacher.pth |
| teacher_control.pth | checkpoints/scenario1/ssl_teacher_control/last_teacher.pth |
| student.pth | checkpoints/scenario1/ssl_student/last_distill.pth |
| student_baseline.pth | checkpoints/scenario1/ssl_student_baseline/last_distill.pth |
| student_baseline_reproduction.pth | checkpoints/scenario1/ssl_student_baseline_reproduction/last_distill.pth |
| bullinger_teacher.pth | checkpoints/scenario1/ssl_bullinger_teacher/last_teacher.pth |
| bullinger_student.pth | checkpoints/scenario1/ssl_bullinger_student/last_distill.pth |

Each student file also contains the map W used at inference.
The DINOv3 initialisation and the reference checkpoint are not part
of this set (steps 3 and 4 above).

## Evaluation

One command per table row; the scripts in reproduce/ list all of them.
Example, the final student on HisIR19 in the base configuration:

    GPU=0 DO_TRAIN=0 RUN_TAG=student_eval VARIANT=std TRAINER=distill \
      CONFIG=configs/stage2_student.yaml EVAL_CONFIG=configs/eval_student.yaml \
      BACKBONE=checkpoints/scenario1/ssl_student/last_distill.pth \
      DISTILL_PROJ=auto IMAGE_SIZE=-1 SIMPLE_WINDOW=1 CODEBOOK_OVERLAP=0 \
      bash run_scenario1.sh

Results are written to results/scenario1_<RUN_TAG>/; the per-query file
LOO_vlad_patch_<VARIANT>_per_query.csv is the input of the paired
bootstrap:

    python tools/paired_bootstrap.py A_per_query.csv B_per_query.csv

Queries are paired by file stem. Add --cluster writer to resample
writers instead of queries. results/README.md maps every included
per-query file to the command that produces it. Each reproduce script
runs all its steps, prints a summary of the failed steps at the end and
exits non-zero if any step failed.

## Reproducing the paper

| script | contents |
|---|---|
| reproduce/table1_baseline.sh | Table 1 (Sec. 3): evaluation ladder and baseline rows |
| reproduce/table2_teacher.sh | Table 2 (Sec. 5.2): Stage 1 training and teacher rows |
| reproduce/table3_distillation.sh | Table 3 (Sec. 5.3): Stage 2 training, student rows, controls |
| reproduce/table4_full_pipeline.sh | Sec. 5.4 and Table 4: full configuration (exemplar-SVM) rows |
| reproduce/table4_bullinger.sh | Table 4, BullingerDB block (Sec. 5.4): trained on the corpus |
| reproduce/table5_grk.sh | Table 5 (Sec. 5.5): GRK-Papyri |
| reproduce/table6_cvl.sh | Table 6 (Sec. 5.5): CVL, fully frozen |
| reproduce/table7_bullinger_zeroshot.sh | Table 7 (Sec. 5.5): BullingerDB, zero-shot |
| reproduce/appendix_window_rule.sh | Appendix A: window-rule variants |

## Determinism

The paired tests on the included per-query files are exact: the bootstrap
uses a fixed seed and reproduces its output exactly. Training and
descriptor extraction are not bit-reproducible: they run on the GPU in
mixed precision (float16 for Stage 1 and for descriptor extraction,
bfloat16 for Stage 2 and for the BullingerDB teacher) with
non-deterministic cuDNN kernels (cudnn.benchmark is enabled), and both
trainers read their data through parallel loader workers. The codebook
fit of the base configuration uses scikit-learn's
MiniBatchKMeans without an explicit n_init, whose default changed between
scikit-learn versions (1.2 and 1.3 run several initialisations, 1.4 and
later one), and the exemplar SVMs (LinearSVC) are fitted in parallel
threads without a fixed random_state. Re-evaluating the same checkpoint
therefore reproduces a row up to small differences rather than
identically: in our re-runs, about 0.05 mAP points on HisIR19 after a re-embedding. The reported rows
were produced with Python 3.11 and PyTorch 2.6.0 (CUDA 12.4);
requirements.txt gives version bounds only.

## Citation

If you use this code or the released weights, please cite the paper
(under review; this entry will be updated once it is published):

```bibtex
@unpublished{mohammed2026rawinput,
  author = {Hussein Mohammed},
  title  = {Raw-input writer retrieval for historical documents by
            privileged-information distillation},
  note   = {Under review},
  year   = {2026}
}
```

## License

Copyright (C) 2026 Hussein Mohammed.

The code is licensed under the GNU Affero General Public License,
version 3 only; see LICENSE.

The student checkpoints (student.pth, student_baseline.pth,
student_baseline_reproduction.pth and bullinger_student.pth) are
derived from the DINOv3 ViT-B/16 checkpoint and are distributed under
the DINOv3 License, a copy of which is provided as LICENSE-DINOv3.md.
The teacher checkpoints (teacher.pth, teacher_reproduction.pth,
teacher_control.pth and bullinger_teacher.pth) are trained from scratch
and are released under the same terms as the code. The DINOv3 source
code and weights fetched in step 3 of Setup are not part of this
release and are subject to the DINOv3 License.

The reference training code fetched under third_party/ is not
redistributed and is subject to its authors' terms (see
third_party/README.md).

The files in results/ are data produced by this code, not code; their
file names and writer labels come from the respective datasets, which
remain subject to their own terms (sources and citations in
data/README.md). The four results/bullinger_*.csv files list file names
and writer labels of the BullingerDB writer-retrieval test split (Peer
et al., https://zenodo.org/records/19728926), which is licensed under
CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); these
four files are distributed under the same licence.
