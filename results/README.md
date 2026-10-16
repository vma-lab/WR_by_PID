# Measured results

This folder contains the per-query CSVs behind the values and paired
tests of the paper (except the BullingerDB corpus-protocol values, see
below, the two-seed range of Sec. 3 and the window-25 check of
Appendix A), including those of the reference checkpoint, so that
the reported comparisons can be verified without retraining anything:

    python tools/paired_bootstrap.py results/A.csv results/B.csv

(run from the repository root). The files are the unmodified evaluation
outputs: one row per query with its writer, average precision, first-hit
rank and top-five neighbours; queries without another page by their
writer have an empty `ap` and are not scored. Naming: `base`/`full` are
the paper's two configurations; `gate`/`mask` are the two GRK-Papyri
cells; the `ladder_r*` files are the evaluation-ladder rows of Table 1
(exact PCA), plus the randomised-PCA variants of r0 and r2 used in
Sec. 3 and Appendix A; `window_rule` rows enable the keep-rule of
Appendix A. The files are data produced by this code; their file names
and writer labels come from the respective datasets (see
data/README.md; licence notes in the License section of README.md).

## File-to-command map

Each file is the output

    results/scenario1_<RUN_TAG>/LOO_vlad_patch_<VARIANT>_per_query.csv

of the reproduce script, RUN_TAG and VARIANT given below, copied here
under its descriptive name.

| file | paper location | script (reproduce/) | RUN_TAG | VARIANT |
|---|---|---|---|---|
| hisir19_ladder_r0_exact_pca | Table 1, published evaluation (printed) | table1_baseline.sh | table1_r0 | std |
| hisir19_ladder_r0_randomised_pca | Appendix A, PCA solver check on the ladder's first row | table1_baseline.sh | table1_r0 | rpca |
| hisir19_ladder_r1_exact_pca | Table 1, all windows kept (printed) | table1_baseline.sh | table1_r1 | std |
| hisir19_ladder_r2_exact_pca | Table 1, + our VLAD normalisation (printed) | table1_baseline.sh | table1_r2 | std |
| hisir19_ladder_r2_randomised_pca | Appendix A solver check; pairing partner of the codebook-fit step (Sec. 3) | table1_baseline.sh | table1_r2 | rpca |
| hisir19_reference_base | Tables 1 to 4, reference checkpoint (base configuration) | table1_baseline.sh | reference | std |
| hisir19_reference_full | Table 4 (Sec. 5.4), reference checkpoint, full configuration | table4_full_pipeline.sh | reference | esvm_nopca |
| hisir19_reference_window_rule_base | Appendix A, window keep-rule, base configuration | appendix_window_rule.sh | reference_rule | std |
| hisir19_reference_window_rule_full | Appendix A, window keep-rule, full configuration | appendix_window_rule.sh | reference_rule | esvm_nopca |
| hisir19_reproduction_base | Table 1 middle block and Table 2, reproduction | table1_baseline.sh | reproduction | std |
| hisir19_teacher_base | Tables 2 and 3, improved teacher | table2_teacher.sh | teacher_eval | std |
| hisir19_teacher_full | Sec. 5.4, improved teacher, full configuration | table4_full_pipeline.sh | teacher_eval | esvm_nopca |
| hisir19_teacher_control_base | Sec. 5.2 (control defined in Sec. 4.2), sampler-only control | table2_teacher.sh | teacher_control_eval | std |
| hisir19_undistilled_control_base | Table 3, DINOv3-B/16 without distillation | table3_distillation.sh | dinov3_control | std |
| hisir19_student_baseline_reproduction_base | Table 3, student from the reproduced teacher | table3_distillation.sh | baseline_reproduction_eval | std |
| hisir19_student_baseline_base | Table 3, student from the improved teacher | table3_distillation.sh | baseline_eval | std |
| hisir19_student_base | Tables 3 and 4, final student | table3_distillation.sh | student_eval | std |
| hisir19_student_full | Table 4, final student, full configuration | table4_full_pipeline.sh | student_eval | esvm_nopca |
| hisir19_student_full_width_base | Table 3, full-width ablation | table3_distillation.sh | student_eval_fullwidth | std |
| hisir19_student_full_width_full | Sec. 5.3, full-width ablation, full configuration | table4_full_pipeline.sh | student_eval_fullwidth | esvm_nopca |
| histwi_reference_base | Tables 1 to 4, reference checkpoint | table1_baseline.sh | reference_histwi | std |
| histwi_reference_full | Table 4 (Sec. 5.4), reference checkpoint, full configuration | table4_full_pipeline.sh | reference_histwi | esvm_nopca |
| histwi_reproduction_base | Table 1 middle block and Table 2, reproduction | table1_baseline.sh | reproduction_histwi | std |
| histwi_teacher_base | Tables 2 and 3, improved teacher | table2_teacher.sh | teacher_eval_histwi | std |
| histwi_student_base | Tables 3 and 4, final student | table3_distillation.sh | student_eval_histwi | std |
| histwi_student_full | Table 4, final student, full configuration | table4_full_pipeline.sh | student_eval_histwi | esvm_nopca |
| grk_reference_gate | Table 5 (Sec. 5.5), standard foreground gate | table5_grk.sh | grk_reference | std |
| grk_student_gate | Table 5 (Sec. 5.5), standard foreground gate | table5_grk.sh | grk_student | std |
| grk_reference_mask | Table 5 (Sec. 5.5), external mask as selector | table5_grk.sh | grk_reference_mask | std |
| grk_student_mask | Table 5 (Sec. 5.5), external mask as selector | table5_grk.sh | grk_student_mask | std |
| cvl_reference_base | Table 6 (Sec. 5.5), base configuration | table6_cvl.sh | cvl_reference | std |
| cvl_reference_full | Table 6 (Sec. 5.5), full configuration | table6_cvl.sh | cvl_reference | esvm_nopca |
| cvl_student_base | Table 6 (Sec. 5.5), base configuration | table6_cvl.sh | cvl_student | std |
| cvl_student_full | Table 6 (Sec. 5.5), full configuration | table6_cvl.sh | cvl_student | esvm_nopca |
| bullinger_trained_base | Sec. 5.4, BullingerDB trained on the corpus, our ranking protocol | table4_bullinger.sh | bullinger_trained | std |
| bullinger_trained_full | Sec. 5.4, the same with exemplar-SVM descriptors | table4_bullinger.sh | bullinger_trained | esvm_nopca |
| bullinger_zeroshot_base | Sec. 5.5, BullingerDB zero-shot, our ranking protocol | table7_bullinger_zeroshot.sh | bullinger_zeroshot | std |
| bullinger_zeroshot_full | Sec. 5.5, the same with exemplar-SVM descriptors | table7_bullinger_zeroshot.sh | bullinger_zeroshot | esvm_nopca |

The Historical-WI files of the binarised-input models name their queries
`<stem>.png` (the inverted binarised pages), those of the student
`<stem>.jpg` (the colour pages); tools/paired_bootstrap.py pairs queries
by file stem, so the two kinds pair directly.

The BullingerDB values of Tables 4 and 7 (mAP, temporal nDCG and Top-1
under plain cosine ranking) are
computed by tools/bullinger_ndcg.py from the descriptors of the same
runs; the reproduce scripts call it right after each variant. It writes
results/scenario1_<RUN_TAG>/bullinger_<VARIANT>_{ours,cosine}_per_query.csv.
These files are not included in this release; tools/bullinger_ndcg.py
regenerates them after the listed evaluations are re-run. The values then
agree with Tables 4 and 7 up to the re-embedding differences
described under Determinism in README.md (up to 0.2 points in our
re-run of the trained-on-target row).
