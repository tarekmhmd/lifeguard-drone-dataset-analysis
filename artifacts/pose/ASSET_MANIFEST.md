# Pose estimation — asset manifest and upload guide

Prepared 2026-11-27. Nothing is staged or committed. The per-file table at the end (path, bytes, sha256, class,
source, Git LFS) and [`asset_inventory.csv`](asset_inventory.csv) are generated from the files actually present by
`python python_scripts/pose/make_asset_inventory.py`; re-run it after any change.

Classes: **RUNTIME** (loaded by the system) · **EXPERIMENTAL** (research only; no configuration loads it) ·
**REFERENCE** (AI-generated annotations — **not human ground truth**) · **METADATA** · **RESULT** · **PROVENANCE**.

## 1. Runtime weight (integrated)

| File | Size | sha256 | Source | Role |
| --- | --- | --- | --- | --- |
| `models/yolo11s-pose.pt` | 20,373,161 B | `1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba` | Official Ultralytics release `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt`, downloaded 2026-11-27; registry `pose_yolo11s_coco_v1` | **RUNTIME** — the only pose model any configuration loads (`configs/ai_z5cc_pose.yaml`). Kept at its existing path; not duplicated. The Drive copy (`pose_baseline/models/yolo11s-pose.pt`, 19.4 MB) was not packaged. |

## 2. Experimental checkpoints (from the Drive package) — EXPERIMENTAL, not adopted

Retrieved from `E:\lifeguardDron\pose_package_20261127.zip` (97,951,521 B, sha256
`44f5d39c5f04c22ba931b06d6e46a13af5e02f18160ee059d4444f1a29815681`, matches the value printed in Colab; every member
verified against the zip's own sha256 list). All four load as 17-keypoint YOLO pose models.

| Path | Bytes | sha256 | Drive source | Run settings (from `args.yaml` / `results.csv`) |
| --- | --- | --- | --- | --- |
| `weights/experimental/yolo11s_pose_distill_960/best.pt` | 20,198,472 | `0cc0896e8145d421d91a09fdef2e3291ed2f78a76ca59e28651f2e819c707a92` | `runs_pose/yolo11s_pose_distill_960/weights/best.pt` | Distillation from ViTPose pseudo-labels; 300 train / 35 val frames (27 test frames held out); imgsz 960, batch 16, 67 of 100 epochs (early stop, patience 30) |
| `weights/experimental/yolo11s_pose_e1_ignore_960/best.pt` | 20,193,032 | `23230614358e4e893ed54eafb21ac40ab63cf89399696e711cf9f28ae65efaa6` | `runs_pose/yolo11s_pose_e1_ignore_960/weights/best.pt` | E1 v1, ignore-aware loss; 2,861 train / 762 val frames; imgsz 960, batch 16, 40 epochs (failed: confidence collapse) |
| `weights/experimental/yolo11s_pose_e1b_centre_960/best.pt` | 20,193,032 | `67c68e0b0e4d12e2ae6c77772dcbeeae6a1f1da92eacf0fa30b49d54eb99dddf` | `runs_pose/yolo11s_pose_e1b_centre_960/weights/best.pt` | E1b, unknown points at box centre; same data; 40 epochs |
| `weights/experimental/yolo11s_pose_e2_softconf_960/best.pt` | 20,193,032 | `6fdd3a3eaa623550910eab7d96cd6913fef097ef7b69f9316c39edbdceb6a7d2` | `runs_pose/yolo11s_pose_e2_softconf_960/weights/best.pt` | E2, soft teacher-confidence targets; same data; 40 epochs (A100) |

`weights/research/` was not created: the only research model, the ViTPose-L SwimXYZ ONNX, is kept external (§8).
None of these checkpoints is better than the pretrained model on the independent reference
(`docs/POSE_ESTIMATION_BLUEPRINT.md` §12).

## 3. Training run outputs — `training/<run>/`

For each of the four runs: `args.yaml` (exact training configuration), `results.csv` / `results.png` (per-epoch
losses and validation metrics), Ultralytics curves (`Box*_curve.png`, `Pose*_curve.png`), confusion matrices,
`labels.jpg`, `train_batch*.jpg`, `val_batch*_labels.jpg` / `_pred.jpg`. Validation metrics are agreement with the
teacher-derived validation labels of that run, **not accuracy**, and are not comparable across runs (different
validation sets). `train_batch*` / `val_batch*` images show frames of the project's source dataset (licence: §8).

## 4. Evaluation outputs — `evaluation/`

| Path | Content |
| --- | --- |
| `evaluation/drive/baseline/overlay_run0.jpg`, `overlay_run1.jpg` | Early baseline overlays (zero-shot comparison) |
| `evaluation/drive/ai_reference/` | Test-27 evaluation of pretrained vs distilled against the ViTPose teacher (`eval_*_test27.csv`, `comparison_test27.csv`), `split_manifest.csv` (train/val/test split of the 362-frame set), sample and label-check images |
| `evaluation/drive/e0/` | E0 diagnostics: `e0_predictions.jsonl` (21.7 MB: pretrained YOLO and ViTPose on all 5,613 frames plus flips), `e0_per_keypoint.csv`, `e0_report.txt`, disputed held-out frames (`e0_disputed_heldout.csv/.jpg`) |
| `evaluation/drive/e1/` | E1 / E1b: `e1_manifest.csv` and `e1_dataset_build.txt` (dataset split and coverage), E3 and Test-27 tables (`e3_*.csv`, `t27_*.csv`) |
| `evaluation/drive/e2/` | E2: `e2_manifest.csv`, `e2_dataset_build.txt`, E3 / Test-27 tables, confidence calibration (`calib_e3.csv`, `calib_t27.csv`) |
| `evaluation/drive/e3/` | E3 (14 hardest held-out frames): ViTPose vs pretrained vs distilled tables |
| `evaluation/drive/human/` | **Despite the folder name, no human labels**: results against the independent **AI** visual reference on Test-27 (`t27vis_*.csv`, `t27vis_calibration.csv`) |
| `evaluation/integration_benchmarks_20261127/` | Pose OFF vs ON benchmarks from the integration (`bench_pose_pipeline.py`) |

## 5. Demo rerun — `demo_results/`

12 per-video JSON files and `demo_results.csv` (full `LifeguardRuntime`, `configs/ai_z5cc_pose.yaml` +
`configs/system_demo_drone.yaml`, each video also with pose off), `run_log.txt`, 36 overlay PNGs drawn by the
dashboard's own `annotate()`, and 4 real dashboard captures from the integration session. The demo videos stay in
`E:\lifeguardDron\demo\` (unchanged; sha256 checked before and after).

## 6. Dataset metadata — `datasets_metadata/`

| Path | Content |
| --- | --- |
| `pose_annotation_set_v2/manifest.csv` | 362-frame pose subset: image, split (`real_train` 300 / `real_val` 62), block, frame, person box |
| `e3_set/manifest.csv` + `images/` (14 JPG) | E3 hard set. **Images come from the project's source dataset, whose licence is not documented: do not upload them publicly until verified**; the manifest alone is safe |
| `audit/AUDIT_REPORT.json` | Phase 0 audit of the 5,613-frame dataset (Drive `pose_audit_out/`) |
| `evaluation/drive/e1/e1_manifest.csv`, `e2/e2_manifest.csv`, `ai_reference/split_manifest.csv` | Split information for E1/E2 and the 362-frame set |

## 7. AI-generated references (NOT human ground truth) — `references/`

| Path | Content | Provenance |
| --- | --- | --- |
| `references/drive_ai_reference/ai_reference_unverified_all362.json`, `…_test27.json` | ViTPose-L SwimXYZ teacher references | `ai_reference_unverified` |
| `references/drive_ai_reference/proposals_coco_keypoints_vitpose_unverified.json` | ViTPose proposals for the 362-frame set (CVAT import) | `model_proposal_unverified` |
| `references/cvat/…` | CVAT exports of task 2796123 (backup) and task 2796177 (E3 AI reference); the proposals imported into CVAT | AI / model proposals |
| `artifacts/pose_e3/` (existing folder, kept in place) | E3 AI visual reference `pose_gt_e3.json` + provenance | `ai_visual_review_unverified` |
| `artifacts/pose_human_ref/` (existing folder, kept in place; **name is historical**) | Test-27 independent AI visual reference `pose_gt_t27_ai_visual.json` (201 points / 20 frames) + provenance | `ai_visual_review_unverified` — **not human** |

Three files in the Drive package were byte-identical to these existing files and were **reused, not duplicated**:
`pose_gt_e3.json` (kept in `artifacts/pose_e3/`), `pose_gt_t27_ai_visual.json` (kept in `artifacts/pose_human_ref/`)
and Drive's `proposals_train_only.json` (kept as `references/cvat/proposals_train_only_imported_to_cvat.json`); see
`drive_package/organize_log.csv`.
No human keypoint annotations exist anywhere in the project.

## 8. Kept external (not in the repository)

| Item | Where | Why |
| --- | --- | --- |
| SwimXYZ (303.7 GB, CC BY 4.0) | Zenodo | Never downloaded |
| ViTPose-L SwimXYZ ONNX `vitpose-l-swimxyz-mv.onnx` (1,177.2 MB per the package log) | Drive `MyDrive/pose_baseline/models/`; upstream Hugging Face `willalexm/vitpose-l-swimxyz` | Offline teacher only, never runtime; decision 2026-11-27: keep external (would need LFS; > 100 MB) |
| Training `last.pt` (4 × 19.3 MB) | Drive `runs_pose/*/weights/` | `best.pt` packaged |
| `datasets (1).zip` (5,613 frames) | Drive `MyDrive/` | Size; licence not verified |
| `pose_annotation_set_v2.zip` (102.9 MB, sha256 `a4b8861105efb33c…`) | `C:\Users\tarek\Downloads\`, Drive | > 100 MB; image licence; manifest packaged |
| `e3_set.zip` (3.9 MB) | Downloads, Drive | Duplicate of `datasets_metadata/e3_set/` |
| `E:\lifeguardDron\pose_package_20261127.zip` | Project folder (outside the repository) | Source archive of §2–§4; kept until you approve the package; do not commit it |
| Demo videos | `E:\lifeguardDron\demo\` | Third-party footage; not redistributed |
| `pose_gt_real_val.json` | — | Expected at Drive `MyDrive/`, **not present there** (the packaging cell found no such file) |

## 9. Scripts and notebooks (kept in place, not duplicated)

- Runtime code: `src/lifeguard/ai/pose/`, `src/lifeguard/ai/config.py`, `ai/perception/factory.py`,
  `ai/perception/real_engine.py`, `app/dashboard_controller.py`, `app/runtime.py`; `configs/ai_z5cc_pose.yaml`;
  `models/registry.yaml`; tests `tests/ai/test_pose.py`, `tests/ai/test_pose_real_model.py`, `tests/app/test_dashboard_pose.py`.
- Scripts `python_scripts/pose/`: frame selection `select_annotation_frames.py`; annotation preparation
  `make_cvat_proposals.py`, `make_ai_reference.py`; datasets `audit_pose_dataset.py`, `build_yolo_pose_dataset.py`,
  `build_e1_dataset.py`; training losses `ignore_aware_pose_loss.py`, `soft_conf_pose_loss.py`, `check_aug_visibility.py`;
  evaluation `zero_shot_eval.py`, `conf_calibration.py`, `e0_diagnostics.py`, `pose_common.py`; integration
  `bench_pose_pipeline.py`, `record_pose_demo.py`, `make_asset_inventory.py`.
- Notebooks `notebooks/`: `pose_audit`, `pose_zero_shot`, `pose_baseline_eval`, `pose_distill_train`, `pose_e0_diagnostics`,
  `pose_e3_eval`, `pose_e1_train`, `pose_e2_train`, `pose_t27_ai_visual_eval`, `pose_human_eval` (ready, never run: no
  human labels). `notebooks/downloads.ipynb` is an empty file not created in these sessions (left untouched).

The requested `training/notebooks|scripts|configs` sub-folders were not created: notebooks and scripts stay where the
repository already keeps them (no duplicates), and each run's configuration is its `training/<run>/args.yaml`.

## 10. Git / Git LFS — findings and proposed changes (NOT applied, nothing staged)

Findings:
- `.gitattributes`: `models/*.pt filter=lfs diff=lfs merge=lfs -text` → covers `models/yolo11s-pose.pt` only.
- `.gitignore` line 13 `*.pt` and line 14 `*.onnx` ignore all weights; the existing detector weights in `models/`
  were force-added and are tracked through LFS (`git lfs ls-files`).
- `artifacts/pose/weights/experimental/*/best.pt` (4 × ~19.3 MB) match **no** LFS pattern and are ignored by `*.pt`.
- No file in `artifacts/pose/` is above 50 MB (largest: `evaluation/drive/e0/e0_predictions.jsonl`, 21.7 MB), so no
  other file needs LFS for size.

Proposed change (needed so the experimental weights go to LFS instead of plain git history):

```gitattributes
# add to .gitattributes
artifacts/pose/weights/**/*.pt filter=lfs diff=lfs merge=lfs -text
```

Then add the weights explicitly (they stay ignored by `*.pt`, like the existing `models/*.pt`):

```powershell
git add .gitattributes
git add -f models/yolo11s-pose.pt
git add -f artifacts/pose/weights/experimental/yolo11s_pose_distill_960/best.pt artifacts/pose/weights/experimental/yolo11s_pose_e1_ignore_960/best.pt artifacts/pose/weights/experimental/yolo11s_pose_e1b_centre_960/best.pt artifacts/pose/weights/experimental/yolo11s_pose_e2_softconf_960/best.pt
git lfs ls-files    # must list all five .pt files before you commit
```

Optional: `models/yolo11s-pose.pt` is the unmodified official file (Ultralytics, AGPL-3.0) and can be re-fetched and
checksum-verified with `python -m lifeguard.common.models fetch pose_yolo11s_coco_v1`, so uploading it is not required.
No `.onnx` or large `.zip` is in the repository, so no rule for them is needed now.

## 11. Upload checklist

1. `git status` (nothing staged); review the diff.
2. Apply §10 before adding any `.pt`.
3. Upload code, configs, tests, `python_scripts/pose/`, `notebooks/pose_*.ipynb`, `docs/POSE_*.md`, `artifacts/pose/`
   (hold back every image that shows source-dataset frames until that dataset's licence is confirmed:
   `datasets_metadata/e3_set/images/`, all `.jpg` under `training/` and `evaluation/drive/`, and the overlays in
   `demo_results/` if the demo footage licence is also unconfirmed),
   `artifacts/pose_e3/`, `artifacts/pose_human_ref/`.
4. Do not upload anything from §8.

## Appendix — every file (generated)

<!-- INVENTORY_START -->
| Path | Bytes | sha256 | Class | Source | LFS |
| --- | --- | --- | --- | --- | --- |
| `artifacts/pose/ASSET_MANIFEST.md` | – | (this file) | DOCUMENT | written in this repository (2026-11-27) | no |
| `artifacts/pose/datasets_metadata/audit/AUDIT_REPORT.json` | 3,986 | `4f72d41dcfe3fcac…` | METADATA | Google Drive MyDrive/pose_audit_out/AUDIT_REPORT.json | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_07996.jpg` | 275,092 | `016e49c7265fd2a2…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_08024.jpg` | 275,179 | `77c46cd891897907…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_08054.jpg` | 272,627 | `d65467eeba481913…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_08064.jpg` | 275,222 | `687db0cd84ff6ab1…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_08128.jpg` | 276,252 | `3d430220dc121577…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_08141.jpg` | 277,365 | `27a32d0327eeafcd…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_08150.jpg` | 282,105 | `1600f7d3a7def4ec…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_08192.jpg` | 278,851 | `3bb170dbc6d1dc8d…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_47155.jpg` | 287,086 | `b380c434028da401…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_47218.jpg` | 299,371 | `f425f5e3f3c3c5b6…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_50192.jpg` | 290,735 | `adf60573718d324c…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_50232.jpg` | 280,810 | `89789c35288b68ba…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_50264.jpg` | 284,859 | `5ef0456d0d4aace5…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/images/frame_50299.jpg` | 281,826 | `a832e34049aae281…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/e3_set/manifest.csv` | 928 | `89c7096ec6c7d780…` | METADATA | e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/ | no |
| `artifacts/pose/datasets_metadata/pose_annotation_set_v2/manifest.csv` | 24,745 | `4fd86430c7570c71…` | METADATA | pose_annotation_set_v2.zip (Drive), local extract | no |
| `artifacts/pose/demo_results/dashboard_screenshots_integration_20261127/dashboard_demo01_skeleton_live.jpg` | 33,123 | `d0fdba738914aabe…` | RESULT (demo rerun) | Streamlit dashboard capture, integration session 2026-11-27 | no |
| `artifacts/pose/demo_results/dashboard_screenshots_integration_20261127/dashboard_demo01_swimmers_table_pose_column.jpg` | 23,487 | `bce347a3ee2d2498…` | RESULT (demo rerun) | Streamlit dashboard capture, integration session 2026-11-27 | no |
| `artifacts/pose/demo_results/dashboard_screenshots_integration_20261127/dashboard_demo06_alarm_prealert.jpg` | 35,234 | `f43b15d7a7f5a0c4…` | RESULT (demo rerun) | Streamlit dashboard capture, integration session 2026-11-27 | no |
| `artifacts/pose/demo_results/dashboard_screenshots_integration_20261127/dashboard_demo06_swimmers_table_pose_column.jpg` | 31,993 | `a37311c7223010cd…` | RESULT (demo rerun) | Streamlit dashboard capture, integration session 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_01_swim_I-EKqLdbysk.json` | 1,610 | `b50ff1b987fa9d20…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_02_drown_I-EKqLdbysk.json` | 1,170 | `47003a3ab9afeede…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_03_swim_L0KTqPloUiU.json` | 1,166 | `d3d1afecdf0e0976…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_04_drown_L0KTqPloUiU.json` | 1,467 | `07054271390f1491…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_05_swim_L0KTqPloUiU.json` | 902 | `67be7199984b0b60…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_06_drown_T5mDQeDkca0.json` | 1,618 | `2ed948243f9a6441…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_07_swim_Y.json` | 1,272 | `03401d4ca087ff00…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_08_swim_Y.json` | 1,125 | `fd5be117e71949c0…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_09_drown_Y.json` | 976 | `1c2ef411f44bb9a1…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_10_swim_Y.json` | 1,126 | `39d7727f92e7b4df…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_11_drown_iVMA7GhSJjM.json` | 1,166 | `186bda184693a98a…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_12_drown_wQdAEA4a8IA.json` | 1,204 | `0bfffd02de20d973…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/demo_results.csv` | 2,160 | `4367209042be0c3f…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/run_log.txt` | 1,752 | `bae7eb434337e07e…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_01_swim_I-EKqLdbysk_frame00395_pose.png` | 494,986 | `3122fdc471331871…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_01_swim_I-EKqLdbysk_frame00455_pose.png` | 483,775 | `e031a7bc16a84301…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_01_swim_I-EKqLdbysk_frame00990_pose.png` | 516,055 | `cce4573399c7663f…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_02_drown_I-EKqLdbysk_frame00096_pose.png` | 314,573 | `3ad3c42a59d92c38…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_02_drown_I-EKqLdbysk_frame00240_pose.png` | 314,383 | `b5039a41d0fd77fd…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_02_drown_I-EKqLdbysk_frame00297_pose.png` | 394,777 | `3405782a0e00ca39…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_03_swim_L0KTqPloUiU_frame00228_pose.png` | 305,279 | `94cddbc6f3dbb26f…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_03_swim_L0KTqPloUiU_frame00295_pose.png` | 406,296 | `50e0c906a01a5fe5…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_03_swim_L0KTqPloUiU_frame00335_pose.png` | 424,453 | `2633ba78c8714827…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_04_drown_L0KTqPloUiU_frame00053_pose.png` | 479,418 | `b3f6d1cb9608bc2b…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_04_drown_L0KTqPloUiU_frame00143_pose.png` | 475,653 | `3bcd1e6a42c6b796…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_04_drown_L0KTqPloUiU_frame00269_pose.png` | 490,826 | `31ed0c7590c23db0…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_05_swim_L0KTqPloUiU_frame00070_pose.png` | 455,110 | `dfb27a34b4571ac2…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_05_swim_L0KTqPloUiU_frame00289_pose.png` | 325,271 | `da68ed1645b11f07…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_06_drown_T5mDQeDkca0_frame00007_pose.png` | 365,646 | `9159abb03b8152ff…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_06_drown_T5mDQeDkca0_frame00066_pose.png` | 349,583 | `be7aaae0698bdfab…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_06_drown_T5mDQeDkca0_frame00105_pose.png` | 326,241 | `b58f68e4dd4ea652…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_07_swim_Y_frame00049_pose.png` | 462,315 | `f05b8886e57b1c6d…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_07_swim_Y_frame00177_pose.png` | 499,455 | `b77d82e24275b66b…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_07_swim_Y_frame00709_pose.png` | 494,305 | `1f0e2e78417a1d0b…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_08_swim_Y_frame00248_pose.png` | 507,417 | `fde0d9b2877ff4cd…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_08_swim_Y_frame00282_pose.png` | 466,596 | `da54f568838631bd…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_08_swim_Y_frame00342_pose.png` | 489,383 | `310f9e7cfe3f9ee2…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_09_drown_Y_frame00001_pose.png` | 420,350 | `26cb26ce72981c04…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_09_drown_Y_frame00317_pose.png` | 409,705 | `7c1d8fd48a2d5926…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_09_drown_Y_frame00372_pose.png` | 348,309 | `5d81b54597472055…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_10_swim_Y_frame00038_pose.png` | 545,821 | `e77c0d445c356c1f…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_10_swim_Y_frame00699_pose.png` | 521,221 | `fa155a5737b57be0…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_10_swim_Y_frame00853_pose.png` | 512,908 | `4be0f11cba2d7f69…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_11_drown_iVMA7GhSJjM_frame00040_pose.png` | 327,990 | `42a15e319fa4044f…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_11_drown_iVMA7GhSJjM_frame00127_pose.png` | 320,165 | `01031053f568b028…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_11_drown_iVMA7GhSJjM_frame00157_pose.png` | 252,397 | `c06bfaaad9ca77a1…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_12_drown_wQdAEA4a8IA_frame00013_pose.png` | 423,956 | `e2bfd8e7aa4a6aa9…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/demo_results/screenshots/demo_12_drown_wQdAEA4a8IA_frame00070_pose.png` | 441,430 | `a3b6fa21b6f2cfc6…` | RESULT (demo rerun) | record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27 | no |
| `artifacts/pose/drive_package/organize_log.csv` | 25,567 | `34c1a525d8ee70bd…` | PROVENANCE | generated while organizing the Drive package | no |
| `artifacts/pose/drive_package/PACKAGE_MANIFEST_SHA256.csv` | 20,489 | `d4c92a7ccbada08c…` | PROVENANCE | Google Drive MyDrive/PACKAGE_MANIFEST_SHA256.csv | no |
| `artifacts/pose/drive_package/PACKAGE_NOT_INCLUDED.csv` | 1,204 | `f42542bcaf1fa056…` | PROVENANCE | Google Drive MyDrive/PACKAGE_NOT_INCLUDED.csv | no |
| `artifacts/pose/evaluation/drive/ai_reference/comparison_test27.csv` | 313 | `26fbbfb4dc561d32…` | RESULT | Google Drive MyDrive/pose_baseline/ai_reference/comparison_test27.csv | no |
| `artifacts/pose/evaluation/drive/ai_reference/eval_finetuned_test27.csv` | 222 | `eeab26773b60972f…` | RESULT | Google Drive MyDrive/pose_baseline/ai_reference/eval_finetuned_test27.csv | no |
| `artifacts/pose/evaluation/drive/ai_reference/eval_pretrained_test27.csv` | 226 | `7202754c77b18221…` | RESULT | Google Drive MyDrive/pose_baseline/ai_reference/eval_pretrained_test27.csv | no |
| `artifacts/pose/evaluation/drive/ai_reference/samples_ref_pretrained_finetuned.jpg` | 433,685 | `486832ece10a9a65…` | RESULT | Google Drive MyDrive/pose_baseline/ai_reference/samples_ref_pretrained_finetuned.jpg | no |
| `artifacts/pose/evaluation/drive/ai_reference/split_manifest.csv` | 16,947 | `783886d097eae67c…` | RESULT | Google Drive MyDrive/pose_baseline/ai_reference/split_manifest.csv | no |
| `artifacts/pose/evaluation/drive/ai_reference/train_label_check.jpg` | 151,677 | `6d7ff44337615651…` | RESULT | Google Drive MyDrive/pose_baseline/ai_reference/train_label_check.jpg | no |
| `artifacts/pose/evaluation/drive/baseline/overlay_run0.jpg` | 258,235 | `95ca76d576e9a4b7…` | RESULT | Google Drive MyDrive/pose_baseline/overlay_run0.jpg | no |
| `artifacts/pose/evaluation/drive/baseline/overlay_run1.jpg` | 286,231 | `165c0ade681a9c8d…` | RESULT | Google Drive MyDrive/pose_baseline/overlay_run1.jpg | no |
| `artifacts/pose/evaluation/drive/e0/e0_disputed_heldout.csv` | 829 | `13af05071cd994ad…` | RESULT | Google Drive MyDrive/pose_baseline/e0/e0_disputed_heldout.csv | no |
| `artifacts/pose/evaluation/drive/e0/e0_disputed_heldout.jpg` | 304,194 | `925c39b802d8d0be…` | RESULT | Google Drive MyDrive/pose_baseline/e0/e0_disputed_heldout.jpg | no |
| `artifacts/pose/evaluation/drive/e0/e0_per_keypoint.csv` | 1,227 | `21e78336838de14a…` | RESULT | Google Drive MyDrive/pose_baseline/e0/e0_per_keypoint.csv | no |
| `artifacts/pose/evaluation/drive/e0/e0_predictions.jsonl` | 22,410,926 | `3131f24aa090caa0…` | RESULT | Google Drive MyDrive/pose_baseline/e0/e0_predictions.jsonl | no |
| `artifacts/pose/evaluation/drive/e0/e0_report.txt` | 958 | `3baa4052fb5e9ee8…` | RESULT | Google Drive MyDrive/pose_baseline/e0/e0_report.txt | no |
| `artifacts/pose/evaluation/drive/e1/e1_dataset_build.txt` | 988 | `cb50cbc452e97bc1…` | RESULT | Google Drive MyDrive/pose_baseline/e1/e1_dataset_build.txt | no |
| `artifacts/pose/evaluation/drive/e1/e1_manifest.csv` | 212,505 | `8b565482627e4df8…` | RESULT | Google Drive MyDrive/pose_baseline/e1/e1_manifest.csv | no |
| `artifacts/pose/evaluation/drive/e1/e3_comparison.csv` | 607 | `2bd2d41f42cc967c…` | RESULT | Google Drive MyDrive/pose_baseline/e1/e3_comparison.csv | no |
| `artifacts/pose/evaluation/drive/e1/e3_distilled.csv` | 413 | `cacac1ea4ce77486…` | RESULT | Google Drive MyDrive/pose_baseline/e1/e3_distilled.csv | no |
| `artifacts/pose/evaluation/drive/e1/e3_e1.csv` | 415 | `4531ac76a93af597…` | RESULT | Google Drive MyDrive/pose_baseline/e1/e3_e1.csv | no |
| `artifacts/pose/evaluation/drive/e1/e3_e1b.csv` | 414 | `f1081a10dbc07daa…` | RESULT | Google Drive MyDrive/pose_baseline/e1/e3_e1b.csv | no |
| `artifacts/pose/evaluation/drive/e1/e3_pretrained.csv` | 420 | `3a0cadd8f0037e89…` | RESULT | Google Drive MyDrive/pose_baseline/e1/e3_pretrained.csv | no |
| `artifacts/pose/evaluation/drive/e1/t27_comparison.csv` | 606 | `4ec4f6646a7cf08c…` | RESULT | Google Drive MyDrive/pose_baseline/e1/t27_comparison.csv | no |
| `artifacts/pose/evaluation/drive/e1/t27_distilled.csv` | 414 | `d2c071991c8d8f4c…` | RESULT | Google Drive MyDrive/pose_baseline/e1/t27_distilled.csv | no |
| `artifacts/pose/evaluation/drive/e1/t27_e1.csv` | 412 | `67c062767c57ac7a…` | RESULT | Google Drive MyDrive/pose_baseline/e1/t27_e1.csv | no |
| `artifacts/pose/evaluation/drive/e1/t27_e1b.csv` | 414 | `37589c9662ffde12…` | RESULT | Google Drive MyDrive/pose_baseline/e1/t27_e1b.csv | no |
| `artifacts/pose/evaluation/drive/e1/t27_pretrained.csv` | 418 | `f5e316523cea910c…` | RESULT | Google Drive MyDrive/pose_baseline/e1/t27_pretrained.csv | no |
| `artifacts/pose/evaluation/drive/e2/calib_e3.csv` | 225 | `5def7089330c2b9b…` | RESULT | Google Drive MyDrive/pose_baseline/e2/calib_e3.csv | no |
| `artifacts/pose/evaluation/drive/e2/calib_t27.csv` | 226 | `fc254411cbcfb1e3…` | RESULT | Google Drive MyDrive/pose_baseline/e2/calib_t27.csv | no |
| `artifacts/pose/evaluation/drive/e2/e2_dataset_build.txt` | 1,045 | `216ff74b171479f0…` | RESULT | Google Drive MyDrive/pose_baseline/e2/e2_dataset_build.txt | no |
| `artifacts/pose/evaluation/drive/e2/e2_manifest.csv` | 212,505 | `8b565482627e4df8…` | RESULT | Google Drive MyDrive/pose_baseline/e2/e2_manifest.csv | no |
| `artifacts/pose/evaluation/drive/e2/e3_comparison.csv` | 698 | `c5f1f268f265a5cf…` | RESULT | Google Drive MyDrive/pose_baseline/e2/e3_comparison.csv | no |
| `artifacts/pose/evaluation/drive/e2/e3_distilled.csv` | 413 | `f510dd32a63e884b…` | RESULT | Google Drive MyDrive/pose_baseline/e2/e3_distilled.csv | no |
| `artifacts/pose/evaluation/drive/e2/e3_e1b.csv` | 414 | `db7b44ed83b18871…` | RESULT | Google Drive MyDrive/pose_baseline/e2/e3_e1b.csv | no |
| `artifacts/pose/evaluation/drive/e2/e3_e2.csv` | 413 | `2c3773bf24e5864e…` | RESULT | Google Drive MyDrive/pose_baseline/e2/e3_e2.csv | no |
| `artifacts/pose/evaluation/drive/e2/e3_pretrained.csv` | 420 | `4375fc01de0c6ec6…` | RESULT | Google Drive MyDrive/pose_baseline/e2/e3_pretrained.csv | no |
| `artifacts/pose/evaluation/drive/e2/t27_comparison.csv` | 697 | `0fbb904ec8ca4c63…` | RESULT | Google Drive MyDrive/pose_baseline/e2/t27_comparison.csv | no |
| `artifacts/pose/evaluation/drive/e2/t27_distilled.csv` | 414 | `43082a9e1d2bd330…` | RESULT | Google Drive MyDrive/pose_baseline/e2/t27_distilled.csv | no |
| `artifacts/pose/evaluation/drive/e2/t27_e1b.csv` | 414 | `fc1c9f0bc26cdee7…` | RESULT | Google Drive MyDrive/pose_baseline/e2/t27_e1b.csv | no |
| `artifacts/pose/evaluation/drive/e2/t27_e2.csv` | 413 | `0431ee70fcdb712e…` | RESULT | Google Drive MyDrive/pose_baseline/e2/t27_e2.csv | no |
| `artifacts/pose/evaluation/drive/e2/t27_pretrained.csv` | 418 | `40d08f786aa08532…` | RESULT | Google Drive MyDrive/pose_baseline/e2/t27_pretrained.csv | no |
| `artifacts/pose/evaluation/drive/e3/e3_comparison.csv` | 618 | `422e89a8566bc657…` | RESULT | Google Drive MyDrive/pose_baseline/e3/e3_comparison.csv | no |
| `artifacts/pose/evaluation/drive/e3/e3_distilled.csv` | 413 | `27e6bf254ad491d4…` | RESULT | Google Drive MyDrive/pose_baseline/e3/e3_distilled.csv | no |
| `artifacts/pose/evaluation/drive/e3/e3_vitpose_and_pretrained.csv` | 523 | `909a47ee6a74e20d…` | RESULT | Google Drive MyDrive/pose_baseline/e3/e3_vitpose_and_pretrained.csv | no |
| `artifacts/pose/evaluation/drive/human/t27vis_calibration.csv` | 227 | `6dfb503e5d5499f4…` | RESULT | Google Drive MyDrive/pose_baseline/human/t27vis_calibration.csv | no |
| `artifacts/pose/evaluation/drive/human/t27vis_comparison.csv` | 675 | `4b71e6119399ebf0…` | RESULT | Google Drive MyDrive/pose_baseline/human/t27vis_comparison.csv | no |
| `artifacts/pose/evaluation/drive/human/t27vis_distilled.csv` | 408 | `06f932a9b329e556…` | RESULT | Google Drive MyDrive/pose_baseline/human/t27vis_distilled.csv | no |
| `artifacts/pose/evaluation/drive/human/t27vis_e1b.csv` | 408 | `cd5afc67394f6dc5…` | RESULT | Google Drive MyDrive/pose_baseline/human/t27vis_e1b.csv | no |
| `artifacts/pose/evaluation/drive/human/t27vis_e2.csv` | 408 | `0da283f3ab83474c…` | RESULT | Google Drive MyDrive/pose_baseline/human/t27vis_e2.csv | no |
| `artifacts/pose/evaluation/drive/human/t27vis_pretrained.csv` | 413 | `be30c7ab3d4ade72…` | RESULT | Google Drive MyDrive/pose_baseline/human/t27vis_pretrained.csv | no |
| `artifacts/pose/evaluation/integration_benchmarks_20261127/bench_demo_01_first600_system.json` | 2,167 | `a582e8b3f71fec16…` | RESULT | bench_pose_pipeline.py, integration session 2026-11-27 | no |
| `artifacts/pose/evaluation/integration_benchmarks_20261127/bench_demo_01_full_system_demo_drone.json` | 2,615 | `22f1a6297669c8bc…` | RESULT | bench_pose_pipeline.py, integration session 2026-11-27 | no |
| `artifacts/pose/evaluation/integration_benchmarks_20261127/bench_demo_05_system.json` | 2,165 | `667ad8a0a1547c68…` | RESULT | bench_pose_pipeline.py, integration session 2026-11-27 | no |
| `artifacts/pose/evaluation/integration_benchmarks_20261127/bench_demo_06_system.json` | 2,516 | `f652ee580f5ae1b8…` | RESULT | bench_pose_pipeline.py, integration session 2026-11-27 | no |
| `artifacts/pose/evaluation/integration_benchmarks_20261127/bench_demo_11_system.json` | 2,165 | `9ab3fcc845aba43f…` | RESULT | bench_pose_pipeline.py, integration session 2026-11-27 | no |
| `artifacts/pose/evaluation/integration_benchmarks_20261127/bench_demo_11_system_demo_drone.json` | 2,277 | `dba082abebb9e35a…` | RESULT | bench_pose_pipeline.py, integration session 2026-11-27 | no |
| `artifacts/pose/references/cvat/cvat_task_2796123_backup_before_train_only_reimport.zip` | 204,600 | `ed7c1cc0e20df418…` | REFERENCE (AI, not human ground truth) | CVAT (app.cvat.ai) export, downloaded 2026-11-27 | no |
| `artifacts/pose/references/cvat/cvat_task_2796177_e3_ai_reference_export.zip` | 8,399 | `f32c7bfb8c415946…` | REFERENCE (AI, not human ground truth) | CVAT (app.cvat.ai) export, downloaded 2026-11-27 | no |
| `artifacts/pose/references/cvat/cvat_task_2796177_person_keypoints_default.json` | 8,219 | `c128441eed889679…` | REFERENCE (AI, not human ground truth) | CVAT (app.cvat.ai) export, downloaded 2026-11-27 | no |
| `artifacts/pose/references/cvat/proposals_train_only_imported_to_cvat.json` | 154,463 | `29576172a4fa0d8c…` | REFERENCE (AI, not human ground truth) | Google Drive MyDrive/pose_baseline/proposals_train_only.json | no |
| `artifacts/pose/references/drive_ai_reference/ai_reference_unverified_all362.json` | 250,507 | `496fa1f61465a752…` | REFERENCE (AI, not human ground truth) | Google Drive MyDrive/pose_baseline/ai_reference/ai_reference_unverified_all362.json | no |
| `artifacts/pose/references/drive_ai_reference/ai_reference_unverified_test27.json` | 18,456 | `93cb885c5774295b…` | REFERENCE (AI, not human ground truth) | Google Drive MyDrive/pose_baseline/ai_reference/ai_reference_unverified_test27.json | no |
| `artifacts/pose/references/drive_ai_reference/proposals_coco_keypoints_vitpose_unverified.json` | 178,035 | `cbc18b68e0f24b51…` | REFERENCE (AI, not human ground truth) | pose_annotation_set_v2.zip (Drive), local extract | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/args.yaml` | 1,833 | `8f59d8cee59ac4f9…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/args.yaml | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/BoxF1_curve.png` | 121,053 | `f7d6c5180c6b2aef…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/BoxF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/BoxP_curve.png` | 106,135 | `6fd9b5812eba7800…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/BoxP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/BoxPR_curve.png` | 74,676 | `97b560bdf41c0c89…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/BoxPR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/BoxR_curve.png` | 97,306 | `cd747a8099c314d1…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/BoxR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/confusion_matrix.png` | 105,023 | `e05801c121e613e9…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/confusion_matrix.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/confusion_matrix_normalized.png` | 94,695 | `86b102e4ddff9a46…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/confusion_matrix_normalized.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/labels.jpg` | 141,627 | `2dfe1ef23ff4fadc…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/PoseF1_curve.png` | 82,638 | `9077316fc0ac7357…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/PoseF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/PoseP_curve.png` | 109,761 | `a5911ed05336af10…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/PoseP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/PosePR_curve.png` | 75,213 | `23842ff0d34a5e9f…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/PosePR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/PoseR_curve.png` | 77,010 | `00efbe71eac3371d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/PoseR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/results.csv` | 12,560 | `5c725106a68ad4bf…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/results.csv | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/results.png` | 502,418 | `3781aca326de7f2d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/results.png | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/train_batch0.jpg` | 365,777 | `10fde3daafbc5c62…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/train_batch0.jpg | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/train_batch1.jpg` | 366,105 | `36cf605ec78f4bb9…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/train_batch1.jpg | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/train_batch2.jpg` | 403,943 | `af1ebf27940f2107…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/train_batch2.jpg | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/val_batch0_labels.jpg` | 262,313 | `0d6dd6559d06206d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/val_batch0_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/val_batch0_pred.jpg` | 301,808 | `344d91f61bce8942…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/val_batch0_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/val_batch1_labels.jpg` | 126,594 | `6e540e7989222834…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/val_batch1_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_distill_960/val_batch1_pred.jpg` | 132,086 | `e4d5d3c0385f283d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/val_batch1_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/args.yaml` | 1,829 | `bc042e7798911ac9…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/args.yaml | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/BoxF1_curve.png` | 102,629 | `2255b83472e7fc63…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/BoxF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/BoxP_curve.png` | 90,620 | `39b8ebe0657934d2…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/BoxP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/BoxPR_curve.png` | 79,182 | `b72b8f59e2872951…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/BoxPR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/BoxR_curve.png` | 106,347 | `6189b6426989a81a…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/BoxR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/confusion_matrix.png` | 106,805 | `4b2065d545cf740b…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/confusion_matrix.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/confusion_matrix_normalized.png` | 94,695 | `86b102e4ddff9a46…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/confusion_matrix_normalized.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/labels.jpg` | 108,830 | `b67b70c2d09e8053…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/PoseF1_curve.png` | 99,774 | `b7a76a06147613cb…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/PoseF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/PoseP_curve.png` | 95,228 | `390364b106387bd2…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/PoseP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/PosePR_curve.png` | 83,461 | `7cc357317722fcef…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/PosePR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/PoseR_curve.png` | 97,968 | `dce37cfe3bc85257…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/PoseR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/results.csv` | 7,808 | `2ec741c8b5b87e35…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/results.csv | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/results.png` | 542,341 | `0bfd578566c6f17b…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/results.png | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/train_batch0.jpg` | 378,227 | `bd3bb7e1c18b41d7…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/train_batch0.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/train_batch1.jpg` | 395,727 | `5800406daf494078…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/train_batch1.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/train_batch2.jpg` | 391,077 | `7f5eecba2757c23c…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/train_batch2.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/train_batch5370.jpg` | 330,784 | `12de59bd97f53c27…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/train_batch5370.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/train_batch5371.jpg` | 324,074 | `0bdfc7d4080ebc97…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/train_batch5371.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/train_batch5372.jpg` | 309,731 | `4ee5478b0991cc36…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/train_batch5372.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/val_batch0_labels.jpg` | 302,311 | `863dd8c5d27d9a2a…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/val_batch0_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/val_batch0_pred.jpg` | 306,092 | `23c62b7f2e5c14f8…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/val_batch0_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/val_batch1_labels.jpg` | 309,923 | `fe46686e7bd257fa…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/val_batch1_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/val_batch1_pred.jpg` | 309,553 | `67e8a9c348778e07…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/val_batch1_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/val_batch2_labels.jpg` | 320,390 | `c306bcb723224fa8…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/val_batch2_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1_ignore_960/val_batch2_pred.jpg` | 322,748 | `bfb1f817cd985e32…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/val_batch2_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/args.yaml` | 1,831 | `5d42e1cf56cafa81…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/args.yaml | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/BoxF1_curve.png` | 101,821 | `4a9d437532ee3258…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/BoxF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/BoxP_curve.png` | 92,913 | `921cd690f4ff78df…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/BoxP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/BoxPR_curve.png` | 83,779 | `474371beb3d454dc…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/BoxPR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/BoxR_curve.png` | 105,090 | `726181480a90e112…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/BoxR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/confusion_matrix.png` | 98,836 | `f5df94f222fac813…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/confusion_matrix.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/confusion_matrix_normalized.png` | 95,389 | `aabdf76d897ba478…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/confusion_matrix_normalized.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/labels.jpg` | 108,830 | `b67b70c2d09e8053…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/PoseF1_curve.png` | 96,492 | `23e3d173bd3dbf7d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/PoseF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/PoseP_curve.png` | 97,075 | `05b3cd4d34b782e8…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/PoseP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/PosePR_curve.png` | 85,698 | `e7422935761d0346…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/PosePR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/PoseR_curve.png` | 96,899 | `914b46d5dc569bcf…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/PoseR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/results.csv` | 7,801 | `6d3e3745e707fd1d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/results.csv | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/results.png` | 564,343 | `b13b2c51db67a4b7…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/results.png | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/train_batch0.jpg` | 370,638 | `25dcf8c2dedd25fa…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/train_batch0.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/train_batch1.jpg` | 383,415 | `a6c66ef9e2904717…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/train_batch1.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/train_batch2.jpg` | 371,065 | `2fd9f92f8fc36fb0…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/train_batch2.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/train_batch5370.jpg` | 323,891 | `940bf71d0878ed7d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/train_batch5370.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/train_batch5371.jpg` | 316,426 | `0af7c9c219b5ac87…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/train_batch5371.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/train_batch5372.jpg` | 304,410 | `796d8e98ae9e3737…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/train_batch5372.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/val_batch0_labels.jpg` | 302,311 | `863dd8c5d27d9a2a…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/val_batch0_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/val_batch0_pred.jpg` | 307,527 | `e01bf34aaa1a71f6…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/val_batch0_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/val_batch1_labels.jpg` | 309,923 | `fe46686e7bd257fa…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/val_batch1_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/val_batch1_pred.jpg` | 313,801 | `851f4733e13df79c…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/val_batch1_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/val_batch2_labels.jpg` | 320,390 | `c306bcb723224fa8…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/val_batch2_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e1b_centre_960/val_batch2_pred.jpg` | 326,418 | `3941ed5a61345e20…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/val_batch2_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/args.yaml` | 1,833 | `98c214bebc680fef…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/args.yaml | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/BoxF1_curve.png` | 101,712 | `912e60be70586f0b…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/BoxF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/BoxP_curve.png` | 89,056 | `bdefac072ffdab97…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/BoxP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/BoxPR_curve.png` | 81,103 | `ee77c1dd34f40fb8…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/BoxPR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/BoxR_curve.png` | 103,375 | `565e54a87c0d59c4…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/BoxR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/confusion_matrix.png` | 101,296 | `69162460a6c1db7b…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/confusion_matrix.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/confusion_matrix_normalized.png` | 99,836 | `35595211e695a833…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/confusion_matrix_normalized.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/labels.jpg` | 108,830 | `b67b70c2d09e8053…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/PoseF1_curve.png` | 98,474 | `63c1cf25ef92d25d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/PoseF1_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/PoseP_curve.png` | 95,928 | `82db5182bab238b0…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/PoseP_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/PosePR_curve.png` | 81,062 | `68b06d24f6cfedba…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/PosePR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/PoseR_curve.png` | 92,369 | `76dd1eed8faf6be0…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/PoseR_curve.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/results.csv` | 7,798 | `5ad4c7fe84b1f640…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/results.csv | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/results.png` | 476,281 | `c615f6d25566d300…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/results.png | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/train_batch0.jpg` | 370,638 | `25dcf8c2dedd25fa…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/train_batch0.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/train_batch1.jpg` | 383,415 | `a6c66ef9e2904717…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/train_batch1.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/train_batch2.jpg` | 371,065 | `2fd9f92f8fc36fb0…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/train_batch2.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/train_batch5370.jpg` | 323,891 | `940bf71d0878ed7d…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/train_batch5370.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/train_batch5371.jpg` | 316,426 | `0af7c9c219b5ac87…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/train_batch5371.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/train_batch5372.jpg` | 304,410 | `796d8e98ae9e3737…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/train_batch5372.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/val_batch0_labels.jpg` | 302,311 | `863dd8c5d27d9a2a…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/val_batch0_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/val_batch0_pred.jpg` | 311,266 | `b335a7154e2fd2bc…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/val_batch0_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/val_batch1_labels.jpg` | 309,923 | `fe46686e7bd257fa…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/val_batch1_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/val_batch1_pred.jpg` | 312,853 | `b92ccb83857e1b82…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/val_batch1_pred.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/val_batch2_labels.jpg` | 320,390 | `c306bcb723224fa8…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/val_batch2_labels.jpg | no |
| `artifacts/pose/training/yolo11s_pose_e2_softconf_960/val_batch2_pred.jpg` | 327,315 | `04266d91bcac4578…` | EXPERIMENTAL (run output) | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/val_batch2_pred.jpg | no |
| `artifacts/pose/weights/experimental/yolo11s_pose_distill_960/best.pt` | 20,198,472 | `0cc0896e8145d421…` | EXPERIMENTAL | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_distill_960/weights/best.pt | yes |
| `artifacts/pose/weights/experimental/yolo11s_pose_e1_ignore_960/best.pt` | 20,193,032 | `23230614358e4e89…` | EXPERIMENTAL | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1_ignore_960/weights/best.pt | yes |
| `artifacts/pose/weights/experimental/yolo11s_pose_e1b_centre_960/best.pt` | 20,193,032 | `67c68e0b0e4d12e2…` | EXPERIMENTAL | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e1b_centre_960/weights/best.pt | yes |
| `artifacts/pose/weights/experimental/yolo11s_pose_e2_softconf_960/best.pt` | 20,193,032 | `6fdd3a3eaa623550…` | EXPERIMENTAL | Google Drive MyDrive/pose_baseline/runs_pose/yolo11s_pose_e2_softconf_960/weights/best.pt | yes |
| `models/yolo11s-pose.pt` | 20,373,161 | `1060bda4a2701206…` | RUNTIME | Ultralytics GitHub release v8.3.0 (official) | yes |
<!-- INVENTORY_END -->
