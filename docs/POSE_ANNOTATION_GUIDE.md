# Pose annotation guide (CVAT) — real validation / training frames

Goal: about 150 `real_val` + 300 `real_train` frames with human-checked COCO-17 keypoints from OUR footage.
The CVAT steps below are written from general knowledge of CVAT and were **not run here [A]**; menu names may differ
in your CVAT version. The Colab notebook `notebooks/pose_zero_shot.ipynb` produces the inputs.

## Inputs (made by the notebook)

- `pose_annotation_set/images/` and `manifest.csv` (image, split, block, frame, box).
- `proposals_coco_keypoints.json`: keypoints PROPOSED by the model (provenance `model_proposal_unverified`).

## Steps

1. Create a CVAT project with one label `person`, type **Skeleton**, with these 17 points in this order:
   nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow,
   left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.
   Edges: any (not used by our tools). Names must match exactly for the import.
2. Create a task from `images/` (do the `real_val` frames first; filter by `manifest.csv`).
3. Import annotations: *Upload annotations* → format **COCO Keypoints 1.0** → `proposals_coco_keypoints.json`.
   **`real_val` frames contain NO model keypoints** (box only): annotate them from scratch so the ground truth is
   not biased toward the proposal model (ViTPose), which would make the ViTPose vs YOLO11-pose comparison unfair.
   Only `real_train` frames are pre-filled. Whether CVAT imports an all-zero skeleton cleanly is untested [A]; if it
   does not, import only the `real_train` frames and create the `real_val` skeletons with CVAT's skeleton tool.
   If the import fails on labels, check that the skeleton point names match the list above.
4. Correct every frame by hand:
   - Move each point onto the joint. "Left/right" are the swimmer's own left/right, not the image's.
   - Point not visible (under water, hidden by splash or by another body part) → mark it **outside/not visible**
     rather than guessing. Do not annotate points you cannot place.
   - One swimmer per frame in the selected set (frames with several boxes were skipped).
   - Mark a frame **unusable** (delete the skeleton) if the body is not identifiable.
5. Export: **COCO Keypoints 1.0** (this is `--gt` for `zero_shot_eval.py`). Keep the export file name with the date.
6. Record provenance: every frame that went through step 4 is `human_corrected`; frames not reviewed stay
   `model_proposal_unverified` and must NOT be used in validation or test. Two people checking the same 20 frames
   gives a rough annotation-noise level (report the mean pixel difference).

## Rules

- Do not use the source classes (Swimming / Struggling / Drowning) in any way here; they are not pose labels,
  and `drowning` labels remain assigned only by a named human reviewer.
- `real_val` frames are never used for training or for choosing augmentations; select models on them, report on a
  second set later.
- Splits are by frame-index block (the source zip has no video ids); the buffer around block edges is dropped.
