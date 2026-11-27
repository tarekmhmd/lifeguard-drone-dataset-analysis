# Pose estimation — dataset and model decision report

Date: 2026-11-26. Evidence level: web pages read on this date (listed per row). Legend: **[V]** read on the
source page, **[U]** could not be verified (page silent, or fetch failed), **[A]** assumption / recommendation.
No dataset was downloaded. Nothing below is measured on our footage.

## 1. Our own dataset (`datasets (1).zip`) — audited, not usable for pose as-is [V]

3 classes (Swimming, Struggling, Drowning), 1 box per image, 5,613 images 1920x1080, no keypoints. Classes are
exactly balanced (1,496 per class in train, 375 in val). Train and val frame indices overlap (7-68,023 vs
616-68,082): likely a random frame-level split, so neighbouring frames probably sit in both splits [A, not measured].
Label provenance unknown; its Drowning/Struggling labels are NOT pose labels and are not used as such.
Use: a **source of images to annotate** (tall boxes, median about 248x384 px at 1080p), nothing more.

## 2. Resources

| | SwimXYZ | Augsburg 2009 | AUT (Cao and Yan) | SJSU D1 Swim | CADDY | ViTPose-L SwimXYZ |
|---|---|---|---|---|---|---|
| Type | Synthetic | Real | Real | Real | Real (scuba divers) | Model |
| View | 5: front, aerial, side underwater, side above-water, side water-level [V paper] | not stated [U] | underwater view [V abstract] | not stated [U] | underwater stereo | same 5 views |
| Annotations | GT 2D+3D joints; formats COCO17, COCO25, 48-joint "Base"; SMPL motions [V] | 14 manual joints, about 1,200 frames, backstroke [V] | keypoint dataset, multi-swimmer, size and joint count not stated [U] | keypoint GT not documented; pose comes from YOLOv7 pose [V repo], Kaggle page unreadable [U] | IMU suit angles as pose GT; no 2D keypoints mentioned [V]; about 12.7K free-swimming stereo frames | 17 COCO keypoints [V] |
| Styles | freestyle, back, breast, butterfly [V] | backstroke | not stated | fly, back, breast, free (+ underwater, diving classes) [V repo] | n/a (diving) | trained on freestyle only (25k frames) [V] |
| Size | 11,520 videos, 5 s at 60 fps, 3.4M frames [V]; labels zips 1.6-1.7 GB per stroke [V]; video part e.g. Freestyle part 1 = 40.3 GB [V] | about 1,200 frames | not stated | not stated | about 10K gesture + 12.7K swimming | PyTorch + ONNX, size not listed |
| Access | Open, Zenodo [V] | report PDF linked; dataset file link not found [U] | thesis PDF only found; dataset release not stated [U] | Kaggle link exists, contents unread [U] | paper says public; download link not found [U] | open, no gating stated [V] |
| License | CC BY 4.0 [V] | not stated [U] | not stated [U] | repo code MIT; data license not stated [V/U] | not found [U] | Apache-2.0 [V] |
| Fit for us | High as pretraining data, domain gap to real | Low (2009, backstroke only, 14 joints) | Highest value **if** obtainable | Possible raw material, unknown | Low: auxiliary only | High as baseline |

Key caveats:

- **SwimXYZ domain gap [V]:** the authors list lack of subject, swimsuit and environment diversity; real-image
  validation was qualitative only ("seem satisfactory"); the only quantitative test was synthetic.
- **ViTPose-L SwimXYZ [V model card]:** top-down (needs person boxes), input 256x192 crops, output 17 COCO heatmaps.
  Trained on 25,000 freestyle frames for 5 epochs. PCK@0.05 0.982 overall, side underwater 0.989, but on a
  held-out **synthetic** split; the base COCO model scored 0.653 on the same split. Also ships a 3D lifter,
  SMPL regressors and a stroke classifier (not needed now). Mesh output needs the licensed SMPL model.
  The card says labels were converted from OpenPose COCO-18 to COCO-17 and checked by overlay: we must re-check.
- **AUT [V]:** authors argue underwater view shows the torso better than side views. This matches a pool with
  underwater or overhead cameras, but we cannot see the data.
- **Survey claim [V]:** the SwimXYZ paper states that, as of 2023, no swimming pose dataset was publicly available.
  This explains why the real ones above are hard to obtain.

## 3. Decision

**Recommended strategy: start from the ViTPose-L SwimXYZ checkpoint, validate zero-shot on a small hand-corrected
set of our own frames, then fine-tune on real + synthetic only if the zero-shot result is not good enough.**
Do not download SwimXYZ videos yet.

Why:

1. A pretrained swimming-specific model exists and is Apache-2.0; training a pose model from scratch needs
   labels we do not have, and its synthetic metrics say nothing about our pool.
2. Real-world performance is the goal, so the real validation set must exist before any training choice.
   Only our own frames show our camera, water and lighting.
3. **Change to my earlier plan [A]:** top-down ViTPose fed with our existing tracker boxes replaces "YOLO11-pose
   alongside the detector". We already have detector + ByteTrack, so no pose-to-track matching is needed.
   YOLO11-pose stays as a speed baseline. Cost: ViTPose-L is heavy; on the GTX 1650 it probably needs
   every-N-frames or only suspicious tracks [A, unmeasured].
4. Pose stays additional evidence, risk weight 0, as before.

## 4. Next steps (in order, each gated)

1. **Real validation set (the missing piece).** Pick about 150 frames from our footage, spaced far apart in
   time, covering both camera types (side Z5cc, overhead ddt) and datasets (1) images. Group by contiguous
   frame-index runs and put whole groups in one split (no video ids exist). Pre-fill keypoints with the
   SwimXYZ model, correct by hand in CVAT, store `provenance`. Roughly 150 val + 300 train frames matches the
   Blueprint estimate of "a few hundred".
2. **Zero-shot comparison** on the corrected frames: ViTPose-L SwimXYZ vs COCO ViTPose-L vs YOLO11-pose, using
   upper-body PCK, detection rate and latency. Report per camera type.
3. **Inspect one SwimXYZ label zip** (Freestyle labels, 1.7 GB) to confirm joint order, COCO17 mapping and
   view naming before using it for anything. Only then decide on fine-tuning data; the labels alone are small
   compared with the 40 GB video parts, and 60 fps frames are highly redundant, so subsample.
4. **Ask for the real data in parallel (cheap):** write to the AUT authors (Cao / Yan) about the underwater
   keypoint dataset and its licence; check the SJSU Kaggle Data tab by hand for labels; try the Augsburg
   authors (Greif / Lienhart) for the 2009 data. Treat all as unavailable until files are in hand.
5. **If zero-shot is not enough:** fine-tune the SwimXYZ checkpoint on real frames (oversampled) mixed with
   subsampled synthetic side-underwater and aerial frames, A100, one change at a time, selecting on the
   real validation split.
6. Then integration as in `POSE_ESTIMATION_PLAN_AND_PROMPT.md` Phase 5-8 (interface, config, registry
   `kind: pose`, dashboard skeleton, tests including "pose alone cannot raise ALARM").

## 5. Missing items we must obtain

- Corrected keypoints on our own frames (nothing else substitutes).
- Licence terms for AUT, SJSU and CADDY data, if any is used.
- A pool-camera clip set with known video ids, so splits can be by video.
- A latency measurement of ViTPose-L on the target GPU.

Sources: Zenodo record 8399376 and 8402009; Hugging Face willalexm/vitpose-l-swimxyz; arXiv 2310.04360
(SwimXYZ); Augsburg project page; AUT repository handle 10292/15043; SJSU swimmervision page and GitHub repo;
arXiv 1807.04856 (CADDY). Failed fetches (ECONNREFUSED): Augsburg report page, Constructor University data page,
AUT thesis PDF; Springer article paywalled.
