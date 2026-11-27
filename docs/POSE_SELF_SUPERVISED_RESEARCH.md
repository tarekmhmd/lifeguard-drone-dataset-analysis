# Pose estimation with unlabelled video: research and experiment plan

Date: 2026-11-27. Research and planning only: nothing trained or changed for this document.
Legend: **[V]** verified (repository, saved notebook outputs, installed source code, or the cited page) ·
**[A]** assumption or estimate · **[R]** recommendation.

## 0. Summary

- **Unlabelled data can make a pose model more consistent, not provably more correct.** Flip consistency, temporal
  smoothness and agreement between independent models are real learning signals, but a model can satisfy all of them
  and still be wrong (for example, a consistent left/right swap). Some human-labelled frames remain the only way to
  *measure* accuracy.
- **The last experiment failed for a specific, fixable reason [V].** In Ultralytics, a keypoint written as unlabelled
  (`v = 0`) is not ignored: the keypoint-confidence loss trains the model to predict "not present" there. Every
  low-confidence ViTPose point became a "this joint is absent" target. That matches the drop in hip/knee/ankle
  confidence from about 0.75 to 0.45 after fine-tuning.
- **Recommended next experiment: E0, a label-free diagnostic pass (no training).** Run the pretrained YOLO11s-pose and
  ViTPose-L SwimXYZ on all 5,613 frames, with flipped copies, and measure per keypoint where they agree, where each is
  stable over time, and where each fails. It costs about 15–30 minutes on a T4 [A], needs no labels, and decides
  whether any pseudo-label training (E1) is worth doing.

## 1. What exists (inspected)

| Item | State |
| --- | --- |
| Unlabelled frames | `datasets (1).zip`: 5,613 frames, 1920x1080, one person box per frame [V audit]. Frame numbers form 22 runs of >= 8 near-consecutive frames (gap <= 3), the longest 1,871 and 1,653 frames [V notebook C1]. Frame rate unknown [A]. |
| Pose subset | 362 frames: 300 `real_train` (8 frame-index blocks), 62 `real_val` (4 blocks: 2, 13, 15, 16); 27 of the 62 selected as test frames [V]. |
| Models | ViTPose-L SwimXYZ ONNX (1.23 GB) and YOLO11s-pose (20 MB) on Drive `pose_baseline/models/`; distilled YOLO11s-pose at `pose_baseline/runs_pose/yolo11s_pose_distill_960/weights/best.pt` (experimental, not adopted) [V]. |
| ViTPose labels | `ai_reference_unverified_all362.json`: mean 11.4 of 17 keypoints >= 0.5; 40/362 frames with < 3 upper-body points; nose missing in 173/362, right ear in 222/362 [V]. |
| Last training run | T4, 67 epochs, best epoch 37. Train pose loss 5.9 to 2.8, val pose loss 9.3 to 8.3 (overfitting); val pose mAP50 about 0.09-0.15 [V]. |
| Last comparison | On 27 test frames vs ViTPose references: PCK@0.2 upper 0.40 to 0.62, PCK@0.05 unchanged; misses 5 to 0; lower-body confidence fell; visual samples showed the teacher was often worse than pretrained YOLO [V]. |
| Hardware | Colab gave a Tesla T4 both times [V]; ViTPose 33.6 ms and YOLO11s-pose 12.4 ms per frame on T4 [V]. |
| Tools | `python_scripts/pose/`: `pose_common.py`, `make_ai_reference.py`, `build_yolo_pose_dataset.py`, `zero_shot_eval.py`, `select_annotation_frames.py`, `make_cvat_proposals.py`, `audit_pose_dataset.py` [V]. |

**Ultralytics loss detail [V, installed 8.3.40 source; Colab ran 8.4.220, re-check there]:**
`kpt_mask = gt_kpt[..., 2] != 0`; the position loss uses only `kpt_mask`, but
`kpts_obj_loss = bce_pose(pred_conf, kpt_mask.float())` applies to **all** 17 keypoints. So the YOLO label format has no
"unknown" state: `0` means "absent", and `1` or `2` both mean "visible, train the position". Any pseudo-label scheme
needs an "ignore" state, which requires a small custom loss.

## 2. Methods: what each one gives

| Method | Learning signal | Needs labels? | Fit for our footage |
| --- | --- | --- | --- |
| **Pseudo-labels from a teacher** | Copies the teacher's predictions | No (teacher needed) | Already tried. Only as good as the teacher, and the teacher is weak on splash and occlusion. |
| **Multi-teacher agreement** | Keeps a point only where *independent* models agree | No | Good: ViTPose (top-down transformer, swimming-tuned) and YOLO11s-pose (bottom-up CNN, COCO) fail differently. |
| **Equivariance (flip/scale consistency)** | A prediction on the flipped image, flipped back, should match | No | Good and cheap. Inconsistency flags unreliable points. Water reflections are flip-sensitive [A]. |
| **Temporal consistency** | Keypoints should move smoothly between near-consecutive frames | No | Useful: we have long runs. But joints genuinely appear and disappear at the surface, so smoothness must not be forced through those transitions. |
| **Optical-flow propagation** | Warp labels or features between frames | No | Weak here [A]: caustics, bubbles and refraction break flow estimates under water. Low priority. |
| **Mean Teacher / consistency regularisation** | Student matches an EMA copy of itself on augmented views | Small labelled set normally | Possible later. Known to collapse on pose heatmaps without care [V Xie 2021]. |
| **Self-supervised representation learning (DINO/MAE style)** | Learns image features without labels | No | Not recommended now [R]: 5,613 frames from a few clips are too few to beat COCO pretraining, and the YOLO backbone has no ready pipeline for it. |
| **Uncertainty estimation** | Disagreement, flip error and temporal jitter as per-point uncertainty | No | Core ingredient for filtering. |
| **Active learning** | Picks the frames a human should label next | Yes, a few | The cheapest way to get a trusted metric: label the 20–30 frames the models disagree on most [V Liu & Ferrari 2017]. |

### A. Can temporal information help? [R]
Yes, as a **filter and a regulariser**, not as a source of truth. Within one run, the same swimmer's limb lengths
should stay roughly constant, and keypoints should not jump. A point that jumps or flips sides between neighbouring
frames is probably wrong.

The constraint must be confidence-weighted, so that a wrist leaving the water can disappear honestly. PoseWarper
showed that sparse labels plus temporal feature warping work in normal video [V Bertasius 2019]. Underwater warping is
untested [A].

### B. Can the two models supervise each other? [R]
Yes, if they are compared **point by point**, not averaged. For each keypoint, record:
1. both confidences;
2. the distance between the two predictions, normalised by box size;
3. each model's flip consistency;
4. each model's temporal stability.

From these signals, three cases follow:
- **Agree:** a point where both models are confident and agree within a small tolerance is a candidate label.
- **Disagree:** where both are confident but disagree, at least one is wrong. Mark the point unknown, and record it as
  a teacher-failure case.
- **Absent:** a point missing in both models is "unknown", never "absent".

SSPCM uses the same idea with two teachers' position inconsistency [V Huang 2023]. Its authors note it can still keep
confidently wrong labels that happen to agree.

### C. Can a model learn without any human labels? [R]
It can learn to be **consistent**:
- with flips;
- with time;
- with an independent model.

That narrows the possible errors but does not identify anatomy by itself. The anatomical meaning comes from the COCO
pretraining, plus any human labels. Self-training on its own predictions mostly reinforces what the model already
believes. Claims of improvement therefore need a trusted check, either the active-learning frames (Section 4) or
lifeguard review of overlays.

### D. Avoiding confirmation bias [R]
1. **Independent teachers:** never train a model on its own raw predictions in the first round.
2. **Ignore, don't negate:** unknown points must be excluded from both the position and the confidence loss. This is
   what went wrong last time.
3. **Several filters at once:** a point must pass agreement, flip and temporal checks.
4. **Fixed held-out data:** all frames in the four validation blocks, including their unlabelled neighbours in
   `datasets (1)`, never enter training.
5. **A fixed diagnostic suite (E0)** is run before and after every experiment. Stop if lower-body confidence or
   coverage drops.
6. **A slow-moving teacher:** an EMA teacher (Mean Teacher) instead of the current student [V Tarvainen 2017], and
   easy-to-hard augmentation pairs [V Xie 2021].
7. **Refresh labels late:** follow the coarse-to-fine pseudo-label refresh used for synthetic-to-real pose adaptation
   [V Li & Lee 2021].

## 3. Dataset readiness

| Question | Answer |
| --- | --- |
| Enough unlabelled frames for filtering and pseudo-labels? | Yes: about 5,600 frames [V]. Diversity is low: one or two swimmers, a few clips, one underwater camera [V visual]. |
| Temporal sequences? | Yes: 22 runs of >= 8 near-consecutive frames [V]. The true frame rate and the gaps between runs are unknown [A]. |
| Person boxes for top-down models? | Yes, one box per frame from the source dataset [V]. |
| Leak-free evaluation split? | Possible: exclude every frame whose index falls in validation blocks 2, 13, 15 and 16 (block size 3,000) from all training [R]. |
| Trusted accuracy labels? | **No.** None exist, so no experiment can currently prove better accuracy [V]. |
| Other camera views (overhead, drone)? | Not in this pool. The ddt and Z5cc clips are detection data at low resolution [V]. Results will not transfer automatically [A]. |

## 4. Experiment sequence

Every experiment compares against the **pretrained YOLO11s-pose**. Agreement and consistency numbers are reported
separately from accuracy. Accuracy is reported only where human labels exist.

| | E0 Diagnostics | E1 Filtered pseudo-labels | E2 Consistency training | E3 Active learning |
| --- | --- | --- | --- | --- |
| Hypothesis | The two models fail in different, measurable places | Training only on points that pass every filter, with unknowns ignored, improves the student without the lower-body collapse | Flip and temporal consistency on unlabelled runs improves stability beyond E1 | 20–30 hand-labelled hard frames give a trusted accuracy measure |
| Data | All 5,613 frames plus flipped copies | E0 outputs; training frames outside the validation blocks | E1 labels plus unlabelled runs | The frames E0 ranks as most disputed |
| Model change | None (inference only) | YOLO11s-pose with a custom loss that has an "ignore" state; lower learning rate | EMA teacher; custom training loop | None |
| Loss | — | YOLO pose loss; position and confidence loss masked for unknown points | E1 loss, plus consistency loss to the EMA teacher on hard-augmented views, plus a confidence-weighted temporal smoothness term | — |
| Cost (estimate) [A] | 15–30 min T4; under 300 MB | 15–30 min T4 per run; under 2 GB | 1–3 h T4; custom code, about 2–4 days to write | About 1 h of human work |
| Evaluation | Per-keypoint agreement, flip error, temporal jitter, limb-length stability, confidence vs agreement, coverage | E0 suite on the held-out blocks; agreement metrics; visual overlays; E3 frames if available | Same as E1 | PCK against human labels for every model so far |
| Fails if | The models agree everywhere (no new information) or almost nowhere (coverage too low to train) | Held-out diagnostics not better than pretrained, lower-body confidence drops, or E3 PCK not better | Collapse (predictions shrink to background), or the E1 failure conditions | — |
| Manual labels? | No | No (yes for a trusted verdict) | No (yes for a trusted verdict) | Yes, few |

E1 should use pseudo-labels from the two *independent* models only. Labelling the 5,251 frames outside the 362
increases volume but not diversity, so its benefit is uncertain [A].

## 5. Risks and limitations

- **Consistent but wrong:** all label-free signals can be satisfied by systematic errors (left/right swaps, arms
  mistaken for legs on inverted swimmers). [V seen on `frame_68013.jpg`]
- **Low diversity:** gains may only reflect memorising a few clips and one camera.
- **Water physics:** reflections at the surface create mirrored limbs [A]. Joints really do disappear and reappear at
  the waterline, which can look like instability.
- **Tooling:** E1 and E2 need a custom Ultralytics loss or trainer. The internals differ between 8.3.x and 8.4.x
  [A], so pin one version.
- **Compute:** Colab has given a T4 both times. Plan for a T4; an A100 only shortens runs [V].
- **Evaluation gap:** without E3, no claim of better accuracy is possible, only better consistency or agreement.

## 6. GPU and storage estimates [A]

| Item | GPU time (T4) | Drive storage |
| --- | --- | --- |
| E0: 5,613 frames x 2 models x 2 (flip) | ~15–30 min | < 300 MB JSON/CSV |
| E1: label build plus 1–3 training runs | ~20–60 min | < 2 GB (dataset copy plus checkpoints) |
| E2: consistency training | ~1–3 h | < 3 GB |

The current Drive use (models 1.25 GB, zips 1.7 GB, runs < 100 MB) stays well within limits.

## 7. Recommendation

1. **Next: E0 diagnostics**, inference only, with no training and no labels. It produces the teacher-failure map and
   shows how many keypoints per frame pass the filters. It is the input E1 needs. If coverage is too low, E1 should
   not be run.
2. **Then E1**, only if E0 shows enough agreeing points, especially on hips, knees and ankles. Use the custom
   ignore-aware loss and keep the validation blocks out entirely.
3. **E3 in parallel** if anyone can spare about an hour. It converts every later result from "more consistent" into
   "more accurate".
4. **E2 only** if E1 is clearly positive.

Until then the pretrained YOLO11s-pose stays the pose baseline, and no experimental checkpoint enters the Lifeguard
system.

## 8. E0 results (run 2026-11-27, Colab T4) [V notebook `pose_e0_diagnostics.ipynb`]

Agreement and consistency only, not accuracy. 5,613 frames (620 held-out, 4,993 training-eligible); 117 ms per frame
for both models plus flips; 11 min in total. Files: `MyDrive/pose_baseline/e0/`.

| Finding | Number |
| --- | --- |
| Pretrained YOLO11s-pose found no person matching the dataset box (IoU >= 0.3) | 788 / 5,613 frames (14 %). Some may be box mismatch rather than a miss [A]. |
| ViTPose confidence vs agreement with YOLO | 0.5-0.7: 39 %; 0.7-0.9: 67 %; >= 0.9: **92 %**. Confidence is informative. |
| YOLO confidence vs agreement with ViTPose | 0.5-0.7: 41 %; 0.7-0.9: 40 %; >= 0.9: **70 %**, and YOLO gives >= 0.9 to 40,434 points. YOLO is overconfident relative to agreement. |
| Flip consistency (median, fraction of box) | ViTPose 0.020-0.028, YOLO 0.014-0.032. Both are self-consistent, so flip is a weak filter here. |
| Cross-model agreement within 0.10 | hips/knees/ankles 72-81 %, shoulders/wrists 64-67 %, elbows 55 %, face 64-75 % |
| Jitter (median second difference) | ViTPose 0.031-0.039, YOLO 0.033-0.053 (arms smoother with ViTPose) |
| Consensus keypoints per training-eligible frame | median 4, mean 5.9; coverage upper body 0.36 (face 0.08-0.18), lower body 0.47 |
| E1 rule fixed before the run (median >= 8 AND lower body >= 0.30) | **Not met** (the median fails; the lower-body part passes) |
| Most disputed held-out frames | mostly head-down/inverted descent (07996, 08024, 08064, 08150, 08151, 08054) and heavy splash (50192, 50194, 50232) |

Interpretation [R]:
1. The models disagree most on **inverted, head-down bodies and splash**, the orientations most relevant to a
   lifeguard system. In several head-down frames, ViTPose appears to force an upright reading, drawing legs where the
   hands are. This is a visual impression, not verified.
2. E1 as pre-registered is not feasible. Lowering the threshold now would be a post-hoc change and must be labelled
   as such.
3. The 30 most disputed held-out frames are the best possible E3 set: labelling them answers which model is right
   exactly where they disagree.

## 9. E3 results (2026-11-27) - agreement with AI reference labels, NOT accuracy

Reference: `artifacts/pose_e3/pose_gt_e3.json`, provenance `ai_visual_review_unverified`. These are keypoints placed
by the AI assistant (Claude) from visual inspection of 14 of the most disputed held-out frames. They are not derived
from either model's output, but the assistant had seen the E0 overlay of both models on 12 of these frames
beforehand. 130 keypoints were labelled (55 v=2, 75 v=1); the face was never labelled. Left/right comes from the
swimmer's watch (right wrist). Estimated placement error is about 5-15 px, so PCK@0.05 is mostly reference noise.
Person boxes come from the dataset manifest.

| Model | PCK@0.05 upper/all | PCK@0.1 upper/all | PCK@0.2 upper/all | Latency (T4) |
| --- | --- | --- | --- | --- |
| ViTPose-L SwimXYZ | 0.514 / 0.369 | 0.703 / 0.615 | **0.730 / 0.692** | 36.0 ms |
| YOLO11s-pose, pretrained | 0.068 / 0.054 | 0.176 / 0.146 | 0.243 / 0.315 | 12.3 ms |
| YOLO11s-pose, distilled | 0.176 / 0.123 | 0.459 / 0.354 | 0.703 / 0.592 | 25.3 ms |

Interpretation [R]:
1. On these hardest frames (head-down descent, splash), the reference agrees far more with ViTPose than with the
   pretrained YOLO. This is the opposite of the visual impression noted in section 8.
2. The distilled YOLO more than doubles the pretrained YOLO's PCK@0.2 on these frames. This revises the earlier
   conclusion that distillation showed no clear benefit, which was based on ViTPose agreement over 27 easier frames.
3. Selection bias: the frames were chosen because the two models disagreed strongly, so a large gap between them was
   guaranteed. The size of the gap is not representative of typical frames.
4. Part of the YOLO gap may be left/right swaps rather than wrong locations; a swap-tolerant PCK would separate the two.

**Swap-tolerant PCK** (one left/right decision per frame, chosen by PCK@0.2; same reference):

| Model | Frames better with L/R swap | PCK@0.05 swaptol upper/all | PCK@0.1 swaptol upper/all | PCK@0.2 swaptol upper/all |
| --- | --- | --- | --- | --- |
| ViTPose-L SwimXYZ | 4 / 14 | 0.554 / 0.392 | **0.770 / 0.654** | 0.851 / 0.762 |
| YOLO11s-pose, pretrained | **10 / 14** | 0.108 / 0.185 | 0.311 / 0.400 | 0.541 / 0.608 |
| YOLO11s-pose, distilled | 4 / 14 | 0.216 / 0.146 | 0.581 / 0.469 | **0.919 / 0.785** |

5. About half of the pretrained YOLO's gap is **side assignment**: on 10 of 14 hard frames it labels left and right
   the opposite way to the reference (the watch rule). Its PCK@0.2 rises from 0.315 to 0.608 when sides are ignored.
   The remaining gap is localisation.
6. "Opposite to the reference" assumes the watch rule is right. On back and inverted views, left/right is genuinely
   hard to decide, for people and for models.
7. When sides are ignored, the distilled YOLO matches or beats ViTPose at the coarse tolerance (0.785 against 0.762)
   at two-thirds of the latency. ViTPose stays clearly better at the finer tolerance (PCK@0.1 0.654 against 0.469).
8. [R] Phase 7 pose features should be **left/right-invariant** (verticality, head height, symmetric arm measures,
   e.g. the maximum over both wrists). Then side errors stop mattering, and swap-tolerant PCK becomes the relevant
   metric.
9. Choosing the better side per frame is slightly optimistic, for every model alike.

## 10. E1 v1 results (2026-11-27) - failed, cause found [V notebook `pose_e1_train.ipynb`]

Setup: YOLO11s-pose from COCO weights, 40 epochs at 960 px on a T4 (1.0 h), 2,861 train / 762 val frames of filtered
ViTPose teacher labels (conf >= 0.7, flip <= 0.05, jump <= 0.25), ignore-aware loss (self-test PASS).
Run: `runs_pose/yolo11s_pose_e1_ignore_960` (kept for reference, not used further).

| held-out set (agreement, not accuracy)  | pretrained | distilled | E1 v1 |
|------------------------------------------|-----------:|----------:|------:|
| E3-14 PCK@0.2 all / swap-tolerant        | 0.315 / 0.608 | 0.592 / 0.785 | 0.454 / 0.562 |
| E3-14 PCK@0.1 all                        | 0.146 | 0.354 | 0.269 |
| E3-14 frames with >= 3 confident upper   | 1.00 | 1.00 | 0.29 |
| Test-27 PCK@0.2 all (vs ViTPose teacher) | 0.376 | 0.519 | 0.451 |
| 76 held-out frames, lower-body mean conf | 0.72 | 0.51 | 0.24 |

Both pre-set criteria failed: E1 v1 is worse than the distilled model and its confidences collapsed further.

Cause [V code + local test]: unknown points were written as `(0, 0, v=1)`. Ultralytics' `RandomPerspective.apply_keypoints`
sets `v=0` for every point outside the image after mosaic/affine, and `(0, 0)` usually lands outside. So most
"unknown" points reached the patched loss as "absent" (confidence target 0), the same failure the patch was meant to
remove, made worse by the stricter filter (more unknowns). On a synthetic check (`check_aug_visibility.py`), the
unknown/known ratio fell from 0.55 in the label files to 0.32 after augmentation; with unknown points placed at the box
centre it stayed at 0.53.

Fix (E1b): `build_e1_dataset.py` writes unknown points at the box centre; the notebook runs `check_aug_visibility.py`
(must print PASS) before training; run name `yolo11s_pose_e1b_centre_960`. Same success criteria as E1.

### E1b results (2026-11-27, run `runs_pose/yolo11s_pose_e1b_centre_960`, 40 epochs, 1.0 h T4)

Pre-training checks: augmentation check PASS (unknown/known ratio 0.73 before and after), loss self-test PASS.

| held-out set (agreement, not accuracy)  | pretrained | distilled | E1b |
|------------------------------------------|-----------:|----------:|----:|
| E3-14 PCK@0.2 all / swap-tolerant        | 0.315 / 0.608 | 0.592 / **0.785** | **0.623** / 0.669 |
| E3-14 PCK@0.1 all / swap-tolerant        | 0.146 / 0.400 | **0.354 / 0.469** | 0.292 / 0.346 |
| E3-14 PCK@0.2 upper swap-tolerant        | 0.541 | **0.919** | 0.649 |
| E3-14 frames better with L/R swap        | 10 | 4 | **1** |
| Test-27 PCK@0.2 all (vs ViTPose teacher) | 0.376 | **0.519** | 0.473 |
| 76 held-out frames, lower-body mean conf | 0.72 | 0.51 | 0.87 |

Verdict: **E1b fails the pre-set criterion** (beat the distilled model on E3, especially PCK@0.1 and swap-tolerant).
Only PCK@0.2 all is slightly higher (+0.03 on 14 frames, within noise); PCK@0.1, swap-tolerant and Test-27 are lower.
The confidence collapse is gone, but it overshot: training kobj loss fell to ~0.002 because almost no keypoint is ever
labelled "absent", so the model now reports high confidence for everything (nose 0.79 and ears 0.73 although faces are
rarely visible in this footage; validation kobj loss rose from 1.1 to 1.6). Confidence that is always high cannot gate
pose features, so this is not an improvement for the system either. One positive signal: fewer left/right swaps.

Decision (as agreed before the run): **stop pose training here.** The distilled model
(`runs_pose/yolo11s_pose_distill_960`) stays the best experimental student. Nothing is integrated; pose risk weight
stays 0. A future attempt would need real human-labelled validation data and a confidence target for unknown points
(e.g. a soft target), not another unsupervised retrain.

### E2 design (2026-11-27, requested by the user after E1b; notebook `pose_e2_train.ipynb`, A100)

One change against E1b: the confidence of every point is trained towards the teacher's own confidence (unknown train
points carry `v = 1 + c`; `soft_conf_pose_loss.py`), while positions still use only filtered confident points.
Verified locally on Ultralytics 8.3.40 before the run: loss self-test PASS, augmentation check PASS (soft values
preserved), 2-epoch smoke train with non-zero pose loss. Found during that test: 8.3.x divides the visibility flag by
the stride, so a `v == 2` mask never matches there; both patches now keep the flag out of that division (the E1 runs
used 8.4.220, where the pose loss was normal, so they are not affected). New check `conf_calibration.py`: AUROC of
confidence for separating hits from misses. Pre-registered success rule: in the notebook's last cell.

### E2 results (2026-11-27, run `runs_pose/yolo11s_pose_e2_softconf_960`, 40 epochs, 0.29 h on A100)

Pre-training checks: augmentation check PASS (unknown/known 0.73 -> 0.70), loss self-test PASS. Agreement, not accuracy.

| metric (held-out)                          | pretrained | distilled | E1b | E2 |
|---------------------------------------------|-----------:|----------:|----:|---:|
| E3-14 PCK@0.1 all                           | 0.146 | **0.354** | 0.292 | 0.269 |
| E3-14 PCK@0.2 all swap-tolerant             | 0.608 | **0.785** | 0.669 | 0.615 |
| Test-27 PCK@0.2 all                         | 0.376 | **0.519** | 0.473 | 0.447 |
| Lower-body mean confidence (76 frames)      | 0.72 | 0.51 | 0.87 | 0.47 |
| Confidence AUROC, E3 / Test-27 (0.5 = none) | 0.750 / 0.811 | 0.686 / 0.774 | 0.677 / 0.758 | 0.637 / 0.795 |

Verdict: **E2 fails the pre-registered rule** (criteria 1, 2 and the lower-body/E3-AUROC parts of 3 are not met; only
Test-27 AUROC beats the distilled model). The confidence level is sane now (train kobj loss stays at 0.36, faces low:
nose 0.37, ears 0.30), so the E1b overshoot is fixed, but positions are worse than the distilled model and confidence
is not more informative. The unmodified COCO-pretrained model has the best confidence AUROC on both sets, even though
its positions are poor.

Decision: **pose training stops.** Three unsupervised retrains (E1, E1b, E2) did not beat the distilled model
(`yolo11s_pose_distill_960`), which stays the best experimental student. Not integrated; pose risk weight stays 0.
The remaining route is human-labelled validation data and a shadow-mode test, not more training on teacher labels.

### Independent reference check (2026-11-27): 20 test frames, AI visual review (not ViTPose, not human)

Reference `artifacts/pose_human_ref/pose_gt_t27_ai_visual.json`: 201 points on 20 of the 27 test frames, estimated by an AI
from zoomed gridded crops (provenance `ai_visual_review_unverified`; 7 frames too blurred; most points v=1 inferred; left/right
uncertain on several frames). Agreement, **not accuracy, not human ground truth**. Notebook `pose_t27_ai_visual_eval.ipynb`.

| metric (20 frames, 201 points)          | pretrained | distilled | E1b | E2 |
|------------------------------------------|-----------:|----------:|----:|---:|
| PCK@0.1 all / swap-tolerant              | 0.423 / **0.522** | 0.328 / 0.453 | **0.443** / 0.498 | 0.428 / 0.483 |
| PCK@0.2 all / swap-tolerant              | 0.632 / 0.726 | 0.632 / **0.771** | **0.701** / 0.751 | 0.662 / 0.711 |
| PCK@0.2 upper swap-tolerant              | 0.700 | **0.820** | 0.740 | 0.720 |
| frames with no detection                 | 2 | 0 | 0 | 0 |
| confidence AUROC (0.5 = no information)  | **0.770** | 0.670 | 0.694 | 0.677 |

Reading: the four models are within a few points of each other (n = 20 frames, so differences under about 0.1 are noise).
The distilled model's lead on the teacher-based sets (E3, Test-27) largely disappears against an independent reference: it
is best only on PCK@0.2 swap-tolerant, worst on PCK@0.1, and its confidence is the least informative of the four except
E2. The unmodified COCO-pretrained model is competitive on position and has the best confidence AUROC (0.77), as it did in
E2's checks (0.75 / 0.81). Teacher-distillation therefore has no demonstrated benefit over the pretrained model on this
evidence; part of the earlier advantage was copying the teacher. This revises the E1b/E2 conclusion that the distilled
model is "the best student": no model is shown to be better than plain YOLO11s-pose here.
Absolute level: about half of the points are within 0.1 of the box size (upper body swap-tolerant 0.51-0.52), roughly
70-82% within 0.2: coarse posture only. Pose risk weight stays 0; nothing is integrated.

### Engineering decision (2026-11-27): pretrained YOLO11s-pose integrated, display only

Following the independent-reference check above, the project integrated the official pretrained YOLO11s-pose (not
the distilled, E1, E1b or E2 checkpoints, and not ViTPose) for display and recording only; pose risk weight stays 0.
Design, configuration, GTX 1650 measurements and tests: [`POSE_INTEGRATION.md`](POSE_INTEGRATION.md). The experimental
results above are unchanged.

## 11. References (verified on the cited pages, 2026-11-27)

- Xie, Wang, Zeng, Wang. *An Empirical Study of the Collapsing Problem in Semi-Supervised 2D Human Pose Estimation.*
  ICCV 2021. <https://arxiv.org/abs/2011.12498> (collapse of consistency methods; easy/hard augmentation pairs).
- Bertasius, Feichtenhofer, Tran, Shi, Torresani. *Learning Temporal Pose Estimation from Sparsely-Labeled Videos*
  (PoseWarper). NeurIPS 2019. <https://arxiv.org/abs/1906.04016>
- Li, Lee. *ScarceNet: Animal Pose Estimation with Scarce Annotations.* CVPR 2023. <https://arxiv.org/abs/2303.15023>
  (small-loss pseudo-label selection, agreement check, student-teacher consistency).
- Li, Lee. *From Synthetic to Real: Unsupervised Domain Adaptation for Animal Pose Estimation.* CVPR 2021.
  <https://arxiv.org/abs/2103.14843> (coarse-to-fine pseudo-label refresh, self-distillation, mean teacher).
- Moskvyak, Maire, Dayoub, Baktashmotlagh. *Semi-supervised Keypoint Localization.* ICLR 2021.
  <https://arxiv.org/abs/2101.07988>
- Huang et al. *Semi-Supervised 2D Human Pose Estimation Driven by Position Inconsistency Pseudo Label Correction
  Module* (SSPCM). CVPR 2023. <https://arxiv.org/abs/2303.04346>
- Liu, Ferrari. *Active Learning for Human Pose Estimation.* ICCV 2017.
  <https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Active_Learning_for_ICCV_2017_paper.pdf>
- Tarvainen, Valpola. *Mean teachers are better role models.* NeurIPS 2017. <https://arxiv.org/abs/1703.01780>
- Fiche et al. *SwimXYZ: A large-scale dataset of synthetic swimming motions and videos.* MIG 2023.
  <https://arxiv.org/abs/2310.04360> (states no public real swimming pose dataset existed).
- Model card, ViTPose-L fine-tuned on SwimXYZ: <https://huggingface.co/willalexm/vitpose-l-swimxyz>
