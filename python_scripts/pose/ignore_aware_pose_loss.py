"""Patch Ultralytics pose training so that UNKNOWN keypoints are ignored instead of trained as absent.

Why: in stock Ultralytics, `kpts_obj_loss = BCE(pred_conf, v != 0)` covers all 17 keypoints, so a keypoint written as
v=0 teaches the model "this joint is not there". Pseudo-labels have many uncertain points, and that taught the
distilled model to drop hips/knees/ankles (E0/E3 analysis, docs/POSE_SELF_SUPERVISED_RESEARCH.md).

Label convention while the patch is active (TRAINING labels only):
  v = 2  confident teacher point -> position loss + confidence target 1
  v = 1  unknown                 -> no loss at all
  v = 0  absent (e.g. pushed outside the image by augmentation) -> confidence target 0
Validation labels must use only 0 / 2 (Ultralytics' OKS metric treats v > 0 as labelled).

The patch rewrites two statements of `v8PoseLoss.calculate_keypoints_loss` and refuses to run if they are not found
exactly once. Known to match Ultralytics 8.3.40 (one-line mask) and 8.4.220 (mask inside an if-block).
"""
from __future__ import annotations

import inspect
import textwrap

OLD_MASK = "kpt_mask = gt_kpt[..., 2] != 0"
NEW_MASK = "kpt_mask = gt_kpt[..., 2] == 2"
OLD_OBJ = "kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())"
NEW_OBJ = ("kpts_obj_loss = (lambda known: torch.nn.functional.binary_cross_entropy_with_logits("
           "pred_kpt[..., 2], kpt_mask.float(), reduction='none')[known].mean() if known.any() "
           "else pred_kpt[..., 2].sum() * 0.0)(gt_kpt[..., 2] != 1 if gt_kpt.shape[-1] == 3 "
           "else torch.ones_like(kpt_mask))")
POS_LINE = "kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)"  # unchanged; now uses v == 2
# 8.3.x divides the whole keypoint tensor by the stride (visibility too: v=2 -> 0.25); keep v intact
OLD_STRIDE = "selected_keypoints /= stride_tensor.view(1, -1, 1, 1)"
NEW_STRIDE = "selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)"


def apply() -> str:
    import ultralytics
    from ultralytics.utils import loss as L

    cls = L.v8PoseLoss
    if getattr(cls, "_ignore_aware", False):
        return "already patched"
    src = textwrap.dedent(inspect.getsource(cls.calculate_keypoints_loss))
    counts = {s: src.count(s) for s in (OLD_MASK, OLD_OBJ, POS_LINE)}
    if any(n != 1 for n in counts.values()):
        raise RuntimeError(f"Ultralytics {ultralytics.__version__}: loss code differs, patch not applied. "
                           f"Occurrences (each must be 1): {counts}")
    src = src.replace(OLD_MASK, NEW_MASK).replace(OLD_OBJ, NEW_OBJ).replace(OLD_STRIDE, NEW_STRIDE)
    ns: dict = {}
    exec(compile(src, "<ignore_aware_pose_loss>", "exec"), L.__dict__, ns)  # noqa: S102 - controlled source
    cls.calculate_keypoints_loss = ns["calculate_keypoints_loss"]
    cls._ignore_aware = True
    return f"patched v8PoseLoss.calculate_keypoints_loss on Ultralytics {ultralytics.__version__} (v=1 ignored)"


if __name__ == "__main__":
    print(apply())
