"""E2: Ultralytics pose loss with SOFT keypoint-confidence targets taken from the teacher.

Why (E1 history, docs/POSE_SELF_SUPERVISED_RESEARCH.md section 10):
  stock / distilled: unknown points trained as "absent" (target 0)  -> confidence collapse on hips/knees/ankles
  E1b (ignore-aware): unknown points get no confidence loss at all  -> confidence always high, carries no information
E2 trains the confidence of EVERY point towards the teacher's own confidence, while the position loss still uses only
the filtered, confident teacher points.

Label convention while the patch is active (TRAINING labels only):
  v = 2             confident teacher point   -> position loss + confidence target 1
  1 <= v < 2        unknown point, teacher confidence c = v - 1 -> no position loss, confidence target c
  v = 0             absent (pushed outside the image by augmentation) -> confidence target 0
Validation labels must use only 0 / 2 (Ultralytics' OKS metric treats v > 0 as labelled).

The patch rewrites two statements of `v8PoseLoss.calculate_keypoints_loss` and refuses to run if they are not found
exactly once (and keeps the visibility flag out of the stride division, which 8.3.x applies to it).
Written against Ultralytics 8.3.40 (one-line mask) and 8.4.220 (mask inside an if-block).
"""
from __future__ import annotations

import inspect
import textwrap

OLD_MASK = "kpt_mask = gt_kpt[..., 2] != 0"
NEW_MASK = "kpt_mask = gt_kpt[..., 2] > 1.995"
OLD_OBJ = "kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())"
NEW_OBJ = ("kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], (torch.where(gt_kpt[..., 2] > 1.995, 1.0, "
           "torch.where(gt_kpt[..., 2] >= 1.0, gt_kpt[..., 2] - 1.0, 0.0)) if gt_kpt.shape[-1] == 3 "
           "else kpt_mask.float()))")
POS_LINE = "kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)"  # unchanged; now uses v == 2
# 8.3.x divides the whole keypoint tensor by the stride (visibility too: v=2 -> 0.25); keep v intact
OLD_STRIDE = "selected_keypoints /= stride_tensor.view(1, -1, 1, 1)"
NEW_STRIDE = "selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)"


def soft_target(v):
    """Reference implementation of the confidence target (used by the self-test)."""
    import torch

    return torch.where(v > 1.995, 1.0, torch.where(v >= 1.0, v - 1.0, 0.0))


def apply() -> str:
    import ultralytics
    from ultralytics.utils import loss as L

    cls = L.v8PoseLoss
    if getattr(cls, "_soft_conf", False):
        return "already patched (soft confidence)"
    if getattr(cls, "_ignore_aware", False):
        raise RuntimeError("the E1 ignore-aware patch is active in this process: restart the runtime first")
    src = textwrap.dedent(inspect.getsource(cls.calculate_keypoints_loss))
    counts = {s: src.count(s) for s in (OLD_MASK, OLD_OBJ, POS_LINE)}
    if any(n != 1 for n in counts.values()):
        raise RuntimeError(f"Ultralytics {ultralytics.__version__}: loss code differs, patch not applied. "
                           f"Occurrences (each must be 1): {counts}")
    if src.count(OLD_STRIDE) > 1:
        raise RuntimeError("stride division found more than once, patch not applied")
    src = src.replace(OLD_MASK, NEW_MASK).replace(OLD_OBJ, NEW_OBJ).replace(OLD_STRIDE, NEW_STRIDE)
    ns: dict = {}
    exec(compile(src, "<soft_conf_pose_loss>", "exec"), L.__dict__, ns)  # noqa: S102 - controlled source
    cls.calculate_keypoints_loss = ns["calculate_keypoints_loss"]
    cls._soft_conf = True
    return f"patched v8PoseLoss.calculate_keypoints_loss on Ultralytics {ultralytics.__version__} (soft confidence)"


def self_test() -> str:
    """Checks the patched loss on a tiny synthetic case (works on 8.3.x and 8.4.x signatures)."""
    import inspect as _inspect

    import torch
    from ultralytics.utils.loss import KeypointLoss, v8PoseLoss
    from ultralytics.utils.metrics import OKS_SIGMA

    assert getattr(v8PoseLoss, "_soft_conf", False), "run apply() first"
    params = list(_inspect.signature(v8PoseLoss.calculate_keypoints_loss).parameters)[1:]
    L = object.__new__(v8PoseLoss)
    L.keypoint_loss = KeypointLoss(sigmas=torch.from_numpy(OKS_SIGMA))
    L.bce_pose = torch.nn.BCEWithLogitsLoss()
    torch.manual_seed(0)
    kp = torch.rand(1, 17, 3) * 50
    kp[0, :, 2] = torch.tensor([2.0] * 6 + [1.3, 1.6, 1.9, 1.0, 1.5, 1.8] + [0.0] * 5)
    tgt = soft_target(kp[0, :, 2]).clamp(1e-4, 1 - 1e-4)
    pred = torch.rand(1, 4, 17, 3) * 50
    pred[..., 2] = torch.log(tgt / (1 - tgt))  # confidence logits exactly at the target

    def run(pr):
        gt = kp.clone() if "batch_idx" in params else kp.clone().unsqueeze(0)  # 8.4: (images, people, 17, 3)
        pool = {"masks": torch.tensor([[True, True, False, False]]), "target_gt_idx": torch.zeros(1, 4, dtype=torch.long),
                "keypoints": gt, "batch_idx": torch.zeros(1, 1), "stride_tensor": torch.full((4, 1), 8.0),
                "target_bboxes": torch.tensor([[[0, 0, 60, 60.0]] * 4]), "pred_kpts": pr.clone()}
        missing = [q for q in params if q not in pool]
        assert not missing, f"unexpected parameter names: {missing}"
        return [float(v) for v in L.calculate_keypoints_loss(*[pool[q] for q in params])]

    base = run(pred)
    def moved(idx, scale):  # put these points exactly on their targets (pixel or stride units, version-dependent)
        q = pred.clone()
        q[..., idx, :2] = kp[0, idx, :2] / scale
        return q

    unk, known = list(range(6, 12)), list(range(0, 6))
    up = pred.clone(); up[..., 6:12, 2] += 0.5    # unknown confidence above the teacher's
    dn = pred.clone(); dn[..., 6:12, 2] -= 0.5    # and below
    assert base[0] > 0, "position loss is zero (visibility flags corrupted, e.g. divided by the stride)"
    assert all(run(moved(unk, sc))[0] == base[0] for sc in (1.0, 8.0)), "unknown points still get a position loss"
    assert min(run(moved(known, sc))[0] for sc in (1.0, 8.0)) < base[0], "confident points no longer get a position loss"
    assert run(up)[1] > base[1] and run(dn)[1] > base[1], "unknown confidence is not pulled to the teacher value"
    return "PASS: position loss only on v=2; confidence of unknown points trained towards the teacher confidence"


if __name__ == "__main__":
    print(apply())
    print(self_test())
