"""Pose integration check and benchmark on a real video: the same frames through the real engine with pose OFF
and ON, each followed by the real risk engine.

Reports (per run): frames, mean/p95 engine latency, processing FPS (perception + risk, no display), pose
latency, GPU memory, tracked swimmers, tracks with pose, per-joint share of keypoints >= 0.5.
Verifies: identical tracks (ids, boxes, states, features) and identical risk events with pose ON vs OFF, i.e.
pose changes nothing except ``TrackObs.keypoints``.

    python python_scripts/pose/bench_pose_pipeline.py --ai-config configs/ai_z5cc_pose.yaml \
        --system-config configs/system.yaml --video E:/lifeguardDron/demo/demo_06_drown_T5mDQeDkca0.mp4
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from lifeguard.ai.config import load_ai_config
from lifeguard.ai.perception.factory import create_engine
from lifeguard.app.runtime import build_geometry
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.system.config import load_system_config
from lifeguard.system.risk.engine import RuleBasedRiskEngine

COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
          "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
T0 = "2026-01-01T00:00:00.000Z"


def gpu_used_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run(ai_cfg, sys_cfg, frames: list[np.ndarray], fps: float, pose: bool, warmup: int) -> dict:
    ai_cfg = ai_cfg.model_copy(deep=True)
    ai_cfg.pose.enabled = pose
    h, w = frames[0].shape[:2]
    geometry, _ = build_geometry(sys_cfg, w, h)
    engine = create_engine(ai_cfg, sys_cfg.cam_id, "bench", geometry=geometry)
    risk = RuleBasedRiskEngine(sys_cfg.risk)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    lat, results, events = [], [], []
    for i, frame in enumerate(frames):
        t = i / fps
        start = time.perf_counter()
        res = engine.process(frame, i, 1000.0 + t, add_seconds_utc(T0, t), t)
        ev = risk.update(res)
        if i >= warmup:
            lat.append((time.perf_counter() - start) * 1000.0)
        results.append(res)
        events += [(e.track_uid, e.from_level.value, e.to_level.value, e.reason.value, round(e.risk, 4),
                    round(e.t_capture, 3)) for e in ev]
        if i == warmup:   # pose timing excludes warm-up frames too
            engine.pose_stats.__init__()
    gpu_peak = None
    if torch is not None and torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated() / 2**20
    stats = engine.pose_stats
    tracks = [t for r in results for t in r.tracks]
    with_pose = [t for t in tracks if t.keypoints is not None]
    vis = np.zeros(17)
    for t in with_pose:
        vis += np.array([c >= 0.5 for _, _, c in t.keypoints], dtype=float)
    return {
        "pose": pose,
        "frames": len(frames),
        "errors": sum(r.status.value == "error" for r in results),
        "latency_ms_mean": round(statistics.mean(lat), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "processing_fps": round(1000.0 / statistics.mean(lat), 1),
        "pose_ms_mean": None if stats.mean_ms is None else round(stats.mean_ms, 1),
        "pose_errors": stats.errors,
        "gpu_torch_peak_mb": None if gpu_peak is None else round(gpu_peak, 0),
        "gpu_used_mb_after": gpu_used_mb(),
        "unique_tracks": len({t.track_id for t in tracks}),
        "max_tracks_in_frame": max((len(r.tracks) for r in results), default=0),
        "track_observations": len(tracks),
        "visible_observations": sum(t.state.value == "tracked" for t in tracks),
        "observations_with_pose": len(with_pose),
        "joint_share_ge_0.5": {COCO17[k]: round(vis[k] / max(len(with_pose), 1), 2) for k in range(17)},
        "model_versions": results[-1].model_versions,
        "_results": results,
        "_events": events,
    }


def strip(result) -> str:
    """FrameResult without keypoints and timing, for the OFF/ON comparison."""
    d = json.loads(result.model_dump_json())
    d.pop("latency_ms", None)
    d.get("model_versions", {}).pop("pose", None)
    for t in d.get("tracks", []):
        t["keypoints"] = None
    return json.dumps(d, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai-config", type=Path, required=True)
    ap.add_argument("--system-config", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, help="write the summary JSON here")
    a = ap.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < a.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {a.video}")
    ai_cfg = load_ai_config(a.ai_config)
    sys_cfg = load_system_config(a.system_config)

    off = run(ai_cfg, sys_cfg, frames, fps, pose=False, warmup=a.warmup)
    on = run(ai_cfg, sys_cfg, frames, fps, pose=True, warmup=a.warmup)
    same_tracks = all(strip(x) == strip(y) for x, y in zip(off["_results"], on["_results"]))
    same_risk = off["_events"] == on["_events"]
    summary = {
        "video": str(a.video), "frame_size": [frames[0].shape[1], frames[0].shape[0]], "video_fps": fps,
        "off": {k: v for k, v in off.items() if not k.startswith("_")},
        "on": {k: v for k, v in on.items() if not k.startswith("_")},
        "tracks_identical_off_vs_on": same_tracks,
        "risk_events_identical_off_vs_on": same_risk,
        "risk_events": off["_events"],
    }
    text = json.dumps(summary, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return 0 if same_tracks and same_risk else 1


if __name__ == "__main__":
    raise SystemExit(main())
