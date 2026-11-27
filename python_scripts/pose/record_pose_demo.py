"""Re-run demo videos through the full LifeguardRuntime (the dashboard's back end) and record the results.

For each video: frames, errors, tracks, risk events (PRE_ALERT / ALARM), latency/FPS, pose coverage, and the same run
with pose disabled for comparison. Saves dashboard-identical overlays (``dashboard_controller.annotate``) of the frames
with the most confident joints. Nothing is changed in detection, tracking or risk; videos are only read.

    python python_scripts/pose/record_pose_demo.py --videos E:/lifeguardDron/demo/demo_06_drown_T5mDQeDkca0.mp4 \
        --out artifacts/pose/demo_results
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2

from lifeguard.ai.config import load_ai_config
from lifeguard.app.dashboard_controller import POSE_DRAW_CONF, annotate
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.system.config import load_system_config
from lifeguard.system.ingest.video import VideoFileSource

ROOT = Path(__file__).resolve().parents[2]


def run(video: Path, ai_path: Path, sys_path: Path, pose: bool, out_dir: Path | None, n_shots: int) -> dict:
    ai = load_ai_config(ai_path)
    ai.pose.enabled = pose
    sys_cfg = load_system_config(sys_path)
    source = VideoFileSource(video)
    rt = LifeguardRuntime(ai, sys_cfg, (source.width, source.height), session_id="demo")
    events, lat, wall, candidates = [], [], [], []
    tracks, visible, with_pose = set(), 0, 0
    t0 = time.perf_counter()
    for frame in source:
        if frame is None:
            continue
        start = time.perf_counter()
        for e in rt.process(frame):
            events.append({"t_media_s": round(e.t_media, 2) if e.t_media is not None else None,
                           "track": e.track_uid.rsplit(":", 1)[-1], "from": e.from_level.value,
                           "to": e.to_level.value, "reason": e.reason.value, "risk": round(e.risk, 3)})
        wall.append((time.perf_counter() - start) * 1000.0)
        snap = rt.snapshot()
        res = snap.result
        if res is None:
            continue
        lat.append(res.latency_ms)
        for t in res.tracks:
            tracks.add(t.track_id)
            if t.state.value == "tracked":
                visible += 1
                if t.keypoints is not None:
                    with_pose += 1
        if pose and out_dir is not None:
            score = sum(1 for t in res.tracks if t.keypoints for _, _, c in t.keypoints if c >= POSE_DRAW_CONF)
            if score:
                candidates.append((score, res.frame_idx, snap))
                candidates = sorted(candidates, key=lambda c: -c[0])[: n_shots * 6]
    total_s = time.perf_counter() - t0
    summary = rt.snapshot().summary
    stats = getattr(rt.engine, "pose_stats", None)
    shots = []
    if pose and out_dir is not None and candidates:
        out_dir.mkdir(parents=True, exist_ok=True)
        chosen: list[tuple] = []
        for cand in candidates:   # best frames, at least 30 frames apart
            if all(abs(cand[1] - c[1]) >= 30 for c in chosen):
                chosen.append(cand)
            if len(chosen) == n_shots:
                break
        for score, idx, snap in sorted(chosen, key=lambda c: c[1]):
            img = annotate(snap.frame, snap.result, snap.levels)
            path = out_dir / f"{video.stem}_frame{idx:05d}_pose.png"
            cv2.imwrite(str(path), img[:, :, ::-1])
            shots.append({"file": path.name, "frame": idx, "joints_drawn": score})
    rt.close()
    return {
        "video": video.name, "pose": pose, "frames": summary.frames, "ok": summary.ok,
        "degraded": summary.degraded, "errors": summary.error, "unique_tracks": len(tracks),
        "visible_track_obs": visible, "obs_with_pose": with_pose,
        "pose_errors": stats.errors if stats else None,
        "pose_ms_mean": round(stats.mean_ms, 1) if stats and stats.mean_ms else None,
        "latency_ms_mean": round(statistics.mean(lat[10:] or lat), 1) if lat else None,
        "latency_ms_p95": round(sorted(lat[10:] or lat)[int(0.95 * len(lat[10:] or lat)) - 1], 1) if lat else None,
        "processing_fps": round(1000.0 / statistics.mean(wall[10:] or wall), 1) if wall else None,
        "wall_s": round(total_s, 1), "video_fps": source.fps,
        "events": events, "pre_alert": any(e["to"] == "PRE_ALERT" for e in events),
        "alarm": any(e["to"] == "ALARM" for e in events), "screenshots": shots,
        "model_versions": rt.engine.heartbeat().detail if hasattr(rt.engine, "heartbeat") else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, nargs="+", required=True)
    ap.add_argument("--ai-config", type=Path, default=ROOT / "configs" / "ai_z5cc_pose.yaml")
    ap.add_argument("--system-config", type=Path, default=ROOT / "configs" / "system_demo_drone.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shots", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for video in a.videos:
        on = run(video, a.ai_config, a.system_config, True, a.out / "screenshots", a.shots)
        off = run(video, a.ai_config, a.system_config, False, None, 0)
        on["events_identical_to_pose_off"] = on["events"] == off["events"]
        on["pose_off"] = {k: off[k] for k in ("frames", "errors", "unique_tracks", "latency_ms_mean", "processing_fps")}
        (a.out / f"{video.stem}.json").write_text(json.dumps(on, indent=1), encoding="utf-8")
        rows.append(on)
        print(f"{video.name}: frames={on['frames']} errors={on['errors']} tracks={on['unique_tracks']} "
              f"pose={on['obs_with_pose']}/{on['visible_track_obs']} fps={on['processing_fps']} "
              f"(off {off['processing_fps']}) pre_alert={on['pre_alert']} alarm={on['alarm']} "
              f"same_events={on['events_identical_to_pose_off']}", flush=True)
    with open(a.out / "demo_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frames", "errors", "unique_tracks", "visible_track_obs", "obs_with_pose",
                    "latency_ms_mean", "latency_ms_p95", "processing_fps", "processing_fps_pose_off", "pose_ms_mean",
                    "pre_alert", "alarm", "events_identical_to_pose_off", "events"])
        for r in rows:
            w.writerow([r["video"], r["frames"], r["errors"], r["unique_tracks"], r["visible_track_obs"],
                        r["obs_with_pose"], r["latency_ms_mean"], r["latency_ms_p95"], r["processing_fps"],
                        r["pose_off"]["processing_fps"], r["pose_ms_mean"], r["pre_alert"], r["alarm"],
                        r["events_identical_to_pose_off"],
                        "; ".join(f"{e['t_media_s']}s #{e['track']} {e['from']}->{e['to']} ({e['reason']})"
                                  for e in r["events"])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
