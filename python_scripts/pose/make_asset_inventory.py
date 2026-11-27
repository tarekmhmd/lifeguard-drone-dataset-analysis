"""Inventory of the pose package: every file under artifacts/pose/ plus the runtime weight.

Writes ``artifacts/pose/asset_inventory.csv`` (path, bytes, sha256, class, source, Git LFS needed) and regenerates the
per-file appendix of ``artifacts/pose/ASSET_MANIFEST.md`` between its INVENTORY markers, so the manifest always
matches the files actually present. Sources of Drive files come from ``drive_package/organize_log.csv``.

    python python_scripts/pose/make_asset_inventory.py
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "artifacts" / "pose"
RUNTIME = ROOT / "models" / "yolo11s-pose.pt"
MANIFEST = PKG / "ASSET_MANIFEST.md"
START, END = "<!-- INVENTORY_START -->", "<!-- INVENTORY_END -->"
LFS_LIMIT = 50 * 2**20   # GitHub warns above 50 MB and rejects above 100 MB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drive_sources() -> dict[str, str]:
    log = PKG / "drive_package" / "organize_log.csv"
    if not log.is_file():
        return {}
    with log.open(encoding="utf-8") as fh:
        return {r["repository_path"]: "Google Drive MyDrive/" + r["drive_path"] for r in csv.DictReader(fh)}


def classify(rel: str) -> str:
    if rel == "models/yolo11s-pose.pt":
        return "RUNTIME"
    sub = rel.split("/")[2] if rel.count("/") >= 3 else ""
    return {"weights": "EXPERIMENTAL", "training": "EXPERIMENTAL (run output)", "evaluation": "RESULT",
            "datasets_metadata": "METADATA", "references": "REFERENCE (AI, not human ground truth)",
            "demo_results": "RESULT (demo rerun)", "drive_package": "PROVENANCE"}.get(sub, "DOCUMENT")


def source(rel: str, drive: dict[str, str]) -> str:
    if rel in drive:
        return drive[rel]
    rules = [
        ("models/yolo11s-pose.pt", "Ultralytics GitHub release v8.3.0 (official)"),
        ("artifacts/pose/evaluation/integration_benchmarks", "bench_pose_pipeline.py, integration session 2026-11-27"),
        ("artifacts/pose/demo_results/dashboard_screenshots", "Streamlit dashboard capture, integration session 2026-11-27"),
        ("artifacts/pose/demo_results", "record_pose_demo.py rerun of E:/lifeguardDron/demo videos, 2026-11-27"),
        ("artifacts/pose/datasets_metadata/pose_annotation_set_v2", "pose_annotation_set_v2.zip (Drive), local extract"),
        ("artifacts/pose/datasets_metadata/e3_set", "e3_set.zip (Drive pose_baseline/e3), formerly repo-root e3_set/"),
        ("artifacts/pose/references/drive_ai_reference/proposals", "pose_annotation_set_v2.zip (Drive), local extract"),
        ("artifacts/pose/references/cvat/cvat_task", "CVAT (app.cvat.ai) export, downloaded 2026-11-27"),
        ("artifacts/pose/references/cvat/proposals", "make_cvat_proposals.py output imported into CVAT"),
        ("artifacts/pose/drive_package/organize_log", "generated while organizing the Drive package"),
    ]
    for prefix, text in rules:
        if rel.startswith(prefix):
            return text
    return "written in this repository (2026-11-27)"


def lfs(path: Path) -> str:
    if path.suffix in (".pt", ".onnx"):
        return "yes"
    return "yes (size)" if path.stat().st_size > LFS_LIMIT else "no"


def main() -> int:
    files = sorted(p for p in PKG.rglob("*") if p.is_file() and p.name != "asset_inventory.csv") + [RUNTIME]
    drive = drive_sources()
    rows = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        rows.append([rel, p.stat().st_size, sha256(p) if p != MANIFEST else "(this file)", classify(rel),
                     source(rel, drive), lfs(p)])
    with (PKG / "asset_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "sha256", "class", "source", "git_lfs_needed"])
        w.writerows(rows)
    if MANIFEST.is_file():
        text = MANIFEST.read_text(encoding="utf-8")
        if START in text and END in text:
            table = ["| Path | Bytes | sha256 | Class | Source | LFS |", "| --- | --- | --- | --- | --- | --- |"]
            table += [f"| `{r[0]}` | {r[1]:,} | `{r[2][:16]}…` | {r[3]} | {r[4]} | {r[5]} |"
                      if r[2] != "(this file)" else f"| `{r[0]}` | – | (this file) | {r[3]} | {r[4]} | no |"
                      for r in rows]
            head, rest = text.split(START, 1)
            tail = rest.split(END, 1)[1]
            MANIFEST.write_text(head + START + "\n" + "\n".join(table) + "\n" + END + tail, encoding="utf-8")
    total = sum(r[1] for r in rows)
    print(f"{len(rows)} files, {total / 2**20:.1f} MB -> artifacts/pose/asset_inventory.csv (+ manifest appendix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
