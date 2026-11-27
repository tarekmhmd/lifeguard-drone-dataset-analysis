"""Phase 0 - read-only audit of a pose / keypoint dataset.

Nothing is written next to the dataset. Outputs go to --out only:
  AUDIT_REPORT.json, AUDIT_SUMMARY.txt, contact_sheet.jpg

Usage (Colab or local):
  python audit_pose_dataset.py --list-candidates /content/drive/MyDrive
  python audit_pose_dataset.py --root /content/pose_ds --out /content/pose_audit_out

Supported annotation formats (auto-detected, parsed from the real files):
  COCO JSON (images + annotations with `keypoints`), YOLO-pose txt, CVAT XML
  (<points>/<skeleton>), per-frame CSV (header + a few rows are only reported).
Everything reported is computed from the files; nothing is assumed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}
EDGE_NAME_PAIRS = [("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "knee"), ("knee", "ankle")]


# --------------------------------------------------------------------------- helpers
def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def split_of(path: Path) -> str:
    for part in reversed(path.parts):
        key = part.lower()
        if key in SPLIT_NAMES:
            return SPLIT_NAMES[key]
    return "unknown"


def video_id_of(stem: str) -> str:
    """Heuristic source-video id from an image file name (reported as a heuristic)."""
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    s = re.sub(r"_(jpg|jpeg|png)$", "", s, flags=re.I)
    s = re.sub(r"([_\-.]?(frame|img|image|f)?[_\-.]?\d+)$", "", s, flags=re.I)
    return s or stem


def list_candidates(root: Path, keywords: tuple[str, ...], depth: int) -> None:
    kw = [k.lower() for k in keywords]
    print(f"Candidate folders under {root} (name contains one of {kw}):")
    found = 0
    base = len(root.parts)
    for p in sorted(root.rglob("*")):
        if len(p.parts) - base > depth:
            continue
        if p.is_dir() and any(k in p.name.lower() for k in kw):
            n_files = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {p}  ({n_files} files)")
            found += 1
    if not found:
        print("  none found - check the Drive mount and the folder name.")


# --------------------------------------------------------------------------- scan
def scan_files(root: Path) -> dict[str, Any]:
    ext_count: collections.Counter[str] = collections.Counter()
    ext_size: collections.Counter[str] = collections.Counter()
    images: list[Path] = []
    videos: list[Path] = []
    others: list[Path] = []
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower() or "(none)"
        size = p.stat().st_size
        ext_count[ext] += 1
        ext_size[ext] += size
        total += size
        if ext in IMG_EXT:
            images.append(p)
        elif ext in VID_EXT:
            videos.append(p)
        else:
            others.append(p)
    return {"ext_count": dict(ext_count), "ext_size": dict(ext_size), "total_bytes": total,
            "images": images, "videos": videos, "others": others}


def tree_text(root: Path, depth: int, max_entries: int = 12) -> str:
    lines = [str(root)]

    def rec(d: Path, level: int, prefix: str) -> None:
        if level >= depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except OSError:
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        for e in dirs[:max_entries]:
            n = sum(1 for _ in e.rglob("*") if _.is_file())
            lines.append(f"{prefix}{e.name}/  ({n} files)")
            rec(e, level + 1, prefix + "    ")
        if len(dirs) > max_entries:
            lines.append(f"{prefix}... {len(dirs) - max_entries} more folders")
        for e in files[:5]:
            lines.append(f"{prefix}{e.name}")
        if len(files) > 5:
            lines.append(f"{prefix}... {len(files) - 5} more files")

    rec(root, 0, "  ")
    return "\n".join(lines)


def image_stats(images: list[Path], do_hash: bool) -> dict[str, Any]:
    from PIL import Image  # Colab has Pillow

    sizes: collections.Counter[str] = collections.Counter()
    corrupt: list[str] = []
    hashes: dict[str, list[str]] = collections.defaultdict(list)
    dims: dict[str, tuple[int, int]] = {}
    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
            dims[str(p)] = (w, h)
            sizes[f"{w}x{h}"] += 1
        except Exception as exc:  # noqa: BLE001 - audit must survive any bad file
            corrupt.append(f"{p}: {type(exc).__name__}")
            continue
        if do_hash:
            hashes[hashlib.md5(p.read_bytes()).hexdigest()].append(str(p))
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    return {"resolution_hist": dict(sizes.most_common(25)), "n_resolutions": len(sizes),
            "corrupt": corrupt, "duplicate_groups": dup_groups, "dims": dims}


# --------------------------------------------------------------------------- parsers
# Unified record: {"image": str, "persons": [{"bbox": [x1,y1,x2,y2]|None,
#                  "kpts": [(x, y, v), ...]}], "w": int|None, "h": int|None}
def parse_coco(path: Path) -> tuple[list[dict], list[str] | None, list | None, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or "annotations" not in data or "images" not in data:
        return None
    imgs = {i["id"]: i for i in data["images"]}
    names = skeleton = None
    for c in data.get("categories", []):
        if "keypoints" in c:
            names, skeleton = c["keypoints"], c.get("skeleton")
            break
    recs: dict[Any, dict] = {}
    for a in data["annotations"]:
        im = imgs.get(a.get("image_id"))
        if im is None:
            continue
        r = recs.setdefault(a["image_id"], {"image": im.get("file_name", ""), "persons": [],
                                            "w": im.get("width"), "h": im.get("height")})
        k = a.get("keypoints") or []
        kp = [(k[i], k[i + 1], k[i + 2]) for i in range(0, len(k) - 2, 3)]
        bb = a.get("bbox")
        bbox = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] if bb and len(bb) == 4 else None
        r["persons"].append({"bbox": bbox, "kpts": kp, "num_keypoints": a.get("num_keypoints")})
    for iid, im in imgs.items():  # images with no annotation at all
        recs.setdefault(iid, {"image": im.get("file_name", ""), "persons": [],
                              "w": im.get("width"), "h": im.get("height")})
    return list(recs.values()), names, skeleton, "coco_json"


def parse_cvat_xml(path: Path) -> tuple[list[dict], list[str] | None, list | None, str] | None:
    try:
        root = ET.parse(path).getroot()
    except Exception:  # noqa: BLE001
        return None
    images = root.findall(".//image")
    if not images:
        return None
    recs = []
    names: list[str] | None = None
    for im in images:
        persons = []
        for sk in im.findall("skeleton"):
            pts = sk.findall("points")
            if names is None:
                names = [p.get("label", "") for p in pts]
            kp = []
            for p in pts:
                x, y = (float(t) for t in p.get("points", "0,0").split(",")[:2])
                occluded = p.get("occluded", "0") == "1"
                outside = p.get("outside", "0") == "1"
                kp.append((x, y, 0 if outside else (1 if occluded else 2)))
            persons.append({"bbox": None, "kpts": kp})
        for pts_el in im.findall("points"):  # flat point annotations
            x, y = (float(t) for t in pts_el.get("points", "0,0").split(",")[:2])
            persons.append({"bbox": None, "kpts": [(x, y, 2)], "flat_label": pts_el.get("label")})
        recs.append({"image": im.get("name", ""), "persons": persons,
                     "w": int(im.get("width", 0)) or None, "h": int(im.get("height", 0)) or None})
    return recs, names, None, "cvat_xml"


def looks_like_yolo_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 5:
        return False
    try:
        [float(t) for t in parts]
    except ValueError:
        return False
    return parts[0].lstrip("-").isdigit()


def parse_yolo_pose(images: list[Path], dims: dict[str, tuple[int, int]]
                    ) -> tuple[list[dict], dict[str, Any]] | None:
    """Pair image with label by stem (images/ -> labels/ swap, or same folder)."""
    recs: list[dict] = []
    token_counts: collections.Counter[int] = collections.Counter()
    pairs = 0
    for img in images:
        cands = [img.with_suffix(".txt")]
        parts = list(img.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].lower() == "images":
                parts[i] = "labels"
                cands.append(Path(*parts).with_suffix(".txt"))
                break
        lbl = next((c for c in cands if c.is_file()), None)
        if lbl is None:
            continue
        lines = [ln for ln in lbl.read_text(errors="replace").splitlines() if ln.strip()]
        if lines and not looks_like_yolo_line(lines[0]):
            continue
        pairs += 1
        rows = []
        for ln in lines:
            toks = ln.split()
            token_counts[len(toks)] += 1
            rows.append(toks)
        recs.append({"image": str(img), "rows": rows, "label": str(lbl)})
    if not pairs:
        return None
    n_tok = token_counts.most_common(1)[0][0]
    info = {"token_count_hist": dict(token_counts), "dominant_tokens": n_tok,
            "images_without_label_file": len(images) - pairs}
    k3 = (n_tok - 5) / 3
    k2 = (n_tok - 5) / 2
    if n_tok > 5 and k3 == int(k3):
        stride, k = 3, int(k3)
        info["layout"] = f"class + box + {k} x (x y v)"
        if k2 == int(k2):
            info["ambiguous_layout"] = (f"{n_tok} tokens also fits {int(k2)} x (x y) "
                                        "without visibility; check data YAML kpt_shape")
    elif n_tok > 5 and k2 == int(k2):
        stride, k = 2, int(k2)
        info["layout"] = f"class + box + {k} x (x y), no visibility"
    else:
        info["layout"] = "box only (no keypoints) or unrecognised"
        stride, k = 0, 0
    out = []
    for r in recs:
        w, h = dims.get(r["image"], (None, None))
        persons = []
        for toks in r["rows"]:
            vals = [float(t) for t in toks]
            sx, sy = (w, h) if w and h else (1, 1)
            bbox = None
            if len(vals) >= 5:
                cx, cy, bw, bh = vals[1:5]
                bbox = [(cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy]
            kp = []
            if stride and len(vals) == 5 + stride * k:
                for i in range(k):
                    base = 5 + i * stride
                    kp.append((vals[base] * sx, vals[base + 1] * sy,
                               vals[base + 2] if stride == 3 else 2))
            persons.append({"bbox": bbox, "kpts": kp, "cls": int(vals[0])})
        out.append({"image": r["image"], "persons": persons, "w": w, "h": h, "label": r["label"]})
    info["n_keypoints_inferred"] = k
    info["labels_normalised"] = bool(dims)
    return out, info


def parse_csv_summary(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return {"file": str(path), "error": str(exc)}
    return {"file": str(path), "rows": len(lines) - 1, "header": lines[0] if lines else "",
            "first_rows": lines[1:4]}


# --------------------------------------------------------------------------- analysis
def find_idx(names: list[str], *needles: str) -> int | None:
    for i, n in enumerate(names):
        low = n.lower().replace("-", "_").replace(" ", "_")
        if all(t in low for t in needles):
            return i
    return None


def side_idx(names: list[str], side: str, part: str) -> int | None:
    for pat in (f"{side}_{part}", f"{side[0]}_{part}", f"{part}_{side}", f"{part}_{side[0]}"):
        i = find_idx(names, pat)
        if i is not None:
            return i
    return None


def analyse(recs: list[dict], names: list[str] | None, dims: dict[str, tuple[int, int]],
            root: Path) -> dict[str, Any]:
    n_img = len(recs)
    persons_per_img: collections.Counter[int] = collections.Counter()
    k_counts: collections.Counter[int] = collections.Counter()
    img_no_kpt = img_all_zero = img_partial = 0
    n_boxes = n_persons = 0
    labelled: collections.Counter[int] = collections.Counter()
    visible: collections.Counter[int] = collections.Counter()
    vis_values: collections.Counter[float] = collections.Counter()
    total_by_idx: collections.Counter[int] = collections.Counter()
    q: collections.Counter[str] = collections.Counter()
    lr_sign: collections.Counter[str] = collections.Counter()
    dup_ann = 0
    in_water_note = collections.Counter()
    splits: dict[str, set[str]] = collections.defaultdict(set)
    split_imgs: collections.Counter[str] = collections.Counter()
    sizes_rel: list[float] = []
    samples_bad: dict[str, list[str]] = collections.defaultdict(list)

    ls = rs = None
    if names:
        ls, rs = side_idx(names, "left", "shoulder"), side_idx(names, "right", "shoulder")

    for r in recs:
        sp = split_of(Path(r["image"]))
        split_imgs[sp] += 1
        splits[sp].add(video_id_of(Path(r["image"]).stem))
        persons = r["persons"]
        persons_per_img[len(persons)] += 1
        w, h = r.get("w"), r.get("h")
        if not (w and h):
            cand = dims.get(r["image"]) or dims.get(str(root / r["image"]))
            if cand:
                w, h = cand
        seen: set[tuple] = set()
        has_any = has_full = False
        for p in persons:
            n_persons += 1
            kp = p["kpts"]
            if p.get("bbox"):
                n_boxes += 1
            k_counts[len(kp)] += 1
            if not kp:
                continue
            vals_ok = all(all(isinstance(c, (int, float)) and math.isfinite(c) for c in t) for t in kp)
            if not vals_ok:
                q["nan_or_inf"] += 1
                samples_bad["nan_or_inf"].append(r["image"])
                continue
            n_lab = 0
            for i, (x, y, v) in enumerate(kp):
                total_by_idx[i] += 1
                vis_values[v] += 1
                is_lab = v > 0 or (x != 0 or y != 0)
                if is_lab:
                    labelled[i] += 1
                    n_lab += 1
                if v >= 2 or (v == 1 and False):
                    visible[i] += 1
                if (x or y) and w and h and (x < 0 or y < 0 or x > w or y > h):
                    q["keypoint_outside_image"] += 1
                    if len(samples_bad["keypoint_outside_image"]) < 5:
                        samples_bad["keypoint_outside_image"].append(r["image"])
            if n_lab:
                has_any = True
            if n_lab == len(kp):
                has_full = True
            xs = [t[0] for t in kp if t[0] or t[1]]
            ys = [t[1] for t in kp if t[0] or t[1]]
            if len(xs) >= 2 and (max(xs) - min(xs) < 2 and max(ys) - min(ys) < 2):
                q["degenerate_skeleton"] += 1
                if len(samples_bad["degenerate_skeleton"]) < 5:
                    samples_bad["degenerate_skeleton"].append(r["image"])
            bb = p.get("bbox")
            if bb and xs:
                bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                tol = 0.05 * max(bw, bh, 1)
                if (min(xs) < bb[0] - tol or max(xs) > bb[2] + tol
                        or min(ys) < bb[1] - tol or max(ys) > bb[3] + tol):
                    q["keypoints_outside_own_box"] += 1
                    if len(samples_bad["keypoints_outside_own_box"]) < 5:
                        samples_bad["keypoints_outside_own_box"].append(r["image"])
                if w and h:
                    sizes_rel.append(math.sqrt(max(bw, 0) * max(bh, 0) / (w * h)))
                diag = math.hypot(bw, bh)
                if names:
                    for a, b in EDGE_NAME_PAIRS:
                        for side in ("left", "right"):
                            ia, ib = side_idx(names, side, a), side_idx(names, side, b)
                            if ia is None or ib is None or ia >= len(kp) or ib >= len(kp):
                                continue
                            pa, pb = kp[ia], kp[ib]
                            if (pa[0] or pa[1]) and (pb[0] or pb[1]):
                                length = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                                if diag and length > 1.2 * diag:
                                    q["implausible_limb_length"] += 1
                                    if len(samples_bad["implausible_limb_length"]) < 5:
                                        samples_bad["implausible_limb_length"].append(r["image"])
            if ls is not None and rs is not None and ls < len(kp) and rs < len(kp):
                a, b = kp[ls], kp[rs]
                if (a[0] or a[1]) and (b[0] or b[1]) and a[0] != b[0]:
                    lr_sign["left_x_greater" if a[0] > b[0] else "left_x_smaller"] += 1
            key = tuple(round(c, 1) for t in kp for c in t)
            if key in seen:
                dup_ann += 1
            seen.add(key)
        if not persons or not has_any:
            img_no_kpt += 1
        elif not has_full:
            img_partial += 1
        if persons and not has_any:
            img_all_zero += 1

    # leakage between splits by heuristic video id
    leak: dict[str, list[str]] = {}
    sp_names = [s for s in splits if s != "unknown"]
    for i, a in enumerate(sp_names):
        for b in sp_names[i + 1:]:
            common = sorted(splits[a] & splits[b])
            if common:
                leak[f"{a}&{b}"] = common[:20] + ([f"... {len(common) - 20} more"] if len(common) > 20 else [])
    n_k = k_counts.most_common(1)[0][0] if k_counts else 0
    per_kp = []
    for i in range(n_k):
        tot = total_by_idx[i] or 1
        per_kp.append({"index": i, "name": names[i] if names and i < len(names) else None,
                       "labelled_fraction": round(labelled[i] / tot, 4),
                       "visible_fraction": round(visible[i] / tot, 4)})
    return {
        "n_images_in_annotations": n_img, "n_persons": n_persons, "n_boxes": n_boxes,
        "persons_per_image_hist": dict(sorted(persons_per_img.items())),
        "keypoints_per_person_hist": dict(k_counts),
        "images_without_any_keypoint": img_no_kpt, "images_partially_labelled": img_partial,
        "images_with_annotations_but_all_zero": img_all_zero,
        "visibility_value_hist": {str(k): v for k, v in vis_values.items()},
        "per_keypoint": per_kp,
        "images_per_split_folder": dict(split_imgs),
        "heuristic_video_ids_per_split": {k: len(v) for k, v in splits.items()},
        "leakage_by_heuristic_video_id": leak,
        "quality_counts": dict(q), "quality_samples": dict(samples_bad),
        "left_right_shoulder_x": dict(lr_sign), "duplicate_annotations_same_image": dup_ann,
        "relative_subject_size_sqrt_area": (
            {"min": round(min(sizes_rel), 3), "median": round(sorted(sizes_rel)[len(sizes_rel) // 2], 3),
             "max": round(max(sizes_rel), 3)} if sizes_rel else None),
        "in_water_note": dict(in_water_note),
    }


def cross_split_duplicates(dup_groups: list[list[str]]) -> list[list[str]]:
    out = []
    for g in dup_groups:
        if len({split_of(Path(p)) for p in g}) > 1:
            out.append(g)
    return out


def diversity_hints(images: list[Path], videos: list[Path]) -> dict[str, Any]:
    words = ["freestyle", "crawl", "butterfly", "breast", "back", "underwater", "above", "surface",
             "drown", "float", "tread", "dive", "diving", "wave", "splash", "kick", "sink"]
    found: collections.Counter[str] = collections.Counter()
    for p in list(images) + list(videos):
        low = str(p).lower()
        for w in words:
            if w in low:
                found[w] += 1
    return {"keywords_found_in_paths": dict(found),
            "note": "Only path/file-name keywords; real diversity must be judged on the contact sheet."}


# --------------------------------------------------------------------------- contact sheet
def contact_sheet(recs: list[dict], names: list[str] | None, out: Path, root: Path,
                  n: int = 24, seed: int = 0) -> str | None:
    from PIL import Image, ImageDraw

    usable = [r for r in recs if r["persons"] and any(p["kpts"] for p in r["persons"])]
    if not usable:
        return None
    random.Random(seed).shuffle(usable)
    cols, cell = 6, 320
    tiles = []
    for r in usable:
        pth = Path(r["image"])
        if not pth.is_file():
            hits = list(root.rglob(pth.name)) if pth.name else []
            if not hits:
                continue
            pth = hits[0]
        try:
            im = Image.open(pth).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        d = ImageDraw.Draw(im)
        s = max(2, int(min(im.size) / 120))
        for pi, p in enumerate(r["persons"]):
            col = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0)][pi % 4]
            if p.get("bbox"):
                d.rectangle(p["bbox"], outline=col, width=max(1, s // 2))
            for i, (x, y, v) in enumerate(p["kpts"]):
                if not (x or y) or v == 0:
                    continue
                d.ellipse([x - s, y - s, x + s, y + s], fill=col if v >= 2 else (255, 255, 255))
                if names and i < len(names) and len(r["persons"]) == 1:
                    d.text((x + s + 1, y - s), str(i), fill=(255, 255, 0))
        im.thumbnail((cell, cell))
        tiles.append((im, pth.name[:38]))
        if len(tiles) == n:
            break
    if not tiles:
        return None
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 14)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(tiles):
        x, y = (i % cols) * cell, (i // cols) * (cell + 14)
        sheet.paste(im, (x, y))
        d.text((x + 2, y + cell + 1), label, fill=(220, 220, 220))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contact_sheet.jpg"
    sheet.save(path, quality=88)
    return str(path)


# --------------------------------------------------------------------------- summary
def summary_text(rep: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    f = rep["files"]
    a("=== POSE DATASET AUDIT SUMMARY (all numbers computed from the files) ===")
    a(f"root: {rep['root']}")
    a(f"total size: {human(f['total_bytes'])}; images: {f['n_images']}; videos: {f['n_videos']}")
    a(f"extensions: {f['ext_count']}")
    im = rep["images"]
    a(f"resolutions ({im['n_resolutions']} distinct, top): {im['resolution_hist']}")
    a(f"corrupt images: {len(im['corrupt'])}; duplicate groups (md5): {len(im['duplicate_groups'])}; "
      f"duplicate groups crossing splits: {len(im['cross_split_duplicate_groups'])}")
    a("--- annotation sources ---")
    for s in rep["annotation_sources"]:
        a(f"  {s}")
    an = rep.get("analysis")
    if not an:
        a("NO PARSEABLE KEYPOINT ANNOTATIONS FOUND. See annotation_sources / csv_files / other files above.")
        return "\n".join(L)
    a(f"format: {rep['format']}; keypoint names/order: {rep['keypoint_names']}")
    if rep.get("yolo_info"):
        a(f"yolo info: {rep['yolo_info']}")
    if rep.get("coco_skeleton"):
        a(f"skeleton edges: {rep['coco_skeleton']}")
    a(f"annotated images: {an['n_images_in_annotations']}; persons: {an['n_persons']}; "
      f"boxes: {an['n_boxes']}")
    a(f"persons per image: {an['persons_per_image_hist']}")
    a(f"keypoints per person: {an['keypoints_per_person_hist']}; visibility values: {an['visibility_value_hist']}")
    a(f"images without keypoints: {an['images_without_any_keypoint']}; partially labelled: "
      f"{an['images_partially_labelled']}")
    a("per-keypoint labelled / visible fraction:")
    for k in an["per_keypoint"]:
        a(f"  {k['index']:>2} {str(k['name']):<18} labelled {k['labelled_fraction']:.2f}  "
          f"visible {k['visible_fraction']:.2f}")
    a(f"images per split folder: {an['images_per_split_folder']}")
    a(f"heuristic video ids per split: {an['heuristic_video_ids_per_split']}")
    a(f"LEAKAGE by heuristic video id: {an['leakage_by_heuristic_video_id'] or 'none detected'}")
    a(f"subject size sqrt(box/img area): {an['relative_subject_size_sqrt_area']}")
    a(f"quality counts: {an['quality_counts'] or 'none'}")
    a(f"left/right shoulder x ordering: {an['left_right_shoulder_x'] or 'n/a'} "
      "(consistent mix is normal; a systematic single value in a mirrored dataset needs the contact sheet)")
    a(f"duplicate annotations in one image: {an['duplicate_annotations_same_image']}")
    a(f"diversity hints: {rep['diversity']}")
    a(f"contact sheet: {rep.get('contact_sheet')}")
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def run(root: Path, out: Path, depth: int, do_hash: bool) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    scan = scan_files(root)
    imgs = scan["images"]
    ims = image_stats(imgs, do_hash)
    dims = ims.pop("dims")
    ims["cross_split_duplicate_groups"] = cross_split_duplicates(ims["duplicate_groups"])
    ims["duplicate_groups"] = ims["duplicate_groups"][:50]
    ims["cross_split_duplicate_groups"] = ims["cross_split_duplicate_groups"][:50]

    sources: list[str] = []
    csvs: list[dict] = []
    recs: list[dict] | None = None
    names = skeleton = None
    fmt = None
    yolo_info = None
    json_paths = [p for p in scan["others"] if p.suffix.lower() == ".json"]
    xml_paths = [p for p in scan["others"] if p.suffix.lower() == ".xml"]
    csv_paths = [p for p in scan["others"] if p.suffix.lower() == ".csv"]

    coco_all: list[dict] = []
    for p in json_paths:
        r = parse_coco(p)
        if r:
            coco_all += [{**x, "image": str(p.parent / x["image"]) if not Path(x["image"]).is_absolute()
                          and (p.parent / x["image"]).exists() else x["image"]} for x in r[0]]
            names = names or r[1]
            skeleton = skeleton or r[2]
            fmt = "coco_json"
            sources.append(f"COCO JSON: {p} ({len(r[0])} images)")
    if coco_all:
        recs = coco_all
    if recs is None:
        cvat_all: list[dict] = []
        for p in xml_paths:
            r = parse_cvat_xml(p)
            if r:
                cvat_all += r[0]
                names = names or r[1]
                fmt = "cvat_xml"
                sources.append(f"CVAT XML: {p} ({len(r[0])} images)")
        if cvat_all:
            recs = cvat_all
    if recs is None:
        y = parse_yolo_pose(imgs, {k: v for k, v in dims.items()})
        if y:
            recs, yolo_info = y
            fmt = "yolo_pose_txt"
            sources.append(f"YOLO txt labels paired with {len(recs)} images; {yolo_info}")
            for yp in [q for q in scan["others"] if q.suffix.lower() in (".yaml", ".yml")]:
                txt = yp.read_text(errors="replace")
                if "kpt_shape" in txt or "flip_idx" in txt or "names" in txt:
                    sources.append(f"data yaml: {yp}\n{txt[:800]}")
                    m = re.search(r"kpt_shape:\s*\[(\d+)\s*,\s*(\d+)\]", txt)
                    if m and yolo_info:
                        yolo_info["yaml_kpt_shape"] = [int(m.group(1)), int(m.group(2))]
    for p in csv_paths:
        s = parse_csv_summary(p)
        csvs.append(s)
        sources.append(f"CSV: {p} rows={s.get('rows')} header={s.get('header')}")

    # resolve names for position-only formats
    analysis = analyse(recs, names, dims, root) if recs else None
    sheet = contact_sheet(recs, names, out, root) if recs else None
    rep: dict[str, Any] = {
        "root": str(root), "tree": tree_text(root, depth),
        "files": {"ext_count": scan["ext_count"], "ext_size": scan["ext_size"],
                  "total_bytes": scan["total_bytes"], "n_images": len(imgs),
                  "n_videos": len(scan["videos"]),
                  "videos_sample": [str(v) for v in scan["videos"][:10]]},
        "images": ims, "format": fmt, "keypoint_names": names, "coco_skeleton": skeleton,
        "yolo_info": yolo_info, "annotation_sources": sources, "csv_files": csvs,
        "raw_examples": [
            {"image": r["image"], "persons": r["persons"][:2]} for r in (recs or [])[:3]],
        "analysis": analysis, "diversity": diversity_hints(imgs, scan["videos"]),
        "contact_sheet": sheet,
    }
    (out / "AUDIT_REPORT.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    text = summary_text(rep)
    (out / "AUDIT_SUMMARY.txt").write_text(text, encoding="utf-8")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="pose dataset folder (read only)")
    ap.add_argument("--out", type=Path, default=Path("pose_audit_out"))
    ap.add_argument("--tree-depth", type=int, default=3)
    ap.add_argument("--no-hash", action="store_true", help="skip md5 duplicate check (faster)")
    ap.add_argument("--list-candidates", type=Path, help="list folders whose name suggests pose data")
    ap.add_argument("--keywords", default="pose,keypoint,skeleton,swim,drown,pool")
    a = ap.parse_args(argv)
    if a.list_candidates:
        list_candidates(a.list_candidates, tuple(a.keywords.split(",")), depth=4)
        return 0
    if not a.root or not a.root.is_dir():
        print("ERROR: --root must be an existing folder", file=sys.stderr)
        return 2
    rep = run(a.root, a.out, a.tree_depth, not a.no_hash)
    print(rep["tree"])
    print()
    print(summary_text(rep))
    print(f"\nWritten: {a.out}/AUDIT_REPORT.json, AUDIT_SUMMARY.txt, contact_sheet.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
