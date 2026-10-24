# Auto-generated Python copy of notebook: downloadDataOnColab.ipynb
# Edits belong in the notebook; this is a faithful code export.
# Colab/notebook-only lines are commented with [shell] or [magic].

# %%

from pathlib import Path

PROJECT = Path("/content/lifeguard_test")

for folder in ["videos", "annotations", "frames"]:
    path = PROJECT / folder

    print(f"\n--- {folder} ---")

    if path.exists():
        files = list(path.iterdir())
        print("Files:", len(files))

        for file in files[:10]:
            print(file.name)
    else:
        print("Folder not found")

# %%
# Cell 1 — Setup and a safe download helper.
import os, urllib.request, urllib.error

BASE = "/content/lifeguard_test"
RAW  = "https://raw.githubusercontent.com/Z5cc/drowning-detection/master/"
os.makedirs(BASE, exist_ok=True)

def download(url, dest, expected=None):
    """Download one file. Skips it if it already exists and is non-empty
    (so your two existing XMLs/videos/frames are never overwritten)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("SKIP (already present):", dest)
        return 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "colab-dataset-prep"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        print(f"ERROR {e.code}: {url}")
        return -1
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {url}")
        return -1
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    print(f"OK {dest} ({len(data)} bytes)")
    if expected is not None and len(data) != expected:
        print(f"  WARNING: size {len(data)} != expected {expected}")
    return len(data)

# %%
# Cell 2 — The manifest table: maps every clip to video id + start/end + crop.
# Verified size from the repo: 6,550 bytes.
download(RAW + "dataset_lifeguard/datalist.csv",
         f"{BASE}/datalist.csv", expected=6550)

# %%
# Cell 3 — All temporal swim/drown annotations (91 XMLs in repo; you already have 2).
import json

API = "https://api.github.com/repos/Z5cc/drowning-detection/git/trees/master?recursive=1"
req = urllib.request.Request(API, headers={"User-Agent": "colab-dataset-prep"})
with urllib.request.urlopen(req, timeout=60) as r:
    tree = json.load(r)

size_by_path = {e["path"]: e["size"] for e in tree["tree"] if e["type"] == "blob"}
label_paths = [p for p in size_by_path
               if p.startswith("dataset_lifeguard/labels/") and p.endswith(".xml")]

downloaded = skipped = errors = total = 0
for p in sorted(label_paths):
    dest = f"{BASE}/annotations/{p.split('/')[-1]}"
    n = download(RAW + p, dest, expected=size_by_path[p])
    if n == 0:   skipped += 1
    elif n < 0:  errors += 1
    else:        downloaded += 1; total += n

print(f"\nLabels: {downloaded} downloaded ({total/1e6:.2f} MB), {skipped} already present, {errors} errors.")

# %%
# Cell 4 — The author's train/test splits (9 folds + final + difficult).
# Uses `size_by_path` from Cell 3, so run the cells in order.
split_paths = [p for p in size_by_path
               if "splitlists" in p and p.endswith(".csv")]

for p in sorted(split_paths):
    download(RAW + p, f"{BASE}/splitlists/{p.split('/')[-1]}", expected=size_by_path[p])

print("Split lists saved under", f"{BASE}/splitlists/")

# %%
# Cell 1 — Parse all annotation XMLs and summarize (read-only)
import os, glob
import xml.etree.ElementTree as ET
import pandas as pd

BASE = "/content/lifeguard_test"
xml_files = sorted(glob.glob(f"{BASE}/annotations/*.xml"))

box_counts = {"swim": 0, "drown": 0}
frame_with = {"swim": 0, "drown": 0}
file_with  = {"swim": 0, "drown": 0}
tot_frames = 0
unexpected = {}        # any label that is not swim/drown
parse_failed = []      # files that could not be read
id_gaps = []           # files where frame ids are not 0,1,2,3...

for path in xml_files:
    name = os.path.basename(path)
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        parse_failed.append((name, str(e)))
        continue

    sw = dw = sw_f = dw_f = 0
    last = -1
    for im in root.iter("image"):
        tot_frames += 1
        iid = int(im.get("id"))
        if iid != last + 1:
            id_gaps.append((name, last, iid))   # frame order is broken here
        last = iid
        s = d = False
        for b in im.iter("box"):
            lab = b.get("label")
            if lab == "swim":   sw += 1; s = True
            elif lab == "drown": dw += 1; d = True
            else: unexpected[lab] = unexpected.get(lab, 0) + 1
        if s: sw_f += 1
        if d: dw_f += 1

    box_counts["swim"] += sw; box_counts["drown"] += dw
    frame_with["swim"] += sw_f; frame_with["drown"] += dw_f
    if sw: file_with["swim"] += 1
    if dw: file_with["drown"] += 1

table = pd.DataFrame([
    ["Annotation files found", len(xml_files)],
    ["Files with parse error", len(parse_failed)],
    ["Total annotated frames", tot_frames],
    ["swim  bounding boxes", box_counts["swim"]],
    ["drown bounding boxes", box_counts["drown"]],
    ["Frames containing swim", frame_with["swim"]],
    ["Frames containing drown", frame_with["drown"]],
    ["Files containing swim", file_with["swim"]],
    ["Files containing drown", file_with["drown"]],
], columns=["Metric", "Value"])

print(table.to_string(index=False))
print("\nUnexpected labels:", unexpected or "none")
print("Frame-id gaps (order broken):", id_gaps or "none")
print("Parse failures:", parse_failed or "none")

# %%
# Cell 2 — Bar chart: swim vs drown box counts
import matplotlib.pyplot as plt

labels = ["swim", "drown"]
values = [box_counts["swim"], box_counts["drown"]]

plt.bar(labels, values, color=["#1f77b4", "#d62728"])
plt.ylabel("Bounding boxes")
plt.title("Box count per class (all annotation files)")
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha="center", va="bottom")
plt.show()

# %%
# Cell 3 — Inspect the train/test split CSVs
import csv, glob

split_files = sorted(glob.glob(f"{BASE}/splitlists/*.csv"))
print("Split files found:", len(split_files), "\n")

for f in split_files:
    with open(f, newline="") as fh:
        rows = list(csv.reader(fh))
    ncols = max((len(r) for r in rows), default=0)
    print(f"{os.path.basename(f):24s} lines={len(rows):3d}  columns={ncols}")

print("\nPreview of train.csv (1 column = clip id):")
with open(f"{BASE}/splitlists/train.csv", newline="") as fh:
    for row in list(csv.reader(fh))[:3]:
        print("  ", row)

print("\nPreview of test.csv (2 columns = clip id, drown flag):")
with open(f"{BASE}/splitlists/test.csv", newline="") as fh:
    for row in list(csv.reader(fh))[:5]:
        print("  ", row)

# %%
# Cell 4 — Which annotation clips match the two videos/frames already on disk?
videos = sorted(glob.glob(f"{BASE}/videos/*.mp4"))
xml_names = [os.path.basename(p) for p in xml_files]

for v in videos:
    vname = os.path.splitext(os.path.basename(v))[0]
    matches = [n for n in xml_names if n.startswith(vname + "_")]
    print(f"\n{vname}.mp4  -> {len(matches)} annotation clip(s):")
    for m in matches:
        root = ET.parse(f"{BASE}/annotations/{m}").getroot()
        n = sum(1 for _ in root.iter("image"))
        print(f"    {m}  ({n} annotated frames)")
    fdir = f"{BASE}/frames/{vname}"
    extracted = len(glob.glob(f"{fdir}/*.jpg")) if os.path.isdir(fdir) else 0
    print(f"  extracted frames on disk: {extracted}")

# %%
# Cell 1 — Load all split CSVs and inspect their columns
import os, glob, csv, re

BASE = "/content/lifeguard_test"

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))

def video_id_of(snippet):
    # snippet = "videoId_00-00-15--00-00-25_0-46--0-33--0-12"
    m = re.match(r"^(.*?)_\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_", snippet)
    return m.group(1) if m else "?"

files = sorted(glob.glob(f"{BASE}/splitlists/*.csv"))
print("Split files:", len(files), "\n")
for f in files:
    rows = read_csv(f)
    ncols = max((len(r) for r in rows), default=0)
    print(f"{os.path.basename(f):24s} rows={len(rows):3d}  columns={ncols}")

print("\nPreview train.csv (1 column = clip id):")
for r in read_csv(f"{BASE}/splitlists/train.csv")[:3]:
    print("  ", r)
print("Preview test.csv (2 columns = clip id, d/s flag):")
for r in read_csv(f"{BASE}/splitlists/test.csv")[:5]:
    print("  ", r)

# %%
# Cell 2 — Distribution of 'd' vs 's' in every two-column (test) file
from collections import Counter

print("Test files: 'd' = clip contains drown, 's' = swim-only")
for f in files:
    rows = read_csv(f)
    if rows and len(rows[0]) >= 2:
        c = Counter(r[1] for r in rows if len(r) >= 2)
        print(f"  {os.path.basename(f):24s} {dict(c)}")

# %%
# Cell 3 — Missing references and orphan annotations
xml_base = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(f"{BASE}/annotations/*.xml")}

all_clips = set()
for f in files:
    for r in read_csv(f):
        if r:
            all_clips.add(r[0])

missing = sorted(c for c in all_clips if c not in xml_base)   # in splits, no XML
orphans = sorted(x for x in xml_base if x not in all_clips)   # XML, not in any split

print("XML annotation files:", len(xml_base))
print("Distinct clip IDs across all splits:", len(all_clips))
print("\nSplit clip IDs with NO matching XML:", len(missing))
for c in missing[:20]: print("  ", c)
print("\nXMLs NOT referenced by any split file:", len(orphans))
for x in orphans[:20]: print("  ", x)

# %%
# Cell 4 — Group clips by original video and check train/test overlap
def vids_of(rows):
    return {video_id_of(r[0]) for r in rows if r}

tr = read_csv(f"{BASE}/splitlists/train.csv")
te = read_csv(f"{BASE}/splitlists/test.csv")
tr_v, te_v = vids_of(tr), vids_of(te)
print("FINAL split: train=%d clips (%d videos) | test=%d clips (%d videos) | videos in BOTH=%d"
      % (len(tr), len(tr_v), len(te), len(te_v), len(tr_v & te_v)))
if tr_v & te_v:
    print("  leaked videos:", sorted(tr_v & te_v))

print("\nNine folds:")
for k in range(1, 10):
    a = read_csv(f"{BASE}/splitlists/{k}_train.csv")
    b = read_csv(f"{BASE}/splitlists/{k}_test.csv")
    a_clips = {r[0] for r in a if r}
    b_clips = {r[0] for r in b if r}
    a_v, b_v = vids_of(a), vids_of(b)
    print(f"  fold {k}: train={len(a_clips)} clips ({len(a_v)} videos) "
          f"test={len(b_clips)} clips ({len(b_v)} videos) "
          f"| clip_overlap={len(a_clips & b_clips)} video_overlap={len(a_v & b_v)}")

# %%
# Cell 5 — One table per split file + coverage of the final split
import pandas as pd

rows = []
for f in files:
    r = read_csv(f)
    ncols = max((len(x) for x in r), default=0)
    is_test = ncols >= 2
    clips = {x[0] for x in r if x}
    flags = Counter(x[1] for x in r if len(x) >= 2) if is_test else {}
    rows.append({
        "file": os.path.basename(f), "rows": len(r), "cols": ncols,
        "clips": len(clips), "videos": len(vids_of(r)),
        "d": flags.get("d", 0), "s": flags.get("s", 0),
    })

print(pd.DataFrame(rows).to_string(index=False))

final = ({x[0] for x in read_csv(f"{BASE}/splitlists/train.csv") if x}
         | {x[0] for x in read_csv(f"{BASE}/splitlists/test.csv") if x})
print("\nFinal train+test clips:", len(final), "| XML files:", len(xml_base))
print("XMLs not covered by final train+test:", sorted(xml_base - final))

# %%
import os, xml.etree.ElementTree as ET
BASE = "/content/lifeguard_test"

def norm(path):
    return ET.tostring(ET.parse(path).getroot(), encoding="unicode")

pairs = [
    ("Z9npOp9WhWM.xml", "Z9npOp9WhWM_00-00-22--00-00-34_0-04--0-25--0-17.xml"),
    ("XPPOcJNk1S8.xml", "XPPOcJNk1S8_00-00-05--00-00-20_0-70--0-27--0-11.xml"),
]
for short, full in pairs:
    a, b = f"{BASE}/annotations/{short}", f"{BASE}/annotations/{full}"
    if os.path.exists(a) and os.path.exists(b):
        print(f"{short} identical to {full} ?  {norm(a) == norm(b)}")
    else:
        print(f"missing: {short} or {full}")

# %%
import os, glob, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

BASE = "/content/lifeguard_test"

def show(video, xml_path, indices):
    root = ET.parse(xml_path).getroot()
    by_id = {}
    for im in root.iter("image"):
        by_id[int(im.get("id"))] = [
            (b.get("label"), float(b.get("xtl")), float(b.get("ytl")),
             float(b.get("xbr")), float(b.get("ybr")))
            for b in im.iter("box")]
    frames = sorted(glob.glob(f"{BASE}/frames/{video}/*.jpg"),
                    key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    for i in indices:
        img = Image.open(frames[i])
        fig, ax = plt.subplots(1, figsize=(5, 5)); ax.imshow(img)
        for lab, xtl, ytl, xbr, ybr in by_id.get(i, []):
            c = "red" if lab == "drown" else "lime"
            ax.add_patch(Rectangle((xtl, ytl), xbr - xtl, ybr - ytl,
                                   fill=False, edgecolor=c, lw=1.5))
            ax.text(xtl, ytl - 3, lab, color=c, fontsize=8)
        ax.set_title(f"{video} index {i} ({os.path.basename(frames[i])})"); ax.axis("off")
        plt.show()

show("Z9npOp9WhWM", f"{BASE}/annotations/Z9npOp9WhWM_00-00-22--00-00-34_0-04--0-25--0-17.xml", [0, 179, 358])
show("XPPOcJNk1S8", f"{BASE}/annotations/XPPOcJNk1S8_00-00-05--00-00-20_0-70--0-27--0-11.xml", [0, 224, 448])

# %%
import os, glob, csv, re, random
BASE = "/content/lifeguard_test"
random.seed(42)

def video_id_of(snippet):
    m = re.match(r"^(.*?)_\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_", snippet)
    return m.group(1) if m else "?"

def read(p):
    with open(p, newline="") as f:
        return list(csv.reader(f))

# 1) canonical clips = full-snippet names only; short-named files are duplicates
xmls = sorted(glob.glob(f"{BASE}/annotations/*.xml"))
canonical = {os.path.splitext(os.path.basename(p))[0] for p in xmls
             if re.search(r"_\d{2}-\d{2}-\d{2}--", os.path.basename(p))}
short = sorted(set(os.path.splitext(os.path.basename(p))[0] for p in xmls) - canonical)
print("Canonical (full-snippet) clips:", len(canonical))
print("Short-named files to ignore (duplicates):", short)

# 2) author's split leakage
train = [r[0] for r in read(f"{BASE}/splitlists/train.csv") if r]
test  = [r[0] for r in read(f"{BASE}/splitlists/test.csv") if r]
tr_v = {video_id_of(c) for c in train}
te_v = {video_id_of(c) for c in test}
print("\nAuthor split: train clips=%d videos=%d | test clips=%d videos=%d"
      % (len(train), len(tr_v), len(test), len(te_v)))
print("Videos in BOTH train and test (leakage):", sorted(tr_v & te_v))

# 3) proposed leak-free VIDEO-level split (reproducible, seed=42)
all_videos = sorted(tr_v | te_v)
test_videos = set(random.sample(all_videos, max(1, round(len(all_videos) * 0.1))))
train_videos = set(all_videos) - test_videos
print("\nProposed leak-free split: train videos=%d, test videos=%d"
      % (len(train_videos), len(test_videos)))
print("Test video ids:", sorted(test_videos))

# %%
import os, glob, csv, re, random
import xml.etree.ElementTree as ET
BASE = "/content/lifeguard_test"

def video_id_of(snippet):
    m = re.match(r"^(.*?)_\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_", snippet)
    return m.group(1) if m else "?"

def read(p):
    with open(p, newline="") as f:
        return list(csv.reader(f))

# canonical clips only (skip the 2 short-named duplicates)
xmls = sorted(glob.glob(f"{BASE}/annotations/*.xml"))
canonical = {os.path.splitext(os.path.basename(p))[0] for p in xmls
             if re.search(r"_\d{2}-\d{2}-\d{2}--", os.path.basename(p))}

# per-clip: video id + "contains drown?"
videos = {}
for c in canonical:
    root = ET.parse(f"{BASE}/annotations/{c}.xml").getroot()
    has_drown = any(b.get("label") == "drown"
                    for im in root.iter("image") for b in im.iter("box"))
    v = video_id_of(c)
    videos.setdefault(v, {"drown": False, "clips": 0})
    videos[v]["drown"] = videos[v]["drown"] or has_drown
    videos[v]["clips"] += 1

drown_videos    = sorted(v for v, i in videos.items() if i["drown"])
swimonly_videos = sorted(v for v, i in videos.items() if not i["drown"])
print("videos:", len(videos), "| with >=1 drown clip:", len(drown_videos),
      "| swim-only:", len(swimonly_videos))

# stratified, leak-free split: hold out ~10% of EACH group for test
random.seed(42)
def holdout(group):
    n = max(1, round(len(group) * 0.1))
    return set(random.sample(group, n))

test_v = holdout(drown_videos) | holdout(swimonly_videos)
train_v = set(videos) - test_v
print("stratified split: train videos=%d, test videos=%d" % (len(train_v), len(test_v)))
print("test videos containing drown:", sorted(v for v in test_v if videos[v]["drown"]))
print("test videos swim-only:", sorted(v for v in test_v if not videos[v]["drown"]))

# %%
import os, glob, re, random
import xml.etree.ElementTree as ET
BASE = "/content/lifeguard_test"

def video_id_of(s):
    m = re.match(r"^(.*?)_\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_", s)
    return m.group(1) if m else "?"

clips = {}          # video -> list of per-clip has_drown
unmatched = []
for p in sorted(glob.glob(f"{BASE}/annotations/*.xml")):
    name = os.path.splitext(os.path.basename(p))[0]
    if not re.search(r"_\d{2}-\d{2}-\d{2}--", name):
        continue     # skip the 2 short-named duplicates
    v = video_id_of(name)
    if v == "?":
        unmatched.append(name); continue
    root = ET.parse(p).getroot()
    has_drown = any(b.get("label") == "drown"
                    for im in root.iter("image") for b in im.iter("box"))
    clips.setdefault(v, []).append(has_drown)

print("unmatched clip names (should be []):", unmatched)

random.seed(42)
drown_v = sorted(v for v, l in clips.items() if any(l))
swim_v  = sorted(v for v, l in clips.items() if not any(l))
test_v  = (set(random.sample(drown_v, round(len(drown_v) * 0.1)))
           | set(random.sample(swim_v, round(len(swim_v) * 0.1))))
train_v = set(clips) - test_v

def summary(vset):
    n = d = s = 0
    for v in vset:
        for has_drown in clips[v]:
            n += 1; d += has_drown; s += (not has_drown)
    return n, d, s

for label, vset in [("TRAIN", train_v), ("TEST", test_v)]:
    n, d, s = summary(vset)
    print(f"{label}: videos={len(vset)}  clips={n}  "
          f"drown_containing_clips={d}  swim_only_clips={s}")

# %%
# Cell 1 — Mount Drive and move the small label/metadata files (and any existing
# videos) into a persistent folder. Idempotent: skips files already present.
from google.colab import drive
import os, shutil, glob

drive.mount('/content/drive')

SRC = "/content/lifeguard_test"                # current temp location (this session)
DST = "/content/drive/MyDrive/lifeguard_test"  # persistent location

os.makedirs(f"{DST}/videos", exist_ok=True)
os.makedirs(f"{DST}/annotations", exist_ok=True)

if os.path.exists(f"{SRC}/datalist.csv") and not os.path.exists(f"{DST}/datalist.csv"):
    shutil.copy(f"{SRC}/datalist.csv", f"{DST}/datalist.csv")
    print("copied datalist.csv -> Drive")

if os.path.exists(f"{SRC}/annotations"):
    n = 0
    for p in glob.glob(f"{SRC}/annotations/*.xml"):
        dst = f"{DST}/annotations/{os.path.basename(p)}"
        if not os.path.exists(dst):
            shutil.copy(p, dst); n += 1
    print(f"copied {n} annotation XMLs -> Drive")

if os.path.exists(f"{SRC}/videos"):
    n = 0
    for p in glob.glob(f"{SRC}/videos/*"):
        dst = f"{DST}/videos/{os.path.basename(p)}"
        if not os.path.exists(dst):
            shutil.copy(p, dst); n += 1
    print(f"copied {n} existing videos -> Drive")

print("Persistent destination:", DST)

# %%
# Cell 2 — Download missing source videos directly into Google Drive.
# Reads datalist.csv + annotations from Drive, skips videos already there,
# and uses --continue so interrupted downloads resume instead of restarting.
import os, csv, glob, subprocess, sys

DST = "/content/drive/MyDrive/lifeguard_test"
VIDEO_DIR = f"{DST}/videos"
DATALIST = f"{DST}/datalist.csv"

if not os.path.exists(DATALIST):
    raise SystemExit("STOP: datalist.csv not in Drive. Run Cell 1 (or the earlier "
                     "GitHub-fetch cells) first.")

def snippet_name(video_id, start, end, x, y, width):
    start = str(start)[:-3].replace(':', '-')
    end   = str(end)[:-3].replace(':', '-')
    x = str(x)[:-2].replace('.', '-')
    y = str(y)[:-2].replace('.', '-')
    width = str(width)[:-2].replace('.', '-')
    return f"{video_id}_{start}--{end}_{x}--{y}--{width}"

needed = set()
with open(DATALIST, newline="") as f:
    for row in csv.reader(f):
        if len(row) < 7:
            continue
        video_id, start, end, x, y, width, flag = row[:7]
        if flag.strip() != "y":
            continue
        if os.path.exists(f"{DST}/annotations/{snippet_name(video_id, start, end, x, y, width)}.xml"):
            needed.add(video_id)

have = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{VIDEO_DIR}/*")}
to_download = sorted(needed - have)

print(f"source videos needed: {len(needed)} | already in Drive: {len(needed & have)} | to download: {len(to_download)}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=False)

ok, fail = [], []
for v in to_download:
    url = f"https://www.youtube.com/watch?v={v}"
    r = subprocess.run(["yt-dlp", "-f", "bv", "--no-playlist", "--continue",
                        "-o", f"{VIDEO_DIR}/%(id)s.%(ext)s", url])
    if r.returncode == 0:
        ok.append(v); print(f"OK   {v}")
    else:
        fail.append(v); print(f"FAIL {v}  ({url})")

print("\n=== Summary ===")
print("destination:", VIDEO_DIR)
print("downloaded:", len(ok))
print("failed:", len(fail))
for v in fail:
    print("  ", v)

# %%
# Cell — Retry only the 10 failed videos (does not touch the 50 already downloaded).
import os, glob, subprocess, sys

VIDEO_DIR = "/content/drive/MyDrive/lifeguard_test/videos"
FAILED = ["1WIDSrr_1lk", "1_l6-wMAvaA", "82PyW_G61MU", "ALbQbfHaRL0",
          "Bx3qXB5O2fY", "K8mSP7f4ORI", "syO6NtNR6xY", "uFT1yAicqsk",
          "usWABZrI75M", "zuZIfy4aBEY"]

# 1) upgrade yt-dlp (fixes most YouTube format/client breakage)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=False)

def reason(stderr):
    for kw in ("unavailable", "private", "removed", "region", "age", "sign in",
               "copyright", "members-only", "not available"):
        for line in stderr.splitlines():
            if kw in line.lower():
                return line.strip()
    return "no clear reason (transient network or format)"

ok, fail = [], []
for v in FAILED:
    files = glob.glob(f"{VIDEO_DIR}/{v}.*")
    big = [f for f in files if not f.endswith(".part") and os.path.getsize(f) > 1_000_000]
    if big:
        print(f"SKIP {v}  (already have {os.path.basename(big[0])})")
        continue
    # remove incomplete leftovers so they are never mistaken for completed downloads
    for f in files:
        os.remove(f); print(f"removed incomplete: {os.path.basename(f)}")

    r = subprocess.run(
        ["yt-dlp",
         "-f", "bv*[ext=mp4]/bv*[ext=webm]/bv/b",   # mp4 video -> webm video -> any video -> best
         "--no-playlist", "--continue", "--retries", "3",
         "-o", f"{VIDEO_DIR}/%(id)s.%(ext)s",
         f"https://www.youtube.com/watch?v={v}"],
        capture_output=True, text=True)

    if r.returncode == 0:
        ok.append(v); print(f"OK   {v}")
    else:
        fail.append(v); print(f"FAIL {v}  — {reason(r.stderr)}")

print("\n=== Result ===")
print("succeeded:", len(ok))
print("still failing:", len(fail))
for v in fail:
    print("  ", v)

# %%
# Cell — Diagnose the 10 failures without downloading (--simulate).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=False)

IDS = ["1WIDSrr_1lk", "1_l6-wMAvaA", "82PyW_G61MU", "ALbQbfHaRL0",
       "Bx3qXB5O2fY", "K8mSP7f4ORI", "syO6NtNR6xY", "uFT1yAicqsk",
       "usWABZrI75M", "zuZIfy4aBEY"]

for v in IDS:
    r = subprocess.run(
        ["yt-dlp", "--simulate", "--no-playlist",
         f"https://www.youtube.com/watch?v={v}"],
        capture_output=True, text=True)
    lines = (r.stderr or r.stdout).strip().splitlines()
    print(f"--- {v} (exit {r.returncode}) ---")
    for ln in lines[-3:]:
        print("   ", ln)

# %%
# Cell — Fix the 7 non-auth failures: install deno (YouTube's JS runtime), then retry.
import os, glob, subprocess, sys

VIDEO_DIR = "/content/drive/MyDrive/lifeguard_test/videos"
RETRY = ["1WIDSrr_1lk", "1_l6-wMAvaA", "82PyW_G61MU", "ALbQbfHaRL0",
         "syO6NtNR6xY", "uFT1yAicqsk", "usWABZrI75M"]   # exclude the 3 auth-gated

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=False)

# install deno (the JS runtime yt-dlp needs to decrypt YouTube formats)
subprocess.run("curl -fsSL https://deno.land/install.sh | sh", shell=True, check=False)
os.environ["PATH"] = os.path.expanduser("~/.deno/bin") + ":" + os.environ["PATH"]

def complete_file(vid):
    for f in glob.glob(f"{VIDEO_DIR}/{vid}.*"):
        if not f.endswith(".part") and os.path.getsize(f) > 1_000_000:
            return f
    return None

ok, fail = [], []
for v in RETRY:
    done = complete_file(v)
    if done:
        print(f"SKIP {v}  (already have {os.path.basename(done)})"); continue
    r = subprocess.run(
        ["yt-dlp", "-f", "bv*[ext=mp4]/bv*[ext=webm]/bv/b",
         "--no-playlist", "--continue", "--retries", "3",
         "-o", f"{VIDEO_DIR}/%(id)s.%(ext)s",
         f"https://www.youtube.com/watch?v={v}"],
        capture_output=True, text=True)
    if r.returncode == 0 and complete_file(v):
        ok.append(v); print(f"OK   {v}")
    else:
        fail.append(v); print(f"FAIL {v}")
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        if tail: print("   ", tail[-1])

print("\nsucceeded:", len(ok), "| still failing:", len(fail), fail)

# %%
# Cell — Retry the 6 remaining videos with the Android player client
# (YouTube 403-blocks the default web client from Colab's datacenter IP).
import os, glob, subprocess, sys

VIDEO_DIR = "/content/drive/MyDrive/lifeguard_test/videos"
RETRY = ["1WIDSrr_1lk", "82PyW_G61MU", "ALbQbfHaRL0",
         "syO6NtNR6xY", "uFT1yAicqsk", "usWABZrI75M"]

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=False)
subprocess.run("curl -fsSL https://deno.land/install.sh | sh", shell=True, check=False)
os.environ["PATH"] = os.path.expanduser("~/.deno/bin") + ":" + os.environ["PATH"]

def complete_file(vid):
    for f in glob.glob(f"{VIDEO_DIR}/{vid}.*"):
        if not f.endswith(".part") and os.path.getsize(f) > 1_000_000:
            return f
    return None

ok, fail = [], []
for v in RETRY:
    done = complete_file(v)
    if done:
        print(f"SKIP {v}  (already have {os.path.basename(done)})"); continue
    r = subprocess.run(
        ["yt-dlp",
         "--extractor-args", "youtube:player_client=android",
         "-f", "bv*[ext=mp4]/bv*[ext=webm]/bv/b",
         "--no-playlist", "--continue", "--retries", "3",
         "-o", f"{VIDEO_DIR}/%(id)s.%(ext)s",
         f"https://www.youtube.com/watch?v={v}"],
        capture_output=True, text=True)
    if r.returncode == 0 and complete_file(v):
        ok.append(v); print(f"OK   {v}")
    else:
        fail.append(v); print(f"FAIL {v}")
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        if tail: print("   ", tail[-1])

print("\nsucceeded:", len(ok), "| still failing:", len(fail), fail)

# %%
# Cell — Try alternate player clients for usWABZrI75M (SABR-only streaming workaround).
import os, glob, subprocess, sys

VIDEO_DIR = "/content/drive/MyDrive/lifeguard_test/videos"
VID = "usWABZrI75M"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=False)
subprocess.run("curl -fsSL https://deno.land/install.sh | sh", shell=True, check=False)
os.environ["PATH"] = os.path.expanduser("~/.deno/bin") + ":" + os.environ["PATH"]

def done():
    for f in glob.glob(f"{VIDEO_DIR}/{VID}.*"):
        if not f.endswith(".part") and os.path.getsize(f) > 1_000_000:
            return f
    return None

CLIENTS = ["tv", "mweb", "ios", "web_embedded"]

got = False
for client in CLIENTS:
    if done():
        got = True; break
    for f in glob.glob(f"{VIDEO_DIR}/{VID}*.part"):
        os.remove(f)
    r = subprocess.run(
        ["yt-dlp",
         "--extractor-args", f"youtube:player_client={client}",
         "-f", "bv*[ext=mp4]/bv*[ext=webm]/bv/b",
         "--no-playlist", "--no-continue",
         "-o", f"{VIDEO_DIR}/%(id)s.%(ext)s",
         f"https://www.youtube.com/watch?v={VID}"],
        capture_output=True, text=True)
    if r.returncode == 0 and done():
        print(f"OK {VID}  (via client {client})"); got = True; break
    else:
        print(f"FAIL via {client}")
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        if tail: print("   ", tail[-1])

if not got:
    print(f"\n{VID} could not be downloaded by any client.")

# %%
import os, csv, glob

DST = "/content/drive/MyDrive/lifeguard_test"

def snippet_name(video_id, start, end, x, y, width):
    start = str(start)[:-3].replace(':', '-')
    end   = str(end)[:-3].replace(':', '-')
    x = str(x)[:-2].replace('.', '-')
    y = str(y)[:-2].replace('.', '-')
    width = str(width)[:-2].replace('.', '-')
    return f"{video_id}_{start}--{end}_{x}--{y}--{width}"

needed = set()
with open(f"{DST}/datalist.csv", newline="") as f:
    for row in csv.reader(f):
        if len(row) < 7: continue
        video_id, start, end, x, y, width, flag = row[:7]
        if flag.strip() != "y": continue
        if os.path.exists(f"{DST}/annotations/{snippet_name(video_id, start, end, x, y, width)}.xml"):
            needed.add(video_id)

have = {os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(f"{DST}/videos/*") if not p.endswith(".part")}
print("needed:", len(needed), "| present:", len(needed & have), "| missing:", sorted(needed - have))

# %%
import os, glob

print("=== /content ===")
for name in sorted(os.listdir("/content")):
    print("  ", name)

print("\n=== .mp4 files under /content ===")
for p in sorted(glob.glob("/content/**/*.mp4", recursive=True)):
    print("  ", p)

print("\n=== .mp4 files in Drive videos/ ===")
drive_videos = "/content/drive/MyDrive/lifeguard_test/videos"
if os.path.isdir(drive_videos):
    for name in sorted(os.listdir(drive_videos)):
        print("  ", name)
else:
    print("  (not found — Drive may not be mounted)")

# %%
import os
from google.colab import files

DEST = "/content/drive/MyDrive/lifeguard_test/videos"

if not os.path.isdir("/content/drive/MyDrive"):
    print("Google Drive is not mounted. Run this first, then re-run this cell:")
    print("  from google.colab import drive")
    print("  drive.mount('/content/drive')")
else:
    os.makedirs(DEST, exist_ok=True)

    print("Select the four MP4 files to upload...")
    uploaded = files.upload()   # {filename: bytes}

    id_map = {"Bx3qXB5O2fY": "Bx3qXB5O2fY.mp4",
              "K8mSP7f4ORI": "K8mSP7f4ORI.mp4",
              "zuZIfy4aBEY": "zuZIfy4aBEY.mp4"}

    for name, data in uploaded.items():
        target = name
        for vid, newname in id_map.items():
            if vid in name:
                target = newname
                break
        dst = os.path.join(DEST, target)
        if os.path.exists(dst):
            print(f"SKIP (already exists): {dst}")
        else:
            with open(dst, "wb") as f:
                f.write(data)
            print(f"OK   {name}\n     -> {dst}  ({len(data)/1e6:.1f} MB)")

    print("\n=== Final check ===")
    expected = ["Bx3qXB5O2fY.mp4", "K8mSP7f4ORI.mp4", "zuZIfy4aBEY.mp4",
                "Wavepool Lifeguard Rescue 43 - Spot the Drowning!.mp4"]
    missing = []
    for e in expected:
        p = os.path.join(DEST, e)
        status = "PRESENT" if os.path.exists(p) else "MISSING"
        print(f"  {status}  {e}")
        if status == "MISSING":
            missing.append(e)
    if missing:
        print("\nStill missing:", missing)
    else:
        print("\nAll four files are present in", DEST)

# %%
from pathlib import Path

video_dir = Path("/content/drive/MyDrive/lifeguard_test/videos")

if not video_dir.exists():
    print("ERROR: videos folder not found")
else:
    files = sorted(video_dir.iterdir())

    print(f"Total files: {len(files)}\n")

    for i, file in enumerate(files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        status = "INCOMPLETE (.part)" if file.name.endswith(".part") else "OK"

        print(f"{i}. {file.name} | {size_mb:.2f} MB | {status}")

# %%

import os
import shutil
from google.colab import drive

# 1) Mount Google Drive
drive.mount("/content/drive")

# 2) API key: paste your NEW Roboflow key here
api_key = "1KYNQ036QaTVoQm0t706"

# 3) Destination + no-overwrite guard
DEST = "/content/drive/MyDrive/lifeguard_datasets/DrowningDetectionTracking_v1"

if os.path.exists(DEST):
    if not os.path.isdir(DEST):
        raise SystemExit(f"STOP: {DEST} exists but is not a folder.")
    if os.listdir(DEST):
        raise SystemExit(f"STOP: {DEST} already exists and is not empty.")
    os.rmdir(DEST)

# 4) Install Roboflow
# [shell] !pip -q install roboflow   # run in Colab/terminal, not standalone Python

from roboflow import Roboflow

# 5) Download Version 1 in YOLOv11 format
rf = Roboflow(api_key=api_key)
project = rf.workspace("drowningdetection-2zzxz").project(
    "drowningdetectiontracking"
)
version = project.version(1)

try:
    dataset = version.download("yolov11")
except Exception as e:
    raise SystemExit(
        f"STOP: download failed ({type(e).__name__}): {e}"
    )

# 6) Move dataset to Google Drive
os.makedirs(os.path.dirname(DEST), exist_ok=True)
shutil.move(os.path.abspath(dataset.location), DEST)

# 7) Verify the downloaded dataset
print("Dataset saved at:", DEST)
print("data.yaml:", os.path.exists(os.path.join(DEST, "data.yaml")))

for split in ("train", "valid", "test"):
    print(
        split,
        "-> images:", os.path.isdir(f"{DEST}/{split}/images"),
        "| labels:", os.path.isdir(f"{DEST}/{split}/labels")
    )

# %%
import os

DRIVE = "/content/drive/MyDrive"

print("=== Top level of MyDrive ===")
try:
    for name in sorted(os.listdir(DRIVE)):
        p = os.path.join(DRIVE, name)
        print(f"  [{'DIR ' if os.path.isdir(p) else 'FILE'}]  {name}")
except Exception as e:
    print("  ERROR reading MyDrive:", e)

def tree(path, depth=0, maxdepth=4):
    if depth > maxdepth:
        return
    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        print("  " * depth + f"(unreadable: {e})")
        return
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if not os.path.isdir(os.path.join(path, e))]
    fsize = sum(os.path.getsize(os.path.join(path, f)) for f in files)
    print("  " * depth + f"{os.path.basename(path.rstrip('/')) or path}/  "
          f"[{len(dirs)} dirs, {len(files)} files, {fsize/1e6:.1f} MB]")
    if len(files) <= 15:  # list files only when few, to keep output small
        for f in files:
            sz = os.path.getsize(os.path.join(path, f))
            size = f"{sz/1e6:.2f} MB" if sz >= 1e6 else f"{sz/1e3:.0f} KB"
            print("  " * (depth + 1) + f"{f}  ({size})")
    for d in dirs:
        tree(os.path.join(path, d), depth + 1, maxdepth)

# Scan every top-level folder whose name looks dataset-related
roots = []
for n in sorted(os.listdir(DRIVE)):
    p = os.path.join(DRIVE, n)
    if os.path.isdir(p) and any(k in n.lower() for k in ("dataset", "lifeguard", "drown", "sea", "mob")):
        roots.append(p)

for r in roots:
    print(f"\n=== {r} ===")
    tree(r)

print("\n=== Search for dataset marker files (data.yaml / dataset.yaml / *.zip / annotations*.json) ===")
for root, dirs, files in os.walk(DRIVE):
    dirs[:] = [d for d in dirs if d not in (".Trash", ".ipynb_checkpoints")]
    for f in files:
        if (f == "data.yaml" or f == "dataset.yaml" or f.endswith(".zip")
                or f.startswith("annotations") and f.endswith(".json")
                or f == "datalist.csv"):
            print("  ", os.path.join(root, f))

# %%
import os, csv, glob
from google.colab import drive
drive.mount("/content/drive")

ZT = "/content/drive/MyDrive/lifeguard_test"

def snippet_name(video_id, start, end, x, y, width):
    start = str(start)[:-3].replace(':', '-'); end = str(end)[:-3].replace(':', '-')
    x = str(x)[:-2].replace('.', '-'); y = str(y)[:-2].replace('.', '-'); width = str(width)[:-2].replace('.', '-')
    return f"{video_id}_{start}--{end}_{x}--{y}--{width}"

expected = set()
with open(f"{ZT}/datalist.csv", newline="") as f:
    for row in csv.reader(f):
        if len(row) < 7: continue
        video_id, start, end, x, y, width, flag = row[:7]
        if flag.strip() != "y": continue
        if os.path.exists(f"{ZT}/annotations/{snippet_name(video_id, start, end, x, y, width)}.xml"):
            expected.add(video_id)

present = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(f"{ZT}/videos/*") if not p.endswith(".part")}
missing = sorted(expected - present)

print("expected videos:", len(expected))
print("present videos:", len(present))
print("MISSING (", len(missing), "):", missing)

print("\n=== MyDrive top level (find the shared-folder shortcut name) ===")
for n in sorted(os.listdir("/content/drive/MyDrive")):
    print("  ", n)

# %%
SHARED = "/content/drive/MyDrive/PASTE_SHORTCUT_NAME_HERE"   # edit this

import os, glob
print("=== Shared folder contents ===")
for p in sorted(glob.glob(f"{SHARED}/**/*", recursive=True)):
    if os.path.isfile(p):
        print(f"  {os.path.relpath(p, SHARED)}  ({os.path.getsize(p)/1e6:.1f} MB)")

print("\n=== Candidate matching (missing id -> filename containing it) ===")
for mid in missing:
    hits = [os.path.basename(p) for p in glob.glob(f"{SHARED}/**/*", recursive=True)
            if os.path.isfile(p) and mid in os.path.basename(p)]
    print(f"  {mid}: {'FOUND -> ' + ', '.join(hits) if hits else 'NO unambiguous filename match'}")

# %%
CONFIRMED = []   # e.g. ["Bx3qXB5O2fY", "K8mSP7f4ORI", "zuZIfy4aBEY"]  <- only unambiguous ones

import os, shutil, glob
ZT = "/content/drive/MyDrive/lifeguard_test"
SHARED = "/content/drive/MyDrive/PASTE_SHORTCUT_NAME_HERE"

copied, skipped, failed = [], [], []
for mid in CONFIRMED:
    srcs = [p for p in glob.glob(f"{SHARED}/**/*", recursive=True)
            if os.path.isfile(p) and mid in os.path.basename(p)]
    if len(srcs) != 1:
        failed.append((mid, "ambiguous/missing source")); continue
    src = srcs[0]
    dst = f"{ZT}/videos/{mid}.mp4"
    if os.path.exists(dst):
        skipped.append((mid, "already present")); continue
    shutil.copyfile(src, dst)
    (copied if os.path.getsize(dst) > 0 else failed).append((mid, os.path.basename(src)))

print("copied:", copied)
print("skipped:", skipped)
print("failed (needs attention):", failed)

present = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(f"{ZT}/videos/*") if not p.endswith(".part")}
print("videos now:", len(present), "of 62")

# %%
import os, glob

SHARED = "/content/drive/MyDrive/missedData"

print("=== Contents of", SHARED, "===")
for p in sorted(glob.glob(f"{SHARED}/**/*", recursive=True)):
    if os.path.isfile(p):
        print(f"  {os.path.relpath(p, SHARED)}  ({os.path.getsize(p)/1e6:.1f} MB)")

missing = ["Bx3qXB5O2fY", "K8mSP7f4ORI", "usWABZrI75M", "zuZIfy4aBEY"]
print("\n=== Match missing id -> file ===")
for mid in missing:
    hits = [os.path.basename(p) for p in glob.glob(f"{SHARED}/**/*", recursive=True)
            if os.path.isfile(p) and mid in os.path.basename(p)]
    print(f"  {mid}: {'FOUND -> ' + ', '.join(hits) if hits else 'NO id in filename (needs manual check)'}")

# %%
CONFIRMED = ["Bx3qXB5O2fY", "K8mSP7f4ORI", "zuZIfy4aBEY"]   # NOT usWABZrI75M yet

import os, shutil, glob
ZT = "/content/drive/MyDrive/lifeguard_test"
SHARED = "/content/drive/MyDrive/missedData"

copied, failed = [], []
for mid in CONFIRMED:
    srcs = [p for p in glob.glob(f"{SHARED}/**/*", recursive=True)
            if os.path.isfile(p) and mid in os.path.basename(p)]
    if len(srcs) != 1:
        failed.append((mid, "ambiguous/missing")); continue
    src = srcs[0]; dst = f"{ZT}/videos/{mid}.mp4"
    if os.path.exists(dst):
        print("SKIP", mid, "(already present)"); continue
    shutil.copyfile(src, dst)
    print("OK", mid, "->", dst, f"({os.path.getsize(dst)/1e6:.1f} MB)")

present = {os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(f"{ZT}/videos/*") if not p.endswith(".part")}
print("videos now:", len(present), "of 62 | still missing:", ["usWABZrI75M"] if "usWABZrI75M" not in present else [])

# %%

from pathlib import Path

root = Path("/content/drive/MyDrive")
target = "usWABZrI75M"

print("Searching for:", target)

for path in root.rglob("*"):
    if path.is_file() and target.lower() in path.name.lower():
        print("FOUND:", path, f"({path.stat().st_size / (1024**2):.1f} MB)")

# %%

from pathlib import Path
import subprocess

video = Path(
    "/content/drive/MyDrive/lifeguard_test/videos/"
    "usWABZrI75M.mp4.part"
)

print("Exists:", video.exists())

if video.exists():
    print("Size (MB):", round(video.stat().st_size / (1024**2), 2))

    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration,size",
         "-of", "default=noprint_wrappers=1",
         str(video)],
        capture_output=True,
        text=True
    )

    print("FFprobe return code:", result.returncode)
    print("FFprobe output:")
    print(result.stdout or result.stderr or "No output")

# %%

from pathlib import Path

folder = Path("/content/drive/MyDrive/lifeguard_test/videos")
source = folder / "usWABZrI75M.mp4.part"
destination = folder / "usWABZrI75M.mp4"

if not source.exists():
    print("STOP: source file not found.")
elif destination.exists():
    print("STOP: destination already exists. No changes made.")
else:
    source.rename(destination)
    print("Renamed successfully:", destination.name)
    print("Size (MB):", round(destination.stat().st_size / (1024**2), 2))

# %%

from pathlib import Path
import subprocess

root = Path("/content/drive/MyDrive/lifeguard_test")
video_dir = root / "videos"
annotation_dir = root / "annotations"

videos = sorted(video_dir.glob("*.mp4"))
video_ids = {p.stem for p in videos}

print("=== VIDEO INVENTORY ===")
print("MP4 videos:", len(videos))
print("Expected:", 62)

print("\n=== VIDEO READABILITY ===")
bad_videos = []

for video in videos:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(video)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0 or not result.stdout.strip():
        bad_videos.append(video.name)

print("Unreadable videos:", len(bad_videos))
for name in bad_videos:
    print("  ", name)

print("\n=== ANNOTATION ID CHECK ===")
annotation_ids = {
    p.name.split("_")[0]
    for p in annotation_dir.glob("*.xml")
}

missing_annotations = sorted(video_ids - annotation_ids)
unmatched_annotations = sorted(annotation_ids - video_ids)

print("Video IDs without XML:", len(missing_annotations))
for item in missing_annotations:
    print("  ", item)

print("Annotation IDs without matching video:", len(unmatched_annotations))
for item in unmatched_annotations:
    print("  ", item)

print("\n=== FINAL STATUS ===")
if len(videos) == 62 and not bad_videos:
    print("All 62 MP4 files are readable.")
else:
    print("Needs attention: count mismatch or unreadable videos.")

# %%
import os, re, glob

ZT = "/content/drive/MyDrive/lifeguard_test"

def video_id_of(snippet):
    m = re.match(r"^(.*?)_\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_", snippet)
    return m.group(1) if m else None

vids = {os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(f"{ZT}/videos/*") if not p.endswith(".part")}

xmls = [os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(f"{ZT}/annotations/*.xml")]
canonical = [x for x in xmls if re.search(r"_\d{2}-\d{2}-\d{2}--", x)]
short = [x for x in xmls if x not in canonical]

xml_vids = {video_id_of(x) for x in canonical if video_id_of(x)}

print("videos:", len(vids))
print("canonical XMLs:", len(canonical), "| short-named duplicates:", short)
print("distinct video IDs in canonical XMLs:", len(xml_vids))
print("videos with NO annotation:", sorted(vids - xml_vids))
print("annotation IDs with NO video:", sorted(xml_vids - vids))

# %%
import os, subprocess, json, urllib.request, zipfile

# 1) Mount Drive only if not already mounted
from google.colab import drive
if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")

DEST = "/content/drive/MyDrive/lifeguard_datasets/MOBDrone/MOBDrone_videos.zip"
URL  = "https://zenodo.org/records/5996890/files/MOBDrone_videos.zip?download=1"

# 2) Expected size + md5 from the Zenodo API (verified record 5996890)
api = json.load(urllib.request.urlopen("https://zenodo.org/api/records/5996890"))
meta = next(x for x in api["files"] if x["key"] == "MOBDrone_videos.zip")
EXPECTED_SIZE = meta["size"]                 # 5,455,391,169 bytes
EXPECTED_MD5  = meta["checksum"].split(":")[-1]

os.makedirs(os.path.dirname(DEST), exist_ok=True)

# 3) Inspect existing file — never overwrite
if os.path.exists(DEST):
    size = os.path.getsize(DEST)
    if size == EXPECTED_SIZE:
        bad = zipfile.ZipFile(DEST).testzip()
        if bad is None:
            print("SKIP: complete, valid archive already present.")
            raise SystemExit(0)
        raise SystemExit(f"STOP: existing archive is corrupt (bad entry: {bad}). Delete it and re-run.")
    raise SystemExit(f"STOP: existing file is partial/suspicious ({size}/{EXPECTED_SIZE} bytes). "
                     f"Re-run to resume with wget -c, or delete the file to restart cleanly.")

# 4) Download (resumable) directly to the Drive destination
print("Downloading MOBDrone_videos.zip (resumable)...")
r = subprocess.run(["wget", "-c", "--progress=dot:giga", "-O", DEST, URL])
if r.returncode != 0:
    raise SystemExit("STOP: download failed. Re-run this cell to resume (wget -c).")

# 5) Verify size
size = os.path.getsize(DEST)
print(f"final size: {size} bytes (expected {EXPECTED_SIZE})")
if size != EXPECTED_SIZE:
    raise SystemExit("STOP: size mismatch after download.")

# 6) Verify integrity without extracting (CRC check)
print("Verifying zip integrity (testzip, no extraction)...")
bad = zipfile.ZipFile(DEST).testzip()
if bad is not None:
    raise SystemExit(f"STOP: archive corrupt at entry {bad}.")
print("OK: archive downloaded and verified.")
print("Optional md5 (slow on 5.5 GB):", EXPECTED_MD5)

# %%
