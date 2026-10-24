# Auto-generated Python copy of notebook: setupDataWithColab.ipynb
# Edits belong in the notebook; this is a faithful code export.
# Colab/notebook-only lines are commented with [shell] or [magic].

# %%

# %%
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

# %%
import os
import torch

print("Drive mounted:", os.path.ismount("/content/drive"))
print("MyDrive accessible:", os.path.isdir("/content/drive/MyDrive"))
print("GPU:", torch.cuda.get_device_name(0))

# %%
import os
import torch

print("Drive mounted:", os.path.ismount("/content/drive"))
print("MyDrive accessible:", os.path.isdir("/content/drive/MyDrive"))
print("GPU:", torch.cuda.get_device_name(0))

# %%
import os
print("Drive mounted:", os.path.ismount("/content/drive"))
print("MyDrive accessible:", os.path.isdir("/content/drive/MyDrive"))

# %%

import sys
import subprocess

video_url = "https://www.youtube.com/watch?v=Z9npOp9WhWM"

result = subprocess.run(
    [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--simulate",
        "--verbose",
        video_url
    ],
    capture_output=True,
    text=True
)

print("Return code:", result.returncode)
print("\n--- ERROR / DIAGNOSTICS ---")
print((result.stderr or result.stdout)[-6000:])

# %%

import shutil
import subprocess
import sys

print("Deno path:", shutil.which("deno"))

print("\nChecking apt package:")
subprocess.run(["bash", "-lc", "apt-cache policy deno"])

print("\nChecking yt-dlp:")
subprocess.run([sys.executable, "-m", "yt_dlp", "--version"])

# %%

# [shell] !curl -fL https://deno.land/install.sh -o /tmp/install_deno.sh   # run in Colab/terminal, not standalone Python
# [shell] !sh /tmp/install_deno.sh   # run in Colab/terminal, not standalone Python

import os
import subprocess

deno = os.path.expanduser("~/.deno/bin/deno")

if os.path.isfile(deno):
    subprocess.run([deno, "--version"], check=True)
else:
    print("Deno was not installed. Review the installation output above.")

# %%
import os
import subprocess

deno = "/root/.deno/bin/deno"

os.environ["PATH"] = os.path.dirname(deno) + ":" + os.environ["PATH"]

subprocess.run([deno, "--version"], check=True)

# %%
import subprocess
import sys

result = subprocess.run(
    [
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "deno:/root/.deno/bin/deno",
        "--simulate",
        "https://www.youtube.com/watch?v=Z9npOp9WhWM"
    ],
    capture_output=True,
    text=True
)

print("Return code:", result.returncode)
print((result.stderr or result.stdout)[-3000:])

# %%
# [magic] %pip install -U "yt-dlp[default]"   # Colab/Jupyter only

# %%
import subprocess
import sys

result = subprocess.run(
    [
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "deno:/root/.deno/bin/deno",
        "--remote-components", "ejs:github",
        "--simulate",
        "https://www.youtube.com/watch?v=Z9npOp9WhWM"
    ],
    capture_output=True,
    text=True
)

print("Return code:", result.returncode)
print("STDOUT:", result.stdout[-1500:])
print("STDERR:", result.stderr[-3000:])

# %%
import os
import subprocess
import sys

output_dir = "/content/lifeguard_test/videos"
os.makedirs(output_dir, exist_ok=True)

video_ids = [
    "Z9npOp9WhWM",
    "XPPOcJNk1S8"
]

for video_id in video_ids:
    url = f"https://www.youtube.com/watch?v={video_id}"

    result = subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "deno:/root/.deno/bin/deno",
        "--remote-components", "ejs:github",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", f"{output_dir}/%(id)s.%(ext)s",
        url
    ])

    if result.returncode != 0:
        print(f"FAILED: {video_id}")
    else:
        print(f"Download command completed: {video_id}")

print("\nFiles:", os.listdir(output_dir))

# %%
import os
import subprocess
import sys

output_dir = "/content/lifeguard_test/videos"
os.makedirs(output_dir, exist_ok=True)

video_ids = [
    "Z9npOp9WhWM",
    "XPPOcJNk1S8"
]

for video_id in video_ids:
    url = f"https://www.youtube.com/watch?v={video_id}"

    result = subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "--js-runtimes", "deno:/root/.deno/bin/deno",
        "--remote-components", "ejs:github",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", f"{output_dir}/%(id)s.%(ext)s",
        url
    ])

    if result.returncode != 0:
        print(f"FAILED: {video_id}")
    else:
        print(f"Download command completed: {video_id}")

print("\nFiles:", os.listdir(output_dir))

# %%
import os
import urllib.request
import xml.etree.ElementTree as ET

output_dir = "/content/lifeguard_test/annotations"
os.makedirs(output_dir, exist_ok=True)

xml_files = {
    "Z9npOp9WhWM": (
        "https://raw.githubusercontent.com/Z5cc/drowning-detection/master/"
        "dataset_lifeguard/labels/"
        "Z9npOp9WhWM_00-00-22--00-00-34_0-04--0-25--0-17.xml"
    ),
    "XPPOcJNk1S8": (
        "https://raw.githubusercontent.com/Z5cc/drowning-detection/master/"
        "dataset_lifeguard/labels/"
        "XPPOcJNk1S8_00-00-05--00-00-20_0-70--0-27--0-11.xml"
    )
}

for video_id, url in xml_files.items():
    xml_path = os.path.join(output_dir, f"{video_id}.xml")
    urllib.request.urlretrieve(url, xml_path)

    root = ET.parse(xml_path).getroot()
    objects = root.findall(".//object")
    classes = {}

    for obj in objects:
        name = obj.findtext("name", default="unknown")
        classes[name] = classes.get(name, 0) + 1

    print(f"\n{video_id}")
    print("XML:", xml_path)
    print("Frames:", len(root.findall(".//image")))
    print("Classes:", classes)

# %%
import xml.etree.ElementTree as ET

for video_id in ["Z9npOp9WhWM", "XPPOcJNk1S8"]:
    path = f"/content/lifeguard_test/annotations/{video_id}.xml"
    root = ET.parse(path).getroot()

    print(f"\n--- {video_id} ---")
    print("Root tag:", root.tag)
    print("Root attributes:", root.attrib)

    for child in list(root)[:5]:
        print(ET.tostring(child, encoding="unicode")[:1500])

# %%
import xml.etree.ElementTree as ET
from collections import Counter

for video_id in ["Z9npOp9WhWM", "XPPOcJNk1S8"]:
    path = f"/content/lifeguard_test/annotations/{video_id}.xml"
    root = ET.parse(path).getroot()

    classes = Counter()
    frames_with_boxes = 0

    for frame in root.findall("image"):
        boxes = frame.findall("box")

        if boxes:
            frames_with_boxes += 1

        for box in boxes:
            classes[box.get("label")] += 1

    print(f"\n{video_id}")
    print("Frames:", len(root.findall("image")))
    print("Frames with annotations:", frames_with_boxes)
    print("Boxes per class:", dict(classes))
    print("Total boxes:", sum(classes.values()))
    

# %%
import cv2
import os

video_dir = "/content/lifeguard_test/videos"

for video_id in ["Z9npOp9WhWM", "XPPOcJNk1S8"]:
    path = os.path.join(video_dir, video_id + ".mp4")
    cap = cv2.VideoCapture(path)

    print(f"\n{video_id}")
    print("Opened:", cap.isOpened())
    print("Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("Video frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    cap.release()

# %%
import urllib.request

url = (
    "https://raw.githubusercontent.com/Z5cc/drowning-detection/"
    "master/dataset_utils/LifeguardUtils.py"
)

with urllib.request.urlopen(url) as response:
    code = response.read().decode("utf-8")

lines = code.splitlines()

for i, line in enumerate(lines):
    if any(word in line.lower() for word in ["crop", "videos_and_frames"]):
        start = max(0, i - 5)
        end = min(len(lines), i + 12)

        print(f"\n--- Lines {start + 1}-{end} ---")
        for j in range(start, end):
            print(f"{j + 1}: {lines[j]}")

# %%
import os
import subprocess

video_dir = "/content/lifeguard_test/videos"
output_dir = "/content/lifeguard_test/frames"
os.makedirs(output_dir, exist_ok=True)

clips = {
    "Z9npOp9WhWM": {
        "start": "00:00:22.73",
        "end": "00:00:34.68",
        "x": 0.0460,
        "y": 0.2514,
        "width": 0.1771,
    },
    "XPPOcJNk1S8": {
        "start": "00:00:05.32",
        "end": "00:00:20.30",
        "x": 0.7065,
        "y": 0.2738,
        "width": 0.1138,
    },
}

video_width = 1280
video_height = 720

for video_id, clip in clips.items():
    crop_size = round(clip["width"] * video_height)
    x = round(clip["x"] * video_width)
    y = round(clip["y"] * video_height)

    frame_dir = os.path.join(output_dir, video_id)
    os.makedirs(frame_dir, exist_ok=True)

    video_path = os.path.join(video_dir, video_id + ".mp4")
    frame_pattern = os.path.join(frame_dir, "%06d.jpg")

    command = [
        "ffmpeg", "-y",
        "-ss", clip["start"],
        "-to", clip["end"],
        "-i", video_path,
        "-vf",
        f"crop={crop_size}:{crop_size}:{x}:{y},scale=224:224,fps=30",
        "-q:v", "1",
        frame_pattern,
    ]

    subprocess.run(command, check=True)

    count = len([
        name for name in os.listdir(frame_dir)
        if name.endswith(".jpg")
    ])

    print(video_id, "frames extracted:", count)

# %%
import urllib.request

url = (
    "https://raw.githubusercontent.com/Z5cc/drowning-detection/"
    "master/dataset_utils/LifeguardUtils.py"
)

code = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

for i in range(68, 103):
    print(f"{i + 1}: {code[i]}")

# %%
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path

video_id = "XPPOcJNk1S8"
frame_id = 0

image_path = Path(
    f"/content/lifeguard_test/frames/{video_id}/{frame_id + 1:06d}.jpg"
)
xml_path = Path(
    f"/content/lifeguard_test/annotations/{video_id}.xml"
)

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

root = ET.parse(xml_path).getroot()
frame = root.find(f"image[@id='{frame_id}']")

for box in frame.findall("box"):
    x1 = int(float(box.get("xtl")))
    y1 = int(float(box.get("ytl")))
    x2 = int(float(box.get("xbr")))
    y2 = int(float(box.get("ybr")))

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.putText(
        image, box.get("label"),
        (x1, max(12, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
    )

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.show()

# %%
# ===== NOT EXECUTED — plan only; run only after your approval =====
import subprocess, sys, os, glob, urllib.request

OUT = "/content/lifeguard_test"
for d in ("source", "clips", "frames", "labels"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=False)
if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
    subprocess.run(["apt-get", "-y", "install", "ffmpeg"], check=False)

CLIPS = [
    dict(name="Z9npOp9WhWM",
         url="https://www.youtube.com/watch?v=Z9npOp9WhWM",
         start="00:00:22.73", end="00:00:34.68",
         x=0.0460, y=0.2514, w=0.1771,
         xml="https://raw.githubusercontent.com/Z5cc/drowning-detection/master/dataset_lifeguard/labels/Z9npOp9WhWM_00-00-22--00-00-34_0-04--0-25--0-17.xml"),
    dict(name="XPPOcJNk1S8",
         url="https://www.youtube.com/watch?v=XPPOcJNk1S8",
         start="00:00:05.32", end="00:00:20.30",
         x=0.7065, y=0.2738, w=0.1138,
         xml="https://raw.githubusercontent.com/Z5cc/drowning-detection/master/dataset_lifeguard/labels/XPPOcJNk1S8_00-00-05--00-00-20_0-70--0-27--0-11.xml"),
]

for c in CLIPS:
    r = subprocess.run(["yt-dlp", "-f", "bv", "--no-playlist",
                        "-o", f"{OUT}/source/{c['name']}.%(ext)s", c["url"]])
    if r.returncode != 0:
        print(f"STOP: could not download {c['name']} ({c['url']}). "
              "Unavailable, region-locked, or requires sign-in. Nothing else was downloaded.")
        sys.exit(1)

    vfile = glob.glob(f"{OUT}/source/{c['name']}.*")[0]
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", vfile],
                           capture_output=True, text=True)
    vw, vh = (int(v) for v in probe.stdout.strip().split(","))
    width_px = min(int(round(c["w"] * vh)), vw, vh)
    x_px, y_px = int(c["x"] * vw), int(c["y"] * vh)
    vf = f"crop={width_px}:{width_px}:{x_px}:{y_px},scale=224:224,fps=30"

    subprocess.run(["ffmpeg", "-y", "-i", vfile, "-ss", c["start"], "-to", c["end"],
                    "-vf", vf, "-preset", "veryslow", "-crf", "17", "-b:v", "0",
                    f"{OUT}/clips/{c['name']}.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", vfile, "-ss", c["start"], "-to", c["end"],
                    "-vf", vf, "-q:v", "1", "-start_number", "0",
                    f"{OUT}/frames/{c['name']}/%06d.jpg"], check=True)
    urllib.request.urlretrieve(c["xml"], f"{OUT}/labels/{c['name']}.xml")
    print(f"OK: {c['name']} -> clips/, frames/, labels/{c['name']}.xml")

print("Done. Exactly two clips staged; no channel-wide or full-dataset download.")

# %%
