# Auto-generated Python copy of notebook: z5cctrain.ipynb
# Edits belong in the notebook; this is a faithful code export.
# Colab/notebook-only lines are commented with [shell] or [magic].

# %%
import torch, subprocess
print("torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB): %.1f" % (p.total_memory / 1e9))
else:
    print("NO GPU — training will be CPU-only (very slow; not recommended).")
# Compute Units are shown in the Colab top-right balance, not via Python.
print("Check Colab's compute-unit balance in the UI before committing.")

# %% [markdown]
# Cell 1 — safe copy + local YAML + path/box-size verification

# %%
import os, glob, shutil, yaml, statistics

SRC = "/content/drive/MyDrive/lifeguard_datasets/Z5cc_v1"
DST = "/content/Z5cc_v1_local"

# 1) refuse to copy into a non-empty destination
if os.path.isdir(DST) and os.listdir(DST):
    raise SystemExit(f"STOP: {DST} exists and is not empty. Remove it or use a new name.")
shutil.copytree(SRC, DST)

# 2) write a LOCAL yaml (does NOT touch the original)
cfg = yaml.safe_load(open(f"{DST}/data.yaml"))
cfg["path"] = DST
yaml.safe_dump(cfg, open(f"{DST}/data_local.yaml", "w"))
print("local yaml:", f"{DST}/data_local.yaml")

# 3) verify every path resolves
for split in ("train", "val", "test"):
    imgs = f"{DST}/{split}/images"; lbls = f"{DST}/{split}/labels"
    ni, nl = len(glob.glob(f"{imgs}/*.jpg")), len(glob.glob(f"{lbls}/*.txt"))
    assert os.path.isdir(imgs) and os.path.isdir(lbls), f"missing dir for {split}"
    print(f"{split}: images={ni} labels={nl}")

# 4) box-size distribution (justifies imgsz)
ws, hs = [], []
for split in ("train", "val", "test"):
    for p in glob.glob(f"{DST}/{split}/labels/*.txt"):
        for ln in open(p).read().splitlines():
            if ln:
                _, _, _, w, h = map(float, ln.split())
                ws.append(w * 224); hs.append(h * 224)
print(f"boxes={len(ws)} | median WxH = {statistics.median(ws):.1f}x{statistics.median(hs):.1f} px "
      f"| max = {max(ws):.1f}x{max(hs):.1f} px")
print("-> if median >= 24px use imgsz=224; else consider 320.")

# %% [markdown]
# Cell 2 — 1-epoch probe (measure throughput, then compute estimate)

# %%
import time
from ultralytics import YOLO

m = YOLO("yolo11n.pt")
t = time.time()
m.train(data="/content/Z5cc_v1_local/data_local.yaml", epochs=1, imgsz=224, batch=32,
        device=0, project="/content/probe_runs", name="probe")   # throwaway, local only
sec = time.time() - t
rate = 12.07   # your reported CU/hour — confirm in the Colab UI
print(f"1 epoch = {sec:.0f}s = {sec/60:.2f} min")
print(f"50 epochs ≈ {sec*50/60:.1f} min ≈ {sec*50/3600*rate:.1f} CU at {rate} CU/hr")

# %% [markdown]
# Cell 3 — full training (unique name, Drive-persistent, resumable)

# %%
from ultralytics import YOLO

DATA   = "/content/Z5cc_v1_local/data_local.yaml"
PROJ   = "/content/drive/MyDrive/lifeguard_datasets/Z5cc_runs"
NAME   = "z5cc_yolo11n_detect_v1"

model = YOLO("yolo11n.pt")
model.train(data=DATA, epochs=50, patience=15, imgsz=224, batch=32,
            device=0, seed=42, deterministic=True, pretrained=True,
            optimizer="auto", close_mosaic=10,
            project=PROJ, name=NAME, exist_ok=False,   # refuses to overwrite
            val=True, split="val")

# %% [markdown]
# Cell 4 — resume after disconnect (only if interrupted)

# %%
from ultralytics import YOLO
# resume from last.pt; if it's corrupt from a mid-write disconnect, use best.pt instead
model = YOLO("/content/drive/MyDrive/lifeguard_datasets/Z5cc_runs/z5cc_yolo11n_detect_v1/weights/last.pt")
model.train(resume=True)

# %% [markdown]
# Cell 5 — final evaluation on the untouched test split

# %%
from ultralytics import YOLO
model = YOLO("/content/drive/MyDrive/lifeguard_datasets/Z5cc_runs/z5cc_yolo11n_detect_v1/weights/best.pt")
model.val(data="/content/Z5cc_v1_local/data_local.yaml", imgsz=224, device=0, split="test")
