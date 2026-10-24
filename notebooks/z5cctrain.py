"""
Z5cc YOLO11n training script (Google Colab GPU).

Trains a YOLO11n object detector on the Z5cc swimming-pool dataset
(labels: swim = 0, drown = 1) and evaluates it on the held-out test split.

Usage in Colab:
  1. Mount Google Drive:
       from google.colab import drive
       drive.mount('/content/drive')
  2. Set the ENABLE_* flags below to True for the stages you want.
  3. Run the script.

All stages are disabled by default. This script never installs packages,
never modifies the dataset or its YAML, and refuses to overwrite an
existing run folder.
"""

import glob
import os
import shutil
import time

import torch
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration (edit this block)
# ---------------------------------------------------------------------------
DATASET_DRIVE = "/content/drive/MyDrive/lifeguard_datasets/Z5cc_v1"   # original (read-only)
DATASET_LOCAL = "/content/Z5cc_v1_local"   # optional fast copy on the Colab SSD

MODEL_WEIGHTS = "yolo11n.pt"    # COCO-pretrained nano model
EPOCHS = 50
IMSZ = 224                      # native Z5cc resolution (224x224 images)
BATCH = 32                      # lower to 16 if you run out of GPU memory
PATIENCE = 15                   # early-stop after this many epochs without improvement
SEED = 42

OUTPUT_DIR = "/content/drive/MyDrive/lifeguard_datasets/Z5cc_runs"
RUN_NAME = "z5cc_yolo11n_detect_v2"    # unique name per experiment

RESUME_CHECKPOINT = f"{OUTPUT_DIR}/{RUN_NAME}/weights/last.pt"

# Which stages to run (all False by default).
ENABLE_COPY = False     # copy the dataset to the local SSD
ENABLE_PROBE = False    # one-epoch timing probe
ENABLE_TRAIN = True    # full training
ENABLE_RESUME = False   # resume from RESUME_CHECKPOINT
ENABLE_TEST = False     # final evaluation on the test split

# Class mapping we expect in data.yaml (verified before any training).
EXPECTED_CLASSES = {0: "swim", 1: "drown"}
# ---------------------------------------------------------------------------


def check_gpu():
    """Print GPU info and return the device to use (0 = GPU, 'cpu' otherwise)."""
    if not torch.cuda.is_available():
        print("No GPU found. Training would run on CPU and be very slow.")
        return "cpu"
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB): %.1f" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    return 0


def get_data_yaml():
    """Return (data_yaml_path, dataset_root), copying to the local SSD if enabled."""
    if not ENABLE_COPY:
        return os.path.join(DATASET_DRIVE, "data.yaml"), DATASET_DRIVE

    if os.path.isdir(DATASET_LOCAL) and os.listdir(DATASET_LOCAL):
        raise SystemExit(f"Abort: {DATASET_LOCAL} already exists and is not empty.")

    shutil.copytree(DATASET_DRIVE, DATASET_LOCAL)
    # Write a local YAML pointing at the copy (the original YAML is untouched).
    cfg = yaml.safe_load(open(os.path.join(DATASET_LOCAL, "data.yaml")))
    cfg["path"] = DATASET_LOCAL
    local_yaml = os.path.join(DATASET_LOCAL, "data_local.yaml")
    yaml.safe_dump(cfg, open(local_yaml, "w"))
    print("Copied dataset to", DATASET_LOCAL)
    return local_yaml, DATASET_LOCAL


def validate_dataset(data_yaml, root):
    """Verify class names and that every split has images and labels."""
    if not os.path.exists(data_yaml):
        raise SystemExit(f"Abort: missing {data_yaml}")
    cfg = yaml.safe_load(open(data_yaml))

    if cfg.get("names") != EXPECTED_CLASSES:
        raise SystemExit(f"Abort: class mapping mismatch: {cfg.get('names')}")

    base = cfg.get("path", root)
    for split in ("train", "val", "test"):
        images_dir = os.path.join(base, cfg[split])
        labels_dir = images_dir.replace("images", "labels")
        n_images = len(glob.glob(os.path.join(images_dir, "*.jpg")))
        n_labels = len(glob.glob(os.path.join(labels_dir, "*.txt")))
        print(f"  {split}: {n_images} images / {n_labels} labels")
        if n_images == 0 or n_labels == 0:
            raise SystemExit(f"Abort: {split} is empty; check the dataset paths.")

    # Sanity check for the image size: median box width in pixels.
    train_labels_dir = os.path.join(base, cfg["train"]).replace("images", "labels")
    widths = []
    for path in glob.glob(os.path.join(train_labels_dir, "*.txt"))[:300]:
        for line in open(path).read().splitlines():
            if line:
                widths.append(float(line.split()[3]) * IMSZ)
    if widths:
        median = sorted(widths)[len(widths) // 2]
        verdict = "OK" if median >= 24 else "consider imgsz=320 for small objects"
        print(f"  median box width ~{median:.0f} px @ imgsz={IMSZ} ({verdict})")


def probe(data_yaml, device):
    """Run one epoch and report the estimated time for the full run."""
    model = YOLO(MODEL_WEIGHTS)
    start = time.time()
    model.train(data=data_yaml, epochs=1, imgsz=IMSZ, batch=BATCH, device=device,
                project="/content/probe_runs", name="probe")
    seconds = time.time() - start
    print(f"1 epoch = {seconds:.0f}s ({seconds / 60:.2f} min)")
    print(f"{EPOCHS} epochs would take about {seconds * EPOCHS / 60:.1f} min")


def train(data_yaml, device):
    """Run the full training. Refuses to overwrite an existing run folder."""
    model = YOLO(MODEL_WEIGHTS)
    model.train(data=data_yaml, epochs=EPOCHS, patience=PATIENCE, imgsz=IMSZ,
                batch=BATCH, device=device, seed=SEED, deterministic=True,
                pretrained=True, optimizer="auto", close_mosaic=10,
                project=OUTPUT_DIR, name=RUN_NAME, exist_ok=False,
                val=True, split="val")


def resume(data_yaml, device):
    """Resume training from RESUME_CHECKPOINT (last.pt)."""
    if not os.path.exists(RESUME_CHECKPOINT):
        raise SystemExit(f"Abort: checkpoint not found: {RESUME_CHECKPOINT}")
    model = YOLO(RESUME_CHECKPOINT)
    model.train(data=data_yaml, resume=True, device=device,
                project=OUTPUT_DIR, name=RUN_NAME)


def report_metrics(metrics, names, split):
    """Print per-class precision, recall, mAP50 and mAP50-95."""
    print(f"\n=== {split} per-class metrics ===")
    for index, class_id in enumerate(map(int, metrics.box.ap_class_index)):
        name = names.get(class_id, str(class_id))
        print(f"  {name:<20s} P={metrics.box.p[index]:.4f} R={metrics.box.r[index]:.4f} "
              f"mAP50={metrics.box.ap50[index]:.4f} mAP50-95={metrics.box.ap[index]:.4f}")


def evaluate(data_yaml, checkpoint, device, split):
    """Evaluate a checkpoint on a split and print per-class metrics."""
    if not os.path.exists(checkpoint):
        raise SystemExit(f"Abort: checkpoint not found: {checkpoint}")
    model = YOLO(checkpoint)
    metrics = model.val(data=data_yaml, imgsz=IMSZ, device=device, split=split)
    report_metrics(metrics, EXPECTED_CLASSES, split)


def main():
    device = check_gpu()

    print("\nDataset validation")
    data_yaml, root = get_data_yaml()
    validate_dataset(data_yaml, root)

    if ENABLE_PROBE:
        print("\nOne-epoch probe")
        probe(data_yaml, device)
    else:
        print("\n[Skipped] probe (ENABLE_PROBE is False)")

    if ENABLE_TRAIN:
        print("\nTraining")
        train(data_yaml, device)
    else:
        print("[Skipped] training (ENABLE_TRAIN is False)")

    if ENABLE_RESUME:
        print("\nResume")
        resume(data_yaml, device)
    else:
        print("[Skipped] resume (ENABLE_RESUME is False)")

    if ENABLE_TEST:
        print("\nTest evaluation")
        best_checkpoint = os.path.join(OUTPUT_DIR, RUN_NAME, "weights", "best.pt")
        evaluate(data_yaml, best_checkpoint, device, split="test")
    else:
        print("[Skipped] test evaluation (ENABLE_TEST is False)")


if __name__ == "__main__":
    main()