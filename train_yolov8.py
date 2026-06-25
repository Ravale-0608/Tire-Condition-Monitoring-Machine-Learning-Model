"""
Trains a YOLOv8 classification model on the unified tire dataset.

Model: yolov8n-cls (nano — fast to train, good baseline)
       swap to yolov8s-cls / yolov8m-cls for better accuracy

Run:
  python train_yolov8.py              # train from scratch
  python train_yolov8.py --resume     # resume last run
  python train_yolov8.py --model m    # use medium model
"""

import sys
from pathlib import Path

from ultralytics import YOLO

DATA_ROOT   = Path(__file__).parent / "data" / "unified"
RUNS_DIR    = Path(__file__).parent / "runs"

# Parse args
args = sys.argv[1:]
RESUME = "--resume" in args
SIZE   = "n"
for a in args:
    if a.startswith("--model"):
        SIZE = a.split("=")[-1] if "=" in a else (args[args.index(a) + 1] if a == "--model" else "n")

MODEL_NAME = f"yolov8{SIZE}-cls.pt"

TRAIN_CONFIG = {
    "data":       str(DATA_ROOT),
    "epochs":     50,
    "imgsz":      224,
    "batch":      32,
    "workers":    4,
    "patience":   10,          # early stopping
    "optimizer":  "AdamW",
    "lr0":        1e-3,
    "lrf":        0.01,        # final lr = lr0 * lrf
    "warmup_epochs": 3,
    "cos_lr":     True,
    "augment":    True,        # built-in YOLOv8 augmentation
    "project":    str(RUNS_DIR),
    "name":       "tire_cls",
    "exist_ok":   True,
    "verbose":    True,
}

# Error persists
def main():
    if not DATA_ROOT.exists():
        print("ERROR: data/unified/ not found. Run build_dataset.py first.")
        sys.exit(1)

    # Count classes
    classes = [d.name for d in (DATA_ROOT / "train").iterdir() if d.is_dir()]
    print(f"Dataset : {DATA_ROOT}")
    print(f"Classes : {sorted(classes)}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Epochs  : {TRAIN_CONFIG['epochs']}")
    print(f"Image   : {TRAIN_CONFIG['imgsz']}px\n")

    if RESUME:
        last = RUNS_DIR / "tire_cls" / "weights" / "last.pt"
        if not last.exists():
            print(f"No checkpoint found at {last} — starting fresh.")
            RESUME_FLAG = False
        else:
            print(f"Resuming from {last}")
            model = YOLO(str(last))
            model.train(resume=True)
            return

    model = YOLO(MODEL_NAME)
    results = model.train(**TRAIN_CONFIG)

    print("\nTraining complete.")
    best = RUNS_DIR / "tire_cls" / "weights" / "best.pt"
    print(f"Best weights : {best}")

    # Quick validation on test set
    print("\nRunning test-set evaluation ...")
    model = YOLO(str(best))
    metrics = model.val(data=str(DATA_ROOT), split="test")
    print(f"Top-1 accuracy : {metrics.top1:.3f}")
    print(f"Top-5 accuracy : {metrics.top5:.3f}")


if __name__ == "__main__":
    main()
