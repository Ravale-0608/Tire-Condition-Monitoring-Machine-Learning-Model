"""
Trains a YOLOv8 classification model on the unified tire dataset.

Default: yolov8s-cls (small — good accuracy/speed balance)
Options:
  python train_yolov8.py              # small model (default)
  python train_yolov8.py --model m    # medium model (best accuracy)
  python train_yolov8.py --model n    # nano (fastest)
  python train_yolov8.py --resume     # resume last run
"""

import sys
from pathlib import Path
from ultralytics import YOLO

DATA_ROOT = Path(__file__).parent / "data" / "unified"
RUNS_DIR  = Path(__file__).parent / "runs"

# ── Args ──────────────────────────────────────────────────────────────────────
args   = sys.argv[1:]
RESUME = "--resume" in args
SIZE   = "s"   # default: small (was nano 'n')
for a in args:
    if a == "--model" and args.index(a) + 1 < len(args):
        SIZE = args[args.index(a) + 1]
    elif a.startswith("--model="):
        SIZE = a.split("=")[1]

MODEL_NAME = f"yolov8{SIZE}-cls.pt"

# ── Training config ────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "data":          str(DATA_ROOT),
    "epochs":        100,          # more epochs with early stopping safety net
    "imgsz":         320,          # larger than 224 — better feature extraction
    "batch":         32,
    "workers":       4,
    "patience":      15,           # more patience before early stop
    "optimizer":     "AdamW",
    "lr0":           5e-4,         # lower initial LR — more stable with larger model
    "lrf":           0.005,        # final LR = lr0 * lrf = 2.5e-6
    "warmup_epochs": 5,
    "cos_lr":        True,
    "weight_decay":  5e-4,
    "dropout":       0.2,          # regularisation to reduce overfitting
    "label_smoothing": 0.1,        # softens hard labels — improves generalisation
    "augment":       True,
    "hsv_h":         0.015,        # hue augmentation
    "hsv_s":         0.7,          # saturation augmentation
    "hsv_v":         0.4,          # brightness augmentation
    "degrees":       15.0,         # rotation
    "fliplr":        0.5,
    "flipud":        0.1,
    "project":       str(RUNS_DIR),
    "name":          "tire_cls",
    "exist_ok":      True,
    "verbose":       True,
}


def evaluate_holdout(model_path: Path):
    """Evaluate on the completely unseen holdout test set."""
    holdout = DATA_ROOT / "holdout"
    if not holdout.exists():
        print("No holdout set found — skipping.")
        return

    model = YOLO(str(model_path))
    print("\nEvaluating on holdout test set (unseen data)...")
    correct, total = 0, 0
    per_class = {}

    for cls_dir in holdout.iterdir():
        if not cls_dir.is_dir():
            continue
        images = list(cls_dir.glob("*.*"))
        cls_correct = 0
        for img in images:
            try:
                res   = model(str(img), verbose=False)
                pred  = model.names[int(res[0].probs.top1)]
                if pred == cls_dir.name:
                    cls_correct += 1
                total += 1
            except Exception:
                pass
        correct += cls_correct
        per_class[cls_dir.name] = (cls_correct, len(images))

    print(f"\nHoldout accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    for cls, (c, t) in per_class.items():
        print(f"  {cls:<14} {c}/{t}  ({c/t*100:.0f}%)")


def main():
    if not DATA_ROOT.exists():
        print("ERROR: data/unified/ not found — run build_dataset.py first.")
        sys.exit(1)

    classes = sorted(d.name for d in (DATA_ROOT / "train").iterdir() if d.is_dir())
    train_n = sum(len(list((DATA_ROOT / "train" / c).iterdir())) for c in classes)
    val_n   = sum(len(list((DATA_ROOT / "val"   / c).iterdir())) for c in classes)

    print(f"\nDataset : {DATA_ROOT}")
    print(f"Classes : {classes}")
    print(f"Train   : {train_n:,} images")
    print(f"Val     : {val_n:,} images")
    print(f"Model   : {MODEL_NAME}")
    print(f"Epochs  : {TRAIN_CONFIG['epochs']}  (patience={TRAIN_CONFIG['patience']})")
    print(f"Image   : {TRAIN_CONFIG['imgsz']}px")
    print(f"Label smoothing: {TRAIN_CONFIG['label_smoothing']}")
    print(f"Dropout : {TRAIN_CONFIG['dropout']}\n")

    if RESUME:
        last = RUNS_DIR / "tire_cls" / "weights" / "last.pt"
        if last.exists():
            print(f"Resuming from {last}")
            YOLO(str(last)).train(resume=True)
            return
        print("No checkpoint found — starting fresh.")

    model = YOLO(MODEL_NAME)
    model.train(**TRAIN_CONFIG)

    best = RUNS_DIR / "tire_cls" / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    # Standard val split evaluation
    print("\nRunning val/test set evaluation...")
    m = YOLO(str(best))
    metrics = m.val(data=str(DATA_ROOT), split="test")
    print(f"Test top-1 : {metrics.top1:.3f}")
    print(f"Test top-5 : {metrics.top5:.3f}")

    # Holdout evaluation
    evaluate_holdout(best)


if __name__ == "__main__":
    main()
