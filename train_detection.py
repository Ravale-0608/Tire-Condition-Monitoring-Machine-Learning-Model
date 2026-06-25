"""
Trains YOLOv8n detection model to locate tires and classify tread condition.
Uses existing annotated tread dataset (BAD_Tyres / BALD_Tyres / NORMAL_Tyres).

The detection boxes are used by the app to draw bounding boxes.
The classification model then runs on the cropped region for the full 6-class condition.

Run:  python train_detection.py
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "data" / "data.yaml"
RUNS_DIR  = Path(__file__).parent / "runs"

model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=40,
    imgsz=640,
    batch=16,
    workers=4,
    patience=10,
    optimizer="AdamW",
    lr0=1e-3,
    cos_lr=True,
    augment=True,
    project=str(RUNS_DIR),
    name="tire_det",
    exist_ok=True,
    verbose=True,
)

print("\nDetection training complete.")
print(f"Weights: {RUNS_DIR / 'tire_det' / 'weights' / 'best.pt'}")
