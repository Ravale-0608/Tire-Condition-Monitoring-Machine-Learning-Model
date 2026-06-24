"""
Augments underrepresented classes to reach a target count.

Targets (based on cleaning report):
  Tyre_Condition/UNUSABLE  :  73 → 300  (needs +227)
  flat.class               : 300 → 300  (already ok, skip)
  no-tire.class            : 300 → 300  (already ok, skip)

Augmentation pipeline (Albumentations):
  - Random horizontal/vertical flip
  - Random rotation ±20°
  - Brightness + contrast jitter
  - Gaussian blur (mild)
  - JPEG compression noise

Output images are saved alongside originals with suffix _aug{N}.jpg

Run:  python augment_minority.py
      python augment_minority.py --dry-run
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DRY_RUN = "--dry-run" in sys.argv

try:
    import albumentations as A
    HAS_ALBUMENTATION = True
except ImportError:
    HAS_ALBUMENTATION = False

DATA_ROOT = Path(__file__).parent / "data"

# (folder, current_count, target_count)
TARGETS = [
    (DATA_ROOT / "flat.class",                          99,   300),
    (DATA_ROOT / "Tyre_Condition_Dataset" / "NEW",      30,   300),
    (DATA_ROOT / "no-tire.class",                       235,  300),
    (DATA_ROOT / "full.class",                          67,   300),
    (DATA_ROOT / "Tyre_Condition_Dataset" / "SERVICEABLE", 87, 300),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_pipeline():
    if HAS_ALBUMENTATION:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=20, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        ])
    return None


def augment_pil(img_bgr, pipeline):
    """Apply augmentation and return new BGR image."""
    if pipeline:
        result = pipeline(image=img_bgr)
        return result["image"]
    # Fallback: simple flip + brightness with PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    if random.random() > 0.5:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    factor = random.uniform(0.7, 1.3)
    from PIL import ImageEnhance
    pil = ImageEnhance.Brightness(pil).enhance(factor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def augment_folder(folder: Path, current: int, target: int, pipeline):
    needed = target - current
    if needed <= 0:
        print(f"  {folder.name}: already at {current} — skipping.")
        return

    sources = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS
               and "_aug" not in f.stem]
    if not sources:
        print(f"  {folder.name}: no source images found — skipping.")
        return

    print(f"  {folder.name}: {current} -> {target}  (generating {needed} images)", flush=True)

    if DRY_RUN:
        print(f"    [DRY] would generate {needed} augmented images")
        return

    generated = 0
    cycle = 0
    while generated < needed:
        src = sources[generated % len(sources)]
        cycle_num = generated // len(sources)

        img = cv2.imread(str(src))
        if img is None:
            generated += 1
            continue

        aug = augment_pil(img, pipeline)
        out_name = f"{src.stem}_aug{cycle_num}_{generated}{src.suffix}"
        out_path = folder / out_name
        cv2.imwrite(str(out_path), aug)
        generated += 1

        if generated % 50 == 0:
            print(f"    {generated}/{needed}", flush=True)

    print(f"    Done — {generated} images written.", flush=True)


def main():
    if DRY_RUN:
        print("DRY RUN — no files will be written.\n")

    pipeline = build_pipeline()
    if HAS_ALBUMENTATION:
        print("Using Albumentations augmentation pipeline.")
    else:
        print("Albumentations not found — using basic PIL fallback.")

    print(f"\nAugmenting minority classes...\n")

    for folder, current, target in TARGETS:
        if not folder.exists():
            print(f"  {folder} — not found, skipping.")
            continue
        augment_folder(folder, current, target, pipeline)

    print("\nAugmentation complete.")
    print("Re-run clean_data.py afterwards to update the cleaning report.")


if __name__ == "__main__":
    main()
