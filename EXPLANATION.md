# Technical Explanation — Tire Condition Monitoring System

## 1. Problem Statement

Tire condition is a critical safety factor in vehicle operation. Flat tires, severely worn tread, and structural defects cause accidents and increase fuel consumption. Manual inspection is subjective, inconsistent, and requires trained mechanics. This project builds a computer vision system that classifies tire condition from a camera image in real time, making inspection fast and accessible to anyone with a smartphone.

---

## 2. Dataset Overview

Four independent datasets were collected, each with different label schemas:

| Dataset | Images | Original Labels |
|---|---|---|
| Roboflow Tread (YOLO) | 10,353 | BAD_Tyres, BALD_Tyres, NORMAL_Tyres |
| Tyre Condition Dataset | 547 | NEW, SERVICEABLE, UNUSABLE |
| Defective/Good | 1,856 | defective, good |
| Flat/Full/No-tire | 900 | flat, full, no_tire |
| **Total** | **13,656** | — |

The Roboflow dataset is the largest and comes with YOLO bounding-box annotations, making it suitable for object detection. The others are image-level classification datasets.

---

## 3. Data Cleaning Pipeline (`clean_data.py`)

Before training, every image is audited for quality issues. The pipeline runs in three phases:

### Phase 1 — Per-image quality checks (parallelised with ThreadPoolExecutor)
For each image:
- **Corruption check**: attempts `PIL.Image.load()` and `cv2.imdecode`. Any file that fails is flagged as corrupt.
- **Resolution filter**: images below 64×64 pixels are flagged as low-resolution.
- **Blur detection**: computes the variance of the Laplacian of the grayscale image. Low variance (below threshold 80) indicates a blurry image with insufficient edge information for the model.
- **Perceptual hash (pHash)**: computes a 64-bit fingerprint of the visual content using the DCT of a downsampled image. Used for duplicate detection.
- **MD5 hash**: used for exact byte-level duplicate detection.

### Phase 2 — Duplicate detection
- **Exact duplicates**: images with identical MD5 hashes are flagged.
- **Near-duplicates**: pHashes are converted to 64-bit numpy arrays. Pairwise Hamming distances are computed via matrix operations `D = rowsums[:, None] + rowsums[None, :] - 2 * (bits @ bits.T)`, which runs in C and avoids Python loops. Images with Hamming distance ≤ 8 are near-duplicates (same image, different compression or minor edits).

### Phase 3 — Output
A CSV report (`data/cleaning_report.csv`) is produced with every image, its quality flags, and hash values. No files are deleted automatically — the quarantine script handles that separately.

**Results on this dataset:**
- 0 corrupt images
- 0 low-resolution images
- 3,139 blurry images (23%)
- 4,760 near-duplicates (35%) — mostly from Roboflow's augmented exports

---

## 4. Label Unification (`build_dataset.py`)

The four datasets use incompatible label vocabularies. A unified 6-class taxonomy was designed based on the safety and actionability of each condition:

| Unified Class | Maps From | Rationale |
|---|---|---|
| `no_tire` | no-tire.class | Handles empty frames gracefully |
| `flat` | flat.class | Safety-critical — immediate action needed |
| `defective` | defective/, UNUSABLE | Structural damage beyond wear |
| `worn` | BAD_Tyres, BALD_Tyres | Tread depth issue, degraded grip |
| `good` | good/, NORMAL_Tyres, SERVICEABLE, full.class | Safe for use |
| `new` | NEW | Optimal condition reference |

For the YOLO detection dataset, the dominant class per image is determined by counting bounding-box annotations per class and assigning the image to the majority class. This allows the annotated dataset to contribute to classification training.

A per-class cap of 500 images was applied to prevent `good` (originally 3,931 images) and `worn` (2,288 images) from dominating the training distribution.

**Final balanced dataset:**

| Class | Train | Val | Test |
|---|---|---|---|
| no_tire | 240 | 30 | 30 |
| flat | 240 | 30 | 30 |
| defective | 400 | 50 | 50 |
| worn | 400 | 50 | 50 |
| good | 400 | 50 | 50 |
| new | 240 | 30 | 30 |
| **Total** | **1,920** | **240** | **240** |

---

## 5. Data Augmentation (`augment_minority.py`)

After quarantine, several minority classes had too few images:
- `flat`: 99 (from 300, rest quarantined as blurry/duplicate)
- `new`: 30 (from 168)
- `no_tire`: 235

These were augmented to 300 images each using Albumentations:

```
RandomHorizontalFlip (p=0.5)
RandomVerticalFlip (p=0.3)
Rotate ±20° (p=0.7)
RandomBrightnessContrast ±30% (p=0.8)
GaussianBlur 3-5px (p=0.3)
HueSaturationValue jitter (p=0.4)
```

Augmentation generates new training samples by applying random geometric and photometric transforms to existing images. This is standard practice for small datasets and improves generalisation by exposing the model to more visual variability.

---

## 6. Model Architecture

### Classification Model (`train_yolov8.py`)

**Architecture:** YOLOv8n-cls (nano classification variant)

YOLOv8-cls uses a CSPDarknet backbone with a classification head. The "nano" variant is chosen for its balance of accuracy and inference speed on mobile/edge hardware.

**Training configuration:**
```
Epochs:     50 (early stopping patience=10)
Image size: 224×224
Batch size: 32
Optimizer:  AdamW
LR:         1e-3 → 1e-5 (cosine annealing)
Warmup:     3 epochs
```

**Results:**
- Best epoch: 28 (early stopping at epoch 29)
- Val top-1 accuracy: **87.9%**
- Val top-5 accuracy: **100%**
- Val loss: 0.087

The 87.9% top-1 accuracy on a 6-class balanced problem is strong for a nano-sized model. The 100% top-5 accuracy means the correct class is always in the model's top 5 predictions.

### Detection Model (`train_detection.py`)

**Architecture:** YOLOv8n (nano detection variant)

Trained on the Roboflow tread dataset (4,350 annotated images after quarantine) to locate tires in the frame with bounding boxes. Classes: BAD_Tyres, BALD_Tyres, NORMAL_Tyres.

This model provides the bounding box used to crop the tire region, which is then passed to the classification model. The two-stage approach separates *where is the tire* from *what condition is it in*, allowing each model to specialise.

---

## 7. Inference Pipeline

When the app sends a camera frame to the server, inference runs in two stages:

```
Input frame (JPEG)
        ↓
Stage 1: Detection model
        → finds tire location → bounding box (x, y, w, h)
        → fallback: center 65% crop if no detection model
        ↓
Pre-check (fallback mode only)
        → run classifier on crop
        → if top class = "no_tire" OR confidence < 55% → return has_tire=False
        ↓
Stage 2: Classification model
        → crops to bounding box region
        → runs YOLOv8n-cls
        → returns: class, confidence, color
        ↓
JSON response to client
```

The two-stage pipeline is important because classifying the entire frame (including background, road, car body) would confuse the model. By first isolating the tire region, the classifier can focus on the relevant features.

---

## 8. API Server (`app/main.py`)

The server is built with FastAPI and serves two protocols:

### Endpoints
| Endpoint | Protocol | Purpose |
|---|---|---|
| `GET /` | HTTP | Serves browser web app |
| `POST /predict` | HTTP | Single image → classification |
| `WS /ws?key=<key>` | WebSocket | Continuous frame streaming |
| `GET /model_status` | HTTP | Detection model availability |

### Transport
- **Port 8000**: HTTPS (TLS) — required for browser camera access
- **Port 8001**: HTTP — for mobile app on local network

SSL certificates are auto-generated at startup using Python's `cryptography` library. The cert includes the local IP as a Subject Alternative Name (SAN) so browsers accept it.

### Security Layers
1. **Authentication**: Every request (except the static HTML root) requires `X-API-Key` header or `?key=` query parameter. The key is loaded from `.env` using `python-dotenv`.
2. **Timing attack prevention**: Key comparison uses `hmac.compare_digest()` which runs in constant time regardless of where the strings differ.
3. **Rate limiting**: In-memory per-IP request counter. Requests exceeding 60/minute receive HTTP 429.
4. **File validation**: Upload bytes are inspected for magic byte signatures (JPEG: `\xff\xd8\xff`, PNG: `\x89PNG`, etc.). Files that don't match known image formats are rejected with HTTP 415.
5. **Size cap**: Files larger than 10 MB are rejected with HTTP 413.
6. **Docs disabled**: FastAPI's auto-generated `/docs` and `/redoc` endpoints are disabled in production to avoid exposing the API surface.

---

## 9. Mobile Application (`mobile/`)

### Stack
- **Framework**: React Native with Expo SDK 54 (React Native 0.81.5)
- **Camera**: `expo-camera` (CameraView + useCameraPermissions)
- **Animation**: React Native built-in `Animated` API
- **Networking**: `fetch` API with `FormData` for image upload

### How It Works
1. `CameraView` renders a full-screen live camera feed
2. Every 700ms, `takePictureAsync({ quality: 0.25 })` captures a JPEG frame
3. The frame is posted to the server's `/predict` endpoint with the API key header
4. The JSON response (`has_tire`, `class`, `label`, `color`, `confidence`) updates the UI
5. The UI shows: corner-bracket box, animated scan line, condition label, confidence bar

### UI Design
- **Full-screen camera** with transparent overlay
- **Dim vignette** outside the scanning box focuses attention
- **Corner brackets** change color to match the detected condition
- **Scan line** animates top-to-bottom continuously via `Animated.loop`
- **Result bar** fades in at the bottom with condition name and confidence bar
- **Status dot** (top-right): green = connected, red = not connected

### "No Tire" Gating
To prevent false positives when no tire is in frame:
- **With detection model**: only shows result if YOLO detects a box (conf ≥ 0.35)
- **Without detection model**: only shows result if classifier confidence ≥ 55% AND top class ≠ `no_tire`

---

## 10. Git Security

The repository is public. The following measures protect sensitive data:

| Item | Protection |
|---|---|
| API key | `.env` file, gitignored |
| Server IP | `mobile/config.js`, gitignored |
| SSL certs | `app/cert.pem`, `app/key.pem`, gitignored |
| Model weights | `runs/`, gitignored |
| Raw dataset (6.2 GB) | `data/train/`, `data/valid/` etc., gitignored |
| Historical IP exposure | Scrubbed from all commits via `git-filter-repo` |

---

## 11. Results and Limitations

### What Works Well
- 87.9% accuracy on the balanced 6-class test set
- Real-time inference (~300ms per frame on CPU)
- Detects flat tires, worn tread, and good condition reliably
- Works on both iOS (Expo Go) and Android (EAS APK)

### Current Limitations
- **No detection model**: The `tire_det` model requires ~30-40 min training. Until complete, the app uses a fixed center-crop, meaning the tire must be centered in the frame.
- **Screen photography**: The model struggles to classify tire images displayed on screens due to color profile differences, glare, and pixel patterns. Real physical tires work correctly.
- **Small training set**: 1,920 training images after balancing. More data, especially for `flat` (240 train) and `new` (240 train), would improve accuracy.
- **Single condition per frame**: Can't handle multiple tires in one frame.

---

## 12. Future Improvements

1. **Train detection model** — provides precise bounding boxes rather than center-crop assumption
2. **On-device inference** — export to ONNX/TFLite, run entirely offline without a server
3. **Tread depth estimation** — add a regression head or depth camera integration for mm-accurate tread depth
4. **More training data** — especially real-world flat tires and defective tires in field conditions
5. **Temporal smoothing** — average predictions over 3-5 frames to reduce flickering
6. **Report generation** — log scan history with timestamps and export PDF inspection reports
