# Tire Condition Monitoring — Machine Learning Model

An AI-powered tire inspection system that uses computer vision to detect tire conditions in real time. Point a camera at a tire and get an instant condition assessment.

## What It Does

Classifies tires into 6 conditions:
| Class | Description |
|---|---|
| 🚫 No Tire | No tire visible in frame |
| ⚠️ Flat | Deflated / flat tire |
| 🔴 Defective | Visible damage: cuts, bulges, cracks |
| 🟡 Worn | Severely worn or bald tread |
| ✅ Good | Safe, normal wear |
| 🔵 New | New tire, deep tread |

## Architecture

```
Raw Datasets (13,656 images, 4 label schemas)
        ↓
Data Cleaning Pipeline      → corrupt/blurry/duplicate removal
        ↓
Label Unification           → 4 schemas → 6-class taxonomy
        ↓
Augmentation                → minority class balancing
        ↓
YOLOv8n-cls Training        → 87.9% top-1 accuracy
        ↓
FastAPI Inference Server    → authenticated REST + WebSocket API
        ↓
React Native Mobile App     → real-time camera scanning
```

## Project Structure

```
├── data/                   Raw datasets (gitignored — 6.2 GB)
├── app/
│   ├── main.py             FastAPI inference server
│   └── static/index.html  Browser-based scanner
├── mobile/                 React Native (Expo) mobile app
├── runs/                   Model weights (gitignored)
├── results/                Training curves, confusion matrix
├── samples/                10 sample images per class
├── clean_data.py           Data quality pipeline
├── quarantine_flagged.py   Move bad images to quarantine/
├── augment_minority.py     Albumentations augmentation
├── build_dataset.py        Unify labels → train/val/test split
├── train_yolov8.py         YOLOv8 classification training
├── train_detection.py      YOLOv8 detection training (bounding boxes)
└── .env.example            Secret key template
```

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Expo Go app (iOS/Android)

### 1. Clone & install Python deps
```bash
git clone https://github.com/Ravale-0608/Tire-Condition-Monitoring-Machine-Learning-Model.git
cd "Tire-Condition-Monitoring-Machine-Learning-Model"
pip install ultralytics fastapi uvicorn python-dotenv pillow opencv-python albumentations imagehash
```

### 2. Set up secrets
```bash
cp .env.example .env
# Edit .env and set: API_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
```

### 3. Add your datasets
Place datasets under `data/` matching the structure in `build_dataset.py`.
Then run the full pipeline:
```bash
python clean_data.py          # quality scan + dedup
python quarantine_flagged.py  # remove bad images
python augment_minority.py    # balance classes
python build_dataset.py       # build unified dataset
python train_yolov8.py        # train classifier
```

### 4. Run the inference server
```bash
python app/main.py
# Browser: https://<your-ip>:8000
# Mobile:  http://<your-ip>:8001
```

### 5. Run the mobile app
```bash
cd mobile
cp config.example.js config.js    # set SERVER_IP and API_KEY
npm install
npx expo start --lan
# Scan QR code with Expo Go
```

## Model Performance

| Metric | Value |
|---|---|
| Architecture | YOLOv8n-cls |
| Classes | 6 |
| Training images | 1,920 |
| Best val top-1 | **87.9%** |
| Best val top-5 | 100% |
| Epochs | 29 (early stop) |

See `results/` for training curves and confusion matrix.

## Security

- All API endpoints require `X-API-Key` header authentication
- Constant-time key comparison (prevents timing attacks)
- Rate limiting: 60 requests/minute per IP
- File uploads validated by magic bytes (not just extension)
- 10 MB upload cap
- API docs disabled in production
- SSL/TLS on port 8000 (auto-generated cert)
- No secrets committed — `.env` and `config.js` are gitignored

## Datasets Used

| Dataset | Source | Classes |
|---|---|---|
| Tire Tread | Roboflow (Public Domain) | BAD / BALD / NORMAL |
| Tyre Condition | Local workshop photos | NEW / SERVICEABLE / UNUSABLE |
| Defective/Good | Kaggle | defective / good |
| Flat/Full/No-tire | Public | flat / full / no_tire |
