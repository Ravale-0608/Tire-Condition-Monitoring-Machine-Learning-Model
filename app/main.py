"""
Tire Scanner API
Run:  python app/main.py
Then open  http://<your-ip>:8000  on your phone (same WiFi)
"""

import io
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "runs" / "tire_cls" / "weights" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_yolov8.py first.")

model = YOLO(str(MODEL_PATH))

CLASS_INFO = {
    "no_tire":   {"label": "No Tire Detected",  "color": "#6b7280", "icon": "🚫"},
    "flat":      {"label": "Flat Tire",          "color": "#ef4444", "icon": "⚠️"},
    "defective": {"label": "Defective Tire",     "color": "#f97316", "icon": "🔴"},
    "worn":      {"label": "Worn Tread",         "color": "#eab308", "icon": "🟡"},
    "good":      {"label": "Good Condition",     "color": "#22c55e", "icon": "✅"},
    "new":       {"label": "New Tire",           "color": "#3b82f6", "icon": "🔵"},
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Tire Scanner")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(img, verbose=False)
        probs   = results[0].probs
        top_idx  = int(probs.top1)
        top_conf = float(probs.top1conf)
        cls_name = model.names[top_idx]

        # Top-3 predictions
        top3 = []
        for idx in probs.top5[:3]:
            name = model.names[int(idx)]
            conf = float(probs.data[int(idx)])
            info = CLASS_INFO.get(name, {"label": name, "color": "#6b7280", "icon": "❓"})
            top3.append({"class": name, "label": info["label"],
                         "confidence": round(conf * 100, 1),
                         "color": info["color"], "icon": info["icon"]})

        info = CLASS_INFO.get(cls_name, {"label": cls_name, "color": "#6b7280", "icon": "❓"})
        return JSONResponse({
            "class":      cls_name,
            "label":      info["label"],
            "confidence": round(top_conf * 100, 1),
            "color":      info["color"],
            "icon":       info["icon"],
            "top3":       top3,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "model": str(MODEL_PATH.name)}


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n  Tire Scanner running!")
    print(f"  Local :  http://localhost:8000")
    print(f"  Phone :  http://{local_ip}:8000  (same WiFi)\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
