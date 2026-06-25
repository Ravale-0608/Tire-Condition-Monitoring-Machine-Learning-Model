"""
Tire Scanner — real-time video inference server.

Two-stage pipeline:
  1. Detection model  → bounding box around tire (runs/tire_det/weights/best.pt)
  2. Classification model → condition label      (runs/tire_cls/weights/best.pt)

If the detection model is not yet trained, falls back to a center-crop box.

Run:  python app/main.py
"""

import asyncio
import io
import json
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
CLS_WEIGHTS = BASE / "runs" / "tire_cls" / "weights" / "best.pt"
DET_WEIGHTS = BASE / "runs" / "tire_det" / "weights" / "best.pt"

# ── Load models ───────────────────────────────────────────────────────────────
if not CLS_WEIGHTS.exists():
    raise FileNotFoundError(f"Classification model not found: {CLS_WEIGHTS}")

cls_model = YOLO(str(CLS_WEIGHTS))
print(f"Classification model loaded: {CLS_WEIGHTS.name}")

det_model = None
if DET_WEIGHTS.exists():
    det_model = YOLO(str(DET_WEIGHTS))
    print(f"Detection model loaded: {DET_WEIGHTS.name}")
else:
    print("Detection model not found — using center-crop fallback until trained.")

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_INFO = {
    "no_tire":   {"label": "No Tire",       "color": "#6b7280"},
    "flat":      {"label": "Flat Tire",     "color": "#ef4444"},
    "defective": {"label": "Defective",     "color": "#f97316"},
    "worn":      {"label": "Worn Tread",    "color": "#eab308"},
    "good":      {"label": "Good",          "color": "#22c55e"},
    "new":       {"label": "New Tire",      "color": "#3b82f6"},
}

FRAME_W = 640
FRAME_H = 480

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text("utf-8"))


@app.get("/model_status")
async def model_status():
    return {"detection_ready": det_model is not None}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_bytes()

            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img_w, img_h = img.size

                box = None          # normalised {x, y, w, h}  (0-1)
                crop = img         # region to classify

                # ── Stage 1: detect tire location ──────────────────────────
                if det_model:
                    det_res = det_model(img, imgsz=640, conf=0.3, verbose=False)
                    boxes   = det_res[0].boxes
                    if boxes is not None and len(boxes):
                        # Pick highest-confidence detection
                        best   = int(boxes.conf.argmax())
                        xyxy   = boxes.xyxy[best].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        # Clamp to image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_w, x2), min(img_h, y2)
                        box  = {
                            "x": x1 / img_w, "y": y1 / img_h,
                            "w": (x2 - x1) / img_w, "h": (y2 - y1) / img_h,
                        }
                        crop = img.crop((x1, y1, x2, y2))
                else:
                    # Fallback: center 65% crop
                    pad_x = int(img_w * 0.175)
                    pad_y = int(img_h * 0.175)
                    box   = {"x": 0.175, "y": 0.175, "w": 0.65, "h": 0.65}
                    crop  = img.crop((pad_x, pad_y, img_w - pad_x, img_h - pad_y))

                # ── Stage 2: classify condition ────────────────────────────
                cls_res  = cls_model(crop, verbose=False)
                probs    = cls_res[0].probs
                top_idx  = int(probs.top1)
                top_conf = float(probs.top1conf)
                cls_name = cls_model.names[top_idx]
                info     = CLASS_INFO.get(cls_name, {"label": cls_name, "color": "#ffffff"})

                payload = {
                    "box":        box,
                    "class":      cls_name,
                    "label":      info["label"],
                    "color":      info["color"],
                    "confidence": round(top_conf * 100, 1),
                    "det_model":  det_model is not None,
                }

            except Exception as e:
                payload = {"error": str(e)}

            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n  Tire Scanner (video mode)")
    print(f"  Local : http://localhost:8000")
    print(f"  Phone : http://{ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
