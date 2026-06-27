"""
Tire Scanner — real-time video inference server.

Security model:
  - All endpoints require X-API-Key header (or ?key= query param for WebSocket)
  - API key loaded from .env — never hardcoded
  - File uploads validated by magic bytes, not just extension
  - 10 MB upload size cap
  - Rate limiting: 60 requests/minute per IP
  - HTTPS on port 8000 (browser), HTTP on port 8001 (LAN mobile only)

Run:  python app/main.py
"""

import asyncio
import datetime
import hmac
import io
import ipaddress
import json
import os
import socket
import time
from collections import defaultdict
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

# ── Secrets ────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    raise RuntimeError("API_KEY not set in .env — refusing to start without authentication.")

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
RATE_LIMIT_RPM   = 60

ALLOWED_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG":      "png",
    b"GIF8":         "gif",
    b"RIFF":         "webp",
    b"BM":           "bmp",
}

# ── Rate limiter ───────────────────────────────────────────────────────────────
_buckets: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(ip: str) -> bool:
    now = time.monotonic()
    _buckets[ip] = [t for t in _buckets[ip] if now - t < 60]
    if len(_buckets[ip]) >= RATE_LIMIT_RPM:
        return False
    _buckets[ip].append(now)
    return True

# ── Auth ───────────────────────────────────────────────────────────────────────
def validate_key(key: str | None) -> bool:
    if not key:
        return False
    return hmac.compare_digest(key, API_KEY)   # constant-time, prevents timing attacks

def validate_image(data: bytes) -> bool:
    for magic in ALLOWED_MAGIC:
        if data[:len(magic)] == magic:
            return True
    return False

# ── SSL cert ───────────────────────────────────────────────────────────────────
def ensure_ssl_cert(cert_path: Path, key_path: Path):
    if cert_path.exists() and key_path.exists():
        return
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    local_ip = socket.gethostbyname(socket.gethostname())
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "tire-scanner")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv4Address(local_ip)),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    print("  SSL cert generated.")

# ── Models ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
CLS_WEIGHTS = BASE / "runs" / "tire_cls" / "weights" / "best.pt"
DET_WEIGHTS = BASE / "runs" / "tire_det" / "weights" / "best.pt"

if not CLS_WEIGHTS.exists():
    raise FileNotFoundError(f"Classification model not found: {CLS_WEIGHTS}")

cls_model = YOLO(str(CLS_WEIGHTS))
print("Classification model loaded.")

det_model = None
if DET_WEIGHTS.exists():
    det_model = YOLO(str(DET_WEIGHTS))
    print("Detection model loaded.")
else:
    print("Detection model not ready — using center-crop fallback.")

CLASS_INFO = {
    "no_tire":   {"label": "No Tire",    "color": "#6b7280"},
    "flat":      {"label": "Flat Tire",  "color": "#ef4444"},
    "defective": {"label": "Defective",  "color": "#f97316"},
    "worn":      {"label": "Worn Tread", "color": "#eab308"},
    "good":      {"label": "Good",       "color": "#22c55e"},
    "new":       {"label": "New Tire",   "color": "#3b82f6"},
}

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(docs_url=None, redoc_url=None)   # disable public API docs
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0] if fwd else (request.client.host or "unknown")


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path == "/":           # static HTML — no auth needed
        return await call_next(request)

    ip = _client_ip(request)
    if not check_rate_limit(ip):
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)

    key = request.headers.get("X-API-Key") or request.query_params.get("key")
    if not validate_key(key):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text("utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")
    if not validate_image(raw):
        raise HTTPException(415, "Unsupported file type")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return JSONResponse(_run_inference(img))
    except Exception:
        raise HTTPException(500, "Inference failed")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    if not validate_key(websocket.query_params.get("key")):
        await websocket.close(code=4401)
        return

    ip = websocket.client.host or "unknown"
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_bytes()
            if not check_rate_limit(ip):
                await websocket.send_text(json.dumps({"error": "Rate limit exceeded"}))
                continue
            if len(raw) > MAX_UPLOAD_BYTES:
                await websocket.send_text(json.dumps({"error": "Frame too large"}))
                continue
            if not validate_image(raw):
                await websocket.send_text(json.dumps({"error": "Invalid image"}))
                continue
            try:
                payload = _run_inference(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                payload = {"error": "Inference failed"}
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass


def _run_inference(img: Image.Image) -> dict:
    img_w, img_h = img.size
    box = None
    has_tire = False
    crop = img

    if det_model:
        det_res = det_model(img, imgsz=640, conf=0.35, verbose=False)
        boxes   = det_res[0].boxes
        if boxes is not None and len(boxes):
            best   = int(boxes.conf.argmax())
            xyxy   = boxes.xyxy[best].cpu().numpy().astype(int)
            x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
            x2, y2 = min(img_w, int(xyxy[2])), min(img_h, int(xyxy[3]))
            box      = {"x": x1/img_w, "y": y1/img_h, "w": (x2-x1)/img_w, "h": (y2-y1)/img_h}
            crop     = img.crop((x1, y1, x2, y2))
            has_tire = True
    else:
        pad_x = int(img_w * 0.175)
        pad_y = int(img_h * 0.175)
        crop  = img.crop((pad_x, pad_y, img_w - pad_x, img_h - pad_y))
        pre   = cls_model(crop, verbose=False)
        pre_name = cls_model.names[int(pre[0].probs.top1)]
        pre_conf = float(pre[0].probs.top1conf)
        has_tire = (pre_name != "no_tire" and pre_conf >= 0.25)
        if has_tire:
            box = {"x": 0.175, "y": 0.175, "w": 0.65, "h": 0.65}

    if not has_tire:
        return {"has_tire": False}

    cls_res  = cls_model(crop, verbose=False)
    probs    = cls_res[0].probs
    top_idx  = int(probs.top1)
    top_conf = float(probs.top1conf)
    cls_name = cls_model.names[top_idx]
    info     = CLASS_INFO.get(cls_name, {"label": cls_name, "color": "#ffffff"})
    return {
        "has_tire":   True,
        "box":        box,
        "class":      cls_name,
        "label":      info["label"],
        "color":      info["color"],
        "confidence": round(top_conf * 100, 1),
    }


async def _run_servers(cert, key):
    https_cfg = uvicorn.Config(app, host="0.0.0.0", port=8000,
                               ssl_certfile=str(cert), ssl_keyfile=str(key), log_level="warning")
    http_cfg  = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="warning")
    await asyncio.gather(uvicorn.Server(https_cfg).serve(), uvicorn.Server(http_cfg).serve())


if __name__ == "__main__":
    ip       = socket.gethostbyname(socket.gethostname())
    cert_dir = Path(__file__).parent
    cert     = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    ensure_ssl_cert(cert, key_file)
    print(f"\n  Tire Scanner (authenticated)")
    print(f"  Browser : https://{ip}:8000")
    print(f"  Mobile  : http://{ip}:8001\n")
    asyncio.run(_run_servers(cert, key_file))
