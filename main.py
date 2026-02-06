from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

app = FastAPI(title="GP Logo Check API")

# Allow all origins (safe for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- SAFE AREAS BY RATIO --------
SAFE_AREAS = {
    "9:16": {"left": 0.1, "right": 0.9, "top": 0.1, "bottom": 0.9},
    "4:5":  {"left": 0.1, "right": 0.9, "top": 0.08, "bottom": 0.92},
    "1:1":  {"left": 0.1, "right": 0.9, "top": 0.1, "bottom": 0.9},
    "5:4":  {"left": 0.08, "right": 0.92, "top": 0.1, "bottom": 0.9},
    "16:9": {"left": 0.1, "right": 0.9, "top": 0.12, "bottom": 0.88},
}

# -------- HEALTH CHECK --------
@app.get("/health")
def health():
    assets_path = "assets"
    templates = []
    if os.path.exists(assets_path):
        templates = os.listdir(assets_path)

    return {
        "ok": True,
        "templates_loaded": len(templates),
        "templates": templates
    }

# -------- CHECK ENDPOINT --------
@app.post("/check")
async def check_logo(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    ratio = round(width / height, 2)

    if abs(ratio - 0.56) < 0.02:
        ratio_label = "9:16"
    elif abs(ratio - 0.8) < 0.02:
        ratio_label = "4:5"
    elif abs(ratio - 1.0) < 0.02:
        ratio_label = "1:1"
    elif abs(ratio - 1.25) < 0.02:
        ratio_label = "5:4"
    elif abs(ratio - 1.78) < 0.02:
        ratio_label = "16:9"
    else:
        ratio_label = "unknown"

    safe = SAFE_AREAS.get(ratio_label)

    return {
        "width": width,
        "height": height,
        "ratio_label": ratio_label,
        "safe_area_norm": safe,
        "logo_detected": False,
        "confidence": 0,
        "logo_bbox": None,
        "logo_position": "unknown",
        "placement_ok": None,
        "placement_reason": None,
        "placement_offset": None
    }
