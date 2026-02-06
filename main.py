import os
import base64
import json
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from openai import OpenAI

app = FastAPI()

# Allow your Supabase Edge Function to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ----------------------------
# Safe-zone rules (tunable)
# ----------------------------
# Idea:
# - Light-blue "safe area" is the middle region.
# - Logo is "correct" only if its CENTER is OUTSIDE that safe rectangle.
# - If it's inside, compute minimal move in px to get it outside.

# These are *fractions* of width/height for the safe-area margins.
# You can tweak later to match your exact templates; this is a solid default.
SAFE_MARGINS = {
    "9:16": {"mx": 0.10, "my_top": 0.10, "my_bottom": 0.10},
    "4:5":  {"mx": 0.08, "my_top": 0.10, "my_bottom": 0.10},
    "1:1":  {"mx": 0.08, "my_top": 0.10, "my_bottom": 0.10},
    "16:9": {"mx": 0.10, "my_top": 0.18, "my_bottom": 0.18},
    "5:4":  {"mx": 0.10, "my_top": 0.12, "my_bottom": 0.12},
}

def closest_ratio_label(w: int, h: int) -> str:
    r = w / h
    candidates = {
        "9:16": 9/16,
        "4:5": 4/5,
        "1:1": 1.0,
        "16:9": 16/9,
        "5:4": 5/4,
    }
    return min(candidates.keys(), key=lambda k: abs(r - candidates[k]))


def safe_rect(w: int, h: int) -> Tuple[int, int, int, int, str]:
    label = closest_ratio_label(w, h)
    m = SAFE_MARGINS.get(label, SAFE_MARGINS["1:1"])
    left = int(w * m["mx"])
    right = int(w * (1 - m["mx"]))
    top = int(h * m["my_top"])
    bottom = int(h * (1 - m["my_bottom"]))
    return left, top, right, bottom, label


def bbox_center(b: Dict[str, int]) -> Tuple[float, float]:
    return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


def classify_position(cx: float, cy: float, w: int, h: int) -> str:
    # coarse 3x3 grid classification
    x_band = "center"
    y_band = "center"
    if cx < w / 3: x_band = "left"
    elif cx > 2*w/3: x_band = "right"
    if cy < h / 3: y_band = "top"
    elif cy > 2*h/3: y_band = "bottom"

    if y_band == "top" and x_band == "left": return "top_left"
    if y_band == "top" and x_band == "right": return "top_right"
    if y_band == "bottom" and x_band == "left": return "bottom_left"
    if y_band == "bottom" and x_band == "right": return "bottom_right"
    if y_band == "top" and x_band == "center": return "top_center"
    if y_band == "bottom" and x_band == "center": return "bottom_center"
    if y_band == "center" and x_band == "left": return "center_left"
    if y_band == "center" and x_band == "right": return "center_right"
    return "center"


def placement_eval(w: int, h: int, logo_bbox: Dict[str, int]) -> Dict[str, Any]:
    left, top, right, bottom, label = safe_rect(w, h)
    cx, cy = bbox_center(logo_bbox)

    inside_safe = (left <= cx <= right) and (top <= cy <= bottom)

    # Compute minimal move to get outside safe rectangle
    fix = {"move_x_px": 0, "move_y_px": 0, "hint": ""}

    if inside_safe:
        # distances to each safe boundary to exit
        d_left = cx - left
        d_right = right - cx
        d_top = cy - top
        d_bottom = bottom - cy

        # choose smallest move out
        options = [
            ("left", int(d_left) + 1, (-int(d_left) - 1, 0)),
            ("right", int(d_right) + 1, (int(d_right) + 1, 0)),
            ("up", int(d_top) + 1, (0, -int(d_top) - 1)),
            ("down", int(d_bottom) + 1, (0, int(d_bottom) + 1)),
        ]
        direction, amount, (mx, my) = min(options, key=lambda x: x[1])
        fix["move_x_px"] = mx
        fix["move_y_px"] = my
        fix["hint"] = f"Move logo {direction} by ~{amount}px to leave safe area."
        return {
            "placement_ok": False,
            "placement_reason": f"Logo center is inside safe area ({label}).",
            "placement_fix": fix,
            "safe_area": {"x1": left, "y1": top, "x2": right, "y2": bottom, "label": label},
        }

    return {
        "placement_ok": True,
        "placement_reason": f"Logo center is outside safe area ({label}).",
        "placement_fix": {"move_x_px": 0, "move_y_px": 0, "hint": "No change needed."},
        "safe_area": {"x1": left, "y1": top, "x2": right, "y2": bottom, "label": label},
    }


# ----------------------------
# OpenAI Vision: find logo bbox
# ----------------------------
def openai_find_logo_bbox(img_bytes: bytes) -> Dict[str, Any]:
    if client is None:
        return {"logo_detected": False, "confidence": 0.0, "logo_bbox": None, "raw": "OPENAI_API_KEY not set"}

    data_url = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
You are detecting a specific telecom brand logo (Grameenphone / GP / GPFi style).
Task:
1) If ANY such logo exists in the image, return ONE bounding box around the most prominent logo.
2) If none, return logo_detected=false.

Return STRICT JSON with these keys only:
{
  "logo_detected": boolean,
  "confidence": number,  // 0..1
  "logo_bbox": { "x1": int, "y1": int, "x2": int, "y2": int } | null
}

Rules:
- Coordinates are in PIXELS relative to the full image.
- x1,y1 is top-left; x2,y2 is bottom-right.
- Keep the bbox tight around the logo mark.
"""

    # Responses API with image input (official pattern)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt.strip()},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    # The model returns text; parse JSON
    txt = resp.output_text.strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"logo_detected": False, "confidence": 0.0, "logo_bbox": None, "raw": txt}


@app.post("/check")
async def check(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # Read image size reliably
    try:
        im = Image.open(io.BytesIO(img_bytes))
        w, h = im.size
    except Exception as e:
        return {
            "error": f"Could not read image: {str(e)}",
            "width": None,
            "height": None,
            "logo_detected": False,
            "confidence": 0.0,
            "logo_bbox": None,
            "logo_position": "unknown",
            "placement_ok": False,
            "placement_reason": "Image read failed",
        }

    # Find logo bbox via OpenAI
    det = openai_find_logo_bbox(img_bytes)
    logo_detected = bool(det.get("logo_detected"))
    confidence = float(det.get("confidence") or 0.0)
    logo_bbox = det.get("logo_bbox")

    out: Dict[str, Any] = {
        "width": w,
        "height": h,
        "logo_detected": logo_detected,
        "confidence": confidence,
        "logo_bbox": logo_bbox,
        "logo_position": "unknown",
        "placement_ok": False,
        "placement_reason": "",
    }

    if not logo_detected or not logo_bbox:
        out["placement_ok"] = False
        out["placement_reason"] = "Logo not detected."
        return out

    # Clamp bbox to image boundaries
    x1 = max(0, min(w, int(logo_bbox["x1"])))
    y1 = max(0, min(h, int(logo_bbox["y1"])))
    x2 = max(0, min(w, int(logo_bbox["x2"])))
    y2 = max(0, min(h, int(logo_bbox["y2"])))
    logo_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    out["logo_bbox"] = logo_bbox

    cx, cy = bbox_center(logo_bbox)
    out["logo_position"] = classify_position(cx, cy, w, h)

    pe = placement_eval(w, h, logo_bbox)
    out["placement_ok"] = pe["placement_ok"]
    out["placement_reason"] = pe["placement_reason"]
    out["placement_fix"] = pe["placement_fix"]
    out["safe_area"] = pe["safe_area"]

    return out
