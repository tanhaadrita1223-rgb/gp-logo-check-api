import os
import io
import json
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# SAFE ZONE + ANCHOR LOGIC
# ----------------------------
# You shared templates where the blue center is the "safe area" for content,
# and logos should be placed OUTSIDE that blue area (near corners / top / bottom zones).
#
# We'll define a "content safe rectangle" (center) as percentages,
# and require the logo CENTER to be OUTSIDE that rectangle.
#
# You can tune these later, but this matches the idea of the templates:
# big centered safe area, margins around it allowed for branding.
#
# For different ratios, margins differ slightly. We'll map by aspect label.

SAFE_ZONE_BY_RATIO = {
    # (w/h) approx values:
    "9:16": {"safe_left": 0.12, "safe_right": 0.88, "safe_top": 0.08, "safe_bottom": 0.92},
    "4:5":  {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.08, "safe_bottom": 0.92},
    "1:1":  {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.10, "safe_bottom": 0.90},
    "5:4":  {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.12, "safe_bottom": 0.88},
    "16:9": {"safe_left": 0.08, "safe_right": 0.92, "safe_top": 0.12, "safe_bottom": 0.88},
}

# Allowed anchor points (logo center targets) outside safe zone:
# We'll propose nearest anchor and return dx/dy.
ANCHORS_BY_RATIO = {
    "9:16": ["top_left", "top_right", "bottom_left", "bottom_right"],
    "4:5":  ["top_left", "top_right", "bottom_left", "bottom_right"],
    "1:1":  ["top_left", "top_center", "top_right", "bottom_left", "bottom_right"],
    "5:4":  ["top_left", "top_right", "bottom_left", "bottom_right"],
    "16:9": ["top_left", "top_right", "bottom_left", "bottom_right"],
}

def guess_ratio_label(w: int, h: int) -> str:
    r = w / h
    candidates = [
        ("9:16", 9/16),
        ("4:5", 4/5),
        ("1:1", 1.0),
        ("5:4", 5/4),
        ("16:9", 16/9),
    ]
    best = min(candidates, key=lambda x: abs(r - x[1]))
    return best[0]

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def anchor_target_xy(ratio_label: str, safe: Dict[str, float], anchor: str) -> Tuple[float, float]:
    # pick target centers in the "allowed" margin zones
    # Example: top_left target is half-way into left margin & top margin.
    left_margin_mid_x = safe["safe_left"] / 2
    right_margin_mid_x = (safe["safe_right"] + 1.0) / 2
    top_margin_mid_y = safe["safe_top"] / 2
    bottom_margin_mid_y = (safe["safe_bottom"] + 1.0) / 2
    top_center_x = 0.5
    top_center_y = top_margin_mid_y

    if anchor == "top_left":
        return (left_margin_mid_x, top_margin_mid_y)
    if anchor == "top_right":
        return (right_margin_mid_x, top_margin_mid_y)
    if anchor == "bottom_left":
        return (left_margin_mid_x, bottom_margin_mid_y)
    if anchor == "bottom_right":
        return (right_margin_mid_x, bottom_margin_mid_y)
    if anchor == "top_center":
        return (top_center_x, top_center_y)

    # fallback
    return (left_margin_mid_x, top_margin_mid_y)

def is_outside_safe_zone(cx: float, cy: float, safe: Dict[str, float]) -> bool:
    inside = (safe["safe_left"] <= cx <= safe["safe_right"]) and (safe["safe_top"] <= cy <= safe["safe_bottom"])
    return not inside

def nearest_anchor_and_offset(
    ratio_label: str,
    safe: Dict[str, float],
    cx: float, cy: float,
    w: int, h: int
) -> Dict[str, Any]:
    anchors = ANCHORS_BY_RATIO.get(ratio_label, ["top_left", "top_right", "bottom_left", "bottom_right"])
    best = None
    for a in anchors:
        tx, ty = anchor_target_xy(ratio_label, safe, a)
        dx = tx - cx
        dy = ty - cy
        dist2 = dx*dx + dy*dy
        if best is None or dist2 < best["dist2"]:
            best = {"anchor": a, "tx": tx, "ty": ty, "dx": dx, "dy": dy, "dist2": dist2}

    # pixel offsets
    dx_px = int(round(best["dx"] * w))
    dy_px = int(round(best["dy"] * h))

    # human direction
    def dir_text(dxpx: int, dypx: int) -> str:
        parts = []
        if dxpx > 0: parts.append(f"move RIGHT ~{abs(dxpx)}px")
        if dxpx < 0: parts.append(f"move LEFT ~{abs(dxpx)}px")
        if dypx > 0: parts.append(f"move DOWN ~{abs(dypx)}px")
        if dypx < 0: parts.append(f"move UP ~{abs(dypx)}px")
        return ", ".join(parts) if parts else "already aligned"

    return {
        "nearest_anchor": best["anchor"],
        "target_center_norm": {"x": round(best["tx"], 4), "y": round(best["ty"], 4)},
        "offset_px": {"dx": dx_px, "dy": dy_px},
        "suggestion": dir_text(dx_px, dy_px),
    }

# ----------------------------
# OPENAI VISION: LOGO BBOX
# ----------------------------
# We ask the model to return a tight bounding box for the GP logo mark.
# Returned bbox format: x1,y1,x2,y2 in PIXELS relative to the image.
#
# This uses OpenAI's vision image input capability. :contentReference[oaicite:2]{index=2}

def openai_detect_logo_bbox(image_bytes: bytes) -> Dict[str, Any]:
    b64 = "data:image/jpeg;base64," + (io.BytesIO(image_bytes).getvalue()).hex()  # placeholder, replaced below

    # Proper base64 encode
    import base64
    b64 = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
You are a vision detector. Find the GP logo symbol (the blue "three-petal" Grameenphone mark).
Return ONLY strict JSON with:
{
  "logo_detected": boolean,
  "confidence": number (0-1),
  "bbox": {"x1": int, "y1": int, "x2": int, "y2": int}  // pixel coords
}
Rules:
- If multiple logos, return the most prominent one.
- If not found, set logo_detected=false and omit bbox.
"""

    resp = client.responses.create(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": b64},
            ]
        }],
    )

    text = resp.output_text.strip()
    try:
        data = json.loads(text)
    except Exception:
        # if model returns extra text, try to extract JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            return {"logo_detected": False, "confidence": 0.0}

    # normalize
    if not isinstance(data, dict):
        return {"logo_detected": False, "confidence": 0.0}

    logo_detected = bool(data.get("logo_detected", False))
    conf = data.get("confidence", 0.0)
    bbox = data.get("bbox")
    if not logo_detected or not bbox:
        return {"logo_detected": False, "confidence": float(conf or 0.0)}

    return {
        "logo_detected": True,
        "confidence": float(conf or 0.0),
        "bbox": {
            "x1": int(bbox["x1"]), "y1": int(bbox["y1"]),
            "x2": int(bbox["x2"]), "y2": int(bbox["y2"]),
        }
    }

# ----------------------------
# API ENDPOINT
# ----------------------------
@app.post("/check")
async def check(file: UploadFile = File(...)):
    # read bytes
    image_bytes = await file.read()

    # read size (guaranteed)
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not parse image: {str(e)}"}
        )

    ratio_label = guess_ratio_label(w, h)
    safe = SAFE_ZONE_BY_RATIO.get(ratio_label, SAFE_ZONE_BY_RATIO["1:1"])

    # detect logo bbox via OpenAI
    logo = {"logo_detected": False, "confidence": 0.0}
    try:
        logo = openai_detect_logo_bbox(image_bytes)
    except Exception as e:
        # fail gracefully
        logo = {"logo_detected": False, "confidence": 0.0, "error": f"logo_detect_error: {str(e)}"}

    # compute placement
    placement_ok = None
    placement_reason = None
    logo_position = "unknown"
    offsets = None

    if logo.get("logo_detected") and logo.get("bbox"):
        x1, y1, x2, y2 = logo["bbox"]["x1"], logo["bbox"]["y1"], logo["bbox"]["x2"], logo["bbox"]["y2"]
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        cx = cx_px / w
        cy = cy_px / h

        logo_position = f"({int(round(cx_px))}, {int(round(cy_px))})"

        outside = is_outside_safe_zone(cx, cy, safe)
        offsets = nearest_anchor_and_offset(ratio_label, safe, cx, cy, w, h)

        placement_ok = bool(outside)
        if placement_ok:
            placement_reason = f"Logo center is OUTSIDE safe area. Nearest anchor: {offsets['nearest_anchor']}."
        else:
            placement_reason = f"Logo center is INSIDE safe area. Suggested fix: {offsets['suggestion']} (towards {offsets['nearest_anchor']})."

    content = {
        "width": w,
        "height": h,
        "ratio_label": ratio_label,
        "safe_area_norm": safe,

        "logo_detected": bool(logo.get("logo_detected", False)),
        "confidence": float(logo.get("confidence", 0.0)) if isinstance(logo.get("confidence", 0.0), (int, float)) else 0.0,
        "logo_bbox": logo.get("bbox"),
        "logo_position": logo_position,

        "placement_ok": placement_ok,
        "placement_reason": placement_reason,
        "placement_offset": offsets,
    }
    return JSONResponse(content=content)
