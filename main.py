import os
import io
import json
import base64
import logging
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
SERVICE_NAME = "gp-logo-check-api"
BUILD_TAG = "debug-2026-02-07-001"

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Supported ratios (tolerance lets minor resize/crop pass)
RATIO_TOL = 0.02  # +/- 2%

# Allowed ratios by asset type (you can expand later)
# NOTE: using labels and numeric (w/h)
ASSET_RATIOS = {
    # Facebook
    "facebook_feed": {
        "1:1": 1.0,
        "4:5": 4 / 5,
        "16:9": 16 / 9,
        "5:4": 5 / 4,
    },
    "facebook_story": {"9:16": 9 / 16},

    # Instagram
    "instagram_feed": {"1:1": 1.0, "4:5": 4 / 5, "16:9": 16 / 9},
    "instagram_story": {"9:16": 9 / 16},
    "instagram_reel": {"9:16": 9 / 16},

    # Generic
    "video": {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0},
}

# Safe-area margins (normalized)
SAFE_AREA_BY_RATIO = {
    "9:16": {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "4:5":  {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "1:1":  {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "5:4":  {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "16:9": {"left": 0.10, "right": 0.90, "top": 0.12, "bottom": 0.88},
}

# If logo center falls in these OUTSIDE-safe-area regions => OK
ALLOWED_REGION_NAMES = [
    "top_left", "top_right", "bottom_left", "bottom_right", "top_band", "bottom_band"
]


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


# -----------------------------
# Helpers
# -----------------------------
def _img_to_b64_data_url(img_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _load_asset_bytes(filename: str) -> Optional[Tuple[bytes, str]]:
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    # these are pngs in your repo
    return data, "image/png"


def _ratio_label(w: int, h: int) -> Tuple[str, float]:
    r = w / h
    candidates = {
        "1:1": 1.0,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
    }
    best = "unknown"
    best_diff = 999.0
    for label, val in candidates.items():
        diff = abs(r - val)
        if diff < best_diff:
            best_diff = diff
            best = label
    if best_diff <= RATIO_TOL:
        return best, r
    return "unknown", r


def _match_asset_ratio(asset_key: str, r: float) -> Optional[str]:
    allowed = ASSET_RATIOS.get(asset_key, {})
    for label, val in allowed.items():
        if abs(r - val) <= RATIO_TOL:
            return label
    return None


def _norm_safe_area(ratio_label: str) -> Optional[Dict[str, float]]:
    return SAFE_AREA_BY_RATIO.get(ratio_label)


def _logo_center_from_bbox(bbox: Dict[str, int]) -> Tuple[float, float]:
    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
    return cx, cy


def _classify_logo_region(cx: float, cy: float, w: int, h: int, safe: Dict[str, float]) -> str:
    nx = cx / w
    ny = cy / h

    left = safe["left"]
    right = safe["right"]
    top = safe["top"]
    bottom = safe["bottom"]

    in_left = nx < left
    in_right = nx > right
    in_top = ny < top
    in_bottom = ny > bottom

    if in_left and in_top:
        return "top_left"
    if in_right and in_top:
        return "top_right"
    if in_left and in_bottom:
        return "bottom_left"
    if in_right and in_bottom:
        return "bottom_right"
    if in_top and (left <= nx <= right):
        return "top_band"
    if in_bottom and (left <= nx <= right):
        return "bottom_band"
    return "center_safe_area"


def _placement_ok_and_offset(cx: float, cy: float, w: int, h: int, safe: Dict[str, float]) -> Tuple[bool, str, Dict[str, Any]]:
    region = _classify_logo_region(cx, cy, w, h, safe)
    if region in ALLOWED_REGION_NAMES:
        return True, region, {"dx_px": 0, "dy_px": 0, "suggestion": "OK"}

    # If in center safe area, push to nearest boundary
    nx = cx / w
    ny = cy / h

    left = safe["left"]
    right = safe["right"]
    top = safe["top"]
    bottom = safe["bottom"]

    d_left = abs(nx - left)
    d_right = abs(right - nx)
    d_top = abs(ny - top)
    d_bottom = abs(bottom - ny)

    candidates = [("left", d_left), ("right", d_right), ("top", d_top), ("bottom", d_bottom)]
    direction = sorted(candidates, key=lambda x: x[1])[0][0]

    eps = 0.005
    target_nx, target_ny = nx, ny
    if direction == "left":
        target_nx = left - eps
    elif direction == "right":
        target_nx = right + eps
    elif direction == "top":
        target_ny = top - eps
    elif direction == "bottom":
        target_ny = bottom + eps

    dx_px = int(round((target_nx - nx) * w))
    dy_px = int(round((target_ny - ny) * h))

    hint_x = ""
    hint_y = ""
    if dx_px < 0:
        hint_x = f"{abs(dx_px)}px left"
    elif dx_px > 0:
        hint_x = f"{abs(dx_px)}px right"
    if dy_px < 0:
        hint_y = f"{abs(dy_px)}px up"
    elif dy_px > 0:
        hint_y = f"{abs(dy_px)}px down"

    suggestion = "Move logo " + " and ".join([p for p in [hint_x, hint_y] if p]).strip()
    if suggestion == "Move logo":
        suggestion = "Move logo slightly outside safe area"

    return False, region, {"dx_px": dx_px, "dy_px": dy_px, "suggestion": suggestion}


# ✅ Updated client getter:
# Supports BOTH env names:
# - OPENAI_API_KEY (recommended)
# - OPEN_AI_API (your Render screenshot shows this)
def _get_openai_client() -> OpenAI:
    key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API") or "").strip()
    if not key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY (or OPEN_AI_API) in Render env vars.")
    return OpenAI(api_key=key)


def _extract_bbox_with_openai(image_bytes: bytes, mime: str) -> Dict[str, Any]:
    """
    Uses OpenAI vision to locate ONLY the GP symbol (blue 3-lobed icon).
    Returns strict JSON:
      {logo_detected:boolean, confidence:number 0..1, bbox:{x_min,y_min,x_max,y_max} or null}
    """
    client = _get_openai_client()

    img_data_url = _img_to_b64_data_url(image_bytes, mime)

    ref1 = _load_asset_bytes("gp_logo.png")
    ref2 = _load_asset_bytes("gp_logo_white.png")

    content = [
        {
            "type": "input_text",
            "text": (
                "Task: Detect ONLY the 'GP' symbol (the blue three-lobed icon), not the text 'gpfi'.\n"
                "Return ONLY strict JSON with keys:\n"
                "  logo_detected (boolean), confidence (number 0..1), bbox (object or null).\n"
                "bbox must be pixel coordinates in the FIRST image:\n"
                "  {x_min:int, y_min:int, x_max:int, y_max:int}\n"
                "If no GP symbol is visible: logo_detected=false, bbox=null, confidence=0.\n"
                "No extra text. JSON only."
            ),
        },
        {"type": "input_image", "image_url": img_data_url},
    ]

    # Add reference logo images (helps the model)
    if ref1:
        content.append({"type": "input_image", "image_url": _img_to_b64_data_url(ref1[0], ref1[1])})
    if ref2:
        content.append({"type": "input_image", "image_url": _img_to_b64_data_url(ref2[0], ref2[1])})

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": content}],
    )

    text = (resp.output_text or "").strip()

    # Robust parse (in case model wraps JSON)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise RuntimeError(f"OpenAI returned non-JSON output: {text[:300]}")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="GP Logo Check API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    assets_exists = os.path.isdir(ASSETS_DIR)
    assets_list = []
    if assets_exists:
        try:
            assets_list = sorted(os.listdir(ASSETS_DIR))
        except Exception:
            assets_list = []

    return {
        "ok": True,
        "service": SERVICE_NAME,
        "build_tag": BUILD_TAG,
        "routes": ["/", "/health", "/check", "/docs"],
        "assets_folder_exists": assets_exists,
        "assets": assets_list,
    }


@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE_NAME, "build_tag": BUILD_TAG}


@app.post("/check")
async def check(
    file: UploadFile = File(...),
    asset: str = Query("facebook_feed", description="e.g. facebook_feed, instagram_story, video"),
):
    try:
        # Read image
        img_bytes = await file.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        mime = (file.content_type or "image/jpeg").strip()

        # PIL load
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        r = w / h

        ratio_lbl, ratio_val = _ratio_label(w, h)
        matched_ratio = _match_asset_ratio(asset, r)

        safe = _norm_safe_area(ratio_lbl)

        # Call OpenAI to get bbox
        logo_data = _extract_bbox_with_openai(img_bytes, mime)

        logo_detected = bool(logo_data.get("logo_detected", False))
        confidence = float(logo_data.get("confidence", 0) or 0)
        bbox = logo_data.get("bbox", None)

        logo_position = "unknown"
        placement_ok = None
        placement_reason = None
        placement_offset = None

        if logo_detected and bbox and safe:
            cx, cy = _logo_center_from_bbox(bbox)
            ok, region2, offset = _placement_ok_and_offset(cx, cy, w, h, safe)
            logo_position = region2
            placement_ok = ok
            placement_offset = offset
            placement_reason = "OK" if ok else "Logo center is inside safe area."

        elif logo_detected and bbox and not safe:
            logo_position = "detected_but_ratio_unknown"
            placement_ok = None
            placement_reason = "Logo detected, but image ratio is not supported for safe-area placement rules."

        return {
            "service": SERVICE_NAME,
            "build_tag": BUILD_TAG,

            "width": w,
            "height": h,
            "ratio": round(ratio_val, 4),

            "asset": asset,
            "asset_ratio_allowed": bool(matched_ratio is not None),
            "matched_ratio_label": matched_ratio,   # allowed ratio label for that asset
            "ratio_label": ratio_lbl,               # detected common ratio label
            "safe_area_norm": safe,                 # margins if known

            "logo_detected": logo_detected,
            "confidence": round(confidence, 3),
            "logo_bbox": bbox,                      # pixel bbox if present

            "logo_position": logo_position,
            "placement_ok": placement_ok,
            "placement_reason": placement_reason,
            "placement_offset": placement_offset,   # dx/dy pixel suggestion
        }

    except HTTPException:
        raise
    except Exception as e:
        # IMPORTANT: return the real reason in JSON so Swagger shows it
        logger.exception("check() failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": str(e),
                "hint": "Check Render Logs for full traceback. Also confirm OPENAI_API_KEY (or OPEN_AI_API) is set.",
                "service": SERVICE_NAME,
                "build_tag": BUILD_TAG,
            },
        )
