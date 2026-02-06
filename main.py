import os
import io
import json
import base64
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
SERVICE_NAME = "gp-logo-check-api"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

RATIO_TOL = 0.02  # +/- 2%

ASSET_RATIOS = {
    "facebook_feed": {"1:1": 1.0, "4:5": 4 / 5, "16:9": 16 / 9, "5:4": 5 / 4},
    "facebook_story": {"9:16": 9 / 16},
    "instagram_feed": {"1:1": 1.0, "4:5": 4 / 5, "16:9": 16 / 9},
    "instagram_story": {"9:16": 9 / 16},
    "instagram_reel": {"9:16": 9 / 16},
    "video": {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0},
}

SAFE_AREA_BY_RATIO = {
    "9:16": {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "4:5":  {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "1:1":  {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "5:4":  {"left": 0.10, "right": 0.90, "top": 0.08, "bottom": 0.92},
    "16:9": {"left": 0.10, "right": 0.90, "top": 0.12, "bottom": 0.88},
}

ALLOWED_REGION_NAMES = [
    "top_left", "top_right", "bottom_left", "bottom_right", "top_band", "bottom_band"
]


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

    eps = 0.005
    direction = sorted(
        [("left", d_left), ("right", d_right), ("top", d_top), ("bottom", d_bottom)],
        key=lambda x: x[1],
    )[0][0]

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

    def _fmt(dx, dy):
        parts = []
        if dx != 0:
            parts.append(f"{abs(dx)}px {'left' if dx < 0 else 'right'}")
        if dy != 0:
            parts.append(f"{abs(dy)}px {'up' if dy < 0 else 'down'}")
        return " and ".join(parts) if parts else "0px"

    return False, region, {
        "dx_px": dx_px,
        "dy_px": dy_px,
        "suggestion": f"Move logo {_fmt(dx_px, dy_px)}",
    }


# ✅ IMPORTANT: supports BOTH env var names:
# - OPENAI_API_KEY (recommended)
# - OPEN_AI_API (your Render env group screenshot shows this one)
def _get_openai_client() -> OpenAI:
    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_AI_API")
        or os.getenv("OPENAI_KEY")
    )
    if not key:
        raise RuntimeError("Missing OpenAI key. Set OPENAI_API_KEY (recommended) or OPEN_AI_API in Render env vars.")
    return OpenAI(api_key=key)


def _extract_bbox_with_openai(image_bytes: bytes, mime: str) -> Dict[str, Any]:
    """
    Detect ONLY the GP symbol (the blue three-lobed icon).
    Return strict JSON:
      {logo_detected:boolean, confidence:number 0..1, bbox:{x_min,y_min,x_max,y_max} or null}
    """
    client = _get_openai_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # fastest cheap default

    img_data_url = _img_to_b64_data_url(image_bytes, mime)
    ref1 = _load_asset_bytes("gp_logo.png")
    ref2 = _load_asset_bytes("gp_logo_white.png")

    content = [
        {
            "type": "input_text",
            "text": (
                "Task: detect ONLY the GP symbol (the blue 3-lobed icon), not text like 'gpfi'. "
                "Return ONLY strict JSON with keys: "
                "{\"logo_detected\": boolean, \"confidence\": number, "
                "\"bbox\": {\"x_min\": int, \"y_min\": int, \"x_max\": int, \"y_max\": int} | null}. "
                "The bbox must tightly cover the GP symbol in the FIRST image. "
                "If not present, set logo_detected=false, bbox=null, confidence=0."
            ),
        },
        {"type": "input_image", "image_url": img_data_url},
    ]

    # Optional references (helps a lot)
    if ref1:
        content.append({"type": "input_image", "image_url": _img_to_b64_data_url(ref1[0], ref1[1])})
    if ref2:
        content.append({"type": "input_image", "image_url": _img_to_b64_data_url(ref2[0], ref2[1])})

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )

    text = (resp.output_text or "").strip()

    # Robust JSON parse
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise RuntimeError(f"OpenAI returned non-JSON output: {text[:250]}")


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
        "routes": ["/", "/health", "/check", "/docs"],
        "assets_folder_exists": assets_exists,
        "assets": assets_list,
        "openai_env_found": bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API") or os.getenv("OPENAI_KEY")),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/check")
async def check(
    file: UploadFile = File(...),
    asset: str = Query("facebook_feed", description="e.g. facebook_feed, instagram_story, video"),
):
    img_bytes = await file.read()
    mime = file.content_type or "image/jpeg"

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    ratio_lbl, ratio_val = _ratio_label(w, h)

    matched_ratio = _match_asset_ratio(asset, ratio_val)
    safe = _norm_safe_area(ratio_lbl)

    # OpenAI call (wrapped so you don't get random 500s)
    openai_error = None
    logo_data: Dict[str, Any] = {"logo_detected": False, "confidence": 0, "bbox": None}

    try:
        logo_data = _extract_bbox_with_openai(img_bytes, mime)
    except Exception as e:
        # Keep API alive (no 500), but report error in JSON
        openai_error = str(e)

    logo_detected = bool(logo_data.get("logo_detected", False))
    confidence = float(logo_data.get("confidence", 0) or 0)
    bbox = logo_data.get("bbox", None)

    logo_position = "unknown"
    placement_ok = None
    placement_reason = None
    placement_offset = None

    if logo_detected and bbox and safe:
        cx, cy = _logo_center_from_bbox(bbox)
        region = _classify_logo_region(cx, cy, w, h, safe)
        ok, region2, offset = _placement_ok_and_offset(cx, cy, w, h, safe)

        logo_position = region2
        placement_ok = ok
        placement_offset = offset
        placement_reason = "OK" if ok else f"Logo center is inside safe area ({region})."

    elif logo_detected and bbox and not safe:
        logo_position = "detected_but_ratio_unknown"
        placement_reason = "Logo detected, but image ratio is not supported for safe-area rules."

    return {
        "width": w,
        "height": h,
        "ratio": round(ratio_val, 4),
        "asset": asset,
        "asset_ratio_allowed": bool(matched_ratio is not None),
        "matched_ratio_label": matched_ratio,
        "ratio_label": ratio_lbl,
        "safe_area_norm": safe,

        "logo_detected": logo_detected,
        "confidence": round(confidence, 3),
        "logo_bbox": bbox,
        "logo_position": logo_position,
        "placement_ok": placement_ok,
        "placement_reason": placement_reason,
        "placement_offset": placement_offset,

        # Helps you debug without breaking the bot
        "openai_error": openai_error,
    }
