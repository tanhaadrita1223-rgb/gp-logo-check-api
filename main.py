from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
from typing import Optional, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- CONFIG ----------
# Put your logo template here (must exist in repo)
LOGO_TEMPLATE_PATHS = [
    "assets/gp_logo.png",     # primary
    "assets/gp_logo_white.png",  # optional (if you sometimes use white logo)
]

# Template matching sensitivity:
# - raise to 0.70 if too many false positives
# - lower to 0.55 if missing real logos
MATCH_THRESHOLD = 0.62

# --------- SAFE AREAS (normalized) ----------
# These match your "Light Blue Safe Area" diagrams:
# Safe area = content-safe area; logo should be in margins OUTSIDE this safe area,
# ideally in corners.
SAFE_AREA_BY_RATIO = {
    "9:16":  {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.08, "safe_bottom": 0.92},
    "4:5":   {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.08, "safe_bottom": 0.92},
    "1:1":   {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.10, "safe_bottom": 0.90},
    "16:9":  {"safe_left": 0.08, "safe_right": 0.92, "safe_top": 0.12, "safe_bottom": 0.88},
    "5:4":   {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.10, "safe_bottom": 0.90},
}

# --------- HELPERS ----------
def ratio_label_from_wh(w: int, h: int) -> str:
    r = w / h
    candidates = {
        "1:1": 1.0,
        "4:5": 4/5,
        "5:4": 5/4,
        "9:16": 9/16,
        "16:9": 16/9,
    }
    # choose closest
    best = min(candidates.items(), key=lambda kv: abs(r - kv[1]))
    return best[0]

def load_templates():
    templates = []
    for p in LOGO_TEMPLATE_PATHS:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # Convert to grayscale for matching
        if img.ndim == 3 and img.shape[2] == 4:
            # if PNG with alpha: use RGB part
            bgr = img[:, :, :3]
        elif img.ndim == 3:
            bgr = img
        else:
            bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        templates.append((p, gray))
    return templates

TEMPLATES = load_templates()

def detect_logo_bbox(image_gray: np.ndarray) -> Dict[str, Any]:
    """
    Returns:
      {logo_detected: bool, confidence: float, bbox: [x1,y1,x2,y2] or None}
    Uses multi-scale template matching.
    """
    if not TEMPLATES:
        return {"logo_detected": False, "confidence": 0.0, "bbox": None, "template_used": None}

    H, W = image_gray.shape[:2]
    best_score = -1.0
    best_bbox = None
    best_tpl = None

    # scales to handle different logo sizes
    scales = [0.35, 0.45, 0.55, 0.70, 0.85, 1.0, 1.15, 1.3]

    for (tpl_path, tpl_gray) in TEMPLATES:
        th, tw = tpl_gray.shape[:2]

        for s in scales:
            rw = int(tw * s)
            rh = int(th * s)
            if rw < 12 or rh < 12:
                continue
            if rw >= W or rh >= H:
                continue

            tpl_rs = cv2.resize(tpl_gray, (rw, rh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(image_gray, tpl_rs, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = float(max_val)
                x1, y1 = max_loc
                x2, y2 = x1 + rw, y1 + rh
                best_bbox = [int(x1), int(y1), int(x2), int(y2)]
                best_tpl = tpl_path

    if best_score >= MATCH_THRESHOLD and best_bbox is not None:
        return {"logo_detected": True, "confidence": best_score, "bbox": best_bbox, "template_used": best_tpl}

    return {"logo_detected": False, "confidence": best_score if best_score > 0 else 0.0, "bbox": None, "template_used": best_tpl}

def logo_position_from_bbox(bbox, w, h) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    left = cx < w * 0.5
    top = cy < h * 0.5

    if top and left:
        return "top-left"
    if top and not left:
        return "top-right"
    if not top and left:
        return "bottom-left"
    return "bottom-right"

def placement_check(bbox, w, h, safe_norm: Dict[str, float]) -> Dict[str, Any]:
    """
    Goal: logo should be OUTSIDE safe area AND in a CORNER zone.
    Safe area = [safe_left..safe_right] x [safe_top..safe_bottom]
    Corner zone requirement: (x in left/right margin) AND (y in top/bottom margin)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    safe_left_px = safe_norm["safe_left"] * w
    safe_right_px = safe_norm["safe_right"] * w
    safe_top_px = safe_norm["safe_top"] * h
    safe_bottom_px = safe_norm["safe_bottom"] * h

    in_left_margin = cx <= safe_left_px
    in_right_margin = cx >= safe_right_px
    in_top_margin = cy <= safe_top_px
    in_bottom_margin = cy >= safe_bottom_px

    # must be corner: one horizontal margin AND one vertical margin
    is_corner = (in_left_margin or in_right_margin) and (in_top_margin or in_bottom_margin)

    # offsets: how much to move center to become corner-valid
    dx = 0.0
    dy = 0.0

    # If x is inside safe area, push to nearest side margin
    if safe_left_px < cx < safe_right_px:
        # choose nearest edge
        if (cx - safe_left_px) < (safe_right_px - cx):
            dx = safe_left_px - cx  # negative means move left
        else:
            dx = safe_right_px - cx  # positive means move right
    # If x is already in margin but not corner because of y, dx stays 0

    # If y is inside safe area, push to nearest top/bottom margin
    if safe_top_px < cy < safe_bottom_px:
        if (cy - safe_top_px) < (safe_bottom_px - cy):
            dy = safe_top_px - cy  # negative means move up
        else:
            dy = safe_bottom_px - cy  # positive means move down

    if is_corner:
        return {
            "placement_ok": True,
            "placement_reason": "Logo is in approved corner safe zone (outside safe area).",
            "placement_offset": {"dx_px": 0, "dy_px": 0, "dx_norm": 0.0, "dy_norm": 0.0},
        }

    # If not corner, explain why
    reason_parts = []
    if safe_left_px < cx < safe_right_px:
        reason_parts.append("logo is inside the content safe area horizontally")
    if safe_top_px < cy < safe_bottom_px:
        reason_parts.append("logo is inside the content safe area vertically")
    if not reason_parts:
        reason_parts.append("logo is not in a corner zone")

    return {
        "placement_ok": False,
        "placement_reason": " / ".join(reason_parts),
        "placement_offset": {
            "dx_px": int(round(dx)),
            "dy_px": int(round(dy)),
            "dx_norm": float(dx / w),
            "dy_norm": float(dy / h),
        },
    }

@app.post("/check")
async def check(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Load image via PIL
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    # Ratio label
    ratio_label = ratio_label_from_wh(w, h)

    # Safe area (fallback to generic if unknown)
    safe_norm = SAFE_AREA_BY_RATIO.get(ratio_label, {"safe_left": 0.10, "safe_right": 0.90, "safe_top": 0.10, "safe_bottom": 0.90})

    # Convert to OpenCV grayscale
    np_img = np.array(img)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    det = detect_logo_bbox(gray)

    resp: Dict[str, Any] = {
        "width": w,
        "height": h,
        "ratio_label": ratio_label,
        "safe_area_norm": safe_norm,
        "logo_detected": bool(det["logo_detected"]),
        "confidence": float(det["confidence"]),
        "logo_bbox": det["bbox"],
        "logo_position": "unknown",
        "placement_ok": None,
        "placement_reason": None,
        "placement_offset": None,
        "template_used": det.get("template_used"),
    }

    if det["logo_detected"] and det["bbox"]:
        bbox = det["bbox"]
        resp["logo_position"] = logo_position_from_bbox(bbox, w, h)

        place = placement_check(bbox, w, h, safe_norm)
        resp["placement_ok"] = place["placement_ok"]
        resp["placement_reason"] = place["placement_reason"]
        resp["placement_offset"] = place["placement_offset"]

    return resp

@app.get("/health")
def health():
    return {"ok": True, "templates_loaded": len(TEMPLATES), "threshold": MATCH_THRESHOLD}
