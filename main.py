from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI(title="GP Logo Check API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATE_PATH = "gp_logo_template.png"

def _load_and_prepare_template():
    tpl_bgr = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
    if tpl_bgr is None:
        raise RuntimeError(f"Template not found at {TEMPLATE_PATH}")

    # Convert to gray
    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

    # Remove white background by auto-cropping to non-white pixels
    # Anything darker than ~245 is considered part of the logo (works for blue-on-white)
    mask = (tpl_gray < 245).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise RuntimeError("Template looks empty after background removal")

    x, y, w, h = cv2.boundingRect(coords)
    tpl_crop = tpl_gray[y:y+h, x:x+w]

    # Edge template (more robust)
    tpl_edge = cv2.Canny(tpl_crop, 50, 150)

    return tpl_edge

TEMPLATE_EDGE = _load_and_prepare_template()

def _match_logo_bbox(img_bgr: np.ndarray):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, 50, 150)

    best = None  # (score, x, y, w, h)

    # Multi-scale matching (handles different logo sizes)
    tpl0 = TEMPLATE_EDGE
    th0, tw0 = tpl0.shape[:2]

    scales = [1.25, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    for s in scales:
        tw = int(tw0 * s)
        th = int(th0 * s)
        if tw < 15 or th < 15:
            continue
        if tw >= img_edge.shape[1] or th >= img_edge.shape[0]:
            continue

        tpl = cv2.resize(tpl0, (tw, th), interpolation=cv2.INTER_AREA)

        res = cv2.matchTemplate(img_edge, tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if best is None or max_val > best[0]:
            best = (float(max_val), int(max_loc[0]), int(max_loc[1]), int(tw), int(th))

    return best  # or None

def _position_label(cx_norm: float, cy_norm: float):
    # 3x3 grid
    if cx_norm < 1/3:
        xlab = "left"
    elif cx_norm > 2/3:
        xlab = "right"
    else:
        xlab = "center"

    if cy_norm < 1/3:
        ylab = "top"
    elif cy_norm > 2/3:
        ylab = "bottom"
    else:
        ylab = "center"

    if xlab == "center" and ylab == "center":
        return "center"
    if xlab == "center":
        return ylab
    if ylab == "center":
        return xlab
    return f"{ylab}_{xlab}"  # top_left, top_right, bottom_left, bottom_right

def _placement_ok(position: str, cx_norm: float, cy_norm: float):
    # Rules:
    # - NOT ok if it's in the center area (we already label "center")
    # - OK if it is in any corner OR edge zones
    # - Also require it isn't too close to exact border (avoid cut-off): margin >= 1.5%
    margin = 0.015
    if cx_norm < margin or cx_norm > 1 - margin or cy_norm < margin or cy_norm > 1 - margin:
        return False, "Logo too close to the edge (risk of cut-off)."

    if position == "center":
        return False, "Logo is in the center area. Should be placed on an edge/corner."

    # Everything else is allowed (top_left/top_right/bottom_left/bottom_right/top/bottom/left/right)
    return True, "Logo placement looks acceptable."

@app.get("/")
def root():
    return {"ok": True, "message": "Use POST /check with multipart file field 'file'."}

@app.post("/check")
async def check_logo(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not decode image. Please upload a valid JPG/PNG."}

    h, w = img.shape[:2]

    best = _match_logo_bbox(img)
    if best is None:
        return {
            "logo_detected": False,
            "width": w,
            "height": h,
            "score": 0.0,
            "message": "No match candidates."
        }

    score, x, y, tw, th = best

    # Threshold: adjust if needed; with edges this is usually stable
    detected = score >= 0.35

    if not detected:
        return {
            "logo_detected": False,
            "width": w,
            "height": h,
            "score": score,
            "message": "Match score below threshold."
        }

    x_min = max(0, x)
    y_min = max(0, y)
    x_max = min(w - 1, x + tw)
    y_max = min(h - 1, y + th)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    cx_norm = cx / w
    cy_norm = cy / h

    pos = _position_label(cx_norm, cy_norm)
    ok, reason = _placement_ok(pos, cx_norm, cy_norm)

    return {
        "logo_detected": True,
        "width": w,
        "height": h,
        "score": score,
        "logo_position": pos,
        "placement_ok": ok,
        "placement_reason": reason,
        "bbox": {"x_min": int(x_min), "y_min": int(y_min), "x_max": int(x_max), "y_max": int(y_max)},
        "center_norm": {"cx": round(cx_norm, 4), "cy": round(cy_norm, 4)},
    }
