from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import os

app = FastAPI(title="GP Logo Check API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "gp_logo_template.png")

def load_image_bytes(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    np_img = np.array(img)  # RGB
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return bgr, w, h

def load_template():
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        raise ValueError("Failed to read template image")
    # Convert to BGR if has alpha
    if tpl.shape[-1] == 4:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
    return tpl

def match_logo(image_bgr, template_bgr):
    """
    Multi-scale template matching.
    Returns best match: (found: bool, bbox, confidence)
    bbox: (x_min, y_min, x_max, y_max)
    """
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tpl_gray_orig = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # Light edge emphasis helps avoid matching big text/blocks
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_edge = cv2.Canny(img_gray, 50, 150)

    best_val = -1.0
    best_bbox = None

    h_img, w_img = img_edge.shape[:2]

    # scales: small → larger
    scales = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.3]

    for s in scales:
        tpl = cv2.resize(tpl_gray_orig, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        th, tw = tpl.shape[:2]

        # Skip if template bigger than image
        if th >= h_img or tw >= w_img or th < 10 or tw < 10:
            continue

        tpl_blur = cv2.GaussianBlur(tpl, (3, 3), 0)
        tpl_edge = cv2.Canny(tpl_blur, 50, 150)

        res = cv2.matchTemplate(img_edge, tpl_edge, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            x, y = max_loc
            best_val = float(max_val)
            best_bbox = (int(x), int(y), int(x + tw), int(y + th))

    # threshold: tuneable
    found = best_bbox is not None and best_val >= 0.55
    return found, best_bbox, best_val

def logo_position_from_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # 3x3 grid
    col = "left" if cx < w/3 else ("right" if cx > 2*w/3 else "center")
    row = "top" if cy < h/3 else ("bottom" if cy > 2*h/3 else "middle")

    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}_{col}"

def placement_ok(bbox, w, h):
    """
    Define "allowed safe corners" (top-left/top-right/bottom-left/bottom-right)
    as 0-25% of width/height areas.
    If logo bbox lies fully inside any of these corner safe boxes => OK.
    """
    x1, y1, x2, y2 = bbox
    safe_w = int(w * 0.25)
    safe_h = int(h * 0.25)

    corners = {
        "top_left": (0, 0, safe_w, safe_h),
        "top_right": (w - safe_w, 0, w, safe_h),
        "bottom_left": (0, h - safe_h, safe_w, h),
        "bottom_right": (w - safe_w, h - safe_h, w, h),
    }

    for name, (cx1, cy1, cx2, cy2) in corners.items():
        if x1 >= cx1 and y1 >= cy1 and x2 <= cx2 and y2 <= cy2:
            return True, f"Inside safe corner zone: {name}"

    return False, "Logo is not inside a safe corner zone (25% x 25%)."

@app.get("/")
def root():
    return {"ok": True, "message": "Use POST /check or /docs"}

@app.post("/check")
async def check_logo(file: UploadFile = File(...)):
    data = await file.read()
    image_bgr, w, h = load_image_bytes(data)

    template_bgr = load_template()
    found, bbox, conf = match_logo(image_bgr, template_bgr)

    if not found:
        return {
            "logo_detected": False,
            "confidence": conf,
            "width": w,
            "height": h,
            "logo_position": "none",
            "bbox": None,
            "placement_ok": False,
            "placement_reason": "Logo not detected reliably."
        }

    pos = logo_position_from_bbox(bbox, w, h)
    ok, reason = placement_ok(bbox, w, h)

    return {
        "logo_detected": True,
        "confidence": conf,
        "width": w,
        "height": h,
        "logo_position": pos,
        "bbox": {"x_min": bbox[0], "y_min": bbox[1], "x_max": bbox[2], "y_max": bbox[3]},
        "placement_ok": ok,
        "placement_reason": reason
    }
