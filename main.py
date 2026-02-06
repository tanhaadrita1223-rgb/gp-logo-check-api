from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAFE_MARGIN_PCT = 0.05
CORNER_ZONE_PCT = 0.35
MIN_MATCH_SCORE = 0.55

TEMPLATE_PATH = "gp_logo_template.png"
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
if template is None:
    raise RuntimeError("Logo template not found")

def to_bgr(img):
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = a.astype(np.float32) / 255.0
        bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.float32) * 255
        fg = cv2.merge([b, g, r]).astype(np.float32)
        out = fg * alpha[..., None] + bg * (1 - alpha[..., None])
        return out.astype(np.uint8)
    return img

template = to_bgr(template)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

def detect_logo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    best = {"score": -1, "bbox": None}

    for scale in [0.4, 0.6, 0.8, 1.0]:
        th = int(template_gray.shape[0] * scale)
        tw = int(template_gray.shape[1] * scale)
        if th >= h or tw >= w:
            continue

        t = cv2.resize(template_gray, (tw, th))
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best["score"]:
            x, y = max_loc
            best["score"] = float(max_val)
            best["bbox"] = (x, y, x + tw, y + th)

    return best["bbox"], best["score"]

def check_placement(bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    margin_x, margin_y = w * SAFE_MARGIN_PCT, h * SAFE_MARGIN_PCT
    safe = x1 > margin_x and y1 > margin_y and x2 < w - margin_x and y2 < h - margin_y

    zone_w, zone_h = w * CORNER_ZONE_PCT, h * CORNER_ZONE_PCT
    corner = (
        (cx < zone_w or cx > w - zone_w) and
        (cy < zone_h or cy > h - zone_h)
    )

    placement = (
        "top-left" if cx < w/2 and cy < h/2 else
        "top-right" if cx > w/2 and cy < h/2 else
        "bottom-left" if cx < w/2 else
        "bottom-right"
    )

    return safe and corner, placement

@app.post("/check")
async def check(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "message": "Invalid image"}

    h, w = img.shape[:2]
    bbox, score = detect_logo(img)

    if bbox is None or score < MIN_MATCH_SCORE:
        return {
            "ok": False,
            "logo_detected": False,
            "placement_ok": False,
            "message": "Logo not detected"
        }

    placement_ok, placement = check_placement(bbox, w, h)

    return {
        "ok": placement_ok,
        "logo_detected": True,
        "placement_ok": placement_ok,
        "placement": placement,
        "score": score,
        "message": "Logo placement OK" if placement_ok else "Logo detected but placement is NOT OK"
    }

@app.get("/")
def root():
    return {"status": "alive"}
