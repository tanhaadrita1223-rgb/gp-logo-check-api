from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np

app = FastAPI()

@app.post("/check")
async def check_logo(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    width, height = img.size

    # Simple heuristic: logo detected if non-transparent pixels exist
    arr = np.array(img)
    alpha = arr[:, :, 3]
    non_transparent = np.count_nonzero(alpha > 10)

    if non_transparent < 100:
        return {
            "logo_detected": False,
            "message": "Logo not detected"
        }

    # Find bounding box of logo
    ys, xs = np.where(alpha > 10)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    position = "center"
    if center_x < width * 0.33:
        position = "left"
    elif center_x > width * 0.66:
        position = "right"
    elif center_y > height * 0.66:
        position = "bottom"

    return {
        "logo_detected": True,
        "logo_position": position,
        "bbox": {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max)
        }
    }
