"""Generate demo 5-face thermal image sets for each AGV.

Output:
  thermal_images/
    AGV-01/front.png
    AGV-01/back.png
    AGV-01/left.png
    AGV-01/right.png
    AGV-01/top.png
    ...
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).parent
CSV_PATH = ROOT / "sample_agv_data.csv"
OUT_DIR = ROOT / "thermal_images"

W, H = 400, 300
FACES = ("front", "back", "left", "right", "top")


def _base_array(face: str) -> np.ndarray:
    arr = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            arr[y, x] = [4 + x // 22, 8, 28 + y // 10]

    # Simple AGV-like thermal template.
    arr[50:250, 50:350] = [18, 48, 88]
    arr[155:250, 50:160] = [28, 65, 98]    # left drive/motor zone
    arr[75:200, 148:258] = [23, 57, 92]    # battery/core zone
    arr[155:250, 248:352] = [28, 65, 98]   # right drive/wheel zone
    arr[50:130, 148:258] = [20, 53, 90]    # controller zone
    arr[228:252, 52:100] = [8, 25, 55]
    arr[228:252, 298:352] = [8, 25, 55]

    # Slight visual variation by face.
    if face == "top":
        arr[60:240, 80:320] = [21, 56, 90]
        arr[95:205, 145:255] = [25, 63, 98]
    elif face == "back":
        arr[70:230, 70:330] = [16, 43, 80]
    elif face == "left":
        arr[140:250, 70:210] = [31, 68, 101]
    elif face == "right":
        arr[140:250, 190:330] = [31, 68, 101]

    return arr


def _add_hotspot(arr: np.ndarray, cx: int, cy: int, radius: int, intensity: float) -> np.ndarray:
    y_grid, x_grid = np.ogrid[:H, :W]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    sigma = radius * 0.55
    mask = np.exp(-(dist ** 2) / (2 * sigma ** 2)).clip(0, 1)

    hot_r = np.clip(255 * mask * intensity, 0, 255)
    hot_g = np.clip(200 * mask * intensity * (1 - dist / (radius + 1) * 0.65), 0, 255)
    hot_b = np.clip(30 * mask * intensity * (1 - dist / (radius + 1)), 0, 255)

    blend = (mask * intensity)[..., np.newaxis]
    out = arr.astype(np.float32)
    out[..., 0] = out[..., 0] * (1 - blend[..., 0]) + hot_r * blend[..., 0]
    out[..., 1] = out[..., 1] * (1 - blend[..., 0]) + hot_g * blend[..., 0]
    out[..., 2] = out[..., 2] * (1 - blend[..., 0]) + hot_b * blend[..., 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def _hotspots_for(agv_id: str, face: str):
    """Return synthetic hotspot definitions matching the demo sensor profiles."""
    profiles = {
        "AGV-01": {},
        "AGV-02": {"front": [(95, 210, 50, 0.85)]},
        "AGV-03": {"right": [(305, 212, 48, 0.92)], "front": [(285, 205, 35, 0.55)]},
        "AGV-04": {"top": [(200, 148, 58, 0.90)]},
        "AGV-05": {},
        "AGV-06": {"back": [(205, 95, 38, 0.70)]},
        "AGV-07": {
            "front": [(95, 210, 55, 1.00)],
            "top": [(200, 148, 60, 0.95)],
            "right": [(308, 213, 48, 0.85)],
        },
    }
    return profiles.get(agv_id, {}).get(face, [])


def _make_image(agv_id: str, face: str) -> Image.Image:
    arr = _base_array(face)
    for cx, cy, radius, intensity in _hotspots_for(agv_id, face):
        arr = _add_hotspot(arr, cx, cy, radius, intensity)

    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 22], fill=(0, 0, 0))
    draw.text((5, 5), f"Thermal Cam | {agv_id} | {face}", fill=(190, 190, 190))
    return img


def main() -> None:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        agv_ids = sorted(df["agv_id"].dropna().unique())
    else:
        agv_ids = [f"AGV-{i:02d}" for i in range(1, 8)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for agv_id in agv_ids:
        agv_dir = OUT_DIR / agv_id
        agv_dir.mkdir(parents=True, exist_ok=True)
        for face in FACES:
            img = _make_image(agv_id, face)
            img.save(agv_dir / f"{face}.png")
            count += 1

    print(f"[OK] generated {count} thermal images in {OUT_DIR}")


if __name__ == "__main__":
    main()
