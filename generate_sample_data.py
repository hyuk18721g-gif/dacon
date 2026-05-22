"""
업로드용 샘플 데이터 생성 스크립트.
실행: python generate_sample_data.py
출력: sample_agv_data.csv, thermal_front.png ~ thermal_top.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).parent
np.random.seed(42)

# ── 1. AGV 센서 CSV ────────────────────────────────────────
INTERVAL = pd.Timedelta(minutes=10)
START    = pd.Timestamp("2025-05-21 00:00:00")
STEPS    = 144   # 24시간 × 6 = 144 타임스텝

AGV_PROFILES = {
    "AGV-01": {"label": "정상",             "motor_bias": 0,   "vib_bias": 0,    "batt_bias": 0},
    "AGV-02": {"label": "모터 과열 진행",   "motor_bias": 14,  "vib_bias": 0.05, "batt_bias": 2},
    "AGV-03": {"label": "진동+전류 이상",   "motor_bias": 8,   "vib_bias": 0.32, "batt_bias": 3},
    "AGV-04": {"label": "배터리 온도 상승", "motor_bias": 2,   "vib_bias": 0.02, "batt_bias": 10},
    "AGV-05": {"label": "정상",             "motor_bias": 0,   "vib_bias": 0,    "batt_bias": 0},
    "AGV-06": {"label": "전압 저하",        "motor_bias": 3,   "vib_bias": 0.08, "batt_bias": 1},
    "AGV-07": {"label": "복합 이상",        "motor_bias": 11,  "vib_bias": 0.20, "batt_bias": 7},
}

rows = []
for agv_id, prof in AGV_PROFILES.items():
    for step in range(STEPS):
        t = START + INTERVAL * step
        prog = step / STEPS   # 0→1 (시간 경과)

        # 이상 AGV는 후반부로 갈수록 악화
        sev = prog ** 1.5

        motor_temp   = 58 + prof["motor_bias"] * sev + np.random.randn() * 1.5
        battery_temp = 33 + prof["batt_bias"] * sev  + np.random.randn() * 1.0
        vibration    = 0.25 + prof["vib_bias"] * sev + np.random.randn() * 0.02
        voltage      = 24.0 - (0.8 if "전압" in prof["label"] else 0) * sev + np.random.randn() * 0.2
        current      = 9.5  + np.random.randn() * 0.8 + (1.5 * sev if prof["vib_bias"] > 0.1 else 0)
        speed        = 1.4  + np.random.randn() * 0.15
        error_count  = max(0, int(np.random.poisson(0.05 + 2 * sev * (prof["motor_bias"] > 5))))
        load_weight  = 45   + np.random.randn() * 5
        driving_time = step * 10   # 분

        rows.append({
            "timestamp":    t.strftime("%Y-%m-%d %H:%M:%S"),
            "agv_id":       agv_id,
            "motor_temp":   round(motor_temp, 2),
            "battery_temp": round(battery_temp, 2),
            "vibration":    round(max(0, vibration), 4),
            "voltage":      round(voltage, 2),
            "current":      round(max(0, current), 2),
            "speed":        round(max(0, speed), 3),
            "error_count":  error_count,
            "load_weight":  round(load_weight, 1),
            "driving_time": driving_time,
        })

df = pd.DataFrame(rows)
csv_path = OUT / "sample_agv_data.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[OK] {csv_path}  ({len(df):,}행, AGV {df['agv_id'].nunique()}대)")


# ── 2. 열화상 이미지 5면 ───────────────────────────────────
W, H = 400, 300


def _base_array() -> np.ndarray:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y, x] = [4 + x // 22, 8, 28 + y // 10]
    arr[50:250, 50:350] = [18, 48, 88]
    arr[155:250, 50:160] = [28, 65, 98]   # 좌측 모터
    arr[75:200, 148:258] = [23, 57, 92]   # 배터리
    arr[155:250, 248:352] = [28, 65, 98]  # 우측 휠
    arr[50:130, 148:258]  = [20, 53, 90]  # 제어 모듈
    arr[228:252, 52:100]  = [8, 25, 55]
    arr[228:252, 298:352] = [8, 25, 55]
    return arr


def _add_hotspot(arr, cx, cy, radius, intensity):
    Y, X = np.ogrid[:H, :W]
    dist  = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    sigma = radius * 0.55
    mask  = np.exp(-dist ** 2 / (2 * sigma ** 2)).clip(0, 1)
    hot_r = np.clip(255 * mask * intensity, 0, 255)
    hot_g = np.clip(200 * mask * intensity * (1 - dist / (radius + 1) * 0.65), 0, 255)
    hot_b = np.clip(30  * mask * intensity * (1 - dist / (radius + 1)), 0, 255)
    blend = (mask * intensity)[..., np.newaxis]
    out   = arr.astype(np.float32)
    out[..., 0] = out[..., 0] * (1 - blend[..., 0]) + hot_r * blend[..., 0]
    out[..., 1] = out[..., 1] * (1 - blend[..., 0]) + hot_g * blend[..., 0]
    out[..., 2] = out[..., 2] * (1 - blend[..., 0]) + hot_b * blend[..., 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_img(label, hotspots):
    """label: 상단 텍스트, hotspots: [(cx, cy, radius, intensity), ...]"""
    arr = _base_array()
    for cx, cy, r, intens in hotspots:
        arr = _add_hotspot(arr, cx, cy, r, intens)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 22], fill=(0, 0, 0))
    draw.text((5, 5), f"Thermal Cam  |  {label}", fill=(190, 190, 190))
    return img


FACES = {
    "thermal_front": ("Front  [좌측 모터 이상]",  [(95,  210, 55, 0.95)]),
    "thermal_back":  ("Back   [정상]",             []),
    "thermal_left":  ("Left   [정상]",             []),
    "thermal_right": ("Right  [휠/구동부 이상]",   [(308, 213, 48, 0.90)]),
    "thermal_top":   ("Top    [배터리 이상]",       [(200, 148, 60, 0.92)]),
}

for fname, (label, hotspots) in FACES.items():
    img  = _make_img(label, hotspots)
    path = OUT / f"{fname}.png"
    img.save(path)
    print(f"[OK] {path}")

print("\n완료! DACON 폴더에서 파일을 확인하세요.")
