"""열화상 이미지 생성 및 Hot-spot 탐지 모듈.

각 면(front/back/left/right/top)별로 AGV 부품 레이아웃이 다르며,
이상 AGV(03/07/11)는 결함 유형에 맞는 물리적 위치에 핫스팟이 생성됩니다.
  - AGV-03 (모터 과열+진동): front 좌측 모터 + left 면 모터 마운트
  - AGV-07 (배터리 과열+전압저하): top 배터리팩 + back 배터리 환기구
  - AGV-11 (휠 진동+속도불안): left/right 양쪽 휠허브 + front 양쪽 휠
"""
from __future__ import annotations
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

THERMAL_DIR = Path(__file__).parent.parent / "data" / "thermal_images"

W, H = 400, 300  # 이미지 크기 (px)
_HEADER_H = 22   # 상단 레이블 영역 높이


# ── 면별 AGV 부품 레이아웃 배열 ──────────────────────────────────

def _build_front_array() -> np.ndarray:
    """전면: 좌측 구동 모터 | 전방 센서 | 우측 구동 모터."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y, x] = [4 + x // 25, 8, 30 + y // 10]
    arr[50:270, 50:355] = [16, 44, 85]          # AGV 본체
    arr[148:255, 52:158] = [26, 62, 96]         # 좌측 구동 모터 하우징
    arr[148:255, 246:358] = [26, 62, 96]        # 우측 구동 모터 하우징
    arr[52:128, 148:258] = [20, 52, 90]         # 전방 센서 어레이
    arr[128:152, 52:358] = [18, 48, 88]         # 섀시 크로스바
    arr[245:272, 55:105] = [8, 22, 52]          # 좌측 앞바퀴 (고무, 어두움)
    arr[245:272, 300:358] = [8, 22, 52]         # 우측 앞바퀴
    return arr


def _build_back_array() -> np.ndarray:
    """후면: 제어 CPU | 배터리 환기구 좌우 | 충전 포트 (좌우 미러)."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y, x] = [4 + x // 25, 8, 30 + y // 10]
    arr[50:270, 50:355] = [16, 44, 85]          # AGV 본체
    arr[52:132, 148:258] = [20, 52, 90]         # 제어 CPU / 메인보드
    arr[148:225, 52:148] = [22, 56, 94]         # 배터리 환기구 (AGV 우측=이미지 좌측)
    arr[148:225, 256:358] = [22, 56, 94]        # 배터리 환기구 (AGV 좌측=이미지 우측)
    arr[228:265, 152:255] = [18, 48, 86]        # 배터리 충전 포트
    arr[262:272, 52:358] = [12, 35, 72]         # 후방 범퍼
    arr[245:272, 55:105] = [8, 22, 52]          # 좌측 뒷바퀴
    arr[245:272, 300:358] = [8, 22, 52]         # 우측 뒷바퀴
    return arr


def _build_left_array() -> np.ndarray:
    """좌측면: 이미지 왼쪽=AGV 전방, 오른쪽=AGV 후방."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y, x] = [4 + x // 25, 8, 30 + y // 10]
    arr[50:270, 50:355] = [16, 44, 85]          # AGV 본체
    arr[52:198, 148:310] = [22, 56, 94]         # 배터리 팩 사이드 패널
    arr[155:250, 52:148] = [26, 62, 96]         # 전방 모터 마운트/구동축 (전방=이미지 좌)
    arr[155:250, 310:358] = [20, 52, 88]        # 후방 섹션
    arr[245:272, 55:115] = [8, 22, 52]          # 전방 좌측 휠허브 (이미지 좌=전방)
    arr[245:272, 295:358] = [8, 22, 52]         # 후방 좌측 휠허브 (이미지 우=후방)
    return arr


def _build_right_array() -> np.ndarray:
    """우측면: 이미지 왼쪽=AGV 후방, 오른쪽=AGV 전방 (좌측면 미러)."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y, x] = [4 + x // 25, 8, 30 + y // 10]
    arr[50:270, 50:355] = [16, 44, 85]          # AGV 본체
    arr[52:198, 95:258] = [22, 56, 94]          # 배터리 팩 사이드 패널
    arr[155:250, 258:358] = [26, 62, 96]        # 전방 모터 마운트/구동축 (전방=이미지 우)
    arr[155:250, 52:95] = [20, 52, 88]          # 후방 섹션 (이미지 좌=후방)
    arr[245:272, 55:115] = [8, 22, 52]          # 후방 우측 휠허브 (이미지 좌=후방)
    arr[245:272, 295:358] = [8, 22, 52]         # 전방 우측 휠허브 (이미지 우=전방)
    return arr


def _build_top_array() -> np.ndarray:
    """상면(조감): 이미지 위=AGV 전방, 아래=AGV 후방."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            arr[y, x] = [4 + x // 25, 8, 30 + y // 10]
    arr[22:285, 60:348] = [16, 44, 85]          # AGV 본체 풋프린트
    arr[25:88, 115:290] = [20, 52, 90]          # 전방 센서 어레이 (위=전방)
    arr[95:225, 80:328] = [22, 56, 94]          # 배터리 팩 (중앙 대형)
    arr[225:275, 62:148] = [26, 62, 96]         # 좌측 구동 모터 (후방-좌)
    arr[225:275, 258:345] = [26, 62, 96]        # 우측 구동 모터 (후방-우)
    arr[248:282, 148:258] = [18, 48, 86]        # 후방 제어 모듈
    return arr


_FACE_BUILDERS = {
    "front": _build_front_array,
    "back":  _build_back_array,
    "left":  _build_left_array,
    "right": _build_right_array,
    "top":   _build_top_array,
}

_FACE_LABEL_KR = {
    "front": "전면 (Front)",
    "back":  "후면 (Back)",
    "left":  "좌측면 (Left)",
    "right": "우측면 (Right)",
    "top":   "상면 (Top)",
}


# ── 면별 부품 매핑 ────────────────────────────────────────────────

def _component_from_face(face: str, rx: float, ry: float) -> str:
    """상대 좌표(rx, ry ∈ [0,1]) + 면 → 부품명."""
    if face == "front":
        if rx < 0.42 and ry > 0.50:
            return "좌측 구동 모터"
        if rx > 0.60 and ry > 0.50:
            return "우측 구동 모터"
        if ry < 0.42:
            return "전방 센서 어레이"
        return "AGV 전면 차체"

    if face == "back":
        if 0.36 < rx < 0.66 and ry > 0.72:
            return "배터리 충전 포트"
        if ry < 0.44:
            return "제어 CPU / 메인보드"
        if rx < 0.40 and ry > 0.48:
            return "배터리 환기구 (우측)"
        if rx > 0.62 and ry > 0.48:
            return "배터리 환기구 (좌측)"
        return "AGV 후면 차체"

    if face == "left":
        if rx < 0.35 and ry > 0.58:
            return "전방 좌측 휠허브"
        if rx > 0.72 and ry > 0.58:
            return "후방 좌측 휠허브"
        if ry < 0.66:
            return "배터리 팩 사이드 패널"
        return "좌측 모터 마운트 / 구동축"

    if face == "right":
        if rx < 0.35 and ry > 0.58:
            return "후방 우측 휠허브"
        if rx > 0.72 and ry > 0.58:
            return "전방 우측 휠허브"
        if ry < 0.66:
            return "배터리 팩 사이드 패널"
        return "우측 모터 마운트 / 구동축"

    if face == "top":
        if ry < 0.28:
            return "전방 센서 어레이"
        if ry >= 0.72 and rx < 0.42:
            return "좌측 구동 모터 (상면)"
        if ry >= 0.72 and rx > 0.58:
            return "우측 구동 모터 (상면)"
        if ry >= 0.82:
            return "후방 제어 모듈"
        if 0.18 < rx < 0.82:
            return "배터리 팩"
        return "AGV 상면 차체"

    return "AGV 차체"


# ── AGV별 핫스팟 설정 ─────────────────────────────────────────────
# 물리적 일관성:
#   center(cx, cy)는 면별 base array의 해당 부품 중심과 일치
#   3D 매핑 검증: front u=cx/W, v=cy/H → x=u*4, y=(1-v)*2, z=0
#                 left  u=cx/W, v=cy/H → x=0, y=(1-v)*2, z=u*2
#                 top   u=cx/W, v=cy/H → x=u*4, y=2.0,   z=v*2

_HS_NONE   = [{"center": (200, 155), "radius": 15, "intensity": 0.06}]  # 완전 정상
_HS_SLIGHT = [{"center": (200, 155), "radius": 22, "intensity": 0.18}]  # 미미한 열

_AGV_HOTSPOT_CONFIGS: dict = {
    # ── 위험 (Critical) ─────────────────────────────────
    "AGV-03": {   # 모터 과열 + 진동 + 전류 변동
        # front: 좌측 모터 하우징 (cx=90, cy=200) — 좌측 모터 영역 정중앙
        "front": [{"center": (90,  200), "radius": 55, "intensity": 0.95}],
        # back: 섀시 전도열 (미약)
        "back":  [{"center": (200, 188), "radius": 22, "intensity": 0.28}],
        # left: 전방 모터 마운트 (이미지 좌측=AGV 전방, cx=95 → z≈0.48)
        "left":  [{"center": (95,  202), "radius": 45, "intensity": 0.70}],
        # right: 거의 정상
        "right": _HS_SLIGHT,
        # top: 좌측 모터 상면 투과열 (cx=105, cy=248 → 후방-좌 모터 영역)
        "top":   [{"center": (105, 248), "radius": 40, "intensity": 0.52}],
    },

    # ── 경고 (Warning) ──────────────────────────────────
    "AGV-07": {   # 배터리 과열 + 전압 저하
        "front": _HS_SLIGHT,
        # back: 배터리 환기구 과열 (cx=200, cy=185 → 배터리 환기구 중심)
        "back":  [{"center": (200, 185), "radius": 52, "intensity": 0.72}],
        # left: 배터리 사이드 패널 (cx=200, cy=128 → 배터리팩 패널)
        "left":  [{"center": (200, 128), "radius": 42, "intensity": 0.48}],
        # right: 대칭 (같은 배터리팩 반대쪽)
        "right": [{"center": (200, 128), "radius": 42, "intensity": 0.46}],
        # top: 배터리팩 전체 과열 (cx=200, cy=158 → 배터리 팩 중심)
        "top":   [{"center": (200, 158), "radius": 68, "intensity": 0.92}],
    },

    "AGV-11": {   # 진동 + 속도 불안정 + 전류 변동 (휠/구동계)
        # front: 좌우 앞바퀴 모두 과열 — (cx=78,cy=258), (cx=328,cy=258)
        "front": [
            {"center": (78,  258), "radius": 38, "intensity": 0.72},
            {"center": (328, 258), "radius": 35, "intensity": 0.65},
        ],
        # back: 좌우 뒷바퀴 (미러 — 이미지 좌=AGV 우)
        "back": [
            {"center": (78,  258), "radius": 28, "intensity": 0.42},
            {"center": (328, 258), "radius": 25, "intensity": 0.38},
        ],
        # left: 전방(이미지 좌, cx=82) + 후방(이미지 우, cx=318) 휠허브
        "left":  [
            {"center": (82,  258), "radius": 50, "intensity": 0.88},
            {"center": (318, 258), "radius": 38, "intensity": 0.62},
        ],
        # right: 전방(이미지 우, cx=320) + 후방(이미지 좌, cx=82) 휠허브 (미러)
        "right": [
            {"center": (320, 258), "radius": 48, "intensity": 0.85},
            {"center": (82,  258), "radius": 35, "intensity": 0.58},
        ],
        # top: 바퀴는 상면에서 거의 안 보임
        "top":   _HS_NONE,
    },
}

# 주의(Caution) AGV 번호
_CAUTION_NUMS = {2, 5, 9}


def _get_hotspot_config(agv_id: str) -> dict:
    """AGV ID → 5면 핫스팟 설정 반환."""
    if agv_id in _AGV_HOTSPOT_CONFIGS:
        return _AGV_HOTSPOT_CONFIGS[agv_id]
    num = int(agv_id.split("-")[1])
    default = [{"center": (200, 155), "radius": 28, "intensity": 0.20}] if num in _CAUTION_NUMS else _HS_NONE
    return {face: default for face in _FACE_BUILDERS}


# ── 이미지 생성 유틸 ─────────────────────────────────────────────

def _add_hotspot(arr: np.ndarray, cx: int, cy: int, radius: int, intensity: float) -> np.ndarray:
    """가우시안 블렌딩으로 열화상 핫스팟 추가 (적색→황색 계열)."""
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    sigma = radius * 0.55
    mask = np.exp(-dist ** 2 / (2 * sigma ** 2)).clip(0, 1)

    hot_r = np.clip(255 * mask * intensity, 0, 255)
    hot_g = np.clip(200 * mask * intensity * (1 - dist / (radius + 1) * 0.65), 0, 255)
    hot_b = np.clip(30  * mask * intensity * (1 - dist / (radius + 1)), 0, 255)

    blend = (mask * intensity)[..., np.newaxis]
    out = arr.astype(np.float32)
    out[..., 0] = out[..., 0] * (1 - blend[..., 0]) + hot_r * blend[..., 0]
    out[..., 1] = out[..., 1] * (1 - blend[..., 0]) + hot_g * blend[..., 0]
    out[..., 2] = out[..., 2] * (1 - blend[..., 0]) + hot_b * blend[..., 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def _generate_face_image(agv_id: str, face: str) -> Image.Image:
    """AGV ID + 면 → 열화상 PIL Image 생성."""
    builder = _FACE_BUILDERS[face]
    arr = builder()

    for hs in _get_hotspot_config(agv_id)[face]:
        arr = _add_hotspot(arr, *hs["center"], hs["radius"], hs["intensity"])

    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, _HEADER_H], fill=(0, 0, 0))
    draw.text((5, 5), f"Thermal | {agv_id}  {_FACE_LABEL_KR[face]}", fill=(190, 190, 190))
    return img


def _face_filename(agv_id: str, face: str) -> str:
    """파일명 규칙: agv03_front.png."""
    num = agv_id.split("-")[1]
    return f"agv{num}_{face}.png"


# ── 공개 API ─────────────────────────────────────────────────────

def generate_all_thermal_images(thermal_dir: Path = None) -> None:
    """AGV 12대 × 5면 = 60장 열화상 이미지 생성 (없을 때만)."""
    if thermal_dir is None:
        thermal_dir = THERMAL_DIR
    thermal_dir.mkdir(parents=True, exist_ok=True)

    agv_ids = [f"AGV-{i:02d}" for i in range(1, 13)]
    for agv_id in agv_ids:
        for face in _FACE_BUILDERS:
            path = thermal_dir / _face_filename(agv_id, face)
            if not path.exists():
                _generate_face_image(agv_id, face).save(path)


def load_agv_face_images(agv_id: str, thermal_dir: Path = None) -> dict:
    """AGV 1대의 5면 PIL Image dict 반환. 파일 없으면 즉시 생성."""
    if thermal_dir is None:
        thermal_dir = THERMAL_DIR
    thermal_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for face in _FACE_BUILDERS:
        path = thermal_dir / _face_filename(agv_id, face)
        if not path.exists():
            img = _generate_face_image(agv_id, face)
            img.save(path)
        else:
            img = Image.open(path).convert("RGB")
        result[face] = img
    return result


def _find_hotspot_peaks(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                        hot_mask: np.ndarray, n_peaks: int = 3,
                        min_dist: int = 50) -> list:
    """열화상 배열에서 개별 핫스팟 피크 좌표 목록 반환.

    Returns: [(cx, cy), ...] — 이미지 절대 좌표, 위험도 내림차순.
    """
    # 온도 강도 맵: 고온=높은 R, 낮은 B
    heat = (r.astype(float) - b.astype(float)).clip(0, None)
    heat[~hot_mask] = 0

    remaining = heat.copy()
    peaks = []
    for _ in range(n_peaks):
        if remaining.max() < 10:
            break
        max_idx = int(np.argmax(remaining))
        cy, cx = divmod(max_idx, remaining.shape[1])
        peaks.append((cx, cy))
        # 이 피크 주변 억제
        Y, X = np.ogrid[:remaining.shape[0], :remaining.shape[1]]
        suppress = (X - cx) ** 2 + (Y - cy) ** 2 < min_dist ** 2
        remaining[suppress] = 0

    return peaks


def detect_hotspot(image_input, thermal_dir: Path = None, face: str = "front") -> dict:
    """열화상 이미지에서 Hot-spot 탐지.

    Parameters
    ----------
    image_input : str(파일명) | PIL.Image | bytes
    thermal_dir : Path (기본=THERMAL_DIR)
    face        : "front"|"back"|"left"|"right"|"top"

    Returns
    -------
    dict with keys:
        hotspot_detected, thermal_risk_score, component, bbox,
        hotspot_centers [(cx,cy)…], processed_image, hot_ratio
    """
    if thermal_dir is None:
        thermal_dir = THERMAL_DIR

    if isinstance(image_input, str):
        path = thermal_dir / image_input
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:
            # agv_id 역추적 시도 (agvNN_face.png 패턴)
            parts = image_input.replace(".png", "").split("_")
            agv_num = parts[0].replace("agv", "")
            agv_id = f"AGV-{agv_num}"
            face_key = parts[1] if len(parts) > 1 else face
            img = _generate_face_image(agv_id, face_key)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(io.BytesIO(image_input)).convert("RGB")

    arr = np.array(img)
    h, w = arr.shape[:2]

    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    # 고온 마스크 (헤더 제외)
    hot_mask = (r > 100) & (r > g * 1.08) & (b < 120)
    hot_mask[:_HEADER_H, :] = False

    hot_px = int(hot_mask.sum())
    total_px = (h - _HEADER_H) * w
    hot_ratio = hot_px / (total_px + 1e-9)

    no_hotspot = dict(
        hotspot_detected=False,
        thermal_risk_score=12.0,
        component="없음",
        bbox=None,
        hotspot_centers=[],
        processed_image=img,
        hot_ratio=hot_ratio,
        face=face,
    )

    if hot_px < 60:
        return no_hotspot

    rows = np.any(hot_mask, axis=1)
    cols = np.any(hot_mask, axis=0)
    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])
    bbox = (cmin, rmin, cmax - cmin, rmax - rmin)

    # 개별 피크 탐지 (최대 3개)
    peaks = _find_hotspot_peaks(r, g, b, hot_mask, n_peaks=3, min_dist=50)

    # 대표 부품: 가장 강한 피크 기준
    cx_main, cy_main = peaks[0] if peaks else ((cmin + cmax) // 2, (rmin + rmax) // 2)
    component = _component_from_face(face, cx_main / w, cy_main / h)

    thermal_risk = float(np.clip(12 + hot_ratio * 2200, 0, 100))

    # 결과 이미지에 BBox + 레이블 그리기
    result_img = img.copy()
    draw = ImageDraw.Draw(result_img)
    draw.rectangle([cmin, rmin, cmax, rmax], outline=(0, 255, 0), width=3)
    label_y = max(_HEADER_H + 2, rmin - 18)
    label_text = f"⚠ {component}"
    draw.rectangle(
        [cmin, label_y - 2, cmin + len(label_text) * 8 + 4, label_y + 14],
        fill=(0, 0, 0),
    )
    draw.text((cmin + 2, label_y), label_text, fill=(0, 255, 0))

    # 개별 피크 마커
    for px, py in peaks:
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=(255, 80, 0), width=2)

    return dict(
        hotspot_detected=True,
        thermal_risk_score=thermal_risk,
        component=component,
        bbox=bbox,
        hotspot_centers=peaks,       # [(cx, cy), ...] 이미지 절대 좌표
        processed_image=result_img,
        hot_ratio=hot_ratio,
        face=face,
    )


def get_thermal_scores_per_agv(thermal_dir: Path = None) -> tuple:
    """AGV별 열화상 위험 점수 dict와 기본값 반환.

    모든 5면의 detect_hotspot 결과 중 최대값을 해당 AGV 점수로 사용.
    """
    if thermal_dir is None:
        thermal_dir = THERMAL_DIR
    generate_all_thermal_images(thermal_dir)

    agv_ids = [f"AGV-{i:02d}" for i in range(1, 13)]
    scores = {}
    for agv_id in agv_ids:
        face_scores = []
        for face in _FACE_BUILDERS:
            fname = _face_filename(agv_id, face)
            det = detect_hotspot(fname, thermal_dir, face=face)
            face_scores.append(det["thermal_risk_score"])
        scores[agv_id] = max(face_scores)

    # 기본값: 정상 AGV 평균
    normal_ids = [aid for aid in agv_ids if aid not in _AGV_HOTSPOT_CONFIGS]
    default_score = float(
        np.mean([scores[aid] for aid in normal_ids]) if normal_ids else 12.0
    )
    return scores, default_score
