"""AGV 5면 열화상 → 3D 박스 매핑 시각화 모듈."""
from __future__ import annotations
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# AGV 박스 치수 (시각화용 단위)
BOX_W = 4.0   # 너비  (x축)
BOX_H = 2.0   # 높이  (y축)
BOX_D = 2.0   # 깊이  (z축)
GRID_N = 30   # 면당 해상도 (NxN 격자)

FACE_KR = {
    "front": "전면",
    "back":  "후면",
    "left":  "좌측면",
    "right": "우측면",
    "top":   "상면",
}

# 레이블 위치 (각 면 중심 바깥쪽)
_LABEL_POS = {
    "front": (BOX_W / 2, BOX_H / 2, -0.25),
    "back":  (BOX_W / 2, BOX_H / 2, BOX_D + 0.25),
    "left":  (-0.25,     BOX_H / 2, BOX_D / 2),
    "right": (BOX_W + 0.25, BOX_H / 2, BOX_D / 2),
    "top":   (BOX_W / 2, BOX_H + 0.25, BOX_D / 2),
}


# ── 내부 유틸 ──────────────────────────────────────────────────

def _img_to_heat(pil_img: Image.Image, n: int = GRID_N) -> np.ndarray:
    """PIL 이미지 → n×n 열화 강도 배열 (0~1).
    상단 22px 헤더 제거 후 (R-B)/255 를 강도로 사용.
    """
    w, h = pil_img.size
    cropped = pil_img.crop((0, 22, w, h))  # 헤더 22px 제거
    arr = np.array(cropped.resize((n, n)).convert("RGB"), dtype=float)
    heat = np.clip((arr[:, :, 0] - arr[:, :, 2]) / 255.0, 0, 1)
    return heat


def _face_xyz(face: str, n: int = GRID_N):
    """면 이름 → 3D 좌표 격자 (X, Y, Z) 각각 n×n ndarray."""
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, v)

    if face == "front":   # z=0 평면, x: 0→W, y: H→0(이미지 위=높이)
        return U * BOX_W, (1 - V) * BOX_H, np.zeros_like(U)
    if face == "back":    # z=D 평면, 좌우 반전 (뒤에서 보면)
        return (1 - U) * BOX_W, (1 - V) * BOX_H, np.full_like(U, BOX_D)
    if face == "left":    # x=0 평면, z: 0→D
        return np.zeros_like(U), (1 - V) * BOX_H, U * BOX_D
    if face == "right":   # x=W 평면, z: D→0 (바깥에서 보면)
        return np.full_like(U, BOX_W), (1 - V) * BOX_H, (1 - U) * BOX_D
    if face == "top":     # y=H 평면, x: 0→W, z: 0→D
        return U * BOX_W, np.full_like(U, BOX_H), V * BOX_D
    raise ValueError(f"Unknown face: {face}")


def _pixel_to_3d(face: str, cx: int, cy: int, iw: int, ih: int) -> tuple:
    """2D 픽셀 핫스팟 좌표 → 3D 박스 표면 좌표."""
    u, v = cx / iw, cy / ih
    if face == "front":  return (u * BOX_W,         (1 - v) * BOX_H, 0.0)
    if face == "back":   return ((1 - u) * BOX_W,   (1 - v) * BOX_H, BOX_D)
    if face == "left":   return (0.0,                (1 - v) * BOX_H, u * BOX_D)
    if face == "right":  return (BOX_W,              (1 - v) * BOX_H, (1 - u) * BOX_D)
    if face == "top":    return (u * BOX_W,          BOX_H,           v * BOX_D)
    raise ValueError(f"Unknown face: {face}")


def _box_wireframe() -> go.Scatter3d:
    """AGV 박스 와이어프레임 (12개 엣지)."""
    W, H, D = BOX_W, BOX_H, BOX_D
    edges = [
        # 전면 사각형
        [(0,0,0),(W,0,0)], [(W,0,0),(W,H,0)], [(W,H,0),(0,H,0)], [(0,H,0),(0,0,0)],
        # 후면 사각형
        [(0,0,D),(W,0,D)], [(W,0,D),(W,H,D)], [(W,H,D),(0,H,D)], [(0,H,D),(0,0,D)],
        # 전후 연결 4개
        [(0,0,0),(0,0,D)], [(W,0,0),(W,0,D)], [(W,H,0),(W,H,D)], [(0,H,0),(0,H,D)],
    ]
    xs, ys, zs = [], [], []
    for (x1,y1,z1), (x2,y2,z2) in edges:
        xs += [x1, x2, None]
        ys += [y1, y2, None]
        zs += [z1, z2, None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="rgba(180,180,200,0.5)", width=2),
        showlegend=False,
        hoverinfo="skip",
    )


# ── 공개 API ──────────────────────────────────────────────────

def build_3d_thermal_figure(
    face_images: dict[str, Image.Image],
    detections: dict[str, dict],
) -> go.Figure:
    """
    5면 열화상 이미지와 탐지 결과로 인터랙티브 3D Plotly Figure 생성.

    Parameters
    ----------
    face_images : {"front": PIL, "back": PIL, "left": PIL, "right": PIL, "top": PIL}
    detections  : {face: detect_hotspot() 반환 dict}

    Returns
    -------
    go.Figure  (Plotly 3D, 마우스 회전/줌 지원)
    """
    fig = go.Figure()
    fig.add_trace(_box_wireframe())

    hs_x, hs_y, hs_z, hs_text, hs_score = [], [], [], [], []

    for face in ["front", "back", "left", "right", "top"]:
        pil_img = face_images.get(face)
        if pil_img is None:
            continue

        heat = _img_to_heat(pil_img)
        X, Y, Z = _face_xyz(face)

        # 면 Surface
        show_colorbar = (face == "front")
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=heat,
            colorscale="Jet",
            cmin=0.0, cmax=0.9,
            showscale=show_colorbar,
            colorbar=dict(
                title=dict(text="열화 강도"),
                thickness=14,
                len=0.5,
                x=1.02,
                tickvals=[0, 0.45, 0.9],
                ticktext=["낮음", "중간", "높음"],
                tickfont=dict(color="white"),
            ) if show_colorbar else None,
            opacity=0.87,
            name=FACE_KR[face],
            hovertemplate=f"면: {FACE_KR[face]}<extra></extra>",
        ))

        # 면 레이블
        lx, ly, lz = _LABEL_POS[face]
        fig.add_trace(go.Scatter3d(
            x=[lx], y=[ly], z=[lz],
            mode="text",
            text=[FACE_KR[face]],
            textfont=dict(size=11, color="rgba(220,220,220,0.9)"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 핫스팟 마커 (개별 피크 지원)
        det = detections.get(face, {})
        if det.get("hotspot_detected"):
            iw, ih = pil_img.width, pil_img.height
            score = det["thermal_risk_score"]
            component = det.get("component", "미확인")

            # hotspot_centers 우선 사용, 없으면 bbox 중심 fallback
            centers = det.get("hotspot_centers", [])
            if not centers and det.get("bbox"):
                bx, by, bw, bh = det["bbox"]
                centers = [(bx + bw // 2, by + bh // 2)]

            for i, (cx, cy) in enumerate(centers):
                x3, y3, z3 = _pixel_to_3d(face, cx, cy, iw, ih)
                hs_x.append(x3);  hs_y.append(y3);  hs_z.append(z3)
                hs_score.append(score)
                hs_text.append(
                    f"<b>⚠ 열화 감지</b><br>"
                    f"면: {FACE_KR[face]}<br>"
                    f"부위: {component}<br>"
                    f"위험도: {score:.1f}"
                    + (f"<br>피크 #{i+1}" if len(centers) > 1 else "")
                )

    # 핫스팟 마커 일괄 렌더링
    if hs_x:
        # 외곽 글로우 효과 (큰 반투명 구)
        fig.add_trace(go.Scatter3d(
            x=hs_x, y=hs_y, z=hs_z,
            mode="markers",
            marker=dict(size=30, color="rgba(255,80,50,0.15)", symbol="circle"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # 실제 마커
        fig.add_trace(go.Scatter3d(
            x=hs_x, y=hs_y, z=hs_z,
            mode="markers",
            marker=dict(
                size=14,
                color=hs_score,
                colorscale=[[0, "#f39c12"], [0.5, "#e74c3c"], [1, "#8e1010"]],
                cmin=20, cmax=100,
                opacity=0.95,
                symbol="circle",
                line=dict(color="white", width=2),
            ),
            customdata=hs_text,
            hovertemplate="%{customdata}<extra></extra>",
            name="열화 감지 지점",
        ))

    # 레이아웃
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-0.6, BOX_W + 0.6],
                showbackground=False,
                showgrid=False,
                zeroline=False,
                title="",
                tickvals=[0, BOX_W],
                ticktext=["좌", "우"],
                tickfont=dict(color="rgba(180,180,180,0.7)"),
            ),
            yaxis=dict(
                range=[-0.4, BOX_H + 0.6],
                showbackground=False,
                showgrid=False,
                zeroline=False,
                title="",
                tickvals=[0, BOX_H],
                ticktext=["하", "상"],
                tickfont=dict(color="rgba(180,180,180,0.7)"),
            ),
            zaxis=dict(
                range=[-0.6, BOX_D + 0.6],
                showbackground=False,
                showgrid=False,
                zeroline=False,
                title="",
                tickvals=[0, BOX_D],
                ticktext=["전", "후"],
                tickfont=dict(color="rgba(180,180,180,0.7)"),
            ),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=2.0, y=1.4, z=2.0),
                up=dict(x=0, y=1, z=0),
            ),
        ),
        paper_bgcolor="rgba(14,17,30,1)",
        margin=dict(t=10, b=10, l=0, r=60),
        height=560,
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(20,25,45,0.85)",
            bordercolor="rgba(100,100,120,0.5)",
            borderwidth=1,
            font=dict(color="white", size=11),
        ),
    )
    return fig
