"""AGV Guardian - AI 기반 AGV 예지보전 및 물류 흐름 안정화 플랫폼."""
from __future__ import annotations
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.data_generator import generate_and_save_data
from utils.risk_scoring import (
    calculate_sensor_scores, calculate_final_risk,
    get_status, get_status_emoji, STATUS_COLORS,
)
from utils.anomaly_detection import calculate_anomaly_scores
from utils.thermal_detection import (
    generate_all_thermal_images, detect_hotspot,
    get_thermal_scores_per_agv, load_agv_face_images, THERMAL_DIR,
)
from utils.thermal_3d import build_3d_thermal_figure, FACE_KR
from utils.maintenance_rules import apply_maintenance_rules, get_anomaly_factors

# ═══════════════════════════════════════════════════════════
# 앱 설정
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AGV Guardian",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border-left: 4px solid #4a90e2;
}
.risk-critical { border-left-color: #e74c3c !important; }
.risk-warning  { border-left-color: #e67e22 !important; }
.risk-caution  { border-left-color: #f1c40f !important; }
.risk-normal   { border-left-color: #2ecc71 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 데이터 로드 및 처리
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="데이터 처리 중...")
def load_and_process_data(w_sensor: float = 0.50, w_ai: float = 0.30, w_thermal: float = 0.20):
    """더미데이터 생성 → 센서 점수 → 이상탐지 → 열화상 점수 → 최종 위험도."""
    thermal_dir = ROOT / "data" / "thermal_images"
    generate_all_thermal_images(thermal_dir)

    df = generate_and_save_data(ROOT / "data")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # current_instability 재계산 (CSV 로드 후 보정)
    df = df.sort_values(["agv_id", "timestamp"]).reset_index(drop=True)
    df["current_instability"] = df.groupby("agv_id")["current"].transform(
        lambda x: x.rolling(6, min_periods=1).std()
        / (x.rolling(6, min_periods=1).mean() + 1e-9)
    )

    df = calculate_sensor_scores(df)
    df = calculate_anomaly_scores(df)

    thermal_scores, default_score = get_thermal_scores_per_agv(thermal_dir)
    df["thermal_risk_score"] = df["agv_id"].map(thermal_scores).fillna(default_score)

    df = calculate_final_risk(df, w_sensor, w_ai, w_thermal)
    df = apply_maintenance_rules(df)
    return df


def get_latest(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp").groupby("agv_id").last().reset_index()


_VALID_FACES = {"front", "back", "left", "right", "top"}


def parse_thermal_uploads(uploaded_files) -> dict:
    """업로드된 다중 파일 목록 → {agv_id: {face: PIL.Image}}.

    파일명 규칙: agvNN_face.png (예: agv03_front.png, agv11_right.png)
    매칭되지 않는 파일은 무시됩니다.
    """
    result: dict = {}
    for f in uploaded_files:
        name = f.name.lower().replace(".jpg", ".png").replace(".jpeg", ".png")
        stem = name.replace(".png", "")
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        agv_part, face = parts[0], parts[1]
        if not agv_part.startswith("agv") or face not in _VALID_FACES:
            continue
        num_str = agv_part[3:]
        if not num_str.isdigit():
            continue
        agv_id = f"AGV-{int(num_str):02d}"
        img = Image.open(f).convert("RGB")
        if agv_id not in result:
            result[agv_id] = {}
        result[agv_id][face] = img
    return result


def thermal_scores_from_all(images_all: dict) -> dict:
    """{agv_id: {face: PIL.Image}} → {agv_id: max_thermal_risk_score}.

    각 AGV별 5면의 detect_hotspot 결과 중 최댓값을 해당 AGV 점수로 사용.
    """
    scores = {}
    for agv_id, face_dict in images_all.items():
        face_scores = []
        for face, img in face_dict.items():
            det = detect_hotspot(img, THERMAL_DIR, face=face)
            face_scores.append(det["thermal_risk_score"])
        scores[agv_id] = max(face_scores) if face_scores else 12.0
    return scores


def process_uploaded_csv(
    raw: pd.DataFrame,
    w_sensor: float,
    w_ai: float,
    w_thermal: float,
    thermal_scores_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """업로드된 CSV를 전체 파이프라인으로 처리 (LSTM 포함).

    thermal_scores_dict: {agv_id: score} — 업로드 이미지 기반 점수.
                         None이면 data/thermal_images 생성 이미지 사용.
    """
    thermal_dir = ROOT / "data" / "thermal_images"
    generate_all_thermal_images(thermal_dir)

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["agv_id", "timestamp"]).reset_index(drop=True)

    if "current_instability" not in df.columns:
        df["current_instability"] = df.groupby("agv_id")["current"].transform(
            lambda x: x.rolling(6, min_periods=1).std()
            / (x.rolling(6, min_periods=1).mean() + 1e-9)
        )

    df = calculate_sensor_scores(df)
    df = calculate_anomaly_scores(df)

    if thermal_scores_dict:
        base_scores, default_score = get_thermal_scores_per_agv(thermal_dir)
        # 업로드 점수 우선 적용, 없는 AGV는 생성 이미지 점수 사용
        merged = {**base_scores, **thermal_scores_dict}
        df["thermal_risk_score"] = df["agv_id"].map(merged).fillna(default_score)
    else:
        thermal_scores, default_score = get_thermal_scores_per_agv(thermal_dir)
        df["thermal_risk_score"] = df["agv_id"].map(thermal_scores).fillna(default_score)

    df = calculate_final_risk(df, w_sensor, w_ai, w_thermal)
    df = apply_maintenance_rules(df)
    return df


def get_active_df(w_sensor: float, w_ai: float, w_thermal: float) -> pd.DataFrame:
    """업로드 데이터가 있으면 반환, 없으면 더미 데이터 사용."""
    if st.session_state.get("processed_df") is not None:
        return st.session_state.processed_df
    return load_and_process_data(w_sensor, w_ai, w_thermal)


# ═══════════════════════════════════════════════════════════
# 세션 상태 초기화 (사이드바 렌더링 전)
# ═══════════════════════════════════════════════════════════
if "w_sensor" not in st.session_state:
    st.session_state.w_sensor  = 0.50
    st.session_state.w_ai      = 0.30
    st.session_state.w_thermal = 0.20
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "face_images_uploaded" not in st.session_state:
    st.session_state.face_images_uploaded = {}
if "thermal_images_all" not in st.session_state:
    st.session_state.thermal_images_all = {}
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ── 네비게이션 메타데이터 ─────────────────────────────────────
_NAV_ITEMS = [
    {"key": "upload",      "icon": "☁️",  "name": "데이터 업로드",    "desc": "AGV 센서/운행 데이터 업로드"},
    {"key": "overview",    "icon": "📊",  "name": "Overview",         "desc": "전체 AGV 상태 요약"},
    {"key": "monitoring",  "icon": "📡",  "name": "AGV Monitoring",   "desc": "AGV별 실시간 운행 상태 확인"},
    {"key": "risk",        "icon": "🧠",  "name": "Risk Analysis",    "desc": "고장/이상 위험도 분석"},
    {"key": "thermal",     "icon": "🌡️", "name": "Thermal Diagnosis","desc": "열화상 기반 이상 진단"},
    {"key": "maintenance", "icon": "🔧",  "name": "Maintenance Guide","desc": "정비 우선순위 및 조치 가이드"},
    {"key": "settings",    "icon": "⚙️",  "name": "Data & Settings",  "desc": "데이터 관리 및 설정"},
]

# ── 사이드바 전역 CSS ─────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════════
   사이드바 컨테이너
════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #f8f9fd !important;
    border-right: 1px solid #e2e6f0 !important;
    min-width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebarUserContent"] {
    padding: 0 14px 20px 14px !important;
}

/* ════════════════════════════════════════════
   헤더
════════════════════════════════════════════ */
.sb2-header {
    padding: 18px 2px 14px 2px;
    border-bottom: 1px solid #e4e8f2;
    margin-bottom: 10px;
}
.sb2-brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 3px;
}
.sb2-brand-left {
    display: flex;
    align-items: center;
    gap: 9px;
}
.sb2-icon-wrap {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 3px 8px rgba(124,58,237,0.30);
}
.sb2-title {
    font-size: 16px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.2px;
    line-height: 1.2;
}
.sb2-sub {
    font-size: 10.5px;
    color: #9ca3af;
    line-height: 1.3;
}
.sb2-collapse-btn {
    width: 26px; height: 26px;
    border-radius: 7px;
    background: #f1f3fa;
    border: 1px solid #e4e8f2;
    display: flex; align-items: center; justify-content: center;
    color: #9ca3af;
    font-size: 13px;
    cursor: pointer;
}
.sb2-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10.5px;
    font-weight: 500;
    margin-top: 10px;
}
.sb2-badge-upload { background:#ede9fe; color:#6d28d9; }
.sb2-badge-dummy  { background:#f3f4f6; color:#6b7280; }

/* ════════════════════════════════════════════
   섹션 레이블
════════════════════════════════════════════ */
.sb2-section-lbl {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: #c0c6d9;
    padding: 2px 2px 6px 2px;
    margin-top: 2px;
}

/* ════════════════════════════════════════════
   네비게이션 버튼 — 공통
════════════════════════════════════════════ */
[data-testid="stSidebar"] .stButton > button {
    position: relative !important;
    display: block !important;
    width: 100% !important;
    text-align: left !important;
    white-space: pre-wrap !important;
    line-height: 1.5 !important;
    padding: 11px 36px 11px 14px !important;
    margin: 3px 0 !important;
    border-radius: 11px !important;
    font-size: 13px !important;
    height: auto !important;
    min-height: 58px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    /* 비활성 기본 */
    background: #ffffff !important;
    border: 1px solid #e8eaf2 !important;
    color: #4b5563 !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06) !important;
}
/* 비활성 — 첫 번째 줄(메뉴명) 강조 */
[data-testid="stSidebar"] .stButton > button::first-line {
    font-weight: 600 !important;
    color: #1f2937 !important;
}
/* 오른쪽 화살표 */
[data-testid="stSidebar"] .stButton > button::after {
    content: "›" !important;
    position: absolute !important;
    right: 13px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-size: 19px !important;
    color: #d1d5db !important;
    font-weight: 400 !important;
    line-height: 1 !important;
}

/* ── hover (비활성) ── */
[data-testid="stSidebar"] .stButton > button:hover {
    background: #f5f3ff !important;
    border-color: #c4b5fd !important;
    color: #5b21b6 !important;
    box-shadow: 0 3px 10px rgba(124,58,237,0.10) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] .stButton > button:hover::after {
    color: #a78bfa !important;
}
[data-testid="stSidebar"] .stButton > button:hover::first-line {
    color: #4c1d95 !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: none !important;
    background: #f5f3ff !important;
    border-color: #c4b5fd !important;
}

/* ── 활성 nav 카드 (HTML div, button 아님) ── */
.sb2-nav-active {
    position: relative;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 36px 11px 14px;
    margin: 3px 0;
    border-radius: 11px;
    background: #f5f3ff;
    border: 1px solid #c4b5fd;
    border-left: 3.5px solid #7c3aed;
    box-shadow: 0 4px 14px rgba(124,58,237,0.13);
    cursor: default;
    min-height: 58px;
    box-sizing: border-box;
}
.sb2-nav-active-arrow {
    position: absolute;
    right: 13px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 19px;
    color: #7c3aed;
    line-height: 1;
    font-weight: 400;
}
.sb2-nav-icon {
    font-size: 18px;
    min-width: 22px;
    text-align: center;
    flex-shrink: 0;
}
.sb2-nav-name {
    font-size: 13px;
    font-weight: 700;
    color: #4c1d95;
    line-height: 1.3;
}
.sb2-nav-desc {
    font-size: 11px;
    color: #7c3aed;
    line-height: 1.3;
    margin-top: 1px;
}

/* ════════════════════════════════════════════
   구분선
════════════════════════════════════════════ */
.sb2-divider {
    height: 1px;
    background: #e8eaf2;
    margin: 12px 0 10px 0;
}

/* ════════════════════════════════════════════
   필터 섹션
════════════════════════════════════════════ */
.sb2-filter-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.sb2-filter-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
    color: #374151;
}
.sb2-filter-icon {
    width: 22px; height: 22px;
    background: #ede9fe;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
}
.sb2-field-lbl {
    font-size: 11px;
    font-weight: 600;
    color: #6b7280;
    margin-bottom: 4px;
    margin-top: 8px;
}
.sb2-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 2px;
}
.sb2-chip {
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.chip-normal   { background:#dcfce7; color:#166534; }
.chip-caution  { background:#fef9c3; color:#854d0e; }
.chip-warning  { background:#ffedd5; color:#9a3412; }
.chip-critical { background:#fee2e2; color:#991b1b; }

/* selectbox / multiselect */
[data-testid="stSidebar"] [data-baseweb="select"] {
    border-radius: 9px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius: 9px !important;
    border-color: #e5e7eb !important;
    background: #ffffff !important;
    font-size: 13px !important;
}

/* ════════════════════════════════════════════
   푸터
════════════════════════════════════════════ */
.sb2-footer {
    text-align: center;
    font-size: 10.5px;
    color: #d1d5db;
    padding: 12px 0 2px 0;
    border-top: 1px solid #e8eaf2;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── 사이드바 렌더링 ───────────────────────────────────────────
with st.sidebar:

    # ══ 헤더 ══════════════════════════════════════════════════
    is_uploaded = st.session_state.processed_df is not None
    badge_cls  = "sb2-badge sb2-badge-upload" if is_uploaded else "sb2-badge sb2-badge-dummy"
    badge_text = "📂 업로드 데이터 사용 중" if is_uploaded else "🔧 더미 데이터 사용 중"
    st.markdown(f"""
    <div class="sb2-header">
        <div class="sb2-brand-row">
            <div class="sb2-brand-left">
                <div class="sb2-icon-wrap">🤖</div>
                <div>
                    <div class="sb2-title">AGV Guardian</div>
                    <div class="sb2-sub">AI 기반 AGV 예지보전 플랫폼</div>
                </div>
            </div>
            <div class="sb2-collapse-btn">‹</div>
        </div>
        <span class="{badge_cls}">{badge_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ══ 네비게이션 ════════════════════════════════════════════
    st.markdown('<div class="sb2-section-lbl">메뉴</div>', unsafe_allow_html=True)
    for item in _NAV_ITEMS:
        is_active = st.session_state.page == item["key"]
        if is_active:
            # 활성 항목: 순수 HTML 카드 (CSS attribute 선택자 불필요)
            st.markdown(f"""
            <div class="sb2-nav-active">
                <span class="sb2-nav-icon">{item['icon']}</span>
                <div>
                    <div class="sb2-nav-name">{item['name']}</div>
                    <div class="sb2-nav-desc">{item['desc']}</div>
                </div>
                <span class="sb2-nav-active-arrow">›</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 비활성 항목: 클릭 가능한 버튼
            btn_label = f"{item['icon']}  {item['name']}\n{item['desc']}"
            if st.button(
                btn_label,
                key=f"nav_{item['key']}",
                use_container_width=True,
            ):
                st.session_state.page = item["key"]
                st.rerun()

    # ══ 필터 ══════════════════════════════════════════════════
    st.markdown('<div class="sb2-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb2-filter-header">
        <div class="sb2-filter-title">
            <span class="sb2-filter-icon">🔍</span> 필터
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_full = get_active_df(
        st.session_state.w_sensor,
        st.session_state.w_ai,
        st.session_state.w_thermal,
    )
    latest_df = get_latest(df_full)

    st.markdown('<div class="sb2-field-lbl">AGV 선택</div>', unsafe_allow_html=True)
    agv_list = sorted(df_full["agv_id"].unique())
    default_idx = agv_list.index("AGV-03") if "AGV-03" in agv_list else 0
    selected_agv = st.selectbox(
        "AGV 선택", agv_list, index=default_idx, label_visibility="collapsed"
    )

    st.markdown('<div class="sb2-field-lbl">상태 등급 필터</div>', unsafe_allow_html=True)
    status_filter = st.multiselect(
        "상태 등급",
        ["Normal", "Caution", "Warning", "Critical"],
        default=["Normal", "Caution", "Warning", "Critical"],
        label_visibility="collapsed",
    )
    # 상태 칩 범례
    st.markdown("""
    <div class="sb2-chips" style="margin-top:6px;">
        <span class="sb2-chip chip-normal">● Normal</span>
        <span class="sb2-chip chip-caution">● Caution</span>
        <span class="sb2-chip chip-warning">● Warning</span>
        <span class="sb2-chip chip-critical">● Critical</span>
    </div>
    """, unsafe_allow_html=True)

    latest_filtered = latest_df[latest_df["status"].isin(status_filter)]

    # ══ 푸터 ══════════════════════════════════════════════════
    st.markdown('<div class="sb2-footer">© 2025 AGV Guardian MVP</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════
def page_overview(df_full: pd.DataFrame, latest: pd.DataFrame, latest_filtered: pd.DataFrame):
    st.title("📊 AGV Fleet Overview")
    st.caption("스마트 공장 내 전체 AGV 운영 현황")

    # ── KPI 카드 ────────────────────────────────────────────
    cnt = latest["status"].value_counts()
    total = len(latest)
    n_normal   = cnt.get("Normal", 0)
    n_caution  = cnt.get("Caution", 0)
    n_warning  = cnt.get("Warning", 0)
    n_critical = cnt.get("Critical", 0)
    avg_risk   = latest["final_risk_score"].mean()
    n_inspect  = n_critical + n_warning

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("총 AGV",          f"{total}대")
    c2.metric("🟢 정상",          f"{n_normal}대")
    c3.metric("🟡 주의",          f"{n_caution}대")
    c4.metric("🟠 경고",          f"{n_warning}대")
    c5.metric("🔴 위험",          f"{n_critical}대")
    c6.metric("평균 위험도",       f"{avg_risk:.1f}")
    c7.metric("오늘 점검 권장",    f"{n_inspect}대")

    st.divider()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # 상태 분포 도넛
        status_order = ["Normal", "Caution", "Warning", "Critical"]
        labels = [f"{get_status_emoji(s)} {s}" for s in status_order]
        values = [cnt.get(s, 0) for s in status_order]
        colors = [STATUS_COLORS[s] for s in status_order]

        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.55, marker_colors=colors,
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            title="상태 등급 분포", height=300,
            margin=dict(t=40, b=0, l=0, r=0),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # 위험도 게이지
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(avg_risk, 1),
            title={"text": "평균 위험도"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#4a90e2"},
                "steps": [
                    {"range": [0,  40], "color": "#2ecc71"},
                    {"range": [40, 60], "color": "#f1c40f"},
                    {"range": [60, 80], "color": "#e67e22"},
                    {"range": [80,100], "color": "#e74c3c"},
                ],
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        # AGV별 위험도 바
        plot_df = latest_filtered.sort_values("final_risk_score", ascending=False)
        bar_colors = [STATUS_COLORS[s] for s in plot_df["status"]]

        fig_bar = go.Figure(go.Bar(
            x=plot_df["agv_id"],
            y=plot_df["final_risk_score"],
            marker_color=bar_colors,
            text=plot_df["final_risk_score"].round(1),
            textposition="outside",
        ))
        fig_bar.add_hline(y=80, line_dash="dash", line_color="#e74c3c",  annotation_text="Critical 임계값")
        fig_bar.add_hline(y=60, line_dash="dash", line_color="#e67e22", annotation_text="Warning 임계값")
        fig_bar.add_hline(y=40, line_dash="dash", line_color="#f1c40f", annotation_text="Caution 임계값")
        fig_bar.update_layout(
            title="AGV별 최종 위험도",
            yaxis_range=[0, 110],
            xaxis_title="AGV ID",
            yaxis_title="위험도 점수",
            height=430,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Top 5 위험 AGV 테이블 ────────────────────────────────
    st.subheader("🚨 위험 AGV Top 5")
    top5 = latest_df_sorted = latest.sort_values("final_risk_score", ascending=False).head(5)
    display_cols = {
        "agv_id": "AGV ID", "final_risk_score": "위험도",
        "status_kr": "상태", "main_anomaly_factors": "주요 이상 요인",
        "suspected_component": "의심 부위",
        "recommended_action": "권장 조치", "recommended_time": "권장 시점",
    }
    top5_disp = top5[list(display_cols.keys())].rename(columns=display_cols).reset_index(drop=True)
    top5_disp.index += 1
    top5_disp["위험도"] = top5_disp["위험도"].round(1)
    st.dataframe(top5_disp, use_container_width=True)

    # ── 운영 알림 ────────────────────────────────────────────
    crit_agvs = latest[latest["status"] == "Critical"]["agv_id"].tolist()
    warn_agvs = latest[latest["status"] == "Warning"]["agv_id"].tolist()
    if crit_agvs or warn_agvs:
        msgs = []
        for agv in crit_agvs:
            row = latest[latest["agv_id"] == agv].iloc[0]
            msgs.append(f"🔴 **{agv}** — {row['main_anomaly_factors']} ({row['recommended_time']})")
        for agv in warn_agvs:
            row = latest[latest["agv_id"] == agv].iloc[0]
            msgs.append(f"🟠 **{agv}** — {row['main_anomaly_factors']} ({row['recommended_time']})")
        st.warning("**운영 알림**\n\n" + "\n\n".join(msgs))


# ═══════════════════════════════════════════════════════════
# PAGE 2: AGV Monitoring
# ═══════════════════════════════════════════════════════════
def page_agv_monitoring(df_full: pd.DataFrame, selected_agv: str):
    st.title("📡 AGV Monitoring")

    agv_df   = df_full[df_full["agv_id"] == selected_agv].sort_values("timestamp")
    latest_r = agv_df.iloc[-1]

    status   = latest_r["status"]
    emoji    = get_status_emoji(status)
    color    = STATUS_COLORS[status]

    # ── 상태 카드 ────────────────────────────────────────────
    st.markdown(f"### {emoji} {selected_agv}  —  {latest_r['status_kr']} ({status})")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("최종 위험도",       f"{latest_r['final_risk_score']:.1f}")
    c2.metric("Sensor Risk",      f"{latest_r['sensor_risk_score']:.1f}")
    c3.metric("AI Anomaly Score", f"{latest_r['ai_anomaly_score']:.1f}")
    c4.metric("Thermal Risk",     f"{latest_r['thermal_risk_score']:.1f}")
    c5.metric("권장 정비",         latest_r.get("recommended_time", "-"))

    st.divider()

    # ── 시계열 그래프 ────────────────────────────────────────
    sensor_cfg = [
        ("motor_temp",   "모터 온도 (℃)",    70.0,  "°C"),
        ("battery_temp", "배터리 온도 (℃)",  45.0,  "°C"),
        ("vibration",    "진동 (g)",          0.70,  "g"),
        ("voltage",      "전압 (V)",          22.0,  "V"),
        ("current",      "전류 (A)",          None,  "A"),
        ("speed",        "속도 (m/s)",        None,  "m/s"),
    ]

    cols = st.columns(2)
    for i, (col_name, title, threshold, unit) in enumerate(sensor_cfg):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agv_df["timestamp"], y=agv_df[col_name],
            mode="lines", name=title,
            line=dict(color=color, width=2),
        ))
        if threshold is not None:
            fig.add_hline(
                y=threshold, line_dash="dash", line_color="#e74c3c",
                annotation_text=f"경고 임계값 ({threshold}{unit})",
            )
        fig.update_layout(
            title=title, height=260,
            margin=dict(t=40, b=30, l=40, r=20),
            xaxis_title="시간", yaxis_title=unit,
        )
        cols[i % 2].plotly_chart(fig, use_container_width=True)

    # ── 최근 이벤트 로그 ─────────────────────────────────────
    st.subheader("📋 최근 이상 이벤트 로그")
    factors = get_anomaly_factors(latest_r)
    if factors:
        for f in factors:
            st.markdown(f"- ⚠️ **{f}** 감지 (최근 측정 기준)")
    else:
        st.success("현재 이상 요인 없음 — 정상 범위 운행 중")

    # 최근 3시간 트렌드
    recent = agv_df.tail(18)
    mt_trend = recent["motor_temp"].iloc[-1] - recent["motor_temp"].iloc[0]
    vib_trend = recent["vibration"].iloc[-1] - recent["vibration"].iloc[0]
    if mt_trend > 3:
        st.markdown(f"- 📈 최근 3시간 모터 온도 **{mt_trend:.1f}℃ 상승** 추세 감지")
    if vib_trend > 0.05:
        st.markdown(f"- 📈 최근 3시간 진동 **{vib_trend:.3f}g 증가** 추세 감지")


# ═══════════════════════════════════════════════════════════
# PAGE 3: Risk Analysis
# ═══════════════════════════════════════════════════════════
def page_risk_analysis(df_full: pd.DataFrame, selected_agv: str):
    st.title("🧠 Risk Analysis")

    agv_df   = df_full[df_full["agv_id"] == selected_agv].sort_values("timestamp")
    latest_r = agv_df.iloc[-1]
    status   = latest_r["status"]
    color    = STATUS_COLORS[status]

    # ── 게이지 ───────────────────────────────────────────────
    col_g, col_scores = st.columns([1, 1])
    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(float(latest_r["final_risk_score"]), 1),
            title={"text": f"{selected_agv} 최종 위험도"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  40], "color": "#d5f5e3"},
                    {"range": [40, 60], "color": "#fef9e7"},
                    {"range": [60, 80], "color": "#fdebd0"},
                    {"range": [80,100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line":      {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value":     latest_r["final_risk_score"],
                },
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_scores:
        st.subheader("위험도 구성 요소")
        st.metric("Sensor Risk Score",  f"{latest_r['sensor_risk_score']:.1f}")
        st.metric("AI Anomaly Score",   f"{latest_r['ai_anomaly_score']:.1f}")
        st.metric("Thermal Risk Score", f"{latest_r['thermal_risk_score']:.1f}")
        st.metric("**Final Risk Score**", f"**{latest_r['final_risk_score']:.1f}**")

        w_s = st.session_state.w_sensor
        w_a = st.session_state.w_ai
        w_t = st.session_state.w_thermal
        st.markdown(f"""
        **위험도 산출 공식**
        ```
        Final Risk Score =
          {w_s:.2f} × Sensor Risk Score
        + {w_a:.2f} × AI Anomaly Score
        + {w_t:.2f} × Thermal Risk Score
        ```
        """)

    st.divider()

    # ── 센서별 기여도 바 차트 ────────────────────────────────
    sensor_scores = {
        "모터 온도":    latest_r.get("motor_temp_score", 0),
        "진동":        latest_r.get("vibration_score", 0),
        "전류 변동성": latest_r.get("current_instability_score", 0),
        "배터리 온도": latest_r.get("battery_temp_score", 0),
        "전압 저하":   latest_r.get("voltage_score", 0),
    }
    bar_colors_s = [
        "#e74c3c" if v >= 70 else "#e67e22" if v >= 50 else "#f1c40f" if v >= 30 else "#2ecc71"
        for v in sensor_scores.values()
    ]
    fig_contrib = go.Figure(go.Bar(
        x=list(sensor_scores.keys()),
        y=list(sensor_scores.values()),
        marker_color=bar_colors_s,
        text=[f"{v:.1f}" for v in sensor_scores.values()],
        textposition="outside",
    ))
    fig_contrib.update_layout(
        title="센서별 위험 기여도",
        yaxis_range=[0, 110],
        yaxis_title="위험 점수",
        height=350,
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

    # ── AI 설명 박스 ─────────────────────────────────────────
    st.info(
        "**AI 이상탐지 방법론 — LSTM AutoEncoder 기반 시계열 이상탐지**\n\n"
        "AGV별 최근 2시간(10분 간격 × 12 시점) 센서 시퀀스를 입력으로 받는 "
        "**LSTM AutoEncoder**가 정상 운행 패턴을 학습합니다. "
        "입력 시계열을 얼마나 잘 복원하지 못하는지를 재구성 오차(reconstruction error)로 계산하며, "
        "모터 온도·진동·전류 변동성처럼 복합적인 패턴이 정상과 달라지면 복원 오차가 증가합니다.\n\n"
        "- **Encoder LSTM** → Bottleneck 잠재 표현 → **RepeatVector** → **Decoder LSTM** → **TimeDistributed Dense**\n"
        "- 재구성 오차(MSE)를 0~100 범위로 정규화하여 AI Anomaly Score로 변환\n"
        "- 최종 AI Anomaly Score = 0.65 × LSTM 재구성 오차 점수 + 0.35 × Sensor Risk Score"
    )


# ═══════════════════════════════════════════════════════════
# PAGE 4: Thermal Diagnosis
# ═══════════════════════════════════════════════════════════

_COMP_RULES = {
    # 전면 부품
    "좌측 구동 모터": {
        "cause":  "모터 과부하 / 베어링 마모 / 냉각팬 이상",
        "action": "모터 하우징 온도 확인 → 베어링 점검 → 냉각팬 확인",
    },
    "우측 구동 모터": {
        "cause":  "모터 과부하 / 베어링 마모 / 냉각팬 이상",
        "action": "모터 하우징 온도 확인 → 베어링 점검 → 냉각팬 확인",
    },
    "전방 센서 어레이": {
        "cause":  "센서 과부하 / 이물질 접촉 / 내부 발열",
        "action": "센서 커버 청소 → 내부 온도 점검 → 펌웨어 확인",
    },
    # 후면 부품
    "제어 CPU / 메인보드": {
        "cause":  "과부하 연산 / 냉각 불량 / 전원 이상",
        "action": "제어 모듈 온도 확인 → 전원 공급 점검 → 히트싱크 청소",
    },
    "배터리 충전 포트": {
        "cause":  "충전 단자 접촉 불량 / 충전기 이상 / 접속부 산화",
        "action": "충전 단자 청소 → 접촉 저항 측정 → 충전기 교체 검토",
    },
    "배터리 환기구 (우측)": {
        "cause":  "배터리 셀 열화 / 환기구 막힘 / 과충전",
        "action": "환기구 청소 → 배터리 온도 모니터링 → 셀 밸런싱",
    },
    "배터리 환기구 (좌측)": {
        "cause":  "배터리 셀 열화 / 환기구 막힘 / 과충전",
        "action": "환기구 청소 → 배터리 온도 모니터링 → 셀 밸런싱",
    },
    # 측면 부품
    "전방 좌측 휠허브": {
        "cause":  "휠 베어링 마모 / 축 정렬 불량 / 이물질 끼임",
        "action": "휠허브 온도 측정 → 베어링 점검 → 휠 정렬 확인",
    },
    "후방 좌측 휠허브": {
        "cause":  "휠 베어링 마모 / 축 정렬 불량 / 이물질 끼임",
        "action": "휠허브 온도 측정 → 베어링 점검 → 휠 정렬 확인",
    },
    "전방 우측 휠허브": {
        "cause":  "휠 베어링 마모 / 축 정렬 불량 / 이물질 끼임",
        "action": "휠허브 온도 측정 → 베어링 점검 → 휠 정렬 확인",
    },
    "후방 우측 휠허브": {
        "cause":  "휠 베어링 마모 / 축 정렬 불량 / 이물질 끼임",
        "action": "휠허브 온도 측정 → 베어링 점검 → 휠 정렬 확인",
    },
    "배터리 팩 사이드 패널": {
        "cause":  "배터리 팩 과열 / 셀 불균형 / 냉각 경로 막힘",
        "action": "배터리 온도 확인 → 셀 밸런싱 → 냉각 경로 점검",
    },
    "좌측 모터 마운트 / 구동축": {
        "cause":  "모터 열전달 / 구동축 마찰 / 베어링 이상",
        "action": "구동축 온도 측정 → 윤활 상태 확인 → 베어링 교환",
    },
    "우측 모터 마운트 / 구동축": {
        "cause":  "모터 열전달 / 구동축 마찰 / 베어링 이상",
        "action": "구동축 온도 측정 → 윤활 상태 확인 → 베어링 교환",
    },
    # 상면 부품
    "배터리 팩": {
        "cause":  "배터리 셀 열화 / 충전 모듈 이상 / 과방전",
        "action": "배터리 팩 온도 확인 → 셀 전압 측정 → 팩 교체 검토",
    },
    "좌측 구동 모터 (상면)": {
        "cause":  "모터 과열 전도 / 냉각 불량",
        "action": "모터 하우징 점검 → 냉각 계통 확인",
    },
    "우측 구동 모터 (상면)": {
        "cause":  "모터 과열 전도 / 냉각 불량",
        "action": "모터 하우징 점검 → 냉각 계통 확인",
    },
    "후방 제어 모듈": {
        "cause":  "제어부 발열 / 냉각 불량",
        "action": "제어 모듈 온도 확인 → 히트싱크 점검",
    },
}

_FACE_UPLOAD_LABELS = {
    "front": "전면 (Front)",
    "back":  "후면 (Back)",
    "left":  "좌측면 (Left)",
    "right": "우측면 (Right)",
    "top":   "상면 (Top)",
}


def page_thermal_diagnosis():
    st.title("🌡️ Thermal Diagnosis")
    st.caption("AGV 5면 열화상 이미지 기반 3D 열화 위치 진단")

    generate_all_thermal_images(THERMAL_DIR)

    # 업로드 페이지에서 이미 올린 이미지가 있으면 바로 사용
    session_imgs: dict = st.session_state.get("face_images_uploaded", {})

    if session_imgs:
        st.info(f"업로드 페이지에서 등록된 이미지 {len(session_imgs)}면을 사용합니다.")
        face_images = session_imgs
        if st.button("이미지 변경 (재업로드)"):
            st.session_state.face_images_uploaded = {}
            st.rerun()
        selected_agv_for_3d = None  # 업로드 모드에서는 AGV 선택 없음
    else:
        mode_3d = st.radio(
            "입력 방식",
            ["AGV별 5면 데모", "5면 이미지 직접 업로드"],
            horizontal=True,
        )
        face_images = {}

        if mode_3d == "AGV별 5면 데모":
            col_sel, col_info = st.columns([2, 5])
            with col_sel:
                # 업로드된 이미지가 있으면 그 AGV만 선택 가능
                uploaded_all = st.session_state.get("thermal_images_all", {})
                if uploaded_all:
                    agv_ids_all = sorted(uploaded_all.keys())
                    st.caption("📁 업로드된 이미지 사용")
                else:
                    agv_ids_all = [f"AGV-{i:02d}" for i in range(1, 13)]
                default_idx = agv_ids_all.index("AGV-03") if "AGV-03" in agv_ids_all else 0
                selected_agv_for_3d = st.selectbox("AGV 선택", agv_ids_all, index=default_idx)
            with col_info:
                _ANOMALY_NOTE = {
                    "AGV-03": "⚠️ 모터 과열+진동 — 전면 좌측 모터 & 좌측면 모터 마운트 핫스팟",
                    "AGV-07": "⚠️ 배터리 과열+전압저하 — 상면 배터리팩 & 후면 환기구 핫스팟",
                    "AGV-11": "⚠️ 휠 진동+속도불안 — 좌/우측면 양쪽 휠허브 핫스팟",
                }
                note = _ANOMALY_NOTE.get(selected_agv_for_3d, "✅ 정상 운행 범위")
                src_label = "📁 업로드 이미지" if (uploaded_all and selected_agv_for_3d in uploaded_all) else "🔧 자동 생성 이미지"
                st.info(f"{note}  |  {src_label}")

            # 업로드된 이미지가 있으면 우선 사용, 없으면 자동 생성
            if uploaded_all and selected_agv_for_3d in uploaded_all:
                face_images = uploaded_all[selected_agv_for_3d]
                missing = [f for f in ["front","back","left","right","top"] if f not in face_images]
                if missing:
                    with st.spinner(f"누락된 면 자동 생성: {missing}"):
                        generated = load_agv_face_images(selected_agv_for_3d, THERMAL_DIR)
                        face_images = {**generated, **face_images}
            else:
                with st.spinner(f"{selected_agv_for_3d} 열화상 이미지 로딩 중..."):
                    face_images = load_agv_face_images(selected_agv_for_3d, THERMAL_DIR)

            # 5면 썸네일 표시
            thumb_cols = st.columns(5)
            for col, (face, img) in zip(thumb_cols, face_images.items()):
                with col:
                    st.caption(_FACE_UPLOAD_LABELS[face])
                    st.image(img, use_container_width=True)

        else:
            selected_agv_for_3d = None
            st.markdown("##### 각 면 이미지 업로드")
            st.caption("업로드하지 않은 면은 분석에서 제외됩니다.")
            cols = st.columns(5)
            for col, face in zip(cols, _FACE_UPLOAD_LABELS):
                with col:
                    up = st.file_uploader(
                        _FACE_UPLOAD_LABELS[face],
                        type=["png", "jpg", "jpeg"],
                        key=f"up3d_{face}",
                    )
                    if up:
                        face_images[face] = Image.open(up).convert("RGB")
                        st.image(face_images[face], use_container_width=True)

    if not face_images:
        st.warning("이미지를 업로드해 주세요.")
        return

    with st.spinner("열화 탐지 중..."):
        detections = {
            face: detect_hotspot(img, THERMAL_DIR, face=face)
            for face, img in face_images.items()
        }

    # ── 탐지 요약 ────────────────────────────────────────────
    detected_faces = [f for f, d in detections.items() if d["hotspot_detected"]]
    if detected_faces:
        st.error(f"⚠️ **열화 감지된 면:** {' / '.join(FACE_KR[f] for f in detected_faces)}")
    else:
        st.success("모든 면에서 유의미한 열화가 탐지되지 않았습니다.")

    # ── 3D 시각화 ────────────────────────────────────────────
    st.markdown("##### 3D 열화 위치 맵 — 마우스로 회전·줌 가능")
    with st.spinner("3D 렌더링 중..."):
        fig3d = build_3d_thermal_figure(face_images, detections)
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption("🔴 마커 = 열화 감지 지점  |  파랑→빨강: 온도 낮음→높음  |  마우스 드래그로 회전, 스크롤로 줌")

    # ── 면별 상세 ────────────────────────────────────────────
    st.divider()
    st.markdown("##### 면별 탐지 상세")
    detail_cols = st.columns(len(face_images))
    for col, (face, det) in zip(detail_cols, detections.items()):
        with col:
            st.markdown(f"**{FACE_KR[face]}**")
            if det["hotspot_detected"]:
                rule = _COMP_RULES.get(
                    det["component"],
                    {"cause": "복합 원인", "action": "전체 점검"},
                )
                st.metric("위험도", f"{det['thermal_risk_score']:.1f}")
                st.markdown(f"부위: **{det['component']}**")
                st.markdown(f"원인: {rule['cause']}")
            else:
                st.metric("위험도", f"{det['thermal_risk_score']:.1f}")
                st.markdown("정상 범위")


# ═══════════════════════════════════════════════════════════
# PAGE 5: Maintenance Guide
# ═══════════════════════════════════════════════════════════
def page_maintenance_guide(latest_df: pd.DataFrame, selected_agv: str):
    st.title("🔧 Maintenance Guide")
    st.caption("정비 우선순위 및 상세 조치 가이드")

    # ── 정비 우선순위 테이블 ─────────────────────────────────
    st.subheader("정비 우선순위 테이블")

    table_cols = [
        "maintenance_priority", "agv_id", "final_risk_score", "status_kr",
        "main_anomaly_factors", "suspected_component",
        "cause_candidates", "recommended_time",
    ]
    rename_map = {
        "maintenance_priority": "우선순위",
        "agv_id": "AGV ID",
        "final_risk_score": "위험도",
        "status_kr": "상태",
        "main_anomaly_factors": "주요 이상 요인",
        "suspected_component": "의심 부위",
        "cause_candidates": "원인 후보",
        "recommended_time": "권장 시점",
    }
    exist_cols = [c for c in table_cols if c in latest_df.columns]
    maint_table = (
        latest_df[exist_cols]
        .sort_values("final_risk_score", ascending=False)
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )
    maint_table.index += 1
    maint_table["위험도"] = maint_table["위험도"].round(1)
    st.dataframe(maint_table, use_container_width=True)

    # CSV 다운로드
    csv_bytes = maint_table.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "📥 Download Maintenance Report (CSV)",
        data=csv_bytes,
        file_name="agv_maintenance_report.csv",
        mime="text/csv",
    )

    st.divider()

    # ── 선택 AGV 상세 가이드 ─────────────────────────────────
    st.subheader(f"📋 {selected_agv} 상세 정비 가이드")
    row = latest_df[latest_df["agv_id"] == selected_agv]
    if row.empty:
        st.warning(f"{selected_agv} 데이터를 찾을 수 없습니다.")
        return

    r = row.iloc[0]
    status  = r["status"]
    emoji   = get_status_emoji(status)
    color   = STATUS_COLORS[status]

    st.markdown(f"""
**위험 등급:** {emoji} {r.get('status_kr', status)} ({status})
**최종 위험도:** {r['final_risk_score']:.1f} / 100
**의심 부위:** {r.get('suspected_component', '-')}
**주요 이상 요인:** {r.get('main_anomaly_factors', '없음')}
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**원인 후보**")
        cause_str = r.get("cause_candidates", "")
        for i, c in enumerate(cause_str.split(" / "), 1):
            st.markdown(f"{i}. {c}")

    with col2:
        st.markdown("**권장 조치**")
        action_str = r.get("recommended_action", "")
        st.markdown(action_str)

    st.markdown(f"\n**권장 정비 시점:** ⏰ {r.get('recommended_time', '-')}")


# ═══════════════════════════════════════════════════════════
# PAGE 0: 데이터 업로드 (랜딩)
# ═══════════════════════════════════════════════════════════
_REQUIRED_COLS = ["timestamp", "agv_id", "motor_temp", "battery_temp",
                  "vibration", "voltage", "current"]

_FACE_UPLOAD_LABELS_INIT = {
    "front": "전면", "back": "후면",
    "left": "좌측면", "right": "우측면", "top": "상면",
}


def page_data_upload():
    st.title("🚀 데이터 업로드")
    st.caption("AGV 센서 CSV와 열화상 5면 이미지를 업로드하여 분석을 시작합니다.")

    # ── 이미 로드된 경우 ─────────────────────────────────────
    if st.session_state.get("processed_df") is not None:
        n_agv = st.session_state.processed_df["agv_id"].nunique()
        n_row = len(st.session_state.processed_df)
        n_img = len(st.session_state.face_images_uploaded)
        st.success(
            f"✅ 데이터 로드 완료 — AGV {n_agv}대 / {n_row:,}행"
            + (f" / 열화상 {n_img}면 업로드됨" if n_img else "")
        )
        st.info("사이드바 메뉴에서 **📊 Overview** 등 다른 페이지로 이동하세요.")
        if st.button("🔄 데이터 초기화 (더미 데이터로 되돌리기)"):
            st.session_state.processed_df = None
            st.session_state.face_images_uploaded = {}
            st.session_state.thermal_images_all = {}
            st.rerun()
        return

    st.markdown("---")
    col_csv, col_img = st.columns([1, 1], gap="large")

    # ── CSV 업로드 ────────────────────────────────────────────
    with col_csv:
        st.subheader("📊 AGV 센서 데이터 (CSV)")
        st.caption(f"필수 컬럼: `{'`, `'.join(_REQUIRED_COLS)}`")
        uploaded_csv = st.file_uploader(
            "CSV 파일 선택", type=["csv"], key="init_csv",
            label_visibility="collapsed",
        )
        raw_df = None
        if uploaded_csv:
            try:
                raw_df = pd.read_csv(uploaded_csv)
                missing = [c for c in _REQUIRED_COLS if c not in raw_df.columns]
                if missing:
                    st.error(f"필수 컬럼 누락: `{missing}`")
                    raw_df = None
                else:
                    st.success(
                        f"✅ AGV **{raw_df['agv_id'].nunique()}**대 / "
                        f"**{len(raw_df):,}**행 확인"
                    )
                    st.dataframe(raw_df.head(4), use_container_width=True)
            except Exception as e:
                st.error(f"파일 읽기 오류: {e}")

    # ── 열화상 이미지 폴더 업로드 ────────────────────────────────
    with col_img:
        st.subheader("🌡️ 열화상 이미지 (선택)")
        st.caption(
            "파일명 규칙 `agvNN_face.png` 로 저장된 이미지를 **한꺼번에 선택**하세요.\n\n"
            "예) `agv03_front.png`, `agv07_top.png` — "
            "업로드하지 않으면 기본 더미 이미지로 분석합니다."
        )
        uploaded_thermal_files = st.file_uploader(
            "이미지 선택 (여러 파일 동시 선택 가능)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="init_thermal_all",
            label_visibility="collapsed",
        )

        thermal_images_all_upload: dict = {}
        if uploaded_thermal_files:
            thermal_images_all_upload = parse_thermal_uploads(uploaded_thermal_files)
            n_agvs = len(thermal_images_all_upload)
            n_faces = sum(len(v) for v in thermal_images_all_upload.values())
            st.success(f"✅ AGV **{n_agvs}**대 / 이미지 **{n_faces}**장 인식")

            # 인식된 AGV별 면 수 표시
            agv_summary = {
                agv_id: list(faces.keys())
                for agv_id, faces in sorted(thermal_images_all_upload.items())
            }
            with st.expander("인식 결과 확인", expanded=False):
                for agv_id, faces in agv_summary.items():
                    face_str = " · ".join(faces)
                    ok = len(faces) == 5
                    icon = "✅" if ok else "⚠️"
                    st.markdown(f"{icon} **{agv_id}**: {face_str}")

            # 경고 AGV 썸네일 미리보기
            warn_agvs = ["AGV-03", "AGV-07", "AGV-11"]
            preview_agvs = [a for a in warn_agvs if a in thermal_images_all_upload]
            if preview_agvs:
                st.caption("경고 AGV 전면 미리보기")
                pcols = st.columns(len(preview_agvs))
                for pcol, agv_id in zip(pcols, preview_agvs):
                    with pcol:
                        face_dict = thermal_images_all_upload[agv_id]
                        preview_img = face_dict.get("front") or next(iter(face_dict.values()))
                        st.image(preview_img, caption=agv_id, use_container_width=True)
        else:
            st.info("업로드하지 않으면 자동 생성 더미 이미지를 사용합니다.")

    st.markdown("---")

    # ── 분석 시작 버튼 ────────────────────────────────────────
    btn_disabled = raw_df is None
    if btn_disabled:
        st.info("CSV 파일을 업로드하면 분석 시작 버튼이 활성화됩니다.")

    if st.button("🚀 분석 시작", type="primary",
                 use_container_width=True, disabled=btn_disabled):
        generate_all_thermal_images(ROOT / "data" / "thermal_images")

        # 업로드 이미지 기반 AGV별 열화 점수 계산
        thermal_scores_dict = None
        if thermal_images_all_upload:
            with st.spinner(f"열화상 분석 중... ({len(thermal_images_all_upload)}대 × 5면)"):
                thermal_scores_dict = thermal_scores_from_all(thermal_images_all_upload)

        with st.spinner("AI 이상탐지 및 위험도 계산 중... (최초 1회 LSTM 학습, 약 1~2분)"):
            processed = process_uploaded_csv(
                raw_df,
                st.session_state.w_sensor,
                st.session_state.w_ai,
                st.session_state.w_thermal,
                thermal_scores_dict=thermal_scores_dict,
            )

        st.session_state.processed_df = processed
        st.session_state.thermal_images_all = thermal_images_all_upload
        st.session_state.face_images_uploaded = {}  # 단일-AGV 업로드는 초기화
        st.success("✅ 분석 완료! 사이드바에서 **📊 Overview**로 이동하세요.")
        st.rerun()


# ═══════════════════════════════════════════════════════════
# PAGE 6: Data & Settings
# ═══════════════════════════════════════════════════════════
def page_data_settings(df_full: pd.DataFrame):
    st.title("⚙️ Data & Settings")

    # ── 데이터 출처 표시 ─────────────────────────────────────
    if st.session_state.get("processed_df") is not None:
        st.info("📂 현재 업로드된 데이터를 사용 중입니다. 데이터를 바꾸려면 **🚀 데이터 업로드** 페이지를 이용하세요.")
    else:
        st.info("🔧 현재 더미 데이터를 사용 중입니다. 실제 데이터는 **🚀 데이터 업로드** 페이지에서 업로드하세요.")

    st.divider()

    # ── 위험도 가중치 조정 ────────────────────────────────────
    st.subheader("⚖️ 위험도 가중치 조정")
    st.caption("가중치 합이 1.00이 되어야 합니다.")

    col1, col2, col3 = st.columns(3)
    new_ws = col1.slider("Sensor Risk Weight",  0.0, 1.0, st.session_state.w_sensor,  0.05)
    new_wa = col2.slider("AI Anomaly Weight",   0.0, 1.0, st.session_state.w_ai,      0.05)
    new_wt = col3.slider("Thermal Risk Weight", 0.0, 1.0, st.session_state.w_thermal, 0.05)

    total_w = new_ws + new_wa + new_wt
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"가중치 합 = {total_w:.2f} (1.00이 아님). 저장 시 자동 정규화됩니다.")

    if st.button("가중치 적용", type="primary"):
        if total_w > 0:
            st.session_state.w_sensor  = new_ws / total_w
            st.session_state.w_ai      = new_wa / total_w
            st.session_state.w_thermal = new_wt / total_w
            msg = (f"Sensor={st.session_state.w_sensor:.2f}, "
                   f"AI={st.session_state.w_ai:.2f}, "
                   f"Thermal={st.session_state.w_thermal:.2f}")
            if st.session_state.get("processed_df") is not None:
                # 업로드 데이터: LSTM 재학습 없이 final_risk만 재계산
                updated = calculate_final_risk(
                    st.session_state.processed_df,
                    st.session_state.w_sensor,
                    st.session_state.w_ai,
                    st.session_state.w_thermal,
                )
                st.session_state.processed_df = apply_maintenance_rules(updated)
            else:
                st.cache_data.clear()
            st.success(f"가중치 적용 완료: {msg}")
            st.rerun()
        else:
            st.error("가중치 합이 0이 될 수 없습니다.")

    st.divider()

    # ── 데이터 미리보기 ──────────────────────────────────────
    st.subheader("📊 현재 데이터 미리보기")
    view_cols = ["timestamp", "agv_id", "motor_temp", "battery_temp",
                 "vibration", "voltage", "current", "sensor_risk_score",
                 "ai_anomaly_score", "thermal_risk_score", "final_risk_score", "status"]
    exist_view = [c for c in view_cols if c in df_full.columns]
    st.dataframe(
        df_full[exist_view].sort_values(["agv_id", "timestamp"]).head(50),
        use_container_width=True,
    )
    st.caption(f"총 {len(df_full):,}행 × {len(df_full.columns)}열  |  AGV {df_full['agv_id'].nunique()}대")


# ═══════════════════════════════════════════════════════════
# 라우팅
# ═══════════════════════════════════════════════════════════
_PAGE = st.session_state.page
if _PAGE == "upload":
    page_data_upload()
elif _PAGE == "overview":
    page_overview(df_full, latest_df, latest_filtered)
elif _PAGE == "monitoring":
    page_agv_monitoring(df_full, selected_agv)
elif _PAGE == "risk":
    page_risk_analysis(df_full, selected_agv)
elif _PAGE == "thermal":
    page_thermal_diagnosis()
elif _PAGE == "maintenance":
    page_maintenance_guide(latest_df, selected_agv)
else:
    page_data_settings(df_full)
