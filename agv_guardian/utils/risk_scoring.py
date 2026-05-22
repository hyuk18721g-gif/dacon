"""센서 기반 위험도 산출 모듈."""
import numpy as np
import pandas as pd


# ─── 개별 센서 → 위험 점수 변환 ───────────────────────────────────────────────

def _score_motor_temp(v: float) -> float:
    if v <= 45:   return float(np.interp(v, [20, 45], [0, 10]))
    elif v <= 60: return float(np.interp(v, [45, 60], [10, 40]))
    elif v <= 70: return float(np.interp(v, [60, 70], [40, 70]))
    else:         return float(np.clip(np.interp(v, [70, 90], [70, 100]), 0, 100))


def _score_battery_temp(v: float) -> float:
    if v <= 35:   return float(np.interp(v, [15, 35], [0, 10]))
    elif v <= 45: return float(np.interp(v, [35, 45], [10, 50]))
    elif v <= 55: return float(np.interp(v, [45, 55], [50, 85]))
    else:         return float(np.clip(np.interp(v, [55, 65], [85, 100]), 0, 100))


def _score_vibration(v: float) -> float:
    if v <= 0.30:  return float(np.interp(v, [0, 0.30], [0, 10]))
    elif v <= 0.50: return float(np.interp(v, [0.30, 0.50], [10, 40]))
    elif v <= 0.70: return float(np.interp(v, [0.50, 0.70], [40, 75]))
    else:           return float(np.clip(np.interp(v, [0.70, 1.20], [75, 100]), 0, 100))


def _score_voltage(v: float) -> float:
    if v >= 24:   return float(np.interp(v, [24, 28], [10, 0]))
    elif v >= 22: return float(np.interp(v, [22, 24], [40, 10]))
    elif v >= 21: return float(np.interp(v, [21, 22], [75, 40]))
    else:         return float(np.clip(np.interp(v, [18, 21], [100, 75]), 0, 100))


def _score_current_instability(cv: float) -> float:
    """전류 변동계수(CV) → 위험 점수 (비선형 스케일)."""
    if cv <= 0.04:   return float(np.interp(cv, [0, 0.04], [0, 10]))
    elif cv <= 0.12: return float(np.interp(cv, [0.04, 0.12], [10, 40]))
    elif cv <= 0.25: return float(np.interp(cv, [0.12, 0.25], [40, 75]))
    else:            return float(np.clip(np.interp(cv, [0.25, 0.55], [75, 100]), 0, 100))


def _score_error_count(count: float) -> float:
    return float(np.clip(count * 10, 0, 100))


# ─── 데이터프레임 단위 계산 ────────────────────────────────────────────────────

def calculate_sensor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """각 row에 대해 센서별 점수 및 sensor_risk_score 계산."""
    df = df.copy()

    df["motor_temp_score"]          = df["motor_temp"].apply(_score_motor_temp)
    df["battery_temp_score"]        = df["battery_temp"].apply(_score_battery_temp)
    df["vibration_score"]           = df["vibration"].apply(_score_vibration)
    df["voltage_score"]             = df["voltage"].apply(_score_voltage)
    df["current_instability_score"] = df["current_instability"].apply(_score_current_instability)
    df["error_score"]               = df["error_count"].apply(_score_error_count)

    df["sensor_risk_score"] = (
        0.30 * df["motor_temp_score"]
        + 0.25 * df["vibration_score"]
        + 0.20 * df["current_instability_score"]
        + 0.15 * df["battery_temp_score"]
        + 0.10 * df["voltage_score"]
    ) * 0.90 + df["error_score"] * 0.10

    df["sensor_risk_score"] = df["sensor_risk_score"].clip(0, 100)
    return df


def calculate_final_risk(
    df: pd.DataFrame,
    w_sensor: float = 0.50,
    w_ai: float = 0.30,
    w_thermal: float = 0.20,
) -> pd.DataFrame:
    """최종 위험도 및 상태 등급 계산."""
    df = df.copy()
    df["final_risk_score"] = (
        w_sensor * df["sensor_risk_score"]
        + w_ai * df["ai_anomaly_score"]
        + w_thermal * df["thermal_risk_score"]
    ).clip(0, 100)

    df["status"]    = df["final_risk_score"].apply(get_status)
    df["status_kr"] = df["final_risk_score"].apply(get_status_korean)
    return df


# ─── 상태 등급 유틸 ────────────────────────────────────────────────────────────

def get_status(score: float) -> str:
    if score < 40:  return "Normal"
    elif score < 60: return "Caution"
    elif score < 80: return "Warning"
    else:            return "Critical"


def get_status_korean(score: float) -> str:
    return {"Normal": "정상", "Caution": "주의", "Warning": "경고", "Critical": "위험"}[get_status(score)]


def get_status_emoji(status: str) -> str:
    return {"Normal": "🟢", "Caution": "🟡", "Warning": "🟠", "Critical": "🔴"}.get(status, "⚪")


STATUS_COLORS = {
    "Normal":   "#2ecc71",
    "Caution":  "#f1c40f",
    "Warning":  "#e67e22",
    "Critical": "#e74c3c",
}
