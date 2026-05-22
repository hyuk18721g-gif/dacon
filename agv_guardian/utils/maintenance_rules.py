"""정비 우선순위 및 조치 가이드 생성 모듈."""
from __future__ import annotations
import pandas as pd

THRESHOLDS = {
    "motor_temp": 65.0,
    "battery_temp": 45.0,
    "vibration": 0.50,
    "voltage": 22.5,
    "current_instability": 0.14,
    "error_count": 3,
}

# (요인 집합) → 정비 규칙
_RULES: list[tuple[frozenset, dict]] = [
    (
        frozenset({"모터 온도 상승", "진동 증가"}),
        {
            "suspected_component": "좌측 구동 모터",
            "cause_candidates": "모터 과부하 / 베어링 마모 / 냉각팬 이상",
            "recommended_action": (
                "① 좌측 구동 모터 하우징 온도 확인\n"
                "② 베어링 마모 여부 점검\n"
                "③ 냉각팬 작동 상태 확인\n"
                "④ 정비 후 센서값 재측정"
            ),
        },
    ),
    (
        frozenset({"배터리 온도 상승", "전압 저하"}),
        {
            "suspected_component": "배터리 팩",
            "cause_candidates": "배터리 셀 열화 / 충전 모듈 이상 / 충전 단자 접촉 불량",
            "recommended_action": (
                "① 배터리 팩 표면 온도 확인\n"
                "② 충전 단자 및 케이블 점검\n"
                "③ 배터리 셀 밸런싱 상태 확인"
            ),
        },
    ),
    (
        frozenset({"진동 증가", "전류 변동성 증가"}),
        {
            "suspected_component": "우측 휠/구동부",
            "cause_candidates": "휠 마모 / 축 정렬 불량 / 구동부 마찰 증가",
            "recommended_action": (
                "① 휠 마모 상태 확인\n"
                "② 축 정렬 상태 점검\n"
                "③ 바닥 이물질 및 주행 경로 장애물 확인"
            ),
        },
    ),
    (
        frozenset({"전류 변동성 증가"}),
        {
            "suspected_component": "구동 모터/전원 시스템",
            "cause_candidates": "모터 부하 증가 / 구동 저항 증가 / 주행 경로 장애물",
            "recommended_action": (
                "① 모터 부하 상태 확인\n"
                "② 구동부 윤활 상태 점검\n"
                "③ 주행 경로 장애물 확인"
            ),
        },
    ),
    (
        frozenset({"모터 온도 상승"}),
        {
            "suspected_component": "구동 모터",
            "cause_candidates": "모터 과부하 / 냉각 시스템 이상 / 주변 온도 상승",
            "recommended_action": (
                "① 모터 하우징 온도 확인\n"
                "② 냉각팬 상태 점검\n"
                "③ 부하 감소 조치"
            ),
        },
    ),
    (
        frozenset({"배터리 온도 상승"}),
        {
            "suspected_component": "배터리 팩",
            "cause_candidates": "배터리 과충전 / 충전 모듈 이상 / 통풍 불량",
            "recommended_action": (
                "① 배터리 팩 점검\n"
                "② 충전 단자 확인\n"
                "③ 냉각 시스템 점검"
            ),
        },
    ),
    (
        frozenset({"오류 횟수 증가"}),
        {
            "suspected_component": "제어 시스템",
            "cause_candidates": "센서 오류 / 통신 장애 / 소프트웨어 이상",
            "recommended_action": (
                "① 에러 로그 분석\n"
                "② 센서 캘리브레이션\n"
                "③ 시스템 재시작 고려"
            ),
        },
    ),
]

_DEFAULT_RULE = {
    "suspected_component": "전체 시스템",
    "cause_candidates": "복합 원인 가능 / 정기 점검 필요",
    "recommended_action": "① 전체 시스템 점검 실시\n② 센서값 트렌드 모니터링",
}

_RECOMMENDED_TIME = {
    "Critical": "24시간 이내 정비 권장",
    "Warning":  "48시간 이내 점검 권장",
    "Caution":  "추세 관찰 및 다음 정기점검 시 확인",
    "Normal":   "정기 모니터링",
}

_PRIORITY_MAP = {"Critical": 1, "Warning": 2, "Caution": 3, "Normal": 4}


def get_anomaly_factors(row: pd.Series) -> list[str]:
    """행 데이터에서 이상 요인 목록 추출."""
    factors = []
    if row.get("motor_temp", 0)          >= THRESHOLDS["motor_temp"]:          factors.append("모터 온도 상승")
    if row.get("battery_temp", 0)        >= THRESHOLDS["battery_temp"]:        factors.append("배터리 온도 상승")
    if row.get("vibration", 0)           >= THRESHOLDS["vibration"]:           factors.append("진동 증가")
    if row.get("voltage", 25)            <= THRESHOLDS["voltage"]:             factors.append("전압 저하")
    if row.get("current_instability", 0) >= THRESHOLDS["current_instability"]: factors.append("전류 변동성 증가")
    if row.get("error_count", 0)         >= THRESHOLDS["error_count"]:         factors.append("오류 횟수 증가")
    return factors


def _match_rule(factors: list[str]) -> dict:
    """요인 집합에 가장 잘 맞는 정비 규칙 반환."""
    fs = frozenset(factors)
    best, best_score = _DEFAULT_RULE, 0
    for rule_key, rule_val in _RULES:
        overlap = len(rule_key & fs)
        if overlap > best_score:
            best_score = overlap
            best = rule_val
    return best


def apply_maintenance_rules(df: pd.DataFrame) -> pd.DataFrame:
    """전체 데이터프레임에 정비 규칙 적용 (AGV별 최신 행 기준)."""
    df = df.copy()

    latest = (
        df.sort_values("timestamp")
        .groupby("agv_id")
        .last()
        .reset_index()
    )

    maint_rows = []
    for _, row in latest.iterrows():
        factors = get_anomaly_factors(row)
        rule = _match_rule(factors)
        status = row.get("status", "Normal")
        maint_rows.append({
            "agv_id":               row["agv_id"],
            "main_anomaly_factors": ", ".join(factors) if factors else "없음",
            "suspected_component":  rule["suspected_component"],
            "cause_candidates":     rule["cause_candidates"],
            "recommended_action":   rule["recommended_action"],
            "recommended_time":     _RECOMMENDED_TIME.get(status, "정기 모니터링"),
            "maintenance_priority": _PRIORITY_MAP.get(status, 4),
        })

    maint_df = pd.DataFrame(maint_rows)
    df = df.merge(maint_df, on="agv_id", how="left")
    return df
