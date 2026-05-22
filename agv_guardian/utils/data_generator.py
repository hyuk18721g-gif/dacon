"""더미 AGV 센서 데이터 생성 모듈."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


THERMAL_IMAGE_MAP = {
    "AGV-03": "agv03_left_motor_hotspot.png",
    "AGV-07": "agv07_battery_hotspot.png",
    "AGV-11": "agv11_wheel_hotspot.png",
}
DEFAULT_THERMAL = "normal_agv.png"


def _gen_agv03(agv_id: str, timestamp, t: float) -> dict:
    """AGV-03: Critical - 모터 과열 + 진동 증가 + 전류 변동성."""
    motor_temp = 60 + 33 * t + np.random.normal(0, 1.5)   # 60→93℃
    battery_temp = 33 + 3 * t + np.random.normal(0, 0.8)
    vibration = 0.50 + 0.55 * t + np.random.normal(0, 0.04)  # 0.50→1.05g
    voltage = 24.5 - 0.8 * t + np.random.normal(0, 0.2)
    current = (13 + 7 * t) + np.random.normal(0, 0.8 + 8.5 * t)  # 변동성 크게 증가
    speed = 1.2 + np.random.normal(0, 0.15)
    error_count = int(np.random.poisson(max(0.1, t * 5)))
    load_weight = 150 + np.random.normal(0, 10)
    driving_time = 8 + t * 16 + np.random.normal(0, 0.5)
    return dict(
        timestamp=timestamp, agv_id=agv_id,
        motor_temp=np.clip(motor_temp, 20, 100),
        battery_temp=np.clip(battery_temp, 15, 70),
        vibration=np.clip(vibration, 0, 2),
        voltage=np.clip(voltage, 18, 30),
        current=np.clip(current, 0, 40),
        driving_time=np.clip(driving_time, 0, 24),
        load_weight=np.clip(load_weight, 0, 500),
        speed=np.clip(speed, 0, 3),
        error_count=error_count,
        thermal_image="agv03_left_motor_hotspot.png",
    )


def _gen_agv07(agv_id: str, timestamp, t: float) -> dict:
    """AGV-07: Warning - 배터리 과열 + 전압 저하 + 오류 증가."""
    motor_temp = 48 + 7 * t + np.random.normal(0, 1.2)
    battery_temp = 44 + 20 * t + np.random.normal(0, 1.5)   # 44→64℃
    vibration = 0.25 + 0.18 * t + np.random.normal(0, 0.02)
    voltage = 24.0 - 3.5 * t + np.random.normal(0, 0.3)     # 24→20.5V
    current = 12 + 2 * t + np.random.normal(0, 0.8 + 2.0 * t)
    speed = 1.1 + np.random.normal(0, 0.1)
    error_count = int(np.random.poisson(max(0.1, t * 6)))
    load_weight = 120 + np.random.normal(0, 8)
    driving_time = 6 + t * 14 + np.random.normal(0, 0.4)
    return dict(
        timestamp=timestamp, agv_id=agv_id,
        motor_temp=np.clip(motor_temp, 20, 100),
        battery_temp=np.clip(battery_temp, 15, 70),
        vibration=np.clip(vibration, 0, 2),
        voltage=np.clip(voltage, 18, 30),
        current=np.clip(current, 0, 40),
        driving_time=np.clip(driving_time, 0, 24),
        load_weight=np.clip(load_weight, 0, 500),
        speed=np.clip(speed, 0, 3),
        error_count=error_count,
        thermal_image="agv07_battery_hotspot.png",
    )


def _gen_agv11(agv_id: str, timestamp, t: float) -> dict:
    """AGV-11: Warning - 진동 + 속도 불안정 + 전류 변동성."""
    motor_temp = 52 + 35 * t + np.random.normal(0, 1.5)   # 52→87℃
    battery_temp = 34 + 5 * t + np.random.normal(0, 0.8)
    vibration = 0.50 + 0.52 * t + np.random.normal(0, 0.04)  # 0.50→1.02g
    voltage = 24.2 - 1.2 * t + np.random.normal(0, 0.3)
    current = (13 + 4 * t) + np.random.normal(0, 1.0 + 6.5 * t)  # 고변동성
    speed = 1.1 + np.random.normal(0, 0.05 + 0.45 * t)
    error_count = int(np.random.poisson(max(0.1, t * 2)))
    load_weight = 180 + np.random.normal(0, 15)
    driving_time = 9 + t * 15 + np.random.normal(0, 0.5)
    return dict(
        timestamp=timestamp, agv_id=agv_id,
        motor_temp=np.clip(motor_temp, 20, 100),
        battery_temp=np.clip(battery_temp, 15, 70),
        vibration=np.clip(vibration, 0, 2),
        voltage=np.clip(voltage, 18, 30),
        current=np.clip(current, 0, 40),
        driving_time=np.clip(driving_time, 0, 24),
        load_weight=np.clip(load_weight, 0, 500),
        speed=np.clip(speed, 0, 3),
        error_count=error_count,
        thermal_image="agv11_wheel_hotspot.png",
    )


def _gen_normal_agv(agv_id: str, agv_num: int, timestamp, t: float) -> dict:
    """정상 또는 주의 수준 AGV 데이터 생성."""
    caution_agvs = {2, 5, 9}
    if agv_num in caution_agvs:
        motor_temp = 44 + 10 * t + np.random.normal(0, 1.5)
        battery_temp = 32 + 8 * t + np.random.normal(0, 1.0)
        vibration = 0.30 + 0.18 * t + np.random.normal(0, 0.03)
        voltage = 24.5 - 1.2 * t + np.random.normal(0, 0.3)
        current = 11 + 2.0 * t + np.random.normal(0, 0.7)
    else:
        motor_temp = 36 + 6 * t + np.random.normal(0, 1.0)
        battery_temp = 27 + 4 * t + np.random.normal(0, 0.7)
        vibration = 0.12 + 0.10 * t + np.random.normal(0, 0.02)
        voltage = 25.2 - 0.4 * t + np.random.normal(0, 0.2)
        current = 10 + 1.0 * t + np.random.normal(0, 0.5)

    speed = 1.2 + np.random.normal(0, 0.08)
    error_count = int(np.random.poisson(max(0.05, t * 0.5)))
    load_weight = 100 + agv_num * 5 + np.random.normal(0, 10)
    driving_time = 5 + t * 12 + np.random.normal(0, 0.4)
    return dict(
        timestamp=timestamp, agv_id=agv_id,
        motor_temp=np.clip(motor_temp, 20, 100),
        battery_temp=np.clip(battery_temp, 15, 70),
        vibration=np.clip(vibration, 0, 2),
        voltage=np.clip(voltage, 18, 30),
        current=np.clip(current, 0, 40),
        driving_time=np.clip(driving_time, 0, 24),
        load_weight=np.clip(load_weight, 0, 500),
        speed=np.clip(speed, 0, 3),
        error_count=error_count,
        thermal_image=DEFAULT_THERMAL,
    )


def generate_agv_data(n_agvs: int = 12, hours: int = 24, interval_min: int = 10) -> pd.DataFrame:
    """AGV 센서 더미데이터 생성 (12대, 24시간, 10분 간격)."""
    np.random.seed(42)

    agv_ids = [f"AGV-{i:02d}" for i in range(1, n_agvs + 1)]
    n_steps = (hours * 60) // interval_min  # 144

    end_time = datetime.now().replace(second=0, microsecond=0)
    end_time = end_time - timedelta(minutes=end_time.minute % interval_min)
    timestamps = [end_time - timedelta(minutes=interval_min * i) for i in range(n_steps - 1, -1, -1)]

    records = []
    special = {"AGV-03": _gen_agv03, "AGV-07": _gen_agv07, "AGV-11": _gen_agv11}

    for agv_id in agv_ids:
        agv_num = int(agv_id.split("-")[1])
        for idx, ts in enumerate(timestamps):
            t = idx / (n_steps - 1)
            if agv_id in special:
                rec = special[agv_id](agv_id, ts, t)
            else:
                rec = _gen_normal_agv(agv_id, agv_num, ts, t)
            records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values(["agv_id", "timestamp"]).reset_index(drop=True)

    # Rolling current instability (CV)
    df["current_instability"] = df.groupby("agv_id")["current"].transform(
        lambda x: x.rolling(window=6, min_periods=1).std()
        / (x.rolling(window=6, min_periods=1).mean() + 1e-9)
    )
    return df


def generate_and_save_data(data_dir: Path = None) -> pd.DataFrame:
    """더미데이터가 없으면 생성 후 CSV 저장, 있으면 로드."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "dummy_agv_sensor_data.csv"
    if not csv_path.exists():
        df = generate_agv_data()
        df.to_csv(csv_path, index=False)

    return pd.read_csv(csv_path)
