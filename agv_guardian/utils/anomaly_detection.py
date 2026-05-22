"""LSTM AutoEncoder 기반 시계열 이상탐지 모듈.

AGV별 최근 2시간(window_size=12, 10분 간격) 센서 시퀀스를 입력으로 받아
정상 패턴 대비 재구성 오차(reconstruction error)를 ai_anomaly_score로 변환한다.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

FEATURES = [
    "motor_temp", "battery_temp", "vibration",
    "voltage", "current", "speed",
    "error_count", "load_weight", "driving_time", "current_instability",
]

WINDOW_SIZE = 12   # 10분 × 12 = 최근 2시간 시퀀스
LATENT_DIM  = 16
EPOCHS      = 5
BATCH_SIZE  = 64
SEED        = 42
BLEND_W     = 0.35  # sensor_risk_score 블렌딩 비율


def _build_lstm_autoencoder(n_features: int):
    """Encoder-Bottleneck-Decoder 구조 LSTM AutoEncoder 생성."""
    import tensorflow as tf
    tf.random.set_seed(SEED)

    inp = tf.keras.Input(shape=(WINDOW_SIZE, n_features))

    # Encoder
    enc = tf.keras.layers.LSTM(LATENT_DIM, return_sequences=False)(inp)

    # Bottleneck → Decoder 입력 복원
    rep = tf.keras.layers.RepeatVector(WINDOW_SIZE)(enc)
    dec = tf.keras.layers.LSTM(LATENT_DIM, return_sequences=True)(rep)

    # 출력: 원본 시퀀스 재구성
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features)
    )(dec)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def _sliding_windows(arr: np.ndarray) -> np.ndarray:
    """(N, F) 배열 → (N-W+1, W, F) 슬라이딩 윈도우 배열."""
    n = len(arr)
    if n < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, arr.shape[1]), dtype=np.float32)
    idx = (
        np.arange(WINDOW_SIZE)[None, :]
        + np.arange(n - WINDOW_SIZE + 1)[:, None]
    )
    return arr[idx].astype(np.float32)


def calculate_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """LSTM AutoEncoder 재구성 오차 기반 ai_anomaly_score를 df에 추가.

    인터페이스: df = calculate_anomaly_scores(df)
    """
    import tensorflow as tf
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    df = df.copy().reset_index(drop=True)

    feats = [f for f in FEATURES if f in df.columns]
    n_f = len(feats)

    # 정규화 (MinMaxScaler)
    X_raw = df[feats].fillna(df[feats].median()).values.astype(np.float32)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 행 위치 추적용 컬럼
    df["_pos"] = np.arange(len(df))

    # AGV별 슬라이딩 윈도우 생성
    all_windows: list[np.ndarray] = []
    window_pos: list[int] = []   # 각 윈도우 마지막 행의 전역 위치

    for _, grp in df.groupby("agv_id"):
        grp_sorted = grp.sort_values("timestamp")
        positions  = grp_sorted["_pos"].values
        X_agv      = X_scaled[positions]

        wins = _sliding_windows(X_agv)
        if len(wins) == 0:
            continue

        all_windows.append(wins)
        # 윈도우 i → 해당 AGV 내 인덱스 (WINDOW_SIZE-1+i)의 전역 위치
        for i in range(len(wins)):
            window_pos.append(int(positions[WINDOW_SIZE - 1 + i]))

    # 윈도우가 하나도 없으면 0점 반환
    if not all_windows:
        df["ai_anomaly_score"] = 0.0
        df.drop(columns=["_pos"], inplace=True)
        return df

    X_all = np.concatenate(all_windows, axis=0)   # (총 윈도우 수, W, F)

    # ── LSTM AutoEncoder 학습 ──────────────────────────────────
    model = _build_lstm_autoencoder(n_f)
    model.fit(
        X_all, X_all,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=True,
    )

    # ── 재구성 오차 계산 ──────────────────────────────────────
    X_pred  = model.predict(X_all, verbose=0)
    # MSE per window: (N_wins, W, F) → 스칼라
    errors  = np.mean(np.mean((X_all - X_pred) ** 2, axis=2), axis=1)

    # 0~100 정규화
    e_min, e_max = errors.min(), errors.max()
    scores_100   = 100.0 * (errors - e_min) / (e_max - e_min + 1e-9)

    # ── 행별 점수 매핑 ────────────────────────────────────────
    row_scores = np.full(len(df), np.nan)
    for pos, sc in zip(window_pos, scores_100):
        row_scores[pos] = sc
    df["ai_anomaly_score"] = row_scores

    # 초기 window 이전 구간(NaN) → 같은 AGV 내 bfill → ffill → 0
    df["ai_anomaly_score"] = (
        df.groupby("agv_id")["ai_anomaly_score"]
        .transform(lambda x: x.bfill().ffill().fillna(0.0))
    )

    # ── sensor_risk_score 블렌딩 ──────────────────────────────
    if "sensor_risk_score" in df.columns:
        df["ai_anomaly_score"] = (
            (1 - BLEND_W) * df["ai_anomaly_score"]
            + BLEND_W * df["sensor_risk_score"]
        ).clip(0, 100)

    df.drop(columns=["_pos"], inplace=True)
    return df
