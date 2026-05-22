"""Dense AutoEncoder + PHM Health Index 기반 시계열 이상탐지 모듈.

AGV별 최근 2시간(window_size=12, 10분 간격) 센서 시퀀스를 Dense AutoEncoder로
압축·복원하여 재구성 오차를 산출하고, PHM Health Index(센서 열화 추세 + 임계값
근접도)와 블렌딩하여 ai_anomaly_score를 생성한다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = [
    "motor_temp", "battery_temp", "vibration",
    "voltage", "current", "speed",
    "error_count", "load_weight", "driving_time", "current_instability",
]

WINDOW_SIZE    = 12    # 10분 × 12 = 최근 2시간 시퀀스
ENCODING_DIM   = 16   # bottleneck 차원
AE_EPOCHS      = 30
AE_LR          = 0.005
BATCH_SIZE     = 64
SEED           = 42
BLEND_W_SENSOR = 0.25  # sensor_risk_score 블렌딩 비율

# PHM 센서별 정상/고장 기준값 및 내부 가중치
PHM_CONFIG: dict[str, dict] = {
    "motor_temp":          {"normal": 45.0, "fail": 90.0, "weight": 0.30, "inverted": False},
    "battery_temp":        {"normal": 35.0, "fail": 65.0, "weight": 0.20, "inverted": False},
    "vibration":           {"normal": 0.30, "fail": 1.20, "weight": 0.25, "inverted": False},
    "voltage":             {"normal": 26.0, "fail": 18.0, "weight": 0.15, "inverted": True},
    "current_instability": {"normal": 0.04, "fail": 0.55, "weight": 0.10, "inverted": False},
}


# ─────────────────────────────────────────────────────────────
# Dense AutoEncoder (numpy only)
# ─────────────────────────────────────────────────────────────

class _DenseAutoEncoder:
    """단일 은닉층 Dense AutoEncoder — numpy SGD 기반.

    구조: Input(n_in) → ReLU(n_in) → Bottleneck(latent) → ReLU(n_in) → Output(n_in)
    """

    def __init__(self, n_in: int, latent: int = 16, lr: float = 0.005, seed: int = 42):
        rng = np.random.default_rng(seed)
        s1  = np.sqrt(2.0 / n_in)
        s2  = np.sqrt(2.0 / latent)
        self.W1 = rng.normal(0, s1, (n_in,    latent)).astype(np.float32)
        self.b1 = np.zeros(latent,  np.float32)
        self.W2 = rng.normal(0, s2, (latent,  n_in  )).astype(np.float32)
        self.b2 = np.zeros(n_in,    np.float32)
        self.lr = lr

    def _forward(self, X: np.ndarray):
        h_pre = X @ self.W1 + self.b1
        h     = np.maximum(0.0, h_pre)
        out   = h @ self.W2 + self.b2
        return h_pre, h, out

    def fit(self, X: np.ndarray) -> None:
        n   = len(X)
        rng = np.random.default_rng(SEED)
        for _ in range(AE_EPOCHS):
            idx = rng.permutation(n)
            for s in range(0, n, BATCH_SIZE):
                b               = X[idx[s: s + BATCH_SIZE]]
                h_pre, h, out   = self._forward(b)
                g_out           = 2.0 * (out - b) / max(len(b), 1)
                dW2             = h.T @ g_out
                db2             = g_out.sum(0)
                g_h             = (g_out @ self.W2.T) * (h_pre > 0)
                dW1             = b.T @ g_h
                db1             = g_h.sum(0)
                self.W1        -= self.lr * dW1
                self.b1        -= self.lr * db1
                self.W2        -= self.lr * dW2
                self.b2        -= self.lr * db2

    def reconstruct_error(self, X: np.ndarray) -> np.ndarray:
        _, _, out = self._forward(X)
        return np.mean((X - out) ** 2, axis=1)


# ─────────────────────────────────────────────────────────────
# 슬라이딩 윈도우
# ─────────────────────────────────────────────────────────────

def _sliding_windows_flat(arr: np.ndarray, window: int) -> np.ndarray:
    """(N, F) 배열 → (N-W+1, W*F) 슬라이딩 윈도우 (평탄화)."""
    n, f = arr.shape
    if n < window:
        return np.empty((0, window * f), dtype=np.float32)
    idx = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
    return arr[idx].reshape(-1, window * f).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# AutoEncoder 점수
# ─────────────────────────────────────────────────────────────

def _ae_scores(df: pd.DataFrame, feats: list) -> np.ndarray:
    """Dense AE 재구성 오차 → 0-100 점수 배열 (df 행 순서와 1:1 대응)."""
    X_raw    = df[feats].fillna(df[feats].median()).values.astype(np.float32)
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    df_tmp         = df.copy()
    df_tmp["_pos"] = np.arange(len(df))

    all_wins: list = []
    win_pos:  list = []

    for _, grp in df_tmp.groupby("agv_id"):
        grp_s = grp.sort_values("timestamp")
        pos   = grp_s["_pos"].values
        wins  = _sliding_windows_flat(X_scaled[pos], WINDOW_SIZE)
        if len(wins) == 0:
            continue
        all_wins.append(wins)
        for i in range(len(wins)):
            win_pos.append(int(pos[WINDOW_SIZE - 1 + i]))

    if not all_wins:
        return np.zeros(len(df), dtype=np.float32)

    X_all = np.concatenate(all_wins, axis=0)

    ae     = _DenseAutoEncoder(X_all.shape[1], latent=ENCODING_DIM, lr=AE_LR)
    ae.fit(X_all)

    errors         = ae.reconstruct_error(X_all)
    e_min, e_max   = errors.min(), errors.max()
    s100           = 100.0 * (errors - e_min) / (e_max - e_min + 1e-9)

    row_scores = np.full(len(df), np.nan, dtype=np.float64)
    for pos, sc in zip(win_pos, s100):
        row_scores[pos] = sc

    filled = pd.Series(row_scores).bfill().ffill().fillna(0.0).values
    return filled.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# PHM Health Index
# ─────────────────────────────────────────────────────────────

def _phm_scores(df: pd.DataFrame) -> np.ndarray:
    """PHM Health Index → 0-100 점수 배열 (df 행 순서와 1:1 대응).

    ① 센서별 정상→고장 구간 정규화 열화도 가중합
    ② 모터 온도·진동 상승 트렌드 보정 (최대 +20점)
    """
    result = np.zeros(len(df), dtype=np.float32)

    for _, grp in df.groupby("agv_id"):
        grp_s = grp.sort_values("timestamp")
        idx   = grp_s.index.values
        n     = len(grp_s)

        hi_parts: list  = []
        weight_sum      = 0.0

        for sensor, cfg in PHM_CONFIG.items():
            if sensor not in grp_s.columns:
                continue
            vals           = grp_s[sensor].values.astype(np.float32)
            normal, fail   = cfg["normal"], cfg["fail"]
            w              = cfg["weight"]

            if cfg["inverted"]:
                norm = (normal - vals) / (abs(normal - fail) + 1e-9)
            else:
                norm = (vals - normal) / (abs(fail - normal) + 1e-9)

            hi_parts.append(np.clip(norm, 0.0, 1.0) * w)
            weight_sum += w

        if not hi_parts:
            continue

        hi  = np.sum(hi_parts, axis=0) / (weight_sum + 1e-9) * 100.0

        # 상승 트렌드 보정: 최근 window의 선형 기울기 → 최대 +20점
        win          = min(WINDOW_SIZE, max(3, n // 3))
        x_lin        = np.arange(win, dtype=np.float32)
        trend_slopes = []
        for sensor in ("motor_temp", "vibration"):
            if sensor not in grp_s.columns:
                continue
            vals   = grp_s[sensor].values.astype(np.float32)
            slopes = np.zeros(n, dtype=np.float32)
            for i in range(win, n):
                slope       = float(np.polyfit(x_lin, vals[i - win: i], 1)[0])
                slopes[i]   = max(0.0, slope)
            trend_slopes.append(slopes)

        if trend_slopes:
            avg_s = np.mean(trend_slopes, axis=0)
            max_s = float(avg_s.max())
            if max_s > 0.0:
                hi = np.clip(hi + avg_s / max_s * 20.0, 0.0, 100.0)

        result[idx] = hi.astype(np.float32)

    return result


# ─────────────────────────────────────────────────────────────
# 공개 인터페이스
# ─────────────────────────────────────────────────────────────

def calculate_anomaly_scores(
    df:    pd.DataFrame,
    w_ae:  float = 0.60,
    w_phm: float = 0.40,
) -> pd.DataFrame:
    """Dense AutoEncoder + PHM Health Index → ai_anomaly_score 컬럼 추가.

    Parameters
    ----------
    w_ae  : AutoEncoder 점수 가중치 (0~1)
    w_phm : PHM Health Index 가중치 (0~1)
            두 값은 자동 정규화되므로 합이 1일 필요 없음.
    """
    np.random.seed(SEED)
    df    = df.copy().reset_index(drop=True)
    feats = [f for f in FEATURES if f in df.columns]

    ae_score  = _ae_scores(df, feats)
    phm_score = _phm_scores(df)

    total    = w_ae + w_phm + 1e-9
    ai_score = (w_ae / total) * ae_score + (w_phm / total) * phm_score

    # sensor_risk_score와 일부 블렌딩해 규칙 기반 신호 보완
    if "sensor_risk_score" in df.columns:
        ai_score = (
            (1.0 - BLEND_W_SENSOR) * ai_score
            + BLEND_W_SENSOR * df["sensor_risk_score"].values
        )

    df["ai_anomaly_score"] = np.clip(ai_score, 0.0, 100.0)
    return df
