# AGV Guardian

## 프로젝트 개요
AI 기반 AGV 예지보전 및 물류 흐름 안정화 Streamlit MVP

스마트 공장 내 AGV의 센서 데이터와 열화상 이미지를 기반으로
고장 위험도를 산출하고, 정비 우선순위와 조치 가이드를 제공합니다.

## 핵심 메시지
- **센서 AI**는 "위험한지"를 판단하고
- **열화상 AI**는 "어디가 문제인지"를 보여주며
- **통합 판단 알고리즘**은 "무엇을 먼저 정비해야 하는지"를 결정합니다

## 주요 기능
- AGV 상태 모니터링 (12대, 실시간 시계열)
- 센서 기반 위험도 산출 (규칙 기반 가중 점수)
- IsolationForest 기반 AI 이상탐지
- 열화상 Hot-spot 탐지 및 부품 위치 매핑
- 정비 우선순위 및 상세 조치 가이드

## 실행 방법

```bash
cd agv_guardian
pip install -r requirements.txt
streamlit run app.py
```

## 파일 구조
```
agv_guardian/
├── app.py                    # 메인 Streamlit 앱
├── requirements.txt
├── README.md
├── data/
│   ├── dummy_agv_sensor_data.csv   # 자동 생성
│   └── thermal_images/             # 자동 생성
│       ├── agv03_left_motor_hotspot.png
│       ├── agv07_battery_hotspot.png
│       ├── agv11_wheel_hotspot.png
│       └── normal_agv.png
└── utils/
    ├── data_generator.py     # 더미 센서 데이터 생성
    ├── risk_scoring.py       # 위험도 산출 로직
    ├── anomaly_detection.py  # IsolationForest 이상탐지
    ├── thermal_detection.py  # 열화상 이미지 처리
    └── maintenance_rules.py  # 정비 가이드 규칙
```

## 시연 시나리오 (AGV-03 Critical 사례)

1. **Overview** 페이지 → AGV-03이 🔴 Critical로 표시
2. **AGV Monitoring** → AGV-03 선택 → 모터 온도/진동/전류 상승 그래프
3. **Risk Analysis** → 최종 위험도 80+ 게이지 확인
4. **Thermal Diagnosis** → `agv03_left_motor_hotspot.png` 선택 → 좌측 모터 Hot-spot 탐지
5. **Maintenance Guide** → AGV-03 정비 우선순위 1위, 24시간 이내 정비 가이드

## 위험도 산출 공식
```
Final Risk Score =
  0.50 × Sensor Risk Score
+ 0.30 × AI Anomaly Score
+ 0.20 × Thermal Risk Score
```

## 상태 등급
| 등급 | 점수 | 권장 조치 |
|------|------|-----------|
| 🟢 Normal  | 0~39  | 정기 모니터링 |
| 🟡 Caution | 40~59 | 다음 정기점검 시 확인 |
| 🟠 Warning | 60~79 | 48시간 이내 점검 |
| 🔴 Critical| 80~100| 24시간 이내 정비 |
