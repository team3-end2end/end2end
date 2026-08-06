> [!WARNING]
> **SAMPLE — 화면 확인용 예시이며 모델 및 평가 수치는 실제 결과가 아닙니다.**

# NYC Yellow Taxi 결제수단 예측 보고서

이 문서는 파이프라인 단계별 결과를 자동으로 모은 팀 내부 기술 보고서입니다.

## 1. 실행 및 데이터 정보

> **상태:** 완료

| 항목 | 값 |
|---|---|
| 프로젝트 | team3-end2end/end2end |
| 실행 ID | `sample-2026-08-06` |
| 코드 리비전 | `SAMPLE` |
| 데이터셋 | `eda/yellow_tripdata_2026-05_snappy.parquet` |
| 관측 기간 | 2026-05-01 ~ 2026-05-31 |
| 원본 크기 | 4,090,836행 × 20열 |
| 타깃 | `payment_label` · multiclass_classification |
| 클래스 | 신용카드, Flex Fare, 현금 |

## 2. EDA 결과

> **상태:** 완료

### 데이터 품질

| 수치형 변수 | 범주형 변수 | 결측 셀 | 중복 행 |
|---:|---:|---:|---:|
| 13 | 5 | 4,776,855 | 0 |

### 타깃 분포

| 클래스 | 건수 | 비율 |
|---|---:|---:|
| 신용카드 | 2,660,062 | 68.36% |
| Flex Fare | 878,256 | 22.57% |
| 현금 | 352,937 | 9.07% |

### 주요 발견

| 중요도 | 발견 | 근거 | 조치 |
|---|---|---|---|
| critical | **결제 후 생성되는 금액은 사용할 수 없습니다** — tip_amount와 total_amount는 결제수단을 직접 드러내 모델 성능을 실제보다 높게 만듭니다. | 현금 팁 0원 100%, fare_amount와 total_amount 상관계수 0.96 | 두 컬럼을 모델 피처에서 제외 |
| warning | **극단적인 거리값이 관계를 왜곡합니다** — 307,491마일 한 건 때문에 거리와 요금의 상관관계가 거의 0으로 보였습니다. | 이상치 처리 전후 상관계수 0.008 → 0.866 | trip_distance를 0~200마일 범위로 필터링 |
| info | **지역과 시간대가 중요한 단서입니다** — 승차 지역과 시간대에 따라 결제수단 구성에 의미 있는 차이가 있습니다. | 지역별 현금 비율 0~77.2%, 새벽 4시 Flex Fare 비율 50.9% | 지역·시간·요일과 공항 여부 피처 생성 |

### 시각화

#### 결제수단별 요금·거리 분포
![결제수단별 요금과 거리 박스플롯](../../eda/handoff/figures/target_box.png)

카드와 현금은 분포가 크게 겹치지만 Flex Fare는 상대적으로 장거리·고액 운행에 많습니다.
#### 원본 데이터 결측 패턴
![컬럼별 결측값 발생 패턴](../../eda/handoff/figures/missing_matrix.png)

다섯 컬럼의 결측이 같은 행에서 발생하며 payment_type 0과 일치합니다.

## 3. 전처리 결과

> **상태:** 완료

- 입력: **4,090,836행 × 20열**
- 출력: **3,891,255행 × 12열**
- 데이터 유지율: **95.12%**

### 단계별 필터

| 단계 | 규칙 | 적용 전 | 적용 후 | 제거 |
|---|---|---:|---:|---:|
| 양수 요금 | `fare_amount > 0` | 4,090,836 | 4,073,654 | 17,182 |
| 양수 거리 | `trip_distance > 0` | 4,073,654 | 3,962,811 | 110,843 |
| 거리 극단값 | `trip_distance < 200` | 3,962,811 | 3,962,738 | 73 |
| 요금·거리 불일치 | `~(fare_amount >= 100 and trip_distance < 1)` | 3,962,738 | 3,962,189 | 549 |
| 타깃 클래스 | `payment_type in {0, 1, 2}` | 3,962,189 | 3,942,906 | 19,283 |
| 관측 기간 | `pickup in 2026-05` | 3,942,906 | 3,942,892 | 14 |
| 운행 시간 | `0 < trip_duration < 180` | 3,942,892 | 3,891,255 | 51,637 |

### 피처 구성

- 선택 피처: trip_distance, fare_amount, trip_duration, cbd_congestion_fee, tolls_amount, PULocationID, DOLocationID, hour, day_of_week, VendorID, is_airport
- 제외 피처:
  - `tip_amount`: 결제 완료 후 기록되어 타깃을 직접 노출
  - `total_amount`: tip_amount를 포함하는 구성 누수
  - `passenger_count`: 결측 패턴이 payment_type=0과 완전히 일치
  - `RatecodeID`: 결측 패턴이 타깃과 일치
  - `mta_tax`: 대부분 0.5달러인 저분산 변수
  - `improvement_surcharge`: 대부분 1달러인 저분산 변수
- 파생 피처:
  - `trip_duration`: 승차·하차 시각 차이를 분 단위로 변환
  - `hour`: 반복되는 시간대 패턴 추출
  - `day_of_week`: 반복되는 요일 패턴 추출
  - `is_airport`: JFK·LGA·EWR 승하차 여부 표시
- 변환:
  - `categorical_encoding`: 범주형 피처를 모델 입력 형태로 변환
  - `numeric_scaling`: 거리·요금·시간의 단위 차이를 조정

## 4. 모델 학습

> **상태:** 완료

### 데이터 분할

| 전략 | 학습 | 검증 | 테스트 | Seed |
|---|---:|---:|---:|---:|
| stratified 70/15/15 split | 2,723,878 | 583,688 | 583,689 | 42 |

### 후보 모델

| ID | 모델 | 라이브러리 | 학습 시간 | 상태 | 산출물 |
|---|---|---|---:|---|---|
| `logistic_regression` | Logistic Regression | scikit-learn | 48.32초 | complete | `models/logistic_regression.joblib` |
| `random_forest` | Random Forest | scikit-learn | 326.18초 | complete | `models/random_forest.joblib` |

#### 하이퍼파라미터

- `logistic_regression`: `{'max_iter': 500, 'class_weight': 'balanced'}`
- `random_forest`: `{'n_estimators': 300, 'max_depth': 18, 'class_weight': 'balanced_subsample'}`

## 5. 평가 결과

> **상태:** 완료

- 주요 평가 지표: **macro_f1**
- 최종 모델: **`random_forest`**
- 선정 근거: 전체 정확도뿐 아니라 소수 클래스인 현금의 재현율과 Macro F1이 가장 높아 결제수단을 비교적 고르게 구분했습니다.

### 모델 비교

| 모델 | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | CV 평균 ± 표준편차 |
|---|---:|---:|---:|---:|---:|---:|
| `logistic_regression` | 71.400% | 58.100% | 62.300% | **59.200%** | 72.800% | 58.800% ± 0.600% |
| `random_forest` | 89.100% | 78.100% | 74.200% | **75.600%** | 88.500% | 74.900% ± 0.400% |

### 클래스별 성능

| 클래스 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 신용카드 | 93.800% | 94.000% | 93.900% | 399,010 |
| Flex Fare | 78.900% | 85.000% | 81.800% | 131,739 |
| 현금 | 79.000% | 62.200% | 69.600% | 52,940 |

### 혼동행렬

| 실제 ＼ 예측 | 신용카드 | Flex Fare | 현금 |
|---|---:|---:|---:|| **신용카드** | 375,000 | 20,000 | 4,010 |
| **Flex Fare** | 15,000 | 112,000 | 4,739 |
| **현금** | 10,000 | 10,000 | 32,940 |

### 한계 및 후속 작업

- 이 모델과 평가 수치는 보고서 화면 검증을 위한 가상 예시입니다.
- 카드와 현금은 요금·거리 분포가 겹쳐 실제 성능이 예시보다 낮을 수 있습니다.
- VendorID 의존도를 확인하기 위한 제외 후 재학습이 필요합니다.

---

_이 보고서는 `python -m reporting.generate`로 생성되었습니다. 직접 수정하지 말고 단계별 JSON 또는 템플릿을 변경하세요._
