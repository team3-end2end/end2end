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

### 이상치와 상관관계

- 이상치 수: **622**
- 이상치 규칙: trip_distance < 200, ~(fare_amount >= 100 and trip_distance < 1)
- 높은 상관관계 기준: **|r| ≥ 0.80**

| 변수 A | 변수 B | 상관계수 |
|---|---|---:|
| `trip_distance` | `fare_amount` | 0.8660 |
| `fare_amount` | `total_amount` | 0.9600 |

### 타깃 분포

| 클래스 | 건수 | 비율 |
|---|---:|---:|
| 신용카드 | 2,660,128 | 68.36% |
| Flex Fare | 878,188 | 22.57% |
| 현금 | 352,939 | 9.07% |

### 컬럼별 결측

| 컬럼 | 결측 수 | 결측률 |
|---|---:|---:|
| `passenger_count` | 955,371 | 23.35% |
| `RatecodeID` | 955,371 | 23.35% |
| `store_and_fwd_flag` | 955,371 | 23.35% |
| `congestion_surcharge` | 955,371 | 23.35% |
| `Airport_fee` | 955,371 | 23.35% |

### 시각화

#### 결제수단별 요금·거리 분포
![결제수단별 요금과 거리 박스플롯](../../eda/handoff/figures/target_box.png)
#### 원본 데이터 결측 패턴
![컬럼별 결측값 발생 패턴](../../eda/handoff/figures/missing_matrix.png)

## 3. 전처리 결과

> **상태:** 완료

- 입력: **4,090,836행 × 20열**
- 출력: **3,891,255행 × 12열**
- 데이터 유지율: **95.12%**
- 제거된 행: **199,581**
- 제거된 열: **10**
- 결측치 처리 변수: 없음
- 인코딩 변수: PULocationID, DOLocationID, hour, day_of_week, VendorID, is_airport

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
  - `tip_amount`
  - `total_amount`
  - `passenger_count`
  - `RatecodeID`
  - `mta_tax`
  - `improvement_surcharge`
- 파생 피처:
  - `trip_duration`
  - `hour`
  - `day_of_week`
  - `is_airport`
- 변환:
  - `OneHotEncoder` · 컬럼: PULocationID, DOLocationID, hour, day_of_week, VendorID, is_airport · 파라미터: `{'handle_unknown': 'ignore'}`
  - `StandardScaler` · 컬럼: trip_distance, fare_amount, trip_duration, cbd_congestion_fee, tolls_amount · 파라미터: `{}`

## 4. 모델 학습

> **상태:** 완료

### 데이터 분할

| 전략 | 학습 | 검증 | 테스트 | Seed |
|---|---:|---:|---:|---:|
| stratified 70/15/15 split | 2,723,878 | 583,688 | 583,689 | 42 |

### 모델 정보

| ID | 모델 | 라이브러리 | 학습 시간 | 산출물 |
|---|---|---|---:|---|
| `random_forest` | Random Forest | scikit-learn | 326.18초 | `models/random_forest.joblib` |

#### 하이퍼파라미터

- `random_forest`: `{'n_estimators': 300, 'max_depth': 18, 'class_weight': 'balanced_subsample'}`

## 5. 평가 결과

> **상태:** 완료

- 평가 모델: **`random_forest`**

### 전체 평가 지표

| Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 89.100% | 78.100% | 74.200% | 75.600% | 88.500% |

### 교차 검증

| 지표 | Fold | 평균 | 표준편차 |
|---|---:|---:|---:|
| macro_f1 | 5 | 74.900% | 0.400% |

### 클래스별 성능

| 클래스 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 신용카드 | 93.800% | 94.000% | 93.900% | 399,010 |
| Flex Fare | 78.900% | 85.000% | 81.800% | 131,739 |
| 현금 | 79.000% | 62.200% | 69.600% | 52,940 |

### 혼동행렬

| 실제 ＼ 예측 | 신용카드 | Flex Fare | 현금 |
|---|---:|---:|---:|
| **신용카드** | 375,000 | 20,000 | 4,010 |
| **Flex Fare** | 15,000 | 112,000 | 4,739 |
| **현금** | 10,000 | 10,000 | 32,940 |


### 평가 시각화

시각화가 등록되지 않았습니다.

---

_이 보고서는 `python -m reporting.generate`로 생성되었습니다. 직접 수정하지 말고 단계별 JSON 또는 템플릿을 변경하세요._
