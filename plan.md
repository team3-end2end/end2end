# End2End 데이터 분석 프로젝트 — 결제수단 예측 (NYC Yellow Taxi)

> **주제**: 2026년 5월 NYC Yellow Taxi 데이터로 승객의 결제수단(카드 vs 현금) 예측
> **데이터**: `yellow_tripdata_2026-05.parquet` (4,090,836행 × 20컬럼, 66.5MB)
> **마감**: 과정 종료 시까지 (~21:00 제출 엄수, 초과 시 감점)

---

## 1. 과제 요구사항 (채점 기준 100점)

### 데이터 준비 + 시각화 (35점)
- [ ] 데이터셋을 **Pandas와 Polars 양쪽으로 로딩**하여 결과 비교
- [ ] **결측치·중복 처리** 및 기본 EDA 수행, 결과 출력
- [ ] **Seaborn 정적 차트 1개 이상** + **Plotly 인터랙티브 차트 1개 이상**
  (분포·상관관계·그룹 비교 중 택일, **제목·축 레이블 필수**)

### 통계 분석 + ML Pipeline (45점)
- [ ] 기술통계(평균·표준편차·분위수) 산출
- [ ] 변수 간 **상관계수** 계산·출력
- [ ] **`scipy.stats.ttest_ind`로 t-test** 수행 및 **p-value 해석** 포함
- [ ] **`sklearn.pipeline.Pipeline`** 객체로 전처리 + 모델 학습 구성
- [ ] **평가지표(정확도·F1 등) 출력**
- [ ] **joblib으로 모델 파일 저장**

### 자동화 + 발표 (20점)
- [ ] 분석 결과를 **report.md로 자동 생성** (스크립트로 생성해야 함)
- [ ] 팀별 발표 5분

### 완성도 (10점)
- [ ] **주석 처리 누락 시 감점** → 모든 코드에 주석 충실히 작성

### 제출물
- [ ] 폴더 포함 전체 코드 → `캠퍼스명_반_이름_실습명.zip` (예: `서울_1반_홍길동_day2종합실습.zip`)
- [ ] 실행결과 화면 캡처 + 코드 분석 결과에 대한 본인 의견(개선 사항, 코드 품질 등) → **PDF 제출**

---

## 2. 데이터 사전 탐색 결과 (핵심 발견 사항)

프로젝트 설계의 근거가 되는, 이미 확인된 사실들:

1. **타깃 누수(target leakage)**: 현금 팁은 기록되지 않아 `tip_amount`가 결제수단을 거의 그대로 노출
   (카드 평균 팁 $4.30 vs 현금 $0.0004). → `tip_amount`, `total_amount`는 **피처에서 제외**하고 그 근거를 보고서에 명시
2. **구조적 결측**: 결측치 955,371개(`passenger_count`, `RatecodeID`, `store_and_fwd_flag`,
   `congestion_surcharge`, `Airport_fee`)가 `payment_type=0`(미상) 행 수와 정확히 일치 → 무작위 결측이 아님
3. **문제 재정의**: payment_type 0(미상)·3(무료)·4(분쟁) 제거 → **카드(1) vs 현금(2) 이진 분류**
   (카드 272.7만 / 현금 37.3만 ≈ 88:12 **클래스 불균형** → 정확도만으론 부족, F1 + 다수 클래스 베이스라인(88%) 비교 필수)
4. **이상치 존재**: 음수 요금(min -$950), trip_distance 최대 307,491마일 → 정제 단계에서 처리

---

## 3. 프로젝트 구조

```
.
├── plan.md                  # 이 문서 — 요구사항·산출물 기준
├── AGENTS.md                # 에이전트·팀원 작업 규약 (PR, 파일 소유권)
├── .github/
│   └── pull_request_template.md
├── data/                    # gitignore — 각자 로컬에 둔다 (README 참고)
│   ├── raw/                 # 원본 parquet. 읽기 전용, 절대 덮어쓰지 않는다
│   └── processed/           # 전처리 결과 (cleaned.parquet)
├── src/
│   ├── 01_load_compare.py   # Pandas vs Polars 로딩 비교
│   ├── 02_clean_eda.py      # 결측치·중복·이상치 처리 + EDA + 시각화
│   ├── 03_stats.py          # 기술통계·상관계수·t-test
│   ├── 04_ml_pipeline.py    # sklearn Pipeline 학습·평가·저장
│   └── 05_report.py         # report.md 자동 생성
├── outputs/
│   ├── figures/             # 차트 (PNG + HTML)
│   ├── model/               # 저장된 모델 (.joblib)
│   ├── results/             # 각 단계 수치 결과 (JSON — report 생성용)
│   └── report.md            # 최종 자동 생성 리포트
└── requirements.txt         # pandas, polars, pyarrow, seaborn, plotly, scipy, scikit-learn, joblib
```

---

## 4. 단계별 Task 및 산출물

### Step 1 — 데이터 로딩 & Pandas vs Polars 비교 (`01_load_compare.py`)
**Task**
- Pandas `read_parquet`과 Polars `read_parquet`으로 각각 로딩
- 로딩 시간(초), 메모리 사용량, shape·dtype 비교 출력 (409만 행이라 차이가 유의미하게 드러남)
- 이후 단계에서 쓸 정제 전 데이터 확인 (행 수, 컬럼 목록)

**산출물**
- 콘솔 출력: 비교 테이블 (라이브러리 / 로딩 시간 / 메모리 / shape)
- `outputs/results/load_compare.json` — report 생성용 수치

**검증**: 두 라이브러리의 행·열 수가 일치하는지 확인 출력

---

### Step 2 — 정제 + EDA + 시각화 (`02_clean_eda.py`)
**Task**
- 결측치 현황 출력 → 구조적 결측(payment_type=0과 일치) 근거 제시
- 중복 행 확인(0건임을 출력하는 것도 "처리했다"는 증빙)
- 필터링: payment_type ∈ {1, 2}, fare_amount > 0, 0 < trip_distance < 100, 유효한 pickup/dropoff 시간
- 파생 변수: `trip_duration_min`, `pickup_hour`, `is_weekend`
- 정제 전후 행 수 변화 출력
- **Seaborn 정적 차트**: 결제수단별 fare_amount 분포 (boxplot/violinplot)
- **Plotly 인터랙티브 차트**: 시간대별 카드/현금 비율 추이 (grouped bar 또는 line)
- 두 차트 모두 제목·축 레이블 포함

**산출물**
- `outputs/figures/seaborn_fare_by_payment.png`
- `outputs/figures/plotly_hourly_payment.html`
- 정제된 데이터: `data/processed/cleaned.parquet` (Step 3·4의 입력)
- `outputs/results/eda.json` — 결측치 수, 정제 전후 행 수 등

**검증**: 정제 후 payment_type이 {1,2}만 남았는지, 차트 파일 2개 존재 확인

---

### Step 3 — 통계 분석 (`03_stats.py`)
**Task**
- 기술통계: 주요 수치형 변수(fare_amount, trip_distance, trip_duration_min 등)의
  평균·표준편차·분위수(25/50/75%) 테이블 출력
- 상관계수: 수치형 변수 간 Pearson 상관행렬 계산·출력
- **t-test**: "카드 결제와 현금 결제의 평균 fare_amount는 다른가"
  - `scipy.stats.ttest_ind(card, cash, equal_var=False)` (Welch's t-test)
  - t-통계량, p-value 출력 + **해석 문장** (대표본이라 p-value가 극단적으로 작음 →
    "통계적 유의 ≠ 실질적 차이 크기" 논의, 평균 차이 자체도 함께 제시)

**산출물**
- 콘솔 출력: 기술통계 테이블, 상관행렬, t-test 결과 + 해석
- `outputs/results/stats.json` — report 생성용 (기술통계, 상관계수, t/p값, 해석 텍스트)

**검증**: p-value가 출력되고 해석 문장이 포함되는지 확인

---

### Step 4 — ML Pipeline (`04_ml_pipeline.py`)
**Task**
- 타깃: `is_card` (payment_type==1), **누수 피처 제외**(tip_amount, total_amount 및 파생 요금 합계류)
- 사용 피처: trip_distance, fare_amount, trip_duration_min, passenger_count,
  pickup_hour, is_weekend, PULocationID/DOLocationID(범주형), RatecodeID 등
- 층화 샘플링 30~50만 행 (전체 409만 행은 불필요, 실행 시간 관리)
- train/test 분할 (`train_test_split`, stratify, random_state 고정)
- **`Pipeline` = `ColumnTransformer`(수치형 StandardScaler + 범주형 OneHotEncoder) + 모델**
  - 베이스라인: LogisticRegression
  - 본 모델: HistGradientBoostingClassifier
- 평가: accuracy, F1, confusion matrix + **다수 클래스 베이스라인(≈88%)과 비교**
- `joblib.dump(pipeline, 'outputs/model/payment_classifier.joblib')`

**산출물**
- 콘솔 출력: 모델별 accuracy·F1·confusion matrix 비교
- `outputs/model/payment_classifier.joblib`
- `outputs/results/ml.json` — 평가지표 수치

**검증**: 저장된 .joblib을 재로드하여 예측이 동일한지 확인 코드 포함

---

### Step 5 — report.md 자동 생성 (`05_report.py`)
**Task**
- `outputs/results/*.json`을 읽어 f-string 템플릿으로 report.md 조립
- 구성: 개요 → 데이터 준비(Pandas/Polars 비교) → 정제 요약 → EDA 차트(이미지 링크) →
  통계 분석(t-test 해석 포함) → ML 결과(지표 테이블) → 결론·한계(타깃 누수 발견 스토리 강조)

**산출물**
- `outputs/report.md` (차트 이미지 링크 포함)

**검증**: 스크립트 재실행 시 report.md가 최신 수치로 재생성되는지 확인

---

### Step 6 — 제출 준비
**Task**
- 전체 실행 순서 확인: `01 → 02 → 03 → 04 → 05` 순차 실행으로 처음부터 끝까지 재현되는지 최종 점검
- 모든 파일 주석 점검 (감점 항목)
- 실행결과 화면 캡처 → 본인 의견(개선 사항, 코드 품질)과 함께 PDF 작성
- 폴더 구조 그대로 zip: `캠퍼스명_반_이름_day2종합실습.zip`

**산출물**
- 제출용 zip + PDF

---

## 5. 발표 스토리라인 (5분)

1. 문제 정의: 운행 정보만으로 결제수단을 예측할 수 있는가?
2. **핵심 발견**: tip_amount 타깃 누수 발견 → 제외 결정 (차별화 포인트)
3. 구조적 결측 분석 → 카드 vs 현금 이진 분류로 재정의
4. 통계: 카드/현금 그룹 간 요금 차이 t-test 결과
5. 모델: 베이스라인(88%) 대비 성능, F1 중심 해석
6. 한계 및 개선 방향

## 6. 시간 배분 가이드

- Step 1~2 (준비·EDA·시각화): 40%
- Step 3~4 (통계·ML): 40%
- Step 5~6 (리포트·제출): 20%
