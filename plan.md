# End2End 데이터 분석 프로젝트 — 결제수단 예측 (NYC Yellow Taxi)

> **주제**: 2026년 5월 NYC Yellow Taxi 데이터로 결제수단(신용카드 / 현금 / Flex Fare) 예측
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

1. **타깃 누수(target leakage)**: 현금 팁은 기록되지 않아 `tip_amount`가 결제수단을 그대로 노출
   (팁 > 0 비율: 카드 90.2%, 그 외 0%). → 누수 컬럼 7개를 **피처에서 제외**하고 근거를 보고서에 명시
   (`tip_amount`, `total_amount`, `congestion_surcharge`, `Airport_fee`, `RatecodeID`,
   `passenger_count`, `store_and_fwd_flag`)
2. **`payment_type=0`은 결측이 아니라 Flex Fare**: 955,371건에서 위 5개 컬럼이 100% 비어 있으나
   (다른 클래스는 0%), 이는 결측이 아니라 **해당 없음**이다. TLC 공식 데이터 사전에 `0 = Flex Fare trip`으로
   명시된 앱 기반 사전 확정요금 제도로, 미터기 세부 요금 필드가 애초에 존재하지 않는다.
   → 정상 클래스로 포함하되, 위 컬럼들은 피처에서 제외해 누수를 차단한다. (판단 번복 경위는 [WORKLOG.md](WORKLOG.md) 참조)
3. **문제 정의**: payment_type 3(무료)·4(분쟁)만 제거(합산 0.85%로 학습 불가) →
   **신용카드(1) / 현금(2) / Flex Fare(0) 3-클래스 분류**
   (정제 후 68.4 : 22.6 : 9.1 → 최소 클래스 9%로 `class_weight="balanced"`로 다룰 수 있는 수준.
   다만 **정확도만으론 부족** — 최빈 클래스만 예측하는 베이스라인이 정확도 0.68이므로 macro F1과 함께 볼 것)
4. **이상치 존재**: 음수 요금(min -$950), trip_distance 최대 307,491마일 → 정제 단계에서 처리
5. **Flex Fare 요금의 끝자리 패턴**: 요금이 $0.50 배수인 비율이 Flex Fare 2.8% vs 카드 27.4% / 현금 25.4%.
   사전 확정요금이라 미터기 요금과 소수점 분포가 다르다. 누수는 아니지만 모델의 Flex Fare 분류가
   이 패턴에 의존할 수 있으므로 EDA에서 확인한다.

---

## 3. 프로젝트 구조

```
.
├── plan.md                  # 이 문서 — 요구사항·산출물 기준
├── WORKLOG.md               # 판단이 바뀐 지점과 근거 (시간순)
├── AGENTS.md                # 에이전트·팀원 작업 규약 (PR, 파일 소유권)
├── .github/
│   └── pull_request_template.md
├── data/                    # gitignore — 각자 로컬에 둔다 (README 참고)
│   ├── raw/                 # 원본 parquet. 읽기 전용, 절대 덮어쓰지 않는다
│   └── processed/           # 전처리 결과 (cleaned.parquet)
├── src/
│   ├── 01_load_compare.py   # Pandas vs Polars 로딩·전처리 성능 비교
│   ├── 02_preprocess.py     # 정제 + 파생변수 → cleaned.parquet  ← 데이터를 바꾸는 유일한 단계
│   ├── 03_eda.py            # EDA (숫자로만) + 피처 화이트리스트 확정
│   ├── 04_stats.py          # t-test + 효과 크기
│   ├── 05_visualize.py      # Seaborn 정적 차트 + Plotly 인터랙티브 차트
│   ├── 06_ml_pipeline.py    # sklearn Pipeline 학습·평가·저장
│   └── 07_report.py         # report.md 자동 생성
├── outputs/
│   ├── figures/             # 차트 (PNG + HTML)
│   ├── model/               # 저장된 모델 (.joblib)
│   ├── results/             # 각 단계 수치 결과 (JSON + MD)
│   └── report.md            # 최종 자동 생성 리포트
└── requirements.txt         # pandas, polars, pyarrow, seaborn, plotly, scipy, scikit-learn, joblib
```

### 단계 설계 원칙

**만든다 → 본다 → 검증한다 → 보여준다** 순서로 나눈다.

| 단계 | 성격 | 데이터를 바꾸나 |
|---|---|---|
| 02 전처리 | 만든다 | **바꾼다 (유일)** |
| 03 EDA | 본다 | 안 바꿈 |
| 04 통계 검정 | 검증한다 | 안 바꿈 |
| 05 시각화 | 보여준다 | 안 바꿈 |

1. **모든 단계는 JSON + MD를 한 쌍으로 만든다.** JSON은 다음 단계와 `07_report.py`가 읽는 기계용,
   MD는 사람이 그 단계만 따로 확인하는 용도. 리포트에 숫자를 직접 타이핑할 일이 없어진다.
2. **데이터를 바꾸는 단계는 02 하나뿐이다.** 03 이후는 `cleaned.parquet`을 읽기만 한다.
   결과가 이상할 때 "정제가 잘못됐나 / 분석이 잘못됐나"를 즉시 가를 수 있다.
3. **누수 차단을 파일로 강제한다.** 03이 만든 `features.json`(쓸 컬럼 목록)을 06이 읽어 그것만 쓴다.
   사람이 실수로 `tip_amount`를 넣을 경로 자체를 없앤다.
4. **03·04·05는 서로 의존하지 않는다.** 순서를 바꿔도 되고 팀원이 나눠 맡아도 된다.
   단 06은 03의 `features.json`이, 07은 전 단계가 필요하다.

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

### Step 2 — 전처리 (`02_preprocess.py`)
> 데이터를 바꾸는 유일한 단계. 시각화·분석은 하지 않는다.

**Task**
- 원본 현황 파악: 행·열 수, 결측, 중복(0건임을 출력하는 것도 "처리했다"는 증빙)
- 결측 구조 교차검증: 결측 955,371건이 `payment_type=0`(Flex Fare)과 100% 일치함을 보이고,
  무작위 결측이 아니라 **해당 없음**임을 근거와 함께 제시
- 분석 대상 선별: payment_type ∈ **{0, 1, 2}** (3·4는 합산 0.85%로 학습 불가)
- 이상값 제거: fare_amount > 0, 0 < trip_distance < 200, 0 < trip_duration_min < 180
  → **규칙별 해당 건수를 각각 출력** (PR·리포트의 근거로 사용)
- 파생변수: `trip_duration_min`, `pickup_hour`, `day_of_week`, `is_weekend`
- 타깃 라벨: `payment_label` (0→Flex Fare, 1→신용카드, 2→현금)
- 클래스 분포는 **정제가 제대로 됐는지 확인하는 용도로만** 출력 (본격 분석은 Step 3)

**산출물**
- **`data/processed/cleaned.parquet`** — Step 3·4·5·6의 입력
- `outputs/results/preprocess.json` / `preprocess.md`

**검증**
- 정제 후 payment_type이 {0,1,2}만 남았는지 (`assert`)
- 남은 결측이 전부 Flex Fare 행 × 5개 컬럼과 일치하는지 (`assert`) — 결함이 아닌 "해당 없음"

---

### Step 3 — EDA (`03_eda.py`)
> 숫자로만 탐색한다. 차트는 Step 5에서 그린다.
> 차트를 먼저 그리면 해석이 그림에 끌려간다(예: 이상치 제거 전 Flex Fare 거리가 3배로 보였으나 실제로는 11% 차이).

**Task**
1. **클래스 분포와 베이스라인** — 68.4 : 22.6 : 9.1, 최빈 클래스만 예측 시 정확도 0.68.
   Step 6의 성능 판단 기준선이므로 여기서 확정한다
2. **전체 기술통계** — 평균·표준편차·분위수(25/50/75%) — 과제 요구사항
3. **결제수단별 기술통계** — 같은 지표를 그룹별로 (EDA의 핵심인 그룹 간 비교)
4. **시간 패턴** — 시간대별·요일별 결제수단 구성비 (표로)
5. **지역 패턴** — 픽업 상위 10개 구역의 결제수단 구성비.
   `PULocationID`는 지역 코드이므로 **상관계수가 아니라 교차표로** 본다
6. **상관계수** — 수치형 변수끼리만 (Pearson). 거리·요금·소요시간이 0.76~0.86으로 강하게 묶임
   → 정보 중복을 뜻하므로 Step 6의 모델 선택 근거가 된다
7. **누수 진단** — 팁 > 0 비율(카드 90.2% vs 나머지 0%), 요금 끝자리 $0.50 배수 비율
   (Flex Fare 2.7% vs 나머지 25% 내외)
8. **피처 화이트리스트 확정** — 위 결과를 근거로 쓸 컬럼 8개 / 뺄 컬럼 7개를 파일로 확정

**산출물**
- `outputs/results/eda.json` / `eda.md`
- **`outputs/results/features.json`** — Step 6이 읽어 쓸 피처 화이트리스트

**검증**: `features.json`에 누수 컬럼 7개가 하나도 포함되지 않았는지 (`assert`)

---

### Step 4 — 통계 검정 (`04_stats.py`)
**Task**
- **t-test**: "신용카드와 현금의 평균 fare_amount는 다른가"
  (두 그룹 비교이므로 3-클래스여도 그대로 성립)
  - `scipy.stats.ttest_ind(card, cash, equal_var=False)` (Welch's t-test)
- **효과 크기(Cohen's d)를 함께 계산한다.** 표본이 389만 건이라 p-value는 사실상 0이 나온다.
  "통계적으로 유의하다"만으로는 아무 말도 못 하므로, 실질적 차이 크기를 수치로 제시해야 한다
  (팀 리포트는 이 구분을 언급만 하고 계산하지 않았다 — 여기서 한 걸음 더 나간다)
- 해석 문장을 함께 출력: 유의성과 효과 크기를 구분해 서술

**산출물**
- `outputs/results/stats.json` / `stats.md`

**검증**: p-value와 Cohen's d가 함께 출력되고 해석 문장이 포함되는지

---

### Step 5 — 시각화 (`05_visualize.py`)
**Task**
- **Seaborn 정적 차트**: 결제수단별 이동거리(또는 요금) 분포 — boxplot
  (극단값에 눌리지 않도록 이상치 표시를 끄고 축 범위 제한)
- **Plotly 인터랙티브 차트**: 시간대별 결제수단 구성비 — stacked bar
  (건수가 아니라 **비율**로 그린다. 시간대별 운행량 차이 때문에 건수로 그리면
  "새벽에 현금이 적다"가 아니라 "새벽에 운행이 적다"만 보인다)
- 두 차트 모두 **제목·축 레이블 필수** (채점 항목)
- 한글 폰트는 OS별 후보를 순서대로 시도 (macOS AppleGothic / Windows Malgun Gothic / Linux NanumGothic)

**산출물**
- `outputs/figures/seaborn_*.png`, `outputs/figures/plotly_*.html`
- `outputs/results/figures.json` — 차트 경로·설명 (report 생성용)

**검증**: 파일 2개 생성, 한글이 깨지지 않는지 육안 확인

---

### Step 6 — ML Pipeline (`06_ml_pipeline.py`)
**Task**
- 타깃: `payment_label` 3-클래스 (신용카드 / 현금 / Flex Fare)
- **피처는 Step 3이 만든 `outputs/results/features.json`을 읽어 그것만 사용한다.**
  코드에 컬럼명을 직접 적지 않는다 — 누수 컬럼이 실수로 섞일 경로를 없애기 위함
  (제외 대상 7개: tip_amount, total_amount, congestion_surcharge, Airport_fee,
  RatecodeID, passenger_count, store_and_fwd_flag)
- 층화 샘플링 30만 행 (전체 389만 행 학습은 5분 이상 소요되어 실습에 비효율적)
- train/test 분할 (`train_test_split`, stratify, `random_state=42`)
- **`Pipeline` = `ColumnTransformer`(수치형 StandardScaler + 범주형 OneHotEncoder) + 모델**
  - 피처 간 상관이 높으므로(0.76~0.86) 트리 기반 모델 사용
  - `class_weight="balanced"` 적용 (클래스 비율 68 : 23 : 9)
- 평가: accuracy, **클래스별 F1 + macro F1**, confusion matrix
  - **베이스라인 2종과 비교**: 최빈 클래스만 예측 시 정확도 0.68 / macro F1 약 0.27
  - `class_weight="balanced"`는 소수 클래스 recall을 얻는 대신 **정확도를 희생**한다.
    정확도가 베이스라인보다 낮게 나올 수 있으며, 이는 의도된 트레이드오프임을 반드시 명시할 것
- `joblib.dump(pipeline, 'outputs/model/payment_classifier.joblib')`

**산출물**
- 콘솔 출력: 모델별 accuracy·F1·confusion matrix 비교
- `outputs/model/payment_classifier.joblib`
- `outputs/results/ml.json` — 평가지표 수치

**검증**: 저장된 .joblib을 재로드하여 예측이 동일한지 확인 코드 포함

---

### Step 7 — report.md 자동 생성 (`07_report.py`)
**Task**
- `outputs/results/*.json`을 읽어 f-string 템플릿으로 report.md 조립
- 구성: 개요 → 데이터 준비(Pandas/Polars 비교) → 정제 요약 → EDA 결과 →
  통계 검정(t-test + 효과 크기) → 차트(이미지 링크) → ML 결과(지표 테이블) →
  결론·한계(타깃 누수 발견, 판단 번복 스토리 강조)

**산출물**
- `outputs/report.md` (차트 이미지 링크 포함)

**검증**: 스크립트 재실행 시 report.md가 최신 수치로 재생성되는지 확인

---

### Step 8 — 제출 준비
**Task**
- 전체 실행 순서 확인: `01 → 02 → 03 → 04 → 05 → 06 → 07` 순차 실행으로 재현되는지 최종 점검
- 모든 파일 주석 점검 (감점 항목)
- 실행결과 화면 캡처 → 본인 의견(개선 사항, 코드 품질)과 함께 PDF 작성
- 폴더 구조 그대로 zip: `캠퍼스명_반_이름_day2종합실습.zip`

**산출물**
- 제출용 zip + PDF

---

## 5. 발표 스토리라인 (5분)

1. 문제 정의: 운행 정보만으로 결제수단을 예측할 수 있는가?
   (원 주제였던 "매칭 위치 예측"이 데이터 한계로 불가능했던 경위 포함)
2. **핵심 발견 1 — 타깃 누수**: tip_amount가 정답을 노출(카드만 팁>0 90.2%) → 7개 컬럼 제외
3. **핵심 발견 2 — 판단 번복**: 결측 95만 건을 불량 데이터로 봤으나 실제로는 Flex Fare였음
   → 제외 대상에서 정상 클래스로 전환, 3-클래스로 재정의 (근거: TLC 공식 데이터 사전)
4. 통계: 신용카드/현금 요금 차이 t-test — 통계적 유의성과 실질적 효과 크기의 구분
5. 모델: 베이스라인(정확도 0.68 / macro F1 0.27) 대비 성능.
   **정확도가 낮아진 것이 소수 클래스를 살린 대가임을 설명** (가장 설명력 있는 대목)
6. 한계: 현금 클래스 precision이 낮음 / 개선 방향

## 6. 시간 배분 가이드

- Step 1~3 (로딩 비교·전처리·EDA): 40%
- Step 4~6 (통계 검정·시각화·ML): 40%
- Step 7~8 (리포트·제출): 20%
