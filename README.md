# NYC Yellow Taxi 결제수단 예측 (end2end)

## 목적

NYC 옐로우 택시 2026년 5월 승차 기록 409만 건을 분석해, 각 트립의 결제수단(카드 / 플렉스 페어 / 현금)을 예측하는 3클래스 분류 모델을 만든다.
Optuna로 XGBoost·HistGBM·로지스틱 회귀를 탐색해 XGBoost를 선택했고, Test macro F1 0.6688 / accuracy 0.7428을 기록했다. 상세 결과와 해석은 [reports/final_report.md](reports/final_report.md) 참고.

## 환경

- Python 3.11
- 주요 패키지: pandas, polars, seaborn, plotly, scipy, scikit-learn, xgboost, optuna, joblib, Jinja2 — 전체 목록과 버전은 `requirements.txt`

## 실행

```bash
pip install -r requirements.txt
python main.py            # 저장된 학습 결과 재사용 → 통계 표·차트 HTML·보고서 생성 (수 분)
python main.py --full     # Optuna 탐색 + 모델 학습부터 전체 재실행 (~40분)
```

테스트:

```bash
pytest
```



## 데이터

- 원본 출처: [https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet) (NYC TLC, 4,090,836행 × 20열)
- 전처리된 학습용 데이터(`eda/data_preparation/payment_type_dataset.parquet`, 3,891,255행)가 저장소에 포함되어 있어 clone 후 바로 실행할 수 있다.
- 전처리 과정: [eda/data_preparation/data_preparation.ipynb](eda/data_preparation/data_preparation.ipynb) (Pandas vs Polars 로딩 비교 포함)



## 산출물


| 파일                                            | 내용                            |
| --------------------------------------------- | ----------------------------- |
| `report.md` / `report.html`                   | 자동 생성 분석 보고서 (기술 문서 / 대시보드)   |
| `reports/final_report.md`                     | 제출용 최종본 — 자동 생성 수치 + 결과 해석    |
| `outputs/model_*.joblib`                      | 학습된 sklearn Pipeline (전처리 포함) |
| `outputs/results.csv`                         | 실행별 설정·성능 기록                  |
| `data_analysis/outputs/figures/plotly_*.html` | Plotly 인터랙티브 차트 5종            |
| `data_analysis/outputs/tables/*.csv`          | 통계 검정(t-test)·효과크기 결과표        |


보고서 시스템 상세: [reporting/README.md](reporting/README.md) · 완성 화면 예시: [reporting/examples/report.example.md](reporting/examples/report.example.md)

## 팀 구성 / 역할


| 담당                 | 작업                                                      |
| ------------------ | ------------------------------------------------------- |
| 데이터 전처리 (이찬혁, 정민교) | 원본 정제·필터링, 학습용 데이터셋 구축 (`eda/data_preparation/`)        |
| EDA·시각화 (이은혜)      | 구조·통계 분석, 차트 (`eda/`, `data_analysis/process.ipynb`)    |
| 통계 분석 (안중범)        | t-test·효과크기·피처 검정 (`data_analysis/feature_analysis.py`) |
| ML 파이프라인 (장소영)     | Optuna 튜닝 학습 파이프라인 (`pipeline.py`, `config.py`)         |
| 보고서 자동화 (이재호)      | report.md/html 생성 시스템 (`reporting/`)                    |
| 결과 해석·통합 (이재호)     | 모델 비교 해석, 최종 보고서 (`reports/`)                           |


