# end2end — NYC Yellow Taxi 결제수단 예측

2026년 5월 NYC Yellow Taxi 운행 데이터로 **결제수단(신용카드 / 현금 / Flex Fare)** 을 예측하는 End2End 분석 프로젝트.

## 문서

| 문서 | 내용 |
|---|---|
| [plan.md](plan.md) | 과제 요구사항, 채점 기준, 단계별 Task·산출물 — **작업의 단일 기준** |
| [WORKLOG.md](WORKLOG.md) | 판단이 바뀐 지점과 근거 (시간순). 번복된 판단도 남긴다 |
| [AGENTS.md](AGENTS.md) | 에이전트·팀원 작업 규약 (파일 소유권, 브랜치, PR, 코드 규칙) |
| [.github/pull_request_template.md](.github/pull_request_template.md) | PR 본문 형식 (의사결정 TL;DR + 근거) |

## 데이터

원본은 용량(66.5MB) 때문에 커밋하지 않는다. 아래에서 받아 `data/raw/`에 둔다.

```bash
mkdir -p data/raw data/processed
curl -o data/raw/yellow_tripdata_2026-05.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet
```

- `data/raw/` — 원본. **읽기 전용, 절대 덮어쓰지 않는다**
- `data/processed/` — 전처리 결과 (`cleaned.parquet`, Step 2가 생성)

## 실행

```bash
# 버전이 고정돼 있으므로 Python 3.11 venv를 새로 만들어 설치할 것.
# 시스템 파이썬에 pandas 1.x가 깔려 있으면 01이 다른 API 동작을 탄다.
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python src/01_load_compare.py   # Pandas vs Polars 로딩·전처리 성능 비교
.venv/bin/python src/02_preprocess.py     # 정제 + 파생변수 → cleaned.parquet
.venv/bin/python src/03_eda.py            # EDA (숫자로만) + 피처 화이트리스트 확정
```

아래는 아직 작성 전이다 (담당자는 [plan.md](plan.md)의 담당자 배정 표 참조).

```bash
.venv/bin/python src/04_stats.py          # t-test + 효과 크기
.venv/bin/python src/05_visualize.py      # Seaborn · Plotly 차트
.venv/bin/python src/06_ml_pipeline.py    # sklearn Pipeline 학습 · 평가 · 저장
.venv/bin/python src/07_report.py         # outputs/report.md 자동 생성
```

데이터를 바꾸는 단계는 `02` 하나뿐이고, `03`·`04`·`05`는 `cleaned.parquet`을 읽기만 하므로
서로 순서를 바꿔도 된다. `06`은 `03`이 만든 `features.json`이, `07`은 앞 단계 전부가 필요하다.

## 협업

1. 브랜치를 판다: `<이름>/<step>-<요약>`
2. 작업 전 [AGENTS.md](AGENTS.md)를 읽는다 (에이전트 사용 시에도 동일)
3. PR 템플릿을 채워 PR을 연다 — **이번 브랜치에서 내린 의사결정과 그 근거**가 핵심
