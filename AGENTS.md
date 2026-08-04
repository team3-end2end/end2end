# AGENTS.md — 에이전트 작업 규약

이 리포에서 코드를 쓰는 모든 에이전트(Claude Code, Cursor, Copilot 등)는 작업 시작 전 이 문서를 읽는다.
사람 팀원도 같은 규칙을 따른다.

## 프로젝트

NYC Yellow Taxi 2026-05 데이터로 **결제수단(신용카드 / 현금 / Flex Fare) 3-클래스 예측**하는 End2End 분석 과제.
과제 요구사항·채점 기준·단계별 산출물은 **[plan.md](plan.md)가 단일 기준(single source of truth)**이다.
판단이 바뀐 경위는 [WORKLOG.md](WORKLOG.md)에 남긴다 — 번복된 판단도 지우지 않는다.
plan.md와 충돌하는 판단이 필요하면 임의로 진행하지 말고 PR 본문에 "plan.md와 다른 선택" 항목으로 명시한다.

## 시작 전 확인

1. `plan.md` — 내가 맡은 Step의 Task / 산출물 / 검증 기준
2. `git log --oneline origin/main -10` — 다른 팀원이 이미 만든 것
3. 내 Step의 **입력 파일이 존재하는지** (앞 Step 산출물에 의존하는 경우)

## 파일 소유권 — 충돌 방지

여러 에이전트가 동시에 작업하므로, **자기 Step 파일 밖은 건드리지 않는다.**

| 담당 | 소유 파일 | 산출물 |
|---|---|---|
| Step 1 | `src/01_load_compare.py` | `outputs/results/load_compare.json` |
| Step 2 | `src/02_clean_eda.py` | `outputs/figures/*`, `data/processed/cleaned.parquet`, `outputs/results/eda.json` |
| Step 3 | `src/03_stats.py` | `outputs/results/stats.json` |
| Step 4 | `src/04_ml_pipeline.py` | `outputs/model/*.joblib`, `outputs/results/ml.json` |
| Step 5 | `src/05_report.py` | `outputs/report.md` |

**공용 파일**(`plan.md`, `README.md`, `AGENTS.md`, `requirements.txt`)을 수정해야 하면
그 변경만 담은 별도 커밋으로 분리하고 PR 본문에 이유를 남긴다.

**단계 간 계약**: 앞 Step은 뒤 Step이 읽을 파일의 **경로와 키 이름을 바꾸지 않는다.**
바꿔야 하면 PR 제목에 `[BREAKING]`을 붙이고 영향받는 담당자를 PR에 멘션한다.

## 브랜치 · 커밋

- 브랜치: `<이름>/<step>-<요약>` (예: `minkyojung/step2-clean-eda`)
- `main`에 직접 푸시 금지. 모든 변경은 PR을 거친다.
- 커밋 메시지: 한 줄 요약 + 필요 시 본문. 무엇을 했는지가 아니라 **왜 했는지**를 남긴다.
- 커밋을 쪼갠다. "Step 2 전부"보다 "정제 로직", "Seaborn 차트", "Plotly 차트"로 나누면 리뷰가 가능해진다.

## PR 규칙 (가장 중요)

PR 본문은 **이번 브랜치에서 내린 의사결정의 기록**이다. 무엇을 했는지 나열하는 changelog가 아니다.
`.github/pull_request_template.md` 형식을 **그대로** 채운다.

### "의사결정"의 정의

**합리적인 다른 선택지가 있었는데 하나를 고른 것**만 의사결정이다.

- ✅ "이상치 기준을 trip_distance < 100으로 잡음" (99퍼센타일, IQR 등 대안 있었음)
- ✅ "tip_amount를 피처에서 제외" (넣으면 정확도는 오르지만 누수)
- ✅ "전체 409만 행 대신 50만 행 층화 샘플링" (실행 시간 vs 데이터 활용)
- ❌ "seaborn을 import함" — 선택지가 없었음. 이건 의사결정이 아니라 작업 내역.

의사결정이 하나도 없는 PR이면 TLDR에 "없음 (plan.md 지침 그대로 구현)"이라고 쓴다.
**억지로 만들어내지 않는다.**

### 근거에는 증거를 붙인다

각 의사결정의 근거에는 검증 가능한 것을 하나 이상 넣는다:
- 실제 수치 (예: "제거된 행 1,204건 = 전체의 0.03%")
- 파일·라인 참조 (예: `src/02_clean_eda.py:47`)
- 실행 출력 발췌

"성능이 더 좋아서", "일반적으로 권장되므로" 같은 근거 없는 서술은 리뷰에서 반려한다.

### PR 생성

```bash
gh pr create --base main --title "[Step N] 요약" --body-file <채운 템플릿 파일>
```

## 코드 규칙

- **모든 함수·주요 블록에 주석 필수** — 과제 채점 항목(완성도 10점)이며 주석 누락 시 감점된다.
- 경로는 스크립트 상단에 상수로 모아둔다. 하드코딩된 절대 경로 금지.
- 랜덤 시드는 `RANDOM_STATE = 42`로 고정한다 (재현성 — 결과 수치가 매번 바뀌면 리포트를 신뢰할 수 없다).
- 각 스크립트는 **단독 실행 가능**해야 한다. `python src/0N_*.py`로 돌아가고, 앞 단계 산출물이 없으면 명확한 에러 메시지를 낸다.
- 수치 결과는 콘솔 출력 **+ `outputs/results/*.json` 저장** 둘 다 한다. Step 5의 report.md 자동 생성이 이 JSON을 읽는다.

## 하지 말 것

- 데이터 파일(`*.parquet`)·모델(`*.joblib`)·차트 결과물 커밋 금지 → `.gitignore` 확인
- 남의 Step 파일 "개선" 금지. 문제를 발견하면 고치지 말고 PR 본문이나 이슈에 남긴다.
- 검증하지 않은 수치를 리포트·PR에 쓰지 않는다. 실행해서 나온 값만 적는다.
- plan.md의 요구사항 체크리스트를 임의로 삭제하지 않는다 (완료 시 `[x]` 체크만).
