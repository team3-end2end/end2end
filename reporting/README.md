# 자동 보고서 사용 가이드

`reporting`은 EDA, 전처리, 모델, 평가 결과를 단계별 JSON으로 받아 팀 내부용 `report.md`와 비개발자용 `report.html`을 함께 생성합니다. 생성된 문서를 직접 수정하지 말고 입력 JSON 또는 템플릿을 수정하세요.

## 빠른 확인

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# 현재 운영 입력: 모든 단계가 준비 중
.venv/bin/python -m reporting.generate

# 값이 모두 채워진 화면 확인
.venv/bin/python -m reporting.generate --example

# 커밋된 결과가 입력과 일치하는지 확인
.venv/bin/python -m reporting.generate --check
.venv/bin/pytest
```

예시의 EDA·전처리 값은 현재 저장소의 분석 결과를 사용하지만 모델·평가 값은 디자인 확인용 가상 값입니다. 예시 보고서에는 이를 알리는 `SAMPLE` 경고가 표시됩니다.

## 단계별 담당 파일

| 단계 | 파일 | 주요 내용 |
|---|---|---|
| 실행 | `reporting/data/run.json` | 실행 ID, 데이터셋, 타깃 정의 |
| EDA | `reporting/data/eda.json` | 품질, 클래스 분포, 발견, 그림 |
| 전처리 | `reporting/data/preprocessing.json` | 필터 이력, 피처, 변환, 최종 크기 |
| 모델 | `reporting/data/model.json` | 데이터 분할, 후보, 파라미터, 학습 시간 |
| 평가 | `reporting/data/evaluation.json` | 모델·클래스 지표, 혼동행렬, 최종 선택 |

모든 파일은 다음 공통 구조를 사용합니다.

```json
{
  "schema_version": 1,
  "stage": "model",
  "status": "pending",
  "generated_at": null,
  "data": {},
  "error": null
}
```

- `pending`: 값이 아직 없으며 보고서에 `준비 중`으로 표시됩니다.
- `complete`: `data`가 단계별 필수 형식을 모두 만족해야 합니다.
- `failed`: `error.message`에 실패 원인을 기록합니다.
- 상세 필드와 타입은 `reporting/schemas/stage-v1.schema.json`, 완성 예시는 `reporting/examples/data/`를 참고합니다.

## Python에서 값 저장하기

노트북이나 파이프라인에서는 직접 파일을 열기보다 검증과 원자적 저장을 제공하는 헬퍼를 권장합니다.

```python
from reporting import write_stage_result

write_stage_result(
    stage="model",
    status="complete",
    data={
        "split": {
            "strategy": "stratified 70/15/15 split",
            "train_samples": 2723878,
            "validation_samples": 583688,
            "test_samples": 583689,
            "random_seed": 42,
        },
        "candidates": [
            {
                "id": "random_forest",
                "name": "Random Forest",
                "library": "scikit-learn",
                "parameters": {"n_estimators": 300},
                "training_seconds": 326.18,
                "artifact_path": "models/random_forest.joblib",
                "status": "complete",
            }
        ],
    },
)
```

실패를 보고서에 남기려면 다음처럼 기록합니다.

```python
write_stage_result(
    stage="model",
    status="failed",
    data={},
    error={"message": "모델 학습 실패", "details": "메모리 부족"},
)
```

## 모델 파이프라인 연결

모델 저장과 평가 계산이 모두 성공한 뒤 마지막 단계에서 보고서를 생성합니다.

```python
from reporting import generate_reports, write_stage_result

write_stage_result("model", model_result)
write_stage_result("evaluation", evaluation_result)
generate_reports(strict=True)
```

`strict=True`는 다섯 단계가 모두 `complete`가 아니면 실패합니다. 개발 중 부분 결과를 확인할 때는 `strict=False`를 사용합니다.

## 이미지와 출력 규칙

- 그림의 `path`는 저장소 루트에서 실행했을 때 찾을 수 있는 상대 경로로 작성합니다.
- Markdown은 출력 위치 기준 상대 링크를 생성합니다.
- HTML은 이미지를 Base64로 포함하므로 `report.html` 하나만 전달할 수 있습니다.
- 그림이 없으면 보고서 생성을 중단하지 않고 `이미지 준비 중`을 표시합니다.
- 숫자, 백분율, 초 단위와 빈 값 표시는 공통 포맷터가 담당합니다.

## CI 연결 시

이번 작업에는 GitHub Actions를 추가하지 않았습니다. 이후 CI에서는 대용량 모델을 다시 학습시키기보다 다음 명령으로 입력과 커밋된 보고서의 일치 여부를 검사할 수 있습니다.

```bash
.venv/bin/python -m reporting.generate --strict --check
.venv/bin/pytest
```

종료 코드 `0`은 성공, `2`는 입력 계약 위반 또는 오래된 보고서를 의미합니다.
