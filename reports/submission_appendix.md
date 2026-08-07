## 7. 실행 결과 화면

모든 화면은 `README.md`에 적힌 명령어를 그대로 실행한 결과다.

### 7.1 전체 파이프라인 실행 — `python main.py`

명령 한 번으로 통계 표 → Seaborn 차트 → Plotly HTML → 보고서까지 생성된다.

![python main.py 실행 시작 — 통계 분석 표 생성](captures/main-py-capture-1.png)

이어지는 화면에서 차트와 보고서가 생성되고, 마지막에 산출물 11개의 존재와 크기를 코드가 직접 확인한다.

![python main.py 실행 완료 — 산출물 확인](captures/main-py-capture-2.png)

> 중간의 `findfont: Failed to find font weight bold` 경고는 한글 폰트에 굵은 두께가 없어 기본 두께로 대체했다는 안내이며, 차트는 정상 생성된다(바로 아래 저장 로그로 확인).

### 7.2 테스트 — `pytest`

보고서 입력 계약(스키마·의미 검증)과 생성기에 대한 테스트 24개가 모두 통과한다.

![pytest 24 passed](captures/py-test-capture.png)

### 7.3 모델 학습 결과 — `tail -35 pipeline_run.log`

2026-08-07 실행분(Optuna 100 trial, 약 40분)의 로그다. 최적 모델 선정 → 평가 → 저장까지의 실제 출력이며, 본 보고서 5장의 수치가 여기서 나왔다. 전체 재현은 `python main.py --full`.

![학습 로그 — 평가 지표와 혼동행렬](captures/accuracy-macrof1-class.png)
