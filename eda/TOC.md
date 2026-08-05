# 실전 Feature Engineering — 전체 목차

> 출처: `2) 데이터분석 및 AIOps_3.실전 Feature Engineering_박병선p.pdf` (SK AX / SKALA, 총 75p)
> 페이지 번호는 PDF 물리 페이지 기준

## 0. 표지 / 인트로 (p1–2)

## 1. Intro (p3–9)
분석 Framework, 분석 프로세스별 고려사항

- p4 — 분석 Framework: "분석 모델 개발"을 위한 작업 활동·절차·산출물 가이드라인
- p5–8 — 분석 방법론 프로세스별 고려사항
  (비즈니스 이해 → 가설 수립 → 데이터 준비 → …)
- p9 — What is Feature Engineering?

## 2. EDA (p10–31)
Why EDA? / Basic Data Exploration / EDA Visualization Toolkit / Question-driven EDA

### 2.1 Why EDA?
- p11 — Why EDA? (Exploratory Data Analysis)
  - In Academia: 데이터를 탐색하여 구조와 특성을 이해하는 과정
  - In Practice: 선입견 없이 문제를 발견하고, FE·모델링 등 다음 분석 방향을 결정하는 과정
  - EDA 4축: 데이터 구조 이해 / 데이터 품질 확인 / 변수의 특성 이해 / 변수 간 관계 탐색 → Feature Engineering 방향 결정
- p12–13 — EDA Workflow (데이터 이해 → 인사이트 도출 → 데이터 개선 → 모델링)
- p14 — [참고] Descriptive Statistics

### 2.2 Basic Data Exploration
- p15 — EDA Checklist Summary
- p16–20 — EDA Checklist (df.shape 등 단계별 점검)
- p21 — Case Study
- p22 — EDA Checklist
- p23–24 — [참고] 결측값 탐색 by missingno
- p25 — EDA Checklist

### 2.3 EDA Visualization Toolkit
- p26 — Quantitative Variable 살펴보기
- p27 — Categorical Variable 살펴보기
- p28 — Quantitative vs Quantitative 살펴보기
- p29 — Quantitative vs Categorical 살펴보기
- p30 — Categorical vs Categorical 살펴보기

### 2.4 Question-driven EDA
- p31–32 — Question-driven EDA: 무엇을 물어볼 것인가? (From Questions to Insights)

## 3. Pre-processing : basic (p32–51)
Missing value / Outlier / Scaling / Encoding

- p33–36 — Missing Value
- p37–41 — Outlier
- p42–46 — Data Scaling
- p47–51 — Data Encoding

## 4. Pre-processing : advance (p52–68)
Transformation / Imbalance

- p53–57 — Variable Transformation
- p58–68 — Imbalanced data

## 5. Creating Better Features (p69–75)
Derived Features: Feature Creation Strategies

- p70 — Feature Creation Strategies: How can we create more informative features?
- p71–72 — From Questions to Features (좋은 질문이 좋은 Feature를 만들고, 좋은 Feature가 좋은 모델을 만든다)
- p73–74 — Where Do Good Features Come From? (Domain Knowledge + Data → Meaningful Features)
- p75 — 마무리
