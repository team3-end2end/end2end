# Step 2 — 전처리

원본 4,090,836행 × 20열 → 정제 후 **3,891,795행 × 25열**
(보존율 95.13%)

## 제거 규칙별 해당 건수

| 규칙 | 건수 |
|---|---|
| `fare_amount <= 0` | 6,087 |
| `trip_distance <= 0` | 107,168 |
| `trip_distance >= 200` | 73 |
| `trip_duration_min <= 0` | 51,367 |
| `trip_duration_min >= 180` | 1,404 |

`payment_type ∉ [0, 1, 2]` 제외: 34,971건 (무료·분쟁, 학습 불가)

## 클래스 분포

| 결제수단 | 비율 |
|---|---|
| 신용카드 | 68.36% |
| Flex Fare | 22.57% |
| 현금 | 9.07% |

## 결측 처리

결측 4,776,855건은 전부 `payment_type=0`(Flex Fare) 행에서 발생하며
(교차검증: True), 결함이 아니라 **해당 없음**이다.
미터기 세부 요금 필드가 애초에 존재하지 않는 요금제이므로 **대체하지 않고 그대로 둔다.**
해당 컬럼: `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, `Airport_fee`
