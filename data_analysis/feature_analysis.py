"""
피처 적합도 통계 분석 — NYC Yellow Taxi 결제수단(payment_type) 예측

목적: 모델링 단계에 넘길 **피처 조합(레시피)** 을 통계 검정으로 검증한다.
      학습 기반 성능 평가는 쓰지 않는다. 판정 지표는 전부 검정/회귀 계열:
      Cohen's d, Cramer's V, eta^2, 카이제곱, McFadden pseudo-R^2, 우도비(LR) 검정.

주의: 여기서 행(row)을 자르는 부분(PART 3)은 **상호작용 구조를 찾기 위한 진단**이지,
      학습 데이터를 걸러내라는 뜻이 아니다. 발견된 구조는 PART 4에서
      전체 데이터에 대한 상호작용 피처로 번역해서 재검증한다.


실행
────
    python data_analysis/feature_analysis.py              # 산출물 생성 (기본, 15초)
    python data_analysis/feature_analysis.py analysis     # 근거 재현 (20초, 화면 출력만)
    python data_analysis/feature_analysis.py all          # 둘 다
    python data_analysis/feature_analysis.py 3            # 특정 PART만

PART 구성
────────
    필수 — 산출물이 나오는 경로
      build_features()  피처 생성 로직 (모델링팀이 그대로 가져다 씀)
      FEATURE_SETS      세트 정의
      TargetEncoder     지역 인코딩. split 이후 train fold 에서만 fit
      PART 5            parquet / CSV 저장

    선택 — 위 설계가 왜 그렇게 됐는지 보여주는 근거. 산출물과 무관하고
           화면 출력만 한다. 인계서(handoff_features.md) 수치가 여기서 나온다.
      PART 1  기초 확인, 표본 크기 함정
      PART 2  원본 피처 개별 검정
      PART 3  조건부 효과크기 → 상호작용 탐지
      PART 4  피처 조합 검정, 세트 비교

의존성: pandas, numpy, scipy, scikit-learn, statsmodels
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import KFold

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

RS = 42
DATA = Path(__file__).resolve().parents[1] / "eda/data_preparation/payment_type_dataset.parquet"

TARGET = "payment_label"
NUM_COLS = ["trip_distance", "fare_amount", "trip_duration",
            "cbd_congestion_fee", "tolls_amount"]
CAT_COLS = ["PULocationID", "DOLocationID", "hour",
            "day_of_week", "VendorID", "is_airport"]
AIRPORT = {132, 138, 1}          # JFK / LaGuardia / Newark
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]

# 분석 표본 크기. 검정통계량은 n에 비례해 커지므로 고정해 두고 비교한다.
N_SAMPLE = 400_000


# ──────────────────────────────────────────────────────────────────────
# 공통 지표
# ──────────────────────────────────────────────────────────────────────
def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """표준화 평균차. |d| 0.2 작음 / 0.5 중간 / 0.8 큼. n에 거의 무관."""
    na, nb = len(a), len(b)
    if min(na, nb) < 50:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def cramers_v(ct: pd.DataFrame) -> float:
    """범주형 연관 강도 0~1. 범주 수가 많으면 신호가 흩어져 과소평가되는 점 주의."""
    if min(ct.shape) < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    return np.sqrt((chi2 / n) / (min(ct.shape) - 1))


def eta_squared(groups: list[np.ndarray]) -> float:
    """일원분산분석 효과크기. 0.01 작음 / 0.06 중간 / 0.14 큼."""
    allv = np.concatenate(groups)
    gm = allv.mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_t = ((allv - gm) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan


def oof_target_encode(col: pd.Series, y: pd.Series, m: int = 100, k: int = 5) -> np.ndarray:
    """K-fold out-of-fold 스무딩 타겟 인코딩.

    같은 행의 정답이 자기 인코딩 값에 섞이면 누수다. fold 밖에서만 산출한다.
    m은 스무딩 강도 — 표본이 적은 범주를 전체 평균 쪽으로 끌어당긴다.
    """
    out = np.zeros(len(col))
    for tr, te in KFold(k, shuffle=True, random_state=RS).split(col):
        g = (pd.DataFrame({"c": col.iloc[tr].values, "y": y.iloc[tr].values})
             .groupby("c")["y"].agg(["mean", "size"]))
        prior = y.iloc[tr].mean()
        mp = (g["mean"] * g["size"] + prior * m) / (g["size"] + m)
        out[te] = col.iloc[te].map(mp).fillna(prior).to_numpy()
    return out


def fit_logit(X: pd.DataFrame, y: pd.Series):
    """표준화 로지스틱 회귀. 계수끼리 크기 비교 가능, exp(coef)=1SD당 오즈비."""
    Xs = (X - X.mean()) / X.std().replace(0, 1)
    return sm.Logit(y, sm.add_constant(Xs, has_constant="add")).fit(disp=0)


def pseudo_r2(X: pd.DataFrame, y: pd.Series) -> float:
    return round(fit_logit(X, y).prsquared, 4)


def coef_table(m) -> pd.DataFrame:
    return (pd.DataFrame({"coef": m.params, "p": m.pvalues, "OR(1SD당)": np.exp(m.params)})
            .round(4).sort_values("coef", key=abs, ascending=False))


def load() -> pd.DataFrame:
    return pd.read_parquet(DATA)


def header(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ──────────────────────────────────────────────────────────────────────
# PART 1 — 기초 확인 / 표본 크기 함정
# ──────────────────────────────────────────────────────────────────────
def part1(df: pd.DataFrame) -> None:
    header("PART 1 — 기초 확인, 그리고 왜 p값을 믿으면 안 되는가")

    print(df[TARGET].value_counts(), "\n")
    print(df[TARGET].value_counts(normalize=True).round(4), "\n")

    print(pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "결측수": df.isna().sum(),
        "결측%": (df.isna().mean() * 100).round(2),
        "고유값": df.nunique(),
    }), "\n")

    print("왜도 — 로그 변환 후보 판정")
    print(pd.DataFrame({
        "skew": df[NUM_COLS].skew().round(2),
        "skew_log1p": np.log1p(df[NUM_COLS].clip(lower=0)).skew().round(2),
    }), "\n")

    # n=389만이면 사소한 차이도 p<0.05가 된다. 효과크기를 함께 봐야 하는 이유.
    card = df.loc[df[TARGET] == "신용카드", "fare_amount"]
    cash = df.loc[df[TARGET] == "현금", "fare_amount"]
    print("표본 크기에 따른 p값 vs 효과크기 (fare_amount, 카드 vs 현금)")
    rows = []
    for n in [30, 100, 500, 2_000, 10_000, 100_000, len(cash)]:
        a = card.sample(n, random_state=RS)
        b = cash.sample(n, random_state=RS)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"n/group": n, "p": round(p, 4),
                     "유의(0.05)": p < 0.05, "Cohen_d": round(cohens_d(a, b), 4)})
    print(pd.DataFrame(rows).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────
# PART 2 — 개별 피처 검정 (전체 데이터)
# ──────────────────────────────────────────────────────────────────────
def part2(df: pd.DataFrame) -> None:
    header("PART 2 — 원본 피처 11개, 개별 검정")

    card = df[df[TARGET] == "신용카드"]
    cash = df[df[TARGET] == "현금"]

    print("[2-1] 수치형 — 카드 vs 현금 (Welch t / Mann-Whitney U / Cohen's d)")
    rows = []
    for c in NUM_COLS:
        a, b = card[c], cash[c]
        t, p = stats.ttest_ind(a, b, equal_var=False)         # H0: 평균이 같다
        _, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")   # 비모수 대응
        rows.append({"feature": c, "카드평균": a.mean(), "현금평균": b.mean(),
                     "t": t, "p(Welch)": p, "p(MWU)": p_u, "Cohen_d": cohens_d(a, b)})
    print(pd.DataFrame(rows).sort_values("Cohen_d", key=abs, ascending=False)
          .round(4).to_string(index=False), "\n")

    print("[2-2] 수치형 — 3클래스 (ANOVA / Kruskal / eta^2)")
    sub = df.groupby(TARGET, group_keys=False).sample(n=5_000, random_state=RS)
    labels = sub[TARGET].unique()
    rows = []
    for c in NUM_COLS:
        gs = [sub.loc[sub[TARGET] == l, c].to_numpy() for l in labels]
        f, p_f = stats.f_oneway(*gs)                          # H0: 세 평균이 모두 같다
        h, p_h = stats.kruskal(*gs)
        gs_full = [df.loc[df[TARGET] == l, c].to_numpy() for l in labels]
        rows.append({"feature": c, "F": f, "p(ANOVA)": p_f,
                     "p(Kruskal)": p_h, "eta2(전체)": eta_squared(gs_full)})
    print(pd.DataFrame(rows).sort_values("eta2(전체)", ascending=False)
          .round(4).to_string(index=False), "\n")

    print("[2-3] 범주형 — 카이제곱 / Cramer's V")
    cc = df[df[TARGET] != "Flex Fare"]
    rows = []
    for c in CAT_COLS:
        ct3 = pd.crosstab(df[c], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct3)         # H0: 피처와 타깃은 독립
        rows.append({"feature": c, "범주수": ct3.shape[0], "chi2": chi2, "p": p,
                     "V(3클래스)": cramers_v(ct3),
                     "V(카드vs현금)": cramers_v(pd.crosstab(cc[c], cc[TARGET]))})
    print(pd.DataFrame(rows).sort_values("V(카드vs현금)", ascending=False)
          .round(4).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────
# PART 3 — 진단: 조건부로 보면 신호가 살아나는가? (상호작용 탐지)
# ──────────────────────────────────────────────────────────────────────
def part3(df: pd.DataFrame) -> None:
    header("PART 3 — 진단: 조건부 효과크기로 상호작용 찾기")
    print("※ 이 PART의 슬라이스는 '학습 데이터를 걸러내라'가 아니라\n"
          "   '어느 조건에서 피처의 효과가 달라지는가'를 찾는 진단이다.\n"
          "   결과는 PART 4에서 상호작용 피처로 번역해 전체 데이터로 재검증한다.\n")

    def slice_signal(name: str, d: pd.DataFrame) -> dict:
        cc = d[d[TARGET] != "Flex Fare"]
        out = {"조건": name, "n": len(d),
               "현금비율": round((cc[TARGET] == "현금").mean(), 3) if len(cc) else np.nan}
        for c in NUM_COLS:
            out["d_" + c] = round(
                cohens_d(cc.loc[cc[TARGET] == "신용카드", c],
                         cc.loc[cc[TARGET] == "현금", c]), 3)
        return out

    night = df.hour.isin(NIGHT_HOURS)
    slices = {
        "전체(기준)":        df,
        "JFK 승차":          df[df.PULocationID == 132],
        "LGA 승차":          df[df.PULocationID == 138],
        "공항 하차":          df[df.DOLocationID.isin({132, 138})],
        "비공항":            df[df.is_airport == 0],
        "Vendor1":          df[df.VendorID == 1],
        "Vendor2":          df[df.VendorID == 2],
        "심야(22-5시)":      df[night],
        "CBD 통과":          df[df.cbd_congestion_fee > 0],
        "CBD 미통과":        df[df.cbd_congestion_fee == 0],
        "Vendor1×CBD미통과": df[(df.VendorID == 1) & (df.cbd_congestion_fee == 0)],
    }
    print(pd.DataFrame([slice_signal(k, v) for k, v in slices.items()]).to_string(index=False))

    # 왜 fare_amount가 전체에서 무력했는가 — JFK 안에 두 요금체계가 섞여 있다
    print("\n[3-1] JFK 승차 내부 구조")
    s = df[(df.PULocationID == 132) & (df[TARGET] != "Flex Fare")].copy()
    s["is_cash"] = (s[TARGET] == "현금").astype(int)
    print(s.groupby(TARGET)[NUM_COLS].median().round(2), "\n")

    s["flat70"] = s.fare_amount.between(68, 72)               # JFK↔맨해튼 $70 정액제
    print("$70 정액 여부별 현금비율")
    print(s.groupby("flat70").agg(n=("is_cash", "size"), 현금비율=("is_cash", "mean")).round(4), "\n")

    s["dbin"] = pd.cut(s.trip_distance, [0, 2, 5, 10, 14, 18, 25, 300])
    print("거리 구간별 현금비율")
    print(s.groupby("dbin", observed=True)
          .agg(n=("is_cash", "size"), 현금비율=("is_cash", "mean")).round(4))

    ct = pd.crosstab(s.DOLocationID, s[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"\nH0: JFK 승차 내에서 하차지역과 결제수단은 독립")
    print(f"chi2={chi2:,.0f}  dof={dof}  p={p:.3g}  Cramer's V={cramers_v(ct):.3f}")


# ──────────────────────────────────────────────────────────────────────
# PART 4 — 피처 조합(레시피) 검정  ★ 모델링팀에 넘길 산출물
# ──────────────────────────────────────────────────────────────────────
def build_features(d: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    """레시피 전체를 한 번에 생성하고 블록별 컬럼 목록을 함께 돌려준다."""
    F = pd.DataFrame(index=d.index)
    h = d.hour

    # 원본 그대로 — 트리 계열은 단조 변환에 불변이라 로그가 필요 없다
    for c in ["trip_distance", "fare_amount", "trip_duration",
              "tolls_amount", "cbd_congestion_fee"]:
        F[c] = d[c].astype(float)
    F["hour"] = h.astype(float)
    F["day_of_week"] = d.day_of_week.astype(float)

    # A. 기본 수치 (로그 변환)
    F["log_dist"] = np.log1p(d.trip_distance)
    F["log_fare"] = np.log1p(d.fare_amount)
    F["log_dur"] = np.log1p(d.trip_duration)
    F["tolls"] = d.tolls_amount
    F["cbd"] = d.cbd_congestion_fee

    # B. 요금 구조 — "요금이 어떻게 매겨졌나"를 드러내는 비율/플래그
    F["fare_per_mile"] = (d.fare_amount / d.trip_distance).clip(0, 100)
    F["fare_per_min"] = (d.fare_amount / d.trip_duration).clip(0, 50)
    F["speed"] = (d.trip_distance / (d.trip_duration / 60)).clip(0, 80)
    F["is_flat70"] = d.fare_amount.between(68, 72).astype(float)     # JFK 정액제
    F["has_tolls"] = (d.tolls_amount > 0).astype(float)

    # C. 시간
    F["hour_sin"] = np.sin(2 * np.pi * h / 24)
    F["hour_cos"] = np.cos(2 * np.pi * h / 24)
    F["is_night"] = h.isin(NIGHT_HOURS).astype(float)
    F["is_weekend"] = d.day_of_week.isin([5, 6]).astype(float)

    # D. 지리 — 고차원 범주는 OOF 타겟 인코딩으로 1차원에 모은다
    F["pu_te"] = oof_target_encode(d.PULocationID, y)
    F["do_te"] = oof_target_encode(d.DOLocationID, y)
    F["route_te"] = oof_target_encode(
        d.PULocationID.astype(str) + "_" + d.DOLocationID.astype(str), y)
    F["is_airport"] = d.is_airport.astype(float)

    # E. 상호작용 — PART 3에서 찾은 구조의 번역
    F["air_x_fare"] = F.is_airport * F.log_fare
    F["air_x_dist"] = F.is_airport * F.log_dist
    F["flat70_x_dist"] = F.is_flat70 * F.log_dist
    F["cbd0_x_dur"] = (d.cbd_congestion_fee == 0).astype(float) * F.log_dur

    # ※ VendorID는 의도적으로 제외했다.
    #   - 이 데이터셋에는 사업자가 3종(1/2/6)뿐이고, 카드 vs 현금만 보면 2종(1/2)이다.
    #     data_preparation 필터에서 Vendor 7은 통째로 빠졌다.
    #   - Vendor 6은 100% Flex Fare라 "항목을 안 적는 사업자라서 항목이 안 적혀 있다"는
    #     동어반복이다 (eda/handoff/3_타겟분석.md).
    #   - 제외 비용은 pseudo-R^2 0.0563 -> 0.0530 (-5.9%).

    blocks = {
        "A 기본수치": ["log_dist", "log_fare", "log_dur", "tolls", "cbd"],
        "B 요금구조": ["fare_per_mile", "fare_per_min", "speed", "is_flat70", "has_tolls"],
        "C 시간": ["hour_sin", "hour_cos", "is_night", "is_weekend"],
        "D 지리": ["pu_te", "do_te", "route_te", "is_airport"],
        "E 상호작용": ["air_x_fare", "air_x_dist", "flat70_x_dist", "cbd0_x_dur"],
    }
    return F, blocks


# ──────────────────────────────────────────────────────────────────────
# 모델 계열별 피처 세트
#
# 하나의 "추천 조합"을 주지 않는 이유: 모델마다 잘 받는 표현이 다르다.
# 트리는 로그·표준화가 무의미하고 상호작용을 스스로 찾는 반면,
# 선형 모델은 셋 다 필요하고 다중공선에 취약하다.
# 아래 세 세트는 PART 4 검정 결과를 각 모델 특성에 맞춰 자른 것이다.
#
# ※ *_te 컬럼은 여기 없다. 타겟 인코딩은 train/test split '이후'
#   train fold에서만 적합해야 해서, 지역 ID를 그대로 넘기고
#   TargetEncoder 를 같이 제공한다. (아래 클래스 참고)
# ──────────────────────────────────────────────────────────────────────
LOC_COLS = ["PULocationID", "DOLocationID"]

FEATURE_SETS: dict[str, dict] = {
    "performance": {
        "설명": "부스팅 계열 (XGBoost / LightGBM / CatBoost)",
        "목표": "설명력을 좀 포기하더라도 성능 우선. 피처가 많고 서로 겹쳐도 됨",
        "cols": [
            # 로그 변환 안 함 — 트리는 단조 변환에 불변 (검정: +0.0002로 무의미)
            "trip_distance", "fare_amount", "trip_duration",
            "tolls_amount", "cbd_congestion_fee",
            # 비율 파생. 트리는 나눗셈을 스스로 만들지 못해 직접 줘야 한다
            "fare_per_mile", "fare_per_min", "speed",
            "is_flat70", "has_tolls",
            # 시간은 정수로. 트리가 알아서 구간을 자르므로 원핫/순환 불필요
            "hour", "day_of_week", "is_night", "is_weekend",
            "is_airport",
            # 상호작용. 트리가 스스로 찾긴 하지만 명시하면 수렴이 빨라지고,
            # 이 세트는 다중공선을 신경 쓰지 않아도 되므로 그냥 넣는다
            "air_x_fare", "air_x_dist", "flat70_x_dist", "cbd0_x_dur",
        ],
        "메모": "정규화·표준화 불필요. 지역은 LightGBM native categorical 로 넣거나 "
                "TargetEncoder 를 쓰거나 둘 중 하나 (TE 쪽이 pseudo-R² 기준 유리)",
    },
    "interpretable": {
        "설명": "회귀 계열 (LogisticRegression / ElasticNet)",
        "목표": "계수를 해석할 수 있고 오버피팅 위험이 낮을 것. 최소 피처",
        "cols": [
            # 로그 변환 필수 — skew 3.3 → 0.9, 계수 안정성 때문
            "log_fare",
            # 총액보다 단가가 정보량이 많음 (base 대비 +0.0021, 파생 중 1위)
            "fare_per_mile",
            # 시간은 hour_cos 하나만. hour_sin 은 p=0.62로 유의하지 않았음
            "hour_cos",
        ],
        # + pu_te, do_te, route_te (TargetEncoder 가 붙임) = 총 6개
        "메모": "최대 VIF 2.91 — 다중공선 없음. 상호작용 항은 일부러 뺐다: "
                "지역 타겟 인코딩이 이미 공항을 알고 있어서 중복이다 (증분 +0.0000). "
                "단 인코딩을 안 쓸 거면 air_x_fare 를 넣어야 한다 (그때는 +0.0035)",
    },
}


TE_COLS = ["pu_te", "do_te", "route_te"]


def set_cols(name: str, encoded: bool = True) -> list[str]:
    """세트의 컬럼 목록. encoded=True면 타겟 인코딩 3개가 붙은 최종 형태."""
    return list(FEATURE_SETS[name]["cols"]) + (TE_COLS if encoded else LOC_COLS)


def max_vif(X: pd.DataFrame) -> float:
    """최대 분산팽창계수. 10 초과면 다중공선 경고, 5 미만이면 안전."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Z = (X - X.mean()) / X.std().replace(0, 1)
    Z = sm.add_constant(Z, has_constant="add").to_numpy(dtype=float)
    return round(max(variance_inflation_factor(Z, i + 1) for i in range(X.shape[1])), 2)


class TargetEncoder:
    """지역 타겟 인코딩. 반드시 train fold로만 fit 할 것.

        enc = TargetEncoder().fit(X_train, y_train)
        X_train = enc.transform(X_train)
        X_test  = enc.transform(X_test)      # test 는 transform 만

    fit 을 전체 데이터로 하면 test 정답이 train 피처에 새어 들어간다.
    """

    def __init__(self, m: int = 100):
        self.m = m
        self.maps_: dict[str, pd.Series] = {}
        self.prior_: float = 0.0

    @staticmethod
    def _route(X: pd.DataFrame) -> pd.Series:
        return X.PULocationID.astype(str) + "_" + X.DOLocationID.astype(str)

    def fit(self, X: pd.DataFrame, y) -> "TargetEncoder":
        y = pd.Series(np.asarray(y), index=X.index)
        self.prior_ = float(y.mean())
        keys = {"pu_te": X.PULocationID, "do_te": X.DOLocationID, "route_te": self._route(X)}
        for name, col in keys.items():
            g = pd.DataFrame({"c": col.values, "y": y.values}).groupby("c")["y"].agg(["mean", "size"])
            self.maps_[name] = (g["mean"] * g["size"] + self.prior_ * self.m) / (g["size"] + self.m)
        return self

    def transform(self, X: pd.DataFrame, drop_ids: bool = True) -> pd.DataFrame:
        out = X.copy()
        keys = {"pu_te": X.PULocationID, "do_te": X.DOLocationID, "route_te": self._route(X)}
        for name, col in keys.items():
            out[name] = col.map(self.maps_[name]).fillna(self.prior_).to_numpy()
        return out.drop(columns=LOC_COLS) if drop_ids else out


def part4(df: pd.DataFrame) -> None:
    header("PART 4 — 피처 조합 검정 (전체 데이터, 행 필터링 없음)")

    # 피처 생성(특히 타겟 인코딩)은 전체 데이터로 해야 PART 5 산출물과 수치가 일치한다.
    # 40만 행만으로 인코딩하면 지역별 표본이 얇아져 pseudo-R^2가 과소평가된다.
    full = df[df[TARGET] != "Flex Fare"].reset_index(drop=True)
    y_full = (full[TARGET] == "현금").astype(int)
    F_full, blocks = build_features(full, y_full)

    # 검정은 표본 크기를 고정해야 통계량끼리 비교가 된다.
    idx = F_full.sample(N_SAMPLE, random_state=RS).index
    F, y, d = F_full.loc[idx], y_full.loc[idx], full.loc[idx]
    print(f"인코딩 {len(full):,}행 → 검정 {len(F):,}행  현금비율={y.mean():.4f}  (카드 vs 현금 2분류)\n")

    print("[4-1] 블록 단독 pseudo-R^2")
    print(pd.DataFrame([{"블록": k, "피처수": len(c), "pseudoR2": pseudo_r2(F[c], y)}
                        for k, c in blocks.items()]).to_string(index=False), "\n")

    print("[4-2] 블록 누적 — 우도비(LR) 검정으로 증분이 유의한지")
    rows, cols, prev = [], [], None
    for k, c in blocks.items():
        cols = cols + c
        m = fit_logit(F[cols], y)
        row = {"모델": "+" + k, "피처수": len(cols), "pseudoR2": round(m.prsquared, 4)}
        if prev is not None:
            lr = 2 * (m.llf - prev.llf)                       # H0: 추가 블록의 계수가 모두 0
            row["ΔR2"] = round(m.prsquared - prev.prsquared, 4)
            row["LR chi2"] = round(lr, 1)
            row["p"] = f"{stats.chi2.sf(lr, len(c)):.3g}"
        rows.append(row)
        prev = m
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    print("[4-3] 인코딩 방식 비교 — 같은 정보를 어떻게 넣느냐")
    h = d.hour
    enc = {
        "거리·요금·시간 원본":     pd.DataFrame({"a": d.trip_distance, "b": d.fare_amount, "c": d.trip_duration}),
        "거리·요금·시간 log1p":    F[["log_dist", "log_fare", "log_dur"]],
        "hour 정수":              pd.DataFrame({"h": h.astype(float)}),
        "hour sin/cos":           F[["hour_sin", "hour_cos"]],
        "hour is_night만":        F[["is_night"]],
        "hour 원핫(23)":          pd.get_dummies(h, prefix="h", drop_first=True).astype(float),
        "지역 ID 정수":            pd.DataFrame({"pu": d.PULocationID.astype(float),
                                                "do": d.DOLocationID.astype(float)}),
        "지역 타겟인코딩":          F[["pu_te", "do_te"]],
        "지역+경로 타겟인코딩":      F[["pu_te", "do_te", "route_te"]],
        "경로 단독":               F[["route_te"]],
    }
    print(pd.DataFrame([{"인코딩": k, "피처수": v.shape[1], "pseudoR2": pseudo_r2(v, y)}
                        for k, v in enc.items()]).to_string(index=False), "\n")

    print("[4-4] 요금구조 파생 — 개별 증분 (base = log 3변수)")
    base = F[["log_dist", "log_fare", "log_dur"]]
    b0 = pseudo_r2(base, y)
    rows = [{"추가": "(base)", "pseudoR2": b0, "증분": 0.0}]
    for c in blocks["B 요금구조"]:
        r = pseudo_r2(base.join(F[c]), y)
        rows.append({"추가": "+" + c, "pseudoR2": r, "증분": round(r - b0, 4)})
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    print("[4-5] 세트 비교 — 성능용 vs 해석용")
    raw_cols = [c for c in NUM_COLS + CAT_COLS if c != "VendorID"]     # 사업자 제외
    rows = [{"세트": "원본 피처 그대로", "피처수": len(raw_cols),
             "pseudoR2": pseudo_r2(d[raw_cols].astype(float), y),
             "maxVIF": max_vif(d[raw_cols].astype(float)), "비고": "기준선"}]
    for name, spec in FEATURE_SETS.items():
        X = F[set_cols(name)]
        rows.append({"세트": name, "피처수": X.shape[1], "pseudoR2": pseudo_r2(X, y),
                     "maxVIF": max_vif(X), "비고": spec["설명"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("  ※ pseudo-R²는 로지스틱 기준이라 performance 세트에는 불리하다.")
    print("     트리는 여기서 못 쓰는 비선형·상호작용을 추가로 쓰므로 실제 격차는 더 벌어진다.")
    print("     VIF는 10 넘으면 다중공선 경고, 5 미만이면 안전.\n")

    print("[4-6] interpretable 세트 계수 (표준화, OR=1SD당 현금 오즈비)")
    print(coef_table(fit_logit(F[set_cols("interpretable")], y)).to_string(), "\n")

    print("[4-7] 상호작용이 지역 인코딩과 중복인지 확인")
    core = set_cols("interpretable")
    rows = []
    for nm, cols in [
        ("지역 인코딩 O, 상호작용 X", core),
        ("지역 인코딩 O, 상호작용 O", core + ["air_x_fare"]),
        ("지역 인코딩 X, 상호작용 X", ["log_fare", "fare_per_mile", "is_airport"]),
        ("지역 인코딩 X, 상호작용 O", ["log_fare", "fare_per_mile", "is_airport", "air_x_fare"]),
    ]:
        rows.append({"구성": nm, "피처수": len(cols), "pseudoR2": pseudo_r2(F[cols], y)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("  → 지역 인코딩이 있으면 상호작용 증분이 사라진다. 인코딩이 이미 공항을 알고 있다.")


# ──────────────────────────────────────────────────────────────────────
# PART 5 — 산출물 저장  ★ 모델링팀이 실제로 받아가는 것
# ──────────────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent / "outputs"


def part5(df: pd.DataFrame) -> None:
    header("PART 5 — 산출물 저장")

    (OUT / "subsets").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    # 전체 데이터에 피처 생성. 타겟 인코딩은 전체 기준 OOF.
    d = df[df[TARGET] != "Flex Fare"].reset_index(drop=True)
    y = (d[TARGET] == "현금").astype(int)
    print(f"피처 생성 중… n={len(d):,}")
    F, blocks = build_features(d, y)
    F["payment_label"] = d[TARGET].to_numpy()
    F["is_cash"] = y.to_numpy()

    # 원본 키 컬럼. 피처로 쓰라는 게 아니라 다른 테이블과 조인하거나
    # 조건별로 쪼개 볼 때 필요해서 all_features 에만 붙여둔다.
    KEYS = ["PULocationID", "DOLocationID", "hour", "day_of_week"]
    for c in KEYS:
        F[c] = d[c].to_numpy()

    # ── 모델 계열별 세트 parquet
    # CSV로 하면 세트당 200MB, 총 1.5GB가 되어서 parquet(snappy)로 저장한다.
    #
    # 중요: *_te 컬럼은 넣지 않는다. 지역 ID를 그대로 넘기고
    #       모델링팀이 split 이후 TargetEncoder 로 train fold에서만 적합해야 한다.
    #       여기서 미리 구워 넣으면 test 정답이 train 피처로 새어 들어간다.
    print("\n[5-1] 모델 계열별 세트 parquet  (타겟 인코딩 미포함 — split 후 직접 적합)")
    rows = []
    for name, spec in FEATURE_SETS.items():
        cols = [c for c in spec["cols"] if c in F] + LOC_COLS
        p = OUT / "subsets" / f"{name}.parquet"
        F[cols + ["payment_label", "is_cash"]].to_parquet(p, index=False, compression="snappy")
        rows.append({"세트": name, "피처수": len(cols), "행": len(F),
                     "MB": round(p.stat().st_size / 1e6, 1), "용도": spec["설명"]})

    # 진단·시각화용. 여기에만 타겟 인코딩(OOF)과 블록 구분이 들어 있다.
    diag = [c for c in F.columns if c not in ("payment_label", "is_cash")]
    p = OUT / "subsets" / "all_features_diagnostic.parquet"
    F[diag + ["payment_label", "is_cash"]].to_parquet(p, index=False, compression="snappy")
    rows.append({"세트": "all_features_diagnostic", "피처수": len(diag), "행": len(F),
                 "MB": round(p.stat().st_size / 1e6, 1), "용도": "⚠️ 학습 금지 — 진단/시각화 전용"})
    print(pd.DataFrame(rows).to_string(index=False))

    # 세트 정의를 사람이 읽을 수 있게 같이 내보낸다
    pd.DataFrame([{"세트": k, "용도": v["설명"], "목표": v["목표"],
                   "피처수(인코딩 후)": len(v["cols"]) + len(TE_COLS),
                   "피처": ", ".join(v["cols"]) + " + pu_te, do_te, route_te",
                   "메모": v["메모"]}
                  for k, v in FEATURE_SETS.items()]
                 ).to_csv(OUT / "tables" / "00_feature_sets.csv", index=False, encoding="utf-8-sig")

    # ── 분석 결과표 CSV (사람이 읽고 문서에 붙이는 용도)
    print("\n[5-2] 결과표 CSV")
    s = F.sample(N_SAMPLE, random_state=RS)
    ys = s["is_cash"]

    t_solo = pd.DataFrame([{"블록": k, "피처수": len(c), "pseudoR2": pseudo_r2(s[c], ys)}
                           for k, c in blocks.items()])

    rows, cols, prev = [], [], None
    for k, c in blocks.items():
        cols = cols + c
        m = fit_logit(s[cols], ys)
        row = {"모델": "+" + k, "피처수": len(cols), "pseudoR2": round(m.prsquared, 4)}
        if prev is not None:
            lr = 2 * (m.llf - prev.llf)
            row |= {"dR2": round(m.prsquared - prev.prsquared, 4),
                    "LR_chi2": round(lr, 1), "p": stats.chi2.sf(lr, len(c))}
        rows.append(row)
        prev = m
    t_cum = pd.DataFrame(rows)

    raw_cols = [c for c in NUM_COLS + CAT_COLS if c != "VendorID"]
    raw = d.loc[s.index, raw_cols].astype(float)
    final_rows = [{"세트": "원본 피처 그대로", "피처수": len(raw_cols),
                   "pseudoR2": pseudo_r2(raw, ys), "maxVIF": max_vif(raw), "비고": "기준선"}]
    for name, spec in FEATURE_SETS.items():
        X = s[set_cols(name)]
        final_rows.append({"세트": name, "피처수": X.shape[1], "pseudoR2": pseudo_r2(X, ys),
                           "maxVIF": max_vif(X), "비고": spec["설명"]})
    t_final = pd.DataFrame(final_rows)

    t_coef = coef_table(fit_logit(s[set_cols("interpretable")], ys)).reset_index(names="feature")

    card = df[df[TARGET] == "신용카드"]
    cash = df[df[TARGET] == "현금"]
    t_effect = pd.DataFrame(
        [{"feature": c, "종류": "수치형", "지표": "Cohen_d",
          "값": round(cohens_d(card[c], cash[c]), 4)} for c in NUM_COLS] +
        [{"feature": c, "종류": "범주형", "지표": "CramersV",
          "값": round(cramers_v(pd.crosstab(df.loc[df[TARGET] != "Flex Fare", c],
                                            df.loc[df[TARGET] != "Flex Fare", TARGET])), 4)}
         for c in CAT_COLS]
    ).sort_values("값", key=abs, ascending=False)

    # t-test 증빙 — Welch t/p와 효과크기를 함께 저장 (n=389만이라 p만으로 판단 금지)
    ttest_rows = []
    for c in NUM_COLS:
        t, p = stats.ttest_ind(card[c], cash[c], equal_var=False)
        ttest_rows.append({"feature": c,
                           "카드평균": round(card[c].mean(), 4), "현금평균": round(cash[c].mean(), 4),
                           "t": round(t, 2), "p(Welch)": p,
                           "Cohen_d": round(cohens_d(card[c], cash[c]), 4)})
    t_ttest = pd.DataFrame(ttest_rows).sort_values("Cohen_d", key=abs, ascending=False)

    for fn, t in [("01_block_solo", t_solo), ("02_block_cumulative", t_cum),
                  ("03_final_compare", t_final), ("04_coefficients", t_coef),
                  ("05_feature_effect", t_effect), ("06_ttest_card_vs_cash", t_ttest)]:
        p = OUT / "tables" / f"{fn}.csv"
        t.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"  {p.relative_to(OUT.parent)}  ({len(t)}행)")

    print(f"\n저장 완료 → {OUT}")


# ──────────────────────────────────────────────────────────────────────
PARTS = {"1": part1, "2": part2, "3": part3, "4": part4, "5": part5}
ANALYSIS = {"1", "2", "3", "4"}      # 근거 재현. 화면 출력만, 파일 안 만듦
GENERATE = {"5"}                     # 산출물 생성


def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        want = GENERATE                       # 기본: 산출물만
    elif "all" in args:
        want = ANALYSIS | GENERATE
    elif "analysis" in args:
        want = ANALYSIS
    else:
        want = {a for a in args if a in PARTS}
        if not want:
            sys.exit(f"알 수 없는 인자: {args}\n{__doc__.split('실행')[1].split('PART 구성')[0]}")

    df = load()
    print(f"데이터: {DATA.name}  {df.shape[0]:,}행 × {df.shape[1]}열")
    for k, fn in PARTS.items():
        if k in want:
            fn(df)


if __name__ == "__main__":
    main()
