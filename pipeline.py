import inspect
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


class GeoEncoder(BaseEstimator, TransformerMixin):
    """config.LOCATION_COLS를 fold의 train 쪽에서만 클래스별 비율로 인코딩해 붙인다.

    타깃이 2클래스든 3클래스(이상)든 그대로 동작한다 — 클래스마다 "이 지역/경로에서
    이 클래스일 비율"을 따로 인코딩하기 때문에(첫 클래스는 기준값이라 생략, 나머지
    클래스 수만큼 컬럼 생성). 단일 평균으로 인코딩하면 명목형 다중클래스에서는
    (예: 0=Flex Fare/1=카드/2=현금의 평균 같은) 의미 없는 숫자가 나와서 이렇게 짰다.
    LOCATION_COLS가 데이터에 없으면 그냥 통과시킨다.
    """

    def __init__(self, m: int = 100):
        self.m = m

    def fit(self, X, y):
        self._active = set(config.LOCATION_COLS).issubset(X.columns)
        if not self._active:
            return self

        y = pd.Series(np.asarray(y), index=X.index)
        self._classes = sorted(y.unique())[1:]  # 첫 클래스는 기준값이라 인코딩에서 뺀다
        self._priors = {c: float((y == c).mean()) for c in self._classes}

        pu_col, do_col = config.LOCATION_COLS
        route = X[pu_col].astype(str) + "_" + X[do_col].astype(str)
        keys = {"pu": X[pu_col], "do": X[do_col], "route": route}

        self._maps = {}
        for key_name, col in keys.items():
            for c in self._classes:
                yc = (y == c).astype(int)
                g = pd.DataFrame({"k": col.values, "y": yc.values}).groupby("k")["y"].agg(["mean", "size"])
                prior = self._priors[c]
                self._maps[(key_name, c)] = (g["mean"] * g["size"] + prior * self.m) / (g["size"] + self.m)
        return self

    def transform(self, X):
        if not self._active:
            return X

        pu_col, do_col = config.LOCATION_COLS
        route = X[pu_col].astype(str) + "_" + X[do_col].astype(str)
        keys = {"pu": X[pu_col], "do": X[do_col], "route": route}

        out = X.drop(columns=[pu_col, do_col])
        for key_name, col in keys.items():
            for c in self._classes:
                m = self._maps[(key_name, c)]
                out[f"{key_name}_te_{c}"] = col.map(m).fillna(self._priors[c]).to_numpy()
        return out


def load_data():
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"config.py의 DATA_PATH에 적은 '{config.DATA_PATH}' 파일이 없습니다. "
            f"DATA_PATH를 실제 파일 경로로 고치세요."
        )
    if str(config.DATA_PATH).endswith(".parquet"):
        df = pd.read_parquet(config.DATA_PATH)
    else:
        df = pd.read_csv(config.DATA_PATH)

    if config.TARGET not in df.columns:
        raise ValueError(
            f"config.py의 TARGET에 적은 '{config.TARGET}'가 데이터에 없습니다. "
            f"컬럼: {list(df.columns)}"
        )
    n_rows, n_cols = df.shape
    n_features = n_cols - 1 - len(config.DROP_COLS)
    print(f"[load_data] 파일: {config.DATA_PATH}")
    print(f"[load_data] 행 수: {n_rows}, 피처 수: {n_features}")

    counts = df[config.TARGET].value_counts()
    ratios = df[config.TARGET].value_counts(normalize=True)
    print("[load_data] 클래스별 개수/비율:")
    for cls in counts.index:
        name = config.CLASS_LABELS.get(cls, cls)
        print(f"  {name} ({cls}): {counts[cls]}건 ({ratios[cls] * 100:.2f}%)")

    na_counts = df.isnull().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts) > 0:
        print("[load_data] 경고: 결측치가 있는 컬럼")
        for col, n in na_counts.items():
            print(f"  {col}: {n}건")

    return df


def split(df):
    X = df.drop(columns=[config.TARGET, *config.DROP_COLS])
    y = df[config.TARGET]

    if config.TARGET_ENCODING and not pd.api.types.is_numeric_dtype(y):
        y = y.map(config.TARGET_ENCODING)
        if y.isnull().any():
            raise ValueError(
                "config.py의 TARGET_ENCODING에 없는 값이 TARGET 컬럼에 있습니다. "
                f"매핑: {config.TARGET_ENCODING}, 실제 값: {list(df[config.TARGET].unique())}"
            )
        y = y.astype(int)

    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=y,
    )


def _class_weight_kwarg(estimator_cls, class_weight):
    """estimator_cls가 class_weight를 지원하면 그 kwarg를, 아니면 경고만 내고 빈 dict를 돌려준다."""
    if class_weight is None:
        return {}
    if "class_weight" in inspect.signature(estimator_cls.__init__).parameters:
        return {"class_weight": class_weight}
    warnings.warn(
        f"{estimator_cls.__name__}는 class_weight를 지원하지 않습니다. "
        f"USE_CLASS_WEIGHT 설정을 이 모델에는 적용하지 못했습니다."
    )
    return {}


def _make_classifier(model_name, params):
    class_weight = "balanced" if config.USE_CLASS_WEIGHT else None

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=config.SEED,
            n_jobs=config.N_JOBS,
            **_class_weight_kwarg(RandomForestClassifier, class_weight),
        )

    if model_name == "hist_gbm":
        return HistGradientBoostingClassifier(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=config.SEED,
            **_class_weight_kwarg(HistGradientBoostingClassifier, class_weight),
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "config.py의 MODELS에 'xgboost'가 있지만 설치되어 있지 않습니다. "
                "'pip install xgboost'로 설치하세요. (macOS는 'brew install libomp'도 필요)"
            )
        # XGBClassifier는 class_weight를 지원하지 않음 -> sample_weight로 대체
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=config.SEED,
            n_jobs=config.N_JOBS,
        )

    if model_name == "logistic":
        return LogisticRegression(
            C=params["C"],
            random_state=config.SEED,
            max_iter=1000,
            n_jobs=config.N_JOBS,
            **_class_weight_kwarg(LogisticRegression, class_weight),
        )

    raise ValueError(
        f"알 수 없는 모델 이름입니다: '{model_name}'. "
        f"config.py의 MODELS에는 random_forest/hist_gbm/xgboost/logistic만 적을 수 있습니다."
    )


def _sample_weight_kwargs(model_name, y):
    """class_weight를 지원하지 않는 모델(xgboost)을 위해 sample_weight를 fit에 전달한다."""
    if model_name == "xgboost" and config.USE_CLASS_WEIGHT:
        from sklearn.utils.class_weight import compute_sample_weight
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def build_pipeline(model_name, params):
    scaler = StandardScaler() if config.SCALE else "passthrough"
    clf = _make_classifier(model_name, params)
    return Pipeline([
        ("geo_encode", GeoEncoder()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
        ("model", clf),
    ])


def tune(X_train, y_train):
    """Optuna로 모델·하이퍼파라미터를 탐색. train만 사용하고 test는 인자로도 받지 않는다."""

    def cv_macro_f1(model_name, params):
        skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            pipe = build_pipeline(model_name, params)
            pipe.fit(X_tr, y_tr, **_sample_weight_kwargs(model_name, y_tr))
            pred = pipe.predict(X_val)
            scores.append(f1_score(y_val, pred, average="macro"))
        return float(np.mean(scores))

    def objective(trial):
        model_name = trial.suggest_categorical("model_name", config.MODELS)
        space = config.SEARCH_SPACE[model_name]
        params = {}
        for name, spec in space.items():
            if len(spec) == 3:
                lo, hi, scale = spec
                log = scale == "log"
            else:
                lo, hi = spec
                log = False
            if isinstance(lo, int) and isinstance(hi, int):
                params[name] = trial.suggest_int(name, lo, hi, log=log)
            else:
                params[name] = trial.suggest_float(name, lo, hi, log=log)
        score = cv_macro_f1(model_name, params)
        print(f"[tune] trial {trial.number}: model={model_name} params={params} -> macro_f1={score:.4f}")
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(objective, n_trials=config.N_TRIALS)

    best = study.best_trial
    best_model_name = best.params["model_name"]
    best_params = {k: v for k, v in best.params.items() if k != "model_name"}
    print(f"[tune] 최적 모델: {best_model_name}, 파라미터: {best_params}, CV macro F1: {best.value:.4f}")
    return best_model_name, best_params, best.value


def fit_final(model_name, params, X_train, y_train):
    pipeline = build_pipeline(model_name, params)
    pipeline.fit(X_train, y_train, **_sample_weight_kwargs(model_name, y_train))
    return pipeline


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    labels = sorted(pd.unique(y_test))
    target_names = [str(config.CLASS_LABELS.get(label, label)) for label in labels]

    print(f"[evaluate] accuracy: {acc:.4f} (참고용)")
    print(f"[evaluate] macro F1: {macro_f1:.4f} (주 지표)")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
    print(f"[evaluate] confusion matrix (행=실제, 열=예측), 클래스 순서: {target_names}")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    return acc, macro_f1


def save(pipeline, model_name, best_params, n_rows, n_features, cv_macro_f1, test_acc, test_macro_f1):
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"outputs/model_{model_name}_{timestamp}.joblib"
    joblib.dump(pipeline, model_path)

    results_path = "outputs/results.csv"
    row = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_file": config.DATA_PATH,
        "n_rows": n_rows,
        "n_features": n_features,
        "best_model": model_name,
        "best_params": best_params,
        "n_trials": config.N_TRIALS,
        "cv_folds": config.CV_FOLDS,
        "seed": config.SEED,
        "test_size": config.TEST_SIZE,
        "class_weight": config.USE_CLASS_WEIGHT,
        "scale": config.SCALE,
        "cv_macro_f1": cv_macro_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "model_path": model_path,
    }
    write_header = not os.path.exists(results_path)
    pd.DataFrame([row]).to_csv(results_path, mode="a", header=write_header, index=False)

    print(f"[save] 모델 저장: {model_path}")
    print(f"[save] 결과 기록: {results_path}")
    return model_path


if __name__ == "__main__":
    print("[1/6] 데이터 불러오는 중...")
    df = load_data()

    print("[2/6] train/test 분리 중...")
    X_train, X_test, y_train, y_test = split(df)

    if config.TUNE_SAMPLE_SIZE and len(X_train) > config.TUNE_SAMPLE_SIZE:
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train,
            train_size=config.TUNE_SAMPLE_SIZE,
            random_state=config.SEED,
            stratify=y_train,
        )
        print(f"[2/6] tune용 표본 추출: {len(X_train)}행 -> {len(X_tune)}행 (최종 재학습은 전체 사용)")
    else:
        X_tune, y_tune = X_train, y_train

    print("[3/6] Optuna로 모델/하이퍼파라미터 탐색 중 (train 표본만 사용)...")
    best_model_name, best_params, cv_macro_f1 = tune(X_tune, y_tune)

    print("[4/6] 최적 설정으로 최종 모델 재학습 중...")
    final_pipeline = fit_final(best_model_name, best_params, X_train, y_train)

    print("[5/6] test 데이터로 평가 중...")
    test_acc, test_macro_f1 = evaluate(final_pipeline, X_test, y_test)

    print("[6/6] 결과 저장 중...")
    model_path = save(
        final_pipeline, best_model_name, best_params,
        n_rows=len(df), n_features=X_train.shape[1],
        cv_macro_f1=cv_macro_f1, test_acc=test_acc, test_macro_f1=test_macro_f1,
    )

    print(f"완료. 저장된 모델: {model_path}, 결과 기록: outputs/results.csv")
