# Общие функции для экспериментов проекта 1: загрузка данных, кросс-валидация, метрика.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "data"
CAT_COLS = ["Balcony", "Walls", "Age", "Lift", "District", "Okrug"]
FEATURES = ["Room", "Size", "Balcony", "Floor", "FloorsTotal", "Walls", "Age", "Lift", "District", "Okrug"]


def load():
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    return tr, te


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def as_category(X, cols=CAT_COLS):
    X = X.copy()
    for c in cols:
        X[c] = X[c].astype("category")
    return X


def cv_oof(fit_predict, X, y, splits=None, groups=None):
    """fit_predict(X_tr, y_tr, X_va) -> прогноз. Возвращает OOF-прогноз и RMSE по фолдам."""
    if splits is None:
        splits = KFold(5, shuffle=True, random_state=SEED).split(X, y, groups)
    oof = np.zeros(len(X))
    scores = []
    for a, b in splits:
        oof[b] = fit_predict(X.iloc[a], y.iloc[a], X.iloc[b])
        scores.append(rmse(y.iloc[b], oof[b]))
    return oof, np.array(scores)


# --- Решения фазы 1 ---
HOUSE_COLS = ["District", "FloorsTotal", "Walls", "Age"]  # «псевдо-дом»
BASE_PARAMS = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, verbose=-1)  # baseline LightGBM


def make_key(d, cols):
    """Строковый ключ по набору столбцов, пропуски — отдельное значение NA."""
    part = lambda c: d[c].astype("object").fillna("NA").astype(str)
    k = part(cols[0])
    for c in cols[1:]:
        k = k + "|" + part(c)
    return k.to_numpy(dtype=object)


def splits(n_rows, seed=SEED):
    """Фиксированные фолды: KFold(5) со своим seed. Все варианты сравниваем на одних и тех же фолдах."""
    return list(KFold(5, shuffle=True, random_state=seed).split(np.zeros(n_rows)))


def target_ppm(df):
    """Рабочий таргет: log(цена за м²). Прогноз цены = exp(прогноз + log(Size))."""
    return np.log(df.Price / df.Size)
