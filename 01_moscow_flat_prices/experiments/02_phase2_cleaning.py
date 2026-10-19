# Фаза 2. Быстрые проверки для решений по очистке: таргет, выбросы, дубли, кодирование категорий.
# Все варианты — LightGBM с параметрами baseline на одних и тех же фолдах (KFold(5), seed 42).
# Ошибка всегда считается по ВСЕМ строкам валидации, даже если из обучения что-то удалено.
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import BASE_PARAMS, CAT_COLS, FEATURES, as_category, load, make_key, rmse, splits, target_ppm

tr, te = load()
y = np.log(tr.Price)
t = target_ppm(tr)
ls = np.log(tr.Size)
X = as_category(tr[FEATURES])
folds = splits(len(tr))
dup_key = make_key(tr, FEATURES)


def run(name, prep=None, params=None, target=t, Xin=X):
    """prep(train_idx) -> (индексы обучения, веса). Возвращает RMSE по log(Price)."""
    oof = np.zeros(len(tr))
    for a, b in folds:
        idx, w = prep(a) if prep else (a, None)
        m = lgb.LGBMRegressor(**{**BASE_PARAMS, **(params or {})})
        m.fit(Xin.iloc[idx], target.iloc[idx], sample_weight=w)
        oof[b] = m.predict(Xin.iloc[b])
    if target is t:
        oof = oof + ls
    score = rmse(y, oof)
    print(f"{name:55s} {score:.4f}")
    sys.stdout.flush()
    return score


def robust_z(idx):
    """robust z цены м² внутри района, считается только по строкам обучения."""
    s, d = t.iloc[idx], tr.District.iloc[idx].fillna("NA")
    med = s.groupby(d).transform("median")
    mad = (s - med).abs().groupby(d).transform("median")
    return ((s - med) / (1.4826 * mad + 1e-9)).values


print("== 2.1 Таргет ==")
run("log(Price)", target=y)
run("log(Price/Size)")

print("== 2.2 Выбросы в train ==")
run("оставить всё")
for thr in [6, 4, 3]:
    run(f"удалить из обучения |z|>{thr}", lambda a, thr=thr: (a[np.abs(robust_z(a)) <= thr], None))
run("вес 0.3 для |z|>4", lambda a: (a, np.where(np.abs(robust_z(a)) > 4, 0.3, 1.0)))
for alpha in [0.2, 0.4]:
    run(f"Huber-лосс, порог {alpha}", params=dict(objective="huber", alpha=alpha))

print("== 2.3 Дубли в train ==")
run("оставить")
cnt = pd.Series(dup_key).map(pd.Series(dup_key).value_counts()).values
run("вес 1/(размер группы)", lambda a: (a, 1.0 / cnt[a]))
run("оставить одну строку из группы", lambda a: (a[~pd.Series(dup_key[a]).duplicated().values], None))

print("== 2.4 Кодирование категорий ==")
run("нативные категории LightGBM")
Xo = X.copy()
for c in CAT_COLS:
    Xo[c] = Xo[c].cat.codes
run("порядковые коды (label encoding)", Xin=Xo)
Xd = pd.get_dummies(tr[FEATURES], columns=CAT_COLS, dummy_na=True, dtype=float)
run("one-hot", Xin=Xd)
run("нативные + сглаживание (cat_smooth=50, min_data_per_group=50)", params=dict(cat_smooth=50, min_data_per_group=50))
