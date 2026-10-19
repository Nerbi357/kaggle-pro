# Фаза 3. Быстрые проверки групп признаков: накопительно и «минус одна группа».
# LightGBM baseline, таргет log(Price/Size), фиксированные фолды KFold(5, seed=42).
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import BASE_PARAMS, FEATURES, as_category, load, rmse, splits, target_ppm
from features import basic, geo, target_encoding, twins

tr, te = load()
y = np.log(tr.Price)
t = target_ppm(tr)
ls = np.log(tr.Size)
folds = splits(len(tr))
def build(a, b, groups, te_m=5):
    A, B = tr.iloc[a], tr.iloc[b]
    parts_a, parts_b = [as_category(A[FEATURES]).reset_index(drop=True)], [as_category(B[FEATURES]).reset_index(drop=True)]
    if "basic" in groups:
        parts_a.append(basic(A).reset_index(drop=True)); parts_b.append(basic(B).reset_index(drop=True))
    if "te" in groups:
        ta, tb = target_encoding(A, t.iloc[a], B, m=te_m)
        parts_a.append(ta.reset_index(drop=True)); parts_b.append(tb.reset_index(drop=True))
    if "twins" in groups:
        parts_a.append(twins(A, t.iloc[a], A, True).reset_index(drop=True))
        parts_b.append(twins(A, t.iloc[a], B, False).reset_index(drop=True))
    if "geo" in groups:
        parts_a.append(geo(A).reset_index(drop=True)); parts_b.append(geo(B).reset_index(drop=True))
    return pd.concat(parts_a, axis=1), pd.concat(parts_b, axis=1)


def run(name, groups, **kw):
    oof = np.zeros(len(tr))
    for a, b in folds:
        Xa, Xb = build(a, b, groups, **kw)
        m = lgb.LGBMRegressor(**BASE_PARAMS).fit(Xa, t.iloc[a])
        oof[b] = m.predict(Xb)
    s = rmse(y, oof + ls)
    print(f"{name:50s} {s:.4f}")
    sys.stdout.flush()
    return s


if __name__ == "__main__":
    print("== Накопительно ==")
    run("сырые признаки", [])
    run("+ 3.1 инженерные", ["basic"])
    run("+ 3.2 target encoding (сглаживание m=5)", ["basic", "te"])
    run("  (то же, m=20)", ["basic", "te"], te_m=20)
    run("+ 3.3 двойники", ["basic", "te", "twins"])
    ALL = ["basic", "te", "twins", "geo"]
    run("+ 3.4 координаты районов", ALL)
    print("== 3.5 Минус одна группа (от полного набора) ==")
    for g in ALL:
        run(f"без {g}", [x for x in ALL if x != g])
