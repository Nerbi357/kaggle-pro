# Фаза 3, итоговый набор признаков: вклад каждой группы («минус одна группа») на двух seed разбиения.
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import BASE_PARAMS, FEATURES, as_category, load, rmse, splits, target_ppm
from features import basic, geo, twins

tr, te = load()
y, t, ls = np.log(tr.Price), target_ppm(tr), np.log(tr.Size)


def run(groups, seed):
    oof = np.zeros(len(tr))
    for a, b in splits(len(tr), seed):
        A, B = tr.iloc[a], tr.iloc[b]
        pa, pb = [as_category(A[FEATURES])], [as_category(B[FEATURES])]
        if "basic" in groups:
            pa.append(basic(A)); pb.append(basic(B))
        if "twins" in groups:
            pa.append(twins(A, t.iloc[a], A, True)); pb.append(twins(A, t.iloc[a], B, False))
        if "geo" in groups:
            pa.append(geo(A)); pb.append(geo(B))
        oof[b] = lgb.LGBMRegressor(**BASE_PARAMS).fit(pd.concat(pa, axis=1), t.iloc[a]).predict(pd.concat(pb, axis=1))
    return rmse(y, oof + ls)


ALL = ["basic", "twins", "geo"]
for seed in [42, 7]:
    print(f"seed {seed}: сырые {run([], seed):.4f} | все {run(ALL, seed):.4f} | " +
          " | ".join(f"без {g} {run([x for x in ALL if x != g], seed):.4f}" for g in ALL), flush=True)
