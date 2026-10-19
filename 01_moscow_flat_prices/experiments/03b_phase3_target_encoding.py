# Фаза 3, разбор target encoding: какие кодировки полезны поверх basic + twins + geo.
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import BASE_PARAMS, FEATURES, HOUSE_COLS, as_category, load, make_key, rmse, splits, target_ppm
from features import TE_KEYS, basic, geo, target_encoding, twins

tr, te = load()
y, t, ls = np.log(tr.Price), target_ppm(tr), np.log(tr.Size)
folds = splits(len(tr))


def run(name, keys=None, inner=5, m=5, counts_only=False):
    oof = np.zeros(len(tr))
    for a, b in folds:
        A, B = tr.iloc[a], tr.iloc[b]
        pa = [as_category(A[FEATURES]), basic(A), twins(A, t.iloc[a], A, True), geo(A)]
        pb = [as_category(B[FEATURES]), basic(B), twins(A, t.iloc[a], B, False), geo(B)]
        if keys:
            ta, tb = target_encoding(A, t.iloc[a], B, keys=keys, m=m, inner_folds=inner)
            if counts_only:
                ta, tb = ta.filter(like="_n"), tb.filter(like="_n")
            pa.append(ta); pb.append(tb)
        Xa, Xb = pd.concat(pa, axis=1), pd.concat(pb, axis=1)
        oof[b] = lgb.LGBMRegressor(**BASE_PARAMS).fit(Xa, t.iloc[a]).predict(Xb)
    print(f"{name:55s} {rmse(y, oof + ls):.4f}")
    sys.stdout.flush()


run("basic + twins + geo (без TE)")
for k in TE_KEYS:
    run(f"+ только {k}", {k: TE_KEYS[k]})
H = {"te_house": HOUSE_COLS}
run("+ te_house, внутренних фолдов 10", H, inner=10)
run("+ te_house, сглаживание m=20", H, m=20)
run("+ только размер псевдо-дома в train (без таргета)", H, counts_only=True)
