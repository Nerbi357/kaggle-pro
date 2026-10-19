# Фаза 3. Быстрые проверки групп признаков: накопительно и «минус одна группа».
# LightGBM baseline, таргет log(Price/Size), фиксированные фолды KFold(5, seed=42).
# Строки валидации обрабатываются так же, как будет обработан test: их позиции по Id скрыты
# и восстанавливаются выравниванием (align_positions).
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import (BASE_PARAMS, FEATURES, HOUSE_COLS, SEED, as_category, load, make_key, rmse,
                    splits, target_ppm)
from features import align_positions, basic, geo, id_neighbors, target_encoding, twins

RADII, SMOOTH = (3000, 800, 200, 60, 20, 10), 25

tr, te = load()
y = np.log(tr.Price)
t = target_ppm(tr)
ls = np.log(tr.Size)
folds = splits(len(tr))
HK = make_key(tr, HOUSE_COLS)


def id_feats(a, b, W, train_mode):
    """Признак соседей по Id для строк обучения a и валидации b (индексы отсортированы по Id)."""
    ta = t.values[a]
    # валидация: позиции неизвестны, выравниваем по порядку
    pos_b = align_positions(HK[a], HK[b], RADII, SMOOTH)
    vb = id_neighbors(HK[a], ta, HK[b], pos_b, W)
    if train_mode == "exact":
        own = np.arange(len(a))
        va = id_neighbors(HK[a], ta, HK[a], own, W, exclude=own)
    else:  # "honest": внутри обучения тоже скрываем позиции по внутренним фолдам
        va = (np.full(len(a), np.nan), np.zeros(len(a)))
        for ia, ib in KFold(5, shuffle=True, random_state=SEED).split(a):
            p = align_positions(HK[a][ia], HK[a][ib], RADII, SMOOTH)
            m, c = id_neighbors(HK[a][ia], ta[ia], HK[a][ib], p, W)
            va[0][ib], va[1][ib] = m, c
    return (pd.DataFrame({"idnb_mean": va[0], "idnb_n": va[1]}),
            pd.DataFrame({"idnb_mean": vb[0], "idnb_n": vb[1]}))


def build(a, b, groups, W=30, id_mode="honest"):
    A, B = tr.iloc[a], tr.iloc[b]
    parts_a, parts_b = [as_category(A[FEATURES]).reset_index(drop=True)], [as_category(B[FEATURES]).reset_index(drop=True)]
    if "basic" in groups:
        parts_a.append(basic(A).reset_index(drop=True)); parts_b.append(basic(B).reset_index(drop=True))
    if "te" in groups:
        ta, tb = target_encoding(A, t.iloc[a], B)
        parts_a.append(ta.reset_index(drop=True)); parts_b.append(tb.reset_index(drop=True))
    if "twins" in groups:
        parts_a.append(twins(A, t.iloc[a], A, True).reset_index(drop=True))
        parts_b.append(twins(A, t.iloc[a], B, False).reset_index(drop=True))
    if "id" in groups:
        fa, fb = id_feats(a, b, W, id_mode)
        parts_a.append(fa); parts_b.append(fb)
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
    run("+ 3.2 target encoding", ["basic", "te"])
    run("+ 3.3а двойники", ["basic", "te", "twins"])
    for W in [10, 30, 100]:
        run(f"+ 3.3б соседи по Id, окно ±{W}", ["basic", "te", "twins", "id"], W=W)
    run("  (то же ±30, train по точным позициям)", ["basic", "te", "twins", "id"], W=30, id_mode="exact")
    ALL = ["basic", "te", "twins", "id", "geo"]
    run("+ 3.4 координаты районов", ALL)
    print("== 3.5 Минус одна группа (от полного набора) ==")
    for g in ALL:
        run(f"без {g}", [x for x in ALL if x != g])
