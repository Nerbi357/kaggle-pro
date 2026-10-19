# Фаза 1. Данные и валидация: все цифры, на которых основаны решения шагов 1.1–1.5.
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (KFold, StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict)

from common import FEATURES, as_category, cv_oof, load, rmse

tr, te = load()
y = np.log(tr.Price)
lppm = y - np.log(tr.Size)
X = as_category(tr[FEATURES])
P = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def key(d, cols):
    return d[cols].astype("object").fillna("NA").astype(str).agg("|".join, axis=1).values


def lgbm(Xa, ya, Xb, seed=0):
    return lgb.LGBMRegressor(**P, random_state=seed).fit(Xa, ya).predict(Xb)


# 1.1 Таргет и аномалии
print("== 1.1 ==")
print("std log(Price)", round(y.std(), 3), "| std log(ppm)", round(lppm.std(), 3))
splits = list(KFold(5, shuffle=True, random_state=42).split(X))
oof_p, _ = cv_oof(lgbm, X, y, splits)
oof_m, _ = cv_oof(lgbm, X, lppm, splits)
print("таргет log(Price):", round(rmse(y, oof_p), 4), "| таргет log(ppm):", round(rmse(y, oof_m + np.log(tr.Size)), 4))
r = y - oof_p
print("доля суммарной ошибки от 1% худших строк:", round(np.sort(r.values ** 2)[::-1][:len(r) // 100].sum() / (r ** 2).sum(), 3))
med = lppm.groupby(tr.District).transform("median")
mad = (lppm - med).abs().groupby(tr.District).transform("median")
z = (lppm - med) / (1.4826 * mad)
print("robust z цены м² внутри района: |z|>4:", int((z.abs() > 4).sum()), "| |z|>6:", int((z.abs() > 6).sum()))
print("1-комнатные >100 м²: train", int(((tr.Room == 1) & (tr.Size > 100)).sum()),
      "test", int(((te.Room == 1) & (te.Size > 100)).sum()))

# 1.2 Повторные объявления, «псевдо-дом», порядок Id
print("== 1.2 ==")
g = pd.Series(key(tr, FEATURES))
vc = g.value_counts()
dup = g.map(vc) > 1
print("групп дублей", int((vc > 1).sum()), "| строк в них", int(dup.sum()),
      "| разброс log-цены внутри группы (max-min, медиана)",
      round(lppm[dup.values].groupby(g[dup].values).agg(lambda s: s.max() - s.min()).median(), 4))
print("доля строк с двойником: train", round(dup.mean(), 4),
      "| test (двойник в train)", round(pd.Series(key(te, FEATURES)).isin(set(g)).mean(), 4))

HOUSE = ["District", "FloorsTotal", "Walls", "Age"]
h = pd.Series(key(tr, HOUSE))
hc = h.map(h.value_counts())
big = (hc >= 5).values
w_h = lppm - lppm.groupby(h.values).transform("mean")
w_d = lppm - lppm.groupby(tr.District).transform("mean")
print("std log(ppm) для домов с ≥5 квартирами: внутри района", round(w_d[big].std(), 3),
      "| внутри псевдо-дома", round(w_h[big].std(), 3))

d_tr, d_te = tr.District.values, te.District.values
print("соседняя по Id строка из того же района: train", round((d_tr[1:] == d_tr[:-1]).mean(), 3),
      "| test", round((d_te[1:] == d_te[:-1]).mean(), 3),
      "| случайно", round((tr.District.value_counts(normalize=True) ** 2).sum(), 3))
# Совпадение порядка: относительная позиция test-строки и train-строк того же псевдо-дома
ktr, kte = key(tr, HOUSE), key(te, HOUSE)
pos = {}
for i, k in enumerate(ktr):
    pos.setdefault(k, []).append(i / len(tr))
diff = np.concatenate([np.array(pos[k]) - j / len(te) for j, k in enumerate(kte) if k in pos])
hist, edges = np.histogram(diff, bins=np.linspace(-0.05, 0.05, 21))
print("гистограмма (позиция в train − позиция в test):", dict(zip(edges[:-1].round(3), hist)))


def id_neighbors(src, tgt, W, exclude_self):
    """Средний log(ppm) строк src из того же псевдо-дома в окне ±W по Id."""
    by = {}
    for i in src:
        by.setdefault(ktr[i], []).append(i)
    by = {k: np.array(v) for k, v in by.items()}
    out, cnt = np.full(len(tgt), np.nan), np.zeros(len(tgt))
    for j, i in enumerate(tgt):
        arr = by.get(ktr[i])
        if arr is None:
            continue
        sel = arr[np.searchsorted(arr, i - W):np.searchsorted(arr, i + W, side="right")]
        if exclude_self:
            sel = sel[sel != i]
        if len(sel):
            out[j], cnt[j] = lppm.values[sel].mean(), len(sel)
    return out, cnt


for W in [20, 100, 500]:
    oof = np.zeros(len(tr))
    for a, b in splits:
        Xa, Xb = X.iloc[a].copy(), X.iloc[b].copy()
        Xa["nb"], Xa["nbc"] = id_neighbors(a, a, W, True)
        Xb["nb"], Xb["nbc"] = id_neighbors(a, b, W, False)
        oof[b] = lgbm(Xa, y.iloc[a], Xb)
    print(f"признак «соседи по Id», окно ±{W}:", round(rmse(y, oof), 4), "(без него", round(rmse(y, oof_p), 4), ")")

# 1.3 Пропуски и смысл признаков
print("== 1.3 ==")
for c in ["Balcony", "Walls", "Lift", "Age"]:
    m = tr[c].isna()
    print(f"{c}: пропусков {m.mean():.3f}, log(ppm) у пропусков относительно района {w_d[m].mean() - w_d[~m].mean():+.3f}")
print("District пропущен ⇔ Okrug пропущен:", int((tr.District.isna() & tr.Okrug.isna()).sum()), "/", int(tr.District.isna().sum()))
print(pd.crosstab(pd.cut(tr.FloorsTotal, [0, 5, 9, 16, 25, 200]), tr.Lift.fillna("NA"), normalize="index").round(2))
print("log(ppm) относительно района по Room:", w_d.groupby(tr.Room).mean().round(3).to_dict())

# 1.4 Train против test
print("== 1.4 ==")
A = as_category(pd.concat([tr[FEATURES], te[FEATURES]], ignore_index=True))
lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1), A, lab,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print("adversarial AUC:", round(roc_auc_score(lab, p), 4))

# 1.5 Схемы валидации и шум
print("== 1.5 ==")
gid = pd.factorize(g)[0]
strat = pd.qcut(y, 10, labels=False)
rows = []
for s in [0, 1, 2]:
    for name, sp in [("KFold", KFold(5, shuffle=True, random_state=s).split(X)),
                     ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=s).split(X, strat)),
                     ("StratifiedGroupKFold(дубли)", StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strat, gid))]:
        oof, sc = cv_oof(lgbm, X, y, sp)
        rows.append((name, s, rmse(y, oof), sc.std()))
r = pd.DataFrame(rows, columns=["схема", "seed", "rmse", "std_фолдов"])
print(r.groupby("схема").agg(rmse=("rmse", "mean"), std_сидов=("rmse", "std"), std_фолдов=("std_фолдов", "mean")).round(4))
