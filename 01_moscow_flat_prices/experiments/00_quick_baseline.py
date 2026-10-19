# Быстрый ориентир: LightGBM на сырых признаках и простое правило "цена м² района × площадь".
# Нужен, чтобы понять масштаб метрики лидерборда (0.17372 — это RMSE в логарифмах).
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import KFold

DATA = Path(__file__).resolve().parents[1] / "data"
tr = pd.read_csv(DATA / "train.csv")
y = np.log(tr.Price)
X = tr.drop(columns=["Id", "Price"])
for c in ["Balcony", "Walls", "Age", "Lift", "District", "Okrug"]:
    X[c] = X[c].astype("category")

kf = KFold(5, shuffle=True, random_state=0)

oof = np.zeros(len(X))
for a, b in kf.split(X):
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X.iloc[a], y.iloc[a])
    oof[b] = m.predict(X.iloc[b])
print("LGBM raw          RMSE", round(np.sqrt(((oof - y) ** 2).mean()), 4))  # ~0.179

lp = np.log(tr.Price / tr.Size)
oof2 = np.zeros(len(X))
for a, b in kf.split(X):
    g = lp.iloc[a].groupby(tr.District.iloc[a]).mean()
    oof2[b] = tr.District.iloc[b].map(g).fillna(lp.iloc[a].mean()) + np.log(tr.Size.iloc[b])
print("district ppm×size RMSE", round(np.sqrt(((oof2 - y) ** 2).mean()), 4))  # ~0.305
print("constant          RMSE", round(y.std(), 4))  # ~0.89
