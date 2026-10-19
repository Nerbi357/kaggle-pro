# Признаки проекта 1. Каждая функция получает обучающую часть (с таргетом) и часть, для которой
# строим признаки, поэтому одинаково работает и внутри CV, и для финального train → test.
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import DATA, FEATURES, HOUSE_COLS, SEED, make_key

KREMLIN = (55.7520, 37.6175)


# --- 3.1 Инженерные признаки ---
def basic(df):
    out = pd.DataFrame(index=df.index)
    out["size_per_room"] = df.Size / df.Room
    out["floor_ratio"] = df.Floor / df.FloorsTotal
    out["is_first"] = (df.Floor == 1).astype(int)
    out["is_last"] = (df.Floor == df.FloorsTotal).astype(int)
    out["floors_bucket"] = pd.cut(df.FloorsTotal, [0, 5, 9, 12, 17, 25, 1000], labels=False)
    walls = df.Walls.fillna("NA")
    out["khrushchevka"] = (walls.isin(["панельный дом", "блочный", "кирпичный"]) & (df.FloorsTotal == 5)).astype(int)
    out["stalinka"] = ((walls == "кирпичный") & df.FloorsTotal.between(6, 10) & (out.size_per_room > 25)).astype(int)
    out["n_missing"] = df[["Balcony", "Walls", "Lift", "Age"]].isna().sum(axis=1)
    return out


# --- 3.2 Target encoding (out-of-fold, со сглаживанием) ---
TE_KEYS = {
    "te_district": ["District"],
    "te_district_age": ["District", "Age"],
    "te_district_walls": ["District", "Walls"],
    "te_district_floors": ["District", "FloorsTotal"],
    "te_house": HOUSE_COLS,
    "te_house_room": HOUSE_COLS + ["Room"],
}


def _te_apply(src, t_src, dst, cols, m):
    k_src, k_dst = make_key(src, cols), make_key(dst, cols)
    stats = pd.DataFrame({"k": k_src, "t": t_src}).groupby("k").t.agg(["sum", "count"])
    prior = t_src.mean()
    s = stats["sum"].reindex(k_dst).values
    c = stats["count"].reindex(k_dst).fillna(0).values
    return np.where(c > 0, (np.nan_to_num(s) + m * prior) / (c + m), prior), c


def target_encoding(train, t_train, apply, keys=TE_KEYS, m=5, inner_folds=5):
    """Для train — OOF по внутренним фолдам, для apply — по всему train. Плюс размеры групп."""
    t_train = pd.Series(np.asarray(t_train), index=train.index)
    tr_out = pd.DataFrame(index=train.index)
    ap_out = pd.DataFrame(index=apply.index)
    for name, cols in keys.items():
        ap_out[name], ap_out[name + "_n"] = _te_apply(train, t_train, apply, cols, m)
        tr_out[name], tr_out[name + "_n"] = np.nan, np.nan
        for a, b in KFold(inner_folds, shuffle=True, random_state=SEED).split(train):
            v, c = _te_apply(train.iloc[a], t_train.iloc[a], train.iloc[b], cols, m)
            tr_out.iloc[b, tr_out.columns.get_loc(name)] = v
            tr_out.iloc[b, tr_out.columns.get_loc(name + "_n")] = c
    return tr_out, ap_out


# --- 3.3 Соседи: точные двойники и соседи по порядку Id ---
def twins(train, t_train, apply, is_same):
    """Средний таргет точных двойников (все 10 признаков совпадают) и их число.
    is_same=True, если apply — это сам train: тогда строка не видит саму себя."""
    k_tr, k_ap = make_key(train, FEATURES), make_key(apply, FEATURES)
    g = pd.DataFrame({"k": k_tr, "t": np.asarray(t_train)}).groupby("k").t.agg(["sum", "count"])
    s = g["sum"].reindex(k_ap).fillna(0).values
    c = g["count"].reindex(k_ap).fillna(0).values
    if is_same:
        s, c = s - np.asarray(t_train), c - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({"twin_mean": np.where(c > 0, s / c, np.nan), "twin_n": c}, index=apply.index)


def align_positions(train_keys, apply_keys, radii=(3000, 800, 200), smooth=25):
    """Оценивает, между какими строками train стояла бы каждая строка apply в общем исходном списке.
    Порядок обеих выборок сохранён, поэтому позиция ≈ линейная + плавно меняющийся сдвиг.
    Сдвиг уточняем итеративно по ближайшим строкам train с тем же ключом псевдо-дома."""
    n, m = len(train_keys), len(apply_keys)
    p0 = np.arange(m) * n / m
    pos = {}
    for i, k in enumerate(train_keys):
        pos.setdefault(k, []).append(i)
    pos = {k: np.array(v) for k, v in pos.items()}
    off = np.zeros(m)
    for R in radii:
        diff = np.full(m, np.nan)
        for j, k in enumerate(apply_keys):
            arr = pos.get(k)
            if arr is None:
                continue
            c = p0[j] + off[j]
            lo, hi = np.searchsorted(arr, c - R), np.searchsorted(arr, c + R)
            if hi > lo:
                cand = arr[lo:hi]
                diff[j] = cand[np.argmin(np.abs(cand - c))] - p0[j]
        s = pd.Series(diff).rolling(2 * smooth + 1, center=True, min_periods=3).median()
        off = s.interpolate(limit_direction="both").fillna(0).values
    return p0 + off


def id_neighbors(train_keys, t_train, query_keys, query_pos, W, exclude=None):
    """Средний таргет строк train того же псевдо-дома в окне ±W позиций вокруг query_pos."""
    t_train = np.asarray(t_train)
    pos = {}
    for i, k in enumerate(train_keys):
        pos.setdefault(k, []).append(i)
    pos = {k: np.array(v) for k, v in pos.items()}
    mean, cnt = np.full(len(query_keys), np.nan), np.zeros(len(query_keys))
    for j, (k, p) in enumerate(zip(query_keys, query_pos)):
        arr = pos.get(k)
        if arr is None:
            continue
        sel = arr[np.searchsorted(arr, p - W):np.searchsorted(arr, p + W, side="right")]
        if exclude is not None:
            sel = sel[sel != exclude[j]]
        if len(sel):
            mean[j], cnt[j] = t_train[sel].mean(), len(sel)
    return mean, cnt


# --- 3.4 Внешние данные ---
def geo(df):
    c = pd.read_csv(DATA / "district_coords.csv").set_index("District")
    lat, lon = df.District.map(c.lat), df.District.map(c.lon)
    dy = (lat - KREMLIN[0]) * 111.2
    dx = (lon - KREMLIN[1]) * 111.2 * np.cos(np.radians(KREMLIN[0]))
    return pd.DataFrame({"lat": lat, "lon": lon, "dist_center": np.hypot(dx, dy),
                         "azimuth": np.degrees(np.arctan2(dx, dy))}, index=df.index)
