"""Shared data access, label construction, candidate pool and MAP@5."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RES = ROOT / "results"
K = 5
WATCHED = 0.8  # screen-time threshold, identical to the competition's label recipe

SPLITS = {
    # name: (train weeks, target weeks)
    "val": ((11, 20), (21, 30)),
    "test": ((11, 30), (31, 42)),
    # re-ranker fitting fold for validation: strictly before the validation target weeks
    "fit_val": ((11, 15), (16, 20)),
}


def load():
    ua = pd.read_parquet(CACHE / "user_airing.parquet")
    air = pd.read_parquet(CACHE / "airings.parquet")
    users = pd.read_parquet(CACHE / "users.parquet")["user_id"].values
    uc = pd.read_parquet(CACHE / "user_channel.parquet")
    items = np.sort(air["tv_show_id"].unique())
    ua["i"] = np.searchsorted(items, ua["tv_show_id"].values).astype(np.int32)
    air["i"] = np.searchsorted(items, air["tv_show_id"].values).astype(np.int32)
    return ua, air, users, items, uc


def in_weeks(df, weeks):
    return df[(df["week"] >= weeks[0]) & (df["week"] <= weeks[1])]


def build_labels(ua, weeks):
    """Competition recipe: per user, count airings with screen >= 0.8 per programme,
    keep users with >= 5 such programmes, take the 5 most frequent.
    Ties (unspecified by the host) are broken by total watched seconds, then item index."""
    w = in_weeks(ua, weeks)
    w = w[w["screen"] >= WATCHED]
    g = w.groupby(["u", "i"], as_index=False).agg(cnt=("screen", "size"), sec=("overlap", "sum"))
    g = g.sort_values(["u", "cnt", "sec", "i"], ascending=[True, False, False, True])
    g["rank"] = g.groupby("u").cumcount()
    n = g.groupby("u")["i"].transform("size")
    g = g[(n >= K) & (g["rank"] < K)]
    return {u: grp["i"].values for u, grp in g.groupby("u")}


@dataclass
class TrainData:
    n_users: int
    n_items: int
    weeks: tuple
    target_weeks: tuple
    cnt: sp.csr_matrix        # (u, i) number of airings watched with screen >= 0.8 in train weeks
    cnt_decay: sp.csr_matrix  # same, each airing weighted by recency
    sec: sp.csr_matrix        # (u, i) overlapping seconds, any screen time
    pool: np.ndarray          # item indices airing (labelled) in the target weeks
    pool_airings: np.ndarray  # number of target-week airings per item (len n_items, 0 outside pool)
    train_airings: np.ndarray  # number of train-week airings per item
    item_channel_share: sp.csr_matrix  # (i, channel) share of target-week airings on that channel
    user_channel: sp.csr_matrix  # (u, channel) viewing seconds in train weeks, all airtime


def make_train(ua, air, uc, n_users, n_items, split, half_life=4.0):
    weeks, target = SPLITS[split]
    tr = in_weeks(ua, weeks)
    hit = tr[tr["screen"] >= WATCHED]
    decay = 0.5 ** ((weeks[1] - hit["week"].values) / half_life)
    shape = (n_users, n_items)
    cnt = sp.csr_matrix((np.ones(len(hit), np.float32), (hit["u"], hit["i"])), shape=shape)
    cnt_decay = sp.csr_matrix((decay.astype(np.float32), (hit["u"], hit["i"])), shape=shape)
    sec = sp.csr_matrix((tr["overlap"].values.astype(np.float32), (tr["u"], tr["i"])), shape=shape)
    for m in (cnt, cnt_decay, sec):
        m.sum_duplicates()

    tgt_air = in_weeks(air, target)  # schedule only: which programme airs when (allowed)
    pool_airings = np.bincount(tgt_air["i"], minlength=n_items).astype(np.float32)
    train_airings = np.bincount(in_weeks(air, weeks)["i"], minlength=n_items).astype(np.float32)
    pool = np.flatnonzero(pool_airings > 0)

    n_ch = int(max(air["channel_id"].max(), uc["channel_id"].max())) + 1
    ic = tgt_air.groupby(["i", "channel_id"]).size().reset_index(name="n")
    ic["share"] = ic["n"] / ic.groupby("i")["n"].transform("sum")
    item_channel_share = sp.csr_matrix((ic["share"], (ic["i"], ic["channel_id"])), shape=(n_items, n_ch))
    ucw = in_weeks(uc, weeks)
    user_channel = sp.csr_matrix(
        (ucw["duraton"].values.astype(np.float32), (ucw["u"], ucw["channel_id"])), shape=(n_users, n_ch))
    user_channel.sum_duplicates()
    return TrainData(n_users, n_items, weeks, target, cnt, cnt_decay, sec, pool, pool_airings,
                     train_airings, item_channel_share, user_channel)


def popularity_scores(td: TrainData):
    """Recency-weighted number of airings watched (>= 0.8) by all users in the train weeks,
    among programmes that air in the target weeks. ((cnt_decay > 0) * cnt_decay == cnt_decay,
    so this is the column sum of decayed counts, not a count of distinct users.)"""
    users_per_item = np.asarray((td.cnt_decay > 0).multiply(td.cnt_decay).sum(axis=0)).ravel()
    s = np.full(td.n_items, -np.inf, np.float32)
    s[td.pool] = users_per_item[td.pool]
    return s


def top_k(scores: np.ndarray, fallback: np.ndarray, k=K):
    """scores: (n_users, n_items) with -inf outside the pool.
    Returns (n_users, k) item indices; slots without a finite positive score are filled
    from the popularity ranking (fallback, already ordered), skipping duplicates."""
    n = scores.shape[0]
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    out = np.empty((n, k), np.int64)
    filled = 0
    for r in range(n):
        good = idx[r][np.isfinite(part[r]) & (part[r] > 0)].tolist()
        if len(good) < k:
            filled += 1
            for f in fallback:
                if len(good) == k:
                    break
                if f not in good:
                    good.append(int(f))
        out[r] = good
    return out, filled


def apk(pred, truth, k=K):
    truth = set(truth.tolist() if hasattr(truth, "tolist") else truth)
    hits, s = 0, 0.0
    for r, p in enumerate(pred[:k], start=1):
        if p in truth:
            hits += 1
            s += hits / r
    return s / min(k, len(truth)) if truth else 0.0


def map_at_k(preds: dict, labels: dict, k=K):
    return float(np.mean([apk(preds[u], labels[u], k) for u in labels]))
