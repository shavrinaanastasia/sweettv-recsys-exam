"""Recommendation methods. Each returns a dense (len(eval_users), n_items) score matrix
built from TrainData only; masking to the candidate pool happens in mask_pool()."""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm

from common import TrainData


def mask_pool(S, td: TrainData):
    out = np.full(S.shape, -np.inf, np.float32)
    out[:, td.pool] = S[:, td.pool]
    return out


# ---------------------------------------------------------------- reference: personal history
def history(td: TrainData, eval_users, decay=True):
    M = td.cnt_decay if decay else td.cnt
    return M[eval_users].toarray()


# ---------------------------------------------------------------- M1: implicit ALS
def als_fit(R: sp.csr_matrix, factors=64, reg=0.1, alpha=10.0, iters=15, seed=42):
    """Implicit-feedback ALS (Hu, Koren, Volinsky, ICDM 2008).
    Preference p_ui = 1 if r_ui > 0; confidence c_ui = 1 + alpha * r_ui.
    Each user row solves (Y^T Y + Y^T (C_u - I) Y + reg I) x_u = Y^T C_u p_u."""
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape
    X = (rng.standard_normal((n_u, factors)) * 0.01).astype(np.float64)
    Y = (rng.standard_normal((n_i, factors)) * 0.01).astype(np.float64)
    Rt = R.T.tocsr()
    I = reg * np.eye(factors)

    def solve(A, B, F):
        FtF = F.T @ F
        for r in range(A.shape[0]):
            s, e = A.indptr[r], A.indptr[r + 1]
            if s == e:
                B[r] = 0.0
                continue
            idx = A.indices[s:e]
            c = 1.0 + alpha * A.data[s:e]
            Fi = F[idx]
            lhs = FtF + (Fi.T * (c - 1.0)) @ Fi + I
            rhs = (Fi.T * c).sum(axis=1)
            B[r] = np.linalg.solve(lhs, rhs)

    for _ in range(iters):
        solve(R, X, Y)
        solve(Rt, Y, X)
    return X, Y


def als(td: TrainData, eval_users, factors=64, reg=0.1, alpha=10.0, iters=15, weight="cnt", seed=42):
    if weight == "cnt":
        R = td.cnt_decay.copy()
    else:  # log of watched seconds, any screen time
        R = td.sec.copy()
        R.data = np.log1p(R.data / 60.0)
    R = R.tocsr().astype(np.float64)
    X, Y = als_fit(R, factors, reg, alpha, iters, seed)
    return (X[eval_users] @ Y.T).astype(np.float32)


# ---------------------------------------------------------------- M2: content-based filtering
def item_content_matrix(air, n_items):
    """TF-IDF over metadata tokens: category, genres, decade, director names, actor names."""
    meta = air.sort_values("start").groupby("i").last()  # metadata is ~98-100% constant per id
    rows, toks = [], []
    for i, r in meta.iterrows():
        t = ["cat=" + r["category"]]
        t += ["genre=" + g for g in (r["genre_1"], r["genre_2"], r["genre_3"]) if g]
        y = re.match(r"(\d{4})", str(r["year"]))
        if y:
            t.append("decade=%d" % (int(y.group(1)) // 10 * 10))
        t += ["dir=" + d.strip() for d in str(r["director"]).split(",") if d.strip()]
        t += ["act=" + a.strip() for a in str(r["actors"]).split(",") if a.strip()]
        rows += [i] * len(t)
        toks += t
    vocab, col = np.unique(np.array(toks), return_inverse=True)
    X = sp.csr_matrix((np.ones(len(col), np.float32), (rows, col)), shape=(n_items, len(vocab)))
    X.data[:] = 1.0
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((1 + n_items) / (1 + df)) + 1.0
    X = X @ sp.diags(idf.astype(np.float32))
    n = spnorm(X, axis=1)
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ X, vocab


def cbf(td: TrainData, eval_users, item_X, profile="cnt_decay", sec_weight=0.0):
    W = td.cnt_decay[eval_users]
    if sec_weight > 0:
        S = td.sec[eval_users].copy()
        S.data = np.log1p(S.data / 60.0)
        W = W + sec_weight * S
    P = W @ item_X  # user profile = weighted sum of watched item vectors
    pn = spnorm(P, axis=1)
    pn[pn == 0] = 1.0
    P = sp.diags(1.0 / pn) @ P
    return np.asarray((P @ item_X.T).todense(), dtype=np.float32)  # cosine similarity


# ---------------------------------------------------------------- M3: personalized PageRank
_GRAPH_CACHE = {}


def coview_graph(td: TrainData, min_sec=600, top_n=50):
    """Item co-view graph: edge weight = number of users who watched both items
    (>= min_sec seconds each) in the train weeks. Only each item's top_n strongest
    edges are kept (the full graph is nearly dense and made one PPR run take ~7.5 min).
    Returned row-stochastic, i.e. P[i, j] = probability of stepping i -> j."""
    key = (id(td), min_sec, top_n)
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    B = td.sec.copy()
    B.data = (B.data >= min_sec).astype(np.float32)
    B.eliminate_zeros()
    A = (B.T @ B).toarray()
    np.fill_diagonal(A, 0)
    if top_n and top_n < A.shape[1]:
        thr = -np.partition(-A, top_n - 1, axis=1)[:, top_n - 1:top_n]
        A[A < thr] = 0
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    P = (A / deg).astype(np.float32)
    if top_n:  # pruned graph -> sparse; full graph stays dense (dense BLAS is much faster here)
        P = sp.csr_matrix(P)
    _GRAPH_CACHE[key] = P
    return P


def ppr(td: TrainData, eval_users, restart=0.3, iters=20, min_sec=600, top_n=50):
    """Random walk with restart on the co-view graph; the restart vector is the user's own
    history (recency-decayed >=0.8 watch counts + light log-seconds signal), normalised to 1."""
    P = coview_graph(td, min_sec, top_n)

    E = td.cnt_decay[eval_users].toarray()
    S = td.sec[eval_users].copy()
    S.data = np.log1p(S.data / 60.0)
    E = E + 0.1 * S.toarray()
    z = E.sum(axis=1, keepdims=True)
    z[z == 0] = 1.0
    E = E / z
    R = E.copy()
    for _ in range(iters):
        RP = (R.astype(np.float32) @ P) if isinstance(P, np.ndarray) else (P.T @ R.T).T
        R = restart * E + (1 - restart) * RP  # one step: r <- a e + (1-a) r P
    return R.astype(np.float32)
