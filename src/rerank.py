"""Candidate generation + learned re-ranking (LightGBM LambdaRank).

Stage 1: the score-fusion hybrid (weights from results/hybrid_val.json) proposes the top N
         programmes per user from the candidate pool.
Stage 2: a LambdaRank model re-orders those N using component scores and raw signals.

No validation labels are used for fitting:
  validation : fit on history 11-15 -> labels 16-20, evaluate on history 11-20 -> labels 21-30
  test       : fit on history 11-20 -> labels 21-30, apply to history 11-30 -> weeks 31-42
Usage: python rerank.py val      |   python rerank.py test
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import hybrid as H
from common import RES, ROOT, SPLITS, apk, build_labels, load, make_train, popularity_scores

N_CAND = int(sys.argv[2]) if len(sys.argv) > 2 else 50  # run 1 used 50 (recall@50 = 0.68 on validation)
SEED = 42
W1 = json.loads((RES / "hybrid_val.json").read_text(encoding="utf-8"))["hybrid"]["weights"]
CONFIGS = {
    "small": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100),
    "large": dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50),
}


def build(td, air, n_items, eval_u, labels=None):
    """Return candidate item matrix (n_u, N), feature frame (n_u*N rows), relevance (or None)."""
    comps, _ = H.components(td, air, n_items, eval_u)
    norm = {k: H.normalise(S, td) for k, S in comps.items()}
    fused = H.fuse(norm, W1, td, n_items)
    cand = np.argpartition(-fused, N_CAND, axis=1)[:, :N_CAND]
    order = np.argsort(-np.take_along_axis(fused, cand, axis=1), axis=1)
    cand = np.take_along_axis(cand, order, axis=1)

    pool_pos = np.full(n_items, -1)
    pool_pos[td.pool] = np.arange(len(td.pool))
    rows = np.repeat(np.arange(len(eval_u)), N_CAND)
    items = cand.ravel()
    f = {}
    for k in comps:
        f[f"raw_{k}"] = comps[k][rows, items]
        f[f"norm_{k}"] = norm[k][rows, pool_pos[items]]
        # rank of the item inside this component's own ordering for the user (0 = top)
        rank = np.empty(norm[k].shape, np.int32)
        np.put_along_axis(rank, np.argsort(-norm[k], axis=1), np.arange(norm[k].shape[1], dtype=np.int32)[None, :], axis=1)
        f[f"rank_{k}"] = rank[rows, pool_pos[items]]
        del rank
    f["fused"] = fused[rows, items]
    f["fused_rank"] = np.tile(np.arange(N_CAND), len(eval_u))
    sec = td.sec[eval_u]
    f["log_sec"] = np.log1p(np.asarray(sec[rows, items]).ravel())
    f["cnt"] = np.asarray(td.cnt[eval_u][rows, items]).ravel()
    f["cnt_decay"] = np.asarray(td.cnt_decay[eval_u][rows, items]).ravel()
    f["target_airings"] = td.pool_airings[items]
    f["train_airings"] = td.train_airings[items]
    f["airing_ratio"] = (td.pool_airings[items] + 1) / (td.train_airings[items] + 1)
    f["hit_rate"] = f["cnt"] / np.maximum(td.train_airings[items], 1)
    f["exp_count"] = f["hit_rate"] * td.pool_airings[items]
    f["user_total_cnt"] = np.asarray(td.cnt[eval_u].sum(axis=1)).ravel()[rows]
    f["user_distinct"] = np.diff(td.cnt[eval_u].indptr)[rows]
    f["user_log_sec"] = np.log1p(np.asarray(sec.sum(axis=1)).ravel())[rows]
    X = pd.DataFrame(f)

    y = None
    if labels is not None:
        rel = np.zeros((len(eval_u), N_CAND), np.int32)
        for r, u in enumerate(eval_u):
            pos = {i: 5 - j for j, i in enumerate(labels[u])}  # label rank 1 -> relevance 5
            rel[r] = [pos.get(i, 0) for i in cand[r]]
        y = rel.ravel()
    return cand, X, y


def fit(X, y, cfg):
    """Native LightGBM API (the sklearn wrapper needs scikit-learn, which we do not install)."""
    c = dict(CONFIGS[cfg])
    params = {"objective": "lambdarank", "metric": "map", "eval_at": [5], "seed": SEED,
              "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1,
              "learning_rate": c["learning_rate"], "num_leaves": c["num_leaves"],
              "min_data_in_leaf": c["min_child_samples"], "deterministic": True, "num_threads": 8}
    ds = lgb.Dataset(X, label=y, group=[N_CAND] * (len(X) // N_CAND))
    return lgb.train(params, ds, num_boost_round=c["n_estimators"])


def rerank_top5(model, cand, X):
    s = model.predict(X).reshape(cand.shape)
    return np.take_along_axis(cand, np.argsort(-s, axis=1)[:, :5], axis=1)


def recall_at_n(cand, labels, eval_u):
    return float(np.mean([np.isin(labels[u], cand[r]).mean() for r, u in enumerate(eval_u)]))


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    ua, air, users, items, uc = load()
    n_u, n_i = len(users), len(items)
    fit_split = "fit_val" if mode == "val" else "val"
    td_fit = make_train(ua, air, uc, n_u, n_i, fit_split)
    lab_fit = build_labels(ua, SPLITS[fit_split][1])
    fit_u = np.array(sorted(lab_fit))
    cand_f, X_f, y_f = build(td_fit, air, n_i, fit_u, lab_fit)
    info = {"mode": mode, "fit_split": fit_split, "fit_users": len(fit_u),
            "fit_candidate_recall@50": recall_at_n(cand_f, lab_fit, fit_u)}
    print(info, flush=True)

    if mode == "val":
        import validate as V  # same validation users, A/B split and labels as every other method
        cand_v, X_v, _ = build(V.td, air, n_i, V.eval_u)
        info["val_candidate_recall@50"] = recall_at_n(cand_v, V.labels, V.eval_u)
        info["configs"] = {}
        for cfg in CONFIGS:
            model = fit(X_f, y_f, cfg)
            P = rerank_top5(model, cand_v, X_v)
            aps = np.array([apk(P[r], V.labels[u]) for r, u in enumerate(V.eval_u)])
            info["configs"][cfg] = {"A": float(aps[V.in_a].mean()), "B": float(aps[~V.in_a].mean()),
                                    "all": float(aps.mean()), "test_users": float(aps[V.is_test_user].mean())}
            imp = pd.Series(model.feature_importance("gain"), index=X_f.columns)
            info["configs"][cfg]["top_features_gain"] = (imp / imp.sum()).sort_values(ascending=False).head(10).round(4).to_dict()
            print(cfg, info["configs"][cfg], flush=True)
        info["runtime_s"] = round(time.time() - t0, 1)
        info["n_cand"] = N_CAND
        (RES / f"rerank_val_n{N_CAND}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    else:
        cfg = json.loads((RES / "rerank_choice.json").read_text(encoding="utf-8"))["config"]
        model = fit(X_f, y_f, cfg)
        td = make_train(ua, air, uc, n_u, n_i, "test")
        sub = pd.read_csv(ROOT / "data" / "submission.csv", dtype={"user_id": np.uint64})
        eval_u = np.searchsorted(users, sub["user_id"].values)
        assert np.all(users[eval_u] == sub["user_id"].values)
        cand_t, X_t, _ = build(td, air, n_i, eval_u)
        P = rerank_top5(model, cand_t, X_t)
        ids = items[P]
        assert all(len(set(r)) == 5 for r in ids) and (ids != 0).all() and np.isin(P, td.pool).all()
        out = ROOT / "submissions" / "submission_rerank.csv"
        pd.DataFrame({"user_id": sub["user_id"].astype(str),
                      "tv_show_id": [" ".join(map(str, r)) for r in ids]}).to_csv(out, index=False)
        info.update({"config": cfg, "file": str(out.relative_to(ROOT)), "runtime_s": round(time.time() - t0, 1)})
        (RES / "rerank_test.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(info)
