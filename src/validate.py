"""Validation: train weeks 11-20, labels from weeks 21-30 (competition recipe).

Validation users are split 50/50 at random (seed 0) into a tuning half A and a reporting
half B, mirroring the public/private user split of the leaderboard. Hyperparameters are
picked on A only; B is reported as the untouched estimate. Every grid point is logged.
"""
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from common import RES, SPLITS, build_labels, load, make_train, map_at_k, popularity_scores, top_k
import methods as M

ua, air, users, items, uc = load()
td = make_train(ua, air, uc, len(users), len(items), "val")
labels = build_labels(ua, SPLITS["val"][1])
eval_u = np.array(sorted(labels))
rng = np.random.default_rng(0)
in_a = rng.random(len(eval_u)) < 0.5
test_ids = pd.read_csv(RES.parent / "data" / "submission.csv", dtype={"user_id": np.uint64})["user_id"].values
is_test_user = np.isin(users[eval_u], test_ids)

pop = popularity_scores(td)
fallback = td.pool[np.argsort(-pop[td.pool])]


def evaluate(S):
    P, filled = top_k(M.mask_pool(S, td), fallback)
    ap = {}
    from common import apk
    aps = np.array([apk(P[r], labels[u]) for r, u in enumerate(eval_u)])
    return {
        "A": float(aps[in_a].mean()), "B": float(aps[~in_a].mean()), "all": float(aps.mean()),
        "test_users": float(aps[is_test_user].mean()), "filled_from_popularity": int(filled),
    }, P


log = []


def run(name, fn, grid):
    best = None
    for params in grid:
        t0 = time.time()
        S = fn(**params)
        res, P = evaluate(S)
        res.update({"method": name, "params": params, "seconds": round(time.time() - t0, 1)})
        log.append(res)
        line = json.dumps(res, ensure_ascii=False)
        print(line, flush=True)
        with open(RES / "val_runs.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
        if best is None or res["A"] > best[0]["A"]:
            best = (res, S, P)
    return best


def grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


if __name__ == "__main__":
    which = sys.argv[1:] or ["pop", "hist", "cbf", "ppr", "als"]
    best = {}
    print(f"val users: {len(eval_u)} (A={in_a.sum()}, B={(~in_a).sum()}, test users among them={is_test_user.sum()})")
    if "pop" in which:
        best["popularity"] = run("popularity", lambda: np.tile(pop, (len(eval_u), 1)), [{}])
    if "hist" in which:
        best["history"] = run("history", lambda decay: M.history(td, eval_u, decay), grid(decay=[False, True]))
    if "cbf" in which:
        item_X, vocab = M.item_content_matrix(air, len(items))
        print("content vocab size:", len(vocab))
        best["cbf"] = run("cbf", lambda sec_weight: M.cbf(td, eval_u, item_X, sec_weight=sec_weight),
                          grid(sec_weight=[0.0, 0.1, 0.5]))
    if "ppr" in which:
        best["ppr"] = run("ppr", lambda **p: M.ppr(td, eval_u, **p),
                          grid(restart=[0.15, 0.3, 0.5], min_sec=[300, 1200], top_n=[50, 0]))
    if "als2" in which:  # second ALS grid: run 1 put the best point on the grid edge (alpha=5, factors=32)
        best["als"] = run("als", lambda **p: M.als(td, eval_u, iters=10, weight="cnt", **p),
                          grid(factors=[16, 32], reg=[1.0, 10.0], alpha=[0.5, 1.0, 2.0]))
    if "als3" in which:  # third ALS grid: run 2 best (factors=16, reg=10, alpha=1) again on the edge
        best["als"] = run("als", lambda **p: M.als(td, eval_u, iters=10, weight="cnt", **p),
                          grid(factors=[8, 16], reg=[10.0, 30.0, 100.0], alpha=[1.0]))
    if "als" in which:
        best["als"] = run("als", lambda **p: M.als(td, eval_u, iters=10, **p),
                          grid(factors=[32, 64], reg=[0.1, 1.0], alpha=[5.0, 20.0], weight=["cnt", "sec"]))
    out = RES / f"val_log_{'_'.join(which)}.json"
    out.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, (r, _, _) in best.items():
        print(f"BEST {k}: A={r['A']:.4f} B={r['B']:.4f} all={r['all']:.4f} params={r['params']}")
