"""Score-fusion hybrid.

Each component score matrix is restricted to the candidate pool and min-max normalised
per user (scores of different methods are not comparable probabilities, so only their
within-user shape is kept). The fused score is a weighted sum. Weights are chosen by
coordinate ascent on validation half A only; half B is reported untouched.

Run:  python hybrid.py            -> validation (train 11-20, labels 21-30)
"""
import json
import time

import numpy as np

import methods as M
from common import RES, apk, top_k

BEST = json.loads((RES / "best_params.json").read_text(encoding="utf-8"))


def channel_affinity(td, eval_users):
    """Expected exposure: share of the user's (all-airtime) viewing on each channel x
    number of target-week airings of the item on that channel. Uses the target-week
    SCHEDULE only (allowed), never target-week views."""
    uc = td.user_channel[eval_users]
    tot = np.asarray(uc.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    ucs = uc.multiply(1.0 / tot[:, None]).tocsr()
    item_air_on_ch = td.item_channel_share.multiply(td.pool_airings[:, None]).tocsr()
    return np.asarray((ucs @ item_air_on_ch.T).todense(), dtype=np.float32)


def components(td, air, n_items, eval_users, log=print):
    out, secs = {}, {}

    def timed(name, fn):
        t0 = time.time()
        out[name] = fn()
        secs[name] = round(time.time() - t0, 1)
        log(f"  component {name}: {secs[name]} s")

    from common import popularity_scores
    p = popularity_scores(td)
    p[~np.isfinite(p)] = 0
    timed("popularity", lambda: np.tile(p, (len(eval_users), 1)))
    timed("history", lambda: M.history(td, eval_users, **BEST["history"]))
    item_X, _ = M.item_content_matrix(air, n_items)
    timed("cbf", lambda: M.cbf(td, eval_users, item_X, **BEST["cbf"]))
    timed("ppr", lambda: M.ppr(td, eval_users, **BEST["ppr"]))
    timed("als", lambda: M.als(td, eval_users, iters=10, **BEST["als"]))
    timed("channel", lambda: channel_affinity(td, eval_users))
    return out, secs


def normalise(S, td):
    S = S[:, td.pool].astype(np.float32)
    lo = S.min(axis=1, keepdims=True)
    hi = S.max(axis=1, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (S - lo) / rng


def fuse(norm, weights, td, n_items):
    fused = np.zeros_like(next(iter(norm.values())))
    for k, w in weights.items():
        if w:
            fused += w * norm[k]
    full = np.full((fused.shape[0], n_items), -np.inf, np.float32)
    full[:, td.pool] = fused  # zero rows (no signal) fall back to popularity inside top_k
    return full


def evaluate_fused(norm, weights, td, n_items, labels, eval_u, fallback, mask):
    P, _ = top_k(fuse(norm, weights, td, n_items), fallback)
    aps = np.array([apk(P[r], labels[u]) for r, u in enumerate(eval_u)])
    return aps, P


def coordinate_ascent(norm, td, n_items, labels, eval_u, fallback, in_a, start, grid, rounds=3, log=print):
    w = dict(start)
    best = evaluate_fused(norm, w, td, n_items, labels, eval_u, fallback, in_a)[0][in_a].mean()
    trace = [{"weights": dict(w), "A": float(best)}]
    for _ in range(rounds):
        improved = False
        for k in w:
            for v in grid:
                if v == w[k]:
                    continue
                cand = dict(w, **{k: v})
                score = evaluate_fused(norm, cand, td, n_items, labels, eval_u, fallback, in_a)[0][in_a].mean()
                if score > best + 1e-6:
                    best, w, improved = score, cand, True
                    trace.append({"weights": dict(w), "A": float(best)})
                    log(f"  A={best:.4f} {w}")
        if not improved:
            break
    return w, trace


if __name__ == "__main__":
    import validate as V  # reuses the exact same split, labels, users and fallback

    norm_input, secs = components(V.td, V.air, len(V.items), V.eval_u)
    norm = {k: normalise(S, V.td) for k, S in norm_input.items()}
    results = {"component_seconds": secs, "single": {}}
    for k in norm:
        aps, _ = evaluate_fused(norm, {k: 1.0}, V.td, len(V.items), V.labels, V.eval_u, V.fallback, V.in_a)
        results["single"][k] = {"A": float(aps[V.in_a].mean()), "B": float(aps[~V.in_a].mean()),
                                "all": float(aps.mean()), "test_users": float(aps[V.is_test_user].mean())}
        print(k, results["single"][k])
    start = {k: 0.0 for k in norm}
    start["history"] = 1.0
    # Run 1 (default, chosen): grid up to 2.0. Run 2 (`python hybrid.py wide`): the channel weight had ended on
    # the 2.0 edge, so the grid was widened; run 2 scored lower on half A (0.34767 vs 0.34838) and was not kept.
    import sys
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    if "wide" in sys.argv[1:]:
        grid += [3.0, 4.0, 6.0]
    w, trace = coordinate_ascent(norm, V.td, len(V.items), V.labels, V.eval_u, V.fallback, V.in_a, start, grid)
    aps, P = evaluate_fused(norm, w, V.td, len(V.items), V.labels, V.eval_u, V.fallback, V.in_a)
    results["hybrid"] = {"weights": w, "A": float(aps[V.in_a].mean()), "B": float(aps[~V.in_a].mean()),
                         "all": float(aps.mean()), "test_users": float(aps[V.is_test_user].mean())}
    results["trace"] = trace
    # Ablation: drop each component from the chosen hybrid
    results["ablation_B"] = {}
    for k in w:
        if w[k]:
            a2, _ = evaluate_fused(norm, dict(w, **{k: 0.0}), V.td, len(V.items), V.labels, V.eval_u, V.fallback, V.in_a)
            results["ablation_B"][f"without_{k}"] = float(a2[~V.in_a].mean())
    (RES / "hybrid_val.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    np.save(RES.parent / "cache" / "val_hybrid_top5.npy", P)
    print(json.dumps(results["hybrid"], indent=1))
    print(json.dumps(results["ablation_B"], indent=1))
