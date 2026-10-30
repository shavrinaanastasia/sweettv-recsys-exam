"""Two example users, chosen by a fixed rule (no cherry-picking):
among validation users who are also Kaggle test users, rank by number of >=0.8 airings
watched in train weeks 11-20; take the user closest to the 90th percentile (heavy viewer)
and the one closest to the 10th percentile (light viewer); ties -> smaller user_id.
For each, every method's Top-5 (validation models, train 11-20) is shown next to the
user's own top programmes in train weeks 11-20 and the actual label set from weeks 21-30.
Output: results/examples.json and results/examples.md
"""
import json

import numpy as np

import hybrid as H
import validate as V
from common import RES, apk, top_k

td, air, items, users, labels = V.td, V.air, V.items, V.users, V.labels
cand = V.eval_u[V.is_test_user]
activity = np.asarray(td.cnt[cand].sum(axis=1)).ravel()


def closest(q):
    target = np.quantile(activity, q)
    d = np.abs(activity - target)
    best = np.flatnonzero(d == d.min())
    return int(cand[best[np.argmin(users[cand[best]])]])


chosen = {"heavy (p90)": closest(0.9), "light (p10)": closest(0.1)}
eu = np.array(list(chosen.values()))
comps, _ = H.components(td, air, len(items), eu)
norm = {k: H.normalise(S, td) for k, S in comps.items()}
weights = json.loads((RES / "hybrid_val.json").read_text(encoding="utf-8"))["hybrid"]["weights"]
variants = {"popularity": {"popularity": 1.0}, "history (reference)": {"history": 1.0},
            "M1 ALS": {"als": 1.0}, "M2 content-based": {"cbf": 1.0}, "M3 PPR": {"ppr": 1.0},
            "hybrid (final)": weights}
meta = air.sort_values("start").groupby("i").last()


def describe(i):
    r = meta.loc[int(i)]
    return {"tv_show_id": int(items[i]), "title": r["title"], "category": r["category"]}


out, md = {}, ["# Example users (validation: train weeks 11-20, actual = labels from weeks 21-30)\n"]
for row, (role, u) in enumerate(chosen.items()):
    truth = labels[u]
    hist_cnt = td.cnt[u].toarray().ravel()
    hist_top = np.argsort(-hist_cnt)[:5]
    entry = {"user_id": str(users[u]), "role": role,
             "train_airings_watched_ge_0_8": int(hist_cnt.sum()),
             "distinct_programmes_watched_ge_0_8": int((hist_cnt > 0).sum()),
             "train_top5": [dict(describe(i), count=int(hist_cnt[i])) for i in hist_top if hist_cnt[i] > 0],
             "actual_21_30": [describe(i) for i in truth], "methods": {}}
    md.append(f"## {role}: user {users[u]}  ({entry['train_airings_watched_ge_0_8']} airings watched >=80% "
              f"in weeks 11-20, {entry['distinct_programmes_watched_ge_0_8']} distinct programmes)\n")
    md.append("Actual weeks 21-30 label: " + "; ".join(
        f"{d['tv_show_id']} {d['title']} [{d['category']}]" for d in entry["actual_21_30"]) + "\n")
    for name, w in variants.items():
        P, _ = top_k(H.fuse({k: v[row:row + 1] for k, v in norm.items()}, w, td, len(items)), V.fallback)
        recs = P[0]
        entry["methods"][name] = {"top5": [dict(describe(i), in_actual=bool(i in set(truth)),
                                                in_train_history=bool(hist_cnt[i] > 0)) for i in recs],
                                  "AP@5": round(apk(recs, truth), 4)}
        md.append(f"**{name}** (AP@5 = {entry['methods'][name]['AP@5']}):")
        for d in entry["methods"][name]["top5"]:
            flag = ("HIT " if d["in_actual"] else "") + ("seen" if d["in_train_history"] else "new")
            md.append(f"- {d['tv_show_id']} | {d['title']} | {d['category']} | {flag}")
        md.append("")
    out[role] = entry

(RES / "examples.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
(RES / "examples.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
