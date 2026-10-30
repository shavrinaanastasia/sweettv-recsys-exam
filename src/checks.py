"""Independent sanity checks. Output: results/checks.json"""
import csv
import json
from datetime import datetime

import numpy as np
import scipy.sparse as sp

import methods as M
from common import RES, ROOT, apk, build_labels, load, make_train, SPLITS

out = {}

# 1. MAP@5 against the two worked examples on the exam slides (relevant = {A, B, C}).
x = apk(list("ABDCE"), set("ABC"))
y = apk(list("DAEBC"), set("ABC"))
out["apk_slide_example"] = {"method_X": round(x, 4), "expected_X": 0.917,
                            "method_Y": round(y, 4), "expected_Y": 0.533,
                            "pass": abs(x - 0.9167) < 1e-3 and abs(y - 0.5333) < 1e-3}

# 2. Re-derive one (user, airing) overlap straight from the raw CSV text, bypassing pandas/join.py.
ua, air, users, items, uc = load()
pick = ua[(ua["screen"] >= 0.8) & (ua["week"] == 15)].iloc[0]
a = air[air["airing_idx"] == pick["airing_idx"]].iloc[0]
uid = str(users[int(pick["u"])])
a_start = datetime.strptime("1970-01-01", "%Y-%m-%d")  # epoch-naive, same convention as prep.py
to_s = lambda s, fmt: int((datetime.strptime(s, fmt) - a_start).total_seconds())
ov = 0
with open(ROOT / "data" / "dataset11-30.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["user_id"] != uid or int(row["vsetv_id"]) != int(a["channel_id"]):
            continue
        s = to_s(row["start_time"], "%Y-%m-%d %H:%M:%S")
        e = to_s(row["stop_time"], "%Y-%m-%d %H:%M:%S")
        ov += max(0, min(e, int(a["end"])) - max(s, int(a["start"])))
ov = min(ov, int(a["duration"]))
out["join_recomputed_from_raw_csv"] = {
    "user_id": uid, "channel_id": int(a["channel_id"]), "tv_show_id": int(a["tv_show_id"]),
    "title": a["title"], "airing_start_epoch_s": int(a["start"]), "duration_s": int(a["duration"]),
    "overlap_join_py": int(pick["overlap"]), "overlap_raw_csv": ov, "pass": ov == int(pick["overlap"]),
}

# 3. Label recipe: every validation label set has 5 distinct items, each watched >= 0.8 at least once
#    in weeks 21-30 by that user.
labels = build_labels(ua, SPLITS["val"][1])
w = ua[(ua["week"] >= 21) & (ua["week"] <= 30) & (ua["screen"] >= 0.8)]
seen = set(zip(w["u"].values, w["i"].values))
out["labels"] = {"users": len(labels),
                 "all_have_5_distinct": all(len(set(v)) == 5 for v in labels.values()),
                 "all_items_watched_ge_0_8": all((u, i) in seen for u, v in labels.items() for i in v)}

# 4. ALS: the weighted squared-error objective must not increase between iterations.
td = make_train(ua, air, uc, len(users), len(items), "val")
R = td.cnt_decay[:600].tocsr().astype(np.float64)
keep = np.flatnonzero(np.asarray(R.sum(axis=0)).ravel() > 0)
R = R[:, keep].tocsr()
alpha, reg = 5.0, 1.0
Pm = (R > 0).astype(np.float64).toarray()
C = 1 + alpha * R.toarray()


def loss(X, Y):
    return float((C * (Pm - X @ Y.T) ** 2).sum() + reg * ((X ** 2).sum() + (Y ** 2).sum()))


losses = []
for it in range(1, 7):
    X, Y = M.als_fit(R, factors=16, reg=reg, alpha=alpha, iters=it, seed=42)
    losses.append(round(loss(X, Y), 2))
out["als_objective_by_iteration"] = {"losses": losses,
                                     "monotone_non_increasing": all(b <= a + 1e-6 for a, b in zip(losses, losses[1:]))}

# 5. PPR: transition matrix rows sum to 1 (or 0 for isolated items); scores keep total mass 1.
P = M.coview_graph(td, min_sec=300, top_n=0)
rs = P.sum(axis=1)
eu = np.arange(50)
Rp = M.ppr(td, eu, restart=0.15, min_sec=300, top_n=0)
out["ppr"] = {"transition_rows_sum_to_1_or_0": bool(np.all((np.abs(rs - 1) < 1e-4) | (rs == 0))),
              "score_mass_range_first_50_users": [float(Rp.sum(axis=1).min()), float(Rp.sum(axis=1).max())]}

(RES / "checks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
