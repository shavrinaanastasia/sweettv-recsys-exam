"""Numbers quoted in the report that no other script saves -> results/report_stats.json:
co-view graph density, titles per programme id, cold-start counts and the sensitivity of the 0.8 cut."""
import json

import numpy as np
import pandas as pd

from common import RES, ROOT, WATCHED, load, make_train

ua, air, users, items, uc = load()
out = {}

# Co-view graph density (validation train weeks 11-20, co-view = both watched >= 300 s), see §5.2.
td = make_train(ua, air, uc, len(users), len(items), "val")
B = td.sec.copy()
B.data = (B.data >= 300).astype(np.float32)
B.eliminate_zeros()
A = (B.T @ B).tocsr()
A.setdiag(0)
A.eliminate_zeros()
active = int((np.asarray(B.sum(axis=0)).ravel() > 0).sum())
out["coview_graph_items_total"] = len(items)
out["coview_graph_items_with_any_viewer_11_20"] = active
out["coview_graph_edge_density_among_those"] = round(A.nnz / (active * (active - 1)), 4)

# One id spans many airings with different titles (§4.1).
t = air.groupby("tv_show_id")["title"].nunique()
out["ids_with_more_than_one_title"] = int((t > 1).sum())
out["distinct_titles_id_12002955"] = int(t.loc[12002955])
out["distinct_titles_id_2400480"] = int(t.loc[2400480])

# Cold start among the 1,260 test users, history weeks 11-30 (§4 Setup).
sub = pd.read_csv(ROOT / "data" / "submission.csv", dtype={"user_id": np.uint64})
tu = np.searchsorted(users, sub["user_id"].values)
h = ua[(ua["week"] <= 30) & (ua["screen"] >= WATCHED)]
n_items = h.groupby("u")["i"].nunique()
k = n_items.reindex(tu).fillna(0).values
out["test_users"] = len(tu)
out["test_users_zero_ge08_programmes"] = int((k == 0).sum())
out["test_users_lt5_ge08_programmes"] = int((k < 5).sum())

# Sensitivity of the 0.8 cut (§3, Figure 1): how many pairs would be added/removed by moving it.
s = ua["screen"].values
out["pairs_total"] = len(s)
out["pairs_ge"] = {str(c): int((s >= c).sum()) for c in (0.7, 0.75, 0.8, 0.85, 0.9)}

(RES / "report_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
