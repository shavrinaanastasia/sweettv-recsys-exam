"""Final predictions: refit every component on weeks 11-30 (all training data), candidate pool =
labelled programmes airing in weeks 31-42 (schedule only), fuse with the weights chosen on
validation, and write one submission file per method plus the hybrid.

Usage: python predict.py
"""
import json
import time

import numpy as np
import pandas as pd

import hybrid as H
from common import RES, ROOT, load, make_train, popularity_scores, top_k

t0 = time.time()
ua, air, users, items, uc = load()
td = make_train(ua, air, uc, len(users), len(items), "test")
sub = pd.read_csv(ROOT / "data" / "submission.csv", dtype={"user_id": np.uint64, "tv_show_id": str})
eval_u = np.searchsorted(users, sub["user_id"].values)
assert np.all(users[eval_u] == sub["user_id"].values), "every test user must be in the view log"

pop = popularity_scores(td)
fallback = td.pool[np.argsort(-pop[td.pool])]
comps, secs = H.components(td, air, len(items), eval_u)
norm = {k: H.normalise(S, td) for k, S in comps.items()}
weights = json.loads((RES / "hybrid_val.json").read_text(encoding="utf-8"))["hybrid"]["weights"]

out_dir = ROOT / "submissions"
out_dir.mkdir(exist_ok=True)
summary = {"pool_size": int(len(td.pool)), "component_seconds": secs, "weights": weights, "files": {}}
variants = {k: {k: 1.0} for k in norm}
variants["hybrid"] = weights
for name, w in variants.items():
    P, filled = top_k(H.fuse(norm, w, td, len(items)), fallback)
    ids = items[P]
    df = pd.DataFrame({"user_id": sub["user_id"].astype(str),
                       "tv_show_id": [" ".join(map(str, r)) for r in ids]})
    path = out_dir / f"submission_{name}.csv"
    df.to_csv(path, index=False)
    # format checks: 1260 rows, header, 5 distinct non-zero ids, all from the test-week pool
    assert len(df) == 1260 and df["user_id"].tolist() == sub["user_id"].astype(str).tolist()
    assert all(len(set(r)) == 5 for r in ids) and (ids != 0).all()
    assert np.isin(P, td.pool).all()
    summary["files"][name] = {"path": str(path.relative_to(ROOT)), "rows_filled_from_popularity": int(filled)}
    print(name, "->", path.name, "filled:", filled)
summary["runtime_s"] = round(time.time() - t0, 1)
(RES / "predict_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
