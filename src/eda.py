"""Measure the data facts the report relies on. Output: results/eda.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

views = pd.read_parquet(CACHE / "views.parquet")
sched = pd.read_parquet(CACHE / "schedule.parquet")
sub = pd.read_csv(ROOT / "data" / "submission.csv", dtype={"user_id": np.uint64, "tv_show_id": str})

out = {}

# ---- view log
d = views["duraton"].values
out["views"] = {
    "rows": len(views),
    "users": int(views["user_id"].nunique()),
    "channels": int(views["channel_id"].nunique()),
    "weeks": sorted(int(w) for w in views["week"].unique()),
    "duraton_equals_stop_minus_start_share": float(np.mean(d == views["stop"].values - views["start"].values)),
    "duraton_quantiles_s": {q: float(np.quantile(d, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]},
    "duraton_max_s": int(d.max()),
    "duraton_min_s": int(d.min()),
}

# ---- schedule
per_file = {}
for tag, g in sched.groupby("file"):
    lab = g[g["tv_show_id"] != 0]
    per_file[tag] = {
        "rows": len(g),
        "labelled_rows": len(lab),
        "labelled_share": len(lab) / len(g),
        "labelled_airtime_share": float(lab["duration"].sum() / g["duration"].sum()),
        "distinct_programmes": int(lab["tv_show_id"].nunique()),
        "channels": int(g["channel_id"].nunique()),
        "channels_with_labelled_rows": int(lab["channel_id"].nunique()),
        "categories": sorted(lab["category"].unique().tolist()),
        "weeks": [int(g["week"].min()), int(g["week"].max())],
        "programme_duration_quantiles_s": {q: float(np.quantile(lab["duration"], q)) for q in [0.1, 0.5, 0.9]},
    }
out["schedule"] = per_file

lab = sched[sched["tv_show_id"] != 0]
train_items = set(lab.loc[lab["file"] != "31-42", "tv_show_id"])
test_items = set(lab.loc[lab["file"] == "31-42", "tv_show_id"])
val_items = set(lab.loc[lab["file"] == "21-30", "tv_show_id"])
w1120_items = set(lab.loc[lab["file"] == "11-20", "tv_show_id"])
out["item_overlap"] = {
    "test_items": len(test_items),
    "test_items_seen_in_train_schedule": len(test_items & train_items),
    "val_items": len(val_items),
    "val_items_seen_in_11_20_schedule": len(val_items & w1120_items),
}

# Does one tv_show_id always carry the same metadata? (needed for content vectors)
meta_cols = ["category", "genre_1", "genre_2", "genre_3", "year", "director", "actors"]
nuniq = lab.groupby("tv_show_id")[meta_cols].nunique()
out["metadata_consistency_share_items_with_single_value"] = {c: float((nuniq[c] <= 1).mean()) for c in meta_cols}
out["metadata_nonempty_share_items"] = {
    c: float((lab.groupby("tv_show_id")[c].first() != "").mean()) for c in meta_cols
}

# ---- channel coverage: do view channels exist in the schedule?
sched_ch = set(sched["channel_id"])
vc = views["channel_id"].values
out["views_on_scheduled_channels_share"] = float(np.isin(vc, list(sched_ch)).mean())

# ---- test users
test_users = sub["user_id"].values
in_log = np.isin(test_users, views["user_id"].unique())
out["test_users"] = {
    "count": len(test_users),
    "unique": int(len(np.unique(test_users))),
    "present_in_view_log": int(in_log.sum()),
    "template_first_row": sub.iloc[0].tolist().__repr__(),
}
tu_views = views[views["user_id"].isin(test_users)]
out["test_users"]["views_rows"] = len(tu_views)
out["test_users"]["present_in_weeks_21_30"] = int(tu_views.loc[tu_views["week"] >= 21, "user_id"].nunique())
out["test_users"]["present_in_weeks_11_20"] = int(tu_views.loc[tu_views["week"] <= 20, "user_id"].nunique())
out["all_users_present_in_weeks_21_30"] = int(views.loc[views["week"] >= 21, "user_id"].nunique())


# ---- clock alignment check: how often does a view start within 60 s of a programme
# boundary on its channel, if the view log is shifted by h hours?
def boundary_hit_rate(shift_h, sample=300_000, tol=60):
    rng = np.random.default_rng(0)
    s = views.sample(n=sample, random_state=0)
    hits = 0
    total = 0
    for ch, g in s.groupby("channel_id"):
        b = np.sort(sched.loc[sched["channel_id"] == ch, "start"].values) if False else BOUNDS.get(ch)
        if b is None or len(b) == 0:
            continue
        t = g["start"].values + shift_h * 3600
        i = np.searchsorted(b, t)
        lo = b[np.clip(i - 1, 0, len(b) - 1)]
        hi = b[np.clip(i, 0, len(b) - 1)]
        dist = np.minimum(np.abs(t - lo), np.abs(t - hi))
        hits += int((dist <= tol).sum())
        total += len(t)
    return hits / total


BOUNDS = {ch: np.sort(g["start"].values) for ch, g in sched.groupby("channel_id")}
out["clock_alignment_view_start_within_60s_of_programme_start"] = {
    str(h): round(boundary_hit_rate(h), 4) for h in [-3, -2, -1, 0, 1, 2, 3]
}

(RES / "eda.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
