"""Join the view log to labelled programme airings.

Rule: same channel AND the view interval overlaps the airing interval.
For every (user, airing) pair the overlapping seconds of all that user's views are summed,
capped at the airing duration, and divided by it -> screen time in [0, 1].

Outputs (cache/):
  airings.parquet       one row per labelled airing (tv_show_id != 0, duration > 0)
  user_airing.parquet   one row per (user, airing) with overlap > 0
  user_channel.parquet  seconds per (user, channel, week) over ALL views, incl. unlabelled airtime
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RES = ROOT / "results"

t0 = time.time()
views = pd.read_parquet(CACHE / "views.parquet")
sched = pd.read_parquet(CACHE / "schedule.parquet")

# Labelled airings only; the three files share boundary days, so drop exact duplicates.
air = sched[(sched["tv_show_id"] != 0) & (sched["duration"] > 0)]
n_before = len(air)
air = air.drop_duplicates(["channel_id", "start", "tv_show_id"]).sort_values(["channel_id", "start"])
air = air.reset_index(drop=True)
air["airing_idx"] = np.arange(len(air), dtype=np.int64)
air.to_parquet(CACHE / "airings.parquet", index=False)
stats = {"labelled_airings_before_dedup": n_before, "labelled_airings": len(air)}

# Users -> dense index
users = np.sort(views["user_id"].unique())
views["u"] = np.searchsorted(users, views["user_id"].values).astype(np.int32)
pd.DataFrame({"user_id": users}).to_parquet(CACHE / "users.parquet", index=False)

# Context from ALL airtime (labelled or not): seconds per user-channel-week.
uc = views.groupby(["u", "channel_id", "week"], as_index=False)["duraton"].sum()
uc.to_parquet(CACHE / "user_channel.parquet", index=False)

pairs_u, pairs_a, pairs_ov = [], [], []
for ch, a in air.groupby("channel_id", sort=False):
    v = views[views["channel_id"] == ch]
    if v.empty:
        continue
    a_start = a["start"].values
    a_end = a["end"].values
    a_idx = a["airing_idx"].values
    end_cummax = np.maximum.accumulate(a_end)  # safe even if airings overlap on a channel
    vs, ve, vu = v["start"].values, v["stop"].values, v["u"].values
    first = np.searchsorted(end_cummax, vs, side="right")  # first airing that may end after view start
    last = np.searchsorted(a_start, ve, side="left")       # airings starting before view end
    n = np.maximum(last - first, 0)
    if n.sum() == 0:
        continue
    rep = np.repeat(np.arange(len(vs)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    j = np.repeat(first, n) + offs
    ov = np.minimum(ve[rep], a_end[j]) - np.maximum(vs[rep], a_start[j])
    keep = ov > 0
    pairs_u.append(vu[rep][keep])
    pairs_a.append(a_idx[j][keep])
    pairs_ov.append(ov[keep])

ua = pd.DataFrame({
    "u": np.concatenate(pairs_u),
    "airing_idx": np.concatenate(pairs_a),
    "overlap": np.concatenate(pairs_ov),
})
stats["view_airing_pairs"] = len(ua)
ua = ua.groupby(["u", "airing_idx"], as_index=False)["overlap"].sum()
meta = air.set_index("airing_idx")
ua["duration"] = meta.loc[ua["airing_idx"], "duration"].values
ua["overlap"] = np.minimum(ua["overlap"].values, ua["duration"].values)
ua["screen"] = (ua["overlap"] / ua["duration"]).astype(np.float32)
ua["tv_show_id"] = meta.loc[ua["airing_idx"], "tv_show_id"].values
ua["week"] = meta.loc[ua["airing_idx"], "week"].values.astype(np.int16)
ua.to_parquet(CACHE / "user_airing.parquet", index=False)

# Share of viewing seconds that land on labelled airtime (the 10.8% caveat, from the user side).
labelled_seconds = float(ua["overlap"].sum())
stats.update({
    "user_airing_pairs": len(ua),
    "pairs_screen_ge_0_8": int((ua["screen"] >= 0.8).sum()),
    "screen_quantiles": {q: float(np.quantile(ua["screen"], q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
    "total_view_seconds": float(views["duraton"].sum()),
    "seconds_on_labelled_airtime": labelled_seconds,
    "share_view_seconds_on_labelled_airtime": labelled_seconds / float(views["duraton"].sum()),
    "users_with_any_ge_0_8_pair": int(ua.loc[ua["screen"] >= 0.8, "u"].nunique()),
    "runtime_s": round(time.time() - t0, 1),
})
(RES / "join_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))
