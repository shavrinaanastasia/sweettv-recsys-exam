"""Parse the raw competition CSVs once and cache them as parquet.

Times are kept as int64 seconds since epoch (naive local time as written in the files).
ISO week numbers are added because the competition splits data by week (11-20, 21-30, 31-42).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

SCHEDULE_FILES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}


def to_seconds(ts: pd.Series) -> np.ndarray:
    return ts.values.astype("datetime64[s]").astype(np.int64)


def iso_week(ts: pd.Series) -> np.ndarray:
    return ts.dt.isocalendar().week.astype(np.int16).values


def prep_views():
    t0 = time.time()
    v = pd.read_csv(
        DATA / "dataset11-30.csv",
        dtype={"user_id": np.uint64, "vsetv_id": np.int32, "duraton": np.int64},
    )
    start = pd.to_datetime(v["start_time"], format="%Y-%m-%d %H:%M:%S")
    stop = pd.to_datetime(v["stop_time"], format="%Y-%m-%d %H:%M:%S")
    out = pd.DataFrame({
        "user_id": v["user_id"].values,
        "channel_id": v["vsetv_id"].values,
        "start": to_seconds(start),
        "stop": to_seconds(stop),
        "duraton": v["duraton"].values,
        "week": iso_week(start),
    })
    out.to_parquet(CACHE / "views.parquet", index=False)
    print(f"views: {len(out):,} rows in {time.time() - t0:.1f}s")


def prep_schedule():
    t0 = time.time()
    parts = []
    for tag, fname in SCHEDULE_FILES.items():
        s = pd.read_csv(DATA / fname, dtype=str, keep_default_na=False)
        start = pd.to_datetime(s["start_time"], format="%d.%m.%Y %H:%M:%S")
        part = pd.DataFrame({
            "channel_id": s["channel_id"].astype(np.int32).values,
            "channel_title": s["channel_title"].values,
            "start": to_seconds(start),
            "duration": s["duration"].astype(np.int64).values,
            "tv_show_id": s["tv_show_id"].astype(np.int64).values,
            "title": s["tv_show_title"].values,
            "category": s["tv_show_category"].values,
            "genre_1": s["tv_show_genre_1"].values,
            "genre_2": s["tv_show_genre_2"].values,
            "genre_3": s["tv_show_genre_3"].values,
            "year": s["year_of_production"].values,
            "director": s["director"].values,
            "actors": s["actors"].values,
            "week": iso_week(start),
            "file": tag,
        })
        part["end"] = part["start"] + part["duration"]
        print(f"schedule {tag}: {len(part):,} rows")
        parts.append(part)
    sched = pd.concat(parts, ignore_index=True)
    sched.to_parquet(CACHE / "schedule.parquet", index=False)
    print(f"schedule total: {len(sched):,} rows in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    prep_views()
    prep_schedule()
