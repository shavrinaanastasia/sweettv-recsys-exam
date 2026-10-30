"""Share of viewing seconds by start hour (all views, weeks 11-30) -> results/viewing_by_hour.json.
Used in §5 to support "people mostly watch TV in the evening"."""
import json

import numpy as np
import pandas as pd

from common import CACHE, RES

v = pd.read_parquet(CACHE / "views.parquet", columns=["start", "duraton"])
hour = (v["start"].values // 3600) % 24
sec = np.bincount(hour, weights=v["duraton"].values, minlength=24)
share = sec / sec.sum()
out = {"share_by_start_hour": {int(h): round(float(s), 4) for h, s in enumerate(share)},
       "share_17_to_24": round(float(share[17:24].sum()), 4), "share_06_to_12": round(float(share[6:12].sum()), 4),
       "peak_hour": int(share.argmax())}
(RES / "viewing_by_hour.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out["share_17_to_24"], out["share_06_to_12"], out["peak_hour"])
