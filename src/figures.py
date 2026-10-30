"""Report figures -> report/fig_screen_time.png, report/fig_scores.png (numbers read from results/)."""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CACHE, RES, ROOT

OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, ACCENT, GRID = "#1f2933", "#7b8794", "#2f6fb0", "#e4e7eb"

# Fig. 1: distribution of screen time (overlap / programme duration) over all user-airing pairs.
ua = pd.read_parquet(CACHE / "user_airing.parquet", columns=["screen"])
s = ua["screen"].values
fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.hist(s, bins=50, range=(0, 1), color=ACCENT, edgecolor="white", linewidth=0.3)
ax.axvline(0.8, color=INK, linestyle="--", linewidth=1)
share = float((s >= 0.8).mean())
ax.text(0.79, ax.get_ylim()[1] * 0.85, f"0.8 threshold\n{share:.1%} of pairs ≥ 0.8", ha="right", color=INK, fontsize=7)
ax.set_xlabel("screen time (watched s / programme duration)")
ax.set_ylabel("user-airing pairs")
ax.grid(axis="y", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_screen_time.png", dpi=300)

# Fig. 2: validation MAP@5 (half B) vs Kaggle private MAP@5 per method.
single = json.loads((RES / "hybrid_val.json").read_text(encoding="utf-8"))["single"]
hyb = json.loads((RES / "hybrid_val.json").read_text(encoding="utf-8"))["hybrid"]
rr = json.loads((RES / "rerank_val_n150.json").read_text(encoding="utf-8"))["configs"]["small"]
with open(RES / "kaggle_submissions.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(line for line in f if line.strip())]
priv = {r["fileName"].replace("submission_", "").replace(".csv", ""): float(r["privateScore"]) for r in rows}
methods = [("CBF", single["cbf"]["B"], priv["cbf"]), ("Popularity", single["popularity"]["B"], priv["popularity"]),
           ("ALS", single["als"]["B"], priv["als"]), ("PPR", single["ppr"]["B"], priv["ppr"]),
           ("Hybrid", hyb["B"], priv["hybrid"]), ("Hybrid +\nre-ranker", rr["B"], priv["rerank"])]
x = np.arange(len(methods))
fig, ax = plt.subplots(figsize=(3.4, 2.0))
b1 = ax.bar(x - 0.19, [m[1] for m in methods], 0.36, color=MUTED, label="validation (half B)")
b2 = ax.bar(x + 0.19, [m[2] for m in methods], 0.36, color=ACCENT, label="Kaggle private")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=5.5, color=INK)
ax.set_xticks(x, [m[0] for m in methods], fontsize=6.5)
ax.set_ylabel("MAP@5")
ax.set_ylim(0, 0.5)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.grid(axis="y", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_scores.png", dpi=300)
json.dump({"share_pairs_ge_0_8": share, "methods": methods}, open(RES / "figures_data.json", "w"), indent=1)
print("share >= 0.8:", round(share, 4))
for m in methods:
    print(m)
