"""Turn results/examples.json into report/examples_tables.md (ids copied from code output, not by hand)."""
import json

from common import RES, ROOT

ex = json.loads((RES / "examples.json").read_text(encoding="utf-8"))
order = ["popularity", "history (reference)", "M1 ALS", "M2 content-based", "M3 PPR", "fixed-weight hybrid",
         "FINAL: hybrid + re-ranker"]
short = {"popularity": "Popularity", "history (reference)": "History (ref.)", "M1 ALS": "M1 ALS",
         "M2 content-based": "M2 CBF", "M3 PPR": "M3 PPR", "fixed-weight hybrid": "Hybrid (fixed)",
         "FINAL: hybrid + re-ranker": "**Final: re-ranker**"}
out, titles = [], {}
for n, (role, e) in enumerate(ex.items(), start=1):
    actual = [d["tv_show_id"] for d in e["actual_21_30"]]
    for d in e["actual_21_30"] + [x for m in e["methods"].values() for x in m["top5"]]:
        titles[d["tv_show_id"]] = (d["title"], d["category"])
    out.append(f"| Method | #1 | #2 | #3 | #4 | #5 | AP@5 |")
    out.append("|---|---|---|---|---|---|---|")
    out.append("| *Actual 21-30* | " + " | ".join(f"*{i}*" for i in actual) + " | |")
    for m in order:
        r = e["methods"][m]
        cells = []
        for d in r["top5"]:
            c = str(d["tv_show_id"]) + ("" if d["in_train_history"] else "ⁿ")
            cells.append(f"**{c}**" if d["in_actual"] else c)
        out.append(f"| {short[m]} | " + " | ".join(cells) + f" | {r['AP@5']:.3f} |")
    out.append(f"Table {n + 3}: {role.split()[0].capitalize()} viewer, user {e['user_id']} "
               f"({e['train_airings_watched_ge_0_8']} airings watched ≥ 80% in weeks 11-20, "
               f"{e['distinct_programmes_watched_ge_0_8']} distinct programmes).")
    out.append("")
# two side-by-side halves to save vertical space
ids = sorted(titles)
half = (len(ids) + 1) // 2
out.append("| tv_show_id | Title (from schedule) | Category | tv_show_id | Title (from schedule) | Category |")
out.append("|---|---|---|---|---|---|")
for a, b in zip(ids[:half], ids[half:] + [None]):
    right = f"{b} | {titles[b][0]} | {titles[b][1]}" if b else " | | "
    out.append(f"| {a} | {titles[a][0]} | {titles[a][1]} | {right} |")
out.append("Table 6: Titles and categories of every programme id in Tables 4-5, looked up from the schedule.")
(ROOT / "report" / "examples_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
