# Reproduction check (done by the author, 2026-10-30)

- Re-ran `python checks.py`: output identical to the committed `results/checks.json` (git showed no change).
- Re-ran `python rerank.py test 150` at 16:20: the rebuilt `submissions/submission_rerank.csv` has the same
  SHA-256 as the submitted file (C80DF2A4221F98BC6417A26A3EED0112D8CA6CC19C44863C7C4DDE718CC3B902),
  i.e. byte-identical. Only the runtime differed (172.5 s vs. 183.2 s in `results/rerank_test.json`, which was kept).
