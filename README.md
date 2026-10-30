# sweet.tv TV-programme recommender — LLM4Rec final exam (FKN)

Anastasia Shavrina · HSE University · LLM4Rec, final exam quiz · Kaggle: [sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) (late submissions)

Predict, for each of 1,260 test users, the five TV programmes they will watch most often in weeks 31-42,
from a view log (weeks 11-30) and the broadcast schedule. Metric: MAP@5.

## Final result

**Designated final submission:** [`submissions/submission_rerank.csv`](submissions/submission_rerank.csv) — score-fusion hybrid as candidate generator + LightGBM LambdaRank re-ranker.

| Submission | Validation MAP@5 (half B, untouched) | Kaggle public | Kaggle private |
|---|---:|---:|---:|
| **Hybrid + re-ranker (final)** | **0.352** | 0.43978 | **0.43344** |
| Hybrid, fixed fusion weights | 0.342 | 0.44153 | 0.42921 |
| M3 Personalized PageRank | 0.318 | 0.39959 | 0.38530 |
| M1 Implicit ALS (matrix factorization) | 0.286 | 0.37750 | 0.36917 |
| Popularity baseline (reference) | 0.224 | 0.30344 | 0.28773 |
| M2 Content-based filtering | 0.183 | 0.22711 | 0.21839 |

Sources: `results/val_runs.log`, `results/hybrid_val.json`, `results/rerank_val_n150.json`, `results/kaggle_submissions.csv`.

## Key design decisions

- **Join rule:** same channel (`vsetv_id` = `channel_id`) and overlapping time intervals; overlap seconds of all of a
  user's views are summed per airing, capped at the airing duration. **Watched = overlap / programme duration ≥ 0.8**,
  the exact rule the host used to build the hidden labels (competition Data tab; see `notes/competition_target.md`).
- **`tv_show_id = 0`** rows are never items. Their airtime is used only as *context*: a user's viewing seconds per
  channel (incl. unlabelled airtime) feed the channel-exposure signal.
- **Validation:** train weeks 11-20 → labels from weeks 21-30 built with the host's recipe (4,030 users with ≥ 5
  qualifying programmes). Users split 50/50 (seed 0): hyper-parameters chosen on half A, half B reported untouched.
- **Candidate pool:** labelled programmes airing in the target weeks (test: the weeks 31-42 *schedule*; no test-period
  views are used anywhere). Re-recommending already-watched programmes is **allowed for every method**.
- **Fallback:** any empty slot is filled from the popularity ranking, identical for all methods.
- **Re-ranker fitting never touches validation labels:** validated by fitting on 11-15 → 16-20; the test model is fit on
  11-20 → 21-30 and applied to 11-30 → 31-42.

## Reproduce

```bash
pip install -r requirements.txt
kaggle competitions download -c sweettv-tv-program-recommender -p data   # accept the rules first
cd data && tar -xf sweettv-tv-program-recommender.zip && cd ..
cd src
python prep.py          # CSV -> parquet cache (~40 s)
python eda.py           # results/eda.json
python join.py          # view x airing join (~12 s), results/join_stats.json
python validate.py pop hist cbf ppr als als2 als3   # grids, appends results/val_runs.log
python hybrid.py        # fusion weights on half A -> results/hybrid_val.json
python predict.py       # one submission per method + fixed-weight hybrid
python rerank.py val 150 && python rerank.py test 150   # re-ranker validation + final submission
python examples.py      # two example users -> results/examples.md
python checks.py        # sanity checks -> results/checks.json
```

Environment: Windows 11, Python 3.11.9, 8 logical CPUs, 16 GB RAM. Seeds: ALS 42, user A/B split 0, LightGBM 42
(deterministic mode). Runtimes are logged in the `results/*.json` files.

## Repository layout

```
src/          prep, eda, join, common (labels, pool, MAP@5), methods (ALS, CBF, PPR), validate, hybrid,
              rerank, predict, examples, checks
results/      every logged experiment, Kaggle score export, sanity checks
submissions/  all submitted files; submission_rerank.csv is the designated final one
notes/        competition label definition (paraphrased from the Data tab)
```

## AI assistance

The code, experiments and report draft were produced with Claude Code (Claude Opus 5.5) under my direction; the
full, unmodified session log is submitted with the report. See §6 of the report for what I verified myself.
