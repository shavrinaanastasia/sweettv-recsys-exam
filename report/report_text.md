<<META Student: Anastasia Shavrina (Шаврина Анастасия Олеговна)>>
<<META Team: Individual>>
<<META Email: shavrina157@gmail.com>>
<<META Date: 2026-10-30>>
<<META Assignment: Final exam quiz (FKN) — TV programme recommendation on sweet.tv (Kaggle account: Maizawitch)>>
<<META Code: https://github.com/shavrinaanastasia/sweettv-recsys-exam>>
# LLM4Rec Final Exam Report — Repeat Viewing Dominates: TV Programme Recommendation on sweet.tv

## Abstract

We predict, for 1,260 sweet.tv users, the five TV programmes they will watch most often in weeks 31-42, given a channel-level view log (weeks 11-30) in which only about 11% of schedule rows carry a programme id. We join views to labelled airings with the host's own "≥ 80% watched" rule, compare a popularity baseline with implicit ALS, content-based filtering and personalized PageRank on one temporal hold-out, and combine them by score fusion followed by a LightGBM LambdaRank re-ranker. The final strategy reaches MAP@5 = 0.352 on untouched validation users (popularity: 0.224) and a returned Kaggle private score of 0.43344 (popularity: 0.28773). The main lesson is that this target rewards predicting repeat viewing on a user's habitual channels; content similarity alone does not even beat popularity.

*Index Terms — implicit feedback, personalized PageRank, matrix factorization, learning to rank, TV recommendation*

## 1. Introduction

**Problem.** sweet.tv logs which channel each user watched and when, and publishes the broadcast grid separately. For each of 1,260 test users we must output five programme ids, ordered, that best match the five programmes the user will watch most often (≥ 80% of an airing) in the next 12 weeks; the metric is MAP@5 [1].

**Motivation.** A TV service's home screen has room for a handful of tiles. If they reflect what a viewer will really come back to (the evening news, a children's series) the screen saves effort; if not, it is ignored.

**Concrete example.** User 8377619604347126107 watched 179 airings to ≥ 80% in weeks 11-20, spread over 22 programmes. In weeks 21-30 their five most-watched programmes were 'ТСН. Неделя', 'Сегодня. Итоги', 'Факты. День', 'Завтрак с 1+1' and a 'Свинка Пеппа' series id — all five already in their history. Personalized PageRank returns exactly these five (AP@5 = 1.0, Table 2).

**Contributions.**
- A reproducible join that applies the host's label recipe, and a validation set built with the same recipe.
- A like-for-like comparison of four methods plus a channel-exposure signal, and a two-stage hybrid whose gain over every single method holds both on held-out validation users and on the Kaggle private split.

## 2. Related Work

Prior work consulted:

- **Implicit-feedback matrix factorization** [2] turns counts into binary preference with a confidence weight and solves it by alternating least squares; our M1 implements those update equations directly.
- **Personalized / topic-sensitive PageRank** [3] biases the random-walk restart vector towards a chosen set of nodes; our M3 restarts on each user's own history over a programme co-view graph.
- **Content-based recommendation** [4] builds item vectors from attributes and a user profile from liked items; our M2 uses TF-IDF over category, genres, decade, director and actor tokens.
- **Learning to rank**: LambdaMART [7] optimises a ranking metric through gradient-boosted trees; we use the LightGBM implementation [6] for the re-ranking stage.

*Alternatives considered.* Association rules [5] were not implemented: the PPR co-view graph, whose edge weight is the number of users who watched both programmes, already encodes the pairwise co-occurrence counts that pairwise rules are built from; whether rule confidence or lift would add to it was not tested. Sequence models (session RNN / transformers) were not tried: views are channel intervals with long gaps rather than ordered item clicks, and only 20 weeks of history exist. A two-tower model was not tried: the only side information is the metadata already used by M2, so it would mostly re-learn the ALS factors at a higher tuning cost (not tested).

## 3. Method

**Approach.** Turn the view log into (user, programme, number of airings watched ≥ 80%) interactions, score every candidate programme with each method, and rank. Every method shares one candidate pool, one fallback and one evaluation protocol. The final system uses the fixed-weight score fusion of all signals as a candidate generator (top 150 per user) and re-orders those candidates with a LambdaRank model.

**Pipeline.**
- *Join.* A view (user, channel, [start, stop]) is matched to a schedule row on the same channel whose airing [start, start + duration] overlaps it. Overlap seconds of all of the user's views are summed per airing and capped at the airing duration. *Watched* means overlap / duration ≥ 0.8 — exactly the host's label rule [1]. Figure 1 justifies this cut on the data as well: screen time is U-shaped (zapping near 0, full watches at 1), and 0.8 sits in the flat valley: moving the cut to 0.7 or 0.9 changes the number of qualifying pairs by +15.6% / −14.5% (`results/report_stats.json`). 20.4% of 3,014,129 user-airing pairs pass.
- *`tv_show_id = 0`.* Unlabelled airtime is never an item (the host removed it from the labels). It is used only as context: each user's viewing seconds per channel, over **all** airtime, feed the channel-exposure signal below.
- *Labels* (validation): per user, count airings watched ≥ 0.8 per programme in the target weeks, keep users with ≥ 5 such programmes, take the 5 most frequent (ties: more watched seconds).
- *Candidate pool:* labelled programmes that air in the target weeks — for the test, read from the weeks 31-42 *schedule* only. Re-recommending a programme the user already watched is **allowed for every method** (the target is repeat-heavy, and the host does not forbid it).
- *Fallback:* empty slots are filled from the popularity ranking, identically for every method, because popularity needs no personal history and is the strongest non-personal signal we have. Each method turns its scores into a list by sorting its own scores inside the pool; scores are never compared across methods without per-user min-max normalisation.

![Figure 1: Screen time of all 3,014,129 user-airing pairs (weeks 11-30). The 0.8 cut is the host's label rule.](fig_screen_time.png){w=6.4}

**Methods.**
- *Popularity (baseline):* recency-weighted number of airings of the programme watched ≥ 0.8 by all users in the train weeks (half-life 4 weeks).
- *History (reference, not a counted method):* the user's own count of ≥ 0.8 airings per programme.
- *M1 implicit ALS* [2]: r = recency-decayed counts, confidence c = 1 + α r; 16 factors, λ = 30, α = 1, 10 iterations, seed 42.
- *M2 content-based* [4]: TF-IDF item vectors over 17,193 metadata tokens; user profile = sum of watched items weighted by recency-decayed ≥ 0.8 counts; cosine score.
- *M3 personalized PageRank* [3]: item graph with edge weight = number of users who watched both items ≥ 20 min; row-normalised; restart probability 0.3 to a vector built from the user's recency-decayed ≥ 0.8 counts plus 0.1 · log(1 + minutes watched) for every programme they touched (so partial watches also count), normalised to sum 1; 20 power iterations.
- *Channel exposure (context signal):* Σ over channels of (user's share of viewing seconds on channel c) × (airings of the item on c in the target weeks).
- *Hybrid:* per-user min-max normalised scores, weighted sum; weights by coordinate ascent on validation half A: channel 2.0, PPR 1.5, history 0.75, ALS 0.5, CBF 0.1, popularity 0.05.
- *Final: hybrid + re-ranker:* top 150 fused candidates per user, re-ordered by LightGBM LambdaRank [6], [7] (300 trees, learning rate 0.05, 31 leaves, ≥ 100 rows per leaf, bagging 0.8, feature fraction 0.8, seed 42, deterministic) on 31 features: every component's raw score, normalised score and rank, fused score and rank, watch counts and seconds, target/train airing counts and their ratio, user activity. Relevance = 5 for the label's #1 down to 1 for #5.

**Key design decisions.** (1) *Use the host's 0.8 rule rather than an absolute-seconds cut,* because the score is computed on labels built with it; an absolute cut would mislabel long films and short news equally. (2) *Fit the re-ranker without validation labels:* for validation it learns history 11-15 → labels 16-20 and is then scored on 11-20 → 21-30; for the test it learns 11-20 → 21-30 (training data at that point) and is applied to 11-30 → 31-42. (3) *Keep PPR's full graph* (dense matrix product): pruning to the 50 strongest edges per item cost 0.013-0.056 MAP on half A at restart 0.15-0.3; at restart 0.5 pruning was 0.004-0.005 better, but those settings were worse overall (`results/val_runs.log`, §5).

**Tools & libraries.**
- Python 3.11.9; numpy 2.4.6 (https://numpy.org/doc/stable/); pandas 3.1.0 (https://pandas.pydata.org/docs/); scipy 1.17.1 sparse matrices (https://docs.scipy.org/doc/scipy/); pyarrow 26.0.0 for the parquet cache.
- LightGBM 4.7.0, native `lgb.train` API with the `lambdarank` objective (https://lightgbm.readthedocs.io/en/stable/).
- matplotlib 3.11.2 (figures), reportlab 5.1.0 (this PDF), Kaggle CLI 2.2.5 (download and submission).
- ALS and PPR are implemented from the equations in [2], [3] (`src/methods.py`), not from a library.

**Configuration.** Windows 11, 8 logical CPUs, 16 GB RAM, no GPU. Seeds: user A/B split 0, ALS 42, LightGBM 42. Runtime: parse 42 s, join 12 s; scoring the 1,260 test users: PPR 41 s, ALS 6 s, CBF 0.3 s, all fixed-weight submission files 52 s (`results/predict_summary.json`); re-ranker test run 183 s end-to-end.

## 4. Experiments

**Setup.** View log: 5,821,987 views by 4,914 users on 202 channels (weeks 11-30); schedule: 381,892 labelled airings. In weeks 11-20 only 10.8% of schedule rows (16.0% of airtime) carry an id (11.3% and 11.2% in the other two schedule files), yet 52.4% of all viewing seconds fall on that labelled airtime. *Validation split:* train weeks 11-20 → labels from weeks 21-30, giving 4,030 users with five labels (1,150 of them are Kaggle test users). The test target window is later than all training weeks, so this split preserves temporal order and has a similar history-to-horizon shape (10/10 weeks vs. 20/12 for the test). Users are split 50/50 at random (seed 0): every hyper-parameter and fusion weight is chosen on half A (2,048 users); half B (1,982 users) is reported untouched, mirroring the leaderboard's public/private user split [1]. Cold start: all 1,260 test users appear in the training log, so none is without views; 4 of them have no programme watched ≥ 0.8 in weeks 11-30 and 19 have fewer than five (`results/report_stats.json`). Such users get popularity in every empty slot; in the history-only list this happens for 22 users, counting only programmes that air in weeks 31-42 (`results/predict_summary.json`).

| Method | Val. A | Val. B (untouched) | Val. test users | Kaggle public | Kaggle private |
|---|---|---|---|---|---|
| Popularity (baseline) | 0.2226 | 0.2244 | 0.2486 | 0.30344 | 0.28773 |
| History (reference) | 0.3105 | 0.3029 | 0.3468 | — | — |
| M1 implicit ALS | 0.2890 | 0.2862 | 0.3120 | 0.37750 | 0.36917 |
| M2 content-based | 0.1871 | 0.1831 | 0.2036 | 0.22711 | 0.21839 |
| M3 personalized PageRank | 0.3239 | 0.3182 | 0.3610 | 0.39959 | 0.38530 |
| Channel exposure (signal) | 0.3227 | 0.3192 | 0.3477 | — | — |
| Hybrid, fixed weights | 0.3484 | 0.3420 | 0.3810 | 0.44153 | 0.42921 |
| **Hybrid + re-ranker (final)** | **0.3563** | **0.3524** | **0.3933** | **0.43978** | **0.43344** |
Table 1: MAP@5. Validation = train 11-20, labels 21-30; Kaggle = train 11-30, hidden labels 31-42. Sources: `results/hybrid_val.json`, `results/rerank_val_n150.json`, `results/kaggle_submissions.csv`.

![Figure 2: Validation MAP@5 (half B) and returned Kaggle private MAP@5 per method. The order of the methods is the same on both.](fig_scores.png){w=6.8}

**Comparison vs. baseline.** ALS (+0.062) and PPR (+0.094) beat popularity on half B; the final strategy gains +0.128 on half B and +0.146 on Kaggle private. Content-based filtering is the only method below the baseline (−0.041 on B). The methods rank in the same order on validation and on Kaggle private (Figure 2); Kaggle scores are higher overall. Two explanations, neither tested: test predictions use 20 instead of 10 weeks of history, and the user population differs — the 1,150 validation users who are also Kaggle test users already score higher than all validation users (Table 1, "Val. test users"), even with popularity.

**Hybrid and ablation.** Removing one component from the fixed-weight hybrid (half B, `results/hybrid_val_run1.json`): channel −0.019, PPR −0.004, history −0.002, CBF −0.001, popularity −0.000, ALS +0.000. A second weight search with a wider grid (up to 6.0) ended *lower* on half A (0.3477 vs. 0.3484) and was not used. The re-ranker was compared over two candidate depths and two model sizes on half A (150 candidates, small model best); recall of the label inside the 150 candidates is 0.767, which bounds what re-ranking can recover. The re-ranker costs more than the fixed hybrid (183 s for its test run vs. 52 s for all fixed-weight submission files) for +0.010 on half B, +0.004 on Kaggle private and −0.002 on Kaggle public.

**Tuning record.** ALS needed three grids because the best point sat on the grid edge in the first two (α = 5, then λ = 10); in the third, λ = 30 beats 10 and 100, and 16 factors beat 8 (32 factors were only tried at λ ≤ 10). At every matched setting of the first grid, weighting ALS by watched seconds instead of ≥ 0.8 counts gave MAP 0.02-0.06 instead of 0.15-0.20 (plausibly because zapping visits then dominate; not tested further). Every grid point, including the losing ones, is logged in `results/val_runs.log` and `results/val_log_*.json`.

**Verification.** (i) MAP@5 code reproduces the course slide's worked example (AP = 0.917 and 0.533). (ii) One user-airing overlap recomputed from the raw CSV with the standard `csv` module equals the join output (5,973 s of a 6,000 s film). (iii) All 4,030 validation label sets contain 5 distinct programmes, each watched ≥ 0.8 in weeks 21-30. (iv) ALS objective decreases every iteration (49,593 → 26,627 over 6 iterations; run on the first 600 users with 16 factors, α = 5, λ = 1). (v) PPR transition rows sum to 1, and each user's score vector keeps total mass 1 (run with restart 0.15 and ≥ 5-min co-views). (vi) Each submission file has 1,260 rows and five distinct non-zero ids from the test-week pool (assertions in `src/predict.py` and `src/rerank.py`), and was scored COMPLETE by Kaggle (`results/kaggle_submissions.csv`). (vii) The dense and sparse PPR implementations return the identical score (A = 0.32295) on the same graph (`results/val_run1_stopped.log` vs. `results/val_runs.log`). (viii) I computed AP@5 by hand for the heavy viewer's popularity list (Table 2), using AP@5 = (1 / min(5, |R|)) · Σₖ P@k · rel(k): the hits are at ranks 1, 3, 4 and 5 and |R| = 5, so AP@5 = (1/5)(1 + 2/3 + 3/4 + 4/5) = 0.6433, the same value the code reports (`results/examples.json`). I also computed the light viewer's final re-ranker list (Table 3): hits at ranks 1 and 2, so AP@5 = (1/5)(1/1 + 2/2 + 0 + 0 + 0) = 2/5 = 0.4, matching the table. My working is shown in Figure 3. Checks (i)-(v) are in `results/checks.json` (`src/checks.py`).

![Figure 3: My hand calculations of AP@5 (photos, cropped): heavy viewer's popularity list, 0.6433 (top); light viewer's final re-ranker list, 0.4 (bottom).](hand_check_ap5.png){w=6.5}

### 4.1 Recommendation examples

Users were chosen by a fixed rule, not by inspecting results: among validation users who are also Kaggle test users, the one closest to the 90th percentile of training activity (heavy) and the one closest to the 10th percentile (light). Lists come from the validation models (train 11-20), so they can be compared with what the users actually watched in weeks 21-30. Titles are the schedule text of the latest airing; one id can cover many titles: 12002955 appears under 207 episode titles of 'Свинка Пеппа', and 2400480 covers both the weekly 'ТСН. Неделя' and the daily 'ТСН.' news (`results/report_stats.json`). In Tables 2-3, **bold** = in the user's actual weeks 21-30 label; ⁿ = not watched ≥ 80% by this user in weeks 11-20.

<<INCLUDE examples_tables.md>>

### 4.2 Kaggle submission

![Figure 4: Kaggle Submissions page (late submissions, account Maizawitch). Designated final submission: submission_rerank.csv, returned private score 0.43344, public 0.43978. The popularity submission (private 0.28773) is below the visible area; all six scores are in Table 1 and results/kaggle_submissions.csv.](kaggle_screenshot.png){w=9.5}

The designated final submission is `submissions/submission_rerank.csv` (ref 57803755), chosen on validation before its Kaggle score was known.

<<INCLUDE my_sections.md>>


<<PAGEBREAK>>
## References

[1] ogurbych, "sweet.tv — TV program recommender," Kaggle community competition, 2020. [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender. [Accessed: 2026-10-30].

[2] Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 263-272, doi: 10.1109/ICDM.2008.22.

[3] T. H. Haveliwala, "Topic-sensitive PageRank," in *Proc. 11th Int. Conf. World Wide Web (WWW)*, 2002, pp. 517-526, doi: 10.1145/511446.511513.

[4] M. J. Pazzani and D. Billsus, "Content-based recommendation systems," in *The Adaptive Web*, LNCS, vol. 4321. Berlin, Germany: Springer, 2007, pp. 325-341, doi: 10.1007/978-3-540-72079-9_10.

[5] R. Agrawal, T. Imieliński, and A. Swami, "Mining association rules between sets of items in large databases," in *Proc. ACM SIGMOD Int. Conf. Management of Data*, 1993, pp. 207-216, doi: 10.1145/170035.170072.

[6] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html. [Accessed: 2026-10-30].

[7] C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An overview," Microsoft Research, Tech. Rep. MSR-TR-2010-82, Jun. 2010. [Online]. Available: https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/. [Accessed: 2026-10-30].
