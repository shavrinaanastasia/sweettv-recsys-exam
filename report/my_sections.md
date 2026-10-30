%% ============================================================================
%% YOUR SECTIONS: edit this file only. Lines starting with %% are notes for you;
%% they never appear in the PDF.
%% - Rewrite §5 in your own words. Keep every number exactly as in the tables.
%% - Replace each <<TODO: ... >> block completely (including << and >>).
%% - **bold**, *italic*; a line starting with "- " is a bullet; keep "##"/"###" headings.
%% - Save as UTF-8 (Notepad default), then tell Claude to rebuild the PDF.
%% ============================================================================

## 5. Discussion

### 5.1 Which recommendation strategy would we select for sweet.tv aand why?

Here we try to predict something that the user will WATCH, not something the user will LIKE. It's important to notice, because a person can watch something regularly as a habit. We can see on our examples that we are able to predict much better already watched content for an active user (AP@5=1.0), than new content for a less active one. Our models work better with a reoccuring data. 

Let's also compare the methods: popularity already has a good result because a lot of users watch the same news and other kinds of regular shows. ALS tries to find more hidden similarities between users. PPR takes into account personal history really good (PPR 0.32 MAP@5 VS ALS 0.29 MAP@5). It happens because a random walk through the graph regularly returns to the programs of a particular user. 

Content filtration is somehow worse than popularity: 0.18 VS 0.22. It is probably due to the fact that the similarity between genres and metadata does not indicate than a person will watch both because they can be on different channels and aat different time.

The best signal here is channel exposure (the info which channels this user regularly watches). If we delete it, our hybrid model quality will drop by 0.019 MAP@5 (on half B). I suppose it is becuase people usually turn on their favourite channels. 

That is why I've chosen a hybrid model with user history, popularity, ALS, PPR, content metadata and channel exposure + a LightGBM re-ranker on top. On validation my model gets MAP@5 =0.35, on Kaggle private 0.43. It is also cheap to compute at this scale: the whole test run takes about 3 minutes on an ordinary 8-core CPU.

But this method is not perfect because only 10.8% of the timetable has tv_show_id != 0. And also watching != being interested and != liking. So we predict that a person will watch something, not that they will like it. Also, if a programme is absent from the log, it does not mean the user never saw it: the log covers only this platform.

### 5.2 Failure case, root cause, fix

The biggest technical problem was with the Personalized PageRank. At first, the validation was 452.8s, but it was too long for finding hyperparameters. So I changed it to 50 strongest connections in the graph, but MAP@5 dropped from 0.32 to 0.28. The problem was in the graph structure: it was almost dense, 71% pairs of the programs (among the 2,902 programs that had any viewer in weeks 11-20) shared a user. That's why we didn't get big advantage. And also for PageRank weak connection also have some useful info and we deleted it. In the end, I decided to use the whole graph, but add dense matrix float32 and optimized multiplication (BLAS). The MAP@5 didn't drop and the time became 141s.

There was also a small problem with LightGBM: the first reranking didn't work because of the dependancy on scikit-learn. I moved to API lgb.train.

Log excerpts:
- `results/val_run1_stopped.log`: `"method": "ppr", "params": {"restart": 0.15, "min_sec": 300}, "seconds": 452.8`
- `results/rerank_val_console_failed_sklearn.log`: `lightgbm.basic.LightGBMError: scikit-learn is required for lightgbm.sklearn.`

### 5.3 What worked, what surprised me, next step

1) the validation seems reliable (80% threshold + check on the future weeks). The ranking of the algorithms is the same as the Kaggle private one.

2) those 89% of the data without tv_show_id where not thrown away: I used it for channel exposure. It was very helpful. 

3) I was a bit surprised that content filtration is worse than popularity??? But then I started thinking about typical TV watcher and understood the reasons. 

4) It is also interesting that ALS worked best with less factors and strong regularisation (30). 

5) It is also interesting that all methods on Kaggle private have bettet scores on MAP@5. it is probably due to the longer history. 

6) Due to the 5 we probably need to check how the length of the history affects our quality. And also we can add re-ranker candidates from the user's favourite channels.

## 6. AI Usage Disclosure

- **Tool:** Claude Code in the Claude desktop app, model Claude Opus 5.5 (`claude-opus-5-5`); one continuous session, exported unmodified as the attached session log.
- **How AI was used:** reading the exam brief, slides and course guidelines (docs); planning; writing all code in `src/` (code generation); running every experiment, and the Kaggle submissions after my explicit approval; debugging (PPR runtime, LightGBM wrapper error, memory blow-up in a feature); searching and checking references (search); figures (viz); drafting this report and the README.
- **What I personally verified:** hand-computed AP@5 for two lists — the heavy viewer's popularity list (0.6433, Table 2) and the light viewer's final re-ranker list (0.4, Table 3), both matching the code (Figure 3); re-ran `src/checks.py` and got output identical to `results/checks.json`; re-ran the final model from scratch (`python rerank.py test 150`) and got a byte-identical `submission_rerank.csv` (same SHA-256, `results/reproduction_check.md`); compared all six Kaggle scores on the Submissions page with Table 1; read the label recipe on the competition's Data tab and compared it with §3; opened the reference links and checked titles, authors and years; checked the submission file format against Kaggle's template; opened the repository while logged out to confirm it is public.
- **What I trusted without verification:** methods implementation; the scoring code beyond the two hand-checked AP@5 values; the data analysis beyond the re-run checks.
- **Reference check:** every reference was checked against Crossref metadata or the publisher's page on 2026-10-30 (`notes/reference_check.md`); the 1999 PageRank technical report was dropped because its server returned an error.
- **Session log:** the attached session export is the audit trail for all of the above.
