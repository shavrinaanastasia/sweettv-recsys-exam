# How the hidden test labels were built (source: competition Data tab)

Source: https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data (read 2026-10-30).
Paraphrase, not a quote:

1. Schedule rows with `tv_show_id = 0` were removed.
2. For each user and each programme airing, screen time = seconds the user watched / programme duration.
   Only airings with screen time >= 0.8 count.
3. For each user, count how many such high-screen-time airings each programme has.
4. Each user's label = exactly 5 programmes with the highest count, ordered by count (most frequent first).
5. Public / private leaderboard split is 50/50 by users, not by time.
6. Window: weeks 31-42 (12 weeks). Metric: MAP@5. 10 submissions per day (Overview tab).

Consequences for this project:
- "Watched" rule = overlap / programme duration >= 0.8 (same as the label definition).
- Validation labels are built with the same recipe on weeks 21-30, users with >= 5 qualifying programmes.
- Every label set has exactly 5 items, so the MAP@5 denominator is min(5, 5) = 5.
- Tie-break inside the label recipe is not specified; we break ties by total watched seconds and say so.
