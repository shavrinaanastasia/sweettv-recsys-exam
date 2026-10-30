# Reference verification (2026-10-30)

Every reference in the report was checked against a primary source before use.
DOIs: Crossref REST API (`https://api.crossref.org/works/<doi>`), which returns the registered metadata.
Publisher pages for IEEE Xplore and ACM DL showed bot checks in the automated browser, so Crossref was used for them.

| # | Reference | Checked at | Result |
|---|---|---|---|
| 1 | ogurbych, sweet.tv — TV program recommender, Kaggle, 2020 | kaggle.com/competitions/sweettv-tv-program-recommender (Overview + Data tabs) | page loads; citation text shown on Overview |
| 2 | Y. Hu, Y. Koren, C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM 2008, pp. 263-272, doi:10.1109/ICDM.2008.22 | Crossref | title, authors, venue, pages match |
| 3 | T. H. Haveliwala, "Topic-sensitive PageRank," WWW 2002, pp. 517-526, doi:10.1145/511446.511513 | Crossref | match |
| 4 | M. J. Pazzani, D. Billsus, "Content-Based Recommendation Systems," The Adaptive Web, LNCS 4321, pp. 325-341, 2007, doi:10.1007/978-3-540-72079-9_10 | Crossref + link.springer.com chapter page (© 2007) | match |
| 5 | R. Agrawal, T. Imieliński, A. Swami, "Mining association rules between sets of items in large databases," SIGMOD 1993, pp. 207-216, doi:10.1145/170035.170072 | Crossref | match |
| 6 | G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 30 (2017) | proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html | page loads, title + 8 authors match |
| 7 | C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An Overview," Microsoft Research, MSR-TR-2010-82, June 2010 | microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/ | page loads, match |

Dropped: Page et al. 1999 PageRank tech report — the Stanford InfoLab server returned "Failed to connect to database" on 2026-10-30, so the link could not be confirmed.
