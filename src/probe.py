"""Quick look at the validation set and two reference rankers before building the methods."""
import numpy as np

from common import load, build_labels, make_train, popularity_scores, top_k, map_at_k, SPLITS

ua, air, users, items, uc = load()
td = make_train(ua, air, uc, len(users), len(items), "val")
labels = build_labels(ua, SPLITS["val"][1])
print("val users with 5 labels:", len(labels), "| pool size:", len(td.pool))
lab_items = np.concatenate(list(labels.values()))
print("label items inside pool:", np.isin(lab_items, td.pool).mean())
cnt = td.cnt.toarray()
eval_u = np.array(sorted(labels))
print("label items the user already watched (>=0.8) in train:",
      np.mean([cnt[u, i] > 0 for u in eval_u for i in labels[u]]))

pop = popularity_scores(td)
fallback = td.pool[np.argsort(-pop[td.pool])]
P, _ = top_k(np.tile(pop, (len(eval_u), 1)), fallback)
print("popularity MAP@5:", map_at_k(dict(zip(eval_u, P)), labels))

for name, M in [("history count", td.cnt), ("history decayed", td.cnt_decay)]:
    S = M[eval_u].toarray()
    S[:, np.setdiff1d(np.arange(len(items)), td.pool)] = -np.inf
    P, f = top_k(S, fallback)
    print(f"{name} MAP@5:", map_at_k(dict(zip(eval_u, P)), labels), "filled:", f)
    S2 = S * td.pool_airings[None, :] / np.maximum(td.train_airings, 1)[None, :]
    P, f = top_k(S2, fallback)
    print(f"{name} x airing ratio MAP@5:", map_at_k(dict(zip(eval_u, P)), labels))
