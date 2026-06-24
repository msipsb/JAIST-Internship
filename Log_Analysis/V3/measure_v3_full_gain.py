"""Measure the incremental value of Change B (trajectory) and Change A (tells).

Loads the full v3 cache (11 V2 + 13 trajectory + 6 Change-A columns) and compares
feature sets with the same models the notebook uses (§5b per-game RF, §8 held-out
LDA), so we can attribute the gain to B, to A, and to both.

Run:  py -3 Log_Analysis/V3/measure_v3_full_gain.py
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playstyle_log_parse_v3_full import build_frames, METRICS, TRAJ_METRICS, A_FEATURES, BASE_DIR

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

STYLE_ORDER = ["aggro", "control", "fatigue", "midrange", "ramp"]
CACHE = os.path.join(BASE_DIR, "playstyle_log_metrics_RenoKazakusMage_v3_full.pkl")
games_df, _, _ = build_frames(cache=CACHE, deck="RenoKazakusMage")
print(f"games_df {games_df.shape}\n")

B, A = list(TRAJ_METRICS), list(A_FEATURES)
FEATURE_SETS = {
    "baseline · 11 V2":            list(METRICS),
    "+B trajectory · 24":          list(METRICS) + B,
    "+A tells · 16":               list(METRICS) + A,
    "full A+B · 29":               list(METRICS) + B + A,
}


def per_game_rf(features):
    X = games_df[features].copy(); X = X.fillna(X.median())
    y = games_df["style"].astype(str).values
    Xs = StandardScaler().fit_transform(X.values)
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    pred = cross_val_predict(rf, Xs, y, cv=5)
    cm = confusion_matrix(y, pred, labels=STYLE_ORDER, normalize="true")
    return accuracy_score(y, pred), dict(zip(STYLE_ORDER, np.diag(cm)))


def games_to_classify(features, thr=0.90, Ns=(1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120)):
    G = games_df.copy(); G[features] = G[features].fillna(G[features].median())
    rng = np.random.default_rng(1)
    tr_idx, te_idx = [], []
    for st in STYLE_ORDER:
        idx = G.index[G["style"] == st].to_numpy().copy(); rng.shuffle(idx)
        h = len(idx) // 2; tr_idx += list(idx[:h]); te_idx += list(idx[h:])
    tr, te = G.loc[tr_idx], G.loc[te_idx]
    sc = StandardScaler().fit(tr[features].values)
    Xtr, Xte = sc.transform(tr[features].values), sc.transform(te[features].values)
    ytr, yte = tr["style"].astype(str).values, te["style"].astype(str).values
    trby = {s: Xtr[ytr == s] for s in STYLE_ORDER}
    teby = {s: Xte[yte == s] for s in STYLE_ORDER}

    def make_fps(by, N, reps, seed):
        r = np.random.default_rng(seed); Xs, ys = [], []
        for si, s in enumerate(STYLE_ORDER):
            a = by[s]
            for _ in range(reps):
                Xs.append(a[r.integers(0, len(a), N)].mean(0)); ys.append(si)
        return np.array(Xs), np.array(ys)

    rows = []
    for N in Ns:
        Xa, ya = make_fps(trby, N, 500, 10)
        lda = LinearDiscriminantAnalysis().fit(Xa, ya)
        Xb, yb = make_fps(teby, N, 500, 20)
        p = lda.predict(Xb)
        row = {"N": N, "overall": (p == yb).mean()}
        for si, s in enumerate(STYLE_ORDER):
            row[s] = (p[yb == si] == si).mean()
        rows.append(row)
    acc = pd.DataFrame(rows).set_index("N")

    def first_N(col):
        hit = acc.index[acc[col] >= thr]
        return int(hit[0]) if len(hit) else f">{Ns[-1]}"

    lda1 = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    cm1 = confusion_matrix(yte, lda1.predict(Xte), labels=STYLE_ORDER, normalize="true")
    one_game = float(np.trace(cm1) / 5)
    return one_game, {s: first_N(s) for s in ["overall"] + STYLE_ORDER}


print("=" * 78)
print("§5b · per-game RandomForest 5-fold CV accuracy  (chance = 0.20)")
print("=" * 78)
rf = pd.DataFrame(
    [{"feature set": n, "overall": (r := per_game_rf(f))[0], **r[1]} for n, f in FEATURE_SETS.items()]
).set_index("feature set")
print(rf.round(3).to_string())

print("\n" + "=" * 78)
print("§8 · held-out LDA — single-game accuracy & smallest N to reach 90% per style")
print("=" * 78)
for name, feats in FEATURE_SETS.items():
    one, n90 = games_to_classify(feats)
    print(f"\n{name}")
    print(f"  single-game accuracy : {one:.3f}")
    print("  N to 90%             : " + " | ".join(f"{k}: {n90[k]}" for k in ["overall"] + STYLE_ORDER))
