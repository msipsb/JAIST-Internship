"""Measure whether V4's deck-agnostic, turn-rate-normalized features **transfer
across decks** better than the V3-full feature set.

The V3 study lived on one deck (RenoKazakusMage). "Deck-agnostic" only means
something across decks, so V4 trains on one deck family and tests on the other:

    * within-deck RF (5-fold CV)         -> reference: V4 ~= V3 on its home deck
    * cross-deck RF (train A -> test B)  -> the headline: does the fingerprint
                                            survive a deck swap?
    * cross-deck aggregate LDA           -> how many games must you pool before a
                                            held-out *other-deck* style is callable

Compared feature sets (all columns live in the V4 frame, a superset of V3):
    V3-full (deck-leaky) : 11 V2 + 13 trajectory + 5 Change-A tells  (incl.
                           avg/max card cost, raw mana — these leak deck identity)
    V4 agnostic          : pure behavioral ratios + the 4 universal currency
                           metrics + normalized trajectory  (no cost/length leakage)

Run:  py -3 Log_Analysis/V4/measure_v4_gain.py
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playstyle_log_parse_v4 import (
    build_frames, BASE_DIR, DECKS, STYLES,
    METRICS, TRAJ_METRICS, A_FEATURES, AGNOSTIC_FEATURES, RATIO_FEATURES, DECK_DEP_FEATURES,
)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

STYLE_ORDER = STYLES
CACHE = os.path.join(BASE_DIR, "playstyle_log_metrics_v4_bothdecks.pkl")

games_df, _, _ = build_frames(cache=CACHE, deck=DECKS)
print(f"games_df {games_df.shape}  decks={sorted(games_df['deck'].unique())}\n")

V3_FULL  = list(METRICS) + list(TRAJ_METRICS) + list(A_FEATURES)
V4_AGN   = list(AGNOSTIC_FEATURES)
V4_RATIO = list(RATIO_FEATURES)
FEATURE_SETS = {
    f"V3-full  (deck-leaky, {len(V3_FULL)})":  V3_FULL,
    f"V4 agnostic ({len(V4_AGN)})":            V4_AGN,
    f"V4 pure-ratio ({len(V4_RATIO)})":        V4_RATIO,
}


def _xy(df, features, fill_from=None):
    """Feature matrix + label vector; NaNs imputed from `fill_from` (defaults to df)."""
    src = df if fill_from is None else fill_from
    med = src[features].median()
    X = df[features].fillna(med).values
    y = df["style"].astype(str).values
    return X, y, med


def within_deck_rf(features, deck):
    """RF 5-fold CV accuracy on a single deck (reference: home-deck performance)."""
    d = games_df[games_df["deck"] == deck]
    X, y, _ = _xy(d, features)
    Xs = StandardScaler().fit_transform(X)
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    return accuracy_score(y, cross_val_predict(rf, Xs, y, cv=5))


def cross_deck_rf(features, train_deck, test_deck):
    """Fit RF on `train_deck`, score on `test_deck`. Scaler + medians come from train."""
    tr = games_df[games_df["deck"] == train_deck]
    te = games_df[games_df["deck"] == test_deck]
    Xtr, ytr, med = _xy(tr, features)
    Xte, yte, _   = _xy(te, features, fill_from=tr)
    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(sc.transform(Xtr), ytr)
    pred = rf.predict(sc.transform(Xte))
    cm = confusion_matrix(yte, pred, labels=STYLE_ORDER, normalize="true")
    return accuracy_score(yte, pred), dict(zip(STYLE_ORDER, np.diag(cm).round(2)))


def cross_deck_aggregate(features, train_deck, test_deck,
                         thr=0.90, Ns=(1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120)):
    """Train LDA on N-game *fingerprints* from train_deck; test on test_deck fingerprints.

    Returns single-game cross-deck accuracy and the smallest N (games pooled) at
    which each held-out other-deck style is recognised >= thr of the time.
    """
    tr = games_df[games_df["deck"] == train_deck]
    te = games_df[games_df["deck"] == test_deck]
    Xtr, ytr, med = _xy(tr, features)
    Xte, yte, _   = _xy(te, features, fill_from=tr)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
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
    one_game = float(np.trace(cm1) / len(STYLE_ORDER))
    return one_game, {s: first_N(s) for s in ["overall"] + STYLE_ORDER}


MAGE, WAR = "RenoKazakusMage", "AggroPirateWarrior"

print("=" * 84)
print("within-deck RandomForest 5-fold CV accuracy  (reference; chance = 0.20)")
print("=" * 84)
ref = pd.DataFrame(
    {name: {dk: within_deck_rf(feats, dk) for dk in DECKS} for name, feats in FEATURE_SETS.items()}
).T
print(ref.round(3).to_string())

print("\n" + "=" * 84)
print("CROSS-DECK RandomForest — train one deck, test the other  (chance = 0.20)")
print("=" * 84)
for name, feats in FEATURE_SETS.items():
    print(f"\n{name}")
    for a, b in [(MAGE, WAR), (WAR, MAGE)]:
        acc, per = cross_deck_rf(feats, a, b)
        print(f"  train {a:18s} -> test {b:18s}: acc {acc:.3f}  per-style {per}")

print("\n" + "=" * 84)
print("CROSS-DECK aggregate LDA — single-game acc & smallest N (other-deck games) to 90%")
print("=" * 84)
for name, feats in FEATURE_SETS.items():
    print(f"\n{name}")
    for a, b in [(MAGE, WAR), (WAR, MAGE)]:
        one, n90 = cross_deck_aggregate(feats, a, b)
        print(f"  train {a} -> test {b}")
        print(f"    single-game acc : {one:.3f}")
        print("    N to 90%        : " + " | ".join(f"{k}: {n90[k]}" for k in ["overall"] + STYLE_ORDER))
