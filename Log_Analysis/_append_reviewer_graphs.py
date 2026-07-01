"""Append a "Reviewer follow-ups" section to every playstyle_log_distribution
notebook, implementing the three graphs Prof. Kokolo asked for in his email.

His three asks (mapped to the report's result sections):
  #1 (§4.2) Confusion matrices — *one game* vs *10 pooled games* — so you can SEE
            which styles are easy and which get confused, before quoting "games to 90%".
  #2 (§4.3) A curve of ARI / classification accuracy as N grows (not only the
            N=1 and N=120 extremes).
  #3 (§4.4) Cross-deck: (a) aggregate several games before classifying across decks,
            and (b) domain-shift correction — normalise features WITHIN each deck
            ("how far this player deviates from the average player of the same deck").

#1 and #2 are appended to every notebook (each recomputes on its own feature set).
#3 only makes sense where two decks are parsed, so it is appended to V4 only.

The appended cells reuse the variables every notebook already defines in §1/§4:
`games_df`, `FEATURES`, `X`, `y`, `STYLE_ORDER`, `STYLE_COLORS` (+ in V4 `games_all`,
`DA`, `DB`, `AGNOSTIC_FEATURES`).  All new cell ids start with "rev_" so the script
is idempotent: re-running it strips the old reviewer cells and re-appends fresh ones.
Existing cells / outputs are left untouched.

Run:  py -3 Log_Analysis/_append_reviewer_graphs.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

NOTEBOOKS = [
    ("V1/playstyle_log_distribution_analysis_AggroPirateWarrior.ipynb", 9,  False),
    ("V1/playstyle_log_distribution_analysis_RenoKazakusMage.ipynb",    9,  False),
    ("V2/playstyle_log_distribution_analysis_RenoKazakusMage_v2.ipynb", 9,  False),
    ("V3/playstyle_log_distribution_analysis_RenoKazakusMage_v3.ipynb", 9,  False),
    ("V3/playstyle_log_distribution_analysis_RenoKazakusMage_v3_full.ipynb", 9, False),
    ("V4/playstyle_log_distribution_analysis_RenoKazakusMage_v4.ipynb", 10, True),
]


def md(num):
    return f"""
## {num} · Reviewer follow-ups — Prof. Kokolo's feedback

This section answers the three graphs Prof. Kokolo asked for after the weekly report.
It re-uses this notebook's own feature set (`FEATURES`) and standardised matrix (`X`, `y`),
so every version reports the figures *on its own metrics*.

* **#1 — Confusion matrices at N = 1, 10, 25, 50 games.** Before quoting "how many games to
  reach 90%", look at *where the mistakes go*: classify a **single** held-out game, then
  fingerprints of **10, 25 and 50 pooled** games, and read the row-normalised confusion matrix of each.
* **#2 — ARI / accuracy as N grows.** The report only showed the N=1 and N=120 extremes;
  here is the whole curve in between (unsupervised cluster–style ARI *and* held-out accuracy).
* **#3 — Cross-deck (V4 only).** Aggregate several games before classifying across decks, and
  apply **domain-shift correction**: express each feature as a deviation from the *average
  player of the same deck*, then re-test transfer.
"""


REV_CONF = r'''
# ===== Reviewer #1 · confusion matrices at N = 1, 10, 25, 50 pooled games =====
# Held-out LDA, trained on N-game fingerprints and tested on N-game fingerprints
# (N=1 is just single games) -- same protocol as §8, shown as confusion matrices.
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score

def _heldout_split(df, feats, seed=1):
    """Disjoint 50/50 split within each style; standardise on the train half."""
    rng = np.random.default_rng(seed)
    G = df.copy(); G[feats] = G[feats].fillna(G[feats].median())
    tr_idx, te_idx = [], []
    for st in STYLE_ORDER:
        idx = G.index[G["style"] == st].to_numpy().copy(); rng.shuffle(idx)
        h = len(idx) // 2; tr_idx += list(idx[:h]); te_idx += list(idx[h:])
    sc = StandardScaler().fit(G.loc[tr_idx, feats].values)
    Xtr = sc.transform(G.loc[tr_idx, feats].values)
    Xte = sc.transform(G.loc[te_idx, feats].values)
    ytr = G.loc[tr_idx, "style"].astype(str).values
    yte = G.loc[te_idx, "style"].astype(str).values
    return Xtr, ytr, Xte, yte

def _fps(Xby, N, reps, seed):
    """reps fingerprints per style, each = mean of N random games of that style."""
    r = np.random.default_rng(seed); Xs, ys = [], []
    for si, s in enumerate(STYLE_ORDER):
        a = Xby[s]
        for _ in range(reps):
            Xs.append(a[r.integers(0, len(a), N)].mean(0)); ys.append(si)
    return np.array(Xs), np.array(ys)

def confusion_at_N(df, feats, N, reps=600, seed=1):
    Xtr, ytr, Xte, yte = _heldout_split(df, feats, seed)
    trby = {s: Xtr[ytr == s] for s in STYLE_ORDER}
    teby = {s: Xte[yte == s] for s in STYLE_ORDER}
    if N == 1:
        pred = LinearDiscriminantAnalysis().fit(Xtr, ytr).predict(Xte)
        cm  = confusion_matrix(yte, pred, labels=STYLE_ORDER, normalize="true")
        acc = accuracy_score(yte, pred)
    else:
        Xa, ya = _fps(trby, N, reps, seed + 10)
        Xb, yb = _fps(teby, N, reps, seed + 20)
        pb  = LinearDiscriminantAnalysis().fit(Xa, ya).predict(Xb)
        cm  = confusion_matrix(yb, pb, labels=range(len(STYLE_ORDER)), normalize="true")
        acc = (pb == yb).mean()
    return cm, acc

N_CONF = [1, 10, 25, 50]
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
for a, N in zip(ax.ravel(), N_CONF):
    cm, acc = confusion_at_N(games_df, FEATURES, N)
    sns.heatmap(pd.DataFrame(cm, index=STYLE_ORDER, columns=STYLE_ORDER), annot=True, fmt=".2f",
                cmap="rocket_r", vmin=0, vmax=1, ax=a, cbar_kws={"label": "row-normalised"})
    a.set_title(f"{'ONE game' if N == 1 else f'{N} games pooled'}  —  overall acc = {acc:.2f}")
    a.set_ylabel("true style"); a.set_xlabel("predicted style")
fig.suptitle("Reviewer #1 · Confusion at N = 1, 10, 25, 50 pooled games "
             f"(held-out LDA, chance 0.20, {len(FEATURES)} features)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
print("Read: the diagonal sharpens from 1 -> 10 -> 25 -> 50 games; the off-diagonal shows which "
      "styles are confused (typically the value styles control/fatigue/midrange/ramp).")
'''


REV_SWEEP = r'''
# ===== Reviewer #2 · ARI and accuracy as N increases (the full sweep) =====
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

Ns = [1, 2, 3, 5, 8, 10, 15, 20, 25, 35, 50, 70, 90, 100, 120]

# (a) UNSUPERVISED: KMeans(k=5) cluster-style ARI on N-pooled fingerprints (all data)
def _boot(N, reps=200, seed=0):
    rng = np.random.default_rng(seed); rows, lab = [], []
    for st in STYLE_ORDER:
        a = X[y == st]
        for _ in range(reps):
            rows.append(a[rng.integers(0, len(a), N)].mean(0)); lab.append(st)
    return np.array(rows), np.array(lab)

ari = []
for N in Ns:
    Xb, yb = _boot(N)
    ari.append(adjusted_rand_score(yb, KMeans(5, n_init=10, random_state=0).fit_predict(Xb)))

# (b) SUPERVISED: held-out LDA accuracy vs N (overall + per style)
Xtr, ytr, Xte, yte = _heldout_split(games_df, FEATURES)
trby = {s: Xtr[ytr == s] for s in STYLE_ORDER}; teby = {s: Xte[yte == s] for s in STYLE_ORDER}
acc_overall, acc_per = [], {s: [] for s in STYLE_ORDER}
for N in Ns:
    Xa, ya = _fps(trby, N, 500, 10); Xc, yc = _fps(teby, N, 500, 20)
    p = LinearDiscriminantAnalysis().fit(Xa, ya).predict(Xc)
    acc_overall.append((p == yc).mean())
    for si, s in enumerate(STYLE_ORDER):
        acc_per[s].append((p[yc == si] == si).mean())

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
xs = range(len(Ns))
ax[0].plot(xs, ari, "o-", color="#34495e"); ax[0].axhline(0.9, color="green", ls=":", lw=1)
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(Ns, rotation=45); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("games pooled per fingerprint (N)"); ax[0].set_ylabel("KMeans cluster–style ARI")
ax[0].set_title("Unsupervised: 5 clusters sharpen as N grows")
for s in STYLE_ORDER:
    ax[1].plot(xs, acc_per[s], "o-", color=STYLE_COLORS[s], label=s, alpha=.85)
ax[1].plot(xs, acc_overall, "k--", lw=2, label="overall")
ax[1].axhline(0.9, color="green", ls=":", lw=1); ax[1].axhline(0.2, color="gray", ls=":", lw=1)
ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(Ns, rotation=45); ax[1].set_ylim(0, 1.02)
ax[1].set_xlabel("games pooled per fingerprint (N)"); ax[1].set_ylabel("held-out accuracy")
ax[1].set_title("Supervised: accuracy vs N (per style + overall)"); ax[1].legend(fontsize=8)
fig.suptitle("Reviewer #2 · ARI & accuracy as N increases — the curve between N=1 and N=120 "
             f"({len(FEATURES)} features)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
'''


# -------------------------- V4-only cross-deck cells --------------------------
REV_XDECK_MD = r"""
### Reviewer #3 · Cross-deck — aggregation + domain-shift correction (V4 only)

These two graphs use both deck families (`games_all`) and the deck-agnostic feature set
(`AGNOSTIC_FEATURES`). First we ask Prof. Kokolo's question — *what if we aggregate several
games before classifying across decks?* — as confusion matrices (1 game vs 10 pooled, both
directions). Then we apply **domain-shift correction**: instead of raw feature values, each
game is re-expressed as how far it deviates from the **average player of its own deck**
(per-deck z-score), which removes the deck-level bias (e.g. an aggressive deck producing more
attacks regardless of style) before we test transfer.
"""

REV_XDECK_CONF = r'''
# ===== Reviewer #3a · cross-deck confusion: ONE game vs TEN pooled, both directions =====
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

DA, DB = "RenoKazakusMage", "AggroPirateWarrior"   # self-sufficient if §9 wasn't run
XFEATS = list(AGNOSTIC_FEATURES)                   # deck-agnostic set

def _deck_mat(deck, feats, fill_from=None):
    d = games_all[games_all["deck"] == deck]
    med = (d if fill_from is None else games_all[games_all["deck"] == fill_from])[feats].median()
    return d[feats].fillna(med).values, d["style"].astype(str).values

def cross_confusion(a, b, feats, N, reps=600, seed=3):
    Xtr, ytr = _deck_mat(a, feats)
    Xte, yte = _deck_mat(b, feats, fill_from=a)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    trby = {s: Xtr[ytr == s] for s in STYLE_ORDER}; teby = {s: Xte[yte == s] for s in STYLE_ORDER}
    if N == 1:
        pred = LinearDiscriminantAnalysis().fit(Xtr, ytr).predict(Xte)
        cm = confusion_matrix(yte, pred, labels=STYLE_ORDER, normalize="true"); acc = (pred == yte).mean()
    else:
        def fps(by, seed_):
            r = np.random.default_rng(seed_); Xs, ys = [], []
            for si, s in enumerate(STYLE_ORDER):
                for _ in range(reps):
                    Xs.append(by[s][r.integers(0, len(by[s]), N)].mean(0)); ys.append(si)
            return np.array(Xs), np.array(ys)
        Xa, ya = fps(trby, seed + 10); Xb, yb = fps(teby, seed + 20)
        p = LinearDiscriminantAnalysis().fit(Xa, ya).predict(Xb)
        cm = confusion_matrix(yb, p, labels=range(len(STYLE_ORDER)), normalize="true"); acc = (p == yb).mean()
    return cm, acc

dirs = [(DA, DB, "train Mage → test Warrior"), (DB, DA, "train Warrior → test Mage")]
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
for row, (a, b, ttl) in enumerate(dirs):
    for col, N in enumerate((1, 10)):
        cm, acc = cross_confusion(a, b, XFEATS, N)
        sns.heatmap(pd.DataFrame(cm, index=STYLE_ORDER, columns=STYLE_ORDER), annot=True, fmt=".2f",
                    cmap="rocket_r", vmin=0, vmax=1, ax=ax[row, col])
        ax[row, col].set_title(f"{ttl}\n{'1 game' if N == 1 else '10 pooled'} — acc {acc:.2f}")
        ax[row, col].set_ylabel("true style"); ax[row, col].set_xlabel("predicted")
fig.suptitle("Reviewer #3a · Cross-deck confusion — pooling 10 games (chance 0.20). "
             "Watch fatigue/ramp survive the swap; aggro/midrange collapse.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
'''

REV_XDECK_NORM = r'''
# ===== Reviewer #3b · domain-shift correction: standardise WITHIN each deck =====
# "how much this player deviates from the AVERAGE player of the same deck"
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def deck_zscore(df, feats):
    g = df.copy()
    g[feats] = g[feats].fillna(g.groupby("deck")[feats].transform("median"))
    g[feats] = g.groupby("deck")[feats].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return g

XFEATS   = list(AGNOSTIC_FEATURES)
games_dz = deck_zscore(games_all, XFEATS)          # domain-corrected features

def cross_single(frame, a, b, feats, model="rf"):
    tr, te = frame[frame["deck"] == a], frame[frame["deck"] == b]
    med = tr[feats].median()                                   # impute from the TRAIN deck
    Xtr, Xte = tr[feats].fillna(med).values, te[feats].fillna(med).values
    ytr, yte = tr["style"].astype(str).values, te["style"].astype(str).values
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = (RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
           if model == "rf" else LinearDiscriminantAnalysis())
    return accuracy_score(yte, clf.fit(Xtr, ytr).predict(Xte))

bars = {}
for label, frame in [("raw V4 agnostic", games_all), ("deck-normalised", games_dz)]:
    bars[label] = {"RF M→W":  cross_single(frame, DA, DB, XFEATS, "rf"),
                   "RF W→M":  cross_single(frame, DB, DA, XFEATS, "rf"),
                   "LDA M→W": cross_single(frame, DA, DB, XFEATS, "lda"),
                   "LDA W→M": cross_single(frame, DB, DA, XFEATS, "lda")}
barsdf = pd.DataFrame(bars)
print("Single-game cross-deck accuracy (chance = 0.20):"); display(barsdf.round(3))

def cross_agg(frame, a, b, feats, Ns=(1, 2, 3, 5, 8, 10, 15, 20, 25, 35, 50, 70, 90, 100, 120), reps=300):
    tr, te = frame[frame["deck"] == a], frame[frame["deck"] == b]
    med = tr[feats].median()                                   # impute from the TRAIN deck
    Xtr, Xte = tr[feats].fillna(med).values, te[feats].fillna(med).values
    ytr, yte = tr["style"].astype(str).values, te["style"].astype(str).values
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    trby = {s: Xtr[ytr == s] for s in STYLE_ORDER}; teby = {s: Xte[yte == s] for s in STYLE_ORDER}
    def fps(by, seed):
        r = np.random.default_rng(seed); Xs, ys = [], []
        for si, s in enumerate(STYLE_ORDER):
            for _ in range(reps):
                Xs.append(by[s][r.integers(0, len(by[s]), N)].mean(0)); ys.append(si)
        return np.array(Xs), np.array(ys)
    out = {}
    for N in Ns:
        Xa, ya = fps(trby, 10); Xb, yb = fps(teby, 20)
        out[N] = (LinearDiscriminantAnalysis().fit(Xa, ya).predict(Xb) == yb).mean()
    return pd.Series(out)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
barsdf.T.plot.bar(ax=ax[0]); ax[0].axhline(0.20, color="k", ls="--", lw=1, label="chance")
ax[0].set_title("Single-game cross-deck: raw vs deck-normalised"); ax[0].set_ylabel("accuracy")
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0); ax[0].legend(fontsize=8, ncol=3)
for label, frame, ls in [("raw", games_all, "o-"), ("deck-normalised", games_dz, "s--")]:
    s = (cross_agg(frame, DA, DB, XFEATS) + cross_agg(frame, DB, DA, XFEATS)) / 2
    xs = range(len(s.index)); ax[1].plot(xs, s.values, ls, label=label)
ax[1].axhline(0.9, color="green", ls=":", lw=1); ax[1].axhline(0.2, color="gray", ls=":", lw=1)
ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(s.index, rotation=45); ax[1].set_ylim(0, 1.02)
ax[1].set_xlabel("games pooled per fingerprint (N)")
ax[1].set_ylabel("cross-deck accuracy (mean of both directions)")
ax[1].set_title("Aggregated cross-deck: raw vs deck-normalised"); ax[1].legend(fontsize=9)
fig.suptitle("Reviewer #3b · Domain-shift correction — deviation from the same-deck average player",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
print("Read: per-deck z-scoring removes the deck's overall bias; cross-deck transfer should "
      "improve (especially once games are aggregated), though deck-entanglement is not fully solved.")
'''


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.strip("\n"), "outputs": [], "execution_count": None}


def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src.strip("\n")}


def patch(rel, num, is_v4):
    path = os.path.join(HERE, rel)
    nb = json.load(open(path, encoding="utf-8"))
    # idempotent: drop any previously-appended reviewer cells
    nb["cells"] = [c for c in nb["cells"] if not str(c.get("id", "")).startswith("rev_")]
    cells = [md_cell("rev_md", md(num)),
             code_cell("rev_conf", REV_CONF),
             code_cell("rev_sweep", REV_SWEEP)]
    if is_v4:
        cells += [md_cell("rev_xdeck_md", REV_XDECK_MD),
                  code_cell("rev_xdeck_conf", REV_XDECK_CONF),
                  code_cell("rev_xdeck_norm", REV_XDECK_NORM)]
    nb["cells"].extend(cells)
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"patched {rel:64s} +{len(cells)} cells (total {len(nb['cells'])})")


if __name__ == "__main__":
    for rel, num, is_v4 in NOTEBOOKS:
        patch(rel, num, is_v4)
    print("done.")
