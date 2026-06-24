"""Assemble playstyle_log_distribution_analysis_RenoKazakusMage.ipynb from cell sources.

Mirrors build_notebook.py (the AggroPirateWarrior version) but points the parser at the
RenoKazakusMage log folders and carries RenoKazakusMage-specific findings in the prose.
"""
import nbformat as nbf

DECK  = "RenoKazakusMage"
CACHE = "playstyle_log_metrics_RenoKazakusMage.pkl"
OUT   = "playstyle_log_distribution_analysis_RenoKazakusMage.ipynb"

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ───────────────────────────────────────────────────────────── title
md(r"""
# Play-style Distribution & Clustering of the 5 AI Archetypes — `RenoKazakusMage`

**Data.** `log/<style>_RenoKazakusMage/` holds 5 folders. Every folder is the **same**
`RenoKazakusMage` deck (a slow, singleton **MAGE** control/highlander list), but driven by a different
SabberStone search/AI play-style: **aggro · control · fatigue · midrange · ramp**. In each log the
folder's style is **P1 (me)**; the opponent (deck *and* style) varies across ~900 games per folder. We
measure **P1 only** and **pool over all opponents** (per the chosen scope).

**Method** — same spirit as `warrior_top20_rank5to10_byGames_analysis.ipynb` and the sibling
`AggroPirateWarrior` notebook: parse each game into a tidy per-game table, then study the **statistical
distribution** of play-style metrics. Here each *"player"* of the reference becomes one of the **5 style
folders**, and we ask whether the distributions form **5 clusters**.

**Metrics** include the reference's tempo/mana family **plus new play-style metrics** (hero-power rate,
face-damage rate, *face-vs-trade* attack ratio, attacks/turn, damage taken, card cost) and — as
requested — **average number of cards in hand** (mean hand size at the end of each of my turns).

> **Headline finding (quantified below).** A single game is too noisy to cluster — the same deck plus
> random opponents makes per-game points overlap (KMeans ARI ≈ 0.09, near zero). But a *play-style is a
> distribution over games*, not one game. Once we look at the **distribution / aggregated fingerprint**,
> the **5 clusters separate almost perfectly** (cluster-vs-style ARI climbs to ≈ 0.99). Both views are
> shown honestly.
>
> **What's different about a control deck.** Forcing an **aggro** search to pilot a slow highlander Mage
> produces a *very* recognisable misfit — so unlike the Warrior list, here **a single game already
> classifies `aggro` ~86% of the time**, and the hard-to-pin ceiling shifts from *control* to the
> **ramp / midrange** value styles, which converge on "spend efficiently for value".
""")

# ───────────────────────────────────────────────────────────── 1 setup
md(r"""## 1 · Setup & parse

The parser lives in [`playstyle_log_parse.py`](playstyle_log_parse.py) (single source of truth, unit-tested
on individual games). It reads every verbose log, attributes events to **P1** by entity id
(`PlayCardTask => [P1]`, `PayPhase 'card[id]'`, `HeroPowerTask`, hero-entity ids for face damage), and
joins each folder's `summary.csv` for the header facts (winner, turns, seconds, start player).
`build_frames(deck="RenoKazakusMage")` selects the RenoKazakusMage folder family; results are cached to
`playstyle_log_metrics_RenoKazakusMage.pkl` — delete it to force a full re-parse.
""")

code(r"""
import os, pickle, collections, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from playstyle_log_parse import build_frames, STYLES

%matplotlib inline
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.width", 200)

# stable order + one colour per play-style (used everywhere)
STYLE_ORDER  = ["aggro", "control", "fatigue", "midrange", "ramp"]
STYLE_COLORS = dict(zip(STYLE_ORDER, ["#d62728", "#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e"]))
PALETTE      = [STYLE_COLORS[s] for s in STYLE_ORDER]

DECK  = "RenoKazakusMage"
CACHE = "playstyle_log_metrics_RenoKazakusMage.pkl"
games_df, cards_df, turns_df = build_frames(cache=CACHE, deck=DECK)
# NB: 'style' clashes with the DataFrame.style accessor -> always index with games_df["style"]
games_df["style"] = pd.Categorical(games_df["style"], categories=STYLE_ORDER, ordered=True)

print(f"deck = {DECK}")
print(f"games_df {games_df.shape} | cards_df {cards_df.shape} | turns_df {turns_df.shape}")
print("\ngames per style:")
print(games_df["style"].value_counts().reindex(STYLE_ORDER).to_string())
print(f"\noverall P1 win rate: {games_df['win'].mean():.1%}")
games_df.head(3)
""")

# ───────────────────────────────────────────────────────────── metric glossary
md(r"""### 1b · Metric glossary

All metrics are for **P1 (the folder's play-style)**, one value per game. *New* = beyond the reference set.

| metric (column) | meaning | new? |
|---|---|---|
| `n_my_turns` | number of P1 turns played | |
| `time_per_turn` | engine seconds per turn (search effort) | ✓ |
| `cards_per_turn` | non-Coin cards played per turn | |
| `minions_per_turn` | minions played per turn | ✓ |
| `mana_eff` | mana spent ÷ mana available | |
| `mana_per_turn` | mana spent per turn | |
| `avg_card_cost` | mean mana cost of cards P1 played (curve centre) | |
| `first_turn` | first P1 turn a card is played | |
| `coin_turn` | P1 turn the Coin is played (NaN if none) | |
| **`avg_cards_in_hand`** | **mean hand size at the end of each P1 turn** | **✓ (requested)** |
| `max_cards_in_hand` | peak hand size in the game | ✓ |
| `hp_per_turn` | hero-power activations per turn | ✓ |
| `face_dmg_per_turn` | damage dealt to the enemy hero per turn (aggression) | ✓ |
| `face_attack_ratio` | share of P1 attacks aimed at the enemy **face** vs minions | ✓ |
| `attacks_per_turn` | P1 attacks per turn | ✓ |
| `taken_dmg` | total damage P1's hero took (game) | ✓ |
""")

# ───────────────────────────────────────────────────────────── 2 per-style grids
md(r"""## 2 · Per-style distribution grids

One distribution grid **per style folder** — the direct analogue of the reference's `show_player()`.
Dashed line = that style's mean. Eyeball how the shapes shift from **aggro** (short games, high face
ratio, but a *hoarded* hand and the most damage taken — the aggressive search mis-pilots a control deck)
toward **ramp / midrange** (longer games, efficient mana, more trading).
""")

code(r"""
# label -> (column, kind);  kind drives the plot style
METRICS = {
    "My turns / game":     ("n_my_turns",        "disc"),
    "Time / turn (s)":     ("time_per_turn",     "kde"),
    "Cards / turn":        ("cards_per_turn",    "kde"),
    "Minions / turn":      ("minions_per_turn",  "kde"),
    "Mana efficiency":     ("mana_eff",          "clip01"),
    "Mana spent / turn":   ("mana_per_turn",     "kde"),
    "Avg card cost":       ("avg_card_cost",     "kde"),
    "First turn to play":  ("first_turn",        "disc"),
    "Avg cards in hand":   ("avg_cards_in_hand", "kde"),
    "Max cards in hand":   ("max_cards_in_hand", "disc"),
    "Hero-power / turn":   ("hp_per_turn",       "kde"),
    "Face dmg / turn":     ("face_dmg_per_turn", "kde"),
    "Face-attack ratio":   ("face_attack_ratio", "clip01"),
    "Attacks / turn":      ("attacks_per_turn",  "kde"),
    "Damage taken (game)": ("taken_dmg",         "kde"),
}
FEATURES = [c for _, (c, _) in METRICS.items()]   # numeric columns used for clustering

def _plot_metric(ax, s, kind, color):
    s = s.dropna()
    if not len(s):
        return
    if kind == "disc":
        sns.histplot(s.round().astype(int), discrete=True, color=color, ax=ax, stat="density")
    elif kind == "clip01":
        sns.histplot(s.clip(0, 1), kde=(s.nunique() > 1), color=color, ax=ax, stat="density")
    else:
        sns.histplot(s, kde=(s.nunique() > 1), color=color, ax=ax, stat="density")
    ax.axvline(s.mean(), color="k", ls="--", lw=1)

def show_style(style):
    gp = games_df[games_df["style"] == style]
    display(Markdown(
        f"### `{style}`  ·  {len(gp)} games  ·  win {gp['win'].mean():.1%}  ·  "
        f"avg {gp['n_my_turns'].mean():.1f} turns"))
    fig, axes = plt.subplots(3, 5, figsize=(19, 9)); axes = axes.ravel()
    for ax, (label, (col, kind)) in zip(axes, METRICS.items()):
        _plot_metric(ax, gp[col], kind, STYLE_COLORS[style])
        ax.set_title(label, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(f"{style} — per-game metric distributions ({DECK})", fontsize=14, color=STYLE_COLORS[style])
    fig.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()
""")

for s in ["aggro", "control", "fatigue", "midrange", "ramp"]:
    code(f'show_style("{s}")')

# ───────────────────────────────────────────────────────────── 3 overlaid
md(r"""## 3 · Overlaid distributions — all 5 styles per metric

The clearest single view of *how* the styles differ: for each metric, all 5 styles' KDEs share one axis.
Where the coloured curves pull apart (e.g. **my turns**, **cards/turn**, **mana spent/turn**,
**attacks/turn**, **hero-power/turn**, **face-attack ratio**, **avg cards in hand**) that metric carries
play-style signal; where they overlap it does not. This is the per-metric, univariate basis for the
multivariate clustering in §5–6.
""")

code(r"""
fig, axes = plt.subplots(3, 5, figsize=(19, 10)); axes = axes.ravel()
for ax, (label, (col, kind)) in zip(axes, METRICS.items()):
    for st in STYLE_ORDER:
        s = games_df.loc[games_df["style"] == st, col].dropna()
        if s.nunique() > 1:
            clip = s.clip(0, 1) if kind == "clip01" else s
            sns.kdeplot(clip, ax=ax, color=STYLE_COLORS[st], lw=1.8, label=st, warn_singular=False)
    ax.set_title(label, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
axes[0].legend(title="style", fontsize=8)
fig.suptitle("Per-metric distributions overlaid across the 5 play-styles (RenoKazakusMage)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

# per-style means (z-scored within each metric so the fingerprint is visible at a glance)
mean_tbl = games_df.groupby("style", observed=True)[FEATURES].mean().reindex(STYLE_ORDER)
z = (mean_tbl - mean_tbl.mean()) / mean_tbl.std()
display(mean_tbl.round(3))
fig, ax = plt.subplots(figsize=(13, 3.4))
sns.heatmap(z, annot=mean_tbl.round(2), fmt="", cmap="vlag", center=0, ax=ax,
            cbar_kws={"label": "z-score across styles"})
ax.set_title("Play-style fingerprints — per-style mean of each metric (colour = z-score across the 5 styles)")
plt.show()
""")

md(r"""### 3b · Per-turn dynamics by style
Mean **cards in hand** as the game progresses, and the **fraction of games still going** at my-turn *t* —
averaged over all games of each style (95% CI band). The aggro search stalls out early (short games) yet
sits on a *fuller* hand; ramp/midrange grind the longest.
""")
code(r"""
tt = turns_df[turns_df["my_turn"] <= 12].copy()
tt["style"] = pd.Categorical(tt["style"], categories=STYLE_ORDER, ordered=True)
mt = games_df[["style", "game_id", "n_my_turns"]]   # (no per-turn mana frame needed for the hand curve)
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
sns.lineplot(data=tt, x="my_turn", y="hand_end", hue="style", hue_order=STYLE_ORDER,
             palette=STYLE_COLORS, errorbar=("ci", 95), ax=ax[1])
ax[1].set_title("Cards in hand at end of my turn"); ax[1].set_xlabel("my turn"); ax[1].set_ylabel("cards in hand")
# turns survived per style (survival-ish curve)
surv = (games_df.groupby("style", observed=True)["n_my_turns"]
        .apply(lambda s: pd.Series({t: (s >= t).mean() for t in range(1, 16)})).unstack().T)
for st in STYLE_ORDER:
    ax[0].plot(surv.index, surv[st], color=STYLE_COLORS[st], marker="o", ms=3, label=st)
ax[0].set_title("Fraction of games still going at my-turn t"); ax[0].set_xlabel("my turn"); ax[0].set_ylabel("share of games"); ax[0].legend(fontsize=8)
fig.tight_layout(); plt.show()
""")

# ───────────────────────────────────────────────────────────── 4 feature matrix
md(r"""## 4 · Feature matrix (standardised)

Stack the per-game metrics into `X` (z-scored). `coin_turn` is mostly NaN (no Coin) so we drop it from the
matrix and impute remaining gaps with the column median.
""")
code(r"""
from sklearn.preprocessing import StandardScaler

X_raw = games_df[FEATURES].copy()
X_raw = X_raw.fillna(X_raw.median())
y = games_df["style"].astype(str).values
scaler = StandardScaler().fit(X_raw.values)
X = scaler.transform(X_raw.values)
print(f"feature matrix: {X.shape}  ({len(FEATURES)} metrics)")
print("features:", ", ".join(FEATURES))
""")

# ───────────────────────────────────────────────────────────── 5 per-game clustering (honest)
md(r"""## 5 · Per-game projection & clustering — the honest baseline

Project the 15-D per-game points to 2-D (**PCA** and **t-SNE**) and colour by true style; then run
**unsupervised** KMeans / GMM with `k=5` and score how well the clusters match the styles
(Adjusted Rand Index, silhouette). Expectation, made explicit: at the **single-game** level the clouds
**overlap** — one game of the same deck against a random opponent isn't a clean style signature (the
`aggro` blob is the one that starts to peel away).
""")
code(r"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment

def cluster_confusion(true_lab, clab, order=STYLE_ORDER):
    # map cluster ids to styles (Hungarian on the contingency table) -> readable confusion matrix
    k = len(order)
    M = pd.crosstab(pd.Series(true_lab, name="style"), pd.Series(clab, name="cluster"))
    M = M.reindex(index=order).reindex(columns=range(k), fill_value=0).fillna(0)
    r, c = linear_sum_assignment(-M.values)
    col_to_style = {c[i]: order[r[i]] for i in range(len(r))}
    mapped = pd.Series(clab).map(lambda x: col_to_style.get(x, x)).values
    cm = pd.crosstab(pd.Series(true_lab, name="true style"),
                     pd.Series(mapped, name="cluster→style")).reindex(index=order, columns=order, fill_value=0)
    return cm, mapped

pca = PCA(n_components=2, random_state=0).fit(X)
P = pca.transform(X)
rng = np.random.default_rng(0)
sub = rng.choice(len(X), size=min(1800, len(X)), replace=False)   # t-SNE on a subsample (speed)
T = TSNE(n_components=2, init="pca", perplexity=40, random_state=0).fit_transform(X[sub])

km = KMeans(5, n_init=10, random_state=0).fit(X)
gm = GaussianMixture(5, covariance_type="full", random_state=0).fit(X)
km_lab, gm_lab = km.labels_, gm.predict(X)

print("Per-game unsupervised clustering vs the 5 true styles")
for name, lab in [("KMeans", km_lab), ("GMM", gm_lab)]:
    print(f"  {name:7s}: ARI={adjusted_rand_score(y, lab):.3f}  "
          f"AMI={adjusted_mutual_info_score(y, lab):.3f}  silhouette={silhouette_score(X, lab):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
for st in STYLE_ORDER:
    m = (y == st); ax[0].scatter(P[m, 0], P[m, 1], s=6, alpha=.35, color=STYLE_COLORS[st], label=st)
    ms = (y[sub] == st); ax[1].scatter(T[ms, 0], T[ms, 1], s=6, alpha=.35, color=STYLE_COLORS[st], label=st)
ax[0].set_title("PCA (per game) — coloured by TRUE style"); ax[0].legend(fontsize=8, markerscale=2)
ax[1].set_title("t-SNE (per game) — coloured by TRUE style")
cm, _ = cluster_confusion(y, km_lab)
sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", ax=ax[2])
ax[2].set_title(f"KMeans(k=5) confusion (ARI={adjusted_rand_score(y, km_lab):.2f})")
for a in ax[:2]:
    a.set_xlabel(""); a.set_ylabel("")
fig.suptitle("§5  Per-game level: the 5 styles mostly OVERLAP — clusters are weak", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
""")

# ───────────────────────────────────────────────────────────── 5b supervised
md(r"""### 5b · Are the styles distinguishable *at all*? (supervised check)

Unsupervised clustering can fail just because clusters aren't round/separated, even when classes are
*distinguishable*. So we also fit a **supervised** classifier (random-forest, 5-fold CV) and an **LDA**
projection (the linear axes that best separate the 5 styles). Accuracy well above the 20% chance line ⇒
the per-game signal is real, only noisy.
""")
code(r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
pred = cross_val_predict(rf, X, y, cv=5)
acc = accuracy_score(y, pred)
print(f"Random-forest 5-fold CV accuracy: {acc:.3f}   (chance = 0.20)")

rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
L = LinearDiscriminantAnalysis(n_components=2).fit(X, y).transform(X)

fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
cm = confusion_matrix(y, pred, labels=STYLE_ORDER, normalize="true")
sns.heatmap(pd.DataFrame(cm, index=STYLE_ORDER, columns=STYLE_ORDER), annot=True, fmt=".2f",
            cmap="rocket_r", ax=ax[0])
ax[0].set_title(f"RF confusion (row-normalised), acc={acc:.2f}"); ax[0].set_ylabel("true"); ax[0].set_xlabel("predicted")
imp.plot.barh(ax=ax[1], color="#34495e"); ax[1].set_title("RF feature importance")
for st in STYLE_ORDER:
    m = (y == st); ax[2].scatter(L[m, 0], L[m, 1], s=6, alpha=.35, color=STYLE_COLORS[st], label=st)
ax[2].set_title("LDA (per game) — best linear separation"); ax[2].legend(fontsize=8, markerscale=2)
fig.tight_layout(); plt.show()
""")

# ───────────────────────────────────────────────────────────── 6 bootstrap fingerprints
md(r"""## 6 · The 5 clusters DO emerge — at the distribution level

A *play-style is a distribution over games*, not a single game. So characterise each style by the
**distribution of its aggregated fingerprints**: repeatedly draw `N` random games of a style and average
their metric vector. Each draw is one "fingerprint" point; many draws give a cloud per style. As `N`
grows, opponent noise averages out and the **5 clouds pull apart into 5 clean clusters** — exactly the
structure the per-game view couldn't show. We confirm it is *intrinsic* by clustering the fingerprints
**unsupervised** (KMeans, no labels) and scoring ARI against the true style.
""")
code(r"""
def bootstrap_fingerprints(N, reps=150, seed=0):
    rng = np.random.default_rng(seed)
    rows, lab = [], []
    for st in STYLE_ORDER:
        a = X[y == st]
        for _ in range(reps):
            rows.append(a[rng.integers(0, len(a), N)].mean(axis=0)); lab.append(st)
    return np.array(rows), np.array(lab)

# ARI of unsupervised KMeans(k=5) on the fingerprints, as a function of N
Ns = [1, 5, 10, 20, 40, 80, 150, 300]
aris, sils = [], []
for N in Ns:
    Xb, yb = bootstrap_fingerprints(N)
    lab = KMeans(5, n_init=10, random_state=0).fit_predict(Xb)
    aris.append(adjusted_rand_score(yb, lab)); sils.append(silhouette_score(Xb, lab))

# a clean picture at a representative N
Xb, yb = bootstrap_fingerprints(120, reps=200)
labb = KMeans(5, n_init=10, random_state=0).fit_predict(Xb)
ari_b = adjusted_rand_score(yb, labb)
Lb = LinearDiscriminantAnalysis(n_components=2).fit(Xb, yb).transform(Xb)

fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
ax[0].plot(Ns, aris, "o-", label="cluster–style ARI")
ax[0].plot(Ns, sils, "s--", color="gray", label="silhouette")
ax[0].set_xscale("log"); ax[0].set_xlabel("games averaged per fingerprint (N)")
ax[0].set_title("5 clusters sharpen as games are aggregated"); ax[0].legend(); ax[0].set_ylim(0, 1)
for st in STYLE_ORDER:
    m = (yb == st); ax[1].scatter(Lb[m, 0], Lb[m, 1], s=14, alpha=.6, color=STYLE_COLORS[st], label=st)
ax[1].set_title(f"Fingerprint clouds (N=120) — 5 clusters\nunsupervised KMeans ARI={ari_b:.2f}")
ax[1].legend(fontsize=8, markerscale=1.5); ax[1].set_xlabel("LD1"); ax[1].set_ylabel("LD2")
cmb, _ = cluster_confusion(yb, labb)
sns.heatmap(cmb, annot=True, fmt="d", cmap="rocket_r", ax=ax[2])
ax[2].set_title("KMeans clusters vs true style (N=120)")
fig.suptitle("§6  At the distribution level the 5 play-styles separate into 5 clusters", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
print(f"unsupervised KMeans(k=5) on N=120 fingerprints:  ARI={ari_b:.3f}")
""")

# ───────────────────────────────────────────────────────────── 7 conclusion
md(r"""## 7 · Conclusion

* **Yes, the 5 archetypes form 5 clusters — but only as distributions, not as single games.** With the
  *same* `RenoKazakusMage` deck and random opponents, one game carries weak style signal: per-game
  KMeans/GMM give ARI ≈ 0.06–0.09 and the PCA/t-SNE clouds largely overlap (§5).
* **The signal is nonetheless real and a bit stronger than for the Warrior list**: a supervised
  classifier reaches ≈ 0.49 (chance 0.20) (§5b), and the styles differ systematically in the overlaid
  distributions and fingerprint heatmap (§3) — most along **search time/turn, mana spent/turn,
  cards- and minions-per-turn, hero-power- and face-damage rate, attacks/turn, and average cards in
  hand**.
* **Aggregation reveals the structure** (§6): averaging games into play-style fingerprints makes the
  5 clusters separate **almost perfectly** (unsupervised ARI ≈ 0.99 by N≈120, →1.0 by N≈150), and
  sharper the more games are pooled.
* **Reading of the styles (control deck)**: *aggro* = the clear outlier — **shortest games** (~6.8 of my
  turns), **highest face-attack ratio**, but it **mis-pilots** the slow deck: lowest mana efficiency,
  fewest cards/attacks per turn, a **hoarded hand** (highest avg cards in hand) and the **most damage
  taken**. *ramp / midrange* = **longest games**, best mana efficiency, most development and trading,
  least damage taken. *control / fatigue* sit between, fatigue grinding slightly longer toward deck-out.

*Practical note*: to **classify an unknown player's style** from this engine, aggregate several of their
games before clustering/scoring — though for this deck a single **aggro** game is already a giveaway.
**§8 quantifies exactly how many games are needed** (and shows the styles are *not* equally separable —
aggro is trivial, the **ramp / midrange** value styles are the ceiling).
""")

# ───────────────────────────────────────────────────────────── 8 how many games
md(r"""## 8 · How many games to classify a player? (and: what does one game tell us?)

§6 showed the 5 clusters are real *in aggregate*. Here we make the practical question precise with a
**held-out test**: split each style's games 50/50, learn the per-style fingerprint **centroids + an LDA
model on the train half only**, then classify fingerprints built from averaging **N held-out games**.

* **Q1 — how many games per style?** Sweep `N` and read accuracy per style (chance = 20%).
* **Q2 — a single unknown game?** The model *always* assigns a nearest cluster (no "reject" option), so
  the question is whether that label is *informative*. The `N=1` confusion matrix answers it.
""")
code(r"""
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# --- held-out 50/50 split within each style (disjoint games) ---
rng = np.random.default_rng(1)
G = games_df.copy(); G[FEATURES] = G[FEATURES].fillna(G[FEATURES].median())
tr_idx, te_idx = [], []
for st in STYLE_ORDER:
    idx = G.index[G["style"] == st].to_numpy().copy(); rng.shuffle(idx)
    h = len(idx) // 2; tr_idx += list(idx[:h]); te_idx += list(idx[h:])
tr, te = G.loc[tr_idx], G.loc[te_idx]
sc8 = StandardScaler().fit(tr[FEATURES].values)
Xtr, Xte = sc8.transform(tr[FEATURES].values), sc8.transform(te[FEATURES].values)
ytr, yte = tr["style"].astype(str).values, te["style"].astype(str).values
trby = {s: Xtr[ytr == s] for s in STYLE_ORDER}
teby = {s: Xte[yte == s] for s in STYLE_ORDER}
Cen  = np.vstack([trby[s].mean(0) for s in STYLE_ORDER])   # train fingerprint centroids = the 5 clusters

def make_fps(by, N, reps, seed):
    r = np.random.default_rng(seed); Xs, ys = [], []
    for si, s in enumerate(STYLE_ORDER):
        a = by[s]
        for _ in range(reps):
            Xs.append(a[r.integers(0, len(a), N)].mean(0)); ys.append(si)
    return np.array(Xs), np.array(ys)

# --- Q1: accuracy vs N (LDA model; nearest-centroid for reference) ---
Ns = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]
rows = []
for N in Ns:
    Xa, ya = make_fps(trby, N, 500, 10)
    lda = LinearDiscriminantAnalysis().fit(Xa, ya)
    Xb, yb = make_fps(teby, N, 500, 20)
    p_lda = lda.predict(Xb)
    p_nc  = (((Xb[:, None] - Cen[None]) ** 2).sum(2)).argmin(1)
    row = {"N": N, "overall (LDA)": (p_lda == yb).mean(), "overall (centroid)": (p_nc == yb).mean()}
    for si, s in enumerate(STYLE_ORDER):
        row[s] = (p_lda[yb == si] == si).mean()
    rows.append(row)
acc = pd.DataFrame(rows).set_index("N")

def first_N(col, thr):
    hit = acc.index[acc[col] >= thr]
    return int(hit[0]) if len(hit) else f">{Ns[-1]}"
print("Smallest N (games averaged) to reach accuracy:")
for thr in (0.80, 0.90):
    print(f"  overall (LDA) >= {thr:.0%}: N = {first_N('overall (LDA)', thr)}")
print("  per style >= 90%:  " + " | ".join(f"{s}: {first_N(s, 0.90)}" for s in STYLE_ORDER))

# --- Q2: single held-out game, classified once each ---
lda1 = LinearDiscriminantAnalysis().fit(Xtr, ytr)
cm1 = confusion_matrix(yte, lda1.predict(Xte), labels=STYLE_ORDER, normalize="true")
print(f"\nSingle-game accuracy: {np.trace(cm1) / 5:.3f}  (chance 0.20)  -> aggro is a giveaway; the rest need pooling")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.3))
# panel 1: accuracy vs N
for s in STYLE_ORDER:
    ax[0].plot(acc.index, acc[s], "o-", color=STYLE_COLORS[s], label=s)
ax[0].plot(acc.index, acc["overall (LDA)"], "k--", lw=2, label="overall")
ax[0].axhline(0.2, color="gray", ls=":", lw=1); ax[0].axhline(0.9, color="green", ls=":", lw=1)
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.02); ax[0].set_xlabel("games averaged per fingerprint (N)")
ax[0].set_ylabel("held-out accuracy"); ax[0].set_title("Q1 · accuracy vs #games (per style)"); ax[0].legend(fontsize=8)
# panel 2: per-style accuracy heatmap across N
hm = acc[STYLE_ORDER].T
sns.heatmap(hm, annot=True, fmt=".2f", cmap="rocket_r", vmin=0.2, vmax=1.0, ax=ax[1],
            cbar_kws={"label": "accuracy"})
ax[1].set_title("Q1 · per-style accuracy by N"); ax[1].set_xlabel("N games"); ax[1].set_ylabel("")
# panel 3: single-game confusion
sns.heatmap(pd.DataFrame(cm1, index=STYLE_ORDER, columns=STYLE_ORDER), annot=True, fmt=".2f",
            cmap="rocket_r", ax=ax[2])
ax[2].set_title(f"Q2 · ONE game: where it lands (acc={np.trace(cm1)/5:.2f})")
ax[2].set_ylabel("true style"); ax[2].set_xlabel("assigned style")
fig.suptitle("§8  How many games are needed to classify a play-style (RenoKazakusMage)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
display(acc.style.format("{:.3f}").background_gradient(cmap="rocket_r", subset=STYLE_ORDER + ["overall (LDA)"])
        .set_caption("Held-out accuracy of an N-game fingerprint"))
""")

md(r"""**Reading §8.**

* **Q1 — games needed (≈90% per style):** **aggro ~2–3** (essentially immediate), **fatigue ~20**,
  **control & midrange ~30**, and **ramp ~50** (the slowest to lock in, since it overlaps the other
  value styles). Rule of thumb: **~12 games → ~80% overall, ~30 → ~90%**, with **ramp/midrange** as the
  ceiling — the opposite end from the Warrior list, where *control* was hardest.
* **Q2 — a single unknown game:** overall ≈ 0.48 (vs 0.20 chance) — **much more informative than for the
  aggro Warrior deck** because **`aggro` is recognised ~86% of the time from one game**: an aggressive
  search piloting a slow highlander Mage produces a glaring misfit (short game, hoarded hand, lots of
  damage taken). The other four styles still need pooling — a single `control` game lands right only ~25%
  of the time, mostly leaking into the neighbouring value styles. **Verdict: one game = a confident
  call *only if it looks like aggro*; otherwise aggregate ~30 games.**
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print(f"wrote {OUT} with {len(cells)} cells")
