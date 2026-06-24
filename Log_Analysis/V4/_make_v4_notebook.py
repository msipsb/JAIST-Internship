"""Assemble the V4 notebook (deck-agnostic, turn-rate-normalized + cross-deck) by
patching the full V3 notebook.

Chain: V1 -> V2 -> V3 Change-B -> V3 full A+B -> **V4 (this file)**.

What this patch does to the V3-full notebook:
  * setup: import the V4 parser, parse BOTH deck families into `games_all`, and
    keep `games_df` = the home deck (RenoKazakusMage) so sections 1-8 run exactly
    as in V3 (now on the V4 columns).
  * §2/§3 grids: add the 4 universal "currency" metrics + 4 normalized
    re-expressions (grid grows 16 -> 24 cells).
  * clustering: `FEATURES` becomes the deck-agnostic set (AGNOSTIC_FEATURES, 29).
  * appends **§9 cross-deck transfer** — the actual deck-agnostic test: train on
    one deck, score the other (RF per-game + aggregate LDA), comparing the
    deck-leaky V3 features, the V4 agnostic set, and the V4 pure-ratio subset.

Run:  py -3 Log_Analysis/V4/_make_v4_notebook.py
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "..", "V3", "playstyle_log_distribution_analysis_RenoKazakusMage_v3_full.ipynb")
DST  = os.path.join(HERE, "playstyle_log_distribution_analysis_RenoKazakusMage_v4.ipynb")
nb = json.load(open(SRC, encoding="utf-8"))


def cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def set_src(cid, text):
    cell(cid)["source"] = text.strip("\n")


def append_md(cid, text):
    c = cell(cid)
    c["source"] = "".join(c["source"]).rstrip("\n") + "\n\n" + text.strip("\n")


def sub_in(cid, pattern, repl, count=1, flags=0):
    c = cell(cid)
    s = "".join(c["source"])
    new, n = re.subn(pattern, repl, s, count=count, flags=flags)
    assert n == count, f"{cid}: expected {count} repl, got {n} for {pattern!r}"
    c["source"] = new


# ---- [2] setup: import V4 parser, parse BOTH decks, home-deck back-compat ------
sub_in("16808bd4", r'os\.path\.abspath\("playstyle_log_parse_v3_full\.py"\)',
       'os.path.abspath("playstyle_log_parse_v4.py")')
sub_in("16808bd4",
       r"from playstyle_log_parse_v3_full import build_frames, STYLES, BASE_DIR, TRAJ_METRICS, A_FEATURES",
       "from playstyle_log_parse_v4 import (build_frames, STYLES, BASE_DIR, TRAJ_METRICS, A_FEATURES,\n"
       "                                    AGNOSTIC_FEATURES, RATIO_FEATURES, DECK_DEP_FEATURES, DECKS)")
sub_in("16808bd4",
       r'DECK  = "RenoKazakusMage"\n'
       r'CACHE = os\.path\.join\(BASE_DIR, "playstyle_log_metrics_RenoKazakusMage_v3_full\.pkl"\)\n'
       r'games_df, cards_df, turns_df = build_frames\(cache=CACHE, deck=DECK\)',
       'DECK  = "RenoKazakusMage"   # "home" deck for the single-deck sections 1-8 (back-compat with V3)\n'
       'CACHE = os.path.join(BASE_DIR, "playstyle_log_metrics_v4_bothdecks.pkl")\n'
       '# V4 parses BOTH deck families so section 9 can test cross-deck transfer.\n'
       'games_all, cards_all, turns_all = build_frames(cache=CACHE, deck=DECKS)\n'
       'games_all["style"] = pd.Categorical(games_all["style"], categories=STYLE_ORDER, ordered=True)\n'
       '# sections 1-8 run on the home deck only (identical to V3); section 9 uses games_all.\n'
       'games_df  = games_all[games_all["deck"] == DECK].copy()\n'
       'cards_df  = cards_all[cards_all["deck"] == DECK].copy()\n'
       'turns_df  = turns_all[turns_all["deck"] == DECK].copy()')

# ---- [5] grids: +4 universal + 4 normalized metrics; FEATURES -> agnostic set --
sub_in("93f623df",
       r'    "First minion turn \(mid\.\)":  \("first_minion_turn",             "disc"\),\n\}',
       '    "First minion turn (mid.)":  ("first_minion_turn",             "disc"),\n'
       '    "Face dmg / turn (aggro)":   ("face_dmg_per_turn",             "kde"),\n'
       '    "Mana floated/turn (greed)": ("mana_floated_per_turn",         "kde"),\n'
       '    "Enemy board left (tol.)":   ("avg_enemy_board_minions",       "kde"),\n'
       '    "Value-turn frac (ramp)":    ("value_turn_fraction",           "clip01"),\n'
       '    "Hand fill ratio":           ("hand_fill_ratio",               "clip01"),\n'
       '    "Cards-left frac":           ("cards_left_frac",               "clip01"),\n'
       '    "First minion frac":         ("first_minion_frac",             "clip01"),\n'
       '    "Max card cost / 10":        ("max_card_cost_norm",            "clip01"),\n}')

sub_in("93f623df",
       r"BASE_FEATURES = \[c for _, \(c, _\) in METRICS\.items\(\)\].*?ramp-less highlander deck",
       "# V4 clusters on the DECK-AGNOSTIC feature set: turn-rate-normalized ratios + the 4 universal\n"
       "# currency metrics + normalized trajectory. Count/cost columns that leak deck identity\n"
       "# (avg/max card cost, raw mana, game length) are deliberately excluded (-> DECK_DEP_FEATURES).\n"
       "FEATURES = list(AGNOSTIC_FEATURES)                             # 29 deck-agnostic features\n"
       "GRID_METRICS = list(METRICS)                                   # 24 columns shown in the grids\n"
       "# note: value_turn_fraction is alive on this Mage deck but ~0 on Pirate Warrior (no draw spells),\n"
       "# the mirror of extra_mana_crystals being dead on this ramp-less highlander list.",
       flags=re.S)

sub_in("93f623df", r"plt\.subplots\(4, 4, figsize=\(16, 12\)\)", "plt.subplots(6, 4, figsize=(16, 18))")

# ---- [12] overlay grid 6x4 + wider fingerprint heatmap ------------------------
sub_in("c3ca5a15", r"plt\.subplots\(4, 4, figsize=\(16, 13\)\)", "plt.subplots(6, 4, figsize=(16, 18))")
sub_in("c3ca5a15", r"fig, ax = plt\.subplots\(figsize=\(22, 3\.8\)\)",
       "fig, ax = plt.subplots(figsize=(26, 4.2))")

# ============================ markdown rewrites ================================
sub_in("e7038b5e",
       r"# Play-style Distribution & Clustering of the 5 AI Archetypes — `RenoKazakusMage` "
       r"\(v3-full · 11 metrics \+ trajectory \+ Change-A tells\)",
       "# Play-style Distribution & Clustering of the 5 AI Archetypes — "
       "`RenoKazakusMage` + `AggroPirateWarrior` (v4 · deck-agnostic, turn-rate-normalized · cross-deck)")

append_md("e7038b5e",
          "> **V4 — deck-agnostic fingerprint.** The single most important change vs V3 is *universal "
          "turn-rate normalization*: every raw count is re-expressed as a per-turn rate or a contextual "
          "fraction, so the features describe the AI's **philosophy** rather than how long the game lasted "
          "or what the deck contained. V4 also adds four **universal currency metrics** (Mana / Cards / "
          "Board / Life) and — because *deck-agnostic only means something across decks* — parses a second "
          "deck family (`AggroPirateWarrior`) so **§9** can train on one deck and test on the other. "
          "Sections 1-8 are the V3 analysis re-run on the home deck (RenoKazakusMage) with the V4 columns; "
          "**§9 is the new cross-deck transfer test.**")

append_md("45bea743",
          "**V4 parser** ([`playstyle_log_parse_v4.py`](playstyle_log_parse_v4.py)) supersets V3: it keeps "
          "every V2 / Change-B / Change-A column, adds turn-rate-normalized re-expressions "
          "(`hand_fill_ratio`, `cards_left_frac`, `first_minion_frac`, `mana_eff_t*`, `hand_frac_t*`) and "
          "four universal currency metrics — `face_dmg_per_turn` (life), `mana_floated_per_turn` (mana / "
          "greed), `avg_enemy_board_minions` (board / threat tolerance), `value_turn_fraction` (cards / "
          "future-value velocity). The two board/ATK-dependent metrics use proxies (the logs carry no "
          "minion ATK stat nor damage-source attribution). `build_frames(deck=DECKS)` parses both deck "
          "families into `games_all` and tags each row with its `deck`; the cache is "
          "`playstyle_log_metrics_v4_bothdecks.pkl` — delete it to force a re-parse.")

append_md("2348a354",
          "*(V4: `FEATURES` is now the **deck-agnostic** set — `AGNOSTIC_FEATURES`, 29 behavioral ratios "
          "with the deck-identity-leaking cost/length columns removed. It still has 29 columns, so §4-§8 "
          "are directly comparable to V3 on the home deck.)*")

append_md("de97b98a",
          "* **What V4 adds:** turn-rate normalization + 4 universal currency metrics, and a **cross-deck "
          "test** (§9). Headline result: V4's agnostic features **transfer across decks better than V3's "
          "deck-leaky features** (train-Mage→test-Warrior and the reverse, both per-game RF and aggregate "
          "LDA) **without losing within-deck accuracy** — but cross-deck per-game classification stays hard "
          "(~0.24 vs 0.20 chance): the AI styles are genuinely deck-entangled. Only the slow archetypes "
          "(**fatigue**, **ramp**) transfer cleanly once games are aggregated; aggro/midrange do not.")

# ============================ §9 appended cells ===============================
S9_MD = """
## 9 · Cross-deck transfer — the deck-agnostic test

Sections 1-8 lived on one deck. *Deck-agnostic* only means something **across** decks, so here we **train
on one deck family and score the other** — `RenoKazakusMage` ⇄ `AggroPirateWarrior` — for three feature
sets:

* **V3-full (deck-leaky)** — includes `avg_card_cost`, `max_card_cost`, raw `mana_at_t*`, game length:
  great *within* a deck, but these encode deck identity.
* **V4 agnostic** — pure behavioral ratios + the 4 universal currency metrics + normalized trajectory.
* **V4 pure-ratio** — the bounded 0-1 subset only (no per-turn rates, no absolute counts).

The bar chart below is the headline; the heatmaps show *which* styles survive a deck swap.
"""

S9_RF = '''
# ===== §9a · within-deck (reference) vs cross-deck RandomForest =====
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from playstyle_log_parse_v4 import METRICS as V2_METRICS, TRAJ_METRICS as _TRAJ, A_FEATURES as _AF

V3_FULL  = list(V2_METRICS) + list(_TRAJ) + list(_AF)
FEATURE_SETS = {
    f"V3-full deck-leaky ({len(V3_FULL)})":      V3_FULL,
    f"V4 agnostic ({len(AGNOSTIC_FEATURES)})":   list(AGNOSTIC_FEATURES),
    f"V4 pure-ratio ({len(RATIO_FEATURES)})":    list(RATIO_FEATURES),
}
DA, DB = "RenoKazakusMage", "AggroPirateWarrior"

def _xy(df, feats, fill_from=None):
    med = (df if fill_from is None else fill_from)[feats].median()
    return df[feats].fillna(med).values, df["style"].astype(str).values

def within_rf(feats, deck):
    d = games_all[games_all["deck"] == deck]
    X, y = _xy(d, feats); Xs = StandardScaler().fit_transform(X)
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    return accuracy_score(y, cross_val_predict(rf, Xs, y, cv=5))

def cross_rf(feats, a, b):
    tr, te = games_all[games_all["deck"] == a], games_all[games_all["deck"] == b]
    Xtr, ytr = _xy(tr, feats); Xte, yte = _xy(te, feats, fill_from=tr)
    sc = StandardScaler().fit(Xtr)
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(sc.transform(Xtr), ytr)
    pred = rf.predict(sc.transform(Xte))
    cm = confusion_matrix(yte, pred, labels=STYLE_ORDER, normalize="true")
    return accuracy_score(yte, pred), dict(zip(STYLE_ORDER, np.diag(cm)))

rows = []
for name, feats in FEATURE_SETS.items():
    wA, wB = within_rf(feats, DA), within_rf(feats, DB)
    cAB, perAB = cross_rf(feats, DA, DB)
    cBA, perBA = cross_rf(feats, DB, DA)
    rows.append(dict(feature_set=name, within_Mage=wA, within_Warrior=wB,
                     cross_M2W=cAB, cross_W2M=cBA))
res = pd.DataFrame(rows).set_index("feature_set")
display(res.round(3))

fig, ax = plt.subplots(figsize=(11, 4.5))
res[["within_Mage", "within_Warrior", "cross_M2W", "cross_W2M"]].plot.bar(ax=ax)
ax.axhline(0.20, color="k", ls="--", lw=1, label="chance (0.20)")
ax.set_ylabel("RF accuracy"); ax.set_xlabel("")
ax.set_title("Within-deck (solid signal) vs cross-deck transfer — V4 agnostic > V3 deck-leaky")
ax.set_xticklabels(ax.get_xticklabels(), rotation=12, ha="right")
ax.legend(fontsize=8, ncol=3); fig.tight_layout(); plt.show()
'''

S9_AGG = '''
# ===== §9b · which styles survive a deck swap? per-style recall + aggregate LDA =====
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, (a, b, ttl) in zip(axes, [(DA, DB, "train Mage → test Warrior"),
                                  (DB, DA, "train Warrior → test Mage")]):
    rec = {name: cross_rf(feats, a, b)[1] for name, feats in FEATURE_SETS.items()}
    M = pd.DataFrame(rec).T[STYLE_ORDER]
    sns.heatmap(M, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "per-style recall"})
    ax.set_title(ttl, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
fig.suptitle("Cross-deck per-style recall (chance = 0.20) — only fatigue/ramp transfer", y=1.04)
fig.tight_layout(); plt.show()

# aggregate: pool N other-deck games into a fingerprint, LDA trained on train-deck fingerprints
def cross_agg(feats, a, b, Ns=(1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120)):
    tr, te = games_all[games_all["deck"] == a], games_all[games_all["deck"] == b]
    Xtr, ytr = _xy(tr, feats); Xte, yte = _xy(te, feats, fill_from=tr)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    trby = {s: Xtr[ytr == s] for s in STYLE_ORDER}; teby = {s: Xte[yte == s] for s in STYLE_ORDER}
    def fps(by, N, reps, seed):
        r = np.random.default_rng(seed); Xs, ys = [], []
        for si, s in enumerate(STYLE_ORDER):
            for _ in range(reps):
                Xs.append(by[s][r.integers(0, len(by[s]), N)].mean(0)); ys.append(si)
        return np.array(Xs), np.array(ys)
    out = {}
    for N in Ns:
        Xa, ya = fps(trby, N, 400, 10); Xb, yb = fps(teby, N, 400, 20)
        p = LinearDiscriminantAnalysis().fit(Xa, ya).predict(Xb)
        out[N] = (p == yb).mean()
    return pd.Series(out)

fig, ax = plt.subplots(figsize=(9, 4))
for name, feats in FEATURE_SETS.items():
    s = (cross_agg(feats, DA, DB) + cross_agg(feats, DB, DA)) / 2
    ax.plot(s.index, s.values, marker="o", label=name)
ax.axhline(0.90, color="g", ls="--", lw=1, label="90% target")
ax.axhline(0.20, color="k", ls=":", lw=1, label="chance")
ax.set_xscale("log"); ax.set_xlabel("games pooled into one cross-deck fingerprint (N)")
ax.set_ylabel("overall accuracy (mean of both directions)")
ax.set_title("Cross-deck aggregate accuracy vs games pooled")
ax.legend(fontsize=8); fig.tight_layout(); plt.show()
'''

S9_CONCL = """
**Reading §9.** Within each deck (left two bars) every feature set carries a clear signal (~0.4-0.5 vs
0.20 chance). Across decks the accuracy collapses toward chance — the AI's style is **entangled with the
deck**, not a free-floating fingerprint. But the comparison is the point: the **V4 agnostic** set transfers
*better* than the deck-leaky V3 set in both directions, while matching it within-deck — so the
normalization did its job (it removed deck-identity leakage without throwing away behavior). The pure-ratio
subset over-prunes (the absolute-but-behavioral signals like enemy-board tolerance carry real transferable
information). Per-style, only the **slow** archetypes — fatigue and ramp — survive a deck swap once games
are aggregated; aggro and midrange do not (on the all-out Pirate Warrior deck almost everything reads as
aggro, and midrange is the mushy middle on both decks).
"""

new_cells = [
    {"cell_type": "markdown", "id": "v4xdeck_md",    "metadata": {}, "source": S9_MD.strip("\n")},
    {"cell_type": "code",     "id": "v4xdeck_rf",    "metadata": {}, "source": S9_RF.strip("\n"),
     "outputs": [], "execution_count": None},
    {"cell_type": "code",     "id": "v4xdeck_agg",   "metadata": {}, "source": S9_AGG.strip("\n"),
     "outputs": [], "execution_count": None},
    {"cell_type": "markdown", "id": "v4xdeck_concl", "metadata": {}, "source": S9_CONCL.strip("\n")},
]
nb["cells"].extend(new_cells)

# ---- clear all outputs / execution counts ------------------------------------
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", os.path.relpath(DST, os.path.join(HERE, "..", "..")), "cells:", len(nb["cells"]))
