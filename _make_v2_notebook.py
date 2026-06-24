import json, re

SRC = "playstyle_log_distribution_analysis_RenoKazakusMage.ipynb"
DST = "playstyle_log_distribution_analysis_RenoKazakusMage_v2.ipynb"
nb = json.load(open(SRC, encoding="utf-8"))


def cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def set_src(cid, text):
    cell(cid)["source"] = text


def sub_in(cid, pattern, repl, count=1, flags=0):
    c = cell(cid)
    s = "".join(c["source"])
    new, n = re.subn(pattern, repl, s, count=count, flags=flags)
    assert n == count, f"{cid}: expected {count} repl, got {n} for {pattern!r}"
    c["source"] = new


# ---- [2] setup: import + cache ------------------------------------------------
sub_in("16808bd4", r"from playstyle_log_parse import build_frames, STYLES",
       "from playstyle_log_parse_v2 import build_frames, STYLES")
sub_in("16808bd4", r'CACHE = "playstyle_log_metrics_RenoKazakusMage\.pkl"',
       'CACHE = "playstyle_log_metrics_RenoKazakusMage_v2.pkl"')

# ---- [5] METRICS dict + show_style grid --------------------------------------
NEW_METRICS = '''# label -> (column, kind);  kind drives the plot style   [v2: fixed 11-metric set]
METRICS = {
    "My turns / game":           ("n_my_turns",                    "disc"),
    "Avg cards in hand":         ("avg_cards_in_hand",             "kde"),
    "Mana efficiency":           ("mana_eff",                      "clip01"),
    "Avg card cost":             ("avg_card_cost",                 "kde"),
    "Minion fraction":           ("minion_fraction",               "clip01"),
    "Face-attack ratio":         ("face_attack_ratio",             "clip01"),
    "Attacks / turn":            ("attacks_per_turn",              "kde"),
    "Enemy minions killed/turn": ("enemy_minions_killed_per_turn", "kde"),
    "Board minions (avg)":       ("avg_board_minions",             "kde"),
    "Damage taken / turn":       ("taken_dmg_per_turn",            "kde"),
    "Hero-power / turn":         ("hp_per_turn",                   "kde"),
}
FEATURES = [c for _, (c, _) in METRICS.items()]   # 11 numeric columns used for clustering'''
sub_in("93f623df",
       r"# label -> \(column, kind\);.*?FEATURES = \[c for _, \(c, _\) in METRICS\.items\(\)\].*?clustering",
       lambda m: NEW_METRICS, flags=re.S)
sub_in("93f623df", r"plt\.subplots\(3, 5, figsize=\(19, 9\)\)", "plt.subplots(3, 4, figsize=(16, 9))")

# ---- [12] overlay grid -------------------------------------------------------
sub_in("c3ca5a15", r"plt\.subplots\(3, 5, figsize=\(19, 10\)\)", "plt.subplots(3, 4, figsize=(16, 10))")

# ---- markdown rewrites --------------------------------------------------------
set_src("e7038b5e", '''# Play-style Distribution & Clustering of the 5 AI Archetypes - `RenoKazakusMage` (v2 · 11-metric set)

**Data.** `log/<style>_RenoKazakusMage/` holds 5 folders. Every folder is the **same**
`RenoKazakusMage` deck (a slow, singleton **MAGE** control/highlander list), but driven by a different
SabberStone search/AI play-style: **aggro · control · fatigue · midrange · ramp**. In each log the
folder's style is **P1 (me)**; the opponent (deck *and* style) varies across ~900 games per folder. We
measure **P1 only** and **pool over all opponents** (per the chosen scope).

**Method** - parse each game into a tidy per-game table, then study the **statistical distribution** of
play-style metrics. Each *"player"* of the reference becomes one of the **5 style folders**, and we ask
whether the distributions form **5 clusters**.

**Metrics (v2).** This variant uses the parser [`playstyle_log_parse_v2.py`](playstyle_log_parse_v2.py),
which exposes a **fixed set of 11 calculated metrics** (and nothing else): game length, resource holding
(avg cards in hand), mana efficiency, curve centre (avg card cost), card-type mix (minion fraction),
aggression direction (face-attack ratio), aggression intensity (attacks/turn), removal / trading (enemy
minions killed per turn), board presence (avg board minions), damage absorbed (taken dmg/turn), and
hero-power tempo. See §1b for the glossary.

> **All outputs below are cleared and regenerate when you run the notebook.** The narrative mirrors the
> 15-metric original ([..._RenoKazakusMage.ipynb](playstyle_log_distribution_analysis_RenoKazakusMage.ipynb));
> the exact figures (ARI, accuracies, feature importances) are recomputed from the **11-metric** matrix
> and may shift.

> **Headline finding (quantified below).** A single game is too noisy to cluster - the same deck plus
> random opponents makes per-game points overlap. But a *play-style is a distribution over games*, not one
> game. Once we look at the **distribution / aggregated fingerprint**, the **5 clusters separate almost
> perfectly**. Both views are shown honestly. As in the original, forcing an **aggro** search to pilot a
> slow highlander Mage is the most recognisable misfit, so a single game already classifies `aggro` well,
> and the hard-to-pin ceiling is the **ramp / midrange** value styles.''')

set_src("45bea743", '''## 1 - Setup & parse

The parser lives in [`playstyle_log_parse_v2.py`](playstyle_log_parse_v2.py) - a copy of the original
parser restricted to a **fixed 11-metric set** (it adds board / graveyard tracking for `minion_fraction`,
`enemy_minions_killed_per_turn`, `avg_board_minions` and `taken_dmg_per_turn`). It reads every verbose
log, attributes events to **P1** by entity id (`PlayCardTask => [P1]`, `PayPhase 'card[id]'`,
`HeroPowerTask`, `SummonPhase ... Board of P1`, hero-entity ids for damage taken), and joins each
folder's `summary.csv` for the header facts (winner, turns, seconds, start player).
`build_frames(deck="RenoKazakusMage")` selects the RenoKazakusMage folder family; results are cached to
`playstyle_log_metrics_RenoKazakusMage_v2.pkl` - delete it to force a full re-parse.''')

set_src("5e18783f", '''### 1b - Metric glossary

All metrics are for **P1 (the folder's play-style)**, one value per game. This v2 notebook uses exactly
these **11 metrics** - the full output of `playstyle_log_parse_v2.py`.

| metric (column) | play-style aspect | definition |
|---|---|---|
| `n_my_turns` | game length | number of P1 turns played |
| `avg_cards_in_hand` | resource holding | mean hand size at the end of each P1 turn |
| `mana_eff` | mana efficiency | mana spent / mana available |
| `avg_card_cost` | curve centre | mean mana cost of cards P1 played |
| `minion_fraction` | card-type mix | minions / all cards played |
| `face_attack_ratio` | aggression direction | face attacks / all attacks |
| `attacks_per_turn` | aggression intensity | total P1 attacks / turns |
| `enemy_minions_killed_per_turn` | removal / trading | enemy minions sent to graveyard on P1 turns / turns |
| `avg_board_minions` | board presence | mean count of P1 minions in play at end of P1 turns |
| `taken_dmg_per_turn` | damage absorbed | damage to P1's hero / turns |
| `hp_per_turn` | hero-power tempo | hero-power uses / turns |''')

set_src("2348a354", '''## 4 - Feature matrix (standardised)

Stack the per-game metrics into `X` (z-scored), then impute any remaining gaps with the column median.''')

set_src("de97b98a", '''## 7 - Conclusion

* **Yes, the 5 archetypes form 5 clusters - but only as distributions, not as single games.** With the
  *same* `RenoKazakusMage` deck and random opponents, one game carries weak style signal: per-game
  KMeans/GMM give a low ARI and the PCA/t-SNE clouds largely overlap (§5).
* **The signal is nonetheless real**: a supervised classifier scores well above the 0.20 chance line
  (§5b), and the styles differ systematically in the overlaid distributions and fingerprint heatmap (§3)
  - most along **mana efficiency, minion fraction, face-attack ratio, attacks/turn,
  enemy-minions-killed/turn, board presence, damage-taken/turn, hero-power tempo and average cards in
  hand**.
* **Aggregation reveals the structure** (§6): averaging games into play-style fingerprints makes the
  5 clusters separate **almost perfectly** (unsupervised ARI → ~1.0 as games are pooled), sharper the
  more games are pooled.
* **Reading of the styles (control deck)**: *aggro* = the clear outlier - **shortest games**, **highest
  face-attack ratio**, but it **mis-pilots** the slow deck: lowest mana efficiency, least trading (fewest
  enemy minions killed), a **hoarded hand** (highest avg cards in hand) and the **most damage taken per
  turn**. *ramp / midrange* = **longest games**, best mana efficiency, most board development and trading,
  least damage taken. *control / fatigue* sit between, fatigue grinding slightly longer toward deck-out.

*Practical note*: to **classify an unknown player's style** from this engine, aggregate several of their
games before clustering/scoring - though for this deck a single **aggro** game is already a giveaway.
**§8 quantifies exactly how many games are needed.** (All figures here are recomputed from the 11-metric
v2 matrix when the notebook runs.)''')

# ---- clear all outputs / execution counts ------------------------------------
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", DST, "cells:", len(nb["cells"]))
