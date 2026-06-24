"""Assemble the V3 notebook by patching the V2 notebook cell-by-cell (by cell id).

Mirrors `_make_v2_notebook.py`: V2 was built by editing specific cells of V1, and
V3 is built by editing specific cells of V2.  V3 = V2 **plus Change B** — keep the
11 whole-game metrics, add the 13 turn-checkpoint trajectory features, point the
parser/cache at v3, and rewrite §3b to draw the board / mana / hand trajectories.

Run:  py -3 Log_Analysis/V3/_make_v3_notebook.py
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "..", "V2", "playstyle_log_distribution_analysis_RenoKazakusMage_v2.ipynb")
DST  = os.path.join(HERE, "playstyle_log_distribution_analysis_RenoKazakusMage_v3.ipynb")
nb = json.load(open(SRC, encoding="utf-8"))


def cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def set_src(cid, text):
    cell(cid)["source"] = text.strip("\n")


def sub_in(cid, pattern, repl, count=1, flags=0):
    c = cell(cid)
    s = "".join(c["source"])
    new, n = re.subn(pattern, repl, s, count=count, flags=flags)
    assert n == count, f"{cid}: expected {count} repl, got {n} for {pattern!r}"
    c["source"] = new


# ---- [2] setup: import v3 parser (+ TRAJ_METRICS) and v3 cache ----------------
sub_in("16808bd4", r'os\.path\.abspath\("playstyle_log_parse_v2\.py"\)',
       'os.path.abspath("playstyle_log_parse_v3.py")')
sub_in("16808bd4", r"from playstyle_log_parse_v2 import build_frames, STYLES, BASE_DIR",
       "from playstyle_log_parse_v3 import build_frames, STYLES, BASE_DIR, TRAJ_METRICS")
sub_in("16808bd4", r'"playstyle_log_metrics_RenoKazakusMage_v2\.pkl"',
       '"playstyle_log_metrics_RenoKazakusMage_v3.pkl"')

# ---- [5] METRICS dict stays the 11; FEATURES = 11 + 13 trajectory -------------
# keep METRICS (drives the 3x4 distribution grids) but extend the clustering FEATURES.
sub_in("93f623df",
       r"FEATURES = \[c for _, \(c, _\) in METRICS\.items\(\)\]   # 11 numeric columns used for clustering",
       'BASE_FEATURES = [c for _, (c, _) in METRICS.items()]            # 11 whole-game V2 metrics\n'
       'TRAJ_FEATURES = list(TRAJ_METRICS)                             # 13 Change-B trajectory metrics\n'
       'FEATURES = BASE_FEATURES + TRAJ_FEATURES                       # 24 features used for clustering')

# ---- [3b] per-turn dynamics -> full trajectory panel --------------------------
set_src("57cfe4cf", '''
tt = turns_df[turns_df["my_turn"] <= 12].copy()
tt["style"] = pd.Categorical(tt["style"], categories=STYLE_ORDER, ordered=True)

fig, ax = plt.subplots(2, 2, figsize=(15, 8)); ax = ax.ravel()
curves = [("board_end",  "Board minions at end of my turn", "board minions"),
          ("mana_spent", "Mana spent during my turn",       "mana spent"),
          ("hand_end",   "Cards in hand at end of my turn",  "cards in hand")]
for a, (col, title, ylab) in zip(ax, curves):
    sns.lineplot(data=tt, x="my_turn", y=col, hue="style", hue_order=STYLE_ORDER,
                 palette=STYLE_COLORS, errorbar=("ci", 95), ax=a, legend=False)
    for cp in (3, 5, 7, 9):
        a.axvline(cp, color="0.8", ls=":", lw=1, zorder=0)   # t3/5/7/9 checkpoints
    a.set_title(title); a.set_xlabel("my turn"); a.set_ylabel(ylab)
handles = [plt.Line2D([0], [0], color=STYLE_COLORS[s], lw=2) for s in STYLE_ORDER]
ax[0].legend(handles, STYLE_ORDER, title="style", fontsize=8)
# turns survived per style (survival-ish curve)
surv = (games_df.groupby("style", observed=True)["n_my_turns"]
        .apply(lambda s: pd.Series({t: (s >= t).mean() for t in range(1, 16)})).unstack().T)
for st in STYLE_ORDER:
    ax[3].plot(surv.index, surv[st], color=STYLE_COLORS[st], marker="o", ms=3, label=st)
ax[3].set_title("Fraction of games still going at my-turn t"); ax[3].set_xlabel("my turn")
ax[3].set_ylabel("share of games"); ax[3].legend(fontsize=8)
fig.suptitle("3b · Per-turn trajectories by style (dotted = t3/5/7/9 checkpoints)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()
''')

# ---- [3] fingerprint heatmap now has 24 columns -> widen + rotate labels -------
sub_in("c3ca5a15", r"fig, ax = plt\.subplots\(figsize=\(13, 3\.4\)\)",
       "fig, ax = plt.subplots(figsize=(20, 3.6))")
sub_in("c3ca5a15",
       r'ax\.set_title\("Play-style fingerprints[^\n]*\)\n',
       'ax.set_title("Play-style fingerprints — per-style mean of each metric '
       '(colour = z-score across the 5 styles)")\n'
       'ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)\n')

# ============================ markdown rewrites ================================
set_src("e7038b5e", '''
# Play-style Distribution & Clustering of the 5 AI Archetypes — `RenoKazakusMage` (v3 · 11 metrics + trajectory)

**Data.** `log/<style>_RenoKazakusMage/` holds 5 folders. Every folder is the **same**
`RenoKazakusMage` deck (a slow, singleton **MAGE** control/highlander list), but driven by a different
SabberStone search/AI play-style: **aggro · control · fatigue · midrange · ramp**. In each log the
folder's style is **P1 (me)**; the opponent (deck *and* style) varies across ~900 games per folder. We
measure **P1 only** and **pool over all opponents**.

**Why v3.** The four *value* styles — **control · fatigue · midrange · ramp** — pilot this slow highlander
Mage so similarly that the 11 whole-game V2 *averages* collapse them onto nearly the same point. v3 keeps
**all 11 V2 averages** (they still carry the aggro-vs-value split) and adds **Change B — trajectory
features**: instead of one game-average, it reads *when* board and mana develop. It samples **board
minions, mana spent, and cards in hand at my-turns 3 / 5 / 7 / 9** plus a **mana slope**, on the intuition
that *midrange commits board early while ramp builds late*, and *ramp's mana-spent curve climbs late while
midrange is flat-high early*. (Change A — new raw ramp/control/fatigue *tell* metrics that need extra
draw / deck-count / mana-crystal log lines — is a separate later stage; **this notebook measures what
trajectory alone buys**, vs the v2 11-metric baseline.)

> **Headline finding (quantified below).** A single game is too noisy to cluster — the same deck plus
> random opponents makes per-game points overlap. But a *play-style is a distribution over games*, not one
> game. Once we look at the **distribution / aggregated fingerprint**, the **5 clusters separate almost
> perfectly**. Both views are shown honestly. Forcing an **aggro** search to pilot a slow highlander Mage
> is the most recognisable misfit, so a single game already classifies `aggro` well; the hard-to-pin
> ceiling is the **ramp / midrange** value styles — which is exactly what the trajectory features target.''')

set_src("45bea743", '''
## 1 · Setup & parse

The parser lives in [`playstyle_log_parse_v3.py`](playstyle_log_parse_v3.py) — V2's fixed 11-metric parser
**plus Change B**, and it needs **no new log-line parsing**: V2 already sampled P1 board size and hand size
at the end of every P1 turn and knew each play's mana cost, so v3 simply *surfaces* that already-tracked
per-turn data. It now emits `board_end` and `mana_spent` into `turns_df` (for the §3b curves) and computes
**13 trajectory columns** into `games_df`: `board_at_t{3,5,7,9}`, `mana_at_t{3,5,7,9}`, `mana_slope`,
`hand_at_t{3,5,7,9}`. It still attributes events to **P1** by entity id (`PlayCardTask => [P1]`,
`PayPhase 'card[id]'`, `HeroPowerTask`, `SummonPhase ... Board of P1`, hero-entity ids for damage taken)
and joins each folder's `summary.csv` (winner, turns, seconds, start player).
`build_frames(deck="RenoKazakusMage")` selects the folder family; results cache to
`playstyle_log_metrics_RenoKazakusMage_v3.pkl` — delete it to force a full re-parse.''')

set_src("5e18783f", '''
### 1b · Metric glossary

All metrics are for **P1 (the folder's play-style)**, one value per game.

**The 11 whole-game metrics (kept verbatim from v2 — they carry the aggro-vs-value split):**

| metric (column) | play-style aspect | definition |
|---|---|---|
| `n_my_turns` | game length | number of P1 turns played |
| `avg_cards_in_hand` | resource holding | mean hand size at the end of each P1 turn |
| `mana_eff` | mana efficiency | mana spent / mana available |
| `avg_card_cost` | curve centre | mean mana cost of cards P1 played |
| `minion_fraction` | card-type mix | minions / all cards played |
| `face_attack_ratio` | aggression direction | face attacks / all attacks |
| `attacks_per_turn` | aggression intensity | total P1 attacks / turns |
| `enemy_minions_killed_per_turn` | removal / trading | enemy minions to graveyard on P1 turns / turns |
| `avg_board_minions` | board presence | mean count of P1 minions in play at end of P1 turns |
| `taken_dmg_per_turn` | damage absorbed | damage to P1's hero / turns |
| `hp_per_turn` | hero-power tempo | hero-power uses / turns |

**Change B — trajectory features (new in v3).** The *same* board / mana / hand quantities, but
**checkpointed by turn** instead of averaged, so the four value styles separate by *timing*:

| metric (column) | target style | definition |
|---|---|---|
| `board_at_t3/5/7/9` | midrange (early board) vs ramp (late) | P1 board minions at end of my-turn 3/5/7/9 |
| `mana_at_t3/5/7/9` | ramp (late ramp-up) vs midrange (flat-high) | mana P1 *spent* during my-turn 3/5/7/9 |
| `mana_slope` | ramp | slope of (my-turn → mana spent) across the game |
| `hand_at_t3/5/7/9` | timing of hoarding | P1 hand size at end of my-turn 3/5/7/9 |

A checkpoint past a game's length is `NaN` and the §4 matrix median-imputes it; since aggro games are
short, its late checkpoints fall back to the median, while the value styles — which reach turn 9 — get
genuine late-game values. **24 features total (11 + 13)** feed the clustering from §4 on.''')

set_src("84ad946d", '''
### 3b · Per-turn trajectories by style

The whole-game averages collapse the value styles; their *trajectories* need not. Board minions, mana
spent and cards in hand are plotted **against my-turn** (mean ± 95% CI), with the **t3/5/7/9 checkpoints**
dotted — this is exactly the signal the new trajectory features encode. Watch **midrange commit board
earlier than ramp**, and **ramp's mana-spent curve keep climbing late** while flatter styles plateau.''')

set_src("2348a354", '''
## 4 - Feature matrix (standardised)

Stack the per-game metrics into `X` (z-scored), then impute any remaining gaps (mostly late-turn
checkpoints in short games) with the column median. `X` now has **24 columns** = the 11 whole-game V2
metrics **+ 13 Change-B trajectory features**.''')

set_src("de97b98a", '''
## 7 - Conclusion

* **Yes, the 5 archetypes form 5 clusters — but only as distributions, not as single games.** With the
  *same* `RenoKazakusMage` deck and random opponents, one game carries weak style signal: per-game
  KMeans/GMM give a low ARI and the PCA/t-SNE clouds largely overlap (§5).
* **The signal is nonetheless real**: a supervised classifier scores well above the 0.20 chance line
  (§5b), and the styles differ systematically in the overlaid distributions and fingerprint heatmap (§3).
* **Aggregation reveals the structure** (§6): averaging games into play-style fingerprints makes the
  5 clusters separate **almost perfectly** (unsupervised ARI → ~1.0 as games are pooled).
* **Reading of the styles (control deck)**: *aggro* = the clear outlier — **shortest games**, **highest
  face-attack ratio**, but it **mis-pilots** the slow deck: lowest mana efficiency, least trading, a
  **hoarded hand** and the **most damage taken per turn**. *ramp / midrange* = **longest games**, best
  mana efficiency, most board development and trading, least damage taken. *control / fatigue* sit between.
* **What v3 adds (Change B):** keeping the 11 averages and adding turn-checkpointed **trajectory**
  features (board / mana / hand at t3/5/7/9 + mana slope) targets the value-style overlap *directly* —
  midrange's early board vs ramp's late build, and ramp's late mana ramp-up. §5b (per-game RF accuracy)
  and §8 (games-to-classify per style) are the scoreboard for whether trajectory tightens the hard-to-pin
  **ramp / midrange / control / fatigue** group relative to v2's 11-metric baseline.

*Practical note*: to **classify an unknown player's style** from this engine, aggregate several of their
games before clustering/scoring — though for this deck a single **aggro** game is already a giveaway.
**§8 quantifies exactly how many games are needed.** (All figures recompute from the 24-feature v3 matrix
when the notebook runs.)''')

# §8 reading: the code recomputes correctly from the 24-feature matrix; flag that the prose numbers below
# are the v2 baseline, to be re-read against this run's outputs.
sub_in("a3d1e93d", r"\*\*Reading §8\.\*\*",
       "**Reading §8.** *(The bullets below are the **v2 11-metric baseline**; compare them against this "
       "run's outputs to see what the trajectory features change — v3 should help most on the value "
       "styles, ramp/midrange/control.)*")

# ---- clear all outputs / execution counts ------------------------------------
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", os.path.relpath(DST, os.path.join(HERE, "..", "..")), "cells:", len(nb["cells"]))
