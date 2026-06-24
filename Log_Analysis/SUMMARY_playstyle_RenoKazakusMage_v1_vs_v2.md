# Play-style Distribution & Clustering — `RenoKazakusMage` — V1 vs V2 Summary

> Self-contained text summary of two Jupyter notebooks for hand-off to another Claude session.
> No figures; all numbers below are the **actual computed outputs** read from the notebooks.
>
> - **V1**: `Log_Analysis/V1/playstyle_log_distribution_analysis_RenoKazakusMage.ipynb` (parser `playstyle_log_parse.py`, **15 metrics**)
> - **V2**: `Log_Analysis/V2/playstyle_log_distribution_analysis_RenoKazakusMage_v2.ipynb` (parser `playstyle_log_parse_v2.py`, **fixed 11 metrics**)

---

## 1. Shared context (identical in both notebooks)

**Goal.** Test whether 5 AI *play-styles* form 5 separable clusters when they all pilot the **same deck**.

**Data.** `log/<style>_RenoKazakusMage/` holds 5 folders. Every folder is the **same** `RenoKazakusMage`
deck — a slow, singleton (highlander) **MAGE** control list — but driven by a different SabberStone
search/AI play-style: **aggro · control · fatigue · midrange · ramp**. In each log the folder's style is
**P1 ("me")**; the opponent (deck *and* style) varies across ~900 games per folder. Only **P1** is
measured, **pooled over all opponents**.

**Dataset size (both notebooks, same parse):**
- `games_df` = **4492 games** — aggro 900, control 898, fatigue 899, midrange 899, ramp 896
- `cards_df` = 48071 rows, `turns_df` = 42941 rows
- Overall P1 win rate = **46.7%**

**Central thesis.** *"A play-style is a distribution over games, not a single game."* One game of the same
deck vs a random opponent is a weak style signal; the structure only appears once games are **aggregated**.

---

## 2. Pipeline (the section-by-section structure is IDENTICAL in V1 and V2)

1. **§1 Setup & parse.** A standalone parser `.py` reads every verbose log, attributes events to **P1** by
   entity id (`PlayCardTask => [P1]`, `PayPhase 'card[id]'`, `HeroPowerTask`, hero-entity ids for face/
   damage), and joins each folder's `summary.csv` (winner, turns, seconds, start player). Output is a tidy
   **one-row-per-game** table, cached to a `.pkl` (delete to force re-parse).
2. **§1b Metric glossary.** Defines every per-game metric (see §3 below for the two metric sets).
3. **§2 Per-style distribution grids.** One grid of per-game metric histograms/KDEs **per style folder**
   (dashed line = that style's mean).
4. **§3 Overlaid distributions + fingerprint heatmap.** All 5 styles' KDEs on one axis per metric, plus a
   heatmap of **per-style metric means** (z-scored across styles) = the "play-style fingerprint."
5. **§3b Per-turn dynamics.** Mean cards-in-hand vs turn number (95% CI) and a survival-style curve
   (fraction of games still going at my-turn *t*).
6. **§4 Feature matrix.** Stack per-game metrics → `X`, z-scored (`StandardScaler`), NaNs median-imputed.
7. **§5 Per-game unsupervised clustering (the honest baseline).** Project `X` to 2-D (**PCA**, **t-SNE**);
   run **KMeans** and **GMM** with k=5; score vs true style with **ARI / AMI / silhouette**. Expectation:
   single-game clouds **overlap**.
8. **§5b Supervised sanity check.** **RandomForest** (5-fold CV accuracy) + feature importance + **LDA**
   2-D projection. Accuracy ≫ 20% chance ⇒ the per-game signal is real, just noisy.
9. **§6 Distribution-level clustering.** Build **bootstrap "fingerprints"**: average `N` random games of a
   style into one vector, repeat many times → a cloud per style. Cluster the fingerprints **unsupervised**
   (KMeans k=5) and track **ARI vs N**. As N grows, opponent noise averages out and 5 clean clusters emerge.
10. **§7 Conclusion** (narrative).
11. **§8 How many games to classify a player?** Held-out **50/50 split per style**; learn LDA + centroids on
    the train half's fingerprints; classify fingerprints made of **N held-out games**. Q1 = accuracy vs N
    per style; Q2 = single-game (N=1) confusion matrix.

---

## 3. The ONLY substantive difference: the metric set (and its parser)

|                | V1                                                | V2                                                          |
|----------------|---------------------------------------------------|-------------------------------------------------------------|
| Parser         | `playstyle_log_parse.py`                          | `playstyle_log_parse_v2.py` (adds board/graveyard tracking) |
| Metric count   | **15**                                            | **fixed 11** (each mapped 1:1 to a play-style aspect)       |
| `games_df`     | (4492, **35**)                                    | (4492, **22**)                                              |
| Cache file     | `playstyle_log_metrics_RenoKazakusMage.pkl`       | `playstyle_log_metrics_RenoKazakusMage_v2.pkl`              |
| Feature matrix | (4492, **15**)                                    | (4492, **11**)                                              |

**Shared 7 metrics (in both):** `n_my_turns`, `avg_cards_in_hand`, `mana_eff`, `avg_card_cost`,
`face_attack_ratio`, `attacks_per_turn`, `hp_per_turn`.

**V1-only (8, dropped in V2):**
- `time_per_turn` — engine **search seconds** per turn (an engine-internal "search-effort" signal, not gameplay)
- `cards_per_turn`, `minions_per_turn`, `mana_per_turn` — tempo rates
- `first_turn` — first turn a card is played
- `max_cards_in_hand` — peak hand size
- `face_dmg_per_turn` — damage dealt to enemy hero per turn
- `taken_dmg` — total damage taken (game total)

**V2-only (4, new — board/trading state):**
- `minion_fraction` — minions / all cards played (card-type mix)
- `enemy_minions_killed_per_turn` — removal / trading intensity
- `avg_board_minions` — board presence
- `taken_dmg_per_turn` — damage absorbed, **normalized per turn** (the per-turn version of V1's `taken_dmg`)

**In one line:** V2 **drops the engine-internal search-time metric and several raw tempo metrics**, and
**adds board-state / trading metrics**, giving a cleaner, purely **behavioral / game-state** 11-metric set
where each feature maps to a distinct play-style dimension (game length, resource holding, mana efficiency,
curve centre, card-type mix, aggression direction, aggression intensity, removal/trading, board presence,
damage absorbed, hero-power tempo).

---

## 4. Results — actual computed numbers, side by side

| Result                                               | V1 (15 metrics)     | V2 (11 metrics)        |
|------------------------------------------------------|---------------------|------------------------|
| Per-game **KMeans** ARI / AMI / silhouette           | 0.094 / 0.136 / 0.140 | 0.094 / 0.132 / 0.137 |
| Per-game **GMM** ARI / AMI / silhouette              | 0.063 / 0.089 / 0.031 | 0.074 / 0.105 / 0.066 |
| Supervised **RandomForest** 5-fold CV acc (chance 0.20) | **0.486**        | **0.448**              |
| Aggregated **fingerprint** KMeans ARI (N=120)        | **0.993**           | **0.990**              |
| **Single-game** held-out LDA accuracy (chance 0.20)  | **0.485**           | **0.435**              |
| Games to reach **80% overall** (LDA)                 | N = **12**          | N = **20**             |
| Games to reach **90% overall** (LDA)                 | N = **30**          | N = **50**             |
| Games to **90% per style**: aggro                    | 2                   | 3                      |
| &nbsp;&nbsp;&nbsp;&nbsp;fatigue                       | 20                  | 20                     |
| &nbsp;&nbsp;&nbsp;&nbsp;midrange                     | 30                  | 30                     |
| &nbsp;&nbsp;&nbsp;&nbsp;control                      | 30                  | **80**                 |
| &nbsp;&nbsp;&nbsp;&nbsp;ramp                          | 50                  | **>120**               |

### Play-style fingerprints (per-style means — shared metrics agree exactly across V1/V2)

| style    | n_my_turns | mana_eff | avg_card_cost | avg_cards_in_hand | face_attack_ratio | attacks_per_turn | hp_per_turn |
|----------|-----------:|---------:|--------------:|------------------:|------------------:|-----------------:|------------:|
| aggro    | 6.82       | 0.566    | 3.06          | 5.42              | 0.839             | 0.183            | 0.536       |
| control  | 9.94       | 0.722    | 3.32          | 4.49              | 0.651             | 0.474            | 0.314       |
| fatigue  | 9.17       | 0.680    | 3.24          | 4.60              | 0.761             | 0.397            | 0.347       |
| midrange | 10.63      | 0.734    | 3.59          | 4.51              | 0.693             | 0.606            | 0.280       |
| ramp     | 11.24      | 0.740    | 3.63          | 4.51              | 0.607             | 0.667            | 0.272       |

**V2-only board/trading means** (confirm the same story):

| style    | minion_fraction | enemy_minions_killed/turn | avg_board_minions | taken_dmg/turn |
|----------|----------------:|--------------------------:|------------------:|---------------:|
| aggro    | 0.439 (lowest)  | 0.022 (lowest)            | 0.058 (lowest)    | 5.59 (highest) |
| control  | 0.517           | 0.185 (highest)           | —                 | 3.50           |
| fatigue  | 0.494           | 0.151                     | —                 | 4.03           |
| midrange | 0.521           | 0.169                     | —                 | 2.85           |
| ramp     | 0.537 (highest) | 0.177                     | —                 | 2.56 (lowest)  |

---

## 5. Interpretation — what's the same, what changed

**Same qualitative conclusion in both notebooks:**
- Single game = weak signal: per-game unsupervised ARI ≈ **0.06–0.09**, PCA/t-SNE clouds overlap.
- Signal is real but noisy: supervised RF ≈ **0.45–0.49** (chance 0.20).
- **Aggregation reveals the structure**: averaging games into fingerprints → **5 clean clusters**,
  unsupervised ARI ≈ **0.99** by N≈120. This headline result is **robust to the metric set** (0.993 vs 0.990).
- **`aggro` is the clear outlier / one-game giveaway**: an aggressive search mis-piloting a slow highlander
  Mage = shortest games, highest face-attack ratio, fewest attacks/turn, **hoards its hand** (highest avg
  cards in hand), almost no board, least trading, and **takes the most damage**.
- **`ramp` / `midrange` are the hard ceiling**: longest games, best mana efficiency, most board development
  and trading, least damage taken — they converge on "spend efficiently for value" and overlap each other.
  `control` / `fatigue` sit in between (fatigue grinds slightly longer toward deck-out).

**What the V2 metric change actually did (the meaningful diff):**
- V2's purely **behavioral** 11-metric set is **slightly less discriminative per game**: RF accuracy drops
  **0.486 → 0.448**, single-game accuracy drops **0.485 → 0.435**.
- V2 needs **more games to lock in a style**: overall 90% at **N=50 vs N=30**; and the value styles get much
  harder — **control 30 → 80**, **ramp 50 → >120** games.
- Most plausible cause: V1's dropped **`time_per_turn`** is almost a direct readout of the AI's search
  effort (engine-internal, not in-game behavior), and the extra tempo metrics add separating power.
  Removing them costs per-game accuracy, concentrated on the already-overlapping value styles (control/ramp).
- **Trade-off:** V2 buys a cleaner, more interpretable, purely game-state feature set (no engine-leak
  feature; each metric = one play-style dimension; adds board/trading signal) at the cost of needing more
  games per classification. The end conclusion — *5 distributions = 5 clusters* — is unchanged.

---

## 6. Caveats about the V2 notebook's prose (trust the code outputs over the text)

The V2 markdown was **copied from V1 and not fully updated**, so its narrative understates the diff:
- Header claims *"All outputs below are cleared and regenerate when you run the notebook"* and that figures
  *"may shift"* — but the notebook **does contain computed outputs** (the V2 numbers in §4 above are real).
- §5 markdown still says *"Project the **15-D** per-game points"* — V2 is **11-D**.
- §3 markdown still lists *"cards/turn, mana spent/turn"* as separating metrics — those are **not** in V2's
  11-metric set.
- §8 "Reading" cell still quotes **V1's** numbers (single-game ≈ 0.48, aggro ~86%, ramp ~50, ~30 games →
  90%). V2's **actual** outputs are single-game **0.435**, overall 90% at **N=50**, **ramp >120**.

When the two conflict, the **code-cell outputs (this summary's §4) are authoritative**; the V2 prose is a
not-yet-refreshed mirror of V1.
