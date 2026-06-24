# Weekly Report — Play-style Distribution Analysis (`Log_Analysis/`)

**Date:** 2026-06-25
**Scope:** `Log_Analysis/` — versions V1 → V4
**Author's one-line thesis:** *"A play-style is a distribution over games, not a single game."*

---

## 0. Executive summary

I am testing whether the **5 SabberStone AI play-styles** (aggro / control / fatigue /
midrange / ramp) form **5 separable clusters** when they all pilot the **same deck**, so the
style signal is the *AI's decision philosophy*, not the deck.

Across four versions I established:

1. **One game is a weak signal.** Per-game unsupervised clustering barely beats random
   (ARI ≈ 0.06–0.09); per-game supervised accuracy ≈ 0.43–0.49 (chance = 0.20).
2. **Aggregation reveals the structure.** Averaging ~120 games of a style into one
   "fingerprint" vector yields **near-perfect, fully unsupervised clusters (ARI ≈ 0.99)**.
   This is the headline result and it is robust to the feature set.
3. **Trajectory + "tell" features (V3) sharpen the hard styles.** Adding *when* board/mana/hand
   develop, plus style-targeted tells, raised per-game accuracy 0.448 → 0.489 and roughly
   **halved the games needed to identify the "value" styles** (e.g. fatigue: 20 → 8 games).
4. **Style does NOT transfer across decks at the single-game level (V4).** Train on the Mage,
   test on the Warrior (or vice-versa) and single-game accuracy collapses to ≈ 0.22–0.29
   (chance 0.20). Only **fatigue** (and partly ramp) carry a deck-independent signature. The
   styles are still **deck-entangled** — the main open problem.

---

## 1. Research question & data

**Question.** Do AI play-styles separate from deck identity? If the same deck is piloted by 5
different search policies, can we recover the policy from the game logs — and how many games do
we need?

**Data.** SabberStone self-play verbose logs under `log/<style>_<deck>/`. Two deck families,
five styles each:

| Deck family | aggro | control | fatigue | midrange | ramp | total |
|-------------|------:|--------:|--------:|---------:|-----:|------:|
| RenoKazakusMage (slow highlander control Mage) | 900 | 898 | 899 | 899 | 896 | **4 492** |
| AggroPirateWarrior (fast aggro Warrior) | 900 | 900 | 900 | 900 | 900 | **4 500** |
| **Total** | | | | | | **8 992** |

- In each folder the style is **P1 ("me")**; the **opponent (deck *and* style) varies** across
  ~900 games. We measure **P1 only, pooled over all opponents**.
- Win rates: Mage **46.7%**, Warrior **62.6%**, overall **54.6%** (the aggro Warrior list simply
  wins more — a deck effect, not a style effect, and a reason to normalize in V4).
- V1–V3 study the **Mage only**; V4 adds the **Warrior** to enable a cross-deck transfer test.

---

## 2. The pipeline (identical section structure in every version)

1. **Parse.** A standalone parser (`playstyle_log_parse*.py`) reads every verbose log and
   attributes events to P1 by entity id (card plays, mana paid, hero power, attacks, damage,
   draws, board/hand sampled at each P1 turn). Joins each folder's `summary.csv` (winner,
   turns, seconds, start player). Output = one tidy **row per game**, cached to a `.pkl`.
2. **Metric glossary** — every per-game metric defined and mapped to a play-style aspect.
3. **Per-style distribution grids** — histograms/KDEs of each metric, per style.
4. **Overlaid distributions + fingerprint heatmap** — all 5 styles per metric; z-scored
   per-style means = the "play-style fingerprint."
5. **Per-turn dynamics** — mean cards-in-hand vs turn (95% CI) + game-length survival curve.
6. **Feature matrix** — stack metrics → `X`, z-score (StandardScaler), median-impute NaNs.
7. **Per-game unsupervised clustering** (the honest baseline) — PCA / t-SNE + KMeans / GMM
   (k=5), scored vs the true style with ARI / AMI / silhouette. *Expectation: clouds overlap.*
8. **Per-game supervised sanity check** — RandomForest (5-fold CV) + feature importance + LDA.
   Accuracy ≫ 20% chance ⇒ the per-game signal is real, just noisy.
9. **Distribution-level clustering** — build **bootstrap fingerprints** (average N random games
   of a style), cluster them unsupervised, track **ARI vs N**. As N grows, opponent noise
   averages out and 5 clean clusters emerge.
10. **"How many games to classify a player?"** — held-out 50/50 split; learn on the train half's
    fingerprints; classify fingerprints of N held-out games. Report smallest **N to reach 90%**
    per style + the single-game (N=1) confusion matrix.

V4 adds an **11th step: cross-deck transfer** — train on one deck, test on the other.

---

## 3. Version-by-version: what changed and why

### V1 — first end-to-end pass (15 metrics)
Established the whole pipeline and the central thesis. Metric set included some
**engine-internal** signals — notably `time_per_turn` (the AI's *search seconds* per turn),
which is a near-direct readout of search effort rather than in-game behavior — plus several raw
tempo rates.

### V2 — clean, purely behavioral metric set (11 metrics)
Dropped the engine-leak metric (`time_per_turn`) and raw tempo rates; **added board/trading
state** (`minion_fraction`, `enemy_minions_killed_per_turn`, `avg_board_minions`,
`taken_dmg_per_turn`). Each of the 11 maps to one play-style dimension. **Trade-off:** slightly
less discriminative per game (it loses the "search-effort" leak) but far more interpretable and
defensible — no feature that secretly encodes the AI's compute budget.

### V3 — trajectory ("Change B") + tells ("Change A")
The four "value" styles (control/fatigue/midrange/ramp) collapse to nearly the same point under
*whole-game averages*. V3 keeps the 11 V2 averages and adds:
- **Change B — trajectory (13 features):** board / mana / hand at **my-turn 3, 5, 7, 9** + the
  mana-spend slope. Captures *when* the game develops, not just the end-state average. Needs **no
  new log parsing** — it surfaces per-turn data the V2 parser already tracked.
- **Change A — tells (6 raw, 5 used):** style-targeted features —
  `proactive_ratio` [control], `cards_drawn_per_turn` & `cards_left_in_deck` [fatigue],
  `first_minion_turn` [midrange], `max_card_cost` [ramp], `extra_mana_crystals` [ramp, **dead**
  on these ramp-less highlander lists, so excluded from clustering].

### V4 — deck-agnostic, turn-rate-normalized + cross-deck test
Goal: a feature set that fingerprints the AI's *philosophy* on **any** deck. Two changes:
- **Turn-rate normalization** — every raw count is re-expressed as a per-turn rate or bounded
  fraction (e.g. `avg_cards_in_hand → hand_fill_ratio`, `mana_at_t{k} → mana_eff_t{k}`,
  `first_minion_turn → first_minion_frac`), so features stop encoding game length / deck curve.
- **Four universal "currency" metrics** (Mana / Cards / Board / Life), each a deck-independent
  behavioral ratio: `face_dmg_per_turn` [life], `mana_floated_per_turn` [mana/greed],
  `avg_enemy_board_minions` [board tolerance], `value_turn_fraction` [card investment].
- **Feature groups:** `AGNOSTIC_FEATURES` (meant to transfer) vs `DECK_DEP_FEATURES`
  (cost/length columns that leak deck identity, excluded from the agnostic set).
- **New experiment:** train on Mage → test on Warrior, and vice-versa.

| | V1 | V2 | V3 (full) | V4 |
|---|---|---|---|---|
| Feature count | 15 | 11 | 29 (11+13+5) | up to 29 agnostic |
| Feature philosophy | mixed (incl. engine leak) | clean behavioral | + trajectory + tells | + normalized, deck-agnostic |
| Decks | Mage | Mage | Mage | **Mage + Warrior** |
| Key question | does aggregation work? | same, cleaner features | separate the value styles | does style transfer decks? |

---

## 4. Results (all numbers are live re-computations from the cached frames)

### 4.1 Per-game supervised accuracy (RandomForest, 5-fold CV; chance = 0.20)

| Feature set | overall | aggro | control | fatigue | midrange | ramp |
|---|---:|---:|---:|---:|---:|---:|
| V1 (15, incl. search-time) | **0.486** | — | — | — | — | — |
| V2 baseline (11) | 0.448 | 0.847 | 0.233 | 0.379 | 0.318 | 0.464 |
| V3 + B trajectory (24) | 0.472 | 0.853 | 0.253 | 0.434 | 0.323 | 0.498 |
| V3 + A tells (16) | 0.482 | 0.846 | 0.258 | **0.529** | 0.294 | 0.484 |
| **V3 full A+B (29)** | **0.489** | 0.854 | 0.264 | 0.526 | 0.315 | 0.487 |

*Read:* aggro is trivially separable (~0.85 every version). Trajectory mainly helps **fatigue**
(0.379 → 0.526); midrange stays the hardest single-game style (~0.31).

### 4.2 How many games to identify a style? (held-out LDA, smallest N to reach 90%)

| Feature set | single-game acc | overall | aggro | control | fatigue | midrange | ramp |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 baseline (11) | 0.435 | 50 | 3 | 80 | 20 | 30 | >120 |
| V3 + B (24) | 0.446 | 50 | 3 | 50 | 20 | 50 | 120 |
| V3 + A (16) | 0.469 | 50 | 2 | 50 | 8 | 30 | >120 |
| **V3 full A+B (29)** | **0.470** | **30** | 2 | 30 | **8** | 50 | 120 |

*Read:* full V3 cuts games-to-90% **overall 50 → 30**, **control 80 → 30**, **fatigue 20 → 8**.
Ramp remains the ceiling (it has no distinguishing "tell" on a ramp-less deck).

### 4.3 The headline aggregation result (V1/V2, robust to feature set)

| | V1 (15) | V2 (11) |
|---|---:|---:|
| Per-game KMeans ARI / AMI / silhouette | 0.094 / 0.136 / 0.140 | 0.094 / 0.132 / 0.137 |
| **Aggregated fingerprint KMeans ARI (N=120)** | **0.993** | **0.990** |

→ Single games overlap; **pooled fingerprints separate almost perfectly, fully unsupervised.**

### 4.4 V4 cross-deck transfer (the new result)

**Within-deck reference** (RF 5-fold CV; chance 0.20) — normalization costs nothing in-deck:

| Feature set | Mage | Warrior |
|---|---:|---:|
| V3-full (deck-leaky, 29) | 0.489 | 0.393 |
| **V4 agnostic (29)** | **0.503** | 0.393 |
| V4 pure-ratio (17) | 0.459 | 0.335 |

**Cross-deck** (train one deck → test the other; chance 0.20):

| Feature set | RF Mage→War | RF War→Mage | LDA single-game Mage→War | War→Mage |
|---|---:|---:|---:|---:|
| V3-full (deck-leaky) | 0.224 | 0.235 | 0.218 | 0.290 |
| **V4 agnostic** | **0.242** | **0.254** | **0.239** | **0.291** |
| V4 pure-ratio | 0.208 | 0.239 | 0.205 | 0.268 |

Per-style, cross-deck recall is carried almost entirely by **fatigue** (0.53–0.63) and sometimes
**ramp**; aggro / control / midrange collapse to ~0.02–0.23. *Fatigue is the AI grinding toward
deck-out — a search-policy signature that survives the deck swap; the others are too
deck-shaped.*

---

## 5. What's good / what's bad, per version

| Version | What's good | What's bad |
|---|---|---|
| **V1** | Built the full pipeline; proved aggregation → ARI 0.99; strong per-game acc (0.486) | `time_per_turn` **leaks engine search effort** — inflates accuracy with a non-behavioral feature |
| **V2** | Clean, interpretable, purely game-state features; 1 metric = 1 style dimension; same 0.99 headline | Loses the leak's separating power → per-game acc 0.486 → 0.448; value styles need more games (control 80, ramp >120) |
| **V3** | Trajectory + tells recover most of that loss *honestly* (0.489); ~halves games for control/fatigue; well-targeted features | Helps fatigue/control, **not** midrange/ramp; `extra_mana_crystals` is dead on these decks; feature set now large (29) and Mage-specific |
| **V4** | Normalization keeps within-deck acc (even +Mage to 0.503) while modestly improving cross-deck; isolates *which* styles are deck-independent (fatigue, ramp) | **Cross-deck single-game ID basically fails (~0.22–0.29)** — styles stay **deck-entangled**; Warrior is just harder (0.393); proxy metrics (no minion-ATK/no damage-source in logs) |

---

## 6. Limitations (study-wide, for the discussion section)

- **AI proxy, not humans.** "Play-styles" are SabberStone **search policies**, not human players.
  Conclusions are about separating *decision philosophies*, a proxy for the human question.
- **Deck entanglement is unsolved.** The V4 result is honest but negative: a single game's style
  does not survive a deck change. Cross-deck recognition needs **aggregation** *and* still leans
  on the one or two styles with deck-independent signatures.
- **Log-format limits.** The verbose logs carry no minion ATK stat and no damage-source
  attribution, so two of the four "currency" metrics are **proxies**. `ramp` has no real tell on
  ramp-less highlander lists.
- **Engine coverage.** ~6% of simulated games crash on unimplemented cards (logged + resumable);
  some opponent archetypes (e.g. MidrangeSecretHunter) fail ~half their games, so the
  opponent pool is mildly skewed.

---

## 7. Conclusions & next steps

**Established.** A play-style *is* a distribution: single-game ID is noisy (~0.44–0.49), but
~30–120 pooled games cluster the 5 styles almost perfectly (ARI ≈ 0.99). Trajectory + tell
features (V3) make the value styles identifiable in far fewer games.

**Open.** Single-game style does not transfer across decks (V4). Only fatigue (and partly ramp)
carry a deck-independent signature today.

**Proposed next steps**
1. **Cross-deck *fingerprint* curve** — V4 only reported single-game + LDA-N cross-deck; run the
   full ARI-vs-N aggregation cross-deck to see whether **pooling** rescues transfer even when one
   game cannot.
2. **Add a 3rd deck** (different class/archetype) to test whether "deck-agnostic" generalizes or
   just overfits the Mage↔Warrior pair.
3. **Per-style feature attribution cross-deck** — confirm *why* fatigue transfers (likely
   game-length / deck-out dynamics) and engineer a deck-independent tell for midrange/ramp.
4. **Hold the feature set fixed and vary the opponent pool** to quantify how much per-game noise
   is opponent-driven vs intrinsic.

---

### Appendix — reproduce these numbers
```
py -3 Log_Analysis/V3/measure_v3_gain.py        # §4.1/§4.2 baseline vs +B
py -3 Log_Analysis/V3/measure_v3_full_gain.py   # §4.1/§4.2 +A, full A+B
py -3 Log_Analysis/V4/measure_v4_gain.py        # §4.4 within- and cross-deck
```
(On Windows set `PYTHONUTF8=1` first — the scripts print `§`/`·`/`—` and the default cp932
console codec will otherwise error. Caches `playstyle_log_metrics_*_v3*.pkl` and
`..._v4_bothdecks.pkl` already exist, so the scripts load in seconds without re-parsing the 1.5 GB
of logs.)
