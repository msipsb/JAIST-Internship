# log_power_analysis — play-style study on the `Power.log` dataset

Analyses [`log_power/`](../log_power/) — 40,500 games written in the **official
Hearthstone `Power.log` format** — with simple per-game statistics.

```sh
py -3 log_power_analysis/powerlog_features.py        # 16 GB of logs -> out/features.csv (~10 min)
jupyter lab log_power_analysis/playstyle_powerlog.ipynb
```

| file | role |
| --- | --- |
| `powerlog_features.py` | parses each `.log` into 35 per-game statistics for P1; labels joined from each cell's `summary.csv` |
| `playstyle_powerlog.ipynb` | the analysis |
| `_make_powerlog_analysis_notebook.py` | rebuilds the notebook |
| `out/features.csv` | 40,497 games × 35 features + label columns |

## Model policy

| | |
| --- | --- |
| unsupervised | **KMeans, k = 5** |
| supervised | **LinearDiscriminantAnalysis** |
| metric | **adjusted Rand index (ARI)** — **0.20 chance line**, perfect 1.000 |

The 0.20 line is the reference bar shared with the `log_v2` notebooks (random
guessing among 5 equiprobable styles). ARI's own null expectation is 0, so a
score between 0 and 0.20 is still above ARI-chance — 0.20 is the comparison bar,
not the floor.

Protocols: pooled (all decks), **within-deck** (fit and score inside one deck),
**LODO** (leave-one-deck-out, 9 folds) and **LOAO** (leave-one-archetype-out,
4 folds, a whole deck family held out).

## How different are the 9 decks?

They are **9 different hero classes, one each** (Warrior, Paladin, Shaman,
Hunter, Rogue, Druid, Priest, Mage, Warlock) — not variants of one archetype.
Class cards cannot cross classes, so the only possible overlap is neutrals and it
is tiny: card-name Jaccard averages **0.020** over the 36 pairs, peaking at
**0.154** for the two Reno highlander decks (8 shared neutral staples) and 0.121
for JadeShaman/MiraclePirateRogue; every other pair is ≤ 0.062. Each deck runs
13–22 cards no other deck runs. **LODO is a real domain shift.**

`deck_family` is therefore a *strategic* grouping, not a compositional one —
within-family card overlap (0.026) is barely above across-family (0.020). The
sibling deck LODO leaves in training was never carrying much of the held-out
deck's card pool, which is why LOAO lands so close to LODO. The grouping does
hold up in behaviour space (mean deck-profile distance 6.06 within family vs 8.46
across; the 3 closest pairs are all same-family) but it is soft. Read LOAO as a
second, slightly stricter LODO rather than a categorically harder test.

## Headline results (40,497 games, 35 statistics)

| | LDA | KMeans (fit on train) | KMeans (fit on held-out) |
| --- | --- | --- | --- |
| pooled, per game | 0.118 | 0.038 | — |
| within-deck (mean of 9) | **0.221** | — | 0.082 |
| cross-deck LODO (mean of 9) | 0.127 | 0.050 | 0.088 |
| cross-family LOAO (mean of 4) | 0.118 | 0.039 | 0.050 |

Within-deck LDA is the **only single-game protocol that clears the 0.20 line**,
and only just. With decks pooled, aggregating games into fingerprints reaches ARI
0.90 at N = 50 (LDA) / N = 90 (KMeans). Across decks the same aggregation
plateaus at 0.469 (LODO) and 0.597 (LOAO) at N = 100 — **pooling fixes noise, not
domain shift.**
Per-deck z-scoring lifts LDA LODO 0.127 → 0.156 and LOAO 0.118 → 0.147, so part
of the gap is a per-deck offset but most is not: the statistics rank the styles
in a *different order* on different decks. KMeans transfer is the sharpest tell —
centroids carried from the training decks (0.050) score **worse** than
re-clustering the held-out deck from scratch (0.088), i.e. they encode the deck.

## Why this dataset is different from `log_v2`

* **No decision records.** A real client log records the action that was *taken*,
  never the legal options it was chosen *from*, so the choice-relative (`ch_*`)
  block of [`log_v2_analysis/v2_features.py`](../log_v2_analysis/v2_features.py)
  cannot be rebuilt. This is necessarily a raw-statistics study.
* **P1's viewpoint only.** P1's own deck is anonymous entities until drawn and
  P2's hand stays hidden — the format discards information the simulator had.
* **Labels never touch the features.** The `.log` never names the agent;
  `style` / `deck` come from `summary.csv`, joined on the file name.

Features are read out of the packet stream: `BLOCK_START BlockType=PLAY`/`ATTACK`
on a P1 entity for what P1 did, the `FULL_ENTITY`/`SHOW_ENTITY` registry for a
card's `COST`/`CARDTYPE`/`ATK`/`HEALTH`, and `TAG_CHANGE` on
`DAMAGE`/`ARMOR`/`ZONE`/`RESOURCES` for life totals, deaths, draws and mana.

Two parser details worth knowing:

* Every turn opens with `RESOURCES=N` / `RESOURCES_USED=N` / `RESOURCES_USED=0`;
  the middle value is the engine restoring the pool before clearing it, so mana
  spent is the running max *after* the last reset to 0.
* The mulligan also moves cards `DECK -> HAND -> DECK -> HAND`, so draws are
  counted only once the first real turn has started.

Hero-power counts were checked against the engine's own
`NUM_TIMES_HERO_POWER_USED_THIS_GAME` tag, and attack/face-attack counts against
an independent scan of the `BlockType=ATTACK` lines.
