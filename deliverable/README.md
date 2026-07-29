# Play-style inference in Hearthstone — code, data and results

Can a model read a **player's style** from a game log, or is it just reading the
**deck** they brought? This folder holds the final answer to that question, the
code that produced it, and the derived data needed to reproduce every number and
figure without re-running the simulator.

Two datasets, one question:

| dataset | what it is | what it can answer |
| --- | --- | --- |
| **`log_v2`** | 40,492 simulated games logged as **per-decision JSONL** — every decision records the *legal options* alongside the *chosen* action | the full question, including choice-relative features |
| **`log_power`** | 40,500 re-simulated games written in the **official Hearthstone `Power.log` format** | what is still recoverable from a *real client log* |

The `log_power` study exists because the `log_v2` result depends on information a
real Hearthstone log does not contain. It is the reality check.

---

## 1. Setup

5 play-styles × 9 decks, fully crossed.

* **Styles** (SabberStone's heuristic AI agents, each a different scoring
  function): `aggro`, `control`, `midrange`, `fatigue`, `ramp`.
* **Decks** (one per hero class): AggroPirateWarrior, MidrangeBuffPaladin,
  MidrangeJadeShaman, MidrangeSecretHunter, MiraclePirateRogue, MurlocDruid,
  RenoKazakusDragonPriest, RenoKazakusMage, ZooDiscardWarlock.
* **Matrix**: 5 P1 agents × 5 P2 agents × 9 P2 decks × 20 games = 4,500 games per
  P1 deck cell; 9 cells = 40,500 games.
* **Model policy** everywhere: unsupervised = KMeans (k=5), supervised = LDA,
  metric = adjusted Rand index (ARI). **Chance line drawn at 0.20** (random guess
  among 5 equiprobable styles). ARI's own null is 0, so 0.20 is the comparison
  bar, not the floor.

**The 9 decks are 9 different hero classes, so cross-deck really is a domain
shift.** Class cards cannot cross classes; card-name Jaccard averages **0.020**
over the 36 deck pairs, peaking at 0.154 for the two Reno highlander decks. Each
deck runs 13–22 cards no other deck runs. `deck_family` is therefore a
*strategic* grouping, not a compositional one — which is why leave-one-deck-out
(LODO) and leave-one-archetype-out (LOAO) land so close together.

---

## 2. Headline results

### 2.1 `log_v2` — choice-relative features (the main contribution)

Every action is scored against the options that were **legally available at that
decision**, so the deck becomes the denominator instead of part of the signal.
A style is "what you picked out of what you were offered".

Leave-one-archetype-out, 5 styles, chance = 0.20:

| feature set | acc N=1 | acc N=10 | ARI_direct N=10 | ARI_lda N=10 |
| --- | --- | --- | --- | --- |
| raw (no norm) | 0.406 | 0.517 | 0.231 | 0.291 |
| **raw+deckz (OLD baseline)** | 0.455 | **0.760** | 0.331 | 0.524 |
| choice only (proposed) | 0.453 | 0.527 | 0.247 | 0.276 |
| choice+deckz | 0.465 | 0.617 | 0.293 | 0.341 |
| **raw+choice+deckz (BEST)** | **0.532** | **0.812** | **0.382** | **0.609** |

**Result: `raw+choice+deckz` beats the old deck-normalisation baseline**, 0.760 →
0.812 at N=10 and 0.455 → 0.532 at N=1, while staying a plain LDA on
interpretable statistics. It wins on **every fold** (aggro 0.773→0.847,
combo_tempo 0.800→0.836, highlander_control 0.692→0.735, midrange 0.775→0.828),
so it is not one lucky deck family.

**The original hypothesis was wrong.** The idea was that choice-relative features
would *replace* deck-normalisation. They do not: choice alone (0.527) loses
badly to old deck-z (0.760) at N=10. They are **complementary** — worth +5.2
points on top of the baseline, but not self-sufficient. Where they do win alone
is the hard case, a **single game, unsupervised**: ARI_direct 0.115 vs 0.064,
nearly double. Pooling 10 games is exactly the regime where averaging rescues the
noisy raw metrics, which is why the advantage evaporates there.

Note the old baseline is deliberately handed an advantage: its per-deck z-score
is fit on the **test** deck's own games, so it peeks at test-deck statistics.
The choice features never do. Beating a transductive baseline with an inductive
model is the stronger claim.

**Why it helps.** `ch_face_pref` — *given that both a face attack and a minion
trade were legal here, did the player go face?* — is the **most cross-deck-stable
of all 32 features** (profile correlation 0.995). It holds aggro at 0.74–0.81 in
every deck family while ramp/control sit at 0.17–0.35. Its absolute counterpart
`raw_face_dmg_per_turn` scores 0.412, because the deck varies that value more
than the style does. The principle: *condition on the dilemma actually arising,
then measure the choice within it.*

Two secondary findings:

1. **No metric reverses across decks.** All 32 features have positive cross-deck
   profile correlation, so there is nothing to prune. The earlier reversal result
   (`mana_eff` −0.86, `cost_tilt` −0.77) came from *card-sequence-only* data and
   does not reproduce once engine-trace features exist.
2. **The sibling-deck leak is negligible.** LODO 0.820 vs LOAO 0.812 — transfer
   is not being propped up by a same-family deck staying in training.

### 2.2 `log_power` — what survives on a real client log

Same 5×9 matrix, but the file is a genuine `Power.log`. **No decision records
exist in a real client log**, so the `ch_*` block cannot be rebuilt; this is
necessarily a raw-statistics study (35 per-game statistics, P1's viewpoint only).

| protocol | LDA | KMeans (fit on train) | KMeans (fit on held-out) |
| --- | --- | --- | --- |
| pooled, per game | 0.118 | 0.038 | — |
| within-deck (mean of 9) | **0.221** | — | 0.082 |
| cross-deck LODO (mean of 9) | 0.127 | 0.050 | 0.088 |
| cross-family LOAO (mean of 4) | 0.118 | 0.039 | 0.050 |

Within-deck LDA is the **only single-game protocol that clears 0.20**, and only
just. Pooling games into fingerprints reaches ARI 0.90 at N=50 (LDA) with decks
pooled, but across decks the same aggregation plateaus at 0.469 (LODO) / 0.597
(LOAO) at N=100 — **pooling fixes noise, not domain shift.** Per-deck z-scoring
lifts LODO 0.127 → 0.156, so part of the gap is a per-deck offset but most is
not: the statistics rank the styles in a *different order* on different decks.
The sharpest tell is KMeans transfer — centroids carried from the training decks
(0.050) score **worse** than re-clustering the held-out deck from scratch
(0.088), i.e. they encode the deck rather than the style.

### 2.3 How much the rich log buys you

Within-deck ARI vs number of games pooled, same games and same method, only the
feature set differs (`out/cardseq_vs_full_ari.csv`):

| games pooled (N) | raw+choice (full log) | card-sequence only |
| --- | --- | --- |
| 1 | 0.152 | 0.047 |
| 10 | 0.568 | 0.218 |
| 25 | 0.826 | 0.358 |
| 50 | 0.962 | 0.499 |
| 100 | 0.997 | 0.692 |

The full log needs ~10 games to pass where the card sequence needs ~50.

### 2.4 Bottom line

Style is recoverable, but **only with the deck accounted for**. Choice-relative
features are the best deck-independent signal found so far and they improve
cross-deck transfer on every fold — but they supplement deck-normalisation, they
do not replace it. On a real client log, where choice-relative features are
impossible, single-game cross-deck inference stays near chance.

---

## 3. What is in this folder

Layout mirrors the source repository, so every script's relative paths work
unchanged if you treat this folder as the project root.

### Simulation pipeline (generates the datasets)

| file | role |
| --- | --- |
| `sabberstone_simulator.py` | loads the SabberStone C# engine in-process via pythonnet and drives the match loop (mulligan → look-ahead search → apply best line) |
| `sim_rich.py` | plays one game and writes **per-decision JSONL**: legal options, chosen task, full pre-action state, attributed post-action events |
| `decks_v2.py` | the 9 deck lists, rebuilt at runtime — `Decks.cs` has crash-prone cards commented out, leaving 6 of 9 decks under 30 cards (Zoo at 11); 47 of the 49 commented copies are in fact implemented, so this restores them |
| `log_v2/run_batch_matrix_v2.py` | the 40,500-game matrix driver — resumable, atomic writes |
| `log_v2/run_batch_v2.ipynb` | notebook runner, one cell per P1 deck |
| `log_power/sim_powerlog.py` | same game, `GameConfig.History=True`, rendered to `Power.log` |
| `log_power/powerlog_render.py` | engine PowerHistory → client text format, plus viewpoint hiding |
| `log_power/run_batch_powerlog.py` / `.ipynb` | the `Power.log` matrix driver |
| `log_power/roundtrip_check.py` | simulate → render → parse with **hslog** (HearthSim's parser) → diff winner, turns, health, armor and both boards against the engine |
| `log_power/README.md` | format details, verification, known limits |

### Analysis — `log_v2`

| file | role |
| --- | --- |
| `log_v2_analysis/README.md` | the detailed write-up |
| `log_v2_analysis/cross_deck_SHOW.ipynb` | **the readable narrative version — start here** |
| `log_v2_analysis/v2_features.py` | parse → `raw_*` (absolute) + `ch_*` (choice-relative) blocks |
| `log_v2_analysis/v2_cross_deck.py` | LOAO/LODO evaluation, all feature sets, figures |
| `log_v2_analysis/cardseq_vs_full_ari.py` | full-log vs card-sequence-only ARI-vs-N |
| `log_v2_analysis/three_blocks/playstyle_v2_raw_v1to4.ipynb` | full 10-section study, raw metrics only |
| `log_v2_analysis/three_blocks/playstyle_v2_choice.ipynb` | same study, choice-relative only |
| `log_v2_analysis/three_blocks/playstyle_v2_both.ipynb` | same study, both blocks |
| `log_v2_analysis/_make_*.py` | notebook generators — **edit these, not the `.ipynb`** |

The three `three_blocks` notebooks are identical section-for-section; only the
feature block selected in section 1 differs, so any difference between them is
attributable to the features alone.

### Analysis — `log_power`

| file | role |
| --- | --- |
| `log_power_analysis/README.md` | the detailed write-up |
| `log_power_analysis/playstyle_powerlog.ipynb` | the analysis |
| `log_power_analysis/powerlog_features.py` | `.log` packet stream → 35 per-game statistics |

### Data included

| file | contents |
| --- | --- |
| `log_v2_analysis/out/features.csv` | **40,492 games × 35 features + labels** (20 MB) |
| `log_power_analysis/out/features.csv` | **40,497 games × 35 features + labels** (20 MB) |
| `log_v2_analysis/out/cross_deck_results.csv` | the headline table |
| `log_v2_analysis/out/cross_deck_per_fold.csv` | per-fold detail |
| `log_v2_analysis/out/feature_stability.csv` | cross-deck profile correlation per feature |
| `log_v2_analysis/out/within_deck_reference.csv` | within-deck ceiling (choice 0.541 vs raw 0.517) |
| `log_v2_analysis/out/cardseq_vs_full_ari.csv` | §2.3 curve |
| `log_v2_analysis/out/fig_*.png` | 5 result figures |

**Raw game logs are not included** — `log_v2/` is 7.2 GB of JSONL and
`log_power/` 16 GB. The two `features.csv` files are the derived tables every
analysis actually reads, so **all notebooks and result scripts run from this
folder as-is**. Only the two parse steps need the raw logs.

---

## 4. Reproducing

### From the included data (no raw logs needed)

```sh
py -3 log_v2_analysis/v2_cross_deck.py          # headline table + figures
py -3 log_v2_analysis/cardseq_vs_full_ari.py    # ARI-vs-N curve
jupyter lab log_v2_analysis/cross_deck_SHOW.ipynb
jupyter lab log_power_analysis/playstyle_powerlog.ipynb
```

The notebooks import `evaluate()` from `v2_cross_deck.py` rather than restating
the logic, so their numbers cannot drift from the script's.

### From raw logs (regenerating `features.csv`)

```sh
py -3 log_v2_analysis/v2_features.py            # 40,492 games -> out/features.csv (~8 min, disk-bound)
py -3 log_power_analysis/powerlog_features.py   # 16 GB of logs -> out/features.csv (~10 min)
```

### From scratch (regenerating the datasets)

Requires the SabberStone C# engine built into DLLs — see the build notes in the
header of `sabberstone_simulator.py`. The engine source is **not** bundled here;
clone it separately.

```sh
py -3 log_v2/run_batch_matrix_v2.py --deck1 all        # ~10-15 h, ~7 GB
py -3 log_power/run_batch_powerlog.py --deck1 all      # ~12-16 h, ~16 GB
```

Both are resumable (existing files are skipped, writes are atomic).

### Environment

Two interpreters are in play and installing into the wrong one fails silently:

* **`py -3`** — the global interpreter, used for all analysis scripts and the
  analysis notebook kernels. Needs `pandas`, `numpy`, `scikit-learn`,
  `matplotlib`, `nbformat`.
* **`.venv`** — has **pythonnet**, required by anything that loads the C# engine
  (`sabberstone_simulator.py`, `sim_rich.py`, `decks_v2.py`, both batch
  runners). Must use the default .NET Framework loader — a 32-bit-only `dotnet`
  makes coreclr fail.

`hslog` is needed only by `roundtrip_check.py`, and only in the interpreter that
runs it.

---

## 5. Caveats

* **`search_score` is never read.** Every decision logs the agent's own internal
  evaluation score. Each play-style *is* a different scoring function, so using
  it would score ~100% and mean nothing.
* **Three `ch_*` columns are excluded from the models** (kept in the CSV):
  `ch_hero_attack_face_pref` is undefined for the 3 weaponless decks, so its
  presence encodes deck identity; `ch_n_options` and `ch_face_dilemma_rate`
  describe the deck's option supply rather than the preference within it.
* **Labels never touch the features.** `style`/`deck` come from each cell's
  `summary.csv`, joined on file name; the `.log` never names the agent.
* **Focal player is P1 only.** The opponent is never used as a labelled sample,
  to avoid correlated duplicate rows.
* **`log_power` games are new games.** They were re-simulated, not converted —
  they do **not** correspond to `log_v2` games; never join the two by index.
* **~0.8% of games raise** on a card SabberStone has not implemented. They are
  recorded as `ERROR` rows in `summary.csv` and skipped, which is why the counts
  are 40,492 / 40,497 rather than 40,500.
* **`deck_family` is a judgement call.** Edit `DECK_FAMILY` in `v2_features.py`
  to regroup. MurlocDruid (mean cost 2.80) is grouped as aggro but curves closest
  to midrange.
* **Timestamps in `log_power` are synthetic**, derived from each game's RNG seed.
  Real think time correlates with the agent, so genuine timing would leak the
  label.
