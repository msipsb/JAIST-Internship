# log_v2 cross-deck transfer analysis

Answers RQ3 on the 9-deck `log_v2` matrix: **is the model reading the player's
style, or just the deck?** Test = train on some decks, predict style on decks
never seen in training.

**Start here: [`cross_deck_SHOW.ipynb`](cross_deck_SHOW.ipynb)** -- the narrative version
of everything below, with the figures and a worked example of a real decision.

Run order (features.csv is cached; step 1 is a one-time ~8 min disk-bound parse):

```
py -3 log_v2_analysis/v2_features.py      # 40,492 games -> out/features.csv
py -3 log_v2_analysis/v2_cross_deck.py    # -> out/cross_deck_results.csv + figures

# the readable notebook (reads the cached features.csv; ~4 min to execute)
py -3 log_v2_analysis/_make_cross_deck_notebook.py
py -3 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 log_v2_analysis/cross_deck_SHOW.ipynb
```

The notebook imports `evaluate()` from `v2_cross_deck.py` rather than restating the
logic, so its numbers cannot drift from the script's.

## Headline

Leave-one-deck-**archetype**-out, 5 styles, chance = 0.20:

| feature set | acc N=1 | acc N=10 | ARI_direct N=10 | ARI_lda N=10 |
|---|---|---|---|---|
| raw (no norm) | 0.406 | 0.517 | 0.231 | 0.291 |
| **raw+deckz (OLD baseline)** | 0.455 | **0.760** | 0.331 | 0.524 |
| choice only (proposed) | 0.453 | 0.527 | 0.247 | 0.276 |
| choice+deckz | 0.465 | 0.617 | 0.293 | 0.341 |
| **raw+choice+deckz (BEST)** | **0.532** | **0.812** | **0.382** | **0.609** |

**The deliverable: `raw+choice+deckz` beats the old deck-normalization**, 0.760 ->
0.812 at N=10 and 0.455 -> 0.532 at N=1. It is still a plain LDA on interpretable
statistics -- the only change is adding the choice-relative block to the old
feature set. It wins on **every** fold, so this is not one lucky deck family:

| held-out family | OLD | BEST |
|---|---|---|
| aggro | 0.773 | 0.847 |
| combo_tempo | 0.800 | 0.836 |
| highlander_control | 0.692 | 0.735 |
| midrange | 0.775 | 0.828 |

## The hypothesis that was WRONG

The idea was that choice-relative features would **replace** deck-normalization:
score each chosen option against the legal options available at that decision, so
the deck cancels out by construction. That is false. **Choice features alone
(0.527) lose badly to old deck-normalization (0.760) at N=10.**

They are **complementary, not a replacement**. Added on top of the old baseline
they buy +5.2 points; on their own they do not carry enough signal. Where they do
win alone is the hard case -- a **single game, unsupervised**: ARI_direct 0.115 vs
0.064, nearly double. Pooling 10 games is exactly the regime where averaging
rescues the noisy raw metrics, which is why the advantage evaporates there.

Note the old baseline is deliberately handed an advantage: its per-deck z-score is
fit on the **test** deck's own games (that is what "distance from that deck's
average player" means, and it is what the earlier reports did). It therefore peeks
at test-deck statistics. Choice features never do.

## Two findings that change next week's plan

**1. No metric reverses across decks.** The report's section 5 plans to "check
whether some metrics point in opposite directions on different decks, keep only
the metrics that order the styles consistently". On this 9-deck rich data, **all
32 features have positive cross-deck profile correlation (see
`feature_stability.csv`) -- nothing reverses, so there is nothing to drop.** The
reversal result (mana_eff -0.86, cost_tilt -0.77) came from the *card-sequence-only*
data, and does not reproduce once engine-trace features are available. That
planned diagnostic is a dead end here.

**2. The sibling-deck leak is negligible.** Holding out one deck (LODO 0.820) vs a
whole archetype family (LOAO 0.812) barely differs. Transfer is not being propped
up by a sibling deck of the same family staying in training, which makes the RQ3
claim stronger than expected.

## Why choice-relative features help at all

`ch_face_pref` -- given that BOTH a face attack and a minion trade were legal at
this decision, did the player go face? -- is the **single most cross-deck-stable
feature of all 32** (profile correlation 0.995). It holds aggro at 0.74-0.81 in
every deck family while ramp/control sit at 0.17-0.35. Contrast
`raw_face_dmg_per_turn` (0.412), where the deck varies the value more than the
style does. See `fig_face_pref_vs_raw.png`.

The principle: condition on the dilemma actually arising, then measure the choice
within it. Absolute rates measure what the deck handed you.

## Files

| file | what |
|---|---|
| **`cross_deck_SHOW.ipynb`** | **the readable write-up -- start here** |
| `_make_cross_deck_notebook.py` | emits the notebook (edit this, not the .ipynb) |
| `v2_features.py` | parse -> `raw_*` (V1-V4 absolute) + `ch_*` (choice-relative) blocks |
| `v2_cross_deck.py` | LOAO/LODO evaluation, all feature sets, figures |
| `out/features.csv` | 40,492 games x 35 features + labels |
| `out/cross_deck_results.csv` | the headline table |
| `out/cross_deck_per_fold.csv` | per-fold detail |
| `out/feature_stability.csv` | cross-deck profile correlation per feature |
| `out/within_deck_reference.csv` | within-deck ceiling (choice 0.541 vs raw 0.517) |
| `out/fig_cross_deck_headline.png` | the headline comparison |
| `out/fig_loao_vs_lodo.png` | sibling-deck leak |
| `out/fig_feature_stability.png` | which features order styles consistently |
| `out/fig_face_pref_vs_raw.png` | flagship feature vs its raw counterpart |

## Methodology notes

- **`search_score` is never read.** Every decision logs the agent's own internal
  evaluation score. Each playstyle *is* a different scoring function, so using it
  would score ~100% and mean nothing.
- **`ch_hero_attack_face_pref` is excluded from the model** (kept in the CSV): it
  is undefined for the 3 weaponless decks (both Reno decks, Zoo), so its presence
  encodes deck identity rather than style.
- **`ch_n_options`, `ch_face_dilemma_rate` excluded**: they describe the deck's
  option supply, not the preference within it.
- Focal player is p1 (the folder's own agent); the opponent is never used as a
  labeled sample, to avoid correlated duplicate rows.
- Deck archetype families are a **judgement call**, edit `DECK_FAMILY` in
  `v2_features.py` to regroup. MurlocDruid (mean cost 2.80) is grouped as aggro
  but curves closest to midrange.
