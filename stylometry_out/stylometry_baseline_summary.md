# Behavioral stylometry -- Stage 1 baseline (summary)

Fingerprint = z-scored mean of per-game features; match a player's query-half fingerprint to reference-half fingerprints by cosine similarity.

Pools are compared with **matched-C10** retrieval (rank the true reference against a fixed 10-way candidate set drawn from the same pool) so chance is constant (1/10) and the pools are directly comparable.  The full-pool (rank-against-everyone) numbers are in the CSV.

## E1 -- SIM sanity check (10 style x deck pseudo-users, full-pool)

- top-1 (cardseq): N=50 -> 0.700,  N=400 -> **1.000** (chance 0.100).  Every confusion at small N is a same-deck style sibling, never cross-deck; at large N all 10 separate perfectly -> harness is sound. (Same-deck scripted styles are subtle and need many games to resolve.)

## E2 -- HUMAN re-identification (label-free), matched-C10 top-1 @N=50

| pool | cardseq | rhythm | role | both | chance |
|------|--------:|-------:|-----:|-----:|-------:|
| all-users | 0.660 | 0.571 | 0.773 | 0.795 | 0.100 |
| within-class | 0.757 | 0.637 | 0.826 | 0.864 | 0.100 |
| within-archetype | 0.579 | 0.505 | 0.637 | 0.709 | 0.100 |

## Verdict

- **Label-free re-identification works**: at N=50 a player's cardseq fingerprint retrieves the same player out of all 164 users with full-pool top-1 = **0.293** (chance 0.006, ~48x). A stable personal card-sequence signal exists -- this substitutes for the missing style labels.
- **Deck-recognition isolation** (identical (user,archetype) pilots, only the distractor scope shrinks, matched-C10, cardseq @N=50): all 0.962 >= class 0.827 >= archetype 0.579 (chance 0.100).
  - The drop from all-distractors to same-archetype-distractors is 0.382: that portion of re-identification was DECK recognition.
  - But same-archetype top-1 = 0.579 still far exceeds chance (0.100): telling apart two players **on the same deck** works, so a genuine personal play style survives after the deck is removed.
- **Ablation**: `rhythm` (identity-free timing/cost/count only) still clears chance within-archetype (matched top-1 0.505 vs 0.100) -> part of the signal is pure tempo, not card choice. `role` and `both` add card-function information on top.
- Note: the three E2 *pools* above are not a controlled deck comparison (they regroup a user's games differently -> mixing all a user's decks into one all-users fingerprint blurs it, which is why within-class can edge above all-users). The deck-recognition isolation is the controlled version.

(Stage 2 GE2E learned embeddings intentionally deferred -- run order says STOP for human review of these Stage-1 numbers first.)
