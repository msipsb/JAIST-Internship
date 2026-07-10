# Stage 2 -- GE2E learned embeddings (summary)

Open-set: 44 held-out eval users (never trained on), reference/query halves = Stage-1 splits/seeds.  Fingerprint = mean of learned game vectors, matched-C10 retrieval (chance 0.100).  Baseline 'both' is rebuilt on the SAME 44 users.

**Caveat**: eval pool is 44 users vs Stage-1's 164, so full-pool chance differs; matched-C10 is the apples-to-apples comparison.

## matched-C10 top-1 @N=50 (mean±sd over 3 seeds)

| model | all-users | within-class | within-archetype |
|-------|----------:|-------------:|-----------------:|
| baseline 'both' (44u) | 0.823±0.000 | 0.842±0.000 | 0.500±0.000 |
| GE2E (a) embed | 0.982±0.007 | 0.975±0.022 | 0.967±0.047 |
| GE2E (b) LSA | 0.976±0.014 | 0.971±0.008 | 0.833±0.125 |
| GE2E (b)+adversarial | 0.968±0.011 | 0.943±0.007 | 0.867±0.047 |

## Deck-recognition gap (identical pilots; all vs same-archetype distractors)

| model | distractors=all | distractors=archetype | gap |
|-------|----------------:|----------------------:|----:|
| baseline 'both' | 0.983 | 0.500 | 0.483 |
| GE2E (a) embed | 0.997 | 0.967 | 0.030 |
| GE2E (b) LSA | 0.993 | 0.833 | 0.159 |
| GE2E (b)+adv | 0.991 | 0.867 | 0.124 |

## Answers

1. **Beat baseline within-archetype?** baseline 'both' = 0.500; best learned = 0.967. -> YES (matched-C10 top-1 @N=50).
2. **Adversarial shrinks the deck gap?** Controlled comparison is (b) LSA vs (b)+adv (same card rep, +/- gradient-reversal head): (b) gap = 0.159, (b)+adv gap = 0.124. -> YES, smaller. (For reference the deck-leaky learned-embed (a) gap = 0.030, baseline = 0.483.)
3. **Frozen-LSA (b) trades all-users for within-archetype robustness?** (a) all-users 0.982 / within-arch 0.967; (b) all-users 0.976 / within-arch 0.833.

(Full-spec run on NVIDIA GeForce RTX 5090 (bf16): steps=15000, seeds=[0, 1, 2].)
