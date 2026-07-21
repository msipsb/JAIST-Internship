"""
Within-deck ARI-vs-N: FULL V1-V4 engine features vs the card-sequence-only
survivable subset.

Both curves are computed on the SAME games (out/features.csv), with the SAME
method the earlier reports used (bootstrap fingerprints -> KMeans(k=5) ->
adjusted Rand index vs the true style), so the only thing that differs between
the two curves is the feature set.

  FULL      every raw_* metric = the V1-V4 absolute engine statistics the
            earlier reports used (needs board / health / hand / combat state).

  CARDSEQ   only the raw_* metrics that survive from the card-play sequence
            alone -- cost / composition / timing. This is the subset that is
            still computable when all you have is the ordered list of cards
            played, which is the situation of the real human dataset.

"Within-deck": fingerprints are built per deck and clustered per deck, then the
ARI is averaged over the 9 decks (band = +/- 1 std across decks). This is the
"within a single deck the styles separate" regime from the reports, not the
cross-deck regime.

Usage:  py -3 log_v2_analysis/cardseq_vs_full_ari.py
Output: out/fig_cardseq_vs_full_ari_vs_N.png  +  out/cardseq_vs_full_ari.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

STYLE_ORDER = ["aggro", "control", "fatigue", "midrange", "ramp"]
N_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 25, 35, 50, 70, 90, 100, 120]
REPS = 150

# ---- the two feature sets -------------------------------------------------
# every V1-V4 absolute engine metric (the "old full features")
FULL = [
    "raw_n_turns", "raw_n_decisions", "raw_face_dmg_per_turn",
    "raw_dmg_taken_per_turn", "raw_heal_per_turn", "raw_face_attack_ratio",
    "raw_attacks_per_turn", "raw_cards_per_turn", "raw_avg_card_cost",
    "raw_max_card_cost", "raw_minion_frac", "raw_mana_spent",
    "raw_mana_spent_per_turn", "raw_mana_floated_per_turn",
    "raw_hero_power_per_turn", "raw_board_size_mean",
    "raw_opp_minions_killed_per_turn", "raw_my_minions_lost_per_turn",
    "raw_first_minion_turn", "raw_deck_count_end",
]
# the subset computable from the ordered card plays alone:
#   turn + (card id -> cost, minion/spell).  No board, health, hand or combat.
# Dropped vs FULL because they need the engine trace: face/dmg-taken/heal per
# turn, face-attack ratio, attacks per turn, mana floated, hero power, board
# size, minions killed/lost, n_decisions (counts attacks/hero-power too),
# deck_count_end (needs draw info).
CARDSEQ = [
    "raw_cards_per_turn", "raw_avg_card_cost", "raw_max_card_cost",
    "raw_minion_frac", "raw_mana_spent", "raw_mana_spent_per_turn",
    "raw_first_minion_turn", "raw_n_turns",
]


def prep(df_deck, feats):
    """Per-deck: median-impute, then z-score. -> (X, y) with y the style."""
    X = df_deck[feats].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)  # a column that is all-NaN in this deck
    X = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return X, df_deck["style"].to_numpy()


def bootstrap_fingerprints(X, y, N, reps, rng):
    """reps fingerprints per style: mean of N games drawn with replacement."""
    rows, lab = [], []
    for st in STYLE_ORDER:
        a = X[y == st]
        if len(a) == 0:
            continue
        for _ in range(reps):
            rows.append(a[rng.integers(0, len(a), N)].mean(axis=0))
            lab.append(st)
    return np.asarray(rows), np.asarray(lab)


def sweep_one_deck(X, y):
    """ARI vs N for one deck (unsupervised KMeans(k=5) on the fingerprints)."""
    out = []
    for N in N_GRID:
        rng = np.random.default_rng(0)  # same draws across feature sets & decks
        Xb, yb = bootstrap_fingerprints(X, y, N, REPS, rng)
        lab = KMeans(5, n_init=10, random_state=0).fit_predict(Xb)
        out.append(adjusted_rand_score(yb, lab))
    return out


def curve(df, feats, name):
    """Per-deck ARI-vs-N, averaged over decks. -> (mean, std) arrays."""
    per_deck = []
    for deck, g in df.groupby("deck"):
        X, y = prep(g, feats)
        per_deck.append(sweep_one_deck(X, y))
        print(f"  [{name}] {deck:26s} "
              f"ARI(N=10)={per_deck[-1][N_GRID.index(10)]:.3f} "
              f"ARI(N=120)={per_deck[-1][-1]:.3f}", flush=True)
    per_deck = np.asarray(per_deck)
    return per_deck.mean(0), per_deck.std(0)


def main():
    df = pd.read_csv(os.path.join(OUT, "features.csv"))
    print(f"{len(df)} games, {df['deck'].nunique()} decks, "
          f"styles={sorted(df['style'].unique())}")
    print(f"FULL: {len(FULL)} features   CARDSEQ: {len(CARDSEQ)} features\n")

    full_m, full_s = curve(df, FULL, "FULL")
    cs_m, cs_s = curve(df, CARDSEQ, "CARDSEQ")

    # ---- save the numbers -------------------------------------------------
    tbl = pd.DataFrame({
        "N": N_GRID,
        "full_ari_mean": full_m, "full_ari_std": full_s,
        "cardseq_ari_mean": cs_m, "cardseq_ari_std": cs_s,
    })
    tbl.to_csv(os.path.join(OUT, "cardseq_vs_full_ari.csv"), index=False)
    print("\n" + tbl.to_string(index=False,
          formatters={c: "{:.3f}".format for c in tbl.columns if c != "N"}))

    # ---- plot -------------------------------------------------------------
    xs = np.arange(len(N_GRID))
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for m, s, color, label in [
        (full_m, full_s, "#2166ac",
         "full V1-V4 features (needs board/health/combat)"),
        (cs_m, cs_s, "#b2182b",
         "card-sequence-only features (order/cost/type)"),
    ]:
        ax.plot(xs, m, "o-", color=color, lw=2.2, ms=6, label=label)
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15)

    ax.axhline(0.0, color="gray", lw=1, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(N_GRID)
    ax.set_xlabel("games averaged per fingerprint  (N)")
    ax.set_ylabel("cluster-vs-style ARI   (1.0 = perfect, 0 = chance)")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Within-deck: the 5 play-styles separate as games are pooled\n"
                 "full engine features vs card-sequence-only features "
                 "(mean over 9 decks, band = ±1 std)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_cardseq_vs_full_ari_vs_N.png")
    fig.savefig(p, dpi=150)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
