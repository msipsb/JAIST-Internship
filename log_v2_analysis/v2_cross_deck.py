"""
Cross-deck (cross-domain) transfer on the 9-deck log_v2 matrix.

Question (RQ3): is the model reading the PLAYER's style, or just the DECK?
The test: train on some decks, predict style on decks never seen in training.

Feature sets compared on identical folds:
  raw          V1-V4 absolute metrics, no normalization          (floor)
  raw+deckz    same metrics, per-deck z-score                    (OLD BASELINE)
  choice       choice-relative metrics only                      (PROPOSED)
  choice+deckz choice-relative, also per-deck z-scored           (does norm still help?)

The OLD BASELINE is deliberately given an advantage: its per-deck z-score is fit
on the TEST deck's own games (that is what "distance from that deck's average
player" means, and it is what the earlier reports did). It therefore sees test
deck statistics at fit time. The choice-relative set needs no such adaptation --
it is deck-normalized by construction. Beating a transductive baseline with an
inductive model is the stronger claim.

Splits:
  LOAO  leave-one-deck-ARCHETYPE-out (headline) -- holds out a whole family, so
        no sibling deck of the same family leaks into training.
  LODO  leave-one-deck-out (diagnostic) -- the gap LODO - LOAO is the sibling leak.

Model is deliberately simple: LDA on standardized features. No neural nets.

Usage:  py -3 log_v2_analysis/v2_cross_deck.py
"""
from __future__ import annotations

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
# NB: the Agg backend is selected in __main__, NOT here. Importing this module
# must not switch the caller's backend -- cross_deck_SHOW.ipynb imports it, and
# an import-time matplotlib.use("Agg") silently kills every inline figure after.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RNG = 0
POOL_NS = (1, 10)
STYLES = ["aggro", "control", "fatigue", "midrange", "ramp"]


def load():
    df = pd.read_csv(os.path.join(OUT, "features.csv"))
    raw = [c for c in df.columns if c.startswith("raw_")]
    ch = [c for c in df.columns if c.startswith("ch_")]
    # Excluded from the model, kept in features.csv as diagnostics:
    #   ch_n_options, ch_face_dilemma_rate  -- describe the deck's option supply,
    #       not the player's preference within it.
    #   ch_hero_attack_face_pref -- undefined for the 3 weaponless decks (both
    #       Reno decks, Zoo), so its mere presence encodes deck identity.
    for c in ("ch_n_options", "ch_face_dilemma_rate", "ch_hero_attack_face_pref"):
        if c in ch:
            ch.remove(c)
    return df, raw, ch


def _pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def _deck_z(df, cols):
    """Per-deck z-score: each game as its distance from that deck's average player."""
    out = df[cols].astype(float)  # integer columns cannot hold z-scores
    for _, idx in df.groupby("deck").groups.items():
        blk = out.loc[idx]
        out.loc[idx] = (blk - blk.mean()) / blk.std(ddof=0).replace(0, np.nan)
    return out


def _pool(X, y, groups, n, rng):
    """Average n games from the same (style, deck) cell into one pooled sample."""
    if n == 1:
        return X, y
    Xs, ys = [], []
    for key in np.unique(groups):
        m = groups == key
        Xi, yi = X[m], y[m]
        order = rng.permutation(len(Xi))
        for i in range(0, len(order) - n + 1, n):
            Xs.append(Xi[order[i:i + n]].mean(axis=0))
            ys.append(yi[order[i]])
    return np.array(Xs), np.array(ys)


def evaluate(df, cols, deckz, splits, split_col, n):
    """One (feature set, pooling N) -> mean scores over folds.

    Three numbers per fold, because the earlier reports quoted ARI from a
    purely unsupervised protocol and we must not quietly compare against an
    easier one:
      acc         LDA trained on train decks, predicting the held-out deck.
      ari_direct  KMeans straight on the held-out deck's features, no training
                  at all. THIS is the old protocol that produced 0.08 -> 0.23.
      ari_lda     KMeans in the LDA space learned from the training decks, i.e.
                  does the learned projection carry over to an unseen deck.
    """
    rng = np.random.RandomState(RNG)
    F = _deck_z(df, cols) if deckz else df[cols]
    F = F.replace([np.inf, -np.inf], np.nan)
    accs, ari_ds, ari_ls, per_fold = [], [], [], []

    for held in splits:
        te = df[split_col] == held
        tr = ~te
        Xtr, ytr = F[tr].to_numpy(float), df.loc[tr, "style"].to_numpy()
        Xte, yte = F[te].to_numpy(float), df.loc[te, "style"].to_numpy()
        gtr = (df.loc[tr, "style"] + "|" + df.loc[tr, "deck"]).to_numpy()
        gte = (df.loc[te, "style"] + "|" + df.loc[te, "deck"]).to_numpy()

        Xtr, ytr = _pool(Xtr, ytr, gtr, n, rng)
        Xte, yte = _pool(Xte, yte, gte, n, rng)

        model = _pipe().fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))

        km_lda = KMeans(len(STYLES), n_init=10, random_state=RNG).fit_predict(
            model.transform(Xte))
        ari_lda = adjusted_rand_score(yte, km_lda)

        # old protocol: no training signal whatsoever
        prep = Pipeline([("imp", SimpleImputer(strategy="mean")),
                         ("sc", StandardScaler())])
        km_dir = KMeans(len(STYLES), n_init=10, random_state=RNG).fit_predict(
            prep.fit_transform(Xte))
        ari_direct = adjusted_rand_score(yte, km_dir)

        accs.append(acc)
        ari_ds.append(ari_direct)
        ari_ls.append(ari_lda)
        per_fold.append({"held_out": held, "acc": acc, "ari_direct": ari_direct,
                         "ari_lda": ari_lda, "n_test": len(yte)})
    return (float(np.mean(accs)), float(np.mean(ari_ds)), float(np.mean(ari_ls)),
            per_fold)


def main():
    df, raw, ch = load()
    print(f"{len(df)} games | {len(raw)} raw features | {len(ch)} choice features")

    sets = {
        "raw": (raw, False),
        "raw+deckz (OLD)": (raw, True),
        "choice (NEW)": (ch, False),
        "choice+deckz": (ch, True),
        # Does the choice block carry signal the old baseline does not already
        # have? Same LDA, just the union of the two feature blocks.
        "raw+choice": (raw + ch, False),
        "raw+choice+deckz": (raw + ch, True),
    }
    protocols = {
        "LOAO": ("deck_family", sorted(df["deck_family"].unique())),
        "LODO": ("deck", sorted(df["deck"].unique())),
    }

    rows, fold_rows = [], []
    for pname, (col, splits) in protocols.items():
        for sname, (cols, dz) in sets.items():
            for n in POOL_NS:
                acc, ari_d, ari_l, pf = evaluate(df, cols, dz, splits, col, n)
                rows.append({"protocol": pname, "features": sname, "pool_N": n,
                             "accuracy": acc, "ARI_direct": ari_d, "ARI_lda": ari_l})
                for r in pf:
                    fold_rows.append({"protocol": pname, "features": sname,
                                      "pool_N": n, **r})
                print(f"  {pname:5s} {sname:16s} N={n:<3d} "
                      f"acc={acc:.3f} ARI_direct={ari_d:.3f} ARI_lda={ari_l:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "cross_deck_results.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "cross_deck_per_fold.csv"), index=False)

    # ---- within-deck ceiling for reference (train and test on same deck) ----
    from sklearn.model_selection import cross_val_score
    within = []
    for name, cols in (("raw", raw), ("choice", ch)):
        for deck, g in df.groupby("deck"):
            X = g[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
            s = cross_val_score(_pipe(), X, g["style"].to_numpy(), cv=5).mean()
            within.append({"features": name, "deck": deck, "within_acc": s})
    wdf = pd.DataFrame(within)
    wdf.to_csv(os.path.join(OUT, "within_deck_reference.csv"), index=False)
    print("\nwithin-deck accuracy (ceiling, N=1):")
    print(wdf.groupby("features")["within_acc"].mean().round(3).to_string())

    _figures(df, res, raw, ch)
    print(f"\nwrote results + figures to {OUT}")


def _figures(df, res, raw, ch):
    # ---- fig 1: headline comparison ----
    titles = {"accuracy": "supervised accuracy (LDA)",
              "ARI_direct": "ARI, KMeans direct (old protocol)",
              "ARI_lda": "ARI, KMeans in transferred LDA space"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, metric in zip(axes, ("accuracy", "ARI_direct", "ARI_lda")):
        sub = res[res.protocol == "LOAO"]
        piv = sub.pivot(index="features", columns="pool_N", values=metric)
        piv = piv.reindex([c for c in ["raw", "raw+deckz (OLD)", "choice (NEW)", "choice+deckz", "raw+choice", "raw+choice+deckz"] if c in piv.index])
        piv.plot(kind="barh", ax=ax, color=["#9ecae1", "#3182bd"])
        chance = 1 / 5 if metric == "accuracy" else 0.0
        ax.axvline(chance, color="crimson", ls="--", lw=1.2,
                   label=f"chance ({chance:.2f})")
        ax.set_title(titles[metric], fontsize=10)
        ax.set_xlabel(metric)
        ax.set_ylabel("")
        ax.legend(title="pooled games N", fontsize=8)
    fig.suptitle("Cross-deck transfer: choice-relative features vs old deck-normalization",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cross_deck_headline.png"), dpi=140)
    plt.close(fig)

    # ---- fig 2: LOAO vs LODO (the sibling-deck leak) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = res[res.pool_N == 10].pivot(index="features", columns="protocol", values="accuracy")
    piv = piv.reindex([c for c in ["raw", "raw+deckz (OLD)", "choice (NEW)", "choice+deckz", "raw+choice", "raw+choice+deckz"] if c in piv.index])
    piv.plot(kind="bar", ax=ax, color=["#fdae6b", "#e6550d"])
    ax.axhline(0.2, color="crimson", ls="--", lw=1.2, label="chance")
    ax.set_title("Sibling-deck leak: holding out one deck vs the whole archetype (N=10)")
    ax.set_ylabel("accuracy")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_loao_vs_lodo.png"), dpi=140)
    plt.close(fig)

    # ---- fig 3: cross-deck stability of each feature ----
    # A feature transfers only if it orders the styles the SAME way on every deck.
    order = []
    for c in ch + raw:
        piv = df.pivot_table(index="style", columns="deck", values=c)
        piv = (piv - piv.mean()) / piv.std(ddof=0)          # compare shape, not scale
        corr = piv.corr().to_numpy()
        iu = np.triu_indices_from(corr, k=1)
        order.append({"feature": c,
                      "block": "choice" if c.startswith("ch_") else "raw",
                      "mean_cross_deck_corr": np.nanmean(corr[iu])})
    od = pd.DataFrame(order).sort_values("mean_cross_deck_corr", ascending=False)
    od.to_csv(os.path.join(OUT, "feature_stability.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(od))))
    colors = ["#3182bd" if b == "choice" else "#bdbdbd" for b in od.block]
    ax.barh(od.feature, od.mean_cross_deck_corr, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean correlation of the style profile between deck pairs")
    ax.set_title("Does the feature rank the styles the same way on every deck?\n"
                 "blue = choice-relative, grey = raw (negative = reverses across decks)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_feature_stability.png"), dpi=140)
    plt.close(fig)

    # ---- fig 4: the flagship feature, style x deck ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, c, t in ((axes[0], "ch_face_pref", "choice-relative: face_pref"),
                     (axes[1], "raw_face_dmg_per_turn", "raw: face damage / turn")):
        piv = df.pivot_table(index="style", columns="deck", values=c).reindex(STYLES)
        im = ax.imshow(piv, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Rows should vary (style signal); columns should not (deck leak)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_face_pref_vs_raw.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")  # headless only when run as a script
    main()
