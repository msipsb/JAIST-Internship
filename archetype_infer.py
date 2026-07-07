"""
archetype_infer.py -- unsupervised deck-archetype labels from me-side card sets.

Purpose (control 4c of the stylometry protocol): so we can compare a player only
against *other players on the same deck archetype*.  Re-identification that
survives that restriction is personal style, not deck recognition.

Method (game-level, because real players switch decks over time):
  1. Each game -> binary vector over me-side played card ids (Coin excluded).
  2. Within each hero class separately:
       a. TF-IDF weight the binary matrix (rare cards define archetypes;
          staples like Fiery War Axe do not).
       b. TruncatedSVD (~20 dims) then KMeans; pick k in 2..8 by silhouette.
       c. Label every game (class, archetype_id).
  3. Merge clusters with < MIN_USERS distinct users into "<class>:other".
  4. Print archetype sizes per class and the top defining cards per archetype.

Sim validation: clustering the sim globally must recover the two deck families
(AggroPirateWarrior vs RenoKazakusMage) at near-perfect purity -- if it does not,
the archetype step is broken and the human numbers cannot be trusted.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).parent
CARDS_META = ROOT / "hearthstonemap-master" / "hearthstonemap-master" / "map" / "cards_meta.json"
OUT_DIR = ROOT / "stylometry_out"

MIN_USERS = 5          # clusters with fewer distinct users are merged to "other"
K_RANGE = range(2, 9)  # candidate cluster counts per hero class
SVD_DIMS = 20
SIL_SAMPLE = 5000      # silhouette on a subsample (full is O(n^2))
SEED = 0


def load_card_names():
    meta = json.loads(CARDS_META.read_text(encoding="utf-8"))
    return {c["id"]: c.get("name", c["id"]) for c in meta if c.get("id")}


# ---------------------------------------------------------------- card-set matrix

def card_set_matrix(me_cards_series):
    """Binary game x card matrix over the vocabulary of a game subset."""
    vocab = {}
    for mc in me_cards_series:
        for cid in set(mc):
            if cid not in vocab:
                vocab[cid] = len(vocab)
    rows, cols = [], []
    for i, mc in enumerate(me_cards_series):
        for cid in set(mc):
            rows.append(i)
            cols.append(vocab[cid])
    data = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(len(me_cards_series), len(vocab)))
    inv_vocab = {j: cid for cid, j in vocab.items()}
    return X, inv_vocab


def _cluster_one(Xbin, seed=SEED):
    """TF-IDF -> SVD -> KMeans with k chosen by silhouette. Returns (labels, Ztfidf)."""
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    n = Xbin.shape[0]
    Xt = TfidfTransformer().fit_transform(Xbin)
    dims = min(SVD_DIMS, Xbin.shape[1] - 1)
    if dims < 2:
        return np.zeros(n, dtype=int), Xt, 1, np.nan
    Z = normalize(TruncatedSVD(n_components=dims, random_state=seed).fit_transform(Xt))

    best = None
    for k in K_RANGE:
        if k >= n:
            break
        lab = KMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(Z)
        if len(np.unique(lab)) < 2:
            continue
        ss = SIL_SAMPLE if n > SIL_SAMPLE else None
        sil = silhouette_score(Z, lab, sample_size=ss, random_state=seed)
        if best is None or sil > best[0]:
            best = (sil, k, lab)
    if best is None:
        return np.zeros(n, dtype=int), Xt, 1, np.nan
    sil, k, labels = best
    return labels, Xt, k, sil


def _defining_cards(Xtfidf, inv_vocab, labels, card_names, topn=10):
    """Top defining card names per cluster by mean TF-IDF weight."""
    out = {}
    Xtfidf = Xtfidf.tocsr()
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        centroid = np.asarray(Xtfidf[idx].mean(axis=0)).ravel()
        top = np.argsort(centroid)[::-1][:topn]
        out[c] = [(card_names.get(inv_vocab[j], inv_vocab[j]), round(float(centroid[j]), 3))
                  for j in top if centroid[j] > 0]
    return out


# ---------------------------------------------------------------- human pipeline

def infer_archetypes(df, cache=None, verbose=True):
    """Add a per-game 'archetype' column (clustered within each hero class)."""
    if cache and Path(cache).exists():
        arche = pd.read_pickle(cache)
        df = df.copy()
        df["archetype"] = arche.reindex(df.index)
        if verbose:
            print(f"[archetype] loaded cached labels ({df['archetype'].nunique()} archetypes)")
        return df

    card_names = load_card_names()
    labels = pd.Series(index=df.index, dtype=object)
    tables = []
    for hero, sub in df.groupby("hero"):
        Xbin, inv_vocab = card_set_matrix(sub["me_cards"])
        lab, Xtfidf, k, sil = _cluster_one(Xbin)

        # merge small clusters (by distinct users) into 'other'
        user_arr = sub["user_hash"].to_numpy()
        keep = {}
        for c in np.unique(lab):
            nusers = len(set(user_arr[lab == c]))
            keep[c] = nusers >= MIN_USERS
        names = {}
        for c in np.unique(lab):
            names[c] = f"{hero}:{c}" if keep[c] else f"{hero}:other"
        arch_labels = np.array([names[c] for c in lab])
        labels.loc[sub.index] = arch_labels

        defs = _defining_cards(Xtfidf, inv_vocab, lab, card_names)
        if verbose:
            print(f"\n[{hero}] {len(sub):,} games  chose k={k} (silhouette={sil:.3f})")
            for c in sorted(np.unique(lab)):
                nusers = len(set(user_arr[lab == c]))
                tag = names[c]
                top = ", ".join(nm for nm, _w in defs[c][:8])
                print(f"   {tag:16s} games={int((lab == c).sum()):6d} users={nusers:4d}  | {top}")
        for c in sorted(np.unique(lab)):
            tables.append(dict(hero=hero, cluster=int(c), label=names[c],
                               games=int((lab == c).sum()),
                               users=len(set(user_arr[lab == c])),
                               top_cards=", ".join(nm for nm, _ in defs[c][:10])))

    df = df.copy()
    df["archetype"] = labels
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(labels, cache)
    if verbose:
        summ = pd.DataFrame(tables)
        print(f"\n[archetype] {df['archetype'].nunique()} archetypes across "
              f"{df['hero'].nunique()} hero classes")
        # cross-check inferred archetype vs the dump-provided hero_deck label
        xt = (pd.crosstab(df["archetype"], df["deck"])
              if df["deck"].notna().any() else None)
    return df


# ---------------------------------------------------------------- sim validation

def validate_sim(sim_df, verbose=True):
    """Cluster sim games globally; purity vs hero_deck must be near-perfect."""
    from sklearn.metrics import adjusted_rand_score
    Xbin, inv_vocab = card_set_matrix(sim_df["me_cards"])
    labels, Xtfidf, k, sil = _cluster_one(Xbin)
    truth = sim_df["deck"].to_numpy()

    ct = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(truth, name="deck"))
    purity = ct.max(axis=1).sum() / ct.values.sum()
    ari = adjusted_rand_score(truth, labels)
    if verbose:
        print("\n=== SIM archetype validation (global clustering vs hero_deck) ===")
        print(f"chosen k={k} (silhouette={sil:.3f})")
        print(ct)
        print(f"purity = {purity:.4f}   adjusted_rand = {ari:.4f}")
        card_names = load_card_names()
        defs = _defining_cards(Xtfidf, inv_vocab, labels, card_names)
        for c in sorted(defs):
            top = ", ".join(nm for nm, _ in defs[c][:8])
            print(f"   cluster {c}: {top}")
    return dict(purity=float(purity), ari=float(ari), k=int(k))


def main():
    from hearthstonemap_load import build_game_frame
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40)

    sim = build_game_frame("sim", cache=OUT_DIR / "frame_sim.pkl", verbose=False)
    res = validate_sim(sim)
    assert res["purity"] > 0.95, "SIM archetype purity too low -- clustering is broken"
    print(">> sim purity check PASSED\n")

    human = build_game_frame("human", cache=OUT_DIR / "frame_human.pkl", verbose=False)
    human = infer_archetypes(human, cache=OUT_DIR / "archetype_human.pkl")


if __name__ == "__main__":
    main()
