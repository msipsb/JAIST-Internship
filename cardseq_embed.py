"""
cardseq_embed.py -- LSA card-text embeddings, functional "role" clusters, and
per-game role features for the stylometry baseline (feature sets (b) and (c)).

Idea: a raw card *id* is deck-leaky (it names the exact card).  A card's
functional *role* -- "cheap aggressive minion", "board wipe", "card draw",
"removal spell" -- is a softer, more style-ish abstraction.  We learn roles
unsupervised from card text:

  1. Each collectible card -> a short document = card type + cost bucket +
     mechanics tags + rules text.
  2. TF-IDF over those documents -> TruncatedSVD (LSA, latent semantics).
  3. KMeans on the LSA vectors -> K roles (K chosen by silhouette).

A game's role features are the fraction of its me-side plays that fall in each
role (+ role entropy + distinct-role ratio).  These are card-derived but do not
name specific cards, so they sit between pure rhythm and raw card identity.

Card metadata: hearthstonemap-master/.../map/cards_meta.json  (id, text, type,
cost, mechanics).  Cards a game plays that are not in the metadata fall into an
"unknown" bucket that is excluded from the role-fraction denominator.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
CARDS_META = ROOT / "hearthstonemap-master" / "hearthstonemap-master" / "map" / "cards_meta.json"
OUT_DIR = ROOT / "stylometry_out"

_TAG_RE = re.compile(r"<[^>]+>")            # strip <b>..</b> markup in rules text
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _cost_bucket(cost):
    if cost is None:
        return "costNA"
    if cost <= 1:
        return "cost_lo"
    if cost <= 3:
        return "cost_mid"
    if cost <= 5:
        return "cost_hi"
    return "cost_top"


def _card_document(c):
    """Build the text document for one card_meta record."""
    parts = []
    t = c.get("type")
    if t:
        parts += [t.lower()] * 3                      # weight the card type
    parts.append(_cost_bucket(c.get("cost")))
    for m in c.get("mechanics", []) or []:
        parts += [m.lower()] * 2                       # weight explicit mechanics
    txt = _TAG_RE.sub(" ", c.get("text") or "")
    parts += _TOKEN_RE.findall(txt.lower())
    return " ".join(parts)


def load_card_documents():
    """id -> document string, over collectible cards with usable text/type."""
    meta = json.loads(CARDS_META.read_text(encoding="utf-8"))
    docs = {}
    for c in meta:
        cid = c.get("id")
        if not cid:
            continue
        # keep collectible playable cards; these are what appears in card_history
        if not (c.get("collectible") or c.get("text")):
            continue
        doc = _card_document(c)
        if doc.strip():
            docs[cid] = doc
    return docs


# ---------------------------------------------------------------- role model

def build_role_index(k_range=range(6, 15), svd_dims=30, seed=0,
                     cache=None, verbose=True):
    """Cluster cards into K functional roles (K picked by silhouette).

    Returns dict with: card_role (id->role int), K, top_cards (role->[ids]),
    and defining_tokens (role->[terms]).
    """
    if cache and Path(cache).exists():
        idx = pd.read_pickle(cache)
        if verbose:
            print(f"[roles] loaded cached role index ({idx['K']} roles)")
        return idx

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    docs = load_card_documents()
    ids = list(docs)
    corpus = [docs[i] for i in ids]

    vec = TfidfVectorizer(min_df=3, max_df=0.6, sublinear_tf=True)
    X = vec.fit_transform(corpus)
    dims = min(svd_dims, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=dims, random_state=seed)
    Z = normalize(svd.fit_transform(X))

    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lab = km.fit_predict(Z)
        sil = silhouette_score(Z, lab)
        if verbose:
            print(f"[roles] K={k:2d}  silhouette={sil:.4f}")
        if best is None or sil > best[0]:
            best = (sil, k, lab, km)
    sil, K, labels, km = best
    if verbose:
        print(f"[roles] chosen K={K} (silhouette={sil:.4f}) over {len(ids)} cards")

    card_role = {cid: int(lbl) for cid, lbl in zip(ids, labels)}

    # defining terms per role: top tf-idf features of the cluster centroid (in
    # original term space, via inverse SVD transform of the KMeans centroid)
    terms = np.array(vec.get_feature_names_out())
    centroid_term = km.cluster_centers_ @ svd.components_       # K x n_terms
    defining = {r: terms[np.argsort(centroid_term[r])[::-1][:12]].tolist()
                for r in range(K)}
    # a few example cards nearest each centroid
    top_cards = {r: [] for r in range(K)}
    d2 = km.transform(Z)                                        # game-card to centroid dist
    for r in range(K):
        order = np.argsort(d2[:, r])[:12]
        top_cards[r] = [ids[i] for i in order]

    idx = dict(card_role=card_role, K=K, defining_tokens=defining,
               top_cards=top_cards, silhouette=float(sil))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(idx, cache)
    return idx


def load_lsa_card_vectors(dims=32, seed=0, cache=None, verbose=True):
    """id -> frozen LSA text vector (TF-IDF + TruncatedSVD, L2-normalized).

    Used by Stage-2 GE2E variant (b) as deck-safer card representations.
    Returns (dict id->np.float32[dim], dim).
    """
    if cache and Path(cache).exists():
        d = pd.read_pickle(cache)
        if verbose:
            print(f"[lsa] loaded cached card vectors ({len(d['vecs'])} cards, dim {d['dim']})")
        return d["vecs"], d["dim"]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    docs = load_card_documents()
    ids = list(docs)
    X = TfidfVectorizer(min_df=3, max_df=0.6, sublinear_tf=True).fit_transform(
        [docs[i] for i in ids])
    dim = min(dims, X.shape[1] - 1)
    Z = normalize(TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X))
    vecs = {cid: Z[i].astype("float32") for i, cid in enumerate(ids)}
    if verbose:
        print(f"[lsa] built {len(vecs)} card vectors, dim {dim}")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"vecs": vecs, "dim": dim}, cache)
    return vecs, dim


ROLE_ENTROPY = "role_entropy"
ROLE_DISTINCT = "role_distinct_ratio"
ROLE_UNKNOWN = "role_unknown_frac"


def role_feature_names(K):
    return [f"role_frac_{r}" for r in range(K)] + [ROLE_ENTROPY, ROLE_DISTINCT, ROLE_UNKNOWN]


def game_role_features(me_cards, card_role, K):
    """me_cards (tuple of card ids) -> role-fraction feature dict."""
    names = role_feature_names(K)
    f = {n: np.nan for n in names}
    if not me_cards:
        return f
    counts = np.zeros(K, dtype=float)
    unknown = 0
    for cid in me_cards:
        r = card_role.get(cid)
        if r is None:
            unknown += 1
        else:
            counts[r] += 1
    total_known = counts.sum()
    f[ROLE_UNKNOWN] = unknown / len(me_cards)
    if total_known <= 0:
        return f
    frac = counts / total_known
    for r in range(K):
        f[f"role_frac_{r}"] = frac[r]
    p = frac[frac > 0]
    f[ROLE_ENTROPY] = float(-(p * np.log2(p)).sum())
    f[ROLE_DISTINCT] = float((counts > 0).sum()) / K
    return f


def add_role_features(df, role_index=None, verbose=True):
    """Append role-fraction feature columns to a game frame. Returns (df, names)."""
    if role_index is None:
        role_index = build_role_index(cache=OUT_DIR / "role_index.pkl", verbose=verbose)
    card_role, K = role_index["card_role"], role_index["K"]
    names = role_feature_names(K)
    feats = [game_role_features(mc, card_role, K) for mc in df["me_cards"]]
    rf = pd.DataFrame(feats, index=df.index).astype("float32")
    out = pd.concat([df, rf], axis=1)
    return out, names


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = build_role_index(cache=OUT_DIR / "role_index.pkl")
    print(f"\n=== {idx['K']} card roles (silhouette {idx['silhouette']:.3f}) ===")
    for r in range(idx["K"]):
        toks = ", ".join(idx["defining_tokens"][r][:8])
        print(f"role {r:2d}: {toks}")


if __name__ == "__main__":
    main()
