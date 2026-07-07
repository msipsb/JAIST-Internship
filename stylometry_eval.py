"""
stylometry_eval.py -- shared re-identification protocol (Stage 1 baseline and,
later, Stage 2 learned embeddings both call into this).

Behavioral stylometry (McIlroy-Young et al., NeurIPS 2021): a player is a
"fingerprint" = the mean of their per-game vectors; identify a player by matching
their query-half fingerprint to reference-half fingerprints via cosine similarity.

Unit / pool model (one framework for every control):
  A re-identification UNIT is a (user, group) pair.  `group` is the pool axis:
    - all-users pool        -> group = "all"        (unit == user)
    - within-hero-class pool -> group = hero class    (unit == user-on-that-class)
    - within-archetype pool  -> group = archetype      (unit == user-on-that-deck)
  For a given N we keep units with >= 2N games, sample 2N (fixed seed), and split
  N reference + N query.  A query unit is matched only against reference units in
  the SAME group; the correct answer is the reference unit with the SAME user.
  This is exactly control 4a/4b/4c: the candidate pool shrinks from "everyone" to
  "same class" to "same deck", so whatever re-identification survives the archetype
  pool is personal style, not deck recognition.

Metrics: top-1, top-5, MRR, and chance = mean over queries of 1/|candidates|.

Feature sets (Stage 1):
  cardseq  -- the 31 public-info cardseq_metrics features
  role     -- LSA card-role fractions (cardseq_embed)
  both     -- cardseq + role
  rhythm   -- cardseq minus card-identity-heavy features (timing/cost/count only)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from cardseq_metrics import FEATURES

# card-identity / card-type heavy features -> dropped for the rhythm-only ablation
IDENTITY_HEAVY = {
    "card_name_entropy", "distinct_card_ratio", "max_card_repeat",
    "minion_fraction", "spell_fraction", "weapon_fraction",
    "first_minion_turn", "first_minion_frac",
}
CARDSEQ_FEATURES = list(FEATURES)
RHYTHM_FEATURES = [f for f in FEATURES if f not in IDENTITY_HEAVY]


def feature_sets(role_names=None):
    """Return the named feature-column lists used by Stage 1."""
    fs = {
        "cardseq": CARDSEQ_FEATURES,
        "rhythm": RHYTHM_FEATURES,
    }
    if role_names:
        fs["role"] = list(role_names)
        fs["both"] = CARDSEQ_FEATURES + list(role_names)
    return fs


# ---------------------------------------------------------------- units

@dataclass
class Unit:
    user: str
    group: str
    ref_pos: np.ndarray     # positional row indices into the feature matrix
    qry_pos: np.ndarray


def build_units(df, group_col, n, seed):
    """(user, group) units with >= 2N games; sample 2N -> N ref + N query.

    df must have a RangeIndex (0..n-1); positions index the feature matrices.
    """
    units = []
    pos = np.asarray(df.index)          # RangeIndex -> identity
    for (user, gval), sub in df.groupby(["user_hash", group_col], sort=False):
        if len(sub) < 2 * n:
            continue
        samp = sub.sample(2 * n, random_state=seed)
        p = samp.index.to_numpy()
        units.append(Unit(user, str(gval), p[:n], p[n:]))
    return units


# ---------------------------------------------------------------- fingerprints

def fingerprints(feat_mat, units):
    """Per-unit reference & query fingerprints.

    Impute (median) and z-score using REFERENCE-set stats only (all reference
    rows across units), then average per unit half.  feat_mat: (n_games, n_feat).
    """
    ref_all = np.concatenate([u.ref_pos for u in units])
    Xref = feat_mat[ref_all]

    med = np.nanmedian(Xref, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xref_i = np.where(np.isnan(Xref), med, Xref)
    mu = Xref_i.mean(axis=0)
    sd = Xref_i.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)

    def fp(pos):
        X = feat_mat[pos]
        X = np.where(np.isnan(X), med, X)
        X = (X - mu) / sd
        return X.mean(axis=0)

    ref_fps = np.vstack([fp(u.ref_pos) for u in units])
    qry_fps = np.vstack([fp(u.qry_pos) for u in units])
    return ref_fps, qry_fps


def _l2(M, eps=1e-9):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + eps)


def score(units, ref_fps, qry_fps, min_candidates=2):
    """Cosine retrieval within each group. Returns top1/top5/mrr/chance/n_units.

    A query is only scored if its group has >= min_candidates reference units
    (so chance < 1 and the number is meaningful).
    """
    R = _l2(ref_fps)
    Q = _l2(qry_fps)

    groups = defaultdict(list)
    for i, u in enumerate(units):
        groups[u.group].append(i)

    top1 = top5 = mrr = chance = 0.0
    nq = 0
    for members in groups.values():
        if len(members) < min_candidates:
            continue
        idxs = np.array(members)
        users = np.array([units[i].user for i in members])
        Rg = R[idxs]
        for qi in members:
            sims = Rg @ Q[qi]
            # correct target = the (unique) reference unit with the same user
            target_local = int(np.where(users == units[qi].user)[0][0])
            order = np.argsort(-sims, kind="stable")
            rank = int(np.where(order == target_local)[0][0]) + 1
            top1 += rank == 1
            top5 += rank <= 5
            mrr += 1.0 / rank
            chance += 1.0 / len(members)
            nq += 1
    if nq == 0:
        return None
    return dict(top1=top1 / nq, top5=top5 / nq, mrr=mrr / nq,
                chance=chance / nq, n_units=nq)


def score_matched(units, ref_fps, qry_fps, C):
    """Pool-size-invariant retrieval: rank the true reference against a FIXED
    C-way candidate set (true + C-1 distractors drawn from the SAME group).

    This makes pools directly comparable (chance == 1/C in every pool).  In the
    all-users pool the distractors have different decks (deck helps -> easier);
    in the within-archetype pool they share the deck (deck useless -> harder), so
    the all-users - within-archetype gap isolates deck recognition.

    Closed form (no Monte-Carlo): if the true ref has b group-mates strictly more
    similar to the query, then over a random (C-1)-subset of the G-1 distractors
    the true ref is top-k iff <= k-1 of those b better ones are sampled -- a
    hypergeometric probability, averaged over queries in groups with >= C units.
    """
    R = _l2(ref_fps)
    Q = _l2(qry_fps)
    groups = defaultdict(list)
    for i, u in enumerate(units):
        groups[u.group].append(i)

    top1 = top5 = 0.0
    nq = 0
    for members in groups.values():
        G = len(members)
        if G < C:
            continue
        idxs = np.array(members)
        users = np.array([units[i].user for i in members])
        Rg = R[idxs]
        denom = comb(G - 1, C - 1)
        for qi in members:
            sims = Rg @ Q[qi]
            t = int(np.where(users == units[qi].user)[0][0])
            b = int(np.sum(sims > sims[t]))                 # distractors above truth
            top1 += comb(G - 1 - b, C - 1) / denom          # none of the b sampled
            k = min(4, b, C - 1)                            # top-5: <=4 of b sampled
            top5 += sum(comb(b, j) * comb(G - 1 - b, C - 1 - j) for j in range(k + 1)) / denom
            nq += 1
    if nq == 0:
        return None
    return dict(top1=top1 / nq, top5=top5 / nq, mrr=np.nan,
                chance=1.0 / C, n_units=nq)


def deck_recognition(df, n_sweep, C=10, seed=0, feature_cols=None, feature_name="cardseq"):
    """Isolate deck recognition on IDENTICAL units.

    Fix the unit = one (user, archetype) pilot fingerprint.  For each pilot query,
    retrieve its own reference in a fixed C-way choice whose C-1 distractors are
    drawn from a shrinking scope:
        distractors=all       -> any pilot          (deck + class help)
        distractors=class     -> same hero class     (archetype still helps)
        distractors=archetype -> same archetype       (deck useless: pure style)
    Because the query/target are unchanged and only the distractor scope shrinks,
    the top-1 drop from 'all' to 'archetype' is exactly how much re-identification
    was riding on the decklist.  Expected: all >= class >= archetype.
    """
    if feature_cols is None:
        feature_cols = CARDSEQ_FEATURES
    mat = df[feature_cols].to_numpy(dtype=float)
    rows = []
    for n in n_sweep:
        units = build_units(df, "archetype", n, seed)
        if len(units) < C:
            continue
        ref_fps, qry_fps = fingerprints(mat, units)
        R, Q = _l2(ref_fps), _l2(qry_fps)
        users = np.array([u.user for u in units])
        arche = np.array([u.group for u in units])
        klass = np.array([g.split(":")[0] for g in arche])
        for scope_name, key in [("all", None), ("class", klass), ("archetype", arche)]:
            top1 = 0.0
            nq = 0
            for qi in range(len(units)):
                cand = np.arange(len(units)) if key is None else np.where(key == key[qi])[0]
                G = len(cand)
                if G < C:
                    continue
                sims = R[cand] @ Q[qi]
                tloc = int(np.where(cand == qi)[0][0])   # this pilot's own reference
                b = int(np.sum(sims > sims[tloc]))
                top1 += comb(G - 1 - b, C - 1) / comb(G - 1, C - 1)
                nq += 1
            if nq:
                rows.append(dict(stage="baseline", pool=f"pilot|distractors={scope_name}",
                                 feature_set=feature_name, N=n, retrieval=f"matched-C{C}",
                                 top1=top1 / nq, top5=np.nan, mrr=np.nan,
                                 chance=1.0 / C, n_units=nq))
    return rows


# ---------------------------------------------------------------- driver

def run_pool(df, stage, pool, group_col, fsets, n_sweep, seed=0, matched_C=10):
    """Sweep N for one pool across all feature sets.

    Emits two retrieval views per (N, feature_set):
      retrieval="full-pool"     -- rank against every reference unit in the group
      retrieval=f"matched-C{C}" -- fixed C-way choice, comparable across pools
    """
    mats = {name: df[cols].to_numpy(dtype=float) for name, cols in fsets.items()}
    rows = []
    for n in n_sweep:
        units = build_units(df, group_col, n, seed)
        if len(units) < 2:
            continue
        for name, mat in mats.items():
            ref_fps, qry_fps = fingerprints(mat, units)
            full = score(units, ref_fps, qry_fps)
            if full is not None:
                rows.append(dict(stage=stage, pool=pool, feature_set=name, N=n,
                                 retrieval="full-pool", **full))
            matched = score_matched(units, ref_fps, qry_fps, matched_C)
            if matched is not None:
                rows.append(dict(stage=stage, pool=pool, feature_set=name, N=n,
                                 retrieval=f"matched-C{matched_C}", **matched))
    return rows
