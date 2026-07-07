"""
Card-text embeddings -> role clusters -> deck-safe role-usage style features.

Pipeline (card2vec prototype, "roles not IDs"):

  1. EMBED   Build a text document per card from cards_meta.json
             (rules text + mechanics + type + rarity + cost bucket), gap-filled
             from SabberStone's CardDefs.xml when present. TF-IDF + truncated
             SVD (LSA) -> one dense vector per card. This follows
             Janusz & Slezak 2018 (card similarity from text embeddings / LSA)
             and Swiechowski et al. 2018 (low-dim card embeddings suffice).

  2. ROLE    KMeans over the vectors of *played* cards only -> N_ROLES clusters
             ("removal", "draw", "taunt wall", "burst", ...). Cards are grouped
             by FUNCTION, not identity. Card identity is the deck's signal
             (cardseq C4/C5); function should travel across decks.

  3. FEATURE One row per game (me-side, Coin excluded):
             role_frac_<r>  share of plays in role r
             role_entropy   diversity of roles used (bits)
             role_gini      concentration on few roles
             unassigned_frac plays whose card had no document (audit)

  4. TEST    Same yardsticks as cardseq_metrics.py so numbers are comparable:
             * U_style vs U_deck (normalized MI) per feature
             * RandomForest: within-deck 5-fold + cross-deck transfer
             * cross-deck profile correlation per feature (the C9 reversal test)
             * optional combined run with cardseq_out/cardseq_features.csv

Outputs (Log_Analysis_Simulated/embed_out/):
  card_roles.csv       card -> role, with distance to centroid
  role_summary.txt     top cards + top terms per role (name the roles here)
  role_features.csv    one row per game: labels + role features
  role_ranking.csv     per-feature U_style / U_deck / U_gap / profile corr
  fig_role_map.png     2-D t-SNE of played cards coloured by role
  fig_role_usage.png   style x role heatmap, one panel per deck
"""
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent          # Log_Analysis_Simulated/
ROOT = HERE.parent                    # project root, where the shared data lives
SIM_JSON = ROOT / "dataset" / "hearthstonemap_sim" / "sim_games.json"
CARDS_META = ROOT / "hearthstonemap-master" / "hearthstonemap-master" / "map" / "cards_meta.json"
CARDDEFS_XML = ROOT / "SabberStone" / "SabberStoneCore" / "resources" / "Data" / "CardDefs.xml"
CARDSEQ_CSV = ROOT / "dataset" / "hearthstonemap_sim" / "cardseq_out" / "cardseq_features.csv"
OUT_DIR = HERE / "embed_out"          # all outputs stay under Log_Analysis_Simulated/

COIN_NAME = "The Coin"
EMBED_DIM = 24          # LSA dims; >16 gives little in card text (Swiechowski et al.)
N_ROLES = 12            # role clusters; tune 8-16 and re-read role_summary.txt
SEED = 0

CARDTYPE_ENUM = {"3": "HERO", "4": "MINION", "5": "SPELL",
                 "6": "ENCHANTMENT", "7": "WEAPON", "10": "HERO_POWER"}

TAG_RE = re.compile(r"<[^>]+>")           # <b>, </b>, <i> ...
MARKUP_RE = re.compile(r"[#$]\d*|\[x\]")  # $4, #2, [x] layout marker
NONWORD_RE = re.compile(r"[^a-z0-9_ ]+")


# ---------------------------------------------------------------- card documents

def clean_text(t):
    t = TAG_RE.sub(" ", t or "")
    t = MARKUP_RE.sub(" ", t)
    t = NONWORD_RE.sub(" ", t.lower())
    return " ".join(t.split())


def cost_bucket(cost):
    if cost is None:
        return None
    c = float(cost)
    return "cost_low" if c <= 2 else "cost_mid" if c <= 5 else "cost_high"


def card_doc(name, text, mechanics, ctype, rarity, cost):
    """One whitespace document per card. Prefixed tokens keep the sources apart
    so TF-IDF can weight rules text vs mechanics vs type on its own."""
    parts = [clean_text(text)]
    parts += [f"mech_{m.lower()}" for m in (mechanics or [])]
    if ctype:
        parts.append(f"type_{ctype.lower()}")
    if rarity:
        parts.append(f"rarity_{rarity.lower()}")
    cb = cost_bucket(cost)
    if cb:
        parts.append(cb)
    return " ".join(p for p in parts if p).strip()


def load_card_docs():
    """id/name -> document. cards_meta.json first, gaps filled from CardDefs.xml
    (enumID 185=name, 184=text, 202=type, 48=cost), mirroring sim_to_hearthstonemap."""
    by_id, by_name = {}, {}
    meta = json.loads(CARDS_META.read_text(encoding="utf-8"))
    for c in meta:
        doc = card_doc(c.get("name"), c.get("text"), c.get("mechanics"),
                       c.get("type"), c.get("rarity"), c.get("cost"))
        if not doc:
            continue
        if c.get("id"):
            by_id[c["id"]] = doc
        nm = c.get("name")
        if nm and (nm not in by_name or c.get("collectible")):
            by_name[nm] = doc

    filled = 0
    if CARDDEFS_XML.exists():
        for _ev, el in ET.iterparse(str(CARDDEFS_XML), events=("end",)):
            if el.tag != "Entity":
                continue
            cardid = el.get("CardID")
            name = text = ctype = cost = None
            for tag in el.findall("Tag"):
                eid = tag.get("enumID")
                en = tag.find("enUS")
                if eid == "185" and en is not None:
                    name = en.text
                elif eid == "184" and en is not None:
                    text = en.text
                elif eid == "202":
                    ctype = CARDTYPE_ENUM.get(tag.get("value"))
                elif eid == "48":
                    cost = tag.get("value")
            doc = card_doc(name, text, None, ctype, None, cost)
            if doc:
                if cardid and cardid not in by_id:
                    by_id[cardid] = doc
                    filled += 1
                if name and name not in by_name:
                    by_name[name] = doc
            el.clear()
    print(f"Card docs: {len(by_id)} ids / {len(by_name)} names ({filled} ids gap-filled)")
    return by_id, by_name


# ---------------------------------------------------------------- embed + roles

def played_cards(sim, by_id, by_name):
    """(key, display name, doc, n_plays) for every distinct card played by either
    side. Both sides feed the role space; features later use the me-side only."""
    seen = {}
    counts = Counter()
    for g in sim["games"]:
        for h in g["card_history"]:
            card = h.get("card", {})
            name = card.get("name")
            if not name or name == COIN_NAME:
                continue
            key = card.get("id") or name
            doc = by_id.get(card.get("id")) or by_name.get(name)
            seen[key] = (name, doc)
            counts[key] += 1
    rows = [(k, nm, doc, counts[k]) for k, (nm, doc) in seen.items()]
    n_missing = sum(1 for _k, _n, doc, _c in rows if not doc)
    print(f"Played vocabulary: {len(rows)} distinct cards ({n_missing} without a document)")
    return rows


def embed_and_cluster(vocab_rows):
    """TF-IDF + SVD on the played cards' documents, then KMeans -> roles.
    Returns role_of (key -> role id), the 2-D map, and per-role descriptions."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    docs_rows = [(k, nm, doc, c) for (k, nm, doc, c) in vocab_rows if doc]
    keys = [k for (k, _n, _d, _c) in docs_rows]
    names = [n for (_k, n, _d, _c) in docs_rows]
    docs = [d for (_k, _n, d, _c) in docs_rows]

    vec = TfidfVectorizer(min_df=1, token_pattern=r"[a-z0-9_]+")
    X = vec.fit_transform(docs)
    dim = min(EMBED_DIM, X.shape[1] - 1, len(docs) - 1)
    svd = TruncatedSVD(n_components=max(2, dim), random_state=SEED)
    Z = svd.fit_transform(X)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    print(f"Embedding: TF-IDF {X.shape} -> SVD {Z.shape[1]}d "
          f"(explained var {svd.explained_variance_ratio_.sum():.2f})")

    k = min(N_ROLES, len(docs))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    role_of = dict(zip(keys, km.labels_))
    dists = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)

    # describe each role: top TF-IDF terms of its centroid + closest cards
    terms = np.array(vec.get_feature_names_out())
    centroids_tfidf = svd.inverse_transform(km.cluster_centers_)
    descriptions = {}
    for r in range(k):
        top_terms = terms[np.argsort(centroids_tfidf[r])[::-1][:6]]
        members = [(names[i], dists[i]) for i in range(len(keys)) if km.labels_[i] == r]
        members.sort(key=lambda t: t[1])
        descriptions[r] = (list(top_terms), [m for m, _ in members[:8]], len(members))

    roles_df = pd.DataFrame({"key": keys, "name": names,
                             "role": km.labels_, "dist_to_centroid": dists.round(4)})
    return role_of, Z, km.labels_, names, roles_df, descriptions


# ---------------------------------------------------------------- per-game features

def role_feature_names(n_roles):
    return ([f"role_frac_{r}" for r in range(n_roles)]
            + ["role_entropy", "role_gini", "unassigned_frac"])


def extract_role_features(game, role_of, n_roles):
    plays, unassigned = [], 0
    for h in game["card_history"]:
        if h.get("player") != "me":
            continue
        card = h.get("card", {})
        name = card.get("name")
        if not name or name == COIN_NAME:
            continue
        key = card.get("id") or name
        r = role_of.get(key)
        if r is None:
            unassigned += 1
        else:
            plays.append(r)
    f = {c: np.nan for c in role_feature_names(n_roles)}
    total = len(plays) + unassigned
    if total == 0:
        return f
    counts = Counter(plays)
    probs = np.array([counts.get(r, 0) for r in range(n_roles)], float)
    probs = probs / total
    for r in range(n_roles):
        f[f"role_frac_{r}"] = probs[r]
    nz = probs[probs > 0]
    f["role_entropy"] = float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0
    srt = np.sort(probs)
    n = len(srt)
    f["role_gini"] = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum())) if srt.sum() else np.nan
    f["unassigned_frac"] = unassigned / total
    return f


def build_frame(sim, role_of, n_roles):
    rows = []
    for g in sim["games"]:
        f = extract_role_features(g, role_of, n_roles)
        f["style"] = g["hero_playstyle"]
        f["deck"] = g["hero_deck"]
        f["result"] = g.get("result")
        rows.append(f)
    df = pd.DataFrame(rows)
    print(f"Built {len(df)} game rows x {len(role_feature_names(n_roles))} role features")
    return df


# ---------------------------------------------------------------- analysis

def discriminativeness(df, features):
    """U_style vs U_deck per feature + cross-deck profile correlation (C9 test)."""
    from sklearn.feature_selection import mutual_info_classif

    X = df[features].apply(lambda c: c.fillna(c.median())).replace([np.inf, -np.inf], 0.0)
    X = X.fillna(0.0).to_numpy()
    mi_style = mutual_info_classif(X, df["style"].to_numpy(), random_state=SEED)
    mi_deck = mutual_info_classif(X, df["deck"].to_numpy(), random_state=SEED)

    def entropy_nats(s):
        p = s.value_counts(normalize=True).to_numpy()
        return float(-(p * np.log(p)).sum())
    H_style, H_deck = entropy_nats(df["style"]), entropy_nats(df["deck"])

    # per-deck style profile (mean per style); corr(+1) = same meaning on both
    # decks, corr(-1) = reversed -- the failure mode deck-normalization can't fix
    decks = sorted(df["deck"].unique())
    prof_corr = {}
    if len(decks) == 2:
        for feat in features:
            a = df[df["deck"] == decks[0]].groupby("style")[feat].mean()
            b = df[df["deck"] == decks[1]].groupby("style")[feat].mean()
            ab = pd.concat([a, b], axis=1).dropna()
            prof_corr[feat] = float(np.corrcoef(ab.iloc[:, 0], ab.iloc[:, 1])[0, 1]) \
                if len(ab) >= 3 and ab.std().min() > 0 else np.nan

    rank = pd.DataFrame({
        "feature": features,
        "U_style": mi_style / H_style,
        "U_deck": mi_deck / H_deck,
    })
    rank["U_gap"] = rank["U_style"] - rank["U_deck"]
    rank["cross_deck_profile_corr"] = rank["feature"].map(prof_corr)
    return rank.sort_values("U_style", ascending=False).reset_index(drop=True)


def classifier_check(df, features, label="role features"):
    """Within-deck 5-fold + cross-deck transfer; same RF setup as cardseq_metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    def Xy(d):
        return d[features].replace([np.inf, -np.inf], np.nan).to_numpy(), d["style"].to_numpy()

    clf = lambda: make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestClassifier(n_estimators=200,
                                                       random_state=SEED, n_jobs=-1))
    out = {}
    for deck in sorted(df["deck"].unique()):
        Xd, yd = Xy(df[df["deck"] == deck])
        out[f"within_{deck}_5fold"] = float(np.mean(
            cross_val_score(clf(), Xd, yd, cv=5, scoring="accuracy")))
    decks = sorted(df["deck"].unique())
    if len(decks) == 2:
        for tr, te in [(decks[0], decks[1]), (decks[1], decks[0])]:
            Xtr, ytr = Xy(df[df["deck"] == tr])
            Xte, yte = Xy(df[df["deck"] == te])
            out[f"transfer_{tr[:4]}->{te[:4]}"] = float(
                (clf().fit(Xtr, ytr).predict(Xte) == yte).mean())
    print(f"\n=== classifier check ({label}) --- chance = 0.200 ===")
    for k, v in out.items():
        print(f"  {k:34s} {v:.3f}")
    return out


def combined_check(df):
    """If cardseq_features.csv exists and rows line up, run hand-made + role
    features together. Rows in that CSV follow sim_games.json order, so a
    positional merge is valid; style/deck labels are asserted to match."""
    if not CARDSEQ_CSV.exists():
        print("\n(cardseq_features.csv not found -- skipping combined run)")
        return
    cs = pd.read_csv(CARDSEQ_CSV)
    if len(cs) != len(df) or not (cs["style"].to_numpy() == df["style"].to_numpy()).all() \
            or not (cs["deck"].to_numpy() == df["deck"].to_numpy()).all():
        print("\n(cardseq_features.csv rows do not line up -- skipping combined run)")
        return
    label_cols = {"style", "deck", "opp_style", "result"}
    cs_feats = [c for c in cs.columns if c not in label_cols]
    both = pd.concat([df.reset_index(drop=True),
                      cs[cs_feats].reset_index(drop=True)], axis=1)
    classifier_check(both, cs_feats, label="cardseq only (baseline)")
    role_feats = [c for c in df.columns if c not in label_cols]
    classifier_check(both, role_feats + cs_feats, label="cardseq + role combined")


# ---------------------------------------------------------------- figures

def make_figures(df, Z, labels, names, role_feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # (1) card map: t-SNE of the embedding, coloured by role
    perp = max(2, min(30, (len(Z) - 1) // 3))
    P = TSNE(n_components=2, random_state=SEED, perplexity=perp, init="pca").fit_transform(Z)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab20", s=22)
    for r in np.unique(labels):
        cx, cy = P[labels == r].mean(axis=0)
        ax.annotate(f"R{r}", (cx, cy), fontsize=11, fontweight="bold",
                    ha="center", va="center")
    ax.set_title("Played cards in embedding space (t-SNE), coloured by role cluster")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_role_map.png", dpi=130)
    plt.close(fig)

    # (2) style x role heatmap, one panel per deck; same colour scale
    fracs = [f for f in role_feats if f.startswith("role_frac_")]
    decks = sorted(df["deck"].unique())
    fig, axes = plt.subplots(1, len(decks), figsize=(6.5 * len(decks), 4.4), squeeze=False)
    for ax, deck in zip(axes[0], decks):
        m = df[df["deck"] == deck].groupby("style")[fracs].mean()
        z = (m - m.mean()) / m.std().replace(0, np.nan)
        im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([f.replace("role_frac_", "R") for f in fracs], fontsize=8)
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index, fontsize=9)
        ax.set_title(deck, fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="z-score within deck")
    fig.suptitle("Role usage by style -- similar row patterns across panels = deck-safe")
    fig.savefig(OUT_DIR / "fig_role_usage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures -> {OUT_DIR}")


# ---------------------------------------------------------------- main

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 80)

    by_id, by_name = load_card_docs()
    sim = json.loads(SIM_JSON.read_text(encoding="utf-8"))
    vocab = played_cards(sim, by_id, by_name)

    role_of, Z, labels, names, roles_df, desc = embed_and_cluster(vocab)
    roles_df.sort_values(["role", "dist_to_centroid"]).to_csv(
        OUT_DIR / "card_roles.csv", index=False)

    lines = []
    print("\n=== roles (name them by reading top terms + closest cards) ===")
    for r, (terms, cards, n) in desc.items():
        if n == 0:
            continue    # KMeans can leave a cluster empty on tiny vocabularies
        line = f"R{r:<2d} ({n:>3d} cards)  terms: {', '.join(terms)}\n" \
               f"    e.g. {', '.join(cards)}"
        print(line)
        lines.append(line)
    (OUT_DIR / "role_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    n_roles = len(desc)
    df = build_frame(sim, role_of, n_roles)
    df.to_csv(OUT_DIR / "role_features.csv", index=False)
    feats = role_feature_names(n_roles)

    rank = discriminativeness(df, feats)
    rank.round(4).to_csv(OUT_DIR / "role_ranking.csv", index=False)
    print("\n=== role features: style signal vs deck leakage ===")
    print("    U_gap > 0 => more style than deck; profile corr near +1 => deck-safe")
    print(rank.round(3).to_string(index=False))

    classifier_check(df, feats, label="role features only")
    combined_check(df)

    make_figures(df, Z, labels, names, feats)
    print(f"\nWrote:\n  " + "\n  ".join(str(OUT_DIR / f) for f in
          ["card_roles.csv", "role_summary.txt", "role_features.csv", "role_ranking.csv"]))


if __name__ == "__main__":
    main()
