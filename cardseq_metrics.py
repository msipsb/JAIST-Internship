"""
Card-sequence-only play-style metrics + discriminativeness analysis.

This is the "public information" counterpart to the V1-V4 engine-trace parsers.
It consumes dataset/hearthstonemap_sim/sim_games.json -- which carries only what
a real hearthstonemap log carries: the per-turn sequence of *played* cards
(turn, player, card{id,name,mana}) plus coin/result -- and computes every metric
that can be reconstructed from that alone.

Two metric groups, tagged in FEATURE_ORIGIN:

  SURVIVING -- the subset of the V1-V4 metrics that only need mana + timing +
               card-type (recovered from the card id).  Everything the old
               parsers built on board state, attacks, damage, hero power, draws,
               or mana crystals is NOT reconstructable and is intentionally absent.

  NEW       -- sequence-structure metrics (entropy / diversity / repetition /
               curve tilt / tempo evenness) and RELATIONAL tempo metrics that use
               the opponent's play sequence (an upside the me-side-only engine
               parsers never used).

Analysis (in __main__), target = hero_playstyle (5 classes):
  * Kruskal-Wallis H + p + epsilon-squared effect size per feature.
  * Mutual information I(feature; playstyle)  AND  I(feature; deck).
    A meaningful *style* feature has high MI with style and LOW MI with deck;
    the MI gap (style - deck) flags deck leakage -- the project's core problem.
  * A compact classifier check: within-deck 5-fold accuracy and cross-deck
    transfer (train on one deck, test on the other), mirroring the V4 study.

Outputs (dataset/hearthstonemap_sim/cardseq_out/):
  cardseq_features.csv   one row per game: labels + all features
  cardseq_ranking.csv    per-feature KW / effect size / MI(style) / MI(deck)
  fig_mi_style_vs_deck.png, fig_top_features_by_style.png
"""
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SIM_JSON = ROOT / "dataset" / "hearthstonemap_sim" / "sim_games.json"
CARDS_META = ROOT / "hearthstonemap-master" / "hearthstonemap-master" / "map" / "cards_meta.json"
CARDDEFS_XML = ROOT / "SabberStone" / "SabberStoneCore" / "resources" / "Data" / "CardDefs.xml"
OUT_DIR = ROOT / "dataset" / "hearthstonemap_sim" / "cardseq_out"

MANA_CAP = 10
CHECKPOINTS = [3, 5, 7, 9]
COIN_NAME = "The Coin"

# CardDefs enumID 202 = CARDTYPE; map the integer values to the same labels
# cards_meta.json uses, so gap-filled ids get a type too.
CARDTYPE_ENUM = {"3": "HERO", "4": "MINION", "5": "SPELL",
                 "6": "ENCHANTMENT", "7": "WEAPON", "10": "HERO_POWER"}


# ---------------------------------------------------------------- card-type index

def load_type_index():
    """id/name -> card type (MINION/SPELL/WEAPON/...), mirroring how
    sim_to_hearthstonemap builds its card index: cards_meta first, then
    gap-fill any missing names from SabberStone's own CardDefs.xml."""
    by_id, by_name = {}, {}
    meta = json.loads(CARDS_META.read_text(encoding="utf-8"))
    for c in meta:
        t = c.get("type")
        if not t:
            continue
        if c.get("id"):
            by_id[c["id"]] = t
        nm = c.get("name")
        if nm and (nm not in by_name or c.get("collectible")):
            by_name[nm] = t

    filled = 0
    for _ev, el in ET.iterparse(str(CARDDEFS_XML), events=("end",)):
        if el.tag != "Entity":
            continue
        cardid = el.get("CardID")
        name = ctype = None
        for tag in el.findall("Tag"):
            eid = tag.get("enumID")
            if eid == "185":
                en = tag.find("enUS")
                if en is not None:
                    name = en.text
            elif eid == "202":
                ctype = CARDTYPE_ENUM.get(tag.get("value"))
        if cardid and ctype and cardid not in by_id:
            by_id[cardid] = ctype
            filled += 1
        if name and ctype and name not in by_name:
            by_name[name] = ctype
        el.clear()
    print(f"Type index: {len(by_id)} ids / {len(by_name)} names ({filled} ids gap-filled)")
    return by_id, by_name


# ---------------------------------------------------------------- feature origin

# origin, source version, and the style each metric was meant to "tell" on
FEATURE_ORIGIN = {
    # ---- SURVIVING from V1-V4 (mana + timing + type only) ----
    "n_my_turns":            ("surviving", "V1-V4", "length"),
    "n_cards":               ("surviving", "V1",    "-"),
    "cards_per_turn":        ("surviving", "V1",    "aggro"),
    "avg_card_cost":         ("surviving", "V1-V4", "curve"),
    "max_card_cost":         ("surviving", "V3/V4", "ramp"),
    "mana_spent":            ("surviving", "V1",    "-"),
    "mana_eff":              ("surviving", "V1-V4", "tempo"),
    "mana_per_turn":         ("surviving", "V1",    "tempo"),
    "mana_floated_per_turn": ("surviving", "V4",    "control"),
    "minion_fraction":       ("surviving", "V2-V4", "midrange"),
    "spell_fraction":        ("surviving", "V2+",   "control"),
    "weapon_fraction":       ("surviving", "V2+",   "aggro"),
    "first_play_turn":       ("surviving", "V1",    "-"),
    "first_minion_turn":     ("surviving", "V3",    "midrange"),
    "first_minion_frac":     ("surviving", "V4",    "midrange"),
    "mana_slope":            ("surviving", "V3/V4", "ramp"),
    "mana_eff_t3":           ("surviving", "V4",    "tempo"),
    "mana_eff_t5":           ("surviving", "V4",    "tempo"),
    "mana_eff_t7":           ("surviving", "V4",    "tempo"),
    "mana_eff_t9":           ("surviving", "V4",    "tempo"),
    "has_coin":              ("surviving", "V1",    "-"),
    "coin_turn":             ("surviving", "V1",    "-"),
    # ---- NEW: sequence-structure ----
    "card_name_entropy":     ("new",        "-",    "diversity"),
    "mana_cost_entropy":     ("new",        "-",    "diversity"),
    "distinct_card_ratio":   ("new",        "-",    "diversity"),
    "max_card_repeat":       ("new",        "-",    "combo"),
    "front_load":            ("new",        "-",    "aggro"),
    "cost_tilt":             ("new",        "-",    "ramp/control"),
    "plays_per_turn_cv":     ("new",        "-",    "tempo evenness"),
    # ---- NEW: relational (uses opponent sequence) ----
    "mana_lead_t5":          ("relational", "-",    "tempo race"),
    "card_lead_t5":          ("relational", "-",    "tempo race"),
}
FEATURES = list(FEATURE_ORIGIN)


# ---------------------------------------------------------------- helpers

def shannon_entropy(counts):
    """Shannon entropy (bits) of a Counter / list of counts."""
    vals = [c for c in (counts.values() if isinstance(counts, dict) else counts) if c > 0]
    tot = sum(vals)
    if tot <= 0:
        return np.nan
    return -sum((c / tot) * math.log2(c / tot) for c in vals)


def mana_available(n_turns, has_coin):
    """Cumulative crystals a player could have spent over n_turns (+1 for Coin),
    matching V1-V4's mana_available()."""
    full = min(n_turns, MANA_CAP)
    base = full * (full + 1) // 2 + max(0, n_turns - MANA_CAP) * MANA_CAP
    return base + (1 if has_coin else 0)


def _side_plays(history, side, type_by_id, type_by_name):
    """Non-coin plays for one side: list of (turn, cost, ctype, name); plus coin_turn."""
    plays, coin_turn = [], np.nan
    for h in history:
        if h.get("player") != side:
            continue
        card = h.get("card", {})
        name = card.get("name")
        turn = h.get("turn")
        if name == COIN_NAME:
            if np.isnan(coin_turn):
                coin_turn = turn
            continue
        cost = card.get("mana")
        cost = np.nan if cost is None else float(cost)
        ctype = type_by_id.get(card.get("id")) or type_by_name.get(name)
        plays.append((turn, cost, ctype, name))
    return plays, coin_turn


# ---------------------------------------------------------------- extraction

def extract_features(game, type_by_id, type_by_name):
    """One sim game -> dict of me-side (+ relational) card-sequence features."""
    hist = game["card_history"]
    me, coin_turn = _side_plays(hist, "me", type_by_id, type_by_name)
    opp, _ = _side_plays(hist, "opponent", type_by_id, type_by_name)
    has_coin = bool(game.get("coin"))

    f = {k: np.nan for k in FEATURES}
    f["has_coin"] = int(has_coin)
    f["coin_turn"] = coin_turn
    if not me:
        return f

    turns = [t for (t, _c, _ct, _n) in me]
    costs = [c for (_t, c, _ct, _n) in me if not np.isnan(c)]
    names = [n for (_t, _c, _ct, n) in me]
    n_turns = max(turns)
    n_cards = len(me)

    # per-turn aggregates (missing turns = 0 spent / 0 plays)
    spent_by_turn = defaultdict(float)
    count_by_turn = defaultdict(int)
    for (t, c, _ct, _n) in me:
        count_by_turn[t] += 1
        if not np.isnan(c):
            spent_by_turn[t] += c

    mana_spent = float(sum(costs)) if costs else 0.0
    avail = mana_available(n_turns, has_coin)

    # ---- SURVIVING ----
    f["n_my_turns"] = n_turns
    f["n_cards"] = n_cards
    f["cards_per_turn"] = n_cards / n_turns
    f["avg_card_cost"] = float(np.mean(costs)) if costs else np.nan
    f["max_card_cost"] = float(np.max(costs)) if costs else np.nan
    f["mana_spent"] = mana_spent
    f["mana_eff"] = mana_spent / avail if avail else np.nan
    f["mana_per_turn"] = mana_spent / n_turns
    f["mana_floated_per_turn"] = max(0.0, avail - mana_spent) / n_turns
    f["first_play_turn"] = float(min(turns))

    types = Counter(ct for (_t, _c, ct, _n) in me if ct)
    typed = sum(types.values())
    if typed:
        f["minion_fraction"] = types.get("MINION", 0) / typed
        f["spell_fraction"] = types.get("SPELL", 0) / typed
        f["weapon_fraction"] = types.get("WEAPON", 0) / typed
    minion_turns = [t for (t, _c, ct, _n) in me if ct == "MINION"]
    if minion_turns:
        f["first_minion_turn"] = float(min(minion_turns))
        f["first_minion_frac"] = min(minion_turns) / n_turns

    if len(spent_by_turn) >= 2:
        xs = list(range(1, n_turns + 1))
        ys = [spent_by_turn.get(t, 0.0) for t in xs]
        f["mana_slope"] = float(np.polyfit(xs, ys, 1)[0])
    for k in CHECKPOINTS:
        if n_turns >= k:                       # only defined if the game reached turn k
            f[f"mana_eff_t{k}"] = spent_by_turn.get(k, 0.0) / min(k, MANA_CAP)

    # ---- NEW: sequence structure ----
    f["card_name_entropy"] = shannon_entropy(Counter(names))
    if costs:
        f["mana_cost_entropy"] = shannon_entropy(Counter(costs))
    f["distinct_card_ratio"] = len(set(names)) / n_cards
    f["max_card_repeat"] = max(Counter(names).values())
    f["front_load"] = sum(1 for t in turns if t <= 3) / n_cards
    early = [c for (t, c, _ct, _n) in me if t <= 3 and not np.isnan(c)]
    late = [c for (t, c, _ct, _n) in me if t >= 4 and not np.isnan(c)]
    if early and late:
        f["cost_tilt"] = float(np.mean(late) - np.mean(early))
    per_turn_counts = [count_by_turn.get(t, 0) for t in range(1, n_turns + 1)]
    mu = np.mean(per_turn_counts)
    f["plays_per_turn_cv"] = float(np.std(per_turn_counts) / mu) if mu > 0 else np.nan

    # ---- NEW: relational (vs opponent sequence) ----
    my_mana_5 = sum(c for (t, c, _ct, _n) in me if t <= 5 and not np.isnan(c))
    opp_mana_5 = sum(c for (t, c, _ct, _n) in opp if t <= 5 and not np.isnan(c))
    f["mana_lead_t5"] = my_mana_5 - opp_mana_5
    my_cards_5 = sum(1 for (t, _c, _ct, _n) in me if t <= 5)
    opp_cards_5 = sum(1 for (t, _c, _ct, _n) in opp if t <= 5)
    f["card_lead_t5"] = my_cards_5 - opp_cards_5
    return f


def build_feature_frame(cache=None, verbose=True):
    """Parse sim_games.json into one feature row per game (+ style/deck labels)."""
    if cache and Path(cache).exists():
        return pd.read_pickle(cache)
    type_by_id, type_by_name = load_type_index()
    sim = json.loads(SIM_JSON.read_text(encoding="utf-8"))
    rows, untyped = [], Counter()
    for g in sim["games"]:
        f = extract_features(g, type_by_id, type_by_name)
        f["style"] = g["hero_playstyle"]
        f["deck"] = g["hero_deck"]
        f["opp_style"] = g["opponent_playstyle"]
        f["result"] = g["result"]
        rows.append(f)
        for h in g["card_history"]:
            if h["player"] == "me" and h["card"]["name"] != COIN_NAME:
                if not (type_by_id.get(h["card"]["id"]) or type_by_name.get(h["card"]["name"])):
                    untyped[h["card"]["name"]] += 1
    df = pd.DataFrame(rows)
    if verbose:
        print(f"Built {len(df)} game rows x {len(FEATURES)} features")
        print(f"Untyped card names (me-side): {len(untyped)} distinct, "
              f"{sum(untyped.values())} plays"
              + (f" e.g. {untyped.most_common(5)}" if untyped else ""))
    if cache:
        pd.to_pickle(df, cache)
    return df


# ---------------------------------------------------------------- analysis

def discriminativeness(df):
    """Per-feature Kruskal-Wallis + epsilon^2 + MI(style) + MI(deck)."""
    from scipy.stats import kruskal
    from sklearn.feature_selection import mutual_info_classif

    styles = sorted(df["style"].unique())

    # impute (median) for MI, which needs finite values
    X = df[FEATURES].apply(lambda c: c.fillna(c.median())).replace([np.inf, -np.inf], 0.0)
    X = X.fillna(0.0).to_numpy()
    mi_style = mutual_info_classif(X, df["style"].to_numpy(), discrete_features=False, random_state=0)
    mi_deck = mutual_info_classif(X, df["deck"].to_numpy(), discrete_features=False, random_state=0)

    # mutual_info_classif is in nats; MI ceilings differ by target (H(style)=ln5,
    # H(deck)=ln2), so raw MI is not comparable across targets. Normalize each by
    # its target entropy -> uncertainty coefficient U in [0,1] = "fraction of the
    # label's uncertainty this feature explains". This makes the style-vs-deck
    # (i.e. signal-vs-leakage) comparison fair.
    def entropy_nats(s):
        p = s.value_counts(normalize=True).to_numpy()
        return float(-(p * np.log(p)).sum())
    H_style, H_deck = entropy_nats(df["style"]), entropy_nats(df["deck"])

    recs = []
    for i, feat in enumerate(FEATURES):
        groups = [df.loc[df["style"] == s, feat].dropna().to_numpy() for s in styles]
        groups = [g for g in groups if len(g) > 0]
        H = p = eps2 = np.nan
        if len(groups) >= 2 and any(len(np.unique(g)) > 1 for g in groups):
            try:
                H, p = kruskal(*groups)
                n = sum(len(g) for g in groups)
                eps2 = (H - len(groups) + 1) / (n - len(groups)) if n > len(groups) else np.nan
            except ValueError:
                pass
        origin, ver, tgt = FEATURE_ORIGIN[feat]
        u_style, u_deck = mi_style[i] / H_style, mi_deck[i] / H_deck
        recs.append(dict(feature=feat, origin=origin, source=ver, target=tgt,
                         KW_H=H, p=p, eps2=eps2,
                         MI_style=mi_style[i], MI_deck=mi_deck[i],
                         U_style=u_style, U_deck=u_deck, U_gap=u_style - u_deck))
    rank = pd.DataFrame(recs).sort_values("U_style", ascending=False).reset_index(drop=True)
    return rank, styles


def classifier_check(df):
    """Within-deck 5-fold accuracy and cross-deck transfer, on card-seq features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    def Xy(d):
        X = d[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy()
        return X, d["style"].to_numpy()

    clf = lambda: make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1))
    out = {}
    Xa, ya = Xy(df)
    out["overall_5fold_acc"] = float(np.mean(cross_val_score(clf(), Xa, ya, cv=5, scoring="accuracy")))
    for deck in sorted(df["deck"].unique()):
        d = df[df["deck"] == deck]
        Xd, yd = Xy(d)
        out[f"within_{deck}_5fold_acc"] = float(np.mean(cross_val_score(clf(), Xd, yd, cv=5, scoring="accuracy")))
    decks = sorted(df["deck"].unique())
    if len(decks) == 2:
        for tr, te in [(decks[0], decks[1]), (decks[1], decks[0])]:
            Xtr, ytr = Xy(df[df["deck"] == tr])
            Xte, yte = Xy(df[df["deck"] == te])
            model = clf().fit(Xtr, ytr)
            out[f"transfer_{tr[:4]}->{te[:4]}_acc"] = float((model.predict(Xte) == yte).mean())
    return out


def make_figures(df, rank, styles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) normalized MI with style vs with deck, per feature
    r = rank.sort_values("U_style")
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(y - 0.2, r["U_style"], height=0.4, label="U with playstyle (signal)", color="#4C78A8")
    ax.barh(y + 0.2, r["U_deck"], height=0.4, label="U with deck (leakage)", color="#E45756")
    ax.set_yticks(y)
    ax.set_yticklabels(r["feature"], fontsize=8)
    ax.set_xlabel("uncertainty coefficient  U = I(feature; label) / H(label)")
    ax.set_title("Card-sequence features: play-style signal vs deck leakage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_mi_style_vs_deck.png", dpi=130)
    plt.close(fig)

    # (2) boxplots of the top-6 features by style
    top = rank.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.ravel(), top):
        data = [df.loc[df["style"] == s, feat].dropna() for s in styles]
        ax.boxplot(data, tick_labels=styles, showfliers=False)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.suptitle("Top-6 card-sequence features by play-style")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_top_features_by_style.png", dpi=130)
    plt.close(fig)
    print(f"Figures -> {OUT_DIR}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 60)

    df = build_feature_frame(cache=OUT_DIR / "cardseq_features.pkl")
    df.to_csv(OUT_DIR / "cardseq_features.csv", index=False)

    rank, styles = discriminativeness(df)
    rank.to_csv(OUT_DIR / "cardseq_ranking.csv", index=False)

    print("\n=== per-style game counts ===")
    print(df.groupby(["deck", "style"]).size().unstack("style"))

    print("\n=== feature discriminativeness (sorted by U_style = normalized MI with playstyle) ===")
    print("    U_style/U_deck = fraction of that label's uncertainty the feature explains (0..1);")
    print("    U_gap>0 => more a STYLE signal than a DECK signal.")
    show = rank.copy()
    for c in ["KW_H", "eps2", "U_style", "U_deck", "U_gap"]:
        show[c] = show[c].round(4)
    show["p"] = show["p"].apply(lambda v: "0" if pd.notna(v) and v < 1e-300 else f"{v:.1e}" if pd.notna(v) else "nan")
    print(show[["feature", "origin", "target", "eps2",
                "U_style", "U_deck", "U_gap"]].to_string(index=False))

    print("\n=== per-style means, top-10 features by U_style ===")
    top = rank.head(10)["feature"].tolist()
    print(df.groupby("style")[top].mean().round(3).T)

    print("\n=== classifier check (RandomForest on card-seq features only) ===")
    for k, v in classifier_check(df).items():
        print(f"  {k:32s} {v:.3f}")
    print("  (5 balanced classes -> chance = 0.200)")

    make_figures(df, rank, styles)
    print(f"\nWrote:\n  {OUT_DIR/'cardseq_features.csv'}\n  {OUT_DIR/'cardseq_ranking.csv'}")


if __name__ == "__main__":
    main()
