"""
Stage 0 -- one source-agnostic loader for the behavioral-stylometry pipeline.

Both the SabberStone simulation and the real hearthstonemap dumps are re-emitted
as the SAME normalized game dict, so every downstream stylometry module is blind
to where a game came from:

    {
      "source": "sim" | "human",
      "user_hash", "hero", "hero_deck", "opponent", "opponent_deck",
      "coin": bool, "result",
      "card_history": [{"player":"me"|"opponent","turn":int,
                        "card":{"id","name","mana"}}, ...],
      "hero_playstyle", "opponent_playstyle",   # ground truth for sim, None for human
    }

Hard rule (engine / skill leak): the real dumps also carry duration, rank, legend,
added, region, mode, id -- NONE of these are ever exposed as a feature.  They are
dropped here at the door so no later module can accidentally use them.

Games with an empty card_history are dropped (a few % of human rows have none).

build_game_frame() turns either source into one feature row per game -- the 31
cardseq_metrics features + labels + the me-side played-card-id list (kept for the
archetype module) -- and caches it as a .pkl (fixed, reproducible).

Run directly to print per-user game counts and the >=2N eligibility table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardseq_metrics import FEATURES, extract_features, load_type_index

ROOT = Path(__file__).parent
HUMAN_DIR = ROOT / "dataset" / "hearthstonemap"
SIM_JSON = ROOT / "dataset" / "hearthstonemap_sim" / "sim_games.json"
OUT_DIR = ROOT / "stylometry_out"

COIN_NAME = "The Coin"

# Fields present in the real dumps that must never become features.
LEAK_FIELDS = ("duration", "rank", "legend", "added", "region", "mode", "id", "note")

# N values the Stage-1 protocol sweeps over (needs >= 2N games per unit).
N_SWEEP = (1, 3, 5, 10, 20, 30, 50)


# ---------------------------------------------------------------- normalization

def _clean_history(hist):
    """Keep only plays with a real card and a valid 1-indexed turn.

    Sim is cleanly 1-indexed; the real dumps carry a few `turn:0` artifacts
    (pre-game/logging noise).  Dropping them here keeps one turn convention
    across both sources so per-turn rates never divide by zero downstream.
    """
    out = []
    for h in hist:
        t = h.get("turn")
        if not isinstance(t, (int, float)) or t < 1:
            continue
        if not isinstance(h.get("card"), dict):
            continue
        out.append(h)
    return out


def normalize_game(raw, source):
    """Raw sim/human record -> common schema, or None if it has no card history."""
    hist = _clean_history(raw.get("card_history") or [])
    if not hist:
        return None
    return {
        "source": source,
        "user_hash": raw.get("user_hash"),
        "hero": raw.get("hero"),
        "hero_deck": raw.get("hero_deck"),
        "opponent": raw.get("opponent"),
        "opponent_deck": raw.get("opponent_deck"),
        "coin": bool(raw.get("coin")),
        "result": raw.get("result"),
        "card_history": hist,
        # ground truth only exists for the simulation
        "hero_playstyle": raw.get("hero_playstyle"),
        "opponent_playstyle": raw.get("opponent_playstyle"),
    }


def human_month_files(months=None):
    files = sorted(HUMAN_DIR.glob("20??-??.json"))
    if months:
        want = set(months)
        files = [f for f in files if f.stem in want]
    return files


def iter_sim_games():
    sim = json.loads(SIM_JSON.read_text(encoding="utf-8"))
    for raw in sim["games"]:
        g = normalize_game(raw, "sim")
        if g is not None:
            yield g


def iter_human_games(months=None):
    """Stream human games month-by-month so we never hold >1 dump in memory."""
    for f in human_month_files(months):
        data = json.loads(f.read_text(encoding="utf-8"))
        for raw in data.get("games", []):
            g = normalize_game(raw, "human")
            if g is not None:
                yield g


def iter_games(source, months=None):
    if source == "sim":
        return iter_sim_games()
    if source == "human":
        return iter_human_games(months)
    raise ValueError(f"unknown source {source!r}")


# ---------------------------------------------------------------- feature frame

def _me_card_ids(hist):
    """Me-side played card ids this game (Coin excluded), in play order, with dupes."""
    out = []
    for h in hist:
        if h.get("player") != "me":
            continue
        card = h.get("card", {})
        if card.get("name") == COIN_NAME:
            continue
        cid = card.get("id")
        if cid:
            out.append(cid)
    return out


def build_game_frame(source, months=None, cache=None, verbose=True):
    """One feature row per game for `source`, cached to .pkl.

    Columns: user_hash, hero, hero_deck, opponent, coin, result, source,
             style/deck ground truth (sim only), me_cards (tuple of ids),
             and the 31 cardseq_metrics features.
    """
    if cache and Path(cache).exists():
        df = pd.read_pickle(cache)
        if verbose:
            print(f"[{source}] loaded cached frame {cache}  ({len(df):,} games)")
        return df

    type_by_id, type_by_name = load_type_index()
    rows = []
    n_seen = n_kept = 0
    for g in iter_games(source, months):
        n_seen += 1
        f = extract_features(g, type_by_id, type_by_name)
        f["user_hash"] = g["user_hash"]
        f["hero"] = g["hero"]
        f["hero_deck"] = g["hero_deck"]
        f["opponent"] = g["opponent"]
        f["coin_flag"] = g["coin"]
        f["result"] = g["result"]
        f["source"] = source
        f["style"] = g["hero_playstyle"]      # None for human
        f["deck"] = g["hero_deck"]
        f["me_cards"] = tuple(_me_card_ids(g["card_history"]))
        # a game with no me-side non-coin play carries no behavioral signal
        if not f["me_cards"]:
            continue
        rows.append(f)
        n_kept += 1
        if verbose and n_seen % 50000 == 0:
            print(f"[{source}] {n_seen:,} games parsed ...")

    df = pd.DataFrame(rows)
    # compact + reproducible: features as float32
    for c in FEATURES:
        df[c] = df[c].astype("float32")
    if verbose:
        print(f"[{source}] built {n_kept:,} feature rows "
              f"(from {n_seen:,} games; {n_seen - n_kept:,} dropped: empty/coin-only)")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(df, cache)
        if verbose:
            print(f"[{source}] cached -> {cache}")
    return df


# ---------------------------------------------------------------- user counts

def per_user_counts(df):
    return df.groupby("user_hash").size().sort_values(ascending=False)


def eligibility_table(df, n_sweep=N_SWEEP):
    """How many users survive the >=2N games requirement for each N."""
    counts = per_user_counts(df)
    recs = []
    for n in n_sweep:
        elig = counts[counts >= 2 * n]
        recs.append(dict(N=n, min_games=2 * n, n_users=int(len(elig)),
                         total_games_in_pool=int(elig.sum())))
    return pd.DataFrame(recs)


def _report(df, source):
    counts = per_user_counts(df)
    print(f"\n=== [{source}] per-user game counts ===")
    print(f"users: {len(counts)}   games: {int(counts.sum()):,}   "
          f"median/user: {int(counts.median())}   "
          f"min: {int(counts.min())}   max: {int(counts.max())}")
    print("top-5 users:", list(counts.head(5).items()))
    print("bottom-5 users:", list(counts.tail(5).items()))
    print(f"\n--- [{source}] users eligible at each N (need >=2N games) ---")
    print(eligibility_table(df).to_string(index=False))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 40)

    sim = build_game_frame("sim", cache=OUT_DIR / "frame_sim.pkl")
    _report(sim, "sim")
    # sim: pseudo-users are (style x deck); confirm the 10-way split
    print("\n[sim] (deck, style) group sizes -- the 10 E1 pseudo-users:")
    print(sim.groupby(["deck", "style"]).size().unstack("style"))

    human = build_game_frame("human", cache=OUT_DIR / "frame_human.pkl")
    _report(human, "human")
    print("\n[human] hero-class distribution:")
    print(human.groupby("hero").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
