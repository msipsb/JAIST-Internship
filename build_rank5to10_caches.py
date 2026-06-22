# Single-pass parser for the rank 5-10 study.
# Scans all 16 monthly dumps once, identifies (a) the most-played CLASS and
# (b) the most-played (class, archetype) DECK among me-side ranked games at
# rank in {5,6,7,8,9,10}, and writes two notebook-format cache pickles plus a
# small selection.json. Per-game feature logic mirrors the warrior notebook's
# build_frames() exactly so the notebooks can load these caches directly.
import json, glob, os, pickle, collections
import numpy as np
import pandas as pd

DATA_DIR  = r"D:\test\dataset\hearthstonemap"
FILE_GLOB = os.path.join(DATA_DIR, "201[0-9]-[0-9][0-9].json")
COIN_ID   = "GAME_005"
MODE      = "ranked"
RANK_SET  = (5, 6, 7, 8, 9, 10)
MANA_CAP  = 10

def mana_available(n_turns, n_coin):
    full = min(n_turns, MANA_CAP)
    base = full * (full + 1) // 2 + max(0, n_turns - MANA_CAP) * MANA_CAP
    return base + n_coin

def feature_row(g):
    """Replicate build_frames() per-game logic; return (game_dict, manas, per_turn, n_turns)."""
    gid = g["id"]; uh = g["user_hash"]
    ch  = g.get("card_history") or []
    me  = [e for e in ch if e["player"] == "me"]
    noncoin = [e for e in me if e["card"].get("id") != COIN_ID]
    coin    = [e for e in me if e["card"].get("id") == COIN_ID]
    manas   = [e["card"]["mana"] for e in noncoin if e["card"].get("mana") is not None]
    n_turns = max((e["turn"] for e in me), default=0)
    rounds  = max((e["turn"] for e in ch), default=0)
    per_turn = collections.Counter()
    for e in noncoin:
        if e["card"].get("mana") is not None:
            per_turn[e["turn"]] += e["card"]["mana"]
    spent = float(sum(manas)); avail = mana_available(n_turns, len(coin))
    row = dict(
        user_hash=uh, game_id=gid, rank=g["rank"], win=(g.get("result") == "win"),
        has_coin=bool(g.get("coin")), hero_deck=g.get("hero_deck") or "Unknown",
        opp=g.get("opponent") or "Unknown", duration=g.get("duration"), rounds=rounds or np.nan,
        n_cards=(len(noncoin) if ch else np.nan), n_me_turns=n_turns,
        mana_spent=spent, mana_available=avail,
        mana_eff=(spent / avail if (ch and n_turns and avail) else np.nan),
        cards_per_turn=(len(noncoin) / n_turns if (ch and n_turns) else np.nan),
        mana_per_turn=(spent / n_turns if (ch and n_turns) else np.nan),
        time_per_turn=(g["duration"] / rounds if (g.get("duration") and rounds) else np.nan),
        first_turn=min((e["turn"] for e in noncoin), default=np.nan),
        coin_turn=(coin[0]["turn"] if coin else np.nan),
    )
    return row, manas, per_turn, n_turns

# master accumulators (all classes, rank 5-10)
game_rows, card_rows, turn_rows = [], [], []
class_counts = collections.Counter()
deck_counts  = collections.Counter()   # keyed by (hero, hero_deck) with real archetype

for f in sorted(glob.glob(FILE_GLOB)):
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    kept = 0
    for g in data["games"]:
        if g.get("mode") != MODE or g.get("rank") not in RANK_SET:
            continue
        hero = g.get("hero") or "Unknown"
        raw_deck = g.get("hero_deck")
        class_counts[hero] += 1
        if raw_deck:                      # only labeled archetypes count toward "deck"
            deck_counts[(hero, raw_deck)] += 1
        row, manas, per_turn, n_turns = feature_row(g)
        row["hero"] = hero
        game_rows.append(row)
        for m in manas:
            card_rows.append((row["user_hash"], m, hero, row["hero_deck"]))
        for t in range(1, n_turns + 1):
            turn_rows.append((row["user_hash"], row["game_id"], t, per_turn.get(t, 0), hero, row["hero_deck"]))
        kept += 1
    print(f"  parsed {os.path.basename(f)}  (+{kept} rank-5-10 ranked games)")

games_all = pd.DataFrame(game_rows)
cards_all = pd.DataFrame(card_rows, columns=["user_hash", "mana", "hero", "hero_deck"])
turns_all = pd.DataFrame(turn_rows, columns=["user_hash", "game_id", "turn", "mana_spent", "hero", "hero_deck"])

print("\n=== Most-played CLASS at rank 5-10 (me-side ranked games) ===")
for h, c in class_counts.most_common():
    print(f"  {h:10s} {c:7d}")
TOP_CLASS = class_counts.most_common(1)[0][0]

print("\n=== Most-played DECK (class + archetype) at rank 5-10 ===")
for (h, d), c in deck_counts.most_common(12):
    print(f"  {h:10s} {d:12s} {c:7d}")
(TOP_DECK_CLASS, TOP_DECK_ARCH) = deck_counts.most_common(1)[0][0]

print(f"\nSelected CLASS file -> {TOP_CLASS}")
print(f"Selected DECK  file -> {TOP_DECK_CLASS} / {TOP_DECK_ARCH}")

GAME_COLS = ["user_hash", "game_id", "rank", "win", "has_coin", "hero_deck", "opp",
             "duration", "rounds", "n_cards", "n_me_turns", "mana_spent", "mana_available",
             "mana_eff", "cards_per_turn", "mana_per_turn", "time_per_turn", "first_turn", "coin_turn"]

def write_cache(path, gmask, cmask, tmask):
    g = games_all[gmask][GAME_COLS].reset_index(drop=True)
    c = cards_all[cmask][["user_hash", "mana"]].reset_index(drop=True)
    t = turns_all[tmask][["user_hash", "game_id", "turn", "mana_spent"]].reset_index(drop=True)
    with open(path, "wb") as fh:
        pickle.dump({"games": g, "cards": c, "turns": t}, fh)
    print(f"  wrote {os.path.basename(path)}: games {g.shape}, cards {c.shape}, turns {t.shape}, "
          f"players {g.user_hash.nunique()}")
    return g.shape, g.user_hash.nunique()

def slug(s):
    return str(s).replace("'", "").replace(" ", "").replace("/", "")

class_cache = os.path.join(r"D:\test", f"{slug(TOP_CLASS).lower()}_dataframes_rank5to10.pkl")
deck_cache  = os.path.join(r"D:\test", f"{slug(TOP_DECK_CLASS).lower()}_{slug(TOP_DECK_ARCH).lower()}_dataframes_rank5to10.pkl")

print("\n=== Writing caches ===")
cls_shape, cls_players = write_cache(
    class_cache,
    games_all.hero == TOP_CLASS,
    cards_all.hero == TOP_CLASS,
    turns_all.hero == TOP_CLASS)
deck_shape, deck_players = write_cache(
    deck_cache,
    (games_all.hero == TOP_DECK_CLASS) & (games_all.hero_deck == TOP_DECK_ARCH),
    (cards_all.hero == TOP_DECK_CLASS) & (cards_all.hero_deck == TOP_DECK_ARCH),
    (turns_all.hero == TOP_DECK_CLASS) & (turns_all.hero_deck == TOP_DECK_ARCH))

selection = dict(
    rank_set=list(RANK_SET),
    top_class=TOP_CLASS, top_class_games=int(class_counts[TOP_CLASS]),
    top_class_cache=class_cache, top_class_players=int(cls_players),
    top_deck_class=TOP_DECK_CLASS, top_deck_arch=TOP_DECK_ARCH,
    top_deck_games=int(deck_counts[(TOP_DECK_CLASS, TOP_DECK_ARCH)]),
    top_deck_cache=deck_cache, top_deck_players=int(deck_players),
    class_counts=dict(class_counts.most_common()),
    deck_counts={f"{h}|{d}": c for (h, d), c in deck_counts.most_common(20)},
)
with open(r"D:\test\rank5to10_selection.json", "w") as fh:
    json.dump(selection, fh, indent=2)
print("\nwrote rank5to10_selection.json")
