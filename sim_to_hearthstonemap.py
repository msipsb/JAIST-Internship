"""Convert SabberStone verbose game logs into hearthstonemap-style records.

Reads the verbose engine logs under ./log/<playstyle>_<deck>/game_*.log and
re-emits each game with the *feature set* of the hearthstonemap 2016-2017
monthly dumps (dataset/hearthstonemap/20??-??.json) -- i.e. compact game
metadata plus a per-turn `card_history` sequence of played cards -- rather than
the full verbose engine trace.

Design decisions (confirmed with the user):
  * hero  = the deck's class; hero_deck = the full sim deck name
            (e.g. WARRIOR / "AggroPirateWarrior").
  * The AI playstyle (aggro/control/fatigue/midrange/ramp) has no field in the
    real schema, so it is kept as an explicit ground-truth label column /
    extra JSON key for evaluation -- never used to build the features.
  * card_history entries carry real HS {id, name, mana}, resolved from
    hearthstonemap-master/map/cards_meta.json.
  * One record per game, from P1's perspective (P1 = "me" = the user);
    user_hash = stable hash of P1's (playstyle + deck).

Outputs (dataset/hearthstonemap_sim/):
  * sim_games.json  -- monthly-dump schema {range_start,...,games:[...]},
                       loadable by the existing extract_and_visualize.py, plus
                       hero_playstyle / opponent_playstyle keys per game.
  * sim_games.csv   -- one row per game, columns mirroring extracted/games_all.csv
                       (cards_me / cards_opp turn-segmented) + label columns.
  * unmapped_cards.txt -- any card names that did not resolve to an id (for audit).
"""

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "log"
CARDS_META = ROOT / "hearthstonemap-master" / "hearthstonemap-master" / "map" / "cards_meta.json"
CARDDEFS_XML = ROOT / "SabberStone" / "SabberStoneCore" / "resources" / "Data" / "CardDefs.xml"
OUT_DIR = ROOT / "dataset" / "hearthstonemap_sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- card database

def load_card_index():
    """name -> {"id":..., "mana":...}, preferring collectible entries on collision.

    Primary source is the 2016-17 hearthstonemap cards_meta.json. Names it does
    not cover (newer sets the sim uses) are filled from SabberStone's own
    CardDefs.xml -- the engine that actually generated the logs -- so every
    played card resolves to a real id + mana.
    """
    meta = json.loads(CARDS_META.read_text(encoding="utf-8"))
    by_name = {}
    for c in meta:
        nm = c.get("name")
        if not nm:
            continue
        prev = by_name.get(nm)
        if prev is None or (c.get("collectible") and not prev.get("collectible")):
            by_name[nm] = c
    index = {nm: {"id": c.get("id"), "mana": c.get("cost")} for nm, c in by_name.items()}

    # ---- gap-fill from CardDefs.xml (enumID 185=CARDNAME, 48=COST, 321=COLLECTIBLE) ----
    defs = {}   # name -> (cardid, cost, collectible)
    for _ev, el in ET.iterparse(str(CARDDEFS_XML), events=("end",)):
        if el.tag != "Entity":
            continue
        cardid = el.get("CardID")
        name = cost = None
        collectible = False
        for tag in el.findall("Tag"):
            eid = tag.get("enumID")
            if eid == "185":
                en = tag.find("enUS")
                if en is not None:
                    name = en.text
            elif eid == "48":
                cost = tag.get("value")
            elif eid == "321":
                collectible = tag.get("value") == "1"
        if name:
            prev = defs.get(name)
            if prev is None or (collectible and not prev[2]):
                defs[name] = (cardid, int(cost) if cost is not None else 0, collectible)
        el.clear()

    filled = 0
    for name, (cardid, cost, _coll) in defs.items():
        if name not in index:      # only fill genuine gaps; leave existing ids stable
            index[name] = {"id": cardid, "mana": cost}
            filled += 1
    print(f"Card index: {len(index)} names ({filled} gap-filled from CardDefs.xml)")
    return index

CARD_INDEX = load_card_index()

# ---------------------------------------------------------------- log parsing

HEADER_RE = re.compile(
    r"^(P\d) agent/deck\s*:\s*(\S+)\s*/\s*(\S+)\s*\((\w+)\)")
START_RE = re.compile(r"^start_player\s*:\s*(P\d)")
TS_RE = re.compile(r"^timestamp\s*:\s*(\S+)")
RESULT_RE = re.compile(r"^result\s*:\s*winner=(P\d)\b.*turns=(\d+)")
TURN_RE = re.compile(r"'Game\[1\]' set data TURN to (\d+)")
PLAY_RE = re.compile(r"PlayCardTask => \[(P\d)\] play '([^\[]+)\[(\d+)\]'\((\w+)\)")

TITLE = {"WARRIOR": "Warrior", "MAGE": "Mage", "HUNTER": "Hunter", "PALADIN": "Paladin",
         "PRIEST": "Priest", "ROGUE": "Rogue", "SHAMAN": "Shaman", "DRUID": "Druid",
         "WARLOCK": "Warlock", "NEUTRAL": "Neutral"}


def parse_log(path):
    """Parse one SabberStone log into a dict, or return None if it is unusable."""
    players = {}          # "P1"/"P2" -> {"playstyle","deck","hero"}
    start_player = None
    timestamp = None
    winner = None
    turns = None
    cur_turn = 0
    plays = []            # (round, "P1"/"P2", card_name)

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line and line[0] == "P" and " agent/deck" in line:
                m = HEADER_RE.match(line)
                if m:
                    pid, style, deck, hero = m.groups()
                    players[pid] = {"playstyle": style, "deck": deck,
                                    "hero": TITLE.get(hero, hero.title())}
                    continue
            if start_player is None and line.startswith("start_player"):
                m = START_RE.match(line)
                if m:
                    start_player = m.group(1)
                    continue
            if timestamp is None and line.startswith("timestamp"):
                m = TS_RE.match(line)
                if m:
                    timestamp = m.group(1)
                    continue
            if winner is None and line.startswith("result"):
                m = RESULT_RE.match(line)
                if m:
                    winner, turns = m.group(1), int(m.group(2))
                    continue
            m = TURN_RE.search(line)
            if m:
                cur_turn = int(m.group(1))
                continue
            m = PLAY_RE.search(line)
            if m:
                pid, name = m.group(1), m.group(2)
                rnd = (cur_turn + 1) // 2      # shared round number, hearthstonemap style
                plays.append((rnd, pid, name))

    if "P1" not in players or "P2" not in players or winner is None:
        return None
    return {"players": players, "start_player": start_player, "timestamp": timestamp,
            "winner": winner, "turns": turns, "plays": plays}


def build_record(parsed, game_id):
    """Turn a parsed log into a hearthstonemap-style game record (P1 = me)."""
    p1, p2 = parsed["players"]["P1"], parsed["players"]["P2"]

    card_history = []
    for rnd, pid, name in parsed["plays"]:
        card = CARD_INDEX.get(name, {"id": None, "mana": None})
        card_history.append({
            "player": "me" if pid == "P1" else "opponent",
            "turn": rnd,
            "card": {"id": card["id"], "name": name, "mana": card["mana"]},
        })

    user_hash = hashlib.md5(
        f"{p1['playstyle']}|{p1['deck']}".encode()).hexdigest().upper()

    # Columns the simulation cannot produce (duration, rank, legend) or that are
    # constant placeholders (region, mode) or bookkeeping (added timestamp) are
    # omitted entirely rather than emitted as null.
    return {
        "user_hash": user_hash,
        "id": game_id,
        "hero": p1["hero"],
        "hero_deck": p1["deck"],
        "opponent": p2["hero"],
        "opponent_deck": p2["deck"],
        "coin": parsed["start_player"] == "P2",   # P1 has the coin iff P1 goes second
        "result": "win" if parsed["winner"] == "P1" else "loss",
        "card_history": card_history,
        # ---- ground-truth labels (not part of the real schema) ----
        "hero_playstyle": p1["playstyle"],
        "opponent_playstyle": p2["playstyle"],
    }


# ---------------------------------------------------------------- CSV helpers

def turn_segmented(history, player):
    """'T1: Coin, War Axe | T2: ...' for one player -- matches extract_and_visualize."""
    plays = (h for h in history if h.get("player") == player)
    return " | ".join(
        f"T{turn}: " + ", ".join(p["card"]["name"] for p in group)
        for turn, group in groupby(plays, key=lambda h: h.get("turn"))
    )

CSV_COLUMNS = ["user_hash", "id", "hero", "hero_deck", "opponent", "opponent_deck",
               "coin", "result", "n_card_plays", "cards_me", "cards_opp",
               "hero_playstyle", "opponent_playstyle", "source_file"]


def main():
    log_files = sorted(LOG_DIR.glob("*/game_*.log"))
    print(f"Found {len(log_files)} game logs under {LOG_DIR}")

    records = []
    skipped = []
    unmapped = Counter()
    combos = Counter()

    csv_path = OUT_DIR / "sim_games.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for i, path in enumerate(log_files, 1):
            parsed = parse_log(path)
            if parsed is None:
                skipped.append(str(path.relative_to(ROOT)))
                continue
            rec = build_record(parsed, game_id=i)
            records.append(rec)

            for h in rec["card_history"]:
                if h["card"]["id"] is None:
                    unmapped[h["card"]["name"]] += 1
            combos[(rec["hero_playstyle"], rec["hero_deck"])] += 1

            hist = rec["card_history"]
            writer.writerow([
                rec["user_hash"], rec["id"], rec["hero"], rec["hero_deck"],
                rec["opponent"], rec["opponent_deck"], rec["coin"], rec["result"],
                len(hist), turn_segmented(hist, "me"), turn_segmented(hist, "opponent"),
                rec["hero_playstyle"], rec["opponent_playstyle"],
                str(path.relative_to(ROOT)).replace("\\", "/"),
            ])

    dump = {
        "unique_users": len({r["user_hash"] for r in records}),
        "total_games": len(records),
        "source": "SabberStone simulation (sim_to_hearthstonemap.py)",
        "games": records,
    }
    json_path = OUT_DIR / "sim_games.json"
    json_path.write_text(json.dumps(dump), encoding="utf-8")

    (OUT_DIR / "unmapped_cards.txt").write_text(
        "\n".join(f"{n}\t{c}" for n, c in unmapped.most_common()), encoding="utf-8")

    # ---- report ----
    print(f"\nConverted : {len(records)} games")
    print(f"Skipped   : {len(skipped)} (missing header/result -- likely crashed sims)")
    print(f"Users     : {dump['unique_users']}  ({json_path.name})")
    print(f"Unmapped card names: {len(unmapped)} distinct "
          f"({sum(unmapped.values())} plays)")
    print(f"\nGames per (playstyle, deck) for P1 = user:")
    for (style, deck), n in sorted(combos.items()):
        print(f"  {style:9s} {deck:22s} {n}")
    print(f"\nWrote:\n  {csv_path}\n  {json_path}")
    if skipped:
        print(f"  (first few skipped: {skipped[:3]})")


if __name__ == "__main__":
    main()
