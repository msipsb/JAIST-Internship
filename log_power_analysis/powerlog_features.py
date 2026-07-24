"""
Simple statistical play-style features from the `log_power` Power.log dataset.

`log_power/` holds each game in the **official Hearthstone `Power.log` format**
(one `CREATE_GAME` followed by timestamped `FULL_ENTITY` / `SHOW_ENTITY` /
`TAG_CHANGE` / `BLOCK_START` / `BLOCK_END` packets), rendered from SabberStone's
own PowerHistory. Unlike `log_v2/`'s JSONL there are **no decision records** in
the file: the legal-option sets the agent chose from are not in a real client
log, so the choice-relative (`ch_*`) block of `log_v2_analysis/v2_features.py`
cannot be rebuilt here. What a real log does give is the resulting *game state
stream*, so every feature below is a **simple statistic** -- a count, a rate, a
mean or a max over one game -- for the focal player **P1** only.

    log_power/<style>_<deck>/game_NNN_*.log     the packets  (features)
    log_power/<style>_<deck>/summary.csv        style/deck   (labels)

Labels are read only from `summary.csv` (the `.log` never names the agent), so
features and labels stay separate exactly as in `sim_to_hearthstonemap.py`.

Viewpoint caveat, straight from `log_power/README.md`: the log is **P1's
client**, so P1's own deck is a stack of anonymous entities until drawn and P2's
hand is hidden. Everything here is therefore computed from what P1 could
actually see -- which is the point of using this format.

Usage:  py -3 log_power_analysis/powerlog_features.py [--limit N] [--jobs N]
Output: log_power_analysis/out/features.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_DIR = os.path.join(ROOT, "log_power")
OUT_DIR = os.path.join(HERE, "out")

# Same grouping as log_v2_analysis/v2_features.py -- LOAO holds out a whole
# family so no sibling deck of the held-out deck stays in training.
DECK_FAMILY = {
    "AggroPirateWarrior": "aggro",
    "ZooDiscardWarlock": "aggro",
    "MurlocDruid": "aggro",
    "MidrangeSecretHunter": "midrange",
    "MidrangeJadeShaman": "midrange",
    "MidrangeBuffPaladin": "midrange",
    "RenoKazakusMage": "highlander_control",
    "RenoKazakusDragonPriest": "highlander_control",
    "MiraclePirateRogue": "combo_tempo",
}

PREFIX = "GameState.DebugPrintPower() - "

# An entity descriptor: [entityName=... id=N zone=Z zonePos=P cardId=C player=N].
# entityName itself can contain brackets ("UNKNOWN ENTITY [cardType=INVALID]"),
# so the pattern anchors on the ` id=.. .. player=N]` tail rather than on `]`.
ENT_RE = re.compile(
    r"\[entityName=(?P<name>.*?) id=(?P<eid>\d+) zone=(?P<zone>\S+) "
    r"zonePos=(?P<pos>-?\d+) cardId=(?P<cid>\S*) player=(?P<pl>\d+)\]"
)
FULL_RE = re.compile(r"^FULL_ENTITY - Creating ID=(\d+) CardID=(\S*)")
SHOW_RE = re.compile(r"^SHOW_ENTITY - Updating Entity=")
BLOCK_RE = re.compile(r"^BLOCK_START BlockType=(\w+) ")

# tags worth keeping on an entity record (everything else is ignored, which is
# most of the file -- a FULL_ENTITY dumps ~30 tags)
ENT_TAGS = {"CARDTYPE", "COST", "ATK", "HEALTH", "CONTROLLER", "ZONE"}
INT_TAGS = {"COST", "ATK", "HEALTH", "CONTROLLER"}

CARD_TYPES = {"MINION", "SPELL", "WEAPON"}


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def _std(xs):
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _ratio(a, b):
    return (a / b) if b else None


class Game:
    """Replays one Power.log and accumulates P1's per-game statistics."""

    def __init__(self):
        self.ent = {}            # eid -> {"cid","type","cost","atk","hp","ctrl","zone"}
        self.cur_ent = None      # entity that bare `tag=` lines belong to
        self.hero = {1: None, 2: None}
        self.dmg = {}            # eid -> last seen DAMAGE value (for deltas)
        self.armor = {}          # eid -> last seen ARMOR value
        self.block_owner = []    # stack of controlling players, one per open block
        self.cur_player = None
        self.turn = 0

        # ---- P1 tallies -------------------------------------------------
        self.p1_turns = 0
        self.plays = 0           # cards played from hand
        self.minion_plays = self.spell_plays = self.weapon_plays = 0
        self.costs = []
        self.minion_atks = []
        self.minion_hps = []
        self.targeted_plays = self.target_face_plays = 0
        self.attacks = self.face_attacks = 0
        self.hero_powers = 0
        self.face_dmg = self.dmg_taken = self.minion_dmg = 0.0
        self.heal_self = self.armor_gain = 0.0
        self.opp_minions_killed = self.own_minions_lost = 0
        self.first_minion_turn = None
        self.cards_drawn = 0
        self.mana_spent = self.mana_avail = 0.0
        self.resources = 0       # P1 RESOURCES (crystals) this turn
        self.temp_res = 0        # P1 TEMP_RESOURCES granted this turn (Coin, Innervate)
        self.temp_res_now = 0    # ... and how much of it is still unspent
        self.res_used_max = 0    # running RESOURCES_USED within the turn
        self.board_sizes = []
        self.hand_sizes = []

    # ---------------------------------------------------------------- state
    def _rec(self, eid):
        r = self.ent.get(eid)
        if r is None:
            r = self.ent[eid] = {"cid": "", "type": "", "cost": None, "atk": None,
                                 "hp": None, "ctrl": None, "zone": ""}
        return r

    def _counts(self):
        board = hand = deck = 0
        for r in self.ent.values():
            if r["ctrl"] != 1:
                continue
            z = r["zone"]
            if z == "PLAY" and r["type"] == "MINION":
                board += 1
            elif z == "HAND":
                hand += 1
            elif z == "DECK":
                deck += 1
        return board, hand, deck

    def _end_p1_turn(self):
        """Snapshot the board/hand and bank the turn's mana usage."""
        board, hand, _ = self._counts()
        self.board_sizes.append(board)
        self.hand_sizes.append(hand)
        self.mana_avail += self.resources + self.temp_res
        self.mana_spent += self.res_used_max + (self.temp_res - self.temp_res_now)
        self.res_used_max = self.temp_res = self.temp_res_now = 0

    def _set_current_player(self, pl):
        if self.cur_player == 1 and pl != 1:
            self._end_p1_turn()
        if pl == 1 and self.cur_player != 1:
            self.p1_turns += 1
        self.cur_player = pl

    # ---------------------------------------------------------------- lines
    def feed(self, c):
        if c.startswith("tag="):                       # bare tag under an entity
            if self.cur_ent is None:
                return
            tag, _, val = c[4:].partition(" value=")
            if tag in ENT_TAGS:
                self._apply_tag(self.cur_ent, tag, val)
            return

        if c.startswith("TAG_CHANGE Entity="):
            i = c.rfind(" tag=")
            if i < 0:
                return
            tag, _, val = c[i + 5:].partition(" value=")
            head = c[18:i]
            if head == "GameEntity":
                if tag == "TURN":
                    self.turn = int(val)
                return
            m = ENT_RE.search(c, 18, i + 1)
            if m is None:
                return
            eid = int(m.group("eid"))
            r = self._rec(eid)
            if r["ctrl"] is None:
                r["ctrl"] = int(m.group("pl"))
            if not r["zone"]:
                r["zone"] = m.group("zone")
            if not r["cid"] and m.group("cid"):
                r["cid"] = m.group("cid")
            self._tag_change(eid, r, tag, val)
            return

        if c.startswith("BLOCK_START"):
            self._block_start(c)
            return
        if c.startswith("BLOCK_END"):
            if self.block_owner:
                self.block_owner.pop()
            return

        m = FULL_RE.match(c)
        if m:
            eid = int(m.group(1))
            r = self._rec(eid)
            if m.group(2):
                r["cid"] = m.group(2)
            self.cur_ent = eid
            return
        if SHOW_RE.match(c):
            m = ENT_RE.search(c)
            if m:
                eid = int(m.group("eid"))
                r = self._rec(eid)
                r["ctrl"] = int(m.group("pl"))
                r["zone"] = m.group("zone")
                j = c.rfind("CardID=")
                if j >= 0:
                    r["cid"] = c[j + 7:].strip()
                self.cur_ent = eid
            return
        if c.startswith("Player EntityID="):
            self.cur_ent = int(c.split("EntityID=", 1)[1].split(" ", 1)[0])
            return
        if c.startswith("GameEntity EntityID="):
            self.cur_ent = None
            return

    def _apply_tag(self, eid, tag, val):
        r = self._rec(eid)
        if tag == "CARDTYPE":
            r["type"] = val
            if val == "HERO" and r["ctrl"] in (1, 2) and self.hero[r["ctrl"]] is None:
                self.hero[r["ctrl"]] = eid
        elif tag == "ZONE":
            r["zone"] = val
        elif tag in INT_TAGS:
            try:
                v = int(val)
            except ValueError:
                return
            if tag == "COST":
                r["cost"] = v
            elif tag == "ATK":
                r["atk"] = v
            elif tag == "HEALTH":
                r["hp"] = v
            else:
                r["ctrl"] = v
                if r["type"] == "HERO" and self.hero.get(v) is None:
                    self.hero[v] = eid

    def _tag_change(self, eid, r, tag, val):
        # --- turn / player bookkeeping ---------------------------------
        if tag == "CURRENT_PLAYER":
            if val == "1":
                self._set_current_player(r["ctrl"])
            return
        if tag == "HERO_ENTITY":
            self.hero[r["ctrl"]] = int(val)
            return
        if tag == "CARDTYPE":
            r["type"] = val
            if val == "HERO" and r["ctrl"] in (1, 2) and self.hero[r["ctrl"]] is None:
                self.hero[r["ctrl"]] = eid
            return

        # --- zone moves: draws, deaths, board size ----------------------
        if tag == "ZONE":
            prev, r["zone"] = r["zone"], val
            # the mulligan also moves cards DECK->HAND->DECK->HAND; only count
            # draws once the first real turn has started
            if r["ctrl"] == 1 and prev == "DECK" and val == "HAND" and self.p1_turns:
                self.cards_drawn += 1
            if val == "GRAVEYARD" and prev == "PLAY" and r["type"] == "MINION":
                if r["ctrl"] == 1:
                    self.own_minions_lost += 1
                elif r["ctrl"] == 2:
                    self.opp_minions_killed += 1
            return

        # --- mana -------------------------------------------------------
        # Each turn starts `RESOURCES=N` / `RESOURCES_USED=N` / `RESOURCES_USED=0`:
        # the engine restores then clears the pool, so the middle value is a
        # phantom. Resetting the running max on every 0 drops it.
        if r["ctrl"] == 1:
            if tag == "RESOURCES":
                self.resources = int(val)
                return
            if tag == "TEMP_RESOURCES":
                self.temp_res_now = int(val)
                self.temp_res = max(self.temp_res, self.temp_res_now)
                return
            if tag == "RESOURCES_USED":
                v = int(val)
                self.res_used_max = 0 if v == 0 else max(self.res_used_max, v)
                return

        # --- health / armour deltas -------------------------------------
        if tag == "DAMAGE":
            v = int(val)
            prev = self.dmg.get(eid, 0)
            self.dmg[eid] = v
            d = v - prev
            if d == 0:
                return
            owner = self.block_owner[0] if self.block_owner else None
            if r["type"] == "HERO":
                if r["ctrl"] == 2 and d > 0:
                    if owner != 2:
                        self.face_dmg += d
                elif r["ctrl"] == 1:
                    if d > 0:
                        self.dmg_taken += d
                    else:
                        self.heal_self += -d
            elif r["type"] == "MINION" and r["ctrl"] == 2 and d > 0 and owner == 1:
                self.minion_dmg += d
            return
        if tag == "ARMOR" and r["type"] == "HERO" and r["ctrl"] == 1:
            v = int(val)
            prev = self.armor.get(eid, 0)
            self.armor[eid] = v
            if v > prev:
                self.armor_gain += v - prev
            return

    def _block_start(self, c):
        m = BLOCK_RE.match(c)
        bt = m.group(1) if m else ""
        head, _, tail = c.partition(" Target=")
        em = ENT_RE.search(head)
        owner = int(em.group("pl")) if em else None
        self.block_owner.append(owner)
        if owner != 1 or em is None or bt not in ("PLAY", "ATTACK"):
            return

        eid = int(em.group("eid"))
        r = self._rec(eid)
        if not r["cid"] and em.group("cid"):
            r["cid"] = em.group("cid")
        tm = ENT_RE.search(tail) if tail and tail[0] == "[" else None

        if bt == "ATTACK":
            self.attacks += 1
            if tm is not None and self._is_hero(tm):
                self.face_attacks += 1
            return

        # BlockType=PLAY -- a hero power is "played" from PLAY zone, a card from HAND
        if r["type"] == "HERO_POWER" or (em.group("zone") == "PLAY" and r["type"] != "MINION"
                                         and r["type"] != "SPELL" and r["type"] != "WEAPON"):
            self.hero_powers += 1
            return

        self.plays += 1
        if r["cost"] is not None:
            self.costs.append(r["cost"])
        t = r["type"]
        if t == "MINION":
            self.minion_plays += 1
            if r["atk"] is not None:
                self.minion_atks.append(r["atk"])
            if r["hp"] is not None:
                self.minion_hps.append(r["hp"])
            if self.first_minion_turn is None:
                self.first_minion_turn = self.p1_turns
        elif t == "SPELL":
            self.spell_plays += 1
        elif t == "WEAPON":
            self.weapon_plays += 1
        if tm is not None:
            self.targeted_plays += 1
            if self._is_hero(tm) and tm.group("pl") == "2":
                self.target_face_plays += 1

    def _is_hero(self, m):
        r = self.ent.get(int(m.group("eid")))
        if r is not None and r["type"]:
            return r["type"] == "HERO"
        return m.group("cid").startswith("HERO_")

    # -------------------------------------------------------------- output
    def features(self):
        if self.cur_player == 1:
            self._end_p1_turn()
        nt = max(self.p1_turns, 1)
        acts = self.plays + self.attacks + self.hero_powers
        _, _, deck_left = self._counts()
        total_dmg = self.face_dmg + self.minion_dmg
        return {
            # volume / tempo
            "f_n_turns": self.p1_turns,
            "f_actions_per_turn": acts / nt,
            "f_cards_per_turn": self.plays / nt,
            "f_cards_drawn_per_turn": self.cards_drawn / nt,
            # curve
            "f_avg_card_cost": _mean(self.costs),
            "f_max_card_cost": max(self.costs) if self.costs else None,
            "f_std_card_cost": _std(self.costs),
            # card-type mix
            "f_minion_frac": _ratio(self.minion_plays, self.plays),
            "f_spell_frac": _ratio(self.spell_plays, self.plays),
            "f_weapon_frac": _ratio(self.weapon_plays, self.plays),
            "f_avg_minion_atk": _mean(self.minion_atks),
            "f_avg_minion_health": _mean(self.minion_hps),
            "f_first_minion_turn": self.first_minion_turn,
            # aggression
            "f_attacks_per_turn": self.attacks / nt,
            "f_face_attack_ratio": _ratio(self.face_attacks, self.attacks),
            "f_hero_power_per_turn": self.hero_powers / nt,
            "f_hero_power_frac": _ratio(self.hero_powers, acts),
            "f_targeted_play_frac": _ratio(self.targeted_plays, self.plays),
            "f_target_face_frac": _ratio(self.target_face_plays, self.targeted_plays),
            # life totals
            "f_face_dmg_per_turn": self.face_dmg / nt,
            "f_minion_dmg_per_turn": self.minion_dmg / nt,
            "f_face_dmg_share": _ratio(self.face_dmg, total_dmg),
            "f_dmg_taken_per_turn": self.dmg_taken / nt,
            "f_heal_per_turn": self.heal_self / nt,
            "f_armor_gain_per_turn": self.armor_gain / nt,
            # board
            "f_board_size_mean": _mean(self.board_sizes),
            "f_board_size_max": max(self.board_sizes) if self.board_sizes else None,
            "f_opp_minions_killed_per_turn": self.opp_minions_killed / nt,
            "f_own_minions_lost_per_turn": self.own_minions_lost / nt,
            "f_trade_ratio": _ratio(self.opp_minions_killed,
                                    self.opp_minions_killed + self.own_minions_lost),
            # resources
            "f_mana_spent_per_turn": self.mana_spent / nt,
            "f_mana_floated_per_turn": max(self.mana_avail - self.mana_spent, 0.0) / nt,
            "f_mana_efficiency": _ratio(self.mana_spent, self.mana_avail),
            "f_hand_size_mean": _mean(self.hand_sizes),
            "f_deck_left_end": deck_left,
        }


FEATURE_COLS = list(Game().features().keys())


def extract_game(path):
    g = Game()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                i = line.find(PREFIX)
                if i < 0:
                    continue
                c = line[i + len(PREFIX):].strip()
                if c:
                    g.feed(c)
    except Exception as exc:
        return {"_error": f"{os.path.basename(path)}: {type(exc).__name__}: {exc}"}
    row = {"game_file": os.path.basename(path)}
    row.update(g.features())
    return row


def _labels(cell_dir):
    """log_file -> label dict, from summary.csv (the .log never names the agent)."""
    out = {}
    p = os.path.join(cell_dir, "summary.csv")
    if not os.path.isfile(p):
        return out
    with open(p, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("winner") == "ERROR" or not r.get("log_file"):
                continue
            out[r["log_file"]] = {
                "style": r["p1_agent"],
                "deck": r["p1_deck"],
                "deck_family": DECK_FAMILY.get(r["p1_deck"], "?"),
                "hero_class": r["p1_class"],
                "opp_style": r["p2_agent"],
                "opp_deck": r["p2_deck"],
                "start_player": int(r["start_player"]) if r["start_player"] else None,
                "won": 1 if r["p1_state"] == "WON" else 0,
                "n_turns_total": int(r["turns"]) if r["turns"] else None,
            }
    return out


def run_cell(args):
    cell_dir, limit = args
    lab = _labels(cell_dir)
    files = sorted(glob(os.path.join(cell_dir, "*.log")))
    if limit:
        files = files[:limit]
    rows, errs = [], []
    for p in files:
        name = os.path.basename(p)
        meta = lab.get(name)
        if meta is None:
            continue                       # ERROR game or missing summary row
        r = extract_game(p)
        if "_error" in r:
            errs.append(r["_error"])
            continue
        r.update(meta)
        rows.append(r)
    return rows, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="games per style x deck cell (0 = all 900)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "features.csv"))
    args = ap.parse_args()

    cells = [d for d in sorted(glob(os.path.join(LOG_DIR, "*"))) if os.path.isdir(d)
             and os.path.isfile(os.path.join(d, "summary.csv"))]
    print(f"{len(cells)} style x deck cells, {args.jobs} workers"
          f"{f', limit {args.limit}/cell' if args.limit else ''}", flush=True)

    rows, errs = [], []
    with ProcessPoolExecutor(args.jobs) as ex:
        for i, (rs, es) in enumerate(ex.map(run_cell, [(c, args.limit) for c in cells]), 1):
            rows += rs
            errs += es
            print(f"  [{i}/{len(cells)}] {os.path.basename(cells[i-1])}: "
                  f"{len(rs)} games ({len(rows)} total)", flush=True)

    df = pd.DataFrame(rows)
    meta_cols = ["game_file", "style", "deck", "deck_family", "hero_class",
                 "opp_style", "opp_deck", "start_player", "won", "n_turns_total"]
    df = df[[c for c in meta_cols if c in df.columns] + FEATURE_COLS]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n{len(df)} games x {len(FEATURE_COLS)} features -> {args.out}")
    if errs:
        print(f"{len(errs)} files failed; first: {errs[:3]}")
    print(f"styles: {sorted(df['style'].unique())}")
    print(f"decks:  {len(df['deck'].unique())}   families: {sorted(df['deck_family'].unique())}")
    print(f"P1 win rate: {df['won'].mean():.1%}")
    miss = df[FEATURE_COLS].isna().mean().sort_values(ascending=False)
    print("\nmissing-rate (top 6):")
    print(miss.head(6).to_string())


if __name__ == "__main__":
    sys.exit(main())
