"""
Feature extraction for the 9-deck log_v2 matrix.

Builds two parallel feature blocks per game, for the focal player (p1) only:

  raw_*     V1-V4 style absolute metrics (face damage/turn, avg card cost, ...).
            These are what the earlier reports used. They measure what the DECK
            handed the player as much as what the player did, which is why they
            do not transfer across decks.

  ch_*      choice-relative metrics. Every one is scored against the set of
            options that were legally available AT THAT DECISION, so the deck's
            contribution is the denominator rather than part of the signal.
            A style is "what you picked out of what you were offered".

NOTE: decision records carry `search_score`, the agent's own internal evaluation
score. That is style-specific by construction, so using it would leak the label.
It is never read here.

Usage:  py -3 log_v2_analysis/v2_features.py [--limit N] [--jobs N]
Output: log_v2_analysis/out/features.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_DIR = os.path.join(ROOT, "log_v2")
OUT_DIR = os.path.join(HERE, "out")

# Deck archetype families. Leaving out a single deck is a weak domain shift when
# a sibling deck of the same family stays in training, so the headline test holds
# out a whole family. Grouped by construction and confirmed by the mana curve:
# the Reno decks are 30-unique singleton highlanders; the Midrange trio sits at
# 2.6-3.5 mean cost. MurlocDruid (2.80) is the judgement call -- it is a tribal
# aggro deck but curves closest to midrange. Edit here to regroup.
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

ATTACK_TYPES = ("MINION_ATTACK", "HERO_ATTACK")


def _is_face(opt) -> bool:
    """An attack option aimed at the enemy hero. Hero entities use HERO_* ids."""
    tgt = opt.get("tgt") or {}
    return str(tgt.get("card_id", "")).startswith("HERO_")


def _is_minion_play(opt) -> bool:
    """PLAY_CARD carries a board position only when the card is a minion."""
    return opt.get("t") == "PLAY_CARD" and "pos" in opt


def _midrank_pct(values, chosen):
    """Where `chosen` sits within `values`, as a 0..1 midrank percentile."""
    n = len(values)
    if n <= 1:
        return None
    less = sum(1 for v in values if v < chosen)
    eq = sum(1 for v in values if v == chosen)
    return (less + 0.5 * eq) / n


class _Acc:
    """Numerator/denominator accumulators -> rate, or None if never applicable."""

    def __init__(self):
        self.num = defaultdict(float)
        self.den = defaultdict(float)

    def hit(self, key, cond: bool):
        self.den[key] += 1
        if cond:
            self.num[key] += 1

    def add(self, key, value):
        if value is not None:
            self.num[key] += value
            self.den[key] += 1

    def rate(self, key):
        d = self.den.get(key, 0)
        return (self.num[key] / d) if d else None


def extract_game(path):
    """One game file -> one feature dict for the focal player (p1)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            objs = [json.loads(line) for line in fh if line.strip()]
    except Exception as exc:  # corrupt/partial file
        return {"_error": f"{os.path.basename(path)}: {exc}"}

    meta = next((o for o in objs if o.get("type") == "game_meta"), None)
    end = next((o for o in objs if o.get("type") == "game_end"), None)
    if meta is None:
        return {"_error": f"{os.path.basename(path)}: no game_meta"}

    focal = 1  # the folder's own agent is always p1
    me, opp = "p1", "p2"
    row = {
        "game_file": os.path.basename(path),
        "style": meta[me]["agent"],
        "deck": meta[me]["deck"],
        "deck_family": DECK_FAMILY.get(meta[me]["deck"], "?"),
        "hero_class": meta[me]["hero_class"],
        "opp_style": meta[opp]["agent"],
        "opp_deck": meta[opp]["deck"],
        "start_player": meta.get("start_player"),
    }
    if end is not None:
        row["n_turns_total"] = end.get("turns")
        row["won"] = 1 if end.get("winner") == "P1" else 0

    acc = _Acc()
    n_dec = 0
    my_turns = set()
    face_dmg = dmg_taken = heal_self = 0.0
    opp_minions_killed = my_minions_lost = 0.0
    n_plays = n_minion_plays = n_attacks = n_face_attacks = n_hero_power = 0
    costs_played = []
    board_sizes = []
    mana_avail_sum = mana_spent_sum = 0.0
    first_minion_turn = None
    deck_count_end = None

    for o in objs:
        if o.get("type") != "decision" or o.get("player") != focal:
            continue
        chosen = o.get("chosen") or {}
        opts = o.get("options") or []
        state = o.get("state") or {}
        turn = o.get("turn")
        my_turns.add(turn)
        n_dec += 1

        side = state.get(me) or {}
        oside = state.get(opp) or {}
        hand_cost = {c["eid"]: c.get("cost", 0) for c in (side.get("hand") or [])}
        board_sizes.append(len(side.get("board") or []))
        if side.get("deck_count") is not None:
            deck_count_end = side["deck_count"]
        mana_now = (side.get("mana") or {}).get("available", 0)

        # ---------- events: absolute (raw) tallies ----------
        for e in o.get("events") or []:
            kind, amt = e.get("e"), e.get("amount", 0)
            if kind == "damage" and e.get("target") == "hero":
                if e.get("player") == 2:
                    face_dmg += amt
                elif e.get("player") == 1:
                    dmg_taken += amt
            elif kind == "heal" and e.get("target") == "hero" and e.get("player") == 1:
                heal_self += amt
            elif kind == "minion_death":
                if e.get("player") == 2:
                    opp_minions_killed += 1
                elif e.get("player") == 1:
                    my_minions_lost += 1

        ct = chosen.get("t")
        if ct == "PLAY_CARD":
            n_plays += 1
            c = hand_cost.get((chosen.get("src") or {}).get("eid"))
            if c is not None:
                costs_played.append(c)
                mana_spent_sum += c
            if _is_minion_play(chosen):
                n_minion_plays += 1
                if first_minion_turn is None:
                    first_minion_turn = turn
        elif ct in ATTACK_TYPES:
            n_attacks += 1
            if _is_face(chosen):
                n_face_attacks += 1
        elif ct == "HERO_POWER":
            n_hero_power += 1
        elif ct == "END_TURN":
            mana_avail_sum += mana_now

        if o.get("kind") != "action":
            continue  # CHOOSE/discover handled separately (rare)

        # ---------- option sets ----------
        atk = [x for x in opts if x.get("t") in ATTACK_TYPES]
        face_atk = [x for x in atk if _is_face(x)]
        minion_atk = [x for x in atk if not _is_face(x)]
        plays = [x for x in opts if x.get("t") == "PLAY_CARD"]
        minion_plays = [x for x in plays if _is_minion_play(x)]
        other_plays = [x for x in plays if not _is_minion_play(x)]
        hp_opts = [x for x in opts if x.get("t") == "HERO_POWER"]
        hero_atk = [x for x in atk if x.get("t") == "HERO_ATTACK"]

        # ---------- choice-relative features ----------
        # Flagship: the face-vs-trade dilemma, counted only when it truly exists.
        if face_atk and minion_atk:
            acc.hit("ch_face_pref", ct in ATTACK_TYPES and _is_face(chosen))
        if atk:
            acc.hit("ch_attack_engage", ct in ATTACK_TYPES)
        if hp_opts:
            acc.hit("ch_heropower_pref", ct == "HERO_POWER")
        if plays:
            acc.hit("ch_play_pref", ct == "PLAY_CARD")
            acc.hit("ch_pass_with_play", ct == "END_TURN")
        if minion_plays and other_plays:
            acc.hit("ch_minion_play_pref", _is_minion_play(chosen))
        if hp_opts and plays:
            acc.hit("ch_hp_over_play", ct == "HERO_POWER")
        if hero_atk:
            hero_face = [x for x in hero_atk if _is_face(x)]
            hero_minion = [x for x in hero_atk if not _is_face(x)]
            if hero_face and hero_minion:
                acc.hit(
                    "ch_hero_attack_face_pref",
                    ct == "HERO_ATTACK" and _is_face(chosen),
                )

        # How expensive was the chosen play, relative to what was playable?
        if ct == "PLAY_CARD" and len(plays) > 1:
            cs = [hand_cost.get((x.get("src") or {}).get("eid")) for x in plays]
            cs = [c for c in cs if c is not None]
            mine = hand_cost.get((chosen.get("src") or {}).get("eid"))
            if mine is not None and len(cs) > 1:
                acc.add("ch_cost_pct", _midrank_pct(cs, mine))
                acc.hit("ch_max_cost_pref", mine == max(cs))
            if mine is not None and mana_now:
                acc.add("ch_mana_commit", min(mine / mana_now, 1.0))

        # Which minion did they attack, relative to the available targets?
        if ct in ATTACK_TYPES and not _is_face(chosen) and len(minion_atk) > 1:
            obm = {m["eid"]: m for m in (oside.get("board") or [])}
            atks, hps = [], []
            for x in minion_atk:
                m = obm.get((x.get("tgt") or {}).get("eid"))
                if m:
                    atks.append(m.get("atk", 0))
                    hps.append(m.get("hp", 0))
            tm = obm.get((chosen.get("tgt") or {}).get("eid"))
            if tm and len(atks) > 1:
                acc.add("ch_target_atk_pct", _midrank_pct(atks, tm.get("atk", 0)))
                acc.add("ch_target_hp_pct", _midrank_pct(hps, tm.get("hp", 0)))

        acc.add("ch_n_options", len(opts))  # diagnostic: deck-leaky by nature

    if n_dec == 0:
        return {"_error": f"{os.path.basename(path)}: no focal decisions"}

    nt = max(len(my_turns), 1)
    for key in (
        "ch_face_pref", "ch_attack_engage", "ch_heropower_pref", "ch_play_pref",
        "ch_pass_with_play", "ch_minion_play_pref", "ch_hp_over_play",
        "ch_hero_attack_face_pref", "ch_cost_pct", "ch_max_cost_pref",
        "ch_mana_commit", "ch_target_atk_pct", "ch_target_hp_pct", "ch_n_options",
    ):
        row[key] = acc.rate(key)
    # how often the dilemma arose at all -- a rate's own support
    row["ch_face_dilemma_rate"] = acc.den.get("ch_face_pref", 0) / n_dec

    row.update({
        "raw_n_turns": nt,
        "raw_n_decisions": n_dec,
        "raw_face_dmg_per_turn": face_dmg / nt,
        "raw_dmg_taken_per_turn": dmg_taken / nt,
        "raw_heal_per_turn": heal_self / nt,
        "raw_face_attack_ratio": (n_face_attacks / n_attacks) if n_attacks else None,
        "raw_attacks_per_turn": n_attacks / nt,
        "raw_cards_per_turn": n_plays / nt,
        "raw_avg_card_cost": (sum(costs_played) / len(costs_played)) if costs_played else None,
        "raw_max_card_cost": max(costs_played) if costs_played else None,
        "raw_minion_frac": (n_minion_plays / n_plays) if n_plays else None,
        "raw_mana_spent": mana_spent_sum,
        "raw_mana_spent_per_turn": mana_spent_sum / nt,
        "raw_mana_floated_per_turn": mana_avail_sum / nt,
        "raw_hero_power_per_turn": n_hero_power / nt,
        "raw_board_size_mean": (sum(board_sizes) / len(board_sizes)) if board_sizes else None,
        "raw_opp_minions_killed_per_turn": opp_minions_killed / nt,
        "raw_my_minions_lost_per_turn": my_minions_lost / nt,
        "raw_first_minion_turn": first_minion_turn,
        "raw_deck_count_end": deck_count_end,
    })
    return row


def _run_dir(d):
    return [extract_game(p) for p in sorted(glob(os.path.join(d, "*.jsonl")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="games per style x deck cell (0 = all)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    dirs = [d for d in sorted(glob(os.path.join(LOG_DIR, "*"))) if os.path.isdir(d)
            and "__pycache__" not in d]
    print(f"{len(dirs)} style x deck cells", flush=True)

    if args.limit:
        tasks = []
        for d in dirs:
            tasks += sorted(glob(os.path.join(d, "*.jsonl")))[: args.limit]
        print(f"parsing {len(tasks)} games (limit {args.limit}/cell)", flush=True)
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(extract_game, tasks, chunksize=32))
    else:
        rows = []
        with ProcessPoolExecutor(args.jobs) as ex:
            for i, batch in enumerate(ex.map(_run_dir, dirs), 1):
                rows += batch
                print(f"  [{i}/{len(dirs)}] {len(rows)} games", flush=True)

    errs = [r["_error"] for r in rows if "_error" in r]
    rows = [r for r in rows if "_error" not in r]
    df = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "features.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} games -> {out}")
    if errs:
        print(f"{len(errs)} files skipped; first few: {errs[:3]}")
    print(f"styles: {sorted(df['style'].unique())}")
    print(f"decks:  {len(df['deck'].unique())}  families: {sorted(df['deck_family'].unique())}")
    miss = df.isna().mean().sort_values(ascending=False)
    print("\nmissing-rate (top 6):")
    print(miss.head(6).to_string())


if __name__ == "__main__":
    main()
