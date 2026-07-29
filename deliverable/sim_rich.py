#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rich per-decision JSONL logging for SabberStone simulations.

Plays a game with exactly the same agents / search / settings as
sabberstone_simulator.simulate_game, but instead of the verbose text log it
records one JSON object per decision:

  * legal_options : every PlayerTask available at that moment (Controller.Options())
  * chosen        : the task the agent actually executed (+ index into options)
  * state         : full both-sides snapshot BEFORE the action (board, hands,
                    heroes, weapons, secrets, mana/overload, deck counts, fatigue)
  * events        : attributed diff AFTER the action (damage, heals, deaths,
                    summons, draws, discards, armor, fatigue, secrets, weapons)

plus per-game records: game_meta (decklists, RNG seed, search settings),
mulligan (offered/kept/replaced per player), and game_end (result, win
condition, final state, card_history projection in hearthstonemap order).

Record types, one JSON object per line ("v": schema version 1):
  {"type":"game_meta", ...}   first line
  {"type":"mulligan", ...}    one per player
  {"type":"decision", ...}    one per executed task, both players
  {"type":"game_end", ...}    last line

Search time is logged (search_secs on the first decision of each computed
line) for debugging only -- do NOT feed it into behavioral metrics.

V2 fixes vs the old simulate_game loop:
  * Decks come from decks_v2.py (all Implemented cards restored; Zoo padded
    to 30), NOT from the trimmed Decks.cs lists.
  * Stale-line fix: the search plans a whole turn, but real RNG can diverge
    from the search's imagined outcomes. Instead of blindly processing the
    planned task (which crashed ~2.7% of games with Zoneposition /
    KeyNotFound errors), each step executes the MATCHED LIVE OPTION from
    Controller.Options(); if the planned task no longer matches any legal
    option, the rest of the line is dropped and the turn is re-searched.
"""
import time
import random as _random

import sabberstone_simulator as sim
import decks_v2

from System.Collections.Generic import List                        # noqa: E402
from SabberStoneCore.Config import GameConfig                       # noqa: E402
from SabberStoneCore.Model import Game, Cards                       # noqa: E402
from SabberStoneCore.Model.Entities import IPlayable                # noqa: E402
from SabberStoneCore.Enums import State                             # noqa: E402
from SabberStoneCore.Tasks.PlayerTasks import ChooseTask, PlayerTask  # noqa: E402
from SabberStoneBasicAI.Nodes import OptionNode                     # noqa: E402

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------
# safe attribute helpers (some flags don't exist on every entity type)
# ----------------------------------------------------------------------------
def _get(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _card_ref(entity):
    """Minimal identity for an entity: entity id + card id + name."""
    if entity is None:
        return None
    card = _get(entity, "Card")
    return {
        "eid": int(_get(entity, "Id", -1)),
        "card_id": str(_get(card, "Id", "?")) if card is not None else "?",
        "name": str(_get(card, "Name", "?")) if card is not None else "?",
    }


# ----------------------------------------------------------------------------
# state snapshot
# ----------------------------------------------------------------------------
def _snap_minion(m):
    d = _card_ref(m)
    d.update({
        "atk": int(_get(m, "AttackDamage", 0)),
        "hp": int(_get(m, "Health", 0)),
        "base_hp": int(_get(m, "BaseHealth", 0)),
        "pos": int(_get(m, "ZonePosition", -1)),
        "exhausted": bool(_get(m, "IsExhausted", False)),
        "attacks_this_turn": int(_get(m, "NumAttacksThisTurn", 0)),
    })
    flags = []
    for attr, tag in (("HasTaunt", "taunt"), ("HasDivineShield", "divine_shield"),
                      ("IsFrozen", "frozen"), ("HasCharge", "charge"),
                      ("HasWindfury", "windfury"), ("HasStealth", "stealth"),
                      ("Poisonous", "poisonous"), ("HasLifeSteal", "lifesteal"),
                      ("CantAttack", "cant_attack"), ("Untouchable", "untouchable")):
        if bool(_get(m, attr, False)):
            flags.append(tag)
    if flags:
        d["flags"] = flags
    return d


def _snap_hand_card(p):
    d = _card_ref(p)
    d["cost"] = int(_get(p, "Cost", 0))
    return d


def _snap_controller(c):
    hero = c.Hero
    weapon = _get(hero, "Weapon")
    power = _get(hero, "HeroPower")
    snap = {
        "hero": {
            "card_id": str(_get(_get(hero, "Card"), "Id", "?")),
            "hp": int(_get(hero, "Health", 0)),
            "armor": int(_get(hero, "Armor", 0)),
            "atk": int(_get(hero, "AttackDamage", 0)),
            "exhausted": bool(_get(hero, "IsExhausted", False)),
            "fatigue": int(_get(hero, "Fatigue", 0)),
        },
        "weapon": None if weapon is None else {
            "card_id": str(_get(_get(weapon, "Card"), "Id", "?")),
            "atk": int(_get(weapon, "AttackDamage", 0)),
            "durability": int(_get(weapon, "Durability", 0)),
        },
        "hero_power": None if power is None else {
            "card_id": str(_get(_get(power, "Card"), "Id", "?")),
            "cost": int(_get(power, "Cost", 0)),
            "exhausted": bool(_get(power, "IsExhausted", False)),
        },
        "mana": {
            "total": int(_get(c, "BaseMana", 0)),
            "available": int(_get(c, "RemainingMana", 0)),
            "overload_owed": int(_get(c, "OverloadOwed", 0)),
            "overload_locked": int(_get(c, "OverloadLocked", 0)),
        },
        "hand": [_snap_hand_card(p) for p in c.HandZone],
        "board": [_snap_minion(m) for m in c.BoardZone],
        "secrets": [_card_ref(s) for s in c.SecretZone],
        "deck_count": int(c.DeckZone.Count),
        "graveyard_eids": [int(_get(e, "Id", -1)) for e in c.GraveyardZone],
    }
    return snap


def snapshot(game):
    return {
        "turn": int(game.Turn),
        "current_player": 1 if game.CurrentPlayer.Id == game.Player1.Id else 2,
        "p1": _snap_controller(game.Player1),
        "p2": _snap_controller(game.Player2),
    }


# ----------------------------------------------------------------------------
# task serialization
# ----------------------------------------------------------------------------
def serialize_task(task, game):
    t = str(task.PlayerTaskType)
    d = {"t": t}
    if isinstance(task, ChooseTask):
        picks = []
        for cid in task.Choices:
            try:
                ent = game.IdEntityDic[cid]
            except Exception:
                ent = None   # stale task referencing a search-imagined entity
            picks.append(_card_ref(ent) if ent is not None
                         else {"eid": int(cid), "card_id": "?", "name": "?"})
        d["picks"] = picks
        return d
    src = _get(task, "Source")
    tgt = _get(task, "Target")
    if src is not None:
        d["src"] = _card_ref(src)
    if tgt is not None:
        d["tgt"] = _card_ref(tgt)
    opt = int(_get(task, "ChooseOne", 0) or 0)
    if opt > 0:
        d["opt"] = opt
    pos = _get(task, "ZonePosition", None)
    if pos is not None and int(pos) >= 0 and t == "PLAY_CARD":
        d["pos"] = int(pos)
    return d


def _task_sig(d):
    """Identity for matching chosen against options (position-insensitive)."""
    if d.get("t") == "CHOOSE":
        return ("CHOOSE", tuple(sorted(p["eid"] for p in d.get("picks", []))))
    src = d.get("src") or {}
    tgt = d.get("tgt") or {}
    return (d.get("t"), src.get("eid"), tgt.get("eid"), d.get("opt", 0))


# ----------------------------------------------------------------------------
# event derivation: diff two snapshots, attribute to the acting task
# ----------------------------------------------------------------------------
def diff_events(pre, post, actor, exclude_eid=None):
    """List of event dicts caused by processing one task."""
    ev = []
    for pn in (1, 2):
        key = "p%d" % pn
        a, b = pre[key], post[key]

        # hero damage / heal / armor / fatigue
        dhp = b["hero"]["hp"] - a["hero"]["hp"]
        if dhp < 0:
            ev.append({"e": "damage", "target": "hero", "player": pn, "amount": -dhp})
        elif dhp > 0:
            ev.append({"e": "heal", "target": "hero", "player": pn, "amount": dhp})
        darm = b["hero"]["armor"] - a["hero"]["armor"]
        if darm > 0:
            ev.append({"e": "armor_gain", "player": pn, "amount": darm})
        elif darm < 0:
            ev.append({"e": "armor_lost", "player": pn, "amount": -darm})
        dfat = b["hero"]["fatigue"] - a["hero"]["fatigue"]
        if dfat > 0:
            ev.append({"e": "fatigue", "player": pn, "fatigue_now": b["hero"]["fatigue"]})

        # minions: damage / heal / buffs on survivors, deaths, summons
        pre_board = {m["eid"]: m for m in a["board"]}
        post_board = {m["eid"]: m for m in b["board"]}
        post_hand_eids = {c["eid"] for c in b["hand"]}
        post_grave = set(b["graveyard_eids"])
        pre_grave = set(a["graveyard_eids"])
        pre_hand = {c["eid"]: c for c in a["hand"]}

        for eid, m0 in pre_board.items():
            m1 = post_board.get(eid)
            if m1 is None:
                if eid in post_grave and eid not in pre_grave:
                    reason = "died"
                elif eid in post_hand_eids:
                    reason = "bounced"
                else:
                    reason = "left_board"
                ev.append({"e": "minion_" + ("death" if reason == "died" else reason),
                           "player": pn, "card_id": m0["card_id"], "eid": eid})
            else:
                dm = m1["hp"] - m0["hp"]
                if dm < 0:
                    ev.append({"e": "damage", "target": "minion", "player": pn,
                               "card_id": m1["card_id"], "eid": eid, "amount": -dm})
                elif dm > 0 and m1["base_hp"] == m0["base_hp"]:
                    ev.append({"e": "heal", "target": "minion", "player": pn,
                               "card_id": m1["card_id"], "eid": eid, "amount": dm})
                da = m1["atk"] - m0["atk"]
                dbh = m1["base_hp"] - m0["base_hp"]
                if da > 0 or dbh > 0:
                    ev.append({"e": "buff", "player": pn, "card_id": m1["card_id"],
                               "eid": eid, "atk": da, "hp": dbh})
        for eid, m1 in post_board.items():
            if eid not in pre_board:
                ev.append({"e": "summon", "player": pn, "card_id": m1["card_id"],
                           "eid": eid, "atk": m1["atk"], "hp": m1["hp"]})

        # hand: draws / generated cards / discards
        pre_hand_eids = set(pre_hand)
        gained = [c for c in b["hand"] if c["eid"] not in pre_hand_eids]
        n_drawn = max(a["deck_count"] - b["deck_count"], 0)
        for i, c in enumerate(gained):
            cause = "draw" if i < n_drawn else "generated"
            ev.append({"e": "card_gained", "player": pn, "cause": cause,
                       "card_id": c["card_id"], "eid": c["eid"]})
        lost = [pre_hand[eid] for eid in pre_hand_eids - post_hand_eids]
        for c in lost:
            if c["eid"] == exclude_eid:
                continue  # the card the task itself played
            if eid_in_board_or_secret(c["eid"], b):
                continue  # entered play some other way (e.g. summoned from hand)
            if c["eid"] in post_grave and c["eid"] not in pre_grave:
                # a hand card that hit the graveyard without being the played
                # card = discard (Warlock) or destroyed-in-hand
                ev.append({"e": "discard", "player": pn,
                           "card_id": c["card_id"], "eid": c["eid"]})
            else:
                ev.append({"e": "left_hand", "player": pn,
                           "card_id": c["card_id"], "eid": c["eid"]})

        # secrets set / revealed
        pre_sec = {s["eid"] for s in a["secrets"]}
        post_sec = {s["eid"] for s in b["secrets"]}
        for s in b["secrets"]:
            if s["eid"] not in pre_sec:
                ev.append({"e": "secret_set", "player": pn, "card_id": s["card_id"]})
        for s in a["secrets"]:
            if s["eid"] not in post_sec:
                ev.append({"e": "secret_revealed", "player": pn, "card_id": s["card_id"]})

        # weapon equipped / lost / used
        w0, w1 = a["weapon"], b["weapon"]
        if w0 is None and w1 is not None:
            ev.append({"e": "weapon_equipped", "player": pn, "card_id": w1["card_id"]})
        elif w0 is not None and w1 is None:
            ev.append({"e": "weapon_destroyed", "player": pn, "card_id": w0["card_id"]})
        elif w0 is not None and w1 is not None and w1["durability"] < w0["durability"]:
            ev.append({"e": "weapon_durability", "player": pn,
                       "card_id": w1["card_id"],
                       "amount": w0["durability"] - w1["durability"]})

    if ev:
        for e in ev:
            e["by"] = actor
    return ev


def eid_in_board_or_secret(eid, side_snap):
    return any(m["eid"] == eid for m in side_snap["board"]) or \
           any(s["eid"] == eid for s in side_snap["secrets"])


# ----------------------------------------------------------------------------
# mulligan (same rule invocation as sabberstone_simulator, but logged)
# ----------------------------------------------------------------------------
def _mulligan_logged(game, player, scorer, pnum):
    offered = []
    for cid in player.Choice.Choices:
        offered.append(_card_ref(game.IdEntityDic[cid]))
    playables = List[IPlayable]()
    for cid in player.Choice.Choices:
        playables.Add(game.IdEntityDic[cid])
    scorer.Controller = player
    kept_ids = scorer.MulliganRule().Invoke(playables)   # .NET List<int>
    kept_set = set(int(k) for k in kept_ids)
    rec = {
        "v": SCHEMA_VERSION, "type": "mulligan", "player": pnum,
        "offered": offered,
        "kept": [o for o in offered if o["eid"] in kept_set],
        "replaced": [o for o in offered if o["eid"] not in kept_set],
    }
    return kept_ids, rec


# ----------------------------------------------------------------------------
# main entry: play one game, return (game, records, seconds)
# ----------------------------------------------------------------------------
def _decklist(deck_name):
    out = []
    for name in decks_v2.DECK_LISTS[deck_name]:
        c = Cards.FromName(name)
        out.append({"card_id": str(c.Id), "name": str(c.Name), "cost": int(c.Cost)})
    return out


def simulate_game_rich(p1_agent, p1_deck, p2_agent, p2_deck,
                       start_player, seed=None, game_index=None):
    """Play one game with full per-decision JSONL records.

    p*_agent : agent name string ("aggro", ...) or Score class
    p*_deck  : deck name string from sim.DECKS
    Returns (game, records, elapsed_seconds); records is a list of dicts.
    """
    import datetime

    a1_cls = sim._resolve_agent(p1_agent) if isinstance(p1_agent, str) else p1_agent
    a2_cls = sim._resolve_agent(p2_agent) if isinstance(p2_agent, str) else p2_agent
    d1_name, _, d1_class = sim._resolve_deck(p1_deck)
    d2_name, _, d2_class = sim._resolve_deck(p2_deck)

    if seed is None:
        seed = _random.getrandbits(48)

    cfg = GameConfig()
    cfg.StartPlayer = start_player
    cfg.Player1Name = "P1"
    cfg.Player1HeroClass = d1_class
    cfg.Player1Deck = decks_v2.build_deck(d1_name)
    cfg.Player2Name = "P2"
    cfg.Player2HeroClass = d2_class
    cfg.Player2Deck = decks_v2.build_deck(d2_name)
    cfg.FillDecks = False
    cfg.Shuffle = sim.SHUFFLE
    cfg.SkipMulligan = False
    cfg.Logging = False          # JSONL replaces the verbose text log
    cfg.History = False
    cfg.RandomSeed = seed        # reproducible engine RNG

    records = []
    t0 = time.perf_counter()
    game = Game(cfg)
    game.StartGame()

    meta = {
        "v": SCHEMA_VERSION, "type": "game_meta",
        "game_index": game_index,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "start_player": start_player,
        "p1": {"agent": str(getattr(p1_agent, "__name__", p1_agent)),
               "deck": d1_name, "hero_class": str(d1_class),
               "decklist": _decklist(d1_name)},
        "p2": {"agent": str(getattr(p2_agent, "__name__", p2_agent)),
               "deck": d2_name, "hero_class": str(d2_class),
               "decklist": _decklist(d2_name)},
        "search": {"depth": sim.SEARCH_MAX_DEPTH, "width": sim.SEARCH_MAX_WIDTH},
        "shuffle": bool(sim.SHUFFLE),
    }
    records.append(meta)

    scorers = {game.Player1.Id: a1_cls(), game.Player2.Id: a2_cls()}

    kept1, rec1 = _mulligan_logged(game, game.Player1, scorers[game.Player1.Id], 1)
    kept2, rec2 = _mulligan_logged(game, game.Player2, scorers[game.Player2.Id], 2)
    records.append(rec1)
    records.append(rec2)
    game.Process(ChooseTask.Mulligan(game.Player1, kept1))
    game.Process(ChooseTask.Mulligan(game.Player2, kept2))
    game.MainReady()

    card_history = []
    decision_index = 0
    safety = 0
    while game.State != State.COMPLETE and safety < 100000:
        cur = game.CurrentPlayer
        scorer = scorers[cur.Id]
        ts = time.perf_counter()
        solutions = OptionNode.GetSolutions(
            game, cur.Id, scorer, sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH)
        search_secs = time.perf_counter() - ts
        if solutions.Count == 0:
            break
        best = max(solutions, key=lambda n: n.Score)
        line = best.PlayerTasks(List[PlayerTask]())
        processed_any = False

        for line_pos, task in enumerate(line):
            if game.State == State.COMPLETE:
                break
            actor = 1 if game.CurrentPlayer.Id == game.Player1.Id else 2
            raw_options = list(game.CurrentPlayer.Options())
            options = [serialize_task(o, game) for o in raw_options]
            planned = serialize_task(task, game)

            # exact match first (keeps the planned board position); fall back
            # to a position-insensitive match if RNG shifted the board
            chosen_index = next(
                (i for i, o in enumerate(options) if o == planned), -1)
            if chosen_index < 0:
                sig = _task_sig(planned)
                chosen_index = next(
                    (i for i, o in enumerate(options) if _task_sig(o) == sig), -1)

            if chosen_index < 0:
                break   # line went stale mid-turn -> re-search the turn

            # always execute the live option, never the search's task object
            exec_task = raw_options[chosen_index]
            chosen = options[chosen_index]

            pre = snapshot(game)
            rec = {
                "v": SCHEMA_VERSION, "type": "decision",
                "di": decision_index, "player": actor, "turn": pre["turn"],
                "kind": "choice" if chosen.get("t") == "CHOOSE" else "action",
                "n_options": len(options),
                "options": options,
                "chosen": chosen, "chosen_index": chosen_index,
                "state": pre,
            }
            if line_pos == 0:
                rec["search_secs"] = round(search_secs, 4)
                rec["search_score"] = int(best.Score)

            game.Process(exec_task)
            processed_any = True

            src_eid = (chosen.get("src") or {}).get("eid")
            rec["events"] = diff_events(pre, snapshot(game), actor, src_eid)
            records.append(rec)
            decision_index += 1

            if chosen.get("t") == "PLAY_CARD" and chosen.get("src"):
                card_history.append(
                    [pre["turn"], actor, chosen["src"]["card_id"]])

            if game.State != State.COMPLETE and game.CurrentPlayer.Choice is not None:
                break   # a mid-line choice opened up -> recompute the turn

        if not processed_any:
            break   # no legal progress possible (a fresh line's first task
                    # always matches Options(), so this should never fire)
        safety += 1

    secs = time.perf_counter() - t0
    winner = sim._winner_label(game)
    final = snapshot(game)
    loser_key = "p2" if winner == "P1" else "p1"
    loser = final[loser_key]["hero"]
    if str(game.Player1.PlayState) == "CONCEDED" or str(game.Player2.PlayState) == "CONCEDED":
        win_condition = "concede"
    elif winner == "DRAW":
        win_condition = "draw"
    elif loser["fatigue"] > 0 and final[loser_key]["deck_count"] == 0:
        win_condition = "fatigue"
    else:
        win_condition = "health"

    records.append({
        "v": SCHEMA_VERSION, "type": "game_end",
        "winner": winner,
        "p1_state": str(game.Player1.PlayState),
        "p2_state": str(game.Player2.PlayState),
        "turns": int(game.Turn),
        "win_condition": win_condition,
        "n_decisions": decision_index,
        "seconds": round(secs, 3),
        "final_state": final,
        "card_history": card_history,
    })
    return game, records, secs
