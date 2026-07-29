#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-trip check: simulate -> render -> parse with the REAL Hearthstone parser.

powerlog_render.validate_lines only checks a log against itself. This goes
further: it feeds each rendered game to hslog (HearthSim's parser, the one
behind HSReplay.net) and compares the state hslog reconstructs against the
SabberStone game object the log was rendered from. If a third-party parser
reads back the same winner, turn count, hero health and board as the engine
holds in memory, the encoding is sound.

Both boards are compared: they are public, so a P1-viewpoint log must still
carry the opponent's board exactly. P2's hand and deck are legitimately unknown
and are not compared.

    py -3 roundtrip_check.py                 # a few games across the matrix
    py -3 roundtrip_check.py --games 30

Needs: py -3 -m pip install hslog
"""
import io
import os
import sys
import random
import argparse
import collections

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_powerlog as sp                    # noqa: E402  (loads the .NET bridge)
import sabberstone_simulator as sim          # noqa: E402
import powerlog_render as plr                # noqa: E402


def require_hslog():
    """Fail with something actionable if HearthSim's parser is missing.

    This repo has both a global `py -3` and d:\\test\\.venv, so it is easy to
    install into the interpreter that is not running this code. Name the one
    that actually needs it rather than guessing.
    """
    try:
        import hslog          # noqa: F401
        import hearthstone    # noqa: F401
    except ImportError as exc:
        raise ImportError(
            '%s\n\nroundtrip_check needs HearthSim\'s parser. Install it into '
            'the interpreter running this code:\n    "%s" -m pip install hslog'
            % (exc, sys.executable)) from exc


def parse_log(text):
    """Run a rendered Power.log through hslog; return the exported game."""
    require_hslog()
    from hslog import LogParser
    from hslog.export import EntityTreeExporter
    parser = LogParser()
    parser.read(io.StringIO(text))
    parser.flush()
    if not parser.games:
        raise ValueError("hslog found no games in the log")
    return EntityTreeExporter(parser.games[0],
                              player_manager=parser.player_manager).export().game


def check_game(a1, d1, a2, d2, start_player, seed):
    """Simulate one game, render, parse it back, diff. -> (problems, info)."""
    require_hslog()
    from hearthstone.enums import GameTag, Zone, CardType

    game, lines, secs = sp.simulate_game_powerlog(a1, d1, a2, d2, start_player,
                                                  seed=seed)
    problems = list(plr.validate_lines(lines))
    winner, p1s, p2s, turns = sp.game_result(game)

    parsed = parse_log("\n".join(lines) + "\n")
    ents = list(parsed.entities)

    # compare by value: on py3.11+ str() of an IntEnum yields the number, not
    # the name, so substring checks on str(PlayState.WON) silently never match
    WON, LOST, TIED = 4, 5, 6
    playstate = {p.player_id: int(p.tags.get(GameTag.PLAYSTATE) or 0)
                 for p in parsed.players}
    got = ("P1" if playstate.get(1) == WON else
           "P2" if playstate.get(2) == WON else
           "DRAW" if playstate.get(1) == TIED else "?")
    if got != winner:
        problems.append("winner: engine=%s parsed=%s" % (winner, got))
    if parsed.tags.get(GameTag.TURN) != turns:
        problems.append("turns: engine=%d parsed=%s"
                        % (turns, parsed.tags.get(GameTag.TURN)))

    for p in parsed.players:
        side = game.Player1 if p.player_id == 1 else game.Player2

        heroes = [e for e in ents
                  if e.tags.get(GameTag.CARDTYPE) == CardType.HERO
                  and e.controller is p]
        if heroes:
            hp = 30 - int(heroes[0].tags.get(GameTag.DAMAGE) or 0)
            if hp != int(side.Hero.Health):
                problems.append("P%d hero hp: engine=%d parsed=%d"
                                % (p.player_id, int(side.Hero.Health), hp))
            armor = int(heroes[0].tags.get(GameTag.ARMOR) or 0)
            if armor != int(side.Hero.Armor):
                problems.append("P%d armor: engine=%d parsed=%d"
                                % (p.player_id, int(side.Hero.Armor), armor))

        # the board is public: a P1-view log must still get P2's board right
        board = sorted(e.card_id for e in ents
                       if e.controller is p
                       and e.tags.get(GameTag.ZONE) == Zone.PLAY
                       and e.tags.get(GameTag.CARDTYPE) == CardType.MINION)
        want = sorted(str(m.Card.Id) for m in side.BoardZone)
        if board != want:
            problems.append("P%d board: engine=%s parsed=%s" % (p.player_id,
                                                                want, board))
    return problems, {"winner": winner, "turns": turns, "lines": len(lines),
                      "secs": secs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--width", type=int, default=14)
    args = ap.parse_args()

    sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH = args.depth, args.width
    rng = random.Random(args.seed)
    agents = ["aggro", "control", "fatigue", "midrange", "ramp"]
    decks = list(sim.DECKS.keys())

    print("round-tripping %d simulated games through hslog ...\n" % args.games)
    counts = collections.Counter()
    clean = 0
    for i in range(args.games):
        a1, a2 = rng.choice(agents), rng.choice(agents)
        d1, d2 = rng.choice(decks), rng.choice(decks)
        sp_ = rng.choice([1, 2])
        seed = rng.getrandbits(40)
        try:
            problems, info = check_game(a1, d1, a2, d2, sp_, seed)
        except Exception as exc:                            # noqa: BLE001
            counts["EXCEPTION: %s" % type(exc).__name__] += 1
            print("  EXC  %-8s %-22s vs %-8s %-22s: %s"
                  % (a1, d1[:22], a2, d2[:22], repr(exc)[:80]))
            continue
        if problems:
            for p in problems:
                counts[p.split(":")[0]] += 1
            print("  DIFF %-8s %-22s vs %-8s %-22s" % (a1, d1[:22], a2, d2[:22]))
            for p in problems[:3]:
                print("         %s" % p[:110])
        else:
            clean += 1
            print("  OK   %-8s %-22s vs %-8s %-22s  %s in %d turns, %d lines"
                  % (a1, d1[:22], a2, d2[:22], info["winner"], info["turns"],
                     info["lines"]))

    print("\n%d/%d games round-trip clean" % (clean, args.games))
    if counts:
        print("mismatches by kind:")
        for k, v in counts.most_common():
            print("  %-40s %d" % (k, v))
    return 0 if clean == args.games else 1


if __name__ == "__main__":
    raise SystemExit(main())
