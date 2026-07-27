#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play one SabberStone game with PowerHistory on and render it as a Power.log.

This is the V2 pipeline (sim_rich.py) with a different output: instead of
per-decision JSONL it turns GameConfig.History on and lets the ENGINE record
what happened, then renders that history into the Hearthstone client's text
format (powerlog_render.py). Ordering and block nesting are the engine's own,
not a reconstruction.

Same agents, same search settings, same decks (decks_v2), and the same
stale-line fix as sim_rich: the search plans a whole turn, but real RNG can
diverge from what the search imagined, so each step executes the MATCHED LIVE
OPTION from Controller.Options(); if the planned task no longer matches a legal
option the rest of the line is dropped and the turn is re-searched.

History costs ~16% runtime and does not bias play: it only appends to
PowerHistory inside tag setters, never touches the RNG, and Game.Clone()
defaults to history=false so the AI search is unaffected. Enabling it does
shift the RNG stream (a game with History=True follows a different line than
the same seed with it off), but over 40 games per arm mean turns matched at
14.68 vs 14.57 (Mann-Whitney p=0.96) -- an RNG offset, not biased play.
Because games are re-simulated they do NOT correspond to the log_v2 games.

    import sim_powerlog as sp
    game, lines, secs = sp.simulate_game_powerlog("aggro", "AggroPirateWarrior",
                                                  "control", "MurlocDruid", 1)
"""
import os
import sys
import time
import random as _random
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
for _p in (PARENT_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sabberstone_simulator as sim          # noqa: E402  (loads the .NET bridge)
import sim_rich                              # noqa: E402  (mulligan + task serialiser)
import decks_v2                              # noqa: E402
import powerlog_render as plr                # noqa: E402

from System.Collections.Generic import List                        # noqa: E402
from SabberStoneCore.Config import GameConfig                       # noqa: E402
from SabberStoneCore.Model import Game                              # noqa: E402
from SabberStoneCore.Enums import State                             # noqa: E402
from SabberStoneCore.Tasks.PlayerTasks import ChooseTask, PlayerTask  # noqa: E402
from SabberStoneBasicAI.Nodes import OptionNode                     # noqa: E402


def simulate_game_powerlog(p1_agent, p1_deck, p2_agent, p2_deck, start_player,
                           seed=None, game_index=None, viewpoint=plr.VIEWPOINT):
    """Play one game and render it. Returns (game, lines, elapsed_seconds)."""
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
    cfg.Logging = False          # the PowerHistory replaces the verbose text log
    cfg.History = True           # <-- the engine records the real power history
    cfg.RandomSeed = seed        # reproducible engine RNG

    t0 = time.perf_counter()
    game = Game(cfg)
    game.StartGame()

    scorers = {game.Player1.Id: a1_cls(), game.Player2.Id: a2_cls()}
    kept1, _ = sim_rich._mulligan_logged(game, game.Player1,
                                         scorers[game.Player1.Id], 1)
    kept2, _ = sim_rich._mulligan_logged(game, game.Player2,
                                         scorers[game.Player2.Id], 2)
    game.Process(ChooseTask.Mulligan(game.Player1, kept1))
    game.Process(ChooseTask.Mulligan(game.Player2, kept2))
    game.MainReady()

    safety = 0
    while game.State != State.COMPLETE and safety < 100000:
        safety += 1
        cur = game.CurrentPlayer
        solutions = OptionNode.GetSolutions(
            game, cur.Id, scorers[cur.Id],
            sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH)
        if solutions.Count == 0:
            break
        best = max(solutions, key=lambda n: n.Score)

        for task in best.PlayerTasks(List[PlayerTask]()):
            if game.State == State.COMPLETE:
                break
            raw_options = list(game.CurrentPlayer.Options())
            planned = sim_rich.serialize_task(task, game)

            # exact match first (keeps the planned board position); fall back to
            # a position-insensitive match if RNG shifted the board
            idx = next((i for i, o in enumerate(raw_options)
                        if sim_rich.serialize_task(o, game) == planned), -1)
            if idx < 0:
                sig = sim_rich._task_sig(planned)
                idx = next((i for i, o in enumerate(raw_options)
                            if sim_rich._task_sig(
                                sim_rich.serialize_task(o, game)) == sig), -1)
            if idx < 0:
                break        # line went stale mid-turn -> re-search the turn
            game.Process(raw_options[idx])

    secs = time.perf_counter() - t0
    entries = plr.history_to_entries(game.PowerHistory)
    start_iso = datetime.datetime.now().isoformat(timespec="seconds")
    lines = plr.render_game(entries, seed=seed, start_iso=start_iso,
                            viewpoint=viewpoint)
    return game, lines, secs


def game_result(game):
    """(winner, p1_state, p2_state, turns) from a finished game."""
    p1 = str(game.Player1.PlayState)
    p2 = str(game.Player2.PlayState)
    winner = "P1" if p1 == "WON" else "P2" if p2 == "WON" else "DRAW"
    return winner, p1, p2, int(game.Turn)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1-agent", default="aggro")
    ap.add_argument("--p1-deck", default="AggroPirateWarrior")
    ap.add_argument("--p2-agent", default="control")
    ap.add_argument("--p2-deck", default="MurlocDruid")
    ap.add_argument("--start-player", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lines", type=int, default=60, help="preview N lines")
    ap.add_argument("--out", default=None, help="write the .log here")
    args = ap.parse_args()

    sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH = 10, 14
    g, lines, secs = simulate_game_powerlog(
        args.p1_agent, args.p1_deck, args.p2_agent, args.p2_deck,
        args.start_player, seed=args.seed)
    winner, p1s, p2s, turns = game_result(g)
    print("winner=%s (P1=%s P2=%s) turns=%d  %d lines  %.1fs"
          % (winner, p1s, p2s, turns, len(lines), secs))
    problems = plr.validate_lines(lines)
    print("validate:", "; ".join(problems) if problems else "CLEAN")
    if args.out:
        plr.write_log(args.out, lines)
        print("wrote", args.out)
    for ln in lines[:args.lines]:
        print(ln)
