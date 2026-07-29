#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SabberStone 1-vs-1 AI match simulator (Python, via pythonnet).

Runs games between two of SabberStone's built-in heuristic AI agents
(aggro / control / midrange / fatigue / ramp), each playing one of the 9
built-in decks from Decks.cs, and saves a full verbose engine log per game.

------------------------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------------------------
SabberStone is a C#/.NET engine. This script loads the compiled engine DLLs
in-process through pythonnet (the default .NET Framework runtime on Windows,
which is 64-bit and supports the netstandard2.0 assemblies) and drives the
match loop (mulligan -> per-turn look-ahead search -> apply best line) directly
from Python. The same look-ahead/scoring used by SabberStoneBasicAI's
`OptionNode.GetSolutions` + `IScore.Rate()` is reused unchanged.

------------------------------------------------------------------------------
ONE-TIME BUILD (produces the DLLs this script loads)
------------------------------------------------------------------------------
    dotnet build SabberStone/core-extensions/SabberStoneBasicAI/SabberStoneBasicAI.csproj -c Debug

The script self-heals the runtime dependency DLLs (System.Memory, etc.) into
the load folder if a clean rebuild removed them.

------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------
    py -3 sabberstone_simulator.py

Edit the CONFIG block below to choose agents, decks, game count and log folder.
"""

import os
import sys
import csv
import time
import datetime

# ============================================================================
# CONFIG  --  edit this block, then run the file
# ============================================================================

# --- Player 1 ---
AGENT_1 = "aggro"                 # aggro | control | midrange | fatigue | ramp
DECK_1  = "AggroPirateWarrior"    # any deck name from DECKS (see list below)

# --- Player 2 ---
AGENT_2 = "control"               # aggro | control | midrange | fatigue | ramp
DECK_2  = "RenoKazakusMage"       # any deck name from DECKS (see list below)

# --- Run settings ---
NUM_GAMES = 20                    # how many games to simulate
LOG_DIR   = "sim_logs"            # folder for per-game logs + summary.csv
                                  # (relative paths are resolved next to this script)

# --- Match settings ---
START_PLAYER = -1                 # 1 = P1 starts, 2 = P2 starts, -1 = random each game
SHUFFLE      = True               # shuffle decks (False = deterministic deck order)

# --- AI look-ahead search (higher = stronger but slower) ---
SEARCH_MAX_DEPTH = 10             # plies of look-ahead per turn
SEARCH_MAX_WIDTH = 14             # beam width kept at each ply

# --- Where the built engine DLLs live (usually leave as-is) ---
DLL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SabberStone", "core-extensions", "SabberStoneBasicAI", "bin", "Debug", "netstandard2.0",
)

# ============================================================================
# End of CONFIG
# ============================================================================


# ----------------------------------------------------------------------------
# .NET bridge bootstrap
# ----------------------------------------------------------------------------
def _ensure_dependency_dlls():
    """Copy runtime dependency DLLs into DLL_DIR if a rebuild removed them."""
    core_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "SabberStone", "SabberStoneCore", "bin", "Debug", "netstandard2.0", "netstandard2.0",
    )
    needed = [
        "System.Memory.dll",
        "System.Buffers.dll",
        "System.Numerics.Vectors.dll",
        "System.Runtime.CompilerServices.Unsafe.dll",
    ]
    import shutil
    for name in needed:
        dst = os.path.join(DLL_DIR, name)
        if not os.path.exists(dst):
            src = os.path.join(core_out, name)
            if os.path.exists(src):
                shutil.copy2(src, dst)


def _load_engine():
    if not os.path.isdir(DLL_DIR):
        sys.exit(
            "ERROR: engine DLL folder not found:\n  %s\n\n"
            "Build it first with:\n"
            "  dotnet build SabberStone/core-extensions/SabberStoneBasicAI/"
            "SabberStoneBasicAI.csproj -c Debug" % DLL_DIR
        )
    if not os.path.exists(os.path.join(DLL_DIR, "SabberStoneBasicAI.dll")):
        sys.exit("ERROR: SabberStoneBasicAI.dll missing in %s -- build the project first." % DLL_DIR)

    _ensure_dependency_dlls()

    import clr  # noqa: F401  (pythonnet)
    if DLL_DIR not in sys.path:
        sys.path.append(DLL_DIR)
    clr.AddReference("SabberStoneCore")
    clr.AddReference("SabberStoneBasicAI")


_load_engine()

from System.Collections.Generic import List                       # noqa: E402
from SabberStoneCore.Config import GameConfig                      # noqa: E402
from SabberStoneCore.Model import Game                            # noqa: E402
from SabberStoneCore.Model.Entities import IPlayable               # noqa: E402
from SabberStoneCore.Enums import CardClass, State, PlayState      # noqa: E402
from SabberStoneCore.Tasks.PlayerTasks import ChooseTask, PlayerTask  # noqa: E402
from SabberStoneBasicAI.Meta import Decks                          # noqa: E402
from SabberStoneBasicAI.Nodes import OptionNode                    # noqa: E402
from SabberStoneBasicAI.Score import (                            # noqa: E402
    AggroScore, ControlScore, FatigueScore, MidRangeScore, RampScore,
)


# ----------------------------------------------------------------------------
# Registries: friendly name -> engine type
# ----------------------------------------------------------------------------
AGENTS = {
    "aggro":    AggroScore,
    "control":  ControlScore,
    "midrange": MidRangeScore,
    "fatigue":  FatigueScore,
    "ramp":     RampScore,
}

# All 9 decks from Decks.cs, each mapped to its required hero class.
DECKS = {
    "MiraclePirateRogue":       ("MiraclePirateRogue",       CardClass.ROGUE),
    "ZooDiscardWarlock":        ("ZooDiscardWarlock",        CardClass.WARLOCK),
    "RenoKazakusDragonPriest":  ("RenoKazakusDragonPriest",  CardClass.PRIEST),
    "MidrangeSecretHunter":     ("MidrangeSecretHunter",     CardClass.HUNTER),
    "MidrangeBuffPaladin":      ("MidrangeBuffPaladin",      CardClass.PALADIN),
    "MurlocDruid":              ("MurlocDruid",              CardClass.DRUID),
    "MidrangeJadeShaman":       ("MidrangeJadeShaman",       CardClass.SHAMAN),
    "AggroPirateWarrior":       ("AggroPirateWarrior",       CardClass.WARRIOR),
    "RenoKazakusMage":          ("RenoKazakusMage",          CardClass.MAGE),
}


def _resolve_agent(name):
    key = str(name).strip().lower()
    if key not in AGENTS:
        sys.exit("ERROR: unknown agent %r. Choose one of: %s"
                 % (name, ", ".join(sorted(AGENTS))))
    return AGENTS[key]


def _resolve_deck(name):
    key = str(name).strip()
    # case-insensitive match for convenience
    if key not in DECKS:
        lookup = {k.lower(): k for k in DECKS}
        if key.lower() in lookup:
            key = lookup[key.lower()]
        else:
            sys.exit("ERROR: unknown deck %r. Choose one of:\n  %s"
                     % (name, "\n  ".join(DECKS)))
    prop, hero = DECKS[key]
    return key, prop, hero


def _deck_cards(prop_name):
    """Fetch a fresh deck List<Card> from Decks (static property/field)."""
    return getattr(Decks, prop_name)


# ----------------------------------------------------------------------------
# Match driving
# ----------------------------------------------------------------------------
def _mulligan_choice(game, player, scorer):
    """Run the agent's MulliganRule over the player's mulligan choices."""
    playables = List[IPlayable]()
    for cid in player.Choice.Choices:
        playables.Add(game.IdEntityDic[cid])
    scorer.Controller = player
    return scorer.MulliganRule().Invoke(playables)


def _play_best_line(game, scorer):
    """Compute the best line for the current player and apply it."""
    cur = game.CurrentPlayer
    solutions = OptionNode.GetSolutions(game, cur.Id, scorer, SEARCH_MAX_DEPTH, SEARCH_MAX_WIDTH)
    if solutions.Count == 0:
        return False
    best = max(solutions, key=lambda n: n.Score)
    line = best.PlayerTasks(List[PlayerTask]())   # ref param -> returned list
    for task in line:
        game.Process(task)
        if game.CurrentPlayer.Choice is not None:
            # a mid-line choice opened up -> stop and let the turn be recomputed
            break
    return True


def simulate_game(p1_agent_cls, p1_deck_prop, p1_class,
                  p2_agent_cls, p2_deck_prop, p2_class, start_player):
    """Play one full game; return (game, elapsed_seconds)."""
    cfg = GameConfig()
    cfg.StartPlayer = start_player
    cfg.Player1Name = "P1"
    cfg.Player1HeroClass = p1_class
    cfg.Player1Deck = _deck_cards(p1_deck_prop)
    cfg.Player2Name = "P2"
    cfg.Player2HeroClass = p2_class
    cfg.Player2Deck = _deck_cards(p2_deck_prop)
    cfg.FillDecks = False
    cfg.Shuffle = SHUFFLE
    cfg.SkipMulligan = False
    cfg.Logging = True          # capture the full verbose engine log
    cfg.History = False

    t0 = time.perf_counter()
    game = Game(cfg)
    game.StartGame()

    scorers = {game.Player1.Id: p1_agent_cls(), game.Player2.Id: p2_agent_cls()}

    game.Process(ChooseTask.Mulligan(
        game.Player1, _mulligan_choice(game, game.Player1, scorers[game.Player1.Id])))
    game.Process(ChooseTask.Mulligan(
        game.Player2, _mulligan_choice(game, game.Player2, scorers[game.Player2.Id])))
    game.MainReady()

    safety = 0
    while game.State != State.COMPLETE and safety < 100000:
        scorer = scorers[game.CurrentPlayer.Id]
        if not _play_best_line(game, scorer):
            break
        safety += 1

    return game, time.perf_counter() - t0


def _winner_label(game):
    if game.Player1.PlayState == PlayState.WON:
        return "P1"
    if game.Player2.PlayState == PlayState.WON:
        return "P2"
    return "DRAW"


def write_game_log(path, header_lines, game):
    with open(path, "w", encoding="utf-8") as f:
        for ln in header_lines:
            f.write(ln + "\n")
        f.write("\n" + "=" * 70 + "\nVERBOSE ENGINE LOG\n" + "=" * 70 + "\n")
        for entry in game.Logs:
            f.write(entry.ToString() + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    import random

    a1 = _resolve_agent(AGENT_1)
    a2 = _resolve_agent(AGENT_2)
    d1_name, d1_prop, d1_class = _resolve_deck(DECK_1)
    d2_name, d2_prop, d2_class = _resolve_deck(DECK_2)

    log_dir = LOG_DIR
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
    os.makedirs(log_dir, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p1_tag = "%s-%s" % (AGENT_1.lower(), d1_name)
    p2_tag = "%s-%s" % (AGENT_2.lower(), d2_name)

    print("=" * 70)
    print("SabberStone simulation")
    print("  P1: agent=%-9s deck=%-24s class=%s" % (AGENT_1, d1_name, d1_class))
    print("  P2: agent=%-9s deck=%-24s class=%s" % (AGENT_2, d2_name, d2_class))
    print("  games=%d  start_player=%s  shuffle=%s  depth=%d  width=%d"
          % (NUM_GAMES, START_PLAYER, SHUFFLE, SEARCH_MAX_DEPTH, SEARCH_MAX_WIDTH))
    print("  logs -> %s" % log_dir)
    print("=" * 70)

    summary_path = os.path.join(log_dir, "summary_%s.csv" % stamp)
    wins = {"P1": 0, "P2": 0, "DRAW": 0}
    total_turns = 0
    t_start = time.perf_counter()

    with open(summary_path, "w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow([
            "game", "start_player",
            "p1_agent", "p1_deck", "p1_class",
            "p2_agent", "p2_deck", "p2_class",
            "winner", "p1_state", "p2_state", "turns", "seconds", "log_file",
        ])

        for i in range(1, NUM_GAMES + 1):
            sp = START_PLAYER
            if sp not in (1, 2):
                sp = random.choice([1, 2])

            try:
                game, secs = simulate_game(a1, d1_prop, d1_class,
                                           a2, d2_prop, d2_class, sp)
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                print("  game %3d  ERROR: %s" % (i, exc))
                writer.writerow([i, sp, AGENT_1, d1_name, d1_class,
                                 AGENT_2, d2_name, d2_class,
                                 "ERROR", "", "", "", "", str(exc)])
                continue

            winner = _winner_label(game)
            wins[winner] += 1
            total_turns += game.Turn

            log_name = "game_%03d_%s_vs_%s.log" % (i, p1_tag, p2_tag)
            log_path = os.path.join(log_dir, log_name)
            header = [
                "SabberStone game %d / %d" % (i, NUM_GAMES),
                "timestamp     : %s" % datetime.datetime.now().isoformat(timespec="seconds"),
                "start_player  : P%d" % sp,
                "P1 agent/deck : %s / %s (%s)" % (AGENT_1, d1_name, d1_class),
                "P2 agent/deck : %s / %s (%s)" % (AGENT_2, d2_name, d2_class),
                "search        : depth=%d width=%d" % (SEARCH_MAX_DEPTH, SEARCH_MAX_WIDTH),
                "result        : winner=%s  P1=%s  P2=%s  turns=%d  (%.2fs)"
                % (winner, game.Player1.PlayState, game.Player2.PlayState, game.Turn, secs),
            ]
            write_game_log(log_path, header, game)

            writer.writerow([i, sp, AGENT_1, d1_name, d1_class,
                             AGENT_2, d2_name, d2_class,
                             winner, game.Player1.PlayState, game.Player2.PlayState,
                             game.Turn, "%.2f" % secs, log_name])
            sf.flush()

            print("  game %3d  start=P%d  winner=%-4s  turns=%-3d  %.2fs  -> %s"
                  % (i, sp, winner, game.Turn, secs, log_name))

    elapsed = time.perf_counter() - t_start
    played = wins["P1"] + wins["P2"] + wins["DRAW"]
    print("=" * 70)
    print("Done: %d games in %.1fs (avg %.2fs/game, avg %.1f turns)"
          % (played, elapsed, elapsed / max(played, 1), total_turns / max(played, 1)))
    if played:
        print("  P1 (%s/%s): %d wins (%.1f%%)"
              % (AGENT_1, d1_name, wins["P1"], 100.0 * wins["P1"] / played))
        print("  P2 (%s/%s): %d wins (%.1f%%)"
              % (AGENT_2, d2_name, wins["P2"], 100.0 * wins["P2"] / played))
        if wins["DRAW"]:
            print("  draws: %d" % wins["DRAW"])
    print("  summary CSV -> %s" % summary_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
