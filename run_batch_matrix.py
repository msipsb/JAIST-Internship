#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch matrix runner for SabberStone simulations.

For each Agent-1 scoring function (aggro, control, fatigue, midrange, ramp)
playing deck AggroPirateWarrior, run 20 games against every combination of
Agent-2 scoring function (5) x deck (9 built-in decks, mirrors included).

  per directory : 5 agent-2 funcs * 9 decks * 20 games = 900 games
  directories   : 5  ->  log/<agent1_func>_AggroPirateWarrior/
  grand total   : 4500 games

Each game writes a full verbose engine log (one .log file per game, so 900
files per directory) plus a combined summary.csv per directory.

Search/match settings:
  depth=10  width=14  start_player=random  shuffle=True

Resumable: if a game's .log file already exists it is skipped, so re-running
after an interruption continues where it left off.

    py -3 run_batch_matrix.py
"""
import os
import csv
import time
import datetime
import random

import sabberstone_simulator as sim

# ---- settings (override the simulator module globals) ----
sim.SEARCH_MAX_DEPTH = 10
sim.SEARCH_MAX_WIDTH = 14
sim.SHUFFLE = True
START_PLAYER = -1          # -1 = random each game

# ---- matrix definition ----
AGENT1_FUNCS = ["aggro", "control", "fatigue", "midrange", "ramp"]
DECK1        = "AggroPirateWarrior"
AGENT2_FUNCS = ["aggro", "control", "midrange", "fatigue", "ramp"]
DECK2_LIST   = list(sim.DECKS.keys())          # all 9 decks, mirrors included
NUM_GAMES    = 20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAND_TOTAL = len(AGENT1_FUNCS) * len(AGENT2_FUNCS) * len(DECK2_LIST) * NUM_GAMES

SUMMARY_HEADER = [
    "game", "start_player",
    "p1_agent", "p1_deck", "p1_class",
    "p2_agent", "p2_deck", "p2_class",
    "winner", "p1_state", "p2_state", "turns", "seconds", "log_file",
]


def run_directory(a1, t_start, progress):
    """Run all 900 games for one Agent-1 function into its own directory."""
    a1_cls = sim._resolve_agent(a1)
    d1_name, d1_prop, d1_class = sim._resolve_deck(DECK1)
    log_dir = os.path.join(SCRIPT_DIR, "log", "%s_%s" % (a1, DECK1))
    os.makedirs(log_dir, exist_ok=True)

    summary_path = os.path.join(log_dir, "summary.csv")
    new_summary = not os.path.exists(summary_path)
    sf = open(summary_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(sf)
    if new_summary:
        writer.writerow(SUMMARY_HEADER)
        sf.flush()

    print("\n" + "#" * 78)
    print("# DIRECTORY  log/%s_%s   (900 games)" % (a1, DECK1))
    print("#" * 78, flush=True)

    p1_tag = "%s-%s" % (a1, d1_name)

    for a2 in AGENT2_FUNCS:
        a2_cls = sim._resolve_agent(a2)
        for deck2 in DECK2_LIST:
            d2_name, d2_prop, d2_class = sim._resolve_deck(deck2)
            p2_tag = "%s-%s" % (a2, d2_name)
            wins = {"P1": 0, "P2": 0, "DRAW": 0, "SKIP": 0, "ERROR": 0}
            combo_t0 = time.perf_counter()

            for i in range(1, NUM_GAMES + 1):
                log_name = "game_%03d_%s_vs_%s.log" % (i, p1_tag, p2_tag)
                log_path = os.path.join(log_dir, log_name)

                if os.path.exists(log_path):
                    wins["SKIP"] += 1
                    progress["done"] += 1
                    continue

                sp = START_PLAYER if START_PLAYER in (1, 2) else random.choice([1, 2])
                try:
                    game, secs = sim.simulate_game(
                        a1_cls, d1_prop, d1_class,
                        a2_cls, d2_prop, d2_class, sp)
                except Exception as exc:                       # noqa: BLE001
                    wins["ERROR"] += 1
                    progress["done"] += 1
                    writer.writerow([i, sp, a1, d1_name, d1_class,
                                     a2, d2_name, d2_class,
                                     "ERROR", "", "", "", "", str(exc)])
                    sf.flush()
                    print("    game %3d ERROR: %s" % (i, exc), flush=True)
                    continue

                winner = sim._winner_label(game)
                wins[winner] += 1
                progress["done"] += 1
                progress["run"] += 1
                progress["secs"] += secs

                header = [
                    "SabberStone game %d / %d" % (i, NUM_GAMES),
                    "timestamp     : %s" % datetime.datetime.now().isoformat(timespec="seconds"),
                    "start_player  : P%d" % sp,
                    "P1 agent/deck : %s / %s (%s)" % (a1, d1_name, d1_class),
                    "P2 agent/deck : %s / %s (%s)" % (a2, d2_name, d2_class),
                    "search        : depth=%d width=%d" % (sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH),
                    "result        : winner=%s  P1=%s  P2=%s  turns=%d  (%.2fs)"
                    % (winner, game.Player1.PlayState, game.Player2.PlayState, game.Turn, secs),
                ]
                sim.write_game_log(log_path, header, game)
                writer.writerow([i, sp, a1, d1_name, d1_class,
                                 a2, d2_name, d2_class,
                                 winner, game.Player1.PlayState, game.Player2.PlayState,
                                 game.Turn, "%.2f" % secs, log_name])
                sf.flush()

            # per-combo progress line + overall ETA
            elapsed = time.perf_counter() - t_start
            avg = progress["secs"] / max(progress["run"], 1)
            remaining = (GRAND_TOTAL - progress["done"]) * avg
            print("  %-9s vs %-9s/%-24s  P1=%2d P2=%2d D=%2d"
                  "  [%4d/%d  %.0f%%]  eta %4.1f min  (%.1fs)"
                  % (a1, a2, d2_name, wins["P1"], wins["P2"], wins["DRAW"],
                     progress["done"], GRAND_TOTAL,
                     100.0 * progress["done"] / GRAND_TOTAL,
                     remaining / 60.0, time.perf_counter() - combo_t0),
                  flush=True)

    sf.close()
    print("  -> wrote %s" % summary_path, flush=True)


def main():
    print("=" * 78)
    print("BATCH MATRIX  (depth=%d width=%d, %d games total)"
          % (sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH, GRAND_TOTAL))
    print("  agent1 funcs : %s   deck1 = %s" % (", ".join(AGENT1_FUNCS), DECK1))
    print("  agent2 funcs : %s" % ", ".join(AGENT2_FUNCS))
    print("  agent2 decks : %s" % ", ".join(DECK2_LIST))
    print("=" * 78, flush=True)

    t_start = time.perf_counter()
    progress = {"done": 0, "run": 0, "secs": 0.0}
    for a1 in AGENT1_FUNCS:
        run_directory(a1, t_start, progress)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print("ALL DONE: %d games processed in %.1f min (%d newly run, avg %.2fs/run)"
          % (progress["done"], elapsed / 60.0, progress["run"],
             progress["secs"] / max(progress["run"], 1)))
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
