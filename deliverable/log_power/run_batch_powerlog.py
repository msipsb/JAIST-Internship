#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch matrix runner: re-simulate the whole dataset as Hearthstone Power.log files.

Same matchup structure and settings as log_v2/run_batch_matrix_v2.py -- Agent 1
plays EVERY deck, one call / notebook cell per P1 deck:

  run_deck1_batch(deck1):
      5 P1 agents x 5 P2 agents x 9 P2 decks x 20 games = 4,500 games
      -> log_power/<agent1>_<deck1>/game_NNN_<a1>-<d1>_vs_<a2>-<d2>.log
         log_power/<agent1>_<deck1>/summary.csv

  all 9 P1 decks -> 45 directories, 40,500 games total.

Search/match settings identical to V2: depth=10 width=14, random start player,
shuffled decks, decks_v2 lists. Resumable: existing .log files are skipped and
files are written atomically (temp + rename), so an interrupted run never
leaves a partial game.

These games are RE-SIMULATED, not converted: they are new games and do NOT
correspond to the log_v2 games. Expect ~12-16 h and ~10 GB for the full matrix.

    py -3 run_batch_powerlog.py --deck1 AggroPirateWarrior
    py -3 run_batch_powerlog.py --deck1 all            # everything
    py -3 run_batch_powerlog.py --smoke                # 1 quick game per deck

Or run cells in run_batch_powerlog.ipynb (one cell per P1 deck).
"""
import os
import sys
import csv
import time
import random
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
for _p in (PARENT_DIR, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_powerlog as sp                    # noqa: E402  (loads the .NET bridge)
import sabberstone_simulator as sim          # noqa: E402
import powerlog_render as plr                # noqa: E402

# ---- settings (identical to the V2 batch) ----
sim.SEARCH_MAX_DEPTH = 10
sim.SEARCH_MAX_WIDTH = 14
sim.SHUFFLE = True
START_PLAYER = -1              # -1 = random each game

AGENT1_FUNCS = ["aggro", "control", "fatigue", "midrange", "ramp"]
AGENT2_FUNCS = ["aggro", "control", "midrange", "fatigue", "ramp"]
DECK_LIST = list(sim.DECKS.keys())
NUM_GAMES = 20

DEFAULT_LOG_ROOT = SCRIPT_DIR              # logs land in log_power/

SUMMARY_HEADER = [
    "game", "seed", "start_player",
    "p1_agent", "p1_deck", "p1_class",
    "p2_agent", "p2_deck", "p2_class",
    "winner", "p1_state", "p2_state", "turns",
    "lines", "seconds", "log_file",
]


def run_deck1_batch(deck1, num_games=NUM_GAMES, log_root=None,
                    agent1_funcs=None, agent2_funcs=None, deck2_list=None,
                    viewpoint=plr.VIEWPOINT):
    """Re-simulate the full matrix for one P1 deck. Resumable."""
    log_root = log_root or DEFAULT_LOG_ROOT
    agent1_funcs = agent1_funcs or AGENT1_FUNCS
    agent2_funcs = agent2_funcs or AGENT2_FUNCS
    deck2_list = deck2_list or DECK_LIST
    d1_name = sim._resolve_deck(deck1)[0]

    total = len(agent1_funcs) * len(agent2_funcs) * len(deck2_list) * num_games
    print("=" * 78)
    print("POWER.LOG BATCH  deck1=%s  (%d games: %d P1 agents x %d P2 agents "
          "x %d decks x %d)" % (d1_name, total, len(agent1_funcs),
                                len(agent2_funcs), len(deck2_list), num_games))
    print("  search depth=%d width=%d  viewpoint=P%d  logs -> %s"
          % (sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH, viewpoint, log_root))
    print("=" * 78, flush=True)

    t_start = time.perf_counter()
    progress = {"done": 0, "run": 0, "secs": 0.0, "errors": 0, "warned": 0}

    for a1 in agent1_funcs:
        log_dir = os.path.join(log_root, "%s_%s" % (a1, d1_name))
        os.makedirs(log_dir, exist_ok=True)
        summary_path = os.path.join(log_dir, "summary.csv")
        new_summary = not os.path.exists(summary_path)
        sf = open(summary_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(sf)
        if new_summary:
            writer.writerow(SUMMARY_HEADER)
            sf.flush()

        print("\n### %s_%s  (%d games)" % (a1, d1_name,
              len(agent2_funcs) * len(deck2_list) * num_games), flush=True)
        p1_tag = "%s-%s" % (a1, d1_name)

        for a2 in agent2_funcs:
            for deck2 in deck2_list:
                d2_name, _, d2_class = sim._resolve_deck(deck2)
                p2_tag = "%s-%s" % (a2, d2_name)
                wins = {"P1": 0, "P2": 0, "DRAW": 0, "SKIP": 0, "ERROR": 0}
                combo_t0 = time.perf_counter()

                for i in range(1, num_games + 1):
                    log_name = "game_%03d_%s_vs_%s.log" % (i, p1_tag, p2_tag)
                    log_path = os.path.join(log_dir, log_name)
                    if os.path.exists(log_path):
                        wins["SKIP"] += 1
                        progress["done"] += 1
                        continue

                    st = START_PLAYER if START_PLAYER in (1, 2) else \
                        random.choice([1, 2])
                    seed = random.getrandbits(48)
                    try:
                        game, lines, secs = sp.simulate_game_powerlog(
                            a1, d1_name, a2, d2_name, st,
                            seed=seed, game_index=i, viewpoint=viewpoint)
                    except Exception as exc:               # noqa: BLE001
                        # ~0.8% of games hit a card SabberStone has not
                        # implemented; record and move on (as the V2 batch did)
                        wins["ERROR"] += 1
                        progress["done"] += 1
                        progress["errors"] += 1
                        writer.writerow([i, seed, st,
                                         a1, d1_name, str(sim.DECKS[d1_name][1]),
                                         a2, d2_name, str(d2_class),
                                         "ERROR", "", "", "", "", "", str(exc)])
                        sf.flush()
                        continue

                    winner, p1_state, p2_state, turns = sp.game_result(game)
                    wins[winner] += 1
                    progress["done"] += 1
                    progress["run"] += 1
                    progress["secs"] += secs

                    problems = plr.validate_lines(lines, viewpoint)
                    if problems:
                        progress["warned"] += 1
                        print("    WARN %s: %s" % (log_name, "; ".join(problems)),
                              flush=True)

                    plr.write_log(log_path, lines)
                    writer.writerow([i, seed, st,
                                     a1, d1_name, str(sim.DECKS[d1_name][1]),
                                     a2, d2_name, str(d2_class),
                                     winner, p1_state, p2_state, turns,
                                     len(lines), "%.2f" % secs, log_name])
                    sf.flush()

                avg = progress["secs"] / max(progress["run"], 1)
                remaining = (total - progress["done"]) * avg
                print("  %-8s vs %-8s/%-24s P1=%2d P2=%2d D=%d E=%d S=%d"
                      "  [%5d/%d %3.0f%%]  eta %5.1f min  (%.1fs)"
                      % (a1, a2, d2_name, wins["P1"], wins["P2"], wins["DRAW"],
                         wins["ERROR"], wins["SKIP"],
                         progress["done"], total,
                         100.0 * progress["done"] / total,
                         remaining / 60.0, time.perf_counter() - combo_t0),
                      flush=True)

        sf.close()

    elapsed = time.perf_counter() - t_start
    print("\ndeck1=%s DONE: %d processed (%d run, %d errors, %d warned) in %.1f min"
          % (d1_name, progress["done"], progress["run"], progress["errors"],
             progress["warned"], elapsed / 60.0), flush=True)
    return progress


# ----------------------------------------------------------------------------
# smoke test: 1 game per P1 deck, rotating agents/opponents, full validation
# ----------------------------------------------------------------------------
def run_smoke(log_root=None, max_retries=6, viewpoint=plr.VIEWPOINT):
    log_root = log_root or os.path.join(SCRIPT_DIR, "_smoke")
    os.makedirs(log_root, exist_ok=True)
    decks = DECK_LIST
    n_ok = 0
    failures = []
    for k, deck1 in enumerate(decks):
        a1 = AGENT1_FUNCS[k % len(AGENT1_FUNCS)]
        a2 = AGENT2_FUNCS[(k + 1) % len(AGENT2_FUNCS)]
        deck2 = decks[(k + 1) % len(decks)]
        ok = False
        for attempt in range(1, max_retries + 1):
            st = random.choice([1, 2])
            try:
                game, lines, secs = sp.simulate_game_powerlog(
                    a1, deck1, a2, deck2, st, game_index=1, viewpoint=viewpoint)
            except Exception as exc:                       # noqa: BLE001
                print("  %-24s attempt %d ERROR: %s" % (deck1, attempt, exc),
                      flush=True)
                continue
            problems = plr.validate_lines(lines, viewpoint)
            path = os.path.join(log_root, "smoke_%s_%s_vs_%s_%s.log"
                                % (a1, deck1, a2, deck2))
            plr.write_log(path, lines)
            winner, _, _, turns = sp.game_result(game)
            size_kb = os.path.getsize(path) / 1024.0
            print("  %s %-10s %-24s vs %-10s %-24s winner=%-4s turns=%2d "
                  "lines=%5d  %6.1f KB  %.1fs"
                  % ("OK " if not problems else "WARN", a1, deck1, a2, deck2,
                     winner, turns, len(lines), size_kb, secs), flush=True)
            for p in problems:
                print("        problem: %s" % p, flush=True)
            if problems:
                failures.append((deck1, problems))
            else:
                n_ok += 1
            ok = True
            break
        if not ok:
            failures.append((deck1, ["all %d attempts raised" % max_retries]))
    print("\nSMOKE: %d/%d decks clean, %d with problems"
          % (n_ok, len(decks), len(failures)), flush=True)
    return failures


def _parse_args():
    p = argparse.ArgumentParser(description="Power.log batch matrix runner.")
    p.add_argument("--deck1", default=None,
                   help="P1 deck name, or 'all' for every deck sequentially.")
    p.add_argument("--num-games", type=int, default=NUM_GAMES)
    p.add_argument("--log-root", default=None,
                   help="output root (default: log_power/ next to this script)")
    p.add_argument("--smoke", action="store_true",
                   help="run 1 validated game per deck and exit")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.smoke:
        failures = run_smoke(args.log_root)
        raise SystemExit(1 if failures else 0)
    if not args.deck1:
        raise SystemExit("use --deck1 <name>|all or --smoke")
    decks = DECK_LIST if args.deck1.lower() == "all" else [args.deck1]
    for d in decks:
        run_deck1_batch(d, num_games=args.num_games, log_root=args.log_root)


if __name__ == "__main__":
    main()
