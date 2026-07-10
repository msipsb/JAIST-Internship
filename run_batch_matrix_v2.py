#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch matrix runner V2: rich per-decision JSONL logs (see sim_rich.py).

Same matchup structure as log/run_batch_matrix.py, but Agent 1 plays EVERY
deck (one call / notebook cell per P1 deck):

  run_deck1_batch(deck1):
      5 P1 agents x 5 P2 agents x 9 P2 decks x 20 games = 4,500 games
      -> log_v2/<agent1>_<deck1>/game_NNN_<a1>-<d1>_vs_<a2>-<d2>.jsonl
         log_v2/<agent1>_<deck1>/summary.csv

  all 9 P1 decks -> 45 directories, 40,500 games total.

Search/match settings identical to V1: depth=10 width=14, random start
player, shuffled decks. Resumable: existing .jsonl files are skipped, and
files are written atomically (temp + rename), so an interrupted run never
leaves a partial game file.

    py -3 run_batch_matrix_v2.py --deck1 AggroPirateWarrior
    py -3 run_batch_matrix_v2.py --deck1 all            # everything (~10-15 h)
    py -3 run_batch_matrix_v2.py --smoke                # 1 quick game per deck

Or run cells in run_batch_v2.ipynb (one cell per P1 deck).
"""
import os
import csv
import json
import time
import random
import argparse

import sabberstone_simulator as sim
import sim_rich

# ---- settings (identical to V1 batch) ----
sim.SEARCH_MAX_DEPTH = 10
sim.SEARCH_MAX_WIDTH = 14
sim.SHUFFLE = True
START_PLAYER = -1              # -1 = random each game

AGENT1_FUNCS = ["aggro", "control", "fatigue", "midrange", "ramp"]
AGENT2_FUNCS = ["aggro", "control", "midrange", "fatigue", "ramp"]
DECK_LIST = list(sim.DECKS.keys())
NUM_GAMES = 20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_ROOT = os.path.join(SCRIPT_DIR, "log_v2")

SUMMARY_HEADER = [
    "game", "seed", "start_player",
    "p1_agent", "p1_deck", "p1_class",
    "p2_agent", "p2_deck", "p2_class",
    "winner", "p1_state", "p2_state", "turns",
    "win_condition", "n_decisions", "seconds", "log_file",
]


def write_jsonl(path, records):
    """Atomic write: full file appears only when the game finished cleanly."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def validate_records(records):
    """Sanity-check one game's records; return a list of problem strings."""
    problems = []
    if not records or records[0].get("type") != "game_meta":
        problems.append("first record is not game_meta")
    mulligans = [r for r in records if r.get("type") == "mulligan"]
    if len(mulligans) != 2:
        problems.append("expected 2 mulligan records, got %d" % len(mulligans))
    for m in mulligans:
        if len(m.get("kept", [])) + len(m.get("replaced", [])) != len(m.get("offered", [])):
            problems.append("mulligan kept+replaced != offered (P%s)" % m.get("player"))
    decisions = [r for r in records if r.get("type") == "decision"]
    if not decisions:
        problems.append("no decision records")
    unmatched = 0
    for d in decisions:
        if not d.get("options"):
            problems.append("decision di=%s has empty options" % d.get("di"))
        if d.get("chosen_index", -1) < 0:
            unmatched += 1
        st = d.get("state", {})
        if "p1" not in st or "p2" not in st:
            problems.append("decision di=%s missing state sides" % d.get("di"))
    if unmatched:
        # sim_rich only executes matched live options, so any -1 is a bug
        problems.append("chosen not found in options for %d/%d decisions"
                        % (unmatched, len(decisions)))
    if not records or records[-1].get("type") != "game_end":
        problems.append("last record is not game_end")
    else:
        end = records[-1]
        if end.get("winner") not in ("P1", "P2", "DRAW"):
            problems.append("bad winner %r" % end.get("winner"))
        if not end.get("card_history"):
            problems.append("empty card_history")
    return problems


def run_deck1_batch(deck1, num_games=NUM_GAMES, log_root=None,
                    agent1_funcs=None, agent2_funcs=None, deck2_list=None):
    """Run the full matrix for one P1 deck (all 5 P1 agents). Resumable."""
    log_root = log_root or DEFAULT_LOG_ROOT
    agent1_funcs = agent1_funcs or AGENT1_FUNCS
    agent2_funcs = agent2_funcs or AGENT2_FUNCS
    deck2_list = deck2_list or DECK_LIST
    d1_name = sim._resolve_deck(deck1)[0]

    total = len(agent1_funcs) * len(agent2_funcs) * len(deck2_list) * num_games
    print("=" * 78)
    print("V2 BATCH  deck1=%s  (%d games: %d P1 agents x %d P2 agents x %d decks x %d)"
          % (d1_name, total, len(agent1_funcs), len(agent2_funcs),
             len(deck2_list), num_games))
    print("  search depth=%d width=%d   logs -> %s"
          % (sim.SEARCH_MAX_DEPTH, sim.SEARCH_MAX_WIDTH, log_root))
    print("=" * 78, flush=True)

    t_start = time.perf_counter()
    progress = {"done": 0, "run": 0, "secs": 0.0, "errors": 0}

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
                    log_name = "game_%03d_%s_vs_%s.jsonl" % (i, p1_tag, p2_tag)
                    log_path = os.path.join(log_dir, log_name)
                    if os.path.exists(log_path):
                        wins["SKIP"] += 1
                        progress["done"] += 1
                        continue

                    sp = START_PLAYER if START_PLAYER in (1, 2) else random.choice([1, 2])
                    seed = random.getrandbits(48)
                    try:
                        game, records, secs = sim_rich.simulate_game_rich(
                            a1, d1_name, a2, d2_name, sp,
                            seed=seed, game_index=i)
                    except Exception as exc:               # noqa: BLE001
                        wins["ERROR"] += 1
                        progress["done"] += 1
                        progress["errors"] += 1
                        writer.writerow([i, seed, sp,
                                         a1, d1_name, str(sim.DECKS[d1_name][1]),
                                         a2, d2_name, str(d2_class),
                                         "ERROR", "", "", "", "", "", "", str(exc)])
                        sf.flush()
                        continue

                    winner = records[-1]["winner"]
                    wins[winner] += 1
                    progress["done"] += 1
                    progress["run"] += 1
                    progress["secs"] += secs

                    write_jsonl(log_path, records)
                    end = records[-1]
                    writer.writerow([i, seed, sp,
                                     a1, d1_name, str(sim.DECKS[d1_name][1]),
                                     a2, d2_name, str(d2_class),
                                     winner, end["p1_state"], end["p2_state"],
                                     end["turns"], end["win_condition"],
                                     end["n_decisions"], "%.2f" % secs, log_name])
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
    print("\ndeck1=%s DONE: %d processed (%d run, %d errors) in %.1f min"
          % (d1_name, progress["done"], progress["run"], progress["errors"],
             elapsed / 60.0), flush=True)
    return progress


# ----------------------------------------------------------------------------
# smoke test: 1 game per P1 deck, rotating agents/opponents, full validation
# ----------------------------------------------------------------------------
def run_smoke(log_root=None, max_retries=6):
    log_root = log_root or os.path.join(SCRIPT_DIR, "log_v2_smoke")
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
            sp = random.choice([1, 2])
            try:
                game, records, secs = sim_rich.simulate_game_rich(
                    a1, deck1, a2, deck2, sp, game_index=1)
            except Exception as exc:                       # noqa: BLE001
                print("  %-24s attempt %d ERROR: %s" % (deck1, attempt, exc),
                      flush=True)
                continue
            problems = validate_records(records)
            path = os.path.join(log_root, "smoke_%s_%s_vs_%s_%s.jsonl"
                                % (a1, deck1, a2, deck2))
            write_jsonl(path, records)
            end = records[-1]
            n_dec = end["n_decisions"]
            size_kb = os.path.getsize(path) / 1024.0
            status = "OK " if not problems else "WARN"
            print("  %s %-10s %-24s vs %-10s %-24s winner=%-4s turns=%2d "
                  "decisions=%3d  %6.1f KB  %.1fs"
                  % (status, a1, deck1, a2, deck2, end["winner"],
                     end["turns"], n_dec, size_kb, secs), flush=True)
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
    p = argparse.ArgumentParser(description="V2 rich-log batch runner.")
    p.add_argument("--deck1", default=None,
                   help="P1 deck name, or 'all' for every deck sequentially.")
    p.add_argument("--num-games", type=int, default=NUM_GAMES)
    p.add_argument("--log-root", default=None,
                   help="output root (default: log_v2/ next to this script)")
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
