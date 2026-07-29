#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build log_power/run_batch_powerlog.ipynb (mirrors log_v2/run_batch_v2.ipynb).

Rebuild with:  py -3 _make_powerlog_notebook.py
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "run_batch_powerlog.ipynb")

DECKS = ["MiraclePirateRogue", "ZooDiscardWarlock", "RenoKazakusDragonPriest",
         "MidrangeSecretHunter", "MidrangeBuffPaladin", "MurlocDruid",
         "MidrangeJadeShaman", "AggroPirateWarrior", "RenoKazakusMage"]


def md(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": source.strip("\n").split("\n")}


def code(cell_id, source):
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": source.strip("\n").split("\n")}


cells = []

cells.append(md("intro", """
# Power.log batch simulation runner -- official-Hearthstone-format logs

Each deck cell **re-simulates** the full matrix for that **P1 deck**:
5 P1 agents x 5 P2 agents x 9 P2 decks x 20 games = **4,500 games**,
writing one `.log` per game into `log_power/<agent1>_<deck1>/` plus a `summary.csv`.

* **Resumable**: finished games are skipped, so you can interrupt the kernel and re-run a cell.
* Run cells in any order; each deck is independent.
* **All 9 cells = 40,500 games. Budget ~15-20 h and ~17 GB** (a real Power.log
  is genuinely 200-500 KB per game, so the size is expected -- check your disk first).

## What these logs are

The real Hearthstone client writes `Logs/Power.log` when `log.config` enables the
Power zone. SabberStone already models that exact packet vocabulary: with
`GameConfig.History = true` the engine records a **PowerHistory** of
`CREATE_GAME` / `FULL_ENTITY` / `SHOW_ENTITY` / `HIDE_ENTITY` / `TAG_CHANGE` /
`BLOCK_START` / `BLOCK_END` entries as it plays. So these logs are **not a
reconstruction**: `powerlog_render.py` re-renders the engine's own history --
real trigger-by-trigger ordering, real block nesting -- into the client's text
format.

Verified end to end: every rendered game is parsed back with **hslog**
(HearthSim's parser, the one behind HSReplay.net) and the reconstructed winner,
turn count, hero health, armor and **both boards** are compared against the
SabberStone game object. See `roundtrip_check.py`.

## Things to know before you run

* **These are NEW games.** They are re-simulated, so they do **not** correspond
  to the `log_v2/` games -- do not join the two datasets by game index.
* **Viewpoint = P1's client.** A real Power.log is written by one client, so it
  only holds what that client saw: deck cards are unknown until drawn (yours
  included), and P2's hand/secrets stay hidden until something makes them public.
  This deliberately discards information the simulator has -- that is the point.
* **Labels never appear inside the `.log`.** The agent/playstyle/deck ground
  truth lives in `summary.csv` only, so features and labels stay separate.
* **Timestamps are synthetic**, derived from each game's RNG seed. Real think
  time correlates with which agent is playing, so using it would leak the
  playstyle label into the timing.
* `History=True` costs ~16% runtime and does not bias play (mean turns 14.68 vs
  14.57 over 40 games/arm, Mann-Whitney p=0.96); it does shift the RNG stream,
  so a seed does not reproduce a `History=False` game.
* ~0.8% of games raise on a card SabberStone has not implemented; those are
  recorded as `ERROR` rows in `summary.csv` and skipped.
"""))

cells.append(md("md_setup", "## Setup"))
cells.append(code("setup", '''
import os, sys
sys.path.insert(0, os.path.abspath("."))   # notebook lives in log_power/
import run_batch_powerlog as pl

print("decks :", ", ".join(pl.DECK_LIST))
print("agents:", ", ".join(pl.AGENT1_FUNCS))
print("games per matchup:", pl.NUM_GAMES,
      " search depth=%d width=%d" % (pl.sim.SEARCH_MAX_DEPTH, pl.sim.SEARCH_MAX_WIDTH))
print("viewpoint: P%d (that client's view only)" % pl.plr.VIEWPOINT)
print("output ->", pl.DEFAULT_LOG_ROOT)
'''))

cells.append(md("md_smoke", """
## Smoke test first

One validated game per deck (~15 s). Run this before committing to the full
matrix: it exercises all 9 decks and checks every rendered log.
"""))
cells.append(code("smoke", "pl.run_smoke()"))

cells.append(md("md_preview", """
## Preview one game

What a single match looks like in the official format. `CREATE_GAME`, then real
engine blocks -- note the `BLOCK_START BlockType=PLAY` with the ordering the
engine actually executed.
"""))
cells.append(code("preview", '''
import sim_powerlog as sp, powerlog_render as plr
game, lines, secs = sp.simulate_game_powerlog(
    "aggro", "AggroPirateWarrior", "control", "MurlocDruid", 1, seed=42)
print("%s in %d turns -- %d lines, %.1fs" % (sp.game_result(game)[0],
                                             sp.game_result(game)[3],
                                             len(lines), secs))
print("validate:", plr.validate_lines(lines) or "CLEAN")
print()
for ln in lines[:25]:
    print(ln)
print("...")
i = next(i for i, l in enumerate(lines) if "BlockType=PLAY" in l)
for ln in lines[i:i + 12]:
    print(ln)
'''))

cells.append(md("md_roundtrip", """
## Verify against the real Hearthstone parser (optional)

Simulates a few games, renders them, then parses them back with **hslog**
(HearthSim's parser) and diffs the reconstructed state against the engine.

Needs `hslog` **in this notebook's kernel**. This repo has both a global `py -3`
and `d:\\test\\.venv`, so install it into the interpreter the kernel is actually
running -- the cell prints the exact command if it is missing. Nothing else in
this notebook needs it; skipping only skips the verification.
"""))
cells.append(code("roundtrip", '''
import sys
try:
    import hslog  # noqa: F401
except ImportError:
    print("hslog is not installed in this kernel -- skipping verification.")
    print('  install with:  "%s" -m pip install hslog' % sys.executable)
else:
    import roundtrip_check as rc
    for i in range(6):
        problems, info = rc.check_game("aggro", "AggroPirateWarrior",
                                       "control", "MurlocDruid", 1, seed=100 + i)
        print("%-5s %s in %2d turns, %5d lines  %s"
              % ("OK" if not problems else "DIFF", info["winner"],
                 info["turns"], info["lines"], "; ".join(problems)))
'''))

for deck in DECKS:
    cells.append(md("md_%s" % deck, "## P1 deck: %s" % deck))
    cells.append(code("run_%s" % deck, 'pl.run_deck1_batch("%s")' % deck))

cells.append(md("md_progress", """
## Progress overview
Re-run anytime to see how far the batch has come.
"""))
cells.append(code("progress", '''
import csv, glob, os
root = pl.DEFAULT_LOG_ROOT
total_done = total_err = total_bytes = 0
print("%-40s %8s %8s %9s %8s" % ("directory", "games", "errors", "size MB", "of 900"))
for sp_ in sorted(glob.glob(os.path.join(root, "*", "summary.csv"))):
    with open(sp_, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    err = sum(1 for r in rows if r["winner"] == "ERROR")
    ok = len(rows) - err
    d = os.path.dirname(sp_)
    mb = sum(os.path.getsize(p) for p in glob.glob(os.path.join(d, "*.log"))) / 1e6
    total_done += ok; total_err += err; total_bytes += mb
    print("%-40s %8d %8d %9.1f %8s" % (os.path.basename(d), ok, err, mb,
                                       "done" if ok >= 900 else ""))
print("\\nTOTAL: %d games logged, %d errors, %.1f GB  (target 40,500)"
      % (total_done, total_err, total_bytes / 1000.0))
'''))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote %s (%d cells)" % (OUT, len(cells)))
