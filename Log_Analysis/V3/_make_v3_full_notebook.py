"""Assemble the FULL V3 notebook (Change A + Change B) by patching the V3-B notebook.

Chain: V1 -> V2 (_make_v2) -> V3 Change-B (_make_v3) -> V3 full A+B (this file).
This patches the Change-B notebook: point parser/cache at `playstyle_log_parse_v3_full`,
add the 5 kept Change-A tell metrics to the per-style grids + clustering FEATURES,
and refresh the glossary / prose.

Run:  py -3 Log_Analysis/V3/_make_v3_full_notebook.py
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "playstyle_log_distribution_analysis_RenoKazakusMage_v3.ipynb")
DST  = os.path.join(HERE, "playstyle_log_distribution_analysis_RenoKazakusMage_v3_full.ipynb")
nb = json.load(open(SRC, encoding="utf-8"))


def cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def set_src(cid, text):
    cell(cid)["source"] = text.strip("\n")


def sub_in(cid, pattern, repl, count=1, flags=0):
    c = cell(cid)
    s = "".join(c["source"])
    new, n = re.subn(pattern, repl, s, count=count, flags=flags)
    assert n == count, f"{cid}: expected {count} repl, got {n} for {pattern!r}"
    c["source"] = new


# ---- [2] setup: import the full parser + A_FEATURES, full cache ---------------
sub_in("16808bd4", r'os\.path\.abspath\("playstyle_log_parse_v3\.py"\)',
       'os.path.abspath("playstyle_log_parse_v3_full.py")')
sub_in("16808bd4",
       r"from playstyle_log_parse_v3 import build_frames, STYLES, BASE_DIR, TRAJ_METRICS",
       "from playstyle_log_parse_v3_full import build_frames, STYLES, BASE_DIR, TRAJ_METRICS, A_FEATURES")
sub_in("16808bd4", r'"playstyle_log_metrics_RenoKazakusMage_v3\.pkl"',
       '"playstyle_log_metrics_RenoKazakusMage_v3_full.pkl"')

# ---- [5] add the 5 kept Change-A tells to the METRICS grid + FEATURES ----------
sub_in("93f623df",
       r'    "Hero-power / turn":         \("hp_per_turn",                   "kde"\),\n\}',
       '    "Hero-power / turn":         ("hp_per_turn",                   "kde"),\n'
       '    "Max card cost (ramp)":      ("max_card_cost",                 "kde"),\n'
       '    "Proactive ratio (control)": ("proactive_ratio",               "clip01"),\n'
       '    "Extra draws/turn (fatigue)":("cards_drawn_per_turn",          "kde"),\n'
       '    "Cards left in deck (fat.)": ("cards_left_in_deck",            "disc"),\n'
       '    "First minion turn (mid.)":  ("first_minion_turn",             "disc"),\n}')

sub_in("93f623df",
       r"BASE_FEATURES = \[c for _, \(c, _\) in METRICS\.items\(\)\].*?"
       r"FEATURES = BASE_FEATURES \+ TRAJ_FEATURES.*?clustering",
       "BASE_FEATURES = [c for _, (c, _) in METRICS.items()]            # 11 V2 + 5 Change-A tells = 16\n"
       "TRAJ_FEATURES = list(TRAJ_METRICS)                             # 13 Change-B trajectory metrics\n"
       "FEATURES = BASE_FEATURES + TRAJ_FEATURES                       # 29 features used for clustering\n"
       "# extra_mana_crystals is intentionally absent: it is identically ~0 on this ramp-less highlander deck",
       flags=re.S)

# grids now hold 16 metrics -> 4x4
sub_in("93f623df", r"plt\.subplots\(3, 4, figsize=\(16, 9\)\)", "plt.subplots(4, 4, figsize=(16, 12))")

# ---- [3] overlay grid 4x4 + slightly wider fingerprint heatmap -----------------
sub_in("c3ca5a15", r"plt\.subplots\(3, 4, figsize=\(16, 10\)\)", "plt.subplots(4, 4, figsize=(16, 13))")
sub_in("c3ca5a15", r"fig, ax = plt\.subplots\(figsize=\(20, 3\.6\)\)",
       "fig, ax = plt.subplots(figsize=(22, 3.8))")

# ============================ markdown rewrites ================================
sub_in("e7038b5e",
       r"# Play-style Distribution & Clustering of the 5 AI Archetypes — `RenoKazakusMage` \(v3 · 11 metrics \+ trajectory\)",
       "# Play-style Distribution & Clustering of the 5 AI Archetypes — `RenoKazakusMage` "
       "(v3-full · 11 metrics + trajectory + Change-A tells)")

sub_in("e7038b5e",
       r"\(Change A — new raw ramp/control/fatigue \*tell\* metrics that need extra\n"
       r"draw / deck-count / mana-crystal log lines — is a separate later stage; \*\*this notebook measures what\n"
       r"trajectory alone buys\*\*, vs the v2 11-metric baseline\.\)",
       "**Change A — raw *tell* metrics** is now also included (this is the *full* V3). Each targets one "
       "value style: `max_card_cost` & `extra_mana_crystals` (ramp), `proactive_ratio` "
       "(control — minions vs removal/trades), `cards_drawn_per_turn` & `cards_left_in_deck` (fatigue), "
       "`first_minion_turn` (midrange). On this **ramp-less highlander** deck `extra_mana_crystals` is "
       "identically ~0 (no ramp cards), so it is computed but dropped from the clustering features; the "
       "fatigue tells are weak (the deck rarely decks out by ~turn 9, as expected). The clear winners are "
       "`max_card_cost` and `proactive_ratio`.")

set_src("45bea743", '''
## 1 · Setup & parse

The parser lives in [`playstyle_log_parse_v3_full.py`](playstyle_log_parse_v3_full.py) — the V2 11-metric
parser **+ Change B (trajectory) + Change A (tells)**. Change B needs no new log lines (it surfaces the
per-turn board/hand/mana V2 already tracked). Change A reads two extra line types confirmed present in the
logs — `DrawPhase: P1 draws ...` (card economy) and `'Player[2]' set data RESOURCES to N` (mana crystals)
— plus derivations from data we already had. It emits `board_end`/`mana_spent` into `turns_df` (for §3b)
and adds **13 trajectory + 6 Change-A columns** to `games_df`. `build_frames(deck="RenoKazakusMage")`
selects the folder family; results cache to `playstyle_log_metrics_RenoKazakusMage_v3_full.pkl` — delete it
to force a full re-parse.''')

sub_in("5e18783f",
       r"A checkpoint past a game's length is `NaN` and the §4 matrix median-imputes it; since aggro games are\n"
       r"short, its late checkpoints fall back to the median, while the value styles — which reach turn 9 — get\n"
       r"genuine late-game values\. \*\*24 features total \(11 \+ 13\)\*\* feed the clustering from §4 on\.",
       '''**Change A — raw "tell" metrics (new in the full v3).** Each targets a value style; on this deck the
strong ones are `max_card_cost` and `proactive_ratio`.

| metric (column) | target style | definition |
|---|---|---|
| `max_card_cost` | ramp | highest mana cost card P1 played (ramp/value reach the top end; aggro mis-pilots and never does) |
| `extra_mana_crystals` | ramp | crystals gained faster than +1/turn — **~0 here** (highlander Mage has no ramp cards), dropped from features |
| `proactive_ratio` | control | minions played / (minions + enemy minions killed) — low = reactive/removal-heavy (control) |
| `cards_drawn_per_turn` | fatigue | draws beyond the opening hand, per turn (~1.0 natural; >1 = card-draw spells). Weak split here |
| `cards_left_in_deck` | fatigue | `30 − P1 draws` (deck remaining; mill ≈ 0). Tracks game length more than deck-out — weak |
| `first_minion_turn` | midrange | my-turn the first P1 minion is played (aggro is latest; value styles ~turn 2.3) |

A checkpoint/tell past a game's length is `NaN` and the §4 matrix median-imputes it. **29 features total
(11 V2 + 13 trajectory + 5 kept Change-A)** feed the clustering from §4 on.''')

sub_in("2348a354",
       r"`X` now has \*\*24 columns\*\* = the 11 whole-game V2\nmetrics \*\*\+ 13 Change-B trajectory features\*\*\.",
       "`X` now has **29 columns** = 11 whole-game V2 metrics **+ 13 Change-B trajectory features + 5 "
       "Change-A tell metrics** (`extra_mana_crystals` excluded as a dead, all-zero feature on this deck).")

sub_in("de97b98a",
       r"\* \*\*What v3 adds \(Change B\):\*\* keeping the 11 averages and adding turn-checkpointed \*\*trajectory\*\*\n"
       r"  features \(board / mana / hand at t3/5/7/9 \+ mana slope\) targets the value-style overlap \*directly\* —\n"
       r"  midrange's early board vs ramp's late build, and ramp's late mana ramp-up\. §5b \(per-game RF accuracy\)\n"
       r"  and §8 \(games-to-classify per style\) are the scoreboard for whether trajectory tightens the hard-to-pin\n"
       r"  \*\*ramp / midrange / control / fatigue\*\* group relative to v2's 11-metric baseline\.",
       "* **What full v3 adds (Change B + Change A):** keep the 11 averages and add (B) turn-checkpointed "
       "**trajectory** features and (A) raw **tell** metrics. On this deck the decisive Change-A tell is "
       "**`max_card_cost`** (aggro ~5.3 → ramp ~7.9, monotone across styles) with **`proactive_ratio`** "
       "isolating control; the ramp-crystal and fatigue tells are dead/weak as predicted (no ramp cards; "
       "the deck rarely decks out). §5b (per-game RF) and §8 (games-to-classify) are the scoreboard vs the "
       "v2 11-metric and Change-B-only baselines.")

# ---- clear all outputs / execution counts ------------------------------------
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", os.path.relpath(DST, os.path.join(HERE, "..", "..")), "cells:", len(nb["cells"]))
