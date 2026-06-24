"""
Parse SabberStone verbose game logs (log/<style>_AggroPirateWarrior/*.log) into
per-game, me-side (P1) play-style metrics.

This is a copy of ``playstyle_log_parse.py`` restricted to a fixed 11-metric set
(see ``METRICS`` below).  Four of them are new relative to the original parser and
need board / graveyard tracking, so the parser additionally follows
``SummonPhase`` lines (minion ownership) and the card-type tag on every
``added to zone`` line.

Each of the 5 style folders is the *same* AggroPirateWarrior deck driven by a
different AI play-style (aggro / control / fatigue / midrange / ramp).  In every
log P1 is the folder's play-style and the opponent (deck *and* style) varies.
We measure P1 only and pool over all opponents.

The 11 calculated metrics (and nothing else) in games_df:
    n_my_turns                     game length           = P1 turns played
    avg_cards_in_hand              resource holding      = mean hand size at end of P1 turns
    mana_eff                       mana efficiency       = mana spent / mana available
    avg_card_cost                  curve center          = mean cost of cards P1 played
    minion_fraction                card-type mix         = minions / all cards played
    face_attack_ratio              aggression direction  = face attacks / all attacks
    attacks_per_turn               aggression intensity  = total P1 attacks / turns
    enemy_minions_killed_per_turn  removal / trading     = enemy minions to graveyard on P1 turns / turns
    avg_board_minions              board presence        = mean count of P1 minions in PLAY at end of P1 turns
    taken_dmg_per_turn             damage absorbed       = damage to my hero / turns
    hp_per_turn                    hero-power tempo       = hero-power uses / turns

Returns three tidy frames (mirroring the rank5to10 reference notebook):
    games_df : one row per game (the feature table used for distributions/clustering)
    cards_df : one row per card P1 played   (style, mana_cost)   -> mana-curve plots
    turns_df : one row per P1 turn           (style, game, my_turn, hand_end)
"""
import os, re, glob, csv, collections
import numpy as np
import pandas as pd

# this file lives in <repo>/Log_Analysis/V2/; the log/ data and cache .pkl live at
# the repo root (two levels up).  Anchor paths there regardless of the kernel's CWD.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOG_ROOT = os.path.join(BASE_DIR, "log")
STYLES   = ["aggro", "control", "fatigue", "midrange", "ramp"]
MANA_CAP = 10

# the exact set of calculated metrics this parser exposes in games_df
METRICS = [
    "n_my_turns", "avg_cards_in_hand", "mana_eff", "avg_card_cost",
    "minion_fraction", "face_attack_ratio", "attacks_per_turn",
    "enemy_minions_killed_per_turn", "avg_board_minions",
    "taken_dmg_per_turn", "hp_per_turn",
]

# ---- regexes (compiled once) ------------------------------------------------
RE_TURN   = re.compile(r"'Game\[1\]' set data TURN to (\d+)")
RE_HEROENT= re.compile(r"'Player\[(\d+)\]' set data HERO_ENTITY to (\d+)")
RE_DRAW   = re.compile(r"DrawPhase: (P[12]) draws '([^\[]+)\[(\d+)\]'")
RE_ZONE   = re.compile(r"Zone: Entity ''([^\[]+)\[(\d+)\]' \(([A-Z]+)\)' has been added to zone '([A-Z]+)'")
RE_SUMMON = re.compile(r"SummonPhase: Summon Minion '([^\[]+)\[(\d+)\]' to Board of (P[12])")
RE_PLAY   = re.compile(r"PlayCardTask => \[(P[12])\] play '([^\[]+)\[(\d+)\]'\(([A-Z]+)\)")
RE_HPOW   = re.compile(r"HeroPowerTask => \[(P[12])\] using")
RE_PAY    = re.compile(r"PayPhase: Paying '([^\[]+)\[(\d+)\]' for (\d+) Mana")
RE_DMG    = re.compile(r"Character: '([^\[]+)\[(\d+)\]' took damage for (\d+)")
RE_ATK    = re.compile(r"\[AttackPhase\]'([^\[]+)\[(\d+)\]'[^']*? attacked '([^\[]+)\[(\d+)\]'")


def mana_available(n_turns, n_coin):
    """Total mana a player could have spent over n_turns (+1 per Coin) -- same as reference."""
    full = min(n_turns, MANA_CAP)
    base = full * (full + 1) // 2 + max(0, n_turns - MANA_CAP) * MANA_CAP
    return base + n_coin


def _active_player(turn, start_player):
    """Which engine player ('P1'/'P2') is active on `turn`, alternating from start_player."""
    # turn 1 == start_player, then alternate
    even = (turn - start_player) % 2 == 0
    return "P1" if even else "P2"


def parse_log(path, start_player):
    """Parse one verbose log -> dict of P1 (me-side) metrics. start_player is 1 or 2."""
    hero_id = {}            # 'P1'/'P2' -> hero entity id
    owner = {}              # entity id -> 'P1'/'P2'
    in_hand = set()         # entity ids currently in *some* hand
    p1_hand = set()         # P1-owned entity ids currently in hand
    p1_board = set()        # P1-owned minion ids currently in PLAY
    p2_board = set()        # P2-owned minion ids currently in PLAY

    cur_turn = 0
    my_turns = 0            # count of P1-active turns reached so far  (== my-turn index)
    seen_my_turn = set()    # which engine TURN numbers were P1's

    plays = []              # (my_turn, ctype, entity_id) for P1 non-coin plays
    pay = {}                # entity id -> mana paid  (PayPhase is logged *after* the play)
    n_minion = 0
    n_coin = 0
    hp_uses = 0
    taken_dmg = 0           # damage dealt to my hero
    enemy_killed = 0        # P2 minions sent to graveyard on P1's turns
    atk_face = atk_minion = 0   # P1 attacks at enemy hero vs at minions
    hand_samples = []       # P1 hand size at end of each P1 turn
    board_samples = []      # P1 minions in PLAY at end of each P1 turn

    other = "P1" if start_player == 2 else "P2"   # the player who goes 2nd gets the Coin

    def active():
        return _active_player(cur_turn, start_player)

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if " set data TURN to " in line:
                m = RE_TURN.search(line)
                if m:
                    new_turn = int(m.group(1))
                    if new_turn != cur_turn:
                        # close out the turn we are leaving: if it was P1's, sample hand+board
                        if cur_turn and active() == "P1":
                            hand_samples.append(len(p1_hand))
                            board_samples.append(len(p1_board))
                        cur_turn = new_turn
                        if active() == "P1" and cur_turn not in seen_my_turn:
                            seen_my_turn.add(cur_turn); my_turns += 1
                continue

            if "HERO_ENTITY" in line:
                m = RE_HEROENT.search(line)
                if m:
                    pl = "P1" if m.group(1) == "2" else "P2"   # Player[2]=P1, Player[3]=P2
                    hero_id[pl] = int(m.group(2))
                continue

            if "DrawPhase:" in line:
                m = RE_DRAW.search(line)
                if m:
                    pl, name, eid = m.group(1), m.group(2), int(m.group(3))
                    owner[eid] = pl
                continue

            if "SummonPhase" in line:
                m = RE_SUMMON.search(line)
                if m:
                    eid, pl = int(m.group(2)), m.group(3)
                    owner.setdefault(eid, pl)
                    (p1_board if pl == "P1" else p2_board).add(eid)
                continue

            if "added to zone" in line:
                m = RE_ZONE.search(line)
                if m:
                    name, eid, ctype, zone = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                    if zone == "HAND":
                        if eid not in owner:
                            # generated card / Coin: Coin -> 2nd player, else active player
                            owner[eid] = other if name.strip() == "The Coin" else active()
                        in_hand.add(eid)
                        if owner.get(eid) == "P1":
                            p1_hand.add(eid)
                    else:
                        in_hand.discard(eid)
                        p1_hand.discard(eid)
                    if zone != "PLAY":
                        # the minion left the board (died / bounced / transformed)
                        was_enemy_minion = eid in p2_board
                        p1_board.discard(eid)
                        p2_board.discard(eid)
                        if (zone == "GRAVEYARD" and ctype == "MINION"
                                and was_enemy_minion and active() == "P1"):
                            enemy_killed += 1
                continue

            if "PayPhase:" in line:
                m = RE_PAY.search(line)
                if m:
                    pay[int(m.group(2))] = int(m.group(3))
                continue

            if "PlayCardTask" in line:
                m = RE_PLAY.search(line)
                if m:
                    pl, name, eid, ctype = m.group(1), m.group(2).strip(), int(m.group(3)), m.group(4)
                    if pl == "P1":
                        is_coin = (name == "The Coin")
                        mt = my_turns if my_turns else 1
                        if is_coin:
                            n_coin += 1
                        else:
                            plays.append((mt, ctype, eid))
                            if ctype == "MINION": n_minion += 1
                continue

            if "HeroPowerTask" in line:
                m = RE_HPOW.search(line)
                if m and m.group(1) == "P1":
                    hp_uses += 1
                continue

            if "took damage for" in line:
                m = RE_DMG.search(line)
                if m:
                    eid, dmg = int(m.group(2)), int(m.group(3))
                    if eid == hero_id.get("P1"): taken_dmg += dmg
                continue

            if "[AttackPhase]" in line and "attacked" in line:
                m = RE_ATK.search(line)
                if m and active() == "P1":   # attacks happen on the attacker's own turn
                    def_id = int(m.group(4))
                    if def_id == hero_id.get("P2"): atk_face += 1
                    else: atk_minion += 1
                continue

    # close the final P1 turn if the game ended on it
    if cur_turn and active() == "P1":
        hand_samples.append(len(p1_hand))
        board_samples.append(len(p1_board))

    # resolve each P1 play's mana cost (PayPhase, keyed by entity id, was logged after the play)
    play_costs = [(mt, ctype, pay.get(eid, np.nan)) for (mt, ctype, eid) in plays]
    costs = [c for _, _, c in play_costs if not np.isnan(c)]
    nt = my_turns if my_turns else np.nan
    noncoin = len(plays)
    mana_spent = float(np.nansum(costs)) if costs else 0.0
    avail = mana_available(my_turns, n_coin) if my_turns else np.nan
    atk_total = atk_face + atk_minion

    return dict(
        # --- the 11 calculated metrics (METRICS), in order ---
        n_my_turns=my_turns,
        avg_cards_in_hand=float(np.mean(hand_samples)) if hand_samples else np.nan,
        mana_eff=mana_spent / avail if (avail and not np.isnan(avail)) else np.nan,
        avg_card_cost=float(np.mean(costs)) if costs else np.nan,
        minion_fraction=n_minion / noncoin if noncoin else np.nan,
        face_attack_ratio=atk_face / atk_total if atk_total else np.nan,
        attacks_per_turn=atk_total / nt if nt else np.nan,
        enemy_minions_killed_per_turn=enemy_killed / nt if nt else np.nan,
        avg_board_minions=float(np.mean(board_samples)) if board_samples else np.nan,
        taken_dmg_per_turn=taken_dmg / nt if nt else np.nan,
        hp_per_turn=hp_uses / nt if nt else np.nan,
        # --- internal raw data for cards_df / turns_df (not metrics) ---
        _plays=play_costs, _hand_samples=hand_samples,
    )


def build_frames(cache=None, verbose=True, deck="AggroPirateWarrior"):
    """Parse every log under log/<style>_<deck>, joining summary.csv for header facts.

    `deck` selects which family of folders to read (e.g. "AggroPirateWarrior"
    or "RenoKazakusMage"); each of the 5 STYLES drives the *same* deck.
    """
    if cache and os.path.exists(cache):
        d = pd.read_pickle(cache)
        return d["games"], d["cards"], d["turns"]

    game_rows, card_rows, turn_rows = [], [], []
    for style in STYLES:
        folder = os.path.join(LOG_ROOT, f"{style}_{deck}")
        summ = os.path.join(folder, "summary.csv")
        meta = {}
        with open(summ, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                meta[r["log_file"]] = r
        files = sorted(glob.glob(os.path.join(folder, "*.log")))
        ok = 0
        for path in files:
            fn = os.path.basename(path)
            row = meta.get(fn)
            if row is None:
                continue
            start_player = int(row["start_player"])
            try:
                m = parse_log(path, start_player)
            except Exception as e:
                continue
            if not m["n_my_turns"]:
                continue
            ok += 1
            duration = float(row["seconds"]); turns = int(row["turns"])
            game_rows.append(dict(
                style=style, game_id=int(row["game"]), log_file=fn,
                opp_style=row["p2_agent"], opp_deck=row["p2_deck"], opp_class=row["p2_class"],
                win=(row["winner"] == "P1"), start_player=start_player,
                duration=duration, turns=turns,
                time_per_turn=duration / turns if turns else np.nan,
                **{k: v for k, v in m.items() if not k.startswith("_")},
            ))
            for (mt, ctype, cost) in m["_plays"]:
                if not (isinstance(cost, float) and np.isnan(cost)):
                    card_rows.append((style, ctype, cost))
            for i, h in enumerate(m["_hand_samples"], 1):
                turn_rows.append((style, int(row["game"]), i, h))
        if verbose:
            print(f"  {style:9s}: parsed {ok}/{len(files)} logs")

    games_df = pd.DataFrame(game_rows)
    cards_df = pd.DataFrame(card_rows, columns=["style", "ctype", "mana"])
    turns_df = pd.DataFrame(turn_rows, columns=["style", "game_id", "my_turn", "hand_end"])
    if cache:
        pd.to_pickle({"games": games_df, "cards": cards_df, "turns": turns_df}, cache)
    return games_df, cards_df, turns_df


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 200, "display.max_columns", 40)
    g, c, t = build_frames(cache=None)
    print(f"\ngames_df {g.shape}  cards_df {c.shape}  turns_df {t.shape}")
    print("\nper-style game counts:\n", g["style"].value_counts())
    print("\nper-style means:\n", g.groupby("style")[METRICS].mean().round(2))
