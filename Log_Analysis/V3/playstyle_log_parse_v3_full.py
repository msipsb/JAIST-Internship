"""
Parse SabberStone verbose game logs into per-game, me-side (P1) play-style metrics.

**Full V3 = Change B (trajectory) + Change A (raw tell metrics).**  This is the
complete V3 parser: it carries everything in ``playstyle_log_parse_v3.py`` (the
11 V2 averages + the 13 Change-B trajectory features) and adds the Change-A
per-style *tell* metrics, which DO need extra log lines (draws, mana crystals)
or new derivations.

Change A metrics added to ``games_df`` (target style in brackets):
    extra_mana_crystals   [ramp]      mana gained faster than +1/turn (RESOURCES deltas)
    max_card_cost         [ramp]      highest mana cost card P1 played
    proactive_ratio       [control]   minions played / (minions + enemy minions killed)
    cards_drawn_per_turn  [fatigue]   draws beyond opening hand, per turn (~1.0 natural)
    cards_left_in_deck    [fatigue]   30 - cards P1 drew (deck remaining at game end)
    first_minion_turn     [midrange]  my-turn the first P1 minion is played

Notes from grepping the RenoKazakusMage logs (the build check the plan asked for):
  * Draw lines exist: ``DrawPhase: P1 draws 'Card[id]'``.  Opening hand = 3 draws
    if P1 goes first, 4 if second (then 1 natural draw per P1 turn); card-draw
    spells are the excess.  We subtract the opening hand so the baseline is ~1.0.
  * Mana-crystal lines exist: ``'Player[2]' set data RESOURCES to N`` (Player[2]=P1).
    BUT on RenoKazakusMage RESOURCES increments exactly 1,2,3,... per P1 turn —
    this highlander Mage has **no ramp cards**, so ``extra_mana_crystals`` is ~0
    for every style.  It is computed for completeness but is expected to be a dead
    feature here (excluded from the notebook's clustering FEATURES).
  * Deck lines exist (``added to zone 'DECK'``) but carry no owner, so
    ``cards_left_in_deck`` uses the robust proxy ``30 - P1 draws`` (mill ~ 0).
    As the plan flags, fatigue stays the hardest split: deck rarely truly decks
    out by ~turn 9, so this metric has limited spread.

Returns three tidy frames:
    games_df : one row per game (11 V2 + 13 trajectory + 6 Change-A columns)
    cards_df : one row per card P1 played   (style, ctype, mana)
    turns_df : one row per P1 turn           (style, game_id, my_turn, hand_end,
                                              board_end, mana_spent)
"""
import os, re, glob, csv, collections
import numpy as np
import pandas as pd

# this file lives in <repo>/Log_Analysis/V3/; the log/ data and cache .pkl live at
# the repo root (two levels up).  Anchor paths there regardless of the kernel's CWD.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOG_ROOT = os.path.join(BASE_DIR, "log")
STYLES   = ["aggro", "control", "fatigue", "midrange", "ramp"]
MANA_CAP = 10
DECK_SIZE = 30           # RenoKazakusMage is a 30-card highlander list

# my-turn checkpoints at which we sample the trajectory features
CHECKPOINTS = [3, 5, 7, 9]

# the 11 whole-game V2 metrics (kept verbatim — they carry the aggro-vs-value split)
METRICS = [
    "n_my_turns", "avg_cards_in_hand", "mana_eff", "avg_card_cost",
    "minion_fraction", "face_attack_ratio", "attacks_per_turn",
    "enemy_minions_killed_per_turn", "avg_board_minions",
    "taken_dmg_per_turn", "hp_per_turn",
]

# Change-B trajectory metrics
TRAJ_METRICS = (
    [f"board_at_t{t}" for t in CHECKPOINTS]
    + [f"mana_at_t{t}" for t in CHECKPOINTS]
    + ["mana_slope"]
    + [f"hand_at_t{t}" for t in CHECKPOINTS]
)

# Change-A raw "tell" metrics (new in the full V3)
A_METRICS = [
    "extra_mana_crystals", "max_card_cost", "proactive_ratio",
    "cards_drawn_per_turn", "cards_left_in_deck", "first_minion_turn",
]

# extra_mana_crystals is identically ~0 on RenoKazakusMage (no ramp cards), so the
# notebook clusters on A_FEATURES, not all of A_METRICS.  See module docstring.
A_DEAD_ON_DECK = ["extra_mana_crystals"]
A_FEATURES = [m for m in A_METRICS if m not in A_DEAD_ON_DECK]

# everything used as a clustering feature in the full V3
ALL_METRICS  = METRICS + TRAJ_METRICS + A_METRICS
ALL_FEATURES = METRICS + TRAJ_METRICS + A_FEATURES

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
RE_RES    = re.compile(r"'Player\[(\d+)\]' set data RESOURCES to (\d+)")


def mana_available(n_turns, n_coin):
    """Total mana a player could have spent over n_turns (+1 per Coin) -- same as reference."""
    full = min(n_turns, MANA_CAP)
    base = full * (full + 1) // 2 + max(0, n_turns - MANA_CAP) * MANA_CAP
    return base + n_coin


def _active_player(turn, start_player):
    """Which engine player ('P1'/'P2') is active on `turn`, alternating from start_player."""
    even = (turn - start_player) % 2 == 0
    return "P1" if even else "P2"


def _at(samples, k):
    """Value of a 1-based per-turn sample series at my-turn k, or NaN if the game was shorter."""
    return float(samples[k - 1]) if len(samples) >= k else np.nan


def parse_log(path, start_player):
    """Parse one verbose log -> dict of P1 (me-side) metrics. start_player is 1 or 2."""
    hero_id = {}            # 'P1'/'P2' -> hero entity id
    owner = {}              # entity id -> 'P1'/'P2'
    in_hand = set()
    p1_hand = set()
    p1_board = set()
    p2_board = set()

    cur_turn = 0
    my_turns = 0
    seen_my_turn = set()

    plays = []
    pay = {}
    n_minion = 0
    n_coin = 0
    hp_uses = 0
    taken_dmg = 0
    enemy_killed = 0
    atk_face = atk_minion = 0
    hand_samples = []
    board_samples = []
    p1_draws_total = 0          # Change A: every DrawPhase 'P1 draws ...' (incl. opening hand)
    p1_res_seq = []             # Change A: monotonic P1 RESOURCES (mana crystal) readings

    other = "P1" if start_player == 2 else "P2"

    def active():
        return _active_player(cur_turn, start_player)

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if " set data TURN to " in line:
                m = RE_TURN.search(line)
                if m:
                    new_turn = int(m.group(1))
                    if new_turn != cur_turn:
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
                    pl = "P1" if m.group(1) == "2" else "P2"
                    hero_id[pl] = int(m.group(2))
                continue

            if "set data RESOURCES to" in line:          # Change A: ramp / mana crystals
                m = RE_RES.search(line)
                if m and m.group(1) == "2":               # Player[2] == P1
                    v = int(m.group(2))
                    if not p1_res_seq or v > p1_res_seq[-1]:
                        p1_res_seq.append(v)
                continue

            if "DrawPhase:" in line:
                m = RE_DRAW.search(line)
                if m:
                    pl, name, eid = m.group(1), m.group(2), int(m.group(3))
                    owner[eid] = pl
                    if pl == "P1":
                        p1_draws_total += 1               # Change A: fatigue / card economy
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
                            owner[eid] = other if name.strip() == "The Coin" else active()
                        in_hand.add(eid)
                        if owner.get(eid) == "P1":
                            p1_hand.add(eid)
                    else:
                        in_hand.discard(eid)
                        p1_hand.discard(eid)
                    if zone != "PLAY":
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
                if m and active() == "P1":
                    def_id = int(m.group(4))
                    if def_id == hero_id.get("P2"): atk_face += 1
                    else: atk_minion += 1
                continue

    if cur_turn and active() == "P1":
        hand_samples.append(len(p1_hand))
        board_samples.append(len(p1_board))

    play_costs = [(mt, ctype, pay.get(eid, np.nan)) for (mt, ctype, eid) in plays]
    costs = [c for _, _, c in play_costs if not np.isnan(c)]
    nt = my_turns if my_turns else np.nan
    noncoin = len(plays)
    mana_spent = float(np.nansum(costs)) if costs else 0.0
    avail = mana_available(my_turns, n_coin) if my_turns else np.nan
    atk_total = atk_face + atk_minion

    # --- Change B: per-turn mana spent + checkpoint / slope trajectory features ---
    mana_by_turn = collections.defaultdict(float)
    for (mt, ctype, c) in play_costs:
        if not (isinstance(c, float) and np.isnan(c)):
            mana_by_turn[mt] += c
    mana_series = [mana_by_turn.get(t, 0.0) for t in range(1, my_turns + 1)]
    if my_turns >= 2:
        mana_slope = float(np.polyfit(np.arange(1, my_turns + 1), mana_series, 1)[0])
    else:
        mana_slope = np.nan

    def mana_at(k):
        return float(mana_by_turn.get(k, 0.0)) if k <= my_turns else np.nan

    traj = {}
    for t in CHECKPOINTS:
        traj[f"board_at_t{t}"] = _at(board_samples, t)
        traj[f"mana_at_t{t}"]  = mana_at(t)
        traj[f"hand_at_t{t}"]  = _at(hand_samples, t)
    traj["mana_slope"] = mana_slope

    # --- Change A: per-style tell metrics ---
    # extra mana crystals: how far P1's crystal count ran ahead of the natural +1/turn
    extra_crystals = max([0] + [v - (i + 1) for i, v in enumerate(p1_res_seq)])
    # fatigue / card economy: draws beyond the opening hand, per turn (~1.0 = natural)
    opening_hand = 3 if start_player == 1 else 4
    cards_drawn_per_turn = (p1_draws_total - opening_hand) / nt if nt else np.nan
    cards_left_in_deck = max(0, DECK_SIZE - p1_draws_total)
    # control vs proactive board-building
    proactive_den = n_minion + enemy_killed
    proactive_ratio = n_minion / proactive_den if proactive_den else np.nan
    # midrange: when the first minion hits the board
    minion_turns = [mt for (mt, ctype, _eid) in plays if ctype == "MINION"]
    first_minion_turn = float(min(minion_turns)) if minion_turns else np.nan

    a = dict(
        extra_mana_crystals=float(extra_crystals),
        max_card_cost=float(max(costs)) if costs else np.nan,
        proactive_ratio=proactive_ratio,
        cards_drawn_per_turn=cards_drawn_per_turn,
        cards_left_in_deck=float(cards_left_in_deck),
        first_minion_turn=first_minion_turn,
    )

    return dict(
        # --- the 11 V2 metrics (METRICS), in order ---
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
        # --- Change-B trajectory metrics ---
        **traj,
        # --- Change-A tell metrics ---
        **a,
        # --- internal raw data for cards_df / turns_df (not metrics) ---
        _plays=play_costs, _hand_samples=hand_samples,
        _board_samples=board_samples, _mana_series=mana_series,
    )


def build_frames(cache=None, verbose=True, deck="RenoKazakusMage"):
    """Parse every log under log/<style>_<deck>, joining summary.csv for header facts."""
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
            except Exception:
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
            hs, bs, ms = m["_hand_samples"], m["_board_samples"], m["_mana_series"]
            for i in range(1, len(hs) + 1):
                b = bs[i - 1] if i <= len(bs) else np.nan
                mn = ms[i - 1] if i <= len(ms) else np.nan
                turn_rows.append((style, int(row["game"]), i, hs[i - 1], b, mn))
        if verbose:
            print(f"  {style:9s}: parsed {ok}/{len(files)} logs")

    games_df = pd.DataFrame(game_rows)
    cards_df = pd.DataFrame(card_rows, columns=["style", "ctype", "mana"])
    turns_df = pd.DataFrame(turn_rows,
                            columns=["style", "game_id", "my_turn", "hand_end", "board_end", "mana_spent"])
    if cache:
        pd.to_pickle({"games": games_df, "cards": cards_df, "turns": turns_df}, cache)
    return games_df, cards_df, turns_df


if __name__ == "__main__":
    pd.set_option("display.width", 240, "display.max_columns", 80)
    g, c, t = build_frames(cache=None)
    print(f"\ngames_df {g.shape}  cards_df {c.shape}  turns_df {t.shape}")
    print("\nper-style V2 means:\n", g.groupby("style")[METRICS].mean().round(2))
    print("\nper-style trajectory means:\n", g.groupby("style")[TRAJ_METRICS].mean().round(2))
    print("\nper-style Change-A means:\n", g.groupby("style")[A_METRICS].mean().round(3))
