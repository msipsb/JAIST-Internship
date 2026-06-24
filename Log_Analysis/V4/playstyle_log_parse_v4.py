"""
V4 — **deck-agnostic, turn-rate-normalized** play-style metrics.

V4 builds on the full V3 parser (V2 averages + Change-B trajectory + Change-A
tells) with two goals, so the same feature set can fingerprint the AI's
*philosophy* on **any** deck — RenoKazakusMage, AggroPirateWarrior, … — not just
the one it was tuned on:

  1. **Universal turn-rate normalization (the single most important change).**
     Raw counts encode *how long the game lasted* and *what the deck contains*,
     not how the AI thinks. Every count is re-expressed as a per-turn rate or a
     contextual fraction:
         avg_cards_in_hand   -> hand_fill_ratio     = hand / 10
         cards_left_in_deck  -> cards_left_frac     = (30 - draws) / 30
         first_minion_turn   -> first_minion_frac   = first_minion_turn / my_turns
         mana_at_t{k}        -> mana_eff_t{k}        = mana spent that turn / crystals
         hand_at_t{k}        -> hand_frac_t{k}       = hand / 10
     (The already-normalized V2 ratios — mana_eff, face_attack_ratio, *_per_turn —
     are kept verbatim; the raw columns are still emitted for the notebook grids
     and for within-deck comparison to V3.)

  2. **Four universal "currency" metrics** (Mana / Cards / Board / Life), each a
     deck-independent behavioral ratio. Two are computed directly; two use
     proxies because the logs carry no minion ATK stat and no damage-source
     attribution (see notes):
         face_attack_ratio       [aggro]    damage *commitment* to the enemy hero
                                            (kept from V2: face attacks / all attacks;
                                            the deck-agnostic proxy for "Face Damage
                                            Commitment Ratio"). Plus face_dmg_per_turn
                                            = damage dealt to the enemy hero / turn.
         mana_floated_per_turn   [control]  unspent crystals / turn (option-preservation /
                                            greed; proxy for "Playable Card Retention")
         avg_enemy_board_minions [fatigue]  enemy minions left alive at the end of my
                                            turns (threat tolerance; proxy for "Enemy
                                            Board Lethality Margin" — no ATK stat in logs)
         value_turn_fraction     [ramp]     fraction of my turns that drew an extra card
                                            (future-value investment velocity, via
                                            DrawPhase counts > the natural 1/turn)

Two feature groups drive the cross-deck study (see measure_v4_gain.py):
    AGNOSTIC_FEATURES — pure behavioral ratios; meant to transfer across decks.
    DECK_DEP_FEATURES — count/cost columns that leak deck identity (avg_card_cost,
                        max_card_cost, raw mana_at_t, game length). Kept for
                        reference and within-deck use, *excluded* from the
                        deck-agnostic feature set.

Notes from grepping the logs (both deck families):
  * Player[2] == P1 (me), Player[3] == P2. HERO_ENTITY maps each player's hero id,
    so "took damage for" against P1's enemy-hero id == face damage P1 dealt.
  * RESOURCES is the crystal count; on these (ramp-less) lists it tracks 1,2,3,…
    per P1 turn, so floated = crystals(turn) - mana spent(turn).
  * DrawPhase is owner-tagged ("P1 draws …"); >1 draw on a turn (after turn 1)
    means a card-draw spell was played -> a "value" turn.

Returns three tidy frames (build_frames tags each game row with its ``deck``):
    games_df : one row per game  (11 V2 + 13 trajectory + 6 Change-A + V4 columns)
    cards_df : one row per card P1 played   (style, ctype, mana)
    turns_df : one row per P1 turn           (style, game_id, my_turn, hand_end,
                                              board_end, mana_spent)
"""
import os, re, glob, csv, collections
import numpy as np
import pandas as pd

# this file lives in <repo>/Log_Analysis/V4/; the log/ data and cache .pkl live at
# the repo root (two levels up).  Anchor paths there regardless of the kernel's CWD.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOG_ROOT = os.path.join(BASE_DIR, "log")
STYLES   = ["aggro", "control", "fatigue", "midrange", "ramp"]
DECKS    = ["RenoKazakusMage", "AggroPirateWarrior"]   # the two simulated deck families
MANA_CAP = 10
DECK_SIZE = 30           # both lists are 30 cards
HAND_CAP  = 10

# my-turn checkpoints at which we sample the trajectory features
CHECKPOINTS = [3, 5, 7, 9]

# ---- the 11 whole-game V2 metrics (kept verbatim) ---------------------------
METRICS = [
    "n_my_turns", "avg_cards_in_hand", "mana_eff", "avg_card_cost",
    "minion_fraction", "face_attack_ratio", "attacks_per_turn",
    "enemy_minions_killed_per_turn", "avg_board_minions",
    "taken_dmg_per_turn", "hp_per_turn",
]

# ---- Change-B trajectory metrics (raw) --------------------------------------
TRAJ_METRICS = (
    [f"board_at_t{t}" for t in CHECKPOINTS]
    + [f"mana_at_t{t}" for t in CHECKPOINTS]
    + ["mana_slope"]
    + [f"hand_at_t{t}" for t in CHECKPOINTS]
)

# ---- Change-A raw "tell" metrics --------------------------------------------
A_METRICS = [
    "extra_mana_crystals", "max_card_cost", "proactive_ratio",
    "cards_drawn_per_turn", "cards_left_in_deck", "first_minion_turn",
]
A_DEAD_ON_DECK = ["extra_mana_crystals"]      # identically ~0 on ramp-less lists
A_FEATURES = [m for m in A_METRICS if m not in A_DEAD_ON_DECK]

# ---- V4: turn-rate-normalized re-expressions of the count metrics ------------
NORM_METRICS = [
    "hand_fill_ratio",          # avg_cards_in_hand / 10
    "cards_left_frac",          # cards_left_in_deck / 30
    "first_minion_frac",        # first_minion_turn / my_turns
    "max_card_cost_norm",       # max_card_cost / 10   (still deck-dependent -> DECK_DEP)
] + [f"mana_eff_t{t}" for t in CHECKPOINTS] \
  + [f"hand_frac_t{t}" for t in CHECKPOINTS]

# ---- V4: the four universal "currency" behavioral metrics --------------------
UNIVERSAL_METRICS = [
    "face_dmg_per_turn",        # life:  damage dealt to enemy hero / turn
    "mana_floated_per_turn",    # mana:  unspent crystals / turn (greed / retention)
    "avg_enemy_board_minions",  # board: enemy minions persisted at my turn-end (tolerance)
    "value_turn_fraction",      # cards: fraction of my turns that drew an extra card
]

ALL_V4 = NORM_METRICS + UNIVERSAL_METRICS

# ---- feature groups for the cross-deck study --------------------------------
# behavioral ratios meant to transfer across decks (the deck-agnostic fingerprint)
AGNOSTIC_FEATURES = [
    # already-normalized V2 ratios
    "mana_eff", "minion_fraction", "face_attack_ratio", "attacks_per_turn",
    "enemy_minions_killed_per_turn", "taken_dmg_per_turn", "hp_per_turn",
    "avg_board_minions",
    # Change-A behavioral ratios (cost-free)
    "proactive_ratio", "cards_drawn_per_turn",
    # V4 normalized re-expressions
    "hand_fill_ratio", "cards_left_frac", "first_minion_frac",
    # V4 universal currency metrics
    "face_dmg_per_turn", "mana_floated_per_turn",
    "avg_enemy_board_minions", "value_turn_fraction",
    # normalized trajectory (board count is ~deck-agnostic; mana/hand normalized)
    *[f"board_at_t{t}" for t in CHECKPOINTS],
    *[f"mana_eff_t{t}" for t in CHECKPOINTS],
    *[f"hand_frac_t{t}" for t in CHECKPOINTS],
]

# count/cost columns that leak deck identity -> excluded from the agnostic set
DECK_DEP_FEATURES = [
    "n_my_turns", "avg_cards_in_hand", "avg_card_cost",
    "max_card_cost", "max_card_cost_norm", "cards_left_in_deck",
    "first_minion_turn",
    *[f"mana_at_t{t}" for t in CHECKPOINTS],
    "mana_slope",
]

# the purest cross-deck set: the subset of AGNOSTIC_FEATURES that are *bounded,
# dimensionless* ratios (~0..1). Unlike per-turn rates (face_dmg_per_turn,
# attacks_per_turn) or absolute counts (avg_*_board_minions), a ratio does not
# shift in magnitude when the deck's damage/curve scale changes, so a scaler fit
# on one deck still lines up on the other.
RATIO_FEATURES = [
    "mana_eff", "minion_fraction", "face_attack_ratio", "proactive_ratio",
    "cards_drawn_per_turn",                       # normalized around the natural ~1 draw/turn
    "hand_fill_ratio", "cards_left_frac", "first_minion_frac", "value_turn_fraction",
    *[f"mana_eff_t{t}" for t in CHECKPOINTS],
    *[f"hand_frac_t{t}" for t in CHECKPOINTS],
]

# everything emitted as a numeric metric column
ALL_METRICS = METRICS + TRAJ_METRICS + A_METRICS + ALL_V4

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
    enemy_board_samples = []     # V4: enemy minions alive at the end of each P1 turn
    p1_draws_total = 0           # Change A: every DrawPhase 'P1 draws ...' (incl. opening hand)
    p1_res_seq = []              # Change A: monotonic P1 RESOURCES (mana crystal) readings
    p2_face_dmg = 0              # V4: damage dealt to the enemy (P2) hero
    crystals_by_turn = {}        # V4: P1 crystal count seen during my-turn k
    draws_by_turn = collections.defaultdict(int)   # V4: P1 DrawPhase events per my-turn

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
                            enemy_board_samples.append(len(p2_board))
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
                    crystals_by_turn[my_turns if my_turns else 1] = v   # V4
                continue

            if "DrawPhase:" in line:
                m = RE_DRAW.search(line)
                if m:
                    pl, name, eid = m.group(1), m.group(2), int(m.group(3))
                    owner[eid] = pl
                    if pl == "P1":
                        p1_draws_total += 1               # Change A: fatigue / card economy
                        draws_by_turn[my_turns if my_turns else 1] += 1   # V4
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
                    elif eid == hero_id.get("P2"): p2_face_dmg += dmg   # V4: face damage dealt
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
        enemy_board_samples.append(len(p2_board))

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
    extra_crystals = max([0] + [v - (i + 1) for i, v in enumerate(p1_res_seq)])
    opening_hand = 3 if start_player == 1 else 4
    cards_drawn_per_turn = (p1_draws_total - opening_hand) / nt if nt else np.nan
    cards_left_in_deck = max(0, DECK_SIZE - p1_draws_total)
    proactive_den = n_minion + enemy_killed
    proactive_ratio = n_minion / proactive_den if proactive_den else np.nan
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

    # --- V4: universal currency metrics (turn-rate / contextual) -------------
    # life: damage dealt to the enemy hero per my-turn (aggression / face commitment)
    face_dmg_per_turn = p2_face_dmg / nt if nt else np.nan
    # mana: unspent crystals per turn (option-preservation / greed) -- proxy for retention
    floated = [max(0.0, crystals_by_turn.get(t, np.nan) - mana_by_turn.get(t, 0.0))
               for t in range(1, my_turns + 1) if t in crystals_by_turn]
    mana_floated_per_turn = float(np.mean(floated)) if floated else np.nan
    # board: enemy minions left alive at the end of my turns (threat tolerance)
    avg_enemy_board_minions = float(np.mean(enemy_board_samples)) if enemy_board_samples else np.nan
    # cards: fraction of my turns (after turn 1) that drew an extra card (value velocity)
    value_turns = sum(1 for t in range(2, my_turns + 1) if draws_by_turn.get(t, 0) >= 2)
    value_turn_fraction = value_turns / (my_turns - 1) if my_turns > 1 else np.nan

    # --- V4: turn-rate-normalized re-expressions of the count metrics --------
    avg_hand = float(np.mean(hand_samples)) if hand_samples else np.nan
    norm = dict(
        hand_fill_ratio=avg_hand / HAND_CAP if not np.isnan(avg_hand) else np.nan,
        cards_left_frac=cards_left_in_deck / DECK_SIZE,
        first_minion_frac=first_minion_turn / nt if (nt and not np.isnan(first_minion_turn)) else np.nan,
        max_card_cost_norm=(float(max(costs)) / MANA_CAP) if costs else np.nan,
    )
    for t in CHECKPOINTS:
        mt = mana_at(t)
        norm[f"mana_eff_t{t}"] = (mt / min(t, MANA_CAP)) if not np.isnan(mt) else np.nan
        h = _at(hand_samples, t)
        norm[f"hand_frac_t{t}"] = (h / HAND_CAP) if not np.isnan(h) else np.nan

    universal = dict(
        face_dmg_per_turn=face_dmg_per_turn,
        mana_floated_per_turn=mana_floated_per_turn,
        avg_enemy_board_minions=avg_enemy_board_minions,
        value_turn_fraction=value_turn_fraction,
    )

    return dict(
        # --- the 11 V2 metrics (METRICS), in order ---
        n_my_turns=my_turns,
        avg_cards_in_hand=avg_hand,
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
        # --- V4 normalized + universal metrics ---
        **norm,
        **universal,
        # --- internal raw data for cards_df / turns_df (not metrics) ---
        _plays=play_costs, _hand_samples=hand_samples,
        _board_samples=board_samples, _mana_series=mana_series,
    )


def build_frames(cache=None, verbose=True, deck="RenoKazakusMage"):
    """Parse every log under log/<style>_<deck>, joining summary.csv for header facts.

    `deck` may be a single deck name or a list of deck names; each game row is
    tagged with its ``deck`` so the frame can hold both families for the
    cross-deck study.
    """
    if cache and os.path.exists(cache):
        d = pd.read_pickle(cache)
        return d["games"], d["cards"], d["turns"]

    decks = [deck] if isinstance(deck, str) else list(deck)
    game_rows, card_rows, turn_rows = [], [], []
    for dk in decks:
        for style in STYLES:
            folder = os.path.join(LOG_ROOT, f"{style}_{dk}")
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
                    deck=dk, style=style, game_id=int(row["game"]), log_file=fn,
                    opp_style=row["p2_agent"], opp_deck=row["p2_deck"], opp_class=row["p2_class"],
                    win=(row["winner"] == "P1"), start_player=start_player,
                    duration=duration, turns=turns,
                    time_per_turn=duration / turns if turns else np.nan,
                    **{k: v for k, v in m.items() if not k.startswith("_")},
                ))
                for (mt, ctype, cost) in m["_plays"]:
                    if not (isinstance(cost, float) and np.isnan(cost)):
                        card_rows.append((dk, style, ctype, cost))
                hs, bs, ms = m["_hand_samples"], m["_board_samples"], m["_mana_series"]
                for i in range(1, len(hs) + 1):
                    b = bs[i - 1] if i <= len(bs) else np.nan
                    mn = ms[i - 1] if i <= len(ms) else np.nan
                    turn_rows.append((dk, style, int(row["game"]), i, hs[i - 1], b, mn))
            if verbose:
                print(f"  {dk:18s} {style:9s}: parsed {ok}/{len(files)} logs")

    games_df = pd.DataFrame(game_rows)
    cards_df = pd.DataFrame(card_rows, columns=["deck", "style", "ctype", "mana"])
    turns_df = pd.DataFrame(turn_rows,
                            columns=["deck", "style", "game_id", "my_turn", "hand_end", "board_end", "mana_spent"])
    if cache:
        pd.to_pickle({"games": games_df, "cards": cards_df, "turns": turns_df}, cache)
    return games_df, cards_df, turns_df


if __name__ == "__main__":
    pd.set_option("display.width", 260, "display.max_columns", 100)
    g, c, t = build_frames(cache=None, deck=DECKS)
    print(f"\ngames_df {g.shape}  cards_df {c.shape}  turns_df {t.shape}")
    print("\nrows per deck/style:\n", g.groupby(["deck", "style"]).size())
    print("\nV4 universal metrics - per deck/style means:\n",
          g.groupby(["deck", "style"])[UNIVERSAL_METRICS].mean().round(3))
    print("\nV4 normalized metrics - per deck/style means:\n",
          g.groupby(["deck", "style"])[["hand_fill_ratio", "cards_left_frac", "first_minion_frac"]].mean().round(3))
