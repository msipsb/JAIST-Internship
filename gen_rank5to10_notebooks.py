# Generate the two rank-5-10 notebooks from the warrior_top20_rank012 pipeline,
# parametrized for (a) the most-played class and (b) the most-played deck, with
# players ranked by GAMES PLAYED (descending) instead of by ladder rank.
import json, os, uuid

with open(r"D:\test\rank5to10_selection.json") as fh:
    SEL = json.load(fh)

def cell_id():
    return uuid.uuid4().hex[:8]

def md(src):
    return {"cell_type": "markdown", "id": cell_id(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "id": cell_id(), "metadata": {},
            "execution_count": None, "outputs": [], "source": src}

# ---- shared code-cell bodies (placeholders __CACHE__/__DECK__/__GAMES_DESC__ filled per notebook) ----

SETUP = r'''import json, glob, os, pickle, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
%matplotlib inline

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
WARRIOR_RED = "#8c1c13"
sns.set_palette("rocket")

DATA_DIR  = r"D:\test\dataset\hearthstonemap"
FILE_GLOB = os.path.join(DATA_DIR, "201[0-9]-[0-9][0-9].json")   # the 16 '201x-xx.json' files
CACHE     = r"__CACHE__"                                          # pre-built rank-5-10 cache

COIN_ID   = "GAME_005"   # The Coin
HERO      = "Warrior"
MODE      = "ranked"
RANK_SET  = (5, 6, 7, 8, 9, 10)   # <-- only games recorded at rank 5..10 (inclusive)
DECK      = __DECK__              # None -> all archetypes (class study); else a single archetype (deck study)
MIN_GAMES = 16           # "more than 15"
TOP_N     = 20           # up to 20 players
MANA_CAP  = 10           # max mana crystals'''

BUILD = r'''def mana_available(n_turns, n_coin):
    # sum of min(t,10) for t=1..n_turns, +1 per Coin
    full = min(n_turns, MANA_CAP)
    base = full * (full + 1) // 2 + max(0, n_turns - MANA_CAP) * MANA_CAP
    return base + n_coin

def build_frames():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            d = pickle.load(fh)
        return d["games"], d["cards"], d["turns"]

    game_rows, card_rows, turn_rows = [], [], []
    for f in sorted(glob.glob(FILE_GLOB)):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for g in data["games"]:
            if (g.get("mode") != MODE or g.get("hero") != HERO
                    or g.get("rank") not in RANK_SET                 # <-- rank 5..10 filter
                    or (DECK is not None and g.get("hero_deck") != DECK)):  # <-- deck filter (deck study only)
                continue
            gid = g["id"]; uh = g["user_hash"]
            ch  = g.get("card_history") or []
            me  = [e for e in ch if e["player"] == "me"]
            noncoin = [e for e in me if e["card"].get("id") != COIN_ID]
            coin    = [e for e in me if e["card"].get("id") == COIN_ID]
            manas   = [e["card"]["mana"] for e in noncoin if e["card"].get("mana") is not None]
            n_turns = max((e["turn"] for e in me), default=0)
            rounds  = max((e["turn"] for e in ch), default=0)

            per_turn = collections.Counter()
            for e in noncoin:
                if e["card"].get("mana") is not None:
                    per_turn[e["turn"]] += e["card"]["mana"]

            spent = float(sum(manas)); avail = mana_available(n_turns, len(coin))
            game_rows.append(dict(
                user_hash=uh, game_id=gid, rank=g["rank"], win=(g.get("result") == "win"),
                has_coin=bool(g.get("coin")), hero_deck=g.get("hero_deck") or "Unknown",
                opp=g.get("opponent") or "Unknown", duration=g.get("duration"), rounds=rounds or np.nan,
                n_cards=(len(noncoin) if ch else np.nan), n_me_turns=n_turns,
                mana_spent=spent, mana_available=avail,
                mana_eff=(spent / avail if (ch and n_turns and avail) else np.nan),
                cards_per_turn=(len(noncoin) / n_turns if (ch and n_turns) else np.nan),
                mana_per_turn=(spent / n_turns if (ch and n_turns) else np.nan),
                time_per_turn=(g["duration"] / rounds if (g.get("duration") and rounds) else np.nan),
                first_turn=min((e["turn"] for e in noncoin), default=np.nan),
                coin_turn=(coin[0]["turn"] if coin else np.nan),
            ))
            for m in manas:
                card_rows.append((uh, m))
            for t in range(1, n_turns + 1):
                turn_rows.append((uh, gid, t, per_turn.get(t, 0)))
        print(f"  parsed {os.path.basename(f)}")

    games_df = pd.DataFrame(game_rows)
    cards_df = pd.DataFrame(card_rows, columns=["user_hash", "mana"])
    turns_df = pd.DataFrame(turn_rows, columns=["user_hash", "game_id", "turn", "mana_spent"])
    with open(CACHE, "wb") as fh:
        pickle.dump({"games": games_df, "cards": cards_df, "turns": turns_df}, fh)
    return games_df, cards_df, turns_df

games_df, cards_df, turns_df = build_frames()
print(f"\ngames_df: {games_df.shape} | cards_df: {cards_df.shape} | turns_df: {turns_df.shape}")
print(f"rank counts: {games_df['rank'].value_counts().sort_index().to_dict()}")
print(f"unique __GAMES_DESC__ players: {games_df.user_hash.nunique()}")
games_df.head(3)'''

SELECT = r'''agg = (games_df.groupby("user_hash")
       .agg(games=("rank", "size"), best=("rank", "min"), avg=("rank", "mean"),
            median=("rank", "median"), win_pct=("win", "mean"), coin_pct=("has_coin", "mean")))
agg["win_pct"]  *= 100
agg["coin_pct"] *= 100

qualified = agg[agg["games"] >= MIN_GAMES].copy()
# RANKED BY GAMES PLAYED (most first); ties broken by peak rank then avg rank.
top = (qualified.sort_values(["games", "best", "avg"], ascending=[False, True, True])
       .head(TOP_N).reset_index())
TOP = top["user_hash"].tolist()
N_PLAYERS = len(TOP)
print(f"Qualifying players (>= {MIN_GAMES} __GAMES_DESC__ games): {len(qualified)} of {len(agg)}")
print(f"Showing top {N_PLAYERS} by GAMES PLAYED (cap {TOP_N}).")

top_deck = (games_df[games_df.user_hash.isin(TOP)]
            .groupby("user_hash")["hero_deck"]
            .agg(lambda s: ", ".join(f"{d} x{c}" for d, c in s.value_counts().head(2).items())))
disp = top.copy()
disp.insert(0, "#", range(1, len(disp) + 1))
disp["player"] = disp["user_hash"].str.slice(0, 12) + "..."
disp["archetypes"] = disp["user_hash"].map(top_deck)
disp = disp[["#", "player", "games", "best", "avg", "median", "win_pct", "coin_pct", "archetypes"]]

(disp.style.hide(axis="index")
   .format({"avg": "{:.2f}", "median": "{:.0f}", "win_pct": "{:.1f}%", "coin_pct": "{:.1f}%"})
   .background_gradient(subset=["games"], cmap="rocket_r")
   .bar(subset=["games"], color="#d4a5a0")
   .set_caption("Top __GAMES_DESC__ players (by games played)"))'''

RENDER = r'''GAME_METRICS = {                      # label -> (column, kind)
    "Game length (s)":     ("duration",      "kde"),
    "Cards / game":        ("n_cards",       "kde"),
    "Cards / turn":        ("cards_per_turn", "kde"),
    "Mana efficiency":     ("mana_eff",      "clip"),
    "First turn to play":  ("first_turn",    "disc"),
    "Coin played on turn": ("coin_turn",     "disc"),
    "Time / turn (s)":     ("time_per_turn", "kde"),
    "Mana spent / turn":   ("mana_per_turn", "kde"),
}
DESCRIBE_COLS = ["duration", "rounds", "n_cards", "cards_per_turn", "mana_eff",
                 "first_turn", "coin_turn", "time_per_turn", "mana_per_turn"]
NICE = {"duration": "Game length (s)", "rounds": "Game length (turns)", "n_cards": "Cards / game",
        "cards_per_turn": "Cards / turn", "mana_eff": "Mana efficiency", "first_turn": "First turn",
        "coin_turn": "Coin turn", "time_per_turn": "Time / turn (s)", "mana_per_turn": "Mana spent / turn"}

def show_player(pos):
    if pos > N_PLAYERS:
        display(Markdown(f"*(Only {N_PLAYERS} players qualified - no player #{pos}.)*")); return
    u   = TOP[pos - 1]
    gp  = games_df[games_df.user_hash == u]
    cp  = cards_df[cards_df.user_hash == u]
    tp  = turns_df[(turns_df.user_hash == u) & (turns_df.turn <= 15)]
    r   = agg.loc[u]
    decks = ", ".join(f"{d} ({c})" for d, c in gp.hero_deck.value_counts().head(4).items())
    opps  = ", ".join(f"{o} ({c})" for o, c in gp.opp.value_counts().head(5).items())

    display(Markdown(
        f"### Player {pos} / {N_PLAYERS} &nbsp;·&nbsp; `{u}`\n"
        f"**{int(r['games'])}** __GAMES_DESC__ games &nbsp;|&nbsp; peak rank **{int(r['best'])}**, "
        f"avg **{r['avg']:.2f}**, median **{r['median']:.0f}** &nbsp;|&nbsp; "
        f"win **{r['win_pct']:.1f}%** &nbsp;|&nbsp; on the coin **{r['coin_pct']:.1f}%**\n\n"
        f"*Archetypes:* {decks}  \n*Vs classes:* {opps}"
    ))

    tbl = gp[DESCRIBE_COLS].describe().T.rename(index=NICE)
    mana_row = cp["mana"].describe(); mana_row.name = "Mana cost of cards"
    tbl = pd.concat([tbl, mana_row.to_frame().T])
    display(tbl.style.format("{:.2f}")
            .background_gradient(subset=["mean"], cmap="rocket_r")
            .set_caption(f"{u[:10]}... - distribution summary"))

    fig, axes = plt.subplots(3, 3, figsize=(14, 10)); axes = axes.ravel()
    for ax, (label, (col, kind)) in zip(axes, GAME_METRICS.items()):
        s = gp[col].dropna()
        if kind == "disc":
            sns.histplot(s.astype(int), discrete=True, color=WARRIOR_RED, ax=ax)
        elif kind == "clip":
            sns.histplot(s.clip(upper=1.5), kde=True, color=WARRIOR_RED, ax=ax)
        else:
            sns.histplot(s, kde=True, color=WARRIOR_RED, ax=ax)
        if len(s):
            ax.axvline(s.mean(), color="k", ls="--", lw=1)
        ax.set_title(label, fontsize=10); ax.set_xlabel("")
    sns.histplot(cp["mana"].astype(int), discrete=True, color=WARRIOR_RED, ax=axes[8])
    axes[8].set_title("Mana cost of played cards (curve)", fontsize=10); axes[8].set_xlabel("")
    fig.suptitle(f"Player {pos} ({u[:10]}...) - metric distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    sns.lineplot(data=tp, x="turn", y="mana_spent", marker="o", color=WARRIOR_RED,
                 errorbar=("ci", 95), ax=ax[0])
    ax[0].set_title("Mean mana spent by turn (95% CI)"); ax[0].set_ylabel("mana")
    dv = gp.hero_deck.value_counts().head(6)
    sns.barplot(x=dv.values, y=dv.index, color=WARRIOR_RED, ax=ax[1]); ax[1].set_title("Warrior archetypes"); ax[1].set_xlabel("games")
    ov = gp.opp.value_counts().head(8)
    sns.barplot(x=ov.values, y=ov.index, color="#34495e", ax=ax[2]); ax[2].set_title("Opponent classes faced"); ax[2].set_xlabel("games")
    fig.tight_layout(); plt.show()


def metric_grid(label, col, kind, source="games"):
    """One metric, 20 sub-plots (4x5) - its distribution for each top-20 player, shared x-range."""
    if source == "cards":
        series = [cards_df.loc[cards_df.user_hash == u, "mana"].dropna() for u in TOP]
    else:
        series = [games_df.loc[games_df.user_hash == u, col].dropna() for u in TOP]
    pooled = pd.concat(series) if series else pd.Series(dtype=float)
    if kind == "clip":
        xlim = (0, 1.5)
    elif kind == "disc":
        xlim = (pooled.min() - 0.5, pooled.max() + 0.5) if len(pooled) else None
    else:                                   # kde / continuous: clip x to the 99th pct so a tail can't squash the rest
        xlim = (pooled.min(), pooled.quantile(0.99)) if len(pooled) else None

    fig, axes = plt.subplots(4, 5, figsize=(18, 11)); axes = axes.ravel()
    for i in range(20):
        ax = axes[i]
        if i >= N_PLAYERS:
            ax.axis("off"); continue
        s = series[i]; kde = (s.nunique() > 1)
        gp_i = games_df[games_df.user_hash == TOP[i]]
        n_games = int(len(gp_i))                      # games this player played
        win_pct = gp_i["win"].mean() * 100            # win rate over those games
        if kind == "disc":
            sns.histplot(s.astype(int), discrete=True, color=WARRIOR_RED, ax=ax)
        elif kind == "clip":
            sns.histplot(s.clip(upper=1.5), kde=kde, color=WARRIOR_RED, ax=ax)
        else:
            sns.histplot(s, kde=kde, color=WARRIOR_RED, ax=ax)
        if len(s):
            ax.axvline(s.mean(), color="k", ls="--", lw=1)
        if xlim and xlim[0] != xlim[1]:
            ax.set_xlim(*xlim)
        ax.set_title(f"P{i + 1} · {n_games} games · {win_pct:.0f}% win", fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle(f"{label} - same metric across the top-20 players (by games)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()'''

COMPARE = r'''sub = games_df[games_df.user_hash.isin(TOP)].copy()
pos_map = {u: i + 1 for i, u in enumerate(TOP)}
sub["player"] = sub.user_hash.map(pos_map)

cmp_cols = ["duration", "n_cards", "cards_per_turn", "mana_eff", "first_turn",
            "coin_turn", "time_per_turn", "mana_per_turn"]
comp = sub.groupby("player")[cmp_cols].mean().rename(columns=NICE)
comp.insert(0, "win%", sub.groupby("player")["win"].mean() * 100)

display(comp.style.format("{:.2f}")
        .background_gradient(cmap="rocket_r")
        .set_caption("Mean of each metric across the top players (player # = games-played order)"))

show = {"duration": "Game length (s)", "mana_eff": "Mana efficiency",
        "coin_turn": "Coin played on turn", "mana_per_turn": "Mana spent / turn"}
fig, axes = plt.subplots(2, 2, figsize=(16, 9)); axes = axes.ravel()
for ax, (col, title) in zip(axes, show.items()):
    sns.boxplot(data=sub, x="player", y=col, hue="player", palette="rocket",
                legend=False, ax=ax)
    ax.set_title(title); ax.set_xlabel("player #")
fig.suptitle("Top __GAMES_DESC__ players compared (full distributions)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()'''


OUTLIER = r'''DURATION_CAP = 3600   # seconds. A real Hearthstone game cannot run > 1 hour;
                      # values above this are AFK / disconnect / paused records that wreck
                      # "Game length (s)" and the derived "Time / turn (s)".

removed = games_df[games_df["duration"] > DURATION_CAP].copy()
n_before = len(games_df)
games_df = games_df[~(games_df["duration"] > DURATION_CAP)].copy()   # keep NaN durations
print(f"Outlier removal: dropped {len(removed)} of {n_before} games "
      f"(duration > {DURATION_CAP}s = 1 h)  ->  {len(games_df)} games remain.")

# ---- side note: exactly which games were removed ----
if len(removed):
    sn = removed.copy()
    sn["player"] = sn["user_hash"].str.slice(0, 12) + "..."
    sn = (sn[["player", "game_id", "rank", "rounds", "duration", "time_per_turn"]]
          .sort_values("duration", ascending=False))
    display(Markdown(
        f"**Removed {len(removed)} impossible-duration game(s)** (`duration > {DURATION_CAP}s`). "
        f"Each inflated *Game length* and *Time / turn* (e.g. a {int(sn['duration'].max())}s record "
        f"= {sn['duration'].max()/3600:.1f} h). All other games are kept unchanged:"))
    display(sn.head(40).style.hide(axis="index")
            .format({"duration": "{:.0f}s", "time_per_turn": "{:.0f}s", "rounds": "{:.0f}", "rank": "{:.0f}"})
            .set_caption("Outlier games removed before every metric below"))
    if len(removed) > 40:
        display(Markdown(f"*(+{len(removed) - 40} more, all with `duration > {DURATION_CAP}s`.)*"))
else:
    display(Markdown(f"*No games exceeded the {DURATION_CAP}s duration cap.*"))'''


# (label, call-arguments) for the 9 per-metric across-player grids in section 7
METRIC_GRIDS = [
    ("Game length (s)",            '"Game length (s)", "duration", "kde"'),
    ("Cards / game",               '"Cards / game", "n_cards", "kde"'),
    ("Cards / turn",               '"Cards / turn", "cards_per_turn", "kde"'),
    ("Mana efficiency",            '"Mana efficiency", "mana_eff", "clip"'),
    ("First turn to play",         '"First turn to play", "first_turn", "disc"'),
    ("Coin played on turn",        '"Coin played on turn", "coin_turn", "disc"'),
    ("Time / turn (s)",            '"Time / turn (s)", "time_per_turn", "kde"'),
    ("Mana spent / turn",          '"Mana spent / turn", "mana_per_turn", "kde"'),
    ("Mana cost of played cards",  '"Mana cost of played cards", None, "disc", source="cards"'),
]


def fill(src, cache, deck_literal, games_desc):
    return (src.replace("__CACHE__", cache)
               .replace("__DECK__", deck_literal)
               .replace("__GAMES_DESC__", games_desc))


def build_notebook(intro_md, cache, deck_literal, games_desc, n_players):
    cells = [md(intro_md)]
    cells += [md("## 1 · Setup"), code(fill(SETUP, cache, deck_literal, games_desc))]
    cells += [md("## 2 · Load & parse into tidy DataFrames (rank 5-10 only)\n"
                 "Uses a pre-built cache pickle (same schema as the rank-0/1/2 study). "
                 "Delete the cache file to force a full re-parse of the 16 monthly dumps."),
              code(fill(BUILD, cache, deck_literal, games_desc))]
    cells += [md("## 2b · Remove undeniable outliers (impossible game durations)\n"
                 "A handful of games carry physically-impossible `duration` values (AFK / disconnect / paused — "
                 "e.g. a ~13-hour record), which wreck **Game length (s)** and the derived **Time / turn (s)**. "
                 "We drop any game with `duration > 3600s` (1 hour) **before all analysis** and print a side note "
                 "listing exactly what was removed. Every other game is kept untouched."),
              code(fill(OUTLIER, cache, deck_literal, games_desc))]
    cells += [md("## 3 · Select the players (top 20 by games played)\n"
                 "*Ranked by **number of games played** (descending) — ties broken by peak then average rank.*"),
              code(fill(SELECT, cache, deck_literal, games_desc))]
    cells += [md("## 4 · Per-player renderer\n"
                 "Each call renders a header, a styled describe() table, a 3x3 seaborn distribution grid "
                 "(with KDE), and a turn-curve / archetype / opponent panel."),
              code(fill(RENDER, cache, deck_literal, games_desc))]
    cells += [md("---\n## 5 · Per-player distributions\n*Each player is rendered in its own code cell.*")]
    for i in range(1, n_players + 1):
        cells += [md(f"### Player {i} of {n_players}"), code(f"show_player({i})")]
    cells += [md("---\n## 6 · Cross-player comparison"),
              code(fill(COMPARE, cache, deck_literal, games_desc))]
    cells += [md("---\n## 7 · Same metric across all players\n"
                 "One block per metric (9 in total). Each block draws a **4×5 grid of 20 sub-plots** — the "
                 "metric's distribution for each of the top-20 players (P1…P20), on a shared x-range so they "
                 "are directly comparable. Each sub-plot title shows the player's **number of games played** and "
                 "**win rate**; dashed line = that player's mean. Uses the outlier-cleaned data.")]
    for label, args in METRIC_GRIDS:
        cells += [md(f"### {label}"), code(f"metric_grid({args})")]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


TOP_N = 20

# ---------- CLASS notebook (most-played class = Warrior, all archetypes) ----------
cls_intro = f"""# Top-20 (by games played) **Rank 5-10** {SEL['top_class']} Players - Playstyle Distribution Analysis

Same pipeline as `warrior_top20_rank012_analysis.ipynb`, re-pointed to the **rank 5-10** ladder band and to
the **most-played class of that band**. Across all 16 monthly dumps, the most-played class among me-side
ranked games at rank 5-10 is **{SEL['top_class']}** ({SEL['top_class_games']:,} games, {SEL['top_class_players']} players).
Players are now **ranked by the number of games they played** (not by ladder rank).

## Data & selection (parameters for this run)
| Choice | Value |
|---|---|
| **Files** | all 16 monthly files `201x-xx.json` (`2016-06` … `2017-09`) |
| **Mode filter** | `mode == "ranked"` |
| **Rank filter** | **`rank in {{5,6,7,8,9,10}}`** — every metric below is computed from rank-5-10 games |
| **Class** | `hero == "{SEL['top_class']}"` — **me-side only** (the recording player), the most-played class at rank 5-10 |
| **Player identity** | `user_hash`. Opponents carry no ID (only a class), so they are excluded |
| **Eligibility** | **>= 16** ranked {SEL['top_class']} games *at rank 5-10* |
| **Players shown** | **top 20**, ranked by **games played (desc)** → peak rank → average rank |
| **The Coin** | excluded from card counts & mana-cost stats; adds **+1** to mana available on the turn it is played |

## Metric definitions (per player, distribution over their rank-5-10 games)
1. **Game length (s)** — `duration`. 2. **Cards / game** — non-Coin `me` cards.
3. **Cards / turn**. 4. **Mana cost of played cards** (the curve). 5. **Mana efficiency / game** = `mana spent / mana available`.
6. **First turn to play a card**. 7. **When The Coin is played**. 8. **Time / turn (s)** = `duration / rounds`. 9. **Mana spent / turn**.

> **Caveats.** `card_history` logs only cards played *from hand* (no hero power, weapon/minion attacks, unplayed cards)
> → mana spent / efficiency / cards-played are lower bounds; efficiency can exceed 1.0. Rank 5-10 is a far larger
> universe (~9.5k {SEL['top_class']} games) than the rank-0/1/2 study, so distributions are tighter.
> Games with `duration > 1 hour` (impossible AFK/disconnect records) are removed up front — see §2b for the list.
"""

cls_nb = build_notebook(cls_intro, SEL['top_class_cache'].replace("\\", "\\\\"),
                        "None", f"rank 5-10 {SEL['top_class']}", TOP_N)
cls_path = r"D:\test\warrior_top20_rank5to10_byGames_analysis.ipynb"
with open(cls_path, "w", encoding="utf-8") as fh:
    json.dump(cls_nb, fh, indent=1)
print("wrote", cls_path)

# ---------- DECK notebook (most-played deck = Warrior Pirate) ----------
deck_label = f"{SEL['top_deck_class']} {SEL['top_deck_arch']}"
deck_intro = f"""# Top-20 (by games played) **Rank 5-10** {deck_label} Players - Playstyle Distribution Analysis

Same pipeline as `warrior_top20_rank012_analysis.ipynb`, re-pointed to the **rank 5-10** ladder band and to
the **most-played deck of that band**. Across all 16 monthly dumps, the most-played (class, archetype) deck
among me-side ranked games at rank 5-10 is **{deck_label}** ({SEL['top_deck_games']:,} games, {SEL['top_deck_players']} players).
Players are **ranked by the number of games they played in this deck** (not by ladder rank).

## Data & selection (parameters for this run)
| Choice | Value |
|---|---|
| **Files** | all 16 monthly files `201x-xx.json` (`2016-06` … `2017-09`) |
| **Mode filter** | `mode == "ranked"` |
| **Rank filter** | **`rank in {{5,6,7,8,9,10}}`** |
| **Deck** | `hero == "{SEL['top_deck_class']}"` **and** `hero_deck == "{SEL['top_deck_arch']}"` — me-side only, the most-played deck at rank 5-10 |
| **Player identity** | `user_hash`. Opponents excluded (no ID) |
| **Eligibility** | **>= 16** ranked {deck_label} games *at rank 5-10* |
| **Players shown** | **top 20**, ranked by **games played (desc)** → peak rank → average rank |
| **The Coin** | excluded from card/mana stats; adds **+1** mana available on the turn it is played |

## Metric definitions (per player, distribution over their rank-5-10 {deck_label} games)
1. **Game length (s)**. 2. **Cards / game**. 3. **Cards / turn**. 4. **Mana cost of played cards** (curve).
5. **Mana efficiency / game**. 6. **First turn to play a card**. 7. **When The Coin is played**.
8. **Time / turn (s)**. 9. **Mana spent / turn**.

> **Note.** Holding the deck fixed (a single archetype) removes archetype as a confound, so the per-player
> "archetypes" panel shows a single bar ({SEL['top_deck_arch']}) by design. **Caveats** as in the class study:
> `card_history` logs only cards played from hand → mana / efficiency / cards are lower bounds.
> Games with `duration > 1 hour` (impossible AFK/disconnect records) are removed up front — see §2b for the list.
"""

deck_nb = build_notebook(deck_intro, SEL['top_deck_cache'].replace("\\", "\\\\"),
                         f'"{SEL["top_deck_arch"]}"', f"rank 5-10 {deck_label}", TOP_N)
deck_path = r"D:\test\warrior_pirate_top20_rank5to10_byGames_analysis.ipynb"
with open(deck_path, "w", encoding="utf-8") as fh:
    json.dump(deck_nb, fh, indent=1)
print("wrote", deck_path)
