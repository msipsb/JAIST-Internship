# log_power — official-Hearthstone-format `Power.log` dataset

Re-simulates the V2 matchup matrix and writes each game as a **Hearthstone
`Power.log`**, the format the real client produces.

Run `run_batch_powerlog.ipynb` — one cell per P1 deck, resumable.

```text
log_power/<agent1>_<deck1>/game_NNN_<a1>-<d1>_vs_<a2>-<d2>.log
log_power/<agent1>_<deck1>/summary.csv
```

5 P1 agents × 5 P2 agents × 9 P2 decks × 20 games = 4,500 per deck cell;
all 9 cells = **40,500 games, ~15-20 h, ~17 GB**.

## What a real Hearthstone log is

The client writes `Hearthstone/Logs/Power.log` when `log.config` enables the
Power zone. One match is a `CREATE_GAME` followed by timestamped packets:

```text
D 09:24:38.5069226 GameState.DebugPrintPower() - CREATE_GAME
D 09:24:38.5069226 GameState.DebugPrintPower() -     GameEntity EntityID=1
D 09:24:38.5069226 GameState.DebugPrintPower() -         tag=TURN value=1
D 09:24:39.1234567 GameState.DebugPrintPower() - BLOCK_START BlockType=PLAY Entity=[entityName=Fiery War Axe id=17 zone=HAND zonePos=3 cardId=CS2_106 player=1] EffectCardId= EffectIndex=0 Target=0 SubOption=-1
D 09:24:39.1234567 GameState.DebugPrintPower() -     TAG_CHANGE Entity=[...] tag=ZONE value=PLAY
D 09:24:39.1234567 GameState.DebugPrintPower() - BLOCK_END
```

Entities are ids (1=GameEntity, 2/3=Players, then heroes/cards), state changes
are `tag=`/`value=` pairs, and blocks nest (`PLAY` → `TRIGGER` → `DEATHS`).

## Why these logs are real, not reconstructed

SabberStone already models that packet vocabulary. With `GameConfig.History =
true` the engine records a **PowerHistory** of `CREATE_GAME` / `FULL_ENTITY` /
`SHOW_ENTITY` / `HIDE_ENTITY` / `CHANGE_ENTITY` / `TAG_CHANGE` / `BLOCK_START` /
`BLOCK_END` entries while it plays. `powerlog_render.py` re-renders that history
into the client's text format, so **trigger ordering and block nesting are the
engine's own**.

Verified end to end: `roundtrip_check.py` parses each rendered game with
**hslog** (HearthSim's parser, behind HSReplay.net) and diffs the reconstructed
winner, turn count, hero health, armor and **both boards** against the
SabberStone game object.

> `hslog` is needed **only** for that verification, and only in the interpreter
> that runs it. This repo has both a global `py -3` and `d:\test\.venv` (the
> notebook kernel), so install it into the right one:
> `d:\test\.venv\Scripts\python.exe -m pip install hslog`. Generating the
> dataset does not need it.

## Files

| file | role |
| --- | --- |
| `run_batch_powerlog.ipynb` | the runner — one cell per P1 deck |
| `run_batch_powerlog.py` | batch matrix driver (mirrors `log_v2/run_batch_matrix_v2.py`) |
| `sim_powerlog.py` | plays one game with `History=True`, renders it |
| `powerlog_render.py` | PowerHistory → `Power.log` text, plus viewpoint hiding and validation |
| `roundtrip_check.py` | simulate → render → parse with hslog → diff vs the engine |
| `_make_powerlog_notebook.py` | rebuilds the notebook |

## Decisions that shape the data

* **These are NEW games.** Re-simulated, so they do **not** correspond to the
  `log_v2/` games — never join the two by game index.
* **Viewpoint = P1's client.** A real log holds only what one client saw: deck
  cards are unknown until drawn (P1's own included), and P2's hand/secrets stay
  hidden until something makes them public. This discards information the
  simulator has — that is the point of the format.
* **Labels are not in the `.log`.** agent/playstyle/deck ground truth lives in
  `summary.csv` only, keeping features and labels separate (same split as
  `sim_to_hearthstonemap.py`).
* **Timestamps are synthetic**, derived from each game's RNG seed. Real think
  time correlates with the agent, so it would leak the playstyle label into the
  timing.

## Known limits

* **29 engine-only GameTags are dropped.** SabberStone models tags the real game
  has no equivalent for (`FATIGUEREFERENCE`, `HEADCRACK_COMBO`, `BACON_*`, …);
  a real `Power.log` cannot contain them and HearthSim's parser rejects the
  names. See `ENGINE_ONLY_TAGS` in `powerlog_render.py`.
* **`History=True` shifts the RNG stream**, so a seed does not reproduce a
  `History=False` game. It costs ~16% runtime and does **not** bias play: over
  40 games/arm, mean turns 14.68 (off) vs 14.57 (on), Mann-Whitney p=0.96. It
  only appends to `PowerHistory` inside tag setters and never touches the RNG,
  and `Game.Clone()` defaults to `history=false` so the AI search is unaffected.
* **~0.8% of games raise** on a card SabberStone has not implemented; they are
  recorded as `ERROR` rows in `summary.csv` and skipped.
* **`SubOption` is always -1** and `EffectIndex` comes from the engine's block
  index; the client uses these for multi-option cards, and SabberStone does not
  track them the same way.
