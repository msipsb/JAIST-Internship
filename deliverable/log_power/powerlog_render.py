#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render SabberStone's own PowerHistory as an official-Hearthstone Power.log.

The real Hearthstone client writes Logs/Power.log (when log.config enables the
Power zone) as a stream of timestamped lines:

    D 09:24:38.5069226 GameState.DebugPrintPower() - CREATE_GAME
    D 09:24:38.5069226 GameState.DebugPrintPower() -     GameEntity EntityID=1
    D 09:24:38.5069226 GameState.DebugPrintPower() -         tag=TURN value=1
    D 09:24:39.1234567 GameState.DebugPrintPower() - BLOCK_START BlockType=PLAY \
Entity=[entityName=Fiery War Axe id=17 zone=HAND zonePos=3 cardId=CS2_106 player=1] \
EffectCardId= EffectIndex=0 Target=0 SubOption=-1
    D 09:24:39.1234567 GameState.DebugPrintPower() -     TAG_CHANGE Entity=[...] tag=ZONE value=PLAY
    D 09:24:39.1234567 GameState.DebugPrintPower() - BLOCK_END

SabberStone already models exactly this. With GameConfig.History = true the
engine records a PowerHistory of CREATE_GAME / FULL_ENTITY / SHOW_ENTITY /
HIDE_ENTITY / TAG_CHANGE / BLOCK_START / BLOCK_END entries as the game runs --
the same packet vocabulary the client logs. So this module does NOT reconstruct
a plausible log: it re-renders the engine's real history, with real
trigger-by-trigger ordering and real block nesting, into the client's text
format.

Design decisions (confirmed with the user):

  * SOURCE = a fresh simulation with History=True (see sim_powerlog.py). Games
    are newly played, so they do NOT correspond to the log_v2 games.
  * VIEWPOINT = P1's client ("true client fidelity"). PowerHistory is the
    server's full-information view; a real Power.log is written by ONE client
    and only holds what that client was shown, so this module hides the rest:
      - a DECK card is unknown to everybody, including its owner, until drawn
      - P2's hand and secrets stay hidden; a P2 card is revealed only when it
        becomes public (played, summoned, triggered, discarded)
      - heroes, hero powers and the board are public
    Hiding is applied on top of the engine stream: when an entity becomes
    visible to P1 a SHOW_ENTITY is injected, exactly as the client emits one.
  * SURFACE = GameState.DebugPrintPower() blocks only. No PowerTaskList mirror
    and no DebugPrintOptions -- this is the subset real parsers (HDT,
    HearthSim/hslog) actually consume.
  * ENTITY IDS are the engine's own: 1=GameEntity, 2/3=Players, then heroes,
    hero powers and cards, exactly as SabberStone allocated them.

TIMESTAMPS are synthetic, derived deterministically from the game's RNG seed.
    They are NOT wall-clock: the AI's think time correlates with which agent is
    playing, so real timings would leak the playstyle label into the log. Same
    seed -> same timestamps; timing carries no agent signal.

LABELS (agent / playstyle / deck) never appear inside the .log. They are ground
    truth for evaluation, not features, so they live in summary.csv only -- the
    same split sim_to_hearthstonemap.py uses.
"""
import os
import re
import json
import random
import datetime as _dt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

VIEWPOINT = 1                              # P1's client

# ---- zone ids (SabberStoneCore.Enums.Zone) ----
Z_PLAY, Z_DECK, Z_HAND, Z_GRAVEYARD, Z_REMOVED, Z_SETASIDE, Z_SECRET = 1, 2, 3, 4, 5, 6, 7

HIDDEN_NAME = "UNKNOWN ENTITY [cardType=INVALID]"

# Tags a hidden entity may carry. The client never tells you an opponent's hand
# card's cost or stats, so anything outside this set is withheld while hidden.
SAFE_TAGS = frozenset(("ZONE", "ZONE_POSITION", "CONTROLLER", "ENTITY_ID"))

# SabberStone models 29 GameTags the real game has no equivalent for: engine
# bookkeeping (FATIGUEREFERENCE, NUM_CARDS_TO_DRAW), per-card helpers
# (HEADCRACK_COMBO, MOAT_LURKER_MINION) and Battlegrounds/later-expansion
# leftovers (BACON_*, TWINSPELL_COPY, SIDEQUEST). A real Power.log cannot
# contain them -- HearthSim's parser raises NoSuchEnum on the names -- so they
# are dropped. (Two more, DORMANT and NUM_SPELLS_PLAYED_THIS_GAME, exist in
# both but with different numbers; harmless, since tags are written by name.)
ENGINE_ONLY_TAGS = frozenset((
    "AMOUNT_HERO_HEALED_THIS_TURN", "AUTOATTACK",
    "BACON_HIGHLIGHT_ATTACKING_MINION_DURING_COMBAT", "BACON_USE_FAST_ANIMATIONS",
    "CASTSWHENDRAWN", "CONTROLLER_CHANGED_THIS_TURN", "EXTRA_END_TURN_EFFECT",
    "FATIGUEREFERENCE", "HEADCRACK_COMBO", "KEEP_ENCHANTMENTS",
    "LAST_CARD_DISCARDED", "LAST_CARD_DRAWN", "MOAT_LURKER_MINION", "MODULAR",
    "NUM_CARDS_TO_DRAW", "NUM_ELEMENTAL_PLAYED_LAST_TURN",
    "NUM_ELEMENTAL_PLAYED_THIS_TURN", "NUM_MURLOCS_PLAYED_THIS_GAME",
    "NUM_SECRETS_PLAYED_THIS_GAME", "NUM_SPELLS_PLAYED_THIS_TURN",
    "NUM_WEAPONS_PLAYED_THIS_GAME", "OUTGOING_DAMAGE_CAP", "RED_MANA_CRYSTALS",
    "SIDEQUEST", "START_OF_GAME", "TAG_LAST_KNOWN_ATK_IN_HAND",
    "TAG_LAST_KNOWN_POSITION_ON_BOARD", "TWINSPELL_COPY", "WEAPON",
))


def keep_tag(name):
    """Would the real client ever write this tag?"""
    return name not in ENGINE_ONLY_TAGS

# ---------------------------------------------------------------- card naming
CARDDEFS_XML = os.path.join(PARENT_DIR, "SabberStone", "SabberStoneCore",
                            "resources", "Data", "CardDefs.xml")
_CARD_NAMES = None


def card_names():
    """card_id -> English name, from SabberStone's own card DB (~0.3 s, cached)."""
    global _CARD_NAMES
    if _CARD_NAMES is not None:
        return _CARD_NAMES
    idx = {}
    try:
        import xml.etree.ElementTree as ET
        for _, el in ET.iterparse(CARDDEFS_XML, events=("end",)):
            if el.tag != "Entity":
                continue
            cid = el.get("CardID")
            for tag in el.findall("Tag"):
                if tag.get("name") == "CARDNAME":
                    en = tag.find("enUS")
                    if en is not None and cid:
                        idx[cid] = en.text
            el.clear()
    except Exception:                        # noqa: BLE001 -- names are cosmetic
        pass
    _CARD_NAMES = idx
    return idx


def card_name(card_id):
    if not card_id:
        return ""
    return card_names().get(card_id, card_id)


# ============================================================================
# PowerHistory (.NET)  ->  plain python entries
# ============================================================================
_SYMBOLS = None


def tag_symbols():
    """{tag_name: {int: SYMBOLIC}} for the enum-typed tags the client spells out.

    Built from SabberStone's own Tag.TypedTags, so ZONE=1 prints as PLAY and
    PLAYSTATE=4 as WON, exactly like a real log. Untyped tags stay numeric.
    """
    global _SYMBOLS
    if _SYMBOLS is not None:
        return _SYMBOLS
    out = {}
    try:
        import System
        from SabberStoneCore.Model import Tag
        for gt in Tag.TypedTags.Keys:
            ty = Tag.TypedTags[gt]
            vals = {}
            for v in System.Enum.GetValues(ty):
                vals[int(v)] = str(v)
            out[str(gt)] = vals
    except Exception:                        # noqa: BLE001
        pass
    _SYMBOLS = out
    return out


def _tags_of(power_entity):
    """[(tag_name, int_value)] for a PowerHistoryEntity/PowerEntity."""
    return [(str(kv.Key), int(kv.Value)) for kv in power_entity.Tags]


def history_to_entries(power_history):
    """SabberStone PowerHistory -> ordered list of plain dicts.

    pythonnet hands back List<IPowerHistoryEntry>, whose concrete fields are
    invisible through the interface. Reflection reads them but costs ~5 us a
    call; LINQ OfType<T> downcasts a whole type at once (~0.2 us/access) and
    preserves order within each type, so the per-type sequences can be zipped
    back onto the interface list's type order to recover the exact stream.
    """
    from System.Linq import Enumerable
    from SabberStoneCore.Kettle import (
        PowerHistoryCreateGame, PowerHistoryFullEntity, PowerHistoryShowEntity,
        PowerHistoryHideEntity, PowerHistoryTagChange, PowerHistoryBlockStart,
        PowerHistoryBlockEnd, PowerHistoryMetaData, PowerHistoryChangeEntity)

    full = power_history.Full
    by_type = {
        "CREATE_GAME": iter(list(Enumerable.OfType[PowerHistoryCreateGame](full))),
        "FULL_ENTITY": iter(list(Enumerable.OfType[PowerHistoryFullEntity](full))),
        "SHOW_ENTITY": iter(list(Enumerable.OfType[PowerHistoryShowEntity](full))),
        "HIDE_ENTITY": iter(list(Enumerable.OfType[PowerHistoryHideEntity](full))),
        "TAG_CHANGE": iter(list(Enumerable.OfType[PowerHistoryTagChange](full))),
        "BLOCK_START": iter(list(Enumerable.OfType[PowerHistoryBlockStart](full))),
        "BLOCK_END": iter(list(Enumerable.OfType[PowerHistoryBlockEnd](full))),
        "META_DATA": iter(list(Enumerable.OfType[PowerHistoryMetaData](full))),
        "CHANGE_ENTITY": iter(list(Enumerable.OfType[PowerHistoryChangeEntity](full))),
    }

    entries = []
    for item in full:
        kind = str(item.PowerType)
        src = by_type.get(kind)
        if src is None:
            # Never skip quietly: dropping CHANGE_ENTITY once left Choose One
            # minions logged as their unplayed base card.
            raise ValueError("unhandled PowerHistory entry type %r" % kind)
        e = next(src)
        if kind == "CREATE_GAME":
            entries.append({
                "t": kind,
                "game_id": int(e.Game.Id),
                "game_tags": _tags_of(e.Game),
                "players": [{"player_id": int(p.PlayerId),
                             "account_id": int(p.AccountId),
                             "entity_id": int(p.PowerEntity.Id),
                             "tags": _tags_of(p.PowerEntity)} for p in e.Players],
            })
        elif kind in ("FULL_ENTITY", "SHOW_ENTITY"):
            entries.append({"t": kind, "id": int(e.Entity.Id),
                            "card_id": str(e.Entity.Name or ""),
                            "tags": _tags_of(e.Entity)})
        elif kind == "HIDE_ENTITY":
            entries.append({"t": kind, "id": int(e.EntityID), "zone": int(e.Zone)})
        elif kind == "TAG_CHANGE":
            entries.append({"t": kind, "id": int(e.EntityId),
                            "tag": str(e.Tag), "value": int(e.Value)})
        elif kind == "BLOCK_START":
            entries.append({"t": kind, "block_type": str(e.BlockType),
                            "source": int(e.Source), "target": int(e.Target),
                            "index": int(e.Index),
                            "effect_card_id": str(e.EffectCardId or "")})
        elif kind == "CHANGE_ENTITY":
            # a Choose One mode / transform: same entity id, different card
            entries.append({"t": kind, "id": int(e.Entity.Id),
                            "card_id": str(e.CardId or e.Entity.Name or ""),
                            "tags": _tags_of(e.Entity)})
        elif kind == "META_DATA":
            entries.append({"t": kind, "meta_type": str(e.Type),
                            "data": int(e.Data),
                            "info": [int(i) for i in (e.Info or [])]})
        else:
            entries.append({"t": kind})
    return entries


# ============================================================================
# rendering
# ============================================================================
class Ent(object):
    """What P1's client knows about one entity."""
    __slots__ = ("eid", "card_id", "zone", "pos", "controller", "revealed",
                 "tags", "label")

    def __init__(self, eid):
        self.eid = eid
        self.card_id = ""
        self.zone = 0
        self.pos = 0
        self.controller = 0
        self.revealed = False
        self.tags = {}
        self.label = None                    # players have a name, not a card

    @property
    def name(self):
        return self.label or card_name(self.card_id)

    def desc(self):
        if self.revealed:
            return ("[entityName=%s id=%d zone=%s zonePos=%d cardId=%s player=%d]"
                    % (self.name or "UNKNOWN", self.eid, zone_name(self.zone),
                       self.pos, self.card_id, self.controller))
        return ("[entityName=%s id=%d zone=%s zonePos=%d cardId= player=%d]"
                % (HIDDEN_NAME, self.eid, zone_name(self.zone), self.pos,
                   self.controller))


def zone_name(z):
    return tag_symbols().get("ZONE", {}).get(z, str(z) if z else "INVALID")


def sym(tag, value):
    """Symbolic value if the tag is enum-typed, else the number."""
    table = tag_symbols().get(tag)
    if table and value in table:
        return table[value]
    return str(value)


class PowerLog(object):
    """Accumulates Power.log lines with a deterministic synthetic clock."""

    def __init__(self, start_iso, seed):
        try:
            self.clock = _dt.datetime.strptime(start_iso, "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError):
            self.clock = _dt.datetime(2017, 1, 1, 12, 0, 0)
        self.rng = random.Random(seed)       # timing depends only on the seed
        self.lines = []
        self.depth = 0
        self._stamp = self._hs_time()

    def tick(self, lo=0.3, hi=2.5):
        """Advance the clock. Everything between two ticks shares a timestamp,
        the way the client stamps a whole block at once."""
        self.clock += _dt.timedelta(seconds=self.rng.uniform(lo, hi))
        self._stamp = self._hs_time()

    def _hs_time(self):
        # HH:MM:SS.fffffff -- the client writes 7 fractional digits
        return "%s%01d" % (self.clock.strftime("%H:%M:%S.%f"),
                           self.rng.randrange(10))

    def emit(self, text, indent=0):
        self.lines.append("D %s GameState.DebugPrintPower() - %s%s"
                          % (self._stamp, "    " * (self.depth + indent), text))

    def tag(self, name, value, indent=1):
        self.emit("tag=%s value=%s" % (name, value), indent=indent)


def _birth_zones(entries):
    """eid -> the zone an entity is created into.

    SabberStone builds an entity before putting it anywhere, so its FULL_ENTITY
    often carries ZONE=INVALID and the real zone arrives in the next TAG_CHANGE.
    Visibility depends on the zone, so without this every entity -- heroes
    included -- would be created hidden and then immediately revealed. The
    client instead creates public entities with their CardID already set.
    """
    birth = {}
    pending = set()
    for entry in entries:
        kind = entry["t"]
        if kind == "FULL_ENTITY":
            zone = dict(entry["tags"]).get("ZONE", 0)
            if zone:
                birth[entry["id"]] = zone
            else:
                pending.add(entry["id"])
        elif (kind == "TAG_CHANGE" and entry["tag"] == "ZONE"
                and entry["id"] in pending):
            birth[entry["id"]] = entry["value"]
            pending.discard(entry["id"])
    return birth


class Renderer(object):
    """Replays the engine's PowerHistory into client-format text."""

    def __init__(self, entries, seed=0, start_iso=None, viewpoint=VIEWPOINT):
        self.entries = entries
        self.view = viewpoint
        self.log = PowerLog(start_iso, seed)
        self.ents = {}
        self.birth_zone = _birth_zones(entries)

    # ---------------------------------------------------------------- helpers
    def ent(self, eid):
        e = self.ents.get(eid)
        if e is None:
            e = Ent(eid)
            self.ents[eid] = e
        return e

    def visible(self, e):
        """Would P1's client know this card's identity right now?"""
        if e.zone in (Z_PLAY, Z_GRAVEYARD, Z_REMOVED):
            return True                      # public
        if e.zone == Z_DECK:
            return False                     # nobody knows a deck card
        if e.zone in (Z_HAND, Z_SECRET, Z_SETASIDE):
            return e.controller == self.view
        return False

    def desc(self, eid):
        if eid in (0, None):
            return "0"
        if eid == 1:
            return "GameEntity"
        return self.ent(eid).desc()

    def apply(self, e, tags):
        """Update our model from a tag list."""
        for name, value in tags:
            e.tags[name] = value
            if name == "ZONE":
                e.zone = value
            elif name == "ZONE_POSITION":
                e.pos = value
            elif name == "CONTROLLER":
                e.controller = value

    def sync_reveal(self, e):
        """Inject the SHOW_ENTITY the client emits when a card becomes known."""
        if e.revealed or not e.card_id or not self.visible(e):
            return
        hidden = ("[entityName=%s id=%d zone=%s zonePos=%d cardId= player=%d]"
                  % (HIDDEN_NAME, e.eid, zone_name(e.zone), e.pos, e.controller))
        self.log.emit("SHOW_ENTITY - Updating Entity=%s CardID=%s"
                      % (hidden, e.card_id))
        e.revealed = True
        for name, value in sorted(e.tags.items()):
            if keep_tag(name):
                self.log.tag(name, sym(name, value))

    def hide(self, e):
        if e.revealed and not self.visible(e):
            e.revealed = False               # back into the unknown

    # ------------------------------------------------------------- rendering
    def render(self):
        log = self.log
        for entry in self.entries:
            kind = entry["t"]

            if kind == "CREATE_GAME":
                log.emit("CREATE_GAME")
                log.emit("GameEntity EntityID=%d" % entry["game_id"], indent=1)
                for name, value in entry["game_tags"]:
                    if keep_tag(name):
                        log.tag(name, sym(name, value), indent=2)
                for p in entry["players"]:
                    log.emit("Player EntityID=%d PlayerID=%d "
                             "GameAccountId=[hi=144115193835963207 lo=%d]"
                             % (p["entity_id"], p["player_id"], p["account_id"]),
                             indent=1)
                    for name, value in p["tags"]:
                        if keep_tag(name):
                            log.tag(name, sym(name, value), indent=2)
                    pe = self.ent(p["entity_id"])
                    self.apply(pe, p["tags"])
                    pe.label = "P%d" % p["player_id"]
                    pe.controller = p["player_id"]
                    pe.revealed = True
                log.tick()

            elif kind == "FULL_ENTITY":
                e = self.ent(entry["id"])
                e.card_id = entry["card_id"]
                self.apply(e, entry["tags"])
                # decide visibility from the zone it is being created INTO,
                # which the engine may only reveal on the next TAG_CHANGE
                probe = Ent(e.eid)
                probe.zone = e.zone or self.birth_zone.get(e.eid, 0)
                probe.controller = e.controller
                show = self.visible(probe)
                e.revealed = show
                log.emit("FULL_ENTITY - Creating ID=%d CardID=%s"
                         % (e.eid, e.card_id if show else ""))
                for name, value in entry["tags"]:
                    if keep_tag(name) and (show or name in SAFE_TAGS):
                        log.tag(name, sym(name, value))

            elif kind == "SHOW_ENTITY":
                e = self.ent(entry["id"])
                e.card_id = entry["card_id"] or e.card_id
                self.apply(e, entry["tags"])
                if self.visible(e):
                    if not e.revealed:
                        hidden = ("[entityName=%s id=%d zone=%s zonePos=%d "
                                  "cardId= player=%d]"
                                  % (HIDDEN_NAME, e.eid, zone_name(e.zone),
                                     e.pos, e.controller))
                        log.emit("SHOW_ENTITY - Updating Entity=%s CardID=%s"
                                 % (hidden, e.card_id))
                    else:
                        log.emit("SHOW_ENTITY - Updating Entity=%s CardID=%s"
                                 % (e.desc(), e.card_id))
                    e.revealed = True
                    for name, value in entry["tags"]:
                        if keep_tag(name):
                            log.tag(name, sym(name, value))

            elif kind == "CHANGE_ENTITY":
                # Choose One / transform: the entity keeps its id but becomes a
                # different card, so the client re-announces it under the new one
                e = self.ent(entry["id"])
                log.emit("CHANGE_ENTITY - Updating Entity=%s CardID=%s"
                         % (e.desc(), entry["card_id"]))
                e.card_id = entry["card_id"]
                self.apply(e, entry["tags"])
                e.revealed = self.visible(e)
                if e.revealed:
                    for name, value in entry["tags"]:
                        if keep_tag(name):
                            log.tag(name, sym(name, value))

            elif kind == "META_DATA":
                log.emit("META_DATA - Meta=%s Data=%d InfoCount=%d"
                         % (entry["meta_type"], entry["data"],
                            len(entry["info"])))
                for i, eid in enumerate(entry["info"]):
                    log.emit("Info[%d] = %s" % (i, self.desc(eid)), indent=1)

            elif kind == "HIDE_ENTITY":
                e = self.ent(entry["id"])
                log.emit("HIDE_ENTITY - Entity=%s tag=ZONE value=%s"
                         % (e.desc(), zone_name(entry["zone"])))
                e.zone = entry["zone"]
                self.hide(e)

            elif kind == "TAG_CHANGE":
                e = self.ent(entry["id"])
                name, value = entry["tag"], entry["value"]
                show = (keep_tag(name)
                        and (e.revealed or name in SAFE_TAGS
                             or entry["id"] in (1, 2, 3)))
                if show:
                    log.emit("TAG_CHANGE Entity=%s tag=%s value=%s"
                             % (self.desc(entry["id"]), name, sym(name, value)))
                e.tags[name] = value
                if name == "ZONE":
                    e.zone = value
                elif name == "ZONE_POSITION":
                    e.pos = value
                elif name == "CONTROLLER":
                    e.controller = value
                if name in ("ZONE", "CONTROLLER"):
                    self.hide(e)
                    self.sync_reveal(e)

            elif kind == "BLOCK_START":
                log.emit("BLOCK_START BlockType=%s Entity=%s EffectCardId=%s "
                         "EffectIndex=%d Target=%s SubOption=-1"
                         % (entry["block_type"], self.desc(entry["source"]),
                            entry["effect_card_id"], entry["index"],
                            self.desc(entry["target"])))
                log.depth += 1
                log.tick()

            elif kind == "BLOCK_END":
                log.depth = max(0, log.depth - 1)
                log.emit("BLOCK_END")

        return log.lines


def render_game(entries, seed=0, start_iso=None, viewpoint=VIEWPOINT):
    return Renderer(entries, seed, start_iso, viewpoint).render()


# ---------------------------------------------------------------- validation
LINE_RE = re.compile(r"^D \d\d:\d\d:\d\d\.\d{7} GameState\.DebugPrintPower\(\) - ")


def validate_lines(lines, viewpoint=VIEWPOINT):
    """Self-consistency checks on one rendered game; returns problem strings."""
    problems = []
    if not lines or "CREATE_GAME" not in lines[0]:
        problems.append("first line is not CREATE_GAME")
    bad = [ln for ln in lines if not LINE_RE.match(ln)]
    if bad:
        problems.append("%d lines do not match the client line format" % len(bad))

    depth = 0
    for ln in lines:
        if "BLOCK_START" in ln:
            depth += 1
        elif "BLOCK_END" in ln:
            depth -= 1
            if depth < 0:
                problems.append("BLOCK_END without BLOCK_START")
                break
    if depth != 0:
        problems.append("%d unclosed BLOCK_START" % depth)

    if not any("tag=STATE value=COMPLETE" in ln for ln in lines):
        problems.append("game never reaches STATE=COMPLETE")
    if not any("tag=PLAYSTATE value=WON" in ln or "tag=PLAYSTATE value=TIED" in ln
               for ln in lines):
        problems.append("no winner recorded")

    # The point of the P1 viewpoint: an opponent card's identity must never
    # appear before something publicly revealed it. (Playing a card IS such a
    # reveal, and the client does stamp that BLOCK_START while the card is
    # still in zone=HAND, so "cardId set in HAND" is not itself a leak.)
    # NB: a hidden descriptor nests brackets -- "UNKNOWN ENTITY [cardType=
    # INVALID]" -- so anchor on id= rather than matching to a closing ].
    opp = 2 if viewpoint == 1 else 1
    revealed = set()
    id_re = re.compile(r"id=(\d+) zone=")
    desc_re = re.compile(r"id=(\d+) zone=\w+ zonePos=-?\d+ cardId=(\w*) player=(\d)")
    full_re = re.compile(r"FULL_ENTITY - Creating ID=(\d+) CardID=(\w+)")
    leaks = []
    for ln in lines:
        if "SHOW_ENTITY - Updating" in ln and re.search(r"CardID=\w+\s*$", ln):
            m = id_re.search(ln)
            if m:
                revealed.add(int(m.group(1)))
        m = full_re.search(ln)
        if m:
            revealed.add(int(m.group(1)))
        for eid, cid, player in desc_re.findall(ln):
            if int(player) == opp and cid and int(eid) not in revealed:
                leaks.append(ln)
    if leaks:
        problems.append("opponent identity leaked before reveal on %d lines"
                        % len(leaks))
    return problems


def write_log(path, lines):
    """Atomic write: a complete game appears, or nothing does."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
