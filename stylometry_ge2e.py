"""
stylometry_ge2e.py -- Stage 2: learned game embeddings (GE2E) for Hearthstone
behavioral stylometry.  Reuses every Stage-1 module; only the game -> vector step
is replaced by a small neural encoder trained with the GE2E loss.

Pieces:
  * build_token_cache -- tokenizes each game into (card_idx, [log(mana+1), turn/30,
    is_me, has_coin]) aligned 1:1 with the Stage-1 human frame (Coin excluded, cap
    MAX_TOK tokens).  Card vocabulary + optional frozen LSA card matrix.
  * GameEncoder -- Transformer (2L, d=128, 4 heads, GELU) or GRU; masked mean-pool
    -> Linear -> L2-normalized 128-d game vector.  card_vec flag:
        "embed" -> learned nn.Embedding(n_cards, 32)      (deck-leaky, upper bound)
        "lsa"   -> frozen LSA text vectors (cardseq_embed) (deck-safer)
  * ge2e_loss -- GE2E softmax variant (Wan et al. 2018), P users x G games,
    self-excluded own centroid, learned scale w>0 and bias b.
  * gradient-reversal deck-adversarial heads (hero_class, archetype_id), lambda
    warmup -- variant (b)+adversarial.
  * train_variant / encode_games / embeddings_to_df -- training loop with
    open-set validation early stopping, and embedding export for stylometry_eval.

No time/duration/rank/legend field is ever read.  Fixed seeds throughout.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from hearthstonemap_load import build_game_frame, iter_games, COIN_NAME, _me_card_ids
from cardseq_embed import load_lsa_card_vectors
from stylometry_eval import run_pool

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "stylometry_out"
TOKENS_CACHE = OUT_DIR / "tokens_human.pkl"

MAX_TOK = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- tokenization

def build_token_cache(cache=TOKENS_CACHE, frame=None, verbose=True):
    """Tokenize every human game, aligned 1:1 with the Stage-1 human frame.

    Returns (games, id2idx) where games[i] = (card_idx:int32[L], feats:float32[L,4]).
    """
    if cache and Path(cache).exists():
        d = pd.read_pickle(cache)
        if verbose:
            print(f"[tokens] loaded cache: {len(d['games']):,} games, "
                  f"{len(d['id2idx'])} cards")
        return d["games"], d["id2idx"]

    id2idx = {}

    def idx(cid):
        j = id2idx.get(cid)
        if j is None:
            j = len(id2idx) + 1          # 0 reserved for pad/unknown
            id2idx[cid] = j
        return j

    games, users = [], []
    for g in iter_games("human"):
        hist = g["card_history"]
        if not _me_card_ids(hist):       # same drop rule as build_game_frame
            continue
        coin = 1.0 if g["coin"] else 0.0
        cidx, feats = [], []
        for h in hist:
            card = h["card"]
            if card.get("name") == COIN_NAME:
                continue
            cid = card.get("id")
            if not cid:
                continue
            mana = card.get("mana")
            mana = 0.0 if mana is None else float(mana)
            cidx.append(idx(cid))
            feats.append((math.log1p(mana), float(h["turn"]) / 30.0,
                          1.0 if h.get("player") == "me" else 0.0, coin))
            if len(cidx) >= MAX_TOK:
                break
        games.append((np.asarray(cidx, dtype=np.int32),
                      np.asarray(feats, dtype=np.float32)))
        users.append(g["user_hash"])

    if frame is not None:
        assert len(games) == len(frame), (len(games), len(frame))
        assert users == list(frame["user_hash"]), "token/frame row misalignment"
    if verbose:
        print(f"[tokens] built {len(games):,} games, {len(id2idx)} cards")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"games": games, "id2idx": id2idx}, cache)
    return games, id2idx


def lsa_matrix_for_vocab(id2idx, dims=32):
    """[vocab+1, dim] frozen LSA matrix aligned to card indices (row 0 = pad = 0)."""
    vecs, dim = load_lsa_card_vectors(dims=dims, cache=OUT_DIR / "lsa_cards.pkl")
    M = np.zeros((len(id2idx) + 1, dim), dtype=np.float32)
    for cid, j in id2idx.items():
        v = vecs.get(cid)
        if v is not None:
            M[j] = v
    return M


def collate(games_subset, device=DEVICE):
    """List of (card_idx, feats) -> padded (card_idx[B,T], feats[B,T,4], mask[B,T])."""
    B = len(games_subset)
    T = max(len(c) for c, _ in games_subset)
    card_idx = np.zeros((B, T), dtype=np.int64)
    feats = np.zeros((B, T, 4), dtype=np.float32)
    mask = np.zeros((B, T), dtype=bool)
    for i, (c, f) in enumerate(games_subset):
        L = len(c)
        card_idx[i, :L] = c
        feats[i, :L] = f
        mask[i, :L] = True
    return (torch.from_numpy(card_idx).to(device),
            torch.from_numpy(feats).to(device),
            torch.from_numpy(mask).to(device))


# ---------------------------------------------------------------- model

class GameEncoder(nn.Module):
    def __init__(self, vocab_size, card_mode="embed", lsa_matrix=None,
                 encoder="transformer", d_model=128, n_heads=4, n_layers=2,
                 dropout=0.1, card_dim=32, out_dim=128, max_len=MAX_TOK):
        super().__init__()
        self.card_mode = card_mode
        if card_mode == "embed":
            self.card_emb = nn.Embedding(vocab_size + 1, card_dim, padding_idx=0)
            cvdim = card_dim
        else:                                       # frozen LSA text vectors
            self.register_buffer("lsa", torch.tensor(lsa_matrix, dtype=torch.float32))
            cvdim = lsa_matrix.shape[1]
        self.in_proj = nn.Linear(cvdim + 4, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.encoder_kind = encoder
        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layers)
        else:
            self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, card_idx, feats, mask):
        cv = self.card_emb(card_idx) if self.card_mode == "embed" else self.lsa[card_idx]
        x = torch.cat([cv, feats], dim=-1)
        h = self.in_proj(x) + self.pos[:x.size(1)]
        if self.encoder_kind == "transformer":
            h = self.enc(h, src_key_padding_mask=~mask)
        else:
            h, _ = self.gru(h)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(self.out(pooled), dim=-1)


class Stage2Model(nn.Module):
    """Encoder + GE2E scale/bias + optional deck-adversarial heads."""
    def __init__(self, encoder, out_dim=128, n_class=9, n_arche=64, adversarial=False):
        super().__init__()
        self.encoder = encoder
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))
        self.adversarial = adversarial
        if adversarial:
            self.head_class = nn.Linear(out_dim, n_class)
            self.head_arche = nn.Linear(out_dim, n_arche)

    def forward(self, *batch):
        return self.encoder(*batch)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lamb * grad, None


def grad_reverse(x, lamb):
    return _GradReverse.apply(x, lamb)


# ---------------------------------------------------------------- GE2E loss

def ge2e_loss(emb, P, G, w, b):
    """GE2E softmax loss. emb: [P*G, D] L2-normalized. Own centroid excludes self.

    Computed in fp32 with autocast disabled -- the einsum/softmax are cheap and
    fp32 avoids bf16 dtype clashes and keeps the softmax numerically stable.
    """
    with torch.autocast(device_type=emb.device.type, enabled=False):
        emb = emb.float()
        D = emb.size(1)
        e = emb.view(P, G, D)
        centroids = F.normalize(e.mean(1), dim=-1)                 # [P, D]
        c_excl = F.normalize((e.sum(1, keepdim=True) - e) / (G - 1), dim=-1)  # [P,G,D]
        sims = torch.einsum("pgd,qd->pgq", e, centroids)           # [P,G,P] cos to all
        own = (e * c_excl).sum(-1)                                 # [P,G] cos to own-excl
        ar = torch.arange(P, device=emb.device)
        sims[ar, :, ar] = own                                      # overwrite own column
        logits = (w.float() * sims + b.float()).reshape(P * G, P)
        target = ar.repeat_interleave(G)
        return F.cross_entropy(logits, target)


# ---------------------------------------------------------------- training

def _user_rows(frame, users):
    """user_hash -> np.array of frame row positions (RangeIndex assumed)."""
    return {u: sub.index.to_numpy() for u, sub in frame.groupby("user_hash") if u in set(users)}


def encode_games(model, games, rows, device=DEVICE, batch=256):
    """Encode the games at `rows` -> np.float32[len(rows), out_dim]."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = [games[r] for r in rows[i:i + batch]]
            ci, fe, ma = collate(chunk, device)
            outs.append(model.encoder(ci, fe, ma).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 128), np.float32)


def embeddings_to_df(model, frame, games, users, device=DEVICE, extra_cols=("hero", "archetype")):
    """Build an eval frame for `users`: emb_0..emb_{D-1} + grouping cols + 'all'."""
    sub = frame[frame["user_hash"].isin(set(users))].copy()
    rows = sub.index.to_numpy()
    emb = encode_games(model, games, rows, device)
    sub = sub.reset_index(drop=True)
    cols = [f"emb_{j}" for j in range(emb.shape[1])]
    edf = pd.DataFrame(emb, columns=cols, index=sub.index)
    sub = pd.concat([sub[["user_hash", *extra_cols]], edf], axis=1)
    sub["all"] = "all"
    return sub, cols


def _val_metric(model, frame, games, val_users, device, n=20, seed=0):
    """Open-set early-stop metric: all-users matched-C10 top-1 @ N on validation users."""
    vdf, cols = embeddings_to_df(model, frame, games, val_users, device)
    rows = run_pool(vdf, "val", "all-users", "all", {"emb": cols}, [n], seed,
                    standardize=False)
    for r in rows:
        if r["retrieval"] == "matched-C10":
            return r["top1"]
    return 0.0


def train_variant(games, frame, id2idx, train_users, val_users, *,
                  card_mode="embed", encoder="transformer", adversarial=False,
                  seed=0, steps=15000, P=16, G=10, lr=1e-3, warmup_frac=0.05,
                  adv_warmup_frac=0.20, eval_every=500, class_map=None,
                  arche_map=None, device=DEVICE, verbose=True, tag=""):
    """Train one GE2E variant; return (best_model_state, history)."""
    set_seed(seed)
    vocab = len(id2idx)
    lsa = lsa_matrix_for_vocab(id2idx, dims=32) if card_mode == "lsa" else None
    enc = GameEncoder(vocab, card_mode=card_mode, lsa_matrix=lsa, encoder=encoder)
    n_arche = (max(arche_map.values()) + 1) if arche_map else 64
    model = Stage2Model(enc, n_class=9, n_arche=n_arche, adversarial=adversarial).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, int(steps * warmup_frac))

    def lr_at(t):
        if t < warm:
            return t / warm
        p = (t - warm) / max(1, steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    u2rows = _user_rows(frame, train_users)
    tusers = [u for u in train_users if len(u2rows.get(u, [])) >= 1]
    cls_arr = frame["_class_idx"].to_numpy() if adversarial else None
    arc_arr = frame["_arche_idx"].to_numpy() if adversarial else None

    best = {"metric": -1.0, "state": None, "step": 0}
    hist = []
    for step in range(steps):
        model.train()
        pick = random.sample(tusers, P)
        batch_rows = []
        for u in pick:
            rr = u2rows[u]
            take = np.random.choice(rr, G, replace=len(rr) < G)
            batch_rows.extend(take.tolist())
        ci, fe, ma = collate([games[r] for r in batch_rows], device)
        # bf16 autocast on GPU (RTX 5090); no-op on CPU
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            emb = model.encoder(ci, fe, ma)
            loss = ge2e_loss(emb, P, G, model.w, model.b)
            if adversarial:
                lamb = min(1.0, step / max(1, int(steps * adv_warmup_frac)))
                rev = grad_reverse(emb, lamb)
                yc = torch.tensor(cls_arr[batch_rows], device=device)
                ya = torch.tensor(arc_arr[batch_rows], device=device)
                loss = loss + F.cross_entropy(model.head_class(rev), yc) \
                            + F.cross_entropy(model.head_arche(rev), ya)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        sched.step()
        model.w.data.clamp_(min=1e-6)

        if (step + 1) % eval_every == 0 or step == steps - 1:
            m = _val_metric(model, frame, games, val_users, device, seed=seed)
            hist.append((step + 1, float(loss.item()), m))
            if m > best["metric"]:
                best = {"metric": m, "state": {k: v.detach().cpu().clone()
                                               for k, v in model.state_dict().items()},
                        "step": step + 1}
            if verbose:
                print(f"    [{tag}] step {step+1:5d}  loss {loss.item():.3f}  "
                      f"val matched-C10@N20 {m:.3f}  (best {best['metric']:.3f})")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, hist, best
