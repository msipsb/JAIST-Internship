"""
run_stage2.py -- orchestrates Stage 2 (GE2E learned embeddings).

Run order (matches the task):
  1. ls/artifact checks, load Stage-1 frame + tokens, rebuild baseline-'both' on the
     44 held-out eval users (the fair head-to-head).
  2. Overfit sanity (8 users) -- training re-ID must go ~1.0.
  3. Variant (a) embed / (b) lsa / (b)+adversarial, 3 seeds each -> eval on 44 users.
  4. Figure + markdown summary answering the three required questions; STOP.

User-level split: eval=44 (>=100 games so the full N-sweep works), val=20 (early
stopping only), train=rest (>=20 games).  No overlap (asserted).  Eval users are
never seen in training -> open set.  Reference/query game halves reuse Stage-1's
exact split seeds so numbers are directly comparable.

CPU note: this machine has CPU-only torch, so the defaults can be reduced with env
vars (STAGE2_STEPS, STAGE2_SEEDS, STAGE2_EVAL_EVERY, STAGE2_SANITY_STEPS).  The
in-code defaults are the full RTX-5090 spec (15k steps, 3 seeds).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hearthstonemap_load import build_game_frame, N_SWEEP
from cardseq_embed import add_role_features
from archetype_infer import infer_archetypes
from stylometry_eval import (CARDSEQ_FEATURES, run_pool, deck_recognition, _l2,
                             build_units, fingerprints, score_matched)
import stylometry_ge2e as g2

OUT = Path("stylometry_out")
CSV = OUT / "stylometry_results.csv"
SEED = 0
SPLIT_SEED = 0
MATCHED_C = 10

STEPS = int(os.environ.get("STAGE2_STEPS", 15000))
SEEDS = [int(x) for x in os.environ.get("STAGE2_SEEDS", "0,1,2").split(",")]
EVAL_EVERY = int(os.environ.get("STAGE2_EVAL_EVERY", 500))
SANITY_STEPS = int(os.environ.get("STAGE2_SANITY_STEPS", 1000))

VARIANTS = [
    ("a_embed",   dict(card_mode="embed", encoder="transformer", adversarial=False)),
    ("b_lsa",     dict(card_mode="lsa",   encoder="transformer", adversarial=False)),
    ("b_lsa_adv", dict(card_mode="lsa",   encoder="transformer", adversarial=True)),
]


# ---------------------------------------------------------------- data prep

def prep_frame():
    df = build_game_frame("human", cache=OUT / "frame_human.pkl", verbose=False).reset_index(drop=True)
    df, role_names = add_role_features(df, verbose=False)
    df = infer_archetypes(df, cache=OUT / "archetype_human.pkl", verbose=False)
    df["all"] = "all"
    classes = sorted(df["hero"].unique())
    arches = sorted(df["archetype"].unique())
    df["_class_idx"] = df["hero"].map({c: i for i, c in enumerate(classes)}).astype(int)
    df["_arche_idx"] = df["archetype"].map({a: i for i, a in enumerate(arches)}).astype(int)
    return df, role_names, len(classes), len(arches)


def make_splits(df):
    counts = df.groupby("user_hash").size()
    rng = np.random.default_rng(SPLIT_SEED)
    elig = np.array(sorted(counts[counts >= 100].index))          # support N=50 eval
    rng.shuffle(elig)
    eval_users = set(elig[:44].tolist())
    rest = [u for u in counts.index if u not in eval_users]
    val_pool = np.array(sorted([u for u in rest if counts[u] >= 40]))
    rng.shuffle(val_pool)
    val_users = set(val_pool[:20].tolist())
    train_users = set(u for u in rest if u not in val_users and counts[u] >= 20)
    assert not (eval_users & val_users), "eval/val overlap"
    assert not (eval_users & train_users), "eval/train overlap"
    assert not (val_users & train_users), "val/train overlap"
    print(f"[split] train={len(train_users)}  val={len(val_users)}  eval={len(eval_users)}  "
          f"(no overlap asserted)")
    return eval_users, val_users, train_users


# ---------------------------------------------------------------- eval helpers

def eval_rows(edf, cols, tag, seed, standardize=False, stage="ge2e"):
    rows = []
    for pool, col in [("all-users", "all"), ("within-class", "hero"),
                      ("within-archetype", "archetype")]:
        rows += run_pool(edf, stage, pool, col, {tag: cols}, N_SWEEP, seed,
                         standardize=standardize)
    rows += deck_recognition(edf, N_SWEEP, C=MATCHED_C, seed=seed, feature_cols=cols,
                             feature_name=tag, standardize=standardize, stage=stage)
    for r in rows:
        r["seed"] = seed
    return rows


def baseline_both_rows(df, role_names, eval_users):
    """Stage-1 'both' hand-features restricted to the 44 eval users (fair head-to-head)."""
    both = CARDSEQ_FEATURES + list(role_names)
    edf = df[df["user_hash"].isin(eval_users)].reset_index(drop=True)
    return eval_rows(edf, both, "baseline_both", SEED, standardize=True, stage="baseline44")


# ---------------------------------------------------------------- sanity

def overfit_sanity(df, games, id2idx, train_users, device):
    print("\n=== Overfit sanity (8 users; training re-ID must approach 1.0) ===")
    counts = df.groupby("user_hash").size()
    users = sorted([u for u in train_users if counts[u] >= 40])[:8]
    model, _hist, _best = g2.train_variant(
        games, df, id2idx, users, users, card_mode="embed", encoder="transformer",
        adversarial=False, seed=0, steps=SANITY_STEPS, P=8, G=10,
        eval_every=max(200, SANITY_STEPS // 4), device=device, verbose=False, tag="sanity")
    edf, cols = g2.embeddings_to_df(model, df, games, users, device)
    units = build_units(edf, "all", 20, SEED)
    R, Q = fingerprints(edf[cols].to_numpy(float), units, standardize=False)
    R, Q = _l2(R), _l2(Q)
    top1 = np.mean([np.argmax(R @ Q[i]) == i for i in range(len(units))])
    print(f"[sanity] 8-user training re-ID full-pool top-1 @N=20 = {top1:.3f} "
          f"(chance {1/len(units):.3f})  -> {'PASS' if top1 >= 0.85 else 'CHECK'}")
    return float(top1)


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 40)
    device = g2.DEVICE
    print(f"device: {device}   steps={STEPS}  seeds={SEEDS}  eval_every={EVAL_EVERY}")

    df, role_names, n_class, n_arche = prep_frame()
    games, id2idx = g2.build_token_cache(frame=build_game_frame("human", cache=OUT / "frame_human.pkl", verbose=False))
    eval_users, val_users, train_users = make_splits(df)

    all_rows = []
    all_rows += baseline_both_rows(df, role_names, eval_users)
    print("[baseline44] rebuilt 'both' on the 44 eval users")

    overfit_sanity(df, games, id2idx, train_users, device)

    for tag, spec in VARIANTS:
        print(f"\n=== Variant {tag}  {spec} ===")
        for seed in SEEDS:
            model, hist, best = g2.train_variant(
                games, df, id2idx, train_users, val_users, seed=seed, steps=STEPS,
                eval_every=EVAL_EVERY, device=device, tag=f"{tag}.s{seed}", **spec)
            edf, cols = g2.embeddings_to_df(model, df, games, eval_users, device)
            all_rows += eval_rows(edf, cols, tag, seed)
            print(f"  [{tag} seed{seed}] best val {best['metric']:.3f} @step {best['step']}")

    # append to results CSV (idempotent for stage2 rows: drop by stage AND by the
    # stage2 feature_set tags, so a re-run never duplicates or leaves stragglers)
    new = pd.DataFrame(all_rows)
    stage2_feats = {t for t, _ in VARIANTS} | {"baseline_both"}
    if CSV.exists():
        old = pd.read_csv(CSV)
        old = old[~old["stage"].isin(["ge2e", "baseline44"])]
        old = old[~old["feature_set"].isin(stage2_feats)]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(CSV, index=False)
    print(f"\nAppended {len(new)} rows -> {CSV}")

    make_figure(out)
    write_summary(out, n_class)


# ---------------------------------------------------------------- figure + summary

def _agg(df, stage, feat, pool, n, retrieval, col="top1"):
    s = df[(df["stage"] == stage) & (df["feature_set"] == feat) & (df["pool"] == pool)
           & (df["N"] == n) & (df["retrieval"] == retrieval)]
    if not len(s):
        return float("nan"), float("nan")
    return float(s[col].mean()), float(s[col].std(ddof=0))


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    series = [("baseline44", "baseline_both", "baseline 'both' (44u)"),
              ("ge2e", "a_embed", "GE2E (a) embed"),
              ("ge2e", "b_lsa", "GE2E (b) LSA"),
              ("ge2e", "b_lsa_adv", "GE2E (b)+adv")]

    def plot(ax, pool, ret, title):
        for stage, feat, label in series:
            xs, ys = [], []
            for n in N_SWEEP:
                m, _ = _agg(out, stage, feat, pool, n, ret)
                if not np.isnan(m):
                    xs.append(n); ys.append(m)
            if xs:
                ax.plot(xs, ys, marker="o", label=label)
        ax.axhline(1.0 / MATCHED_C, ls="--", c="gray", label="chance")
        ax.set(title=title, xlabel="N games / fingerprint", ylabel="top-1", ylim=(0, 1.02))
        ax.legend(fontsize=7)

    plot(axes[0], "all-users", "matched-C10", "all-users (matched-C10)")
    plot(axes[1], "within-archetype", "matched-C10", "within-archetype (matched-C10)")
    plot(axes[2], "pilot|distractors=archetype", "matched-C10",
         "deck-recognition: same-archetype distractors")
    fig.suptitle("Stage 2 -- GE2E learned embeddings vs Stage-1 baseline (44 held-out users)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage2_top1_vs_N.png", dpi=130)
    plt.close(fig)
    print(f"Figure -> {OUT/'fig_stage2_top1_vs_N.png'}")


def write_summary(out, n_class):
    n = 50
    L = ["# Stage 2 -- GE2E learned embeddings (summary)\n"]
    L.append(f"Open-set: 44 held-out eval users (never trained on), reference/query halves = "
             "Stage-1 splits/seeds.  Fingerprint = mean of learned game vectors, matched-C10 "
             "retrieval (chance 0.100).  Baseline 'both' is rebuilt on the SAME 44 users.\n")
    L.append("**Caveat**: eval pool is 44 users vs Stage-1's 164, so full-pool chance differs; "
             "matched-C10 is the apples-to-apples comparison.\n")

    def row(stage, feat, label):
        cells = []
        for pool in ["all-users", "within-class", "within-archetype"]:
            m, sd = _agg(out, stage, feat, pool, n, "matched-C10")
            cells.append(f"{m:.3f}±{sd:.3f}" if not np.isnan(m) else "-")
        return f"| {label} | " + " | ".join(cells) + " |"

    L.append(f"## matched-C10 top-1 @N={n} (mean±sd over {len(SEEDS)} seeds)\n")
    L.append("| model | all-users | within-class | within-archetype |")
    L.append("|-------|----------:|-------------:|-----------------:|")
    L.append(row("baseline44", "baseline_both", "baseline 'both' (44u)"))
    L.append(row("ge2e", "a_embed", "GE2E (a) embed"))
    L.append(row("ge2e", "b_lsa", "GE2E (b) LSA"))
    L.append(row("ge2e", "b_lsa_adv", "GE2E (b)+adversarial"))

    base_wa = _agg(out, "baseline44", "baseline_both", "within-archetype", n, "matched-C10")[0]
    a_wa = _agg(out, "ge2e", "a_embed", "within-archetype", n, "matched-C10")[0]
    b_wa = _agg(out, "ge2e", "b_lsa", "within-archetype", n, "matched-C10")[0]
    adv_wa = _agg(out, "ge2e", "b_lsa_adv", "within-archetype", n, "matched-C10")[0]

    def gap(stage, feat):
        a = _agg(out, stage, feat, "pilot|distractors=all", n, "matched-C10")[0]
        c = _agg(out, stage, feat, "pilot|distractors=archetype", n, "matched-C10")[0]
        return a, c, a - c

    L.append("\n## Deck-recognition gap (identical pilots; all vs same-archetype distractors)\n")
    L.append("| model | distractors=all | distractors=archetype | gap |")
    L.append("|-------|----------------:|----------------------:|----:|")
    for stage, feat, label in [("baseline44", "baseline_both", "baseline 'both'"),
                               ("ge2e", "a_embed", "GE2E (a) embed"),
                               ("ge2e", "b_lsa", "GE2E (b) LSA"),
                               ("ge2e", "b_lsa_adv", "GE2E (b)+adv")]:
        a, c, g = gap(stage, feat)
        L.append(f"| {label} | {a:.3f} | {c:.3f} | {g:.3f} |")

    L.append("\n## Answers\n")
    best_learned = max(v for v in [a_wa, b_wa, adv_wa] if not np.isnan(v)) if any(
        not np.isnan(v) for v in [a_wa, b_wa, adv_wa]) else float("nan")
    L.append(f"1. **Beat baseline within-archetype?** baseline 'both' = {base_wa:.3f}; best learned "
             f"= {best_learned:.3f}. -> {'YES' if best_learned > base_wa else 'NOT YET'} "
             "(matched-C10 top-1 @N=50).")
    _, _, a_gap = gap("ge2e", "a_embed")
    _, _, b_gap = gap("ge2e", "b_lsa")
    _, _, adv_gap = gap("ge2e", "b_lsa_adv")
    verd = "YES, smaller" if adv_gap < b_gap else "no -- it did NOT shrink (in this short run it grew)"
    L.append(f"2. **Adversarial shrinks the deck gap?** Controlled comparison is (b) LSA vs "
             f"(b)+adv (same card rep, +/- gradient-reversal head): (b) gap = {b_gap:.3f}, "
             f"(b)+adv gap = {adv_gap:.3f}. -> {verd}. "
             f"(For reference the deck-leaky learned-embed (a) gap = {a_gap:.3f}, baseline = "
             f"{_agg(out,'baseline44','baseline_both','pilot|distractors=all',n,'matched-C10')[0] - _agg(out,'baseline44','baseline_both','pilot|distractors=archetype',n,'matched-C10')[0]:.3f}.)")
    a_all = _agg(out, "ge2e", "a_embed", "all-users", n, "matched-C10")[0]
    b_all = _agg(out, "ge2e", "b_lsa", "all-users", n, "matched-C10")[0]
    L.append(f"3. **Frozen-LSA (b) trades all-users for within-archetype robustness?** "
             f"(a) all-users {a_all:.3f} / within-arch {a_wa:.3f}; (b) all-users {b_all:.3f} / "
             f"within-arch {b_wa:.3f}.")
    dev = g2.DEVICE.type
    full = (STEPS >= 15000 and len(SEEDS) >= 3)
    if dev == "cuda" and full:
        prov = (f"Full-spec run on {torch.cuda.get_device_name(0)} (bf16): "
                f"steps={STEPS}, seeds={SEEDS}.")
    else:
        prov = (f"Reduced run ({dev.upper()}): steps={STEPS}, seeds={SEEDS}. "
                "Full spec = 15k steps x 3 seeds.")
    L.append(f"\n({prov})\n")

    txt = "\n".join(L)
    (OUT / "stage2_summary.md").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
