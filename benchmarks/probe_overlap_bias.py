"""Probe Von for a lexical-overlap shortcut, forward or inverse.

Motivation: JevBench v1.4 sealed aggregates put Von at 7% on the
`paraphrase_robustness` family (chance ~29%, Laya 50%). Far-below-chance
means a systematic inverted rule, not ignorance. The suspect is
`training/harden_corpus.py`, whose own docstring warns that pushing
gold-is-top-overlap too low "would just teach the inverse shortcut".

Three views, all on public items only (never sealed):

1. **Base split.** Accuracy on items where the gold option is the strict
   highest-overlap option vs items where it is not, plus how often Von's pick
   *is* the top-overlap option and the bottom-overlap option.
   forward shortcut:  acc(gold=top) >> acc(gold!=top), P(pick=top) high
   inverse shortcut:  acc(gold=top) << acc(gold!=top), P(pick=bottom) high

2. **Meaning-preserving perturbations.** Two transforms that change lexical
   overlap without changing which answer is correct:
   A `state_deoverlap`   synonym-swap state words that the gold option echoes
                         (gold becomes *less* overlapping)
   B `wrong_synonymized` synonym-swap wrong options' text
                         (gold becomes *relatively more* overlapping)
   forward shortcut:  acc(A) < base < acc(B)
   inverse shortcut:  acc(A) > base > acc(B)
   no shortcut:       flat

3. **Paraphrase pairs.** original.jsonl ships 36 `-0`/`-1` pairs of the same
   scenario. Same-answer rate and both-correct rate.

For `temporal_numeric` items the dataset ships a `surface_answer` (what a
shallow read yields). P(pick == surface_answer) is reported as a bonus.

Usage (CPU box, 8 cores):
    uv run python -m benchmarks.probe_overlap_bias --ckpt checkpoints/von-1.2 \
        --workers 4 --threads 2 --dump benchmarks/data/overlap_probe_v12.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from training.harden_corpus import (  # noqa: E402
    PROTECTED, SYNONYMS, WORD_RE, content_tokens, deoverlap,
)

DATA_DIR = os.environ.get("JEVBENCH_PUBLIC", "/tmp/scratch/jevbench/datasets/public")
TIER_FILES = {
    "hard": os.path.join(DATA_DIR, "hard.jsonl"),
    "standard": os.path.join(DATA_DIR, "original.jsonl"),
}


# --------------------------------------------------------------------------- data

def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def criteria_map(row: dict) -> Dict[str, str]:
    """label -> option text, as the model sees it (plain shape)."""
    q = row["question"]
    crit = q.get("criteria") or {}
    if q.get("type") == "score" and isinstance(crit, list):
        return {str(i): str(v) for i, v in enumerate(crit)}
    if isinstance(crit, dict):
        return {k: str(v or k) for k, v in crit.items()}
    return {str(i): str(v) for i, v in enumerate(crit or [])}


def gold_label(row: dict) -> str:
    """Gold expressed in the criteria's key space."""
    exp = str(row["expected"]).strip().lower()
    if row["question"].get("type") == "noul":
        return "true" if exp == "yes" else "false"
    return exp


def overlap_rank(state: str, crit: Dict[str, str]) -> Tuple[Dict[str, int], Optional[str], Optional[str]]:
    """Per-option overlap counts plus the strict top/bottom labels (None on tie)."""
    st = content_tokens(state)
    scores = {k: len(content_tokens(v) & st) for k, v in crit.items()}
    vals = sorted(scores.values())
    top = bottom = None
    if vals and vals[-1] > 0 and vals.count(vals[-1]) == 1:
        top = max(scores, key=scores.get)
    if vals and vals.count(vals[0]) == 1:
        bottom = min(scores, key=scores.get)
    return scores, top, bottom


def synonymize(text: str, rng: random.Random) -> Tuple[str, int]:
    """Meaning-preserving synonym swap of every eligible word."""
    swaps = 0

    def replace(match) -> str:
        nonlocal swaps
        word = match.group(0)
        low = word.lower()
        if low in PROTECTED:
            return word
        syn = SYNONYMS.get(low)
        if not syn:
            return word
        swaps += 1
        return syn[0].upper() + syn[1:] if word[0].isupper() else syn

    return WORD_RE.sub(replace, text), swaps


def make_variants(row: dict, rng: random.Random) -> List[Tuple[str, dict]]:
    """(variant_name, row) for base + the two perturbations that actually changed text."""
    out = [("base", row)]
    crit = criteria_map(row)
    gold = gold_label(row)
    if gold not in crit:
        return out
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)

    new_state, swaps = deoverlap(state, crit[gold], rng)
    if swaps:
        r = dict(row)
        r["state"] = new_state
        out.append(("state_deoverlap", r))

    q = dict(row["question"])
    new_crit = dict(crit)
    total = 0
    for k, v in crit.items():
        if k == gold:
            continue
        nv, s = synonymize(v, rng)
        new_crit[k] = nv
        total += s
    if total:
        if q.get("type") == "score" and isinstance(q.get("criteria"), list):
            q["criteria"] = [new_crit[str(i)] for i in range(len(new_crit))]
        else:
            q["criteria"] = new_crit
        r = dict(row)
        r["question"] = q
        out.append(("wrong_synonymized", r))
    return out


# ------------------------------------------------------------------------ scoring

_BACKEND = None


def _init(ckpt: str, threads: int) -> None:
    global _BACKEND
    import torch
    torch.set_num_threads(threads)
    from von.backends.option_marker_backend import OptionMarkerBackend
    _BACKEND = OptionMarkerBackend(checkpoint_dir=ckpt)
    _BACKEND._get_model()


def _score(job: Tuple[str, dict]) -> dict:
    assert _BACKEND is not None
    variant, row = job
    q = row["question"]
    qtype = q.get("type", "choice")
    crit = criteria_map(row)
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    base = {"id": row.get("id"), "variant": variant, "family": row.get("family", "?"),
            "type": qtype, "group": row.get("group")}
    if len(crit) < 2:
        return {**base, "error": "single option"}
    try:
        payload = dict(crit) if qtype != "score" else [crit[k] for k in sorted(crit, key=int)]
        res = _BACKEND.evaluate(state=state, questions={"q": {
            "type": qtype, "instructions": q.get("instructions", ""), "criteria": payload}})
        ans = res.answers["q"]
        if qtype == "noul":
            p = float(getattr(ans, "noul"))
            pick = "true" if p >= 0.5 else "false"
            conf = abs(p - 0.5) * 2
        elif qtype == "score":
            # Official harness: argmax over the returned distribution. round(E[level])
            # regresses every 0-3 answer to 1 or 2 (ordinal 3/12 vs argmax 9/12).
            probs = getattr(ans, "probabilities")
            pick = max(probs, key=probs.get)
            conf = float(getattr(ans, "confidence"))
        else:
            pick, conf = str(getattr(ans, "choice")), float(getattr(ans, "confidence"))
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:160]}

    gold = gold_label(row)
    scores, top, bottom = overlap_rank(state, crit)
    surface = row.get("provenance", {}).get("surface_answer")
    if surface is not None and qtype == "noul":
        surface = "true" if str(surface).lower() == "yes" else "false"
    return {
        **base,
        "pick": pick, "gold": gold, "hit": pick.lower() == gold.lower(), "confidence": conf,
        "overlap": scores, "top": top, "bottom": bottom,
        "gold_is_top": top is not None and top == gold,
        "pick_is_top": top is not None and pick == top,
        "pick_is_bottom": bottom is not None and pick == bottom,
        "surface": str(surface).lower() if surface is not None else None,
        "pick_is_surface": surface is not None and pick.lower() == str(surface).lower(),
    }


def run(ckpt: str, jobs: Sequence[Tuple[str, dict]], workers: int, threads: int) -> List[dict]:
    ordered = sorted(jobs, key=lambda j: -len(str(j[1].get("state", ""))))
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(ckpt, threads)) as pool:
        return pool.map(_score, ordered, chunksize=1)


# ------------------------------------------------------------------------- report

def pct(n: int, d: int) -> str:
    return f"{n / d:6.1%} ({n}/{d})" if d else "   n/a"


def report(results: List[dict]) -> dict:
    good = [r for r in results if not r.get("error")]
    errs = [r for r in results if r.get("error")]
    summary: dict = {"n": len(good), "errors": len(errs)}
    if errs:
        print(f"{len(errs)} errors, first: {errs[0]['error']}")

    print("\n== 1. Base split by overlap (variant=base) ==")
    base = [r for r in good if r["variant"] == "base"]
    for label, sel in [("all", base), ("choice", [r for r in base if r["type"] == "choice"]),
                       ("noul", [r for r in base if r["type"] == "noul"])]:
        gt = [r for r in sel if r["gold_is_top"]]
        gn = [r for r in sel if r["top"] is not None and not r["gold_is_top"]]
        tie = [r for r in sel if r["top"] is None]
        strict = gt + gn
        line = {
            "acc": pct(sum(r["hit"] for r in sel), len(sel)),
            "acc_gold_top": pct(sum(r["hit"] for r in gt), len(gt)),
            "acc_gold_not_top": pct(sum(r["hit"] for r in gn), len(gn)),
            "acc_tie": pct(sum(r["hit"] for r in tie), len(tie)),
            "P_pick_top": pct(sum(r["pick_is_top"] for r in strict), len(strict)),
            "P_pick_bottom": pct(sum(r["pick_is_bottom"] for r in strict), len(strict)),
        }
        summary[f"base_{label}"] = line
        print(f"  [{label}]")
        for k, v in line.items():
            print(f"    {k:<18}{v}")

    print("\n== 2. Perturbations (paired on items that have every variant) ==")
    by_id: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in good:
        by_id[r["id"]][r["variant"]] = r
    for variant in ("state_deoverlap", "wrong_synonymized"):
        pairs = [(v["base"], v[variant]) for v in by_id.values() if "base" in v and variant in v]
        if not pairs:
            continue
        b = sum(x["hit"] for x, _ in pairs)
        p = sum(y["hit"] for _, y in pairs)
        flips = sum(x["pick"] != y["pick"] for x, y in pairs)
        dtop = sum(y["pick_is_top"] for _, y in pairs) - sum(x["pick_is_top"] for x, _ in pairs)
        line = {"n": len(pairs), "acc_base": b / len(pairs), "acc_variant": p / len(pairs),
                "delta_pp": 100 * (p - b) / len(pairs), "answer_flips": flips,
                "delta_pick_top": dtop}
        summary[variant] = line
        print(f"  {variant:<18} n={len(pairs):<4} base {b / len(pairs):.1%} -> {p / len(pairs):.1%} "
              f"({line['delta_pp']:+.1f}pp)  flips={flips}  Δpick_top={dtop:+d}")

    print("\n== 3. Paraphrase pairs (standard tier, base) ==")
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in base:
        if r["group"] and r["id"] and r["id"][-2:] in ("-0", "-1"):
            groups[r["group"]].append(r)
    pairs2 = [g for g in groups.values() if len(g) == 2]
    if pairs2:
        same = sum(g[0]["pick"] == g[1]["pick"] for g in pairs2)
        both = sum(g[0]["hit"] and g[1]["hit"] for g in pairs2)
        either = sum(g[0]["hit"] != g[1]["hit"] for g in pairs2)
        summary["paraphrase_pairs"] = {"n": len(pairs2), "same_answer": same / len(pairs2),
                                       "both_correct": both / len(pairs2),
                                       "exactly_one_correct": either / len(pairs2)}
        print(f"  n={len(pairs2)}  same-answer {same / len(pairs2):.1%}  both-correct "
              f"{both / len(pairs2):.1%}  exactly-one-correct {either / len(pairs2):.1%}")

    print("\n== 4. temporal_numeric (hard tier, base) ==")
    tn = [r for r in base if r["family"] == "temporal_numeric"]
    if tn:
        sur = [r for r in tn if r["surface"] is not None]
        line = {"n": len(tn), "acc": pct(sum(r["hit"] for r in tn), len(tn)),
                "P_pick_surface": pct(sum(r["pick_is_surface"] for r in sur), len(sur))}
        summary["temporal_numeric"] = line
        for k, v in line.items():
            print(f"    {k:<18}{v}")
        for r in sorted(tn, key=lambda r: r["id"]):
            mark = "✓" if r["hit"] else ("S" if r["pick_is_surface"] else "✗")
            print(f"    {mark} {r['id']:<40} pick={r['pick']:<6} gold={r['gold']:<6} conf={r['confidence']:.2f}")

    print("\n== 5. Per-family accuracy (hard tier, base) ==")
    fam: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in base:
        fam[r["family"]][0] += r["hit"]
        fam[r["family"]][1] += 1
    for f, (c, t) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
        print(f"    {f:<18} {pct(c, t)}")
    summary["by_family"] = {f: c / t for f, (c, t) in fam.items()}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/von-1.2")
    ap.add_argument("--tiers", default="hard,standard")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-perturb", action="store_true")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: List[dict] = []
    for tier in args.tiers.split(","):
        rows.extend(load_rows(TIER_FILES[tier.strip()]))
    if args.limit:
        rows = rows[:args.limit]

    jobs: List[Tuple[str, dict]] = []
    for row in rows:
        jobs.extend([("base", row)] if args.no_perturb else make_variants(row, rng))
    print(f"{len(rows)} items -> {len(jobs)} jobs on {os.path.basename(args.ckpt)}")

    t0 = time.time()
    results = run(args.ckpt, jobs, args.workers, args.threads)
    print(f"scored in {time.time() - t0:.0f}s")
    summary = report(results)
    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "summary": summary, "results": results}, f, indent=1)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
