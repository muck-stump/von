"""Does Von react to numbers at all?

JevBench v1.4 sealed `temporal_numeric` is the largest sealed family (56 items,
18% of the set) and Von's worst relative to chance: 9% (Laya 20%, jeff 29%).
On the 15 public items Von scores 4/15 and gives the dataset's `surface_answer`
7/15 times. Before writing numeric training data, check whether the model
reads numerals at all: scramble every number in the state and count how often
the answer moves. A model that never changes its pick is not reading digits;
one that changes constantly is reading them but computing wrong -- different
fixes.

Variants per item (k random scrambles): each integer/decimal token in the
state is replaced by a different random number of the same digit length;
dates and times are covered because their components are digit runs.
Gold is unknown after scrambling, so only the *flip rate* is reported, plus
the pick distribution.

Usage:
    uv run python -m benchmarks.probe_numeral_sensitivity --ckpt checkpoints/von-1.2 \
        --k 4 --workers 4 --threads 2 --dump benchmarks/data/numeral_probe_v12.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from benchmarks.probe_overlap_bias import criteria_map, gold_label  # noqa: E402

HARD = os.path.join(os.environ.get("JEVBENCH_PUBLIC", "/tmp/scratch/jevbench/datasets/public"), "hard.jsonl")
NUM_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?![A-Za-z])")


def scramble_numbers(text: str, rng: random.Random) -> str:
    def rep(m: re.Match) -> str:
        s = m.group(0)
        out = []
        for ch in s:
            if ch.isdigit():
                d = rng.choice([c for c in "0123456789" if c != ch])
                out.append(d)
            else:
                out.append(ch)
        return "".join(out)
    return NUM_RE.sub(rep, text)


def variants(row: dict, k: int, rng: random.Random) -> List[dict]:
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    q = row["question"]
    base = {"id": row["id"], "type": q.get("type", "choice"), "instructions": q.get("instructions", ""),
            "criteria": criteria_map(row), "gold": gold_label(row),
            "surface": str(row.get("provenance", {}).get("surface_answer", "")).lower() or None}
    out = [{**base, "variant": "base", "state": state}]
    for i in range(k):
        out.append({**base, "variant": f"scramble_{i}", "state": scramble_numbers(state, rng)})
    return out


_BACKEND = None


def _init(ckpt: str, threads: int) -> None:
    global _BACKEND
    import torch
    torch.set_num_threads(threads)
    from von.backends.option_marker_backend import OptionMarkerBackend
    _BACKEND = OptionMarkerBackend(checkpoint_dir=ckpt)
    _BACKEND._get_model()


def _score(job: dict) -> dict:
    assert _BACKEND is not None
    out = {k: job[k] for k in ("id", "variant", "type", "gold", "surface")}
    crit = job["criteria"]
    qtype = job["type"]
    try:
        payload = dict(crit) if qtype != "score" else [crit[k] for k in sorted(crit, key=int)]
        res = _BACKEND.evaluate(state=job["state"], questions={"q": {
            "type": qtype, "instructions": job["instructions"], "criteria": payload}})
        ans = res.answers["q"]
        if qtype == "noul":
            pick = "true" if getattr(ans, "noul") >= 0.5 else "false"
        elif qtype == "score":
            probs = getattr(ans, "probabilities")
            pick = max(probs, key=probs.get)  # argmax, as the official harness reads it
        else:
            pick = str(getattr(ans, "choice"))
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": f"{type(exc).__name__}: {exc}"[:160]}
    return {**out, "pick": pick}


def report(results: List[dict]) -> dict:
    good = [r for r in results if not r.get("error")]
    by_id: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in good:
        by_id[r["id"]][r["variant"]] = r
    flips = total = 0
    unchanged_items = 0
    print(f"\n{len(by_id)} temporal_numeric items")
    print(f"  {'id':<40} {'gold':<20} {'base':<20} scrambled picks")
    for iid, vs in sorted(by_id.items()):
        base = vs["base"]
        scr = [v["pick"] for k, v in vs.items() if k != "base"]
        f = sum(p != base["pick"] for p in scr)
        flips += f
        total += len(scr)
        unchanged_items += f == 0
        mark = "✓" if base["pick"].lower() == base["gold"].lower() else ("S" if base["surface"] and base["pick"].lower() == base["surface"] else "✗")
        print(f"  {mark} {iid:<38} {base['gold'][:20]:<20} {base['pick'][:20]:<20} {dict(Counter(scr))}")
    summary = {"items": len(by_id), "flip_rate": flips / max(1, total),
               "items_never_flip": unchanged_items}
    print(f"\n  flip rate under number scrambling: {summary['flip_rate']:.1%}   "
          f"items whose pick never moved: {unchanged_items}/{len(by_id)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/von-1.2")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(HARD, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    jobs = [v for r in rows if r.get("family") == "temporal_numeric" for v in variants(r, args.k, rng)]
    jobs.sort(key=lambda j: -len(j["state"]))
    print(f"{len(jobs)} jobs on {os.path.basename(args.ckpt)}")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.ckpt, args.threads)) as pool:
        results = pool.map(_score, jobs, chunksize=1)
    print(f"scored in {time.time() - t0:.0f}s")
    summary = report(results)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "summary": summary, "results": results}, f, indent=1)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
