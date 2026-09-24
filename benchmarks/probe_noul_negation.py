"""Does Von's Noul path read meaning or surface polarity of the true/false descriptions?

`probe_overlap_bias.py` measured a forward overlap dependence on the Noul path
(73% accuracy when the gold description is the overlap winner, 27% otherwise).
If Von decides Noul by "which description echoes the state", then a
meaning-preserving rewrite that swaps the surface polarity of the descriptions
should flip its answers wholesale -- and a sealed family built around
paraphrased/negated rubrics would land far below chance, as
`paraphrase_robustness` did (7%).

Variants on every public Noul item (gold never changes):
  base      shipped criteria
  neg_wrap  true  <- "It is not the case that " + <old false description>
            false <- "It is not the case that " + <old true description>
  swap_ids  criteria texts swapped between the keys, gold key swapped too,
            i.e. the identical decision with the labels renamed. Any accuracy
            change here is pure label/position bias, not semantics.

Usage:
    uv run python -m benchmarks.probe_noul_negation --ckpt checkpoints/von-1.2 \
        --workers 4 --threads 2 --dump benchmarks/data/noul_negation_v12.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_DIR = os.environ.get("JEVBENCH_PUBLIC", "/tmp/scratch/jevbench/datasets/public")
FILES = [os.path.join(DATA_DIR, f) for f in ("hard.jsonl", "original.jsonl", "easy.jsonl")]


def _lc_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def variants(row: dict) -> List[dict]:
    crit = row["question"]["criteria"]
    t, f = str(crit["true"]), str(crit["false"])
    gold = "true" if str(row["expected"]).lower() == "yes" else "false"
    base = {"id": row["id"], "family": row.get("family", "?"), "state": row["state"],
            "instructions": row["question"].get("instructions", "")}
    return [
        {**base, "variant": "base", "criteria": {"true": t, "false": f}, "gold": gold},
        {**base, "variant": "neg_wrap",
         "criteria": {"true": "It is not the case that " + _lc_first(f),
                      "false": "It is not the case that " + _lc_first(t)}, "gold": gold},
        {**base, "variant": "swap_ids", "criteria": {"true": f, "false": t},
         "gold": "false" if gold == "true" else "true"},
    ]


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
    state = job["state"] if isinstance(job["state"], str) else json.dumps(job["state"], ensure_ascii=False)
    out = {k: job[k] for k in ("id", "family", "variant", "gold")}
    try:
        res = _BACKEND.evaluate(state=state, questions={"q": {
            "type": "noul", "instructions": job["instructions"], "criteria": job["criteria"]}})
        p = float(getattr(res.answers["q"], "noul"))
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": f"{type(exc).__name__}: {exc}"[:160]}
    pick = "true" if p >= 0.5 else "false"
    return {**out, "p_true": p, "pick": pick, "hit": pick == job["gold"]}


def report(results: List[dict]) -> dict:
    good = [r for r in results if not r.get("error")]
    by_id: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in good:
        by_id[r["id"]][r["variant"]] = r
    summary: dict = {}
    print(f"\n{len(by_id)} noul items")
    for v in ("base", "neg_wrap", "swap_ids"):
        rs = [x[v] for x in by_id.values() if v in x]
        acc = sum(r["hit"] for r in rs) / len(rs)
        say_true = sum(r["pick"] == "true" for r in rs) / len(rs)
        line = {"n": len(rs), "acc": acc, "P_say_true": say_true}
        if v != "base":
            pairs = [(x["base"], x[v]) for x in by_id.values() if "base" in x and v in x]
            # for swap_ids the *decision* is the same when picks are opposite keys
            same = (sum(a["pick"] != b["pick"] for a, b in pairs) if v == "swap_ids"
                    else sum(a["pick"] == b["pick"] for a, b in pairs))
            line["same_decision"] = same / len(pairs)
        summary[v] = line
        print(f"  {v:<10} acc {acc:6.1%}   P(say true) {say_true:5.1%}"
              + (f"   same decision as base {line['same_decision']:5.1%}" if v != "base" else ""))
    fam: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for x in by_id.values():
        for v, r in x.items():
            fam[r["family"]][v].append(int(r["hit"]))
    print("\n  per family (base / neg_wrap / swap_ids):")
    for f, d in sorted(fam.items(), key=lambda kv: -len(kv[1]["base"])):
        cells = "  ".join(f"{sum(d[v]) / len(d[v]):5.0%}" if d[v] else "  n/a" for v in ("base", "neg_wrap", "swap_ids"))
        print(f"    {f:<18} n={len(d['base']):<3} {cells}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/von-1.2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    rows = []
    for path in FILES:
        with open(path, "r", encoding="utf-8") as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    jobs = [v for r in rows if r["question"].get("type") == "noul" and r["question"].get("criteria")
            for v in variants(r)]
    jobs.sort(key=lambda j: -len(str(j["state"])))
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
