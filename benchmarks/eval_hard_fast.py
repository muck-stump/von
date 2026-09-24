"""Fast, parallel JevBench hard-tier scorer with swappable adapter shapes.

Serial scoring took ~11 minutes per checkpoint, which is too slow to iterate on
an adapter that currently reproduces 20.7% where the official harness measured
36.9%. The work is pure compute -- 122,013 packed tokens over 111 items -- so
the fix is parallelism plus load balancing, not cleverness.

Two things make it fast:

* Four worker processes at one torch thread each. Measured on this 4-core box,
  a single call is 877ms at 1 thread and 468ms at 4, so threads scale at about
  0.47x while processes scale at ~1.0x. Four independent workers beat one
  four-threaded worker by roughly 2x.
* Longest-first scheduling. Item cost spans 501 tokens at the median to 3,677
  at the max, so handing out work in file order leaves one worker holding a
  3.6k-token item while the others idle.

`--shape` selects how the request is built, because the gap between this
harness and the official number has to be found empirically rather than
guessed. Shapes are the hypotheses.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_JEVBENCH_PUBLIC = os.environ.get("JEVBENCH_PUBLIC", "/tmp/scratch/jevbench/datasets/public")
TIER_FILES = {
    "hard": os.path.join(_JEVBENCH_PUBLIC, "hard.jsonl"),
    "standard": os.path.join(_JEVBENCH_PUBLIC, "original.jsonl"),
    "easy": os.path.join(_JEVBENCH_PUBLIC, "easy.jsonl"),
}


def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_request(row: dict, shape: str) -> Tuple[str, str, Dict[str, str]]:
    """Return (state, instructions, criteria) for one item under a given shape."""
    q = row["question"]
    crit = q.get("criteria") or {}
    labels = row.get("labels") or list(crit)
    state = row["state"] if isinstance(row["state"], str) else json.dumps(
        row["state"], ensure_ascii=False)
    instructions = q.get("instructions", "")

    qtype = q.get("type", "choice")
    if qtype == "score" and isinstance(crit, list):
        # Score ships an ordered list of level descriptions, not a mapping.
        criteria = {str(i): str(v) for i, v in enumerate(crit)}
    elif isinstance(crit, dict):
        criteria = {k: (v or k) for k, v in crit.items()}
    else:
        criteria = {str(i): str(v) for i, v in enumerate(crit or [])}

    if shape == "plain":
        # What the current harness does: descriptions only.
        return state, instructions, criteria

    if shape == "rubric":
        # What local_openjev does: append the rubric JSON to the instructions,
        # so the allowed answers appear in the instruction text as well.
        rubric = json.dumps(criteria, ensure_ascii=False)
        return state, f"{instructions}\nAllowed answers and rubric: {rubric}", criteria

    if shape == "labels":
        # Descriptions replaced by the bare label, testing whether the long
        # rubric text is crowding the state out of the context window.
        return state, instructions, {k: k.replace("_", " ") for k in labels}

    if shape == "label_desc":
        # Label prefixed to its description, so the marker sees the identifier
        # it must return adjacent to the rationale for returning it.
        return state, instructions, {
            k: f"{k.replace('_', ' ')}: {v}" for k, v in criteria.items()}

    raise SystemExit(f"unknown shape: {shape}")


_BACKEND = None
_SHAPE = "plain"
_CKPT = ""


def _init(ckpt: str, shape: str, threads: int) -> None:
    global _BACKEND, _SHAPE, _CKPT
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    from von.backends.option_marker_backend import OptionMarkerBackend
    _SHAPE, _CKPT = shape, ckpt
    _BACKEND = OptionMarkerBackend(checkpoint_dir=ckpt)
    _BACKEND._get_model()  # pay the load once per worker, not per item


def _score(row: dict) -> Optional[dict]:
    state, instructions, criteria = build_request(row, _SHAPE)
    if len(criteria) < 2:
        return None
    # Route on the item's declared type. Forcing every item through `choice`
    # skipped the noul path entirely for 38 of 111 hard items (34%), including
    # its zero-shot debiasing, so the harness was not measuring what ships.
    qtype = row["question"].get("type", "choice")
    try:
        # Score's criteria is an ordered list of level descriptions; Choice and
        # Noul take a mapping. Sending the wrong shape is a validation error,
        # not a silent wrong answer, which is how these six surfaced.
        payload = dict(criteria) if qtype != "score" else [
            criteria[k] for k in sorted(criteria, key=int)]
        res = _BACKEND.evaluate(state=state, questions={"q": {
            "type": qtype,
            "instructions": instructions,
            "criteria": payload,
        }})
        ans = res.answers["q"]
        if qtype == "noul":
            pick = "yes" if ans.noul >= 0.5 else "no"
            conf = abs(ans.noul - 0.5) * 2
        elif qtype == "score":
            # ans.score is a continuous position on the scale (1.62), while the
            # dataset expects a discrete level. Comparing the raw float as a
            # string scored every score item wrong regardless of the answer.
            # JevBench scores "argmax over the exact label set" on the returned
            # distribution. round(E[level]) collapses 0-3 scales to 1 or 2:
            # public ordinal reads 3/12 that way and 9/12 by argmax.
            pick = max(ans.probabilities, key=ans.probabilities.get)
            conf = ans.confidence
        else:
            pick, conf = ans.choice, ans.confidence
    except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
        return {"id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"[:120]}
    # Score items carry an int expected value while every pick is a string, so
    # a raw == comparison is False for every score item no matter the answer.
    expected = row["expected"]
    return {
        "id": row.get("id"),
        "family": row.get("family", "?"),
        "pick": pick,
        "expected": expected,
        "hit": str(pick).strip().lower() == str(expected).strip().lower(),
        "confidence": conf,
    }


def evaluate(ckpt: str, shape: str, workers: int, threads: int,
             rows: List[dict]) -> Tuple[List[dict], float]:
    # Longest first: a 3.6k-token item handed out last strands three workers.
    ordered = sorted(rows, key=lambda r: -len(str(r.get("state", ""))))
    t0 = time.time()
    with mp.get_context("spawn").Pool(
            workers, initializer=_init, initargs=(ckpt, shape, threads)) as pool:
        out = [r for r in pool.map(_score, ordered, chunksize=1) if r]
    return out, time.time() - t0


def report(name: str, results: List[dict], secs: float) -> float:
    errs = [r for r in results if r.get("error")]
    good = [r for r in results if not r.get("error")]
    hits = sum(1 for r in good if r["hit"])
    n = len(good)
    acc = hits / max(n, 1)
    print(f"{name}: {hits}/{n} = {acc:.1%}   ({secs:.0f}s)")
    if errs:
        print(f"   {len(errs)} errors, first: {errs[0]['error']}")
    fam: Dict[str, List[int]] = {}
    for r in good:
        d = fam.setdefault(r["family"], [0, 0])
        d[1] += 1
        d[0] += r["hit"]
    for f, (c, t) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
        print(f"   {f:<18} {c:>3}/{t:<3} {c / t:.0%}")
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/von-option-marker-universal")
    ap.add_argument("--shape", default="plain")
    ap.add_argument("--shapes", default="", help="comma list to sweep")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--tier", default="hard", choices=list(TIER_FILES))
    args = ap.parse_args()

    rows = load_rows(TIER_FILES[args.tier])
    if args.limit:
        rows = rows[:args.limit]

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()] or [args.shape]
    best = None
    for shape in shapes:
        results, secs = evaluate(args.ckpt, shape, args.workers, args.threads, rows)
        acc = report(f"{os.path.basename(args.ckpt)} [{shape}]", results, secs)
        if best is None or acc > best[1]:
            best = (shape, acc, results)
        print()
    if args.dump and best:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"shape": best[0], "accuracy": best[1],
                       "results": best[2]}, f, indent=2)
        print(f"wrote {args.dump} (best shape: {best[0]} at {best[1]:.1%})")


if __name__ == "__main__":
    main()
