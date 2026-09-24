"""Probe Von on paraphrase-vs-lexical-twin decisions, PAWS-style.

JevBench v1.4 sealed `paraphrase_robustness`: Von 7% (chance ~29%, Laya 50%,
kev-0.6B 64%). `probe_overlap_bias.py` found no overlap shortcut on ordinary
public Choice items, so the failure is not a general bias -- it needs the
specific adversarial shape where a *wrong* option is a near-verbatim copy of
the state and the *right* option says the same thing in different words.

PAWS is exactly that material: label-0 pairs share ~90% of their words and
differ in meaning (word-order / entity swaps). Two shapes are tested, both
built from the PAWS test split (never used in training; train split was):

  noul    state = both sentences, "Do they have the same meaning?"
          Chance 50%. Von trained on 8k PAWS train rows in this shape.
  choice  state = sentence S. Options: [a PAWS label-1 partner of S (true
          paraphrase, gold), a PAWS label-0 partner of S (lexical twin, wrong),
          an unrelated PAWS sentence (wrong)]. Chance 33%. Only sentences with
          both partner kinds qualify (~139 in test+validation).
          "Which option means the same as the text?"

If choice accuracy lands near 7%, the sealed collapse is reproduced locally
and becomes a training target. Also reports P(pick = lexical twin).

Usage:
    uv run python -m benchmarks.probe_paraphrase_adversary --ckpt checkpoints/von-1.2 \
        --n 200 --workers 4 --threads 2 --dump benchmarks/data/paraphrase_probe_v12.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from benchmarks.probe_overlap_bias import content_tokens  # noqa: E402

PAWS_FILES = os.environ.get("PAWS_FILES", "/tmp/scratch/paws/test.jsonl,/tmp/scratch/paws/validation.jsonl")

NOUL_Q = "Do `sentence1` and `sentence2` have the same meaning?"
NOUL_CRIT = {"true": "The two sentences state the same thing in different words.",
             "false": "The two sentences differ in meaning despite sharing wording."}
CHOICE_Q = "Which option states the same thing as the text, possibly in different words?"


def load_paws(paths: str) -> List[dict]:
    rows = []
    for path in paths.split(","):
        with open(path.strip(), "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                st = json.loads(r["state"]) if isinstance(r["state"], str) else r["state"]
                rows.append({"s1": st["sentence1"], "s2": st["sentence2"], "label": str(r["label"])})
    return rows


def jaccard(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


def build_jobs(rows: List[dict], n: int, rng: random.Random) -> List[dict]:
    jobs: List[dict] = []
    neg = [r for r in rows if r["label"] == "0"]
    pos = [r for r in rows if r["label"] == "1"]
    rng.shuffle(neg)
    rng.shuffle(pos)

    # noul: balanced positives/negatives
    for r in (neg[: n // 2] + pos[: n // 2]):
        jobs.append({"shape": "noul", "state": f"sentence1: {r['s1']}\nsentence2: {r['s2']}",
                     "instructions": NOUL_Q, "criteria": NOUL_CRIT,
                     "gold": "true" if r["label"] == "1" else "false"})

    # choice: sentences that have both a true-paraphrase partner and a lexical-twin partner
    partners: Dict[str, Dict[str, List[str]]] = {}
    for r in rows:
        for a, b in ((r["s1"], r["s2"]), (r["s2"], r["s1"])):
            partners.setdefault(a, {"1": [], "0": []})[r["label"]].append(b)
    anchors = sorted(s for s, p in partners.items() if p["1"] and p["0"])
    rng.shuffle(anchors)
    for s in anchors[:n]:
        para = rng.choice(partners[s]["1"])
        twin = rng.choice(partners[s]["0"])
        other = rng.choice(rows)["s1"]
        opts = [("paraphrase", para), ("twin", twin), ("unrelated", other)]
        rng.shuffle(opts)
        labels = [f"option_{i + 1}" for i in range(3)]
        crit = {lab: text for lab, (_, text) in zip(labels, opts)}
        roles = {lab: role for lab, (role, _) in zip(labels, opts)}
        jobs.append({"shape": "choice", "state": s, "instructions": CHOICE_Q,
                     "criteria": crit, "roles": roles,
                     "gold": next(l for l, ro in roles.items() if ro == "paraphrase"),
                     "overlap": {ro: round(jaccard(s, txt), 3) for ro, txt in opts}})
    return jobs


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
    out = {k: v for k, v in job.items() if k != "criteria"}
    try:
        res = _BACKEND.evaluate(state=job["state"], questions={"q": {
            "type": job["shape"], "instructions": job["instructions"], "criteria": job["criteria"]}})
        ans = res.answers["q"]
        if job["shape"] == "noul":
            p = float(getattr(ans, "noul"))
            pick, conf = ("true" if p >= 0.5 else "false"), abs(p - 0.5) * 2
        else:
            pick, conf = str(getattr(ans, "choice")), float(getattr(ans, "confidence"))
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": f"{type(exc).__name__}: {exc}"[:160]}
    out.update(pick=pick, hit=pick == job["gold"], confidence=conf)
    if job["shape"] == "choice":
        out["picked_role"] = job["roles"].get(pick)
    return out


def report(results: List[dict]) -> dict:
    good = [r for r in results if not r.get("error")]
    errs = [r for r in results if r.get("error")]
    if errs:
        print(f"{len(errs)} errors, first: {errs[0]['error']}")
    summary: Dict[str, object] = {}

    noul = [r for r in good if r["shape"] == "noul"]
    if noul:
        acc = sum(r["hit"] for r in noul) / len(noul)
        tp = [r for r in noul if r["gold"] == "true"]
        tn = [r for r in noul if r["gold"] == "false"]
        say_true = sum(r["pick"] == "true" for r in noul) / len(noul)
        summary["noul"] = {"n": len(noul), "acc": acc, "P_say_true": say_true,
                           "acc_on_paraphrase": sum(r["hit"] for r in tp) / max(1, len(tp)),
                           "acc_on_twin": sum(r["hit"] for r in tn) / max(1, len(tn))}
        print(f"\n== PAWS noul (chance 50%) ==  n={len(noul)}  acc {acc:.1%}  P(say true) {say_true:.1%}")
        print(f"   on true paraphrases: {summary['noul']['acc_on_paraphrase']:.1%}   "
              f"on lexical twins: {summary['noul']['acc_on_twin']:.1%}")

    choice = [r for r in good if r["shape"] == "choice"]
    if choice:
        acc = sum(r["hit"] for r in choice) / len(choice)
        roles = {ro: sum(r["picked_role"] == ro for r in choice) / len(choice)
                 for ro in ("paraphrase", "twin", "unrelated")}
        mean_ov = {ro: sum(r["overlap"][ro] for r in choice) / len(choice)
                   for ro in ("paraphrase", "twin", "unrelated")}
        summary["choice"] = {"n": len(choice), "acc": acc, "picked": roles, "mean_jaccard": mean_ov}
        print(f"\n== PAWS choice (chance 33%) ==  n={len(choice)}  acc {acc:.1%}")
        print("   picked:  " + "  ".join(f"{k} {v:.1%}" for k, v in roles.items()))
        print("   jaccard: " + "  ".join(f"{k} {v:.2f}" for k, v in mean_ov.items()))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/von-1.2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    jobs = build_jobs(load_paws(PAWS_FILES), args.n, rng)
    print(f"{len(jobs)} jobs on {os.path.basename(args.ckpt)}")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                      initargs=(args.ckpt, args.threads)) as pool:
        results = pool.map(_score, jobs, chunksize=4)
    print(f"scored in {time.time() - t0:.0f}s")
    summary = report(results)
    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "summary": summary, "results": results}, f, indent=1)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
