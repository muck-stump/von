"""MDE-aware ship gates for Von checkpoint comparisons.

Why this exists: six retrain-class runs were judged null on per-family public
gates of n=12 and n=15. A 20pp effect on n=15 has a Wilson 95% CI wider than
the effect itself, so those gates could never resolve what they were asked to
resolve. This module makes the resolving power explicit and refuses to let an
unresolvable gate drive a ship/no-ship decision.

Two tools:

  mcnemar_gate(baseline_hits, candidate_hits)
      Paired comparison on the same items. Reports discordant pairs, exact
      two-sided McNemar p, and the minimum detectable effect (MDE, in pp of
      accuracy) at 80% power for that n. If |delta| < MDE the gate is
      UNRESOLVABLE regardless of the sign of the delta.

  numeric_slice(rows)
      Pools every public item across every family whose state carries >= 2
      numerals (digits or spelled-out month names next to digits). This is the
      only public statistic large enough (~80-100 items) to move a decision.
      Hand-curated family labels stay reported but stop being gates.

Usage:
  uv run python -m benchmarks.stat_gate \\
      --baseline benchmarks/data/baseline_v12_hard.json benchmarks/data/baseline_v12_standard.json \\
      --candidate /tmp/v13_hard.json /tmp/v13_standard.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.eval_hard_fast import TIER_FILES, load_rows  # noqa: E402

_NUMERAL_RE = re.compile(r"\d+(?:[.,:]\d+)*")
_MONTH_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d", re.I
)


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test at p=0.5 (the McNemar exact statistic)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mde_pp(n: int, p_base: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable accuracy change (in pp) for a one-sample proportion
    at n items, baseline accuracy p_base, two-sided alpha, given power.
    Normal approximation; good enough as a gate sanity bound."""
    if n <= 0:
        return 100.0
    z_a = 1.959964  # two-sided 0.05
    z_b = {0.80: 0.841621, 0.90: 1.281552}.get(power, 0.841621)
    se = math.sqrt(p_base * (1 - p_base) / n)
    return 100.0 * (z_a + z_b) * se


def mcnemar_gate(
    baseline_hits: Sequence[bool],
    candidate_hits: Sequence[bool],
    *,
    threshold_pp: float | None = None,
    label: str = "",
) -> Dict[str, object]:
    if len(baseline_hits) != len(candidate_hits):
        raise ValueError("paired comparison needs identical item lists")
    n = len(baseline_hits)
    b = sum(1 for x, y in zip(baseline_hits, candidate_hits) if x and not y)   # base right, cand wrong
    c = sum(1 for x, y in zip(baseline_hits, candidate_hits) if not x and y)   # base wrong, cand right
    acc_b = sum(baseline_hits) / n if n else 0.0
    acc_c = sum(candidate_hits) / n if n else 0.0
    delta_pp = 100.0 * (acc_c - acc_b)
    p = _binom_two_sided_p(c, b + c)
    mde = mde_pp(n, acc_b if 0 < acc_b < 1 else 0.5)
    resolvable = abs(delta_pp) >= mde
    if threshold_pp is not None and threshold_pp < mde:
        verdict = "UNRESOLVABLE"
    elif not resolvable:
        verdict = "UNRESOLVABLE"
    elif p < 0.05:
        verdict = "PASS" if delta_pp > 0 else "FAIL"
    else:
        verdict = "UNRESOLVABLE"
    return {
        "label": label,
        "n": n,
        "acc_baseline": acc_b,
        "acc_candidate": acc_c,
        "delta_pp": delta_pp,
        "discordant_base_only": b,
        "discordant_cand_only": c,
        "mcnemar_p": p,
        "mde_pp_80pct": mde,
        "verdict": verdict,
    }


def is_numeric_item(row: dict) -> bool:
    state = row.get("state")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    return len(_NUMERAL_RE.findall(state)) >= 2 or bool(_MONTH_RE.search(state))


def numeric_slice(rows: Iterable[dict]) -> List[str]:
    return [r["id"] for r in rows if is_numeric_item(r)]


def _load_results(paths: Sequence[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        for r in d["results"]:
            out[r["id"]] = r
    return out


def _fmt(g: Dict[str, object]) -> str:
    return (f"{g['label']:<28} n={g['n']:<4} "
            f"{100*g['acc_baseline']:5.1f}% -> {100*g['acc_candidate']:5.1f}%  "
            f"d={g['delta_pp']:+5.1f}pp  MDE={g['mde_pp_80pct']:4.1f}pp  "
            f"disc={g['discordant_base_only']}/{g['discordant_cand_only']}  "
            f"p={g['mcnemar_p']:.3f}  {g['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", nargs="+", required=True, help="eval_hard_fast dump(s) for the baseline")
    ap.add_argument("--candidate", nargs="+", required=True, help="eval_hard_fast dump(s) for the candidate")
    ap.add_argument("--tiers", nargs="+", default=["hard", "standard"], choices=list(TIER_FILES))
    args = ap.parse_args()

    base = _load_results(args.baseline)
    cand = _load_results(args.candidate)
    rows: List[dict] = []
    for t in args.tiers:
        rows.extend(load_rows(TIER_FILES[t]))
    rows = [r for r in rows if r["id"] in base and r["id"] in cand]
    ids = [r["id"] for r in rows]
    fam_of = {r["id"]: base[r["id"]].get("family", "?") for r in rows}

    def gate(sel: List[str], label: str, threshold_pp: float | None = None):
        return mcnemar_gate([base[i]["hit"] for i in sel], [cand[i]["hit"] for i in sel],
                            threshold_pp=threshold_pp, label=label)

    numeric_ids = set(numeric_slice(rows))
    print(f"\n{len(ids)} paired items across {args.tiers}; numeric-sensitive slice = {len(numeric_ids)} items\n")
    print("== decision gates (pooled, MDE-aware) ==")
    print(_fmt(gate(ids, "ALL public (hard+standard)")))
    print(_fmt(gate([i for i in ids if i in numeric_ids], "numeric slice (>=2 numerals)")))
    print(_fmt(gate([i for i in ids if i not in numeric_ids], "non-numeric complement")))
    print("\n== per-family (REPORT ONLY -- not gates; note MDE vs n) ==")
    fams = sorted({fam_of[i] for i in ids})
    for f in fams:
        sel = [i for i in ids if fam_of[i] == f]
        print(_fmt(gate(sel, f)))


if __name__ == "__main__":
    main()
