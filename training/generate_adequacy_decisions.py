"""Generate response-adequacy judgment records: does a response satisfy a request?

JevBench's public `adequacy` family (standard tier, 12 items) is a Noul task:
"Does the response fully satisfy the request, using the supplied reference
when present?" von-1.2 scores 2/12 (17%, at/below chance) under the correct
argmax readout (see PROGRESS.md 2026-09-23) -- a genuine deficit, not a
readout artefact like the ordinal-tier bug this same session found and fixed.

The public items span six constraint shapes: exact arithmetic, exhaustive
selection from a reference set, structural output format, exact length,
negation/polarity resolution, and comparison/ranking. Each generator below
covers one shape with a deterministic, exactly-computable verdict (this is
constraint-checking, not judgment calls) and emits a **contrastive pair**:
identical request/reference wording, one response that satisfies every
constraint and one that violates exactly one -- so the training signal
isolates the constraint, not the topic.

Family E (`negation_polarity`) is deliberately the same underlying skill as
`training/generate_numeric_decisions.py`'s sibling fix in
`probe_noul_negation.py` (Von's 22pp yes-bias, 24% flip rate under rubric
negation): judging whether a response's polarity correctly resolves a
reference fact stated with different surface polarity than the question.

Usage:
    uv run python -m training.generate_adequacy_decisions --n 10000 \
        --out data_adequacy/adequacy.jsonl --audit
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import List, Tuple

from training.generate_numeric_decisions import maybe_filler, write_jsonl

TRUE_DESC = "Correct, complete, and follows every explicit constraint in the request."
FALSE_DESC = "Wrong, incomplete, or violates at least one explicit constraint in the request."
INSTRUCTIONS = ("Do the response's contents fully and correctly satisfy the request, "
                "respecting the reference where one is supplied?")



def _record(state: str, label: str, family: str) -> dict:
    return {
        "state": state,
        "question": INSTRUCTIONS,
        "options": [{"id": "true", "description": TRUE_DESC}, {"id": "false", "description": FALSE_DESC}],
        "label": label,
        "family": family,
        "source": {"kind": f"synth_adequacy_{family}"},
    }


# --------------------------------------------------------------------- A ---
# exact_arithmetic: does the response give the exact requested computation?

def make_arithmetic_pair(rng: random.Random) -> Tuple[dict, dict]:
    a, b = rng.randint(2, 500), rng.randint(2, 500)
    op = rng.choice(["+", "-", "*"])
    correct = {"+": a + b, "-": a - b, "*": a * b}[op]
    op_word = {"+": "sum", "-": "difference", "*": "product"}[op]
    phrasing = rng.choice([
        f"Return only the {op_word} of {a} and {b}.",
        f"Give just {a}{op}{b}.",
        f"Compute {a} {op} {b} and reply with the number alone.",
    ])
    # A plausible near-miss, never the correct value: off by a small delta,
    # a transposition, or the wrong operand order for non-commutative ops.
    wrong = correct + rng.choice([d for d in range(-9, 10) if d != 0])
    if op == "-" and rng.random() < 0.3:
        wrong = b - a  # classic operand-order mistake
    base = _record(f"Request: {phrasing} Response: {correct}", "true", "exact_arithmetic")
    twin = _record(f"Request: {phrasing} Response: {wrong}", "false", "exact_arithmetic")
    return base, twin


# --------------------------------------------------------------------- B ---
# exhaustive_selection: does the response name every item matching a criterion?

def make_selection_pair(rng: random.Random) -> Tuple[dict, dict]:
    pool = rng.sample(["red", "teal", "amber", "violet", "slate", "coral", "olive", "navy"], 4)
    flags = rng.sample(range(4), k=rng.choice([2, 3]))
    matching = sorted(pool[i] for i in flags)
    ref = ", ".join(f"{c}={'match' if i in flags else 'no match'}" for i, c in enumerate(pool))
    req = f"Name every color that is a match, per the reference."
    correct_resp = " and ".join(matching)
    dropped = matching[:-1] if len(matching) > 1 else []
    wrong_resp = " and ".join(dropped) if dropped else pool[flags[0]] + " only"
    base = _record(f"Request: {req} Reference: {ref}. Response: {correct_resp}", "true", "exhaustive_selection")
    twin = _record(f"Request: {req} Reference: {ref}. Response: {wrong_resp}", "false", "exhaustive_selection")
    return base, twin


# --------------------------------------------------------------------- C ---
# exact_format: does the response match the requested structural shape?

def make_format_pair(rng: random.Random) -> Tuple[dict, dict]:
    n = rng.randint(2, 5)
    values = sorted(rng.sample(range(1, 99), n))
    shape = rng.choice(["array", "csv"])
    if shape == "array":
        req = f"Return a JSON array containing exactly the integers {values}, in that order."
        correct_resp = json.dumps(values)
        # violates structure while keeping the same content: wraps in an object.
        wrong_resp = json.dumps({"values": values})
    else:
        req = f"Return a comma-separated list of exactly the integers {values}, in that order, no brackets."
        correct_resp = ", ".join(str(v) for v in values)
        wrong_resp = json.dumps(values)  # bracketed instead of a bare CSV line
    base = _record(f"Request: {req} Response: {correct_resp}", "true", "exact_format")
    twin = _record(f"Request: {req} Response: {wrong_resp}", "false", "exact_format")
    return base, twin


# --------------------------------------------------------------------- D ---
# exact_length: does the response satisfy an explicit word-count constraint?

FILLER_WORDS = ["done", "ready", "confirmed", "sent", "received", "logged", "closed", "noted",
                "checked", "cleared", "queued", "posted"]


def make_length_pair(rng: random.Random) -> Tuple[dict, dict]:
    n = rng.randint(2, 5)
    req = f"Reply with exactly {n} words confirming the task is complete, nothing else."
    correct_resp = " ".join(rng.sample(FILLER_WORDS, n))
    off_by = rng.choice([-2, -1, 1, 2])
    wrong_n = max(1, n + off_by)
    wrong_resp = " ".join(rng.sample(FILLER_WORDS, min(wrong_n, len(FILLER_WORDS))))
    base = _record(f"Request: {req} Response: {correct_resp}", "true", "exact_length")
    twin = _record(f"Request: {req} Response: {wrong_resp}", "false", "exact_length")
    return base, twin


# --------------------------------------------------------------------- E ---
# negation_polarity: does the response resolve a fact stated with opposite
# surface polarity to the question? Same skill as probe_noul_negation.

POLARITY_DOMAINS = [
    ("open", "closed", "Is {subj} open on {day}?"),
    ("available", "unavailable", "Is {subj} available for {day}?"),
    ("eligible", "ineligible", "Is {subj} eligible on {day}?"),
    ("included", "excluded", "Is {subj} included for {day}?"),
]
SUBJECTS = ["the library", "the clinic", "the warehouse", "the account", "the route", "the plan"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Sunday", "the holiday period", "the trial period"]


def make_polarity_pair(rng: random.Random) -> Tuple[dict, dict]:
    pos_word, neg_word, tmpl = rng.choice(POLARITY_DOMAINS)
    subj, day = rng.choice(SUBJECTS), rng.choice(DAYS)
    req = tmpl.format(subj=subj, day=day)
    is_neg_fact = rng.random() < 0.5
    ref = f"Reference: {subj.capitalize()} is {neg_word if is_neg_fact else pos_word} on/for {day}."
    correct_answer = "No" if is_neg_fact else "Yes"
    wrong_answer = "Yes" if is_neg_fact else "No"
    correct_resp = f"{correct_answer}, it is {neg_word if is_neg_fact else pos_word}."
    wrong_resp = f"{wrong_answer}, it is {neg_word if is_neg_fact else pos_word}."  # surface-plausible flip
    base = _record(f"Request: {req} {ref} Response: {correct_resp}", "true", "negation_polarity")
    twin = _record(f"Request: {req} {ref} Response: {wrong_resp}", "false", "negation_polarity")
    return base, twin


# --------------------------------------------------------------------- F ---
# comparison_ranking: does the response name the correct extreme from a
# small reference set (earliest / cheapest / most expensive / latest)?

def make_comparison_pair(rng: random.Random) -> Tuple[dict, dict]:
    n = rng.randint(2, 4)
    labels = rng.sample(["A", "B", "C", "D"], n)
    kind = rng.choice(["time", "price"])
    if kind == "time":
        values = rng.sample(range(0, 1440), n)  # minutes past midnight, distinct
        fmt = lambda v: f"{v // 60:02d}:{v % 60:02d}"
        criterion, pick = rng.choice([("earliest", min), ("latest", max)])
        req = f"Which option is {criterion}?"
        ref = ", ".join(f"{l} at {fmt(v)}" for l, v in zip(labels, values))
    else:
        values = rng.sample(range(5, 999), n)
        fmt = lambda v: f"${v}"
        criterion, pick = rng.choice([("cheapest", min), ("most expensive", max)])
        req = f"Which option is {criterion}?"
        ref = ", ".join(f"{l} costs {fmt(v)}" for l, v in zip(labels, values))
    correct_label = labels[values.index(pick(values))]
    # a close, plausible wrong pick: the runner-up by the same criterion.
    ranked = sorted(range(n), key=lambda i: values[i], reverse=(pick is max))
    wrong_label = labels[ranked[1]] if n > 1 else labels[0]
    base = _record(f"Request: {req} Reference: {ref}. Response: {correct_label}", "true", "comparison_ranking")
    twin = _record(f"Request: {req} Reference: {ref}. Response: {wrong_label}", "false", "comparison_ranking")
    return base, twin


FAMILIES = [
    ("exact_arithmetic", make_arithmetic_pair),
    ("exhaustive_selection", make_selection_pair),
    ("exact_format", make_format_pair),
    ("exact_length", make_length_pair),
    ("negation_polarity", make_polarity_pair),
    ("comparison_ranking", make_comparison_pair),
]


def generate(n: int, seed: int, dedupe: bool = True) -> List[dict]:
    rng = random.Random(seed)
    out: List[dict] = []
    seen = set()
    per_family = max(2, n // len(FAMILIES) // 2 * 2)
    for family, make_pair in FAMILIES:
        made = 0
        stalled = 0
        while made < per_family and stalled < 20000:
            base, twin = make_pair(rng)
            if rng.random() < 0.5:
                base["state"] = maybe_filler(rng, base["state"])
                twin["state"] = maybe_filler(rng, twin["state"])
            if base["label"] == twin["label"]:
                stalled += 1
                continue
            key_b, key_t = base["state"], twin["state"]
            if dedupe and (key_b in seen or key_t in seen):
                stalled += 1
                continue
            seen.add(key_b)
            seen.add(key_t)
            out.append(base)
            out.append(twin)
            made += 2
            stalled = 0
        if made < per_family:
            print(f"NOTE: {family} generated {made:,} of {per_family:,} requested.")
    rng.shuffle(out)
    return out


def audit(records: List[dict]) -> None:
    from training.harden_corpus import gold_is_top, overlap_scores

    print(f"Total: {len(records):,}")
    for f, c in Counter(r["family"] for r in records).most_common():
        print(f"  {f:<22} {c:,}")
    labels = Counter(r["label"] for r in records)
    print(f"\nLabel balance: {dict(labels)} ({labels['true'] / len(records):.1%} true)")
    measurable = [r for r in records
                  if max(overlap_scores(str(r["state"]), r["options"]), default=0) > 0]
    top = sum(1 for r in measurable if gold_is_top(r))
    print(f"Gold-is-highest-overlap (measurable only): {top}/{len(measurable)} = "
          f"{top / max(1, len(measurable)):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="data_adequacy/adequacy.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    records = generate(args.n, args.seed)
    write_jsonl(records, args.out)
    print(f"Wrote {len(records):,} records to {args.out}")
    if args.audit:
        audit(records)


if __name__ == "__main__":
    main()
