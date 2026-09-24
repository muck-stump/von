"""Generate numeric/temporal decision records where the gold is exact arithmetic.

Motivation: `benchmarks/probe_numeral_sensitivity.py` scrambled every digit in
JevBench's public `temporal_numeric` items and found Von's pick survived
unchanged 70% of the time (6/15 items never moved at all). The existing
numeric generator in `generate_synthetic_decisions.py` (~10k items, one
template: sum 2-3 quantities against a cap) did not fix this -- it is 3.4%
of the 290k corpus and covers none of the arithmetic shapes JevBench's public
items actually use (time-zone conversion, month-end/leap-year rollover,
proration, cross-sentence cumulative totals, elapsed-duration comparison).

Design, one rule per generator, all stdlib-only and exactly correct (no
sigmoid estimation -- these are facts, not judgment calls):

1. **Contrastive twins are the load-bearing mechanism.** Every scenario is
   emitted twice: the base draw, and a twin where exactly the decisive
   number(s) are perturbed so the gold flips. Wording, entities, and every
   other digit are identical between the pair. A model that ignores numerals
   cannot do better than chance across a twin pair; one that reads them can
   solve both. This is the only way the training signal forces attention
   onto digits rather than surrounding prose.
2. Every family produces true/false (Noul) records: `criteria={"true":...,
   "false":...}`-shaped, matching `evaluate_noul`'s framing and the public
   `temporal_numeric` family's own type mix (noul-heavy).
3. No lexical shortcut: `harden_corpus.gold_is_top` is checked in `audit()`
   and stays near the corpus-wide 0.32 target, because numbers, not words,
   decide these items by construction -- the true/false descriptions never
   contain the specific values being compared.

Five families, matching JevBench's `temporal_numeric` construction (a
deadline/timestamp cutoff, a month-end/leap-year rollover, a proration cap,
a cumulative total, a date/duration comparison):

    deadline_tz          cross-timezone deadline vs. event timestamp
    month_end_leap        N-months-later cutoff with month-end/leap clamping
    proration_percent      prorated-and-capped refund/credit vs. request
    cumulative_vs_limit    2-3 stated quantities summed against a cap
    date_order_duration    elapsed days between two dates vs. a policy window

Usage:
    uv run python -m training.generate_numeric_decisions --n 40000 \
        --out data_numeric/numeric.jsonl --audit
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# --- shared vocabulary -------------------------------------------------------

# (utc_offset_hours, label). Includes half/quarter-hour zones so offset math
# is not always whole-hour trivial.
OFFSETS: List[Tuple[float, str]] = [
    (-8.0, "Los Angeles time"), (-7.0, "Denver time"), (-5.0, "New York time"),
    (-4.0, "Santiago time"), (-3.0, "Buenos Aires time"), (0.0, "London time"),
    (1.0, "Rotterdam time"), (2.0, "Cairo time"), (3.0, "Nairobi time"),
    (5.5, "Mumbai time"), (7.0, "Bangkok time"), (8.0, "Singapore time"),
    (9.0, "Tokyo time"), (9.5, "Adelaide time"), (10.0, "Sydney time"),
    (12.75, "Chatham Islands time"),
]

NEUTRAL_FILLER = [
    "Account status is otherwise in good standing.",
    "No prior exceptions have been recorded on this account.",
    "The relevant policy has not been amended since issuance.",
    "All other supporting documents were received and are in order.",
    "The counterparty confirmed receipt of the notice by the usual channel.",
]


def _lc(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def fmt_offset(h: float) -> str:
    sign = "+" if h >= 0 else "-"
    ah = abs(h)
    hh = int(ah)
    mm = round((ah - hh) * 60)
    return f"{sign}{hh:02d}:{mm:02d}"


def rand_date(rng: random.Random, y_lo: int = 2023, y_hi: int = 2032) -> date:
    y = rng.randint(y_lo, y_hi)
    m = rng.randint(1, 12)
    d = rng.randint(1, calendar.monthrange(y, m)[1])
    return date(y, m, d)


def fmt_date(d: date) -> str:
    return d.strftime("%d %b %Y")


def bool_options(rng: random.Random, true_desc: str, false_desc: str) -> List[Dict[str, str]]:
    pair = [("true", true_desc), ("false", false_desc)]
    rng.shuffle(pair)
    return [{"id": i, "description": d} for i, d in pair]


def maybe_filler(rng: random.Random, text: str) -> str:
    if rng.random() < 0.5:
        filler = rng.choice(NEUTRAL_FILLER)
        return f"{text} {filler}" if rng.random() < 0.5 else f"{filler} {text}"
    return text


# --------------------------------------------------------------------- A ---
# deadline_tz: deadline and event given in different (possibly fractional)
# UTC offsets; gold requires converting both to a common reference.

def _draw_deadline_tz(rng: random.Random) -> dict:
    d_date, d_off = rand_date(rng), rng.choice(OFFSETS)
    d_h, d_m = rng.randint(0, 23), rng.randint(0, 59)
    deadline_local = datetime(d_date.year, d_date.month, d_date.day, d_h, d_m)
    deadline_utc = deadline_local - timedelta(hours=d_off[0])

    # margin can be small (near-boundary, adversarial) or large; never zero.
    margin_min = rng.choice([rng.randint(1, 240), rng.randint(1, 10080)])
    margin_min *= rng.choice([1, -1])
    event_utc = deadline_utc + timedelta(minutes=margin_min)
    e_off = rng.choice(OFFSETS)
    event_local = event_utc + timedelta(hours=e_off[0])

    return {"deadline_local": deadline_local, "d_off": d_off, "e_off": e_off,
            "margin_min": margin_min, "event_local": event_local}


def _twin_deadline_tz(params: dict, rng: random.Random) -> dict:
    out = dict(params)
    new_margin = -params["margin_min"]
    deadline_utc = params["deadline_local"] - timedelta(hours=params["d_off"][0])
    event_utc = deadline_utc + timedelta(minutes=new_margin)
    out["margin_min"] = new_margin
    out["event_local"] = event_utc + timedelta(hours=params["e_off"][0])
    return out


def make_deadline_tz(params: dict, rng: random.Random) -> dict:
    label = "false" if params["margin_min"] > 0 else "true"  # true = on time
    state = maybe_filler(rng,
        f"Deadline: {params['deadline_local'].strftime('%d %b %Y %H:%M')} "
        f"{params['d_off'][1]} (UTC{fmt_offset(params['d_off'][0])}). "
        f"Submission recorded: {params['event_local'].strftime('%d %b %Y %H:%M')} "
        f"{params['e_off'][1]} (UTC{fmt_offset(params['e_off'][0])}).")
    return {
        "state": state,
        "question": "Converting both timestamps to a common time reference, "
                    "was the submission made at or before the deadline?",
        "options": bool_options(rng,
            "Yes, once converted to a common reference the submission is at or before the deadline.",
            "No, once converted to a common reference the submission is after the deadline."),
        "label": label,
        "source": {"kind": "synth_numeric_deadline_tz"},
    }


# --------------------------------------------------------------------- B ---
# month_end_leap: N months after a start date, clamped to month-end / leap
# year, compared against a report date.

def _add_months_clamped(d: date, months: int) -> date:
    m0 = d.month - 1 + months
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)


def _draw_month_end_leap(rng: random.Random) -> dict:
    start = rand_date(rng, 2023, 2030)
    # Bias toward day-31 starts and Feb/30-day targets so clamping actually bites.
    if rng.random() < 0.6:
        start = start.replace(day=min(31, calendar.monthrange(start.year, start.month)[1]))
    months = rng.choice([1, 2, 3, 6, 9, 12, 15, 18, 24, 30, 36])
    cutoff = _add_months_clamped(start, months)
    margin_days = rng.choice([rng.randint(1, 5), rng.randint(1, 60)]) * rng.choice([1, -1])
    report = cutoff + timedelta(days=margin_days)
    return {"start": start, "months": months, "cutoff": cutoff,
            "margin_days": margin_days, "report": report}


def _twin_month_end_leap(params: dict, rng: random.Random) -> dict:
    out = dict(params)
    new_margin = -params["margin_days"]
    out["margin_days"] = new_margin
    out["report"] = params["cutoff"] + timedelta(days=new_margin)
    return out


def make_month_end_leap(params: dict, rng: random.Random) -> dict:
    label = "true" if params["margin_days"] <= 0 else "false"  # true = within window
    state = maybe_filler(rng,
        f"Reference date: {fmt_date(params['start'])}. Policy window: "
        f"{params['months']} months from the reference date, extending to the last "
        f"day of the ending month if that day does not exist in the target month. "
        f"Report filed: {fmt_date(params['report'])}.")
    return {
        "state": state,
        "question": "Was the report filed on or before the end of the policy window?",
        "options": bool_options(rng,
            "Yes, the report was filed on or before the end of the policy window.",
            "No, the report was filed after the end of the policy window."),
        "label": label,
        "source": {"kind": "synth_numeric_month_end_leap"},
    }


# --------------------------------------------------------------------- C ---
# proration_percent: prorated refund/credit, capped at a percentage, vs a
# requested amount.

PRORATION_DOMAINS = [
    ("subscription", "refund"), ("service contract", "credit"),
    ("membership", "refund"), ("equipment lease", "credit"),
]


def _draw_proration(rng: random.Random) -> dict:
    domain, kind = rng.choice(PRORATION_DOMAINS)
    total = round(rng.uniform(80, 4800), 2)
    period_days = rng.choice([30, 60, 90, 180, 365])
    elapsed = rng.randint(1, period_days - 1)
    remaining = period_days - elapsed
    cap_pct = rng.choice([50, 60, 70, 75, 80, 90, 100])
    raw_fraction = remaining / period_days
    capped_fraction = min(raw_fraction, cap_pct / 100.0)
    allowed = round(total * capped_fraction, 2)
    delta = round(rng.uniform(0.5, max(1.0, total * 0.05)), 2) * rng.choice([1, -1])
    requested = round(allowed + delta, 2)
    return {"domain": domain, "kind": kind, "total": total, "period_days": period_days,
            "elapsed": elapsed, "cap_pct": cap_pct, "allowed": allowed,
            "delta": delta, "requested": requested}


def _twin_proration(params: dict, rng: random.Random) -> dict:
    out = dict(params)
    out["delta"] = -params["delta"]
    out["requested"] = round(params["allowed"] + out["delta"], 2)
    return out


def make_proration(params: dict, rng: random.Random) -> dict:
    label = "true" if params["requested"] <= params["allowed"] + 0.005 else "false"
    state = maybe_filler(rng,
        f"{params['domain'].capitalize()} total paid: ${params['total']:.2f}. "
        f"Billing period: {params['period_days']} days; days elapsed before "
        f"cancellation: {params['elapsed']}. Policy caps any prorated {params['kind']} "
        f"at {params['cap_pct']}% of the total paid. Customer requests a "
        f"{params['kind']} of ${params['requested']:.2f}.")
    return {
        "state": state,
        "question": f"Is the requested {params['kind']} within what the policy allows?",
        "options": bool_options(rng,
            f"Yes, the requested {params['kind']} is within the policy's prorated-and-capped allowance.",
            f"No, the requested {params['kind']} exceeds the policy's prorated-and-capped allowance."),
        "label": label,
        "source": {"kind": "synth_numeric_proration"},
    }


# --------------------------------------------------------------------- D ---
# cumulative_vs_limit: 2-3 stated component quantities summed against a cap,
# plus a large irrelevant "decoy" number to defeat a biggest-number shortcut.

CUMULATIVE_DOMAINS = [
    {"item": "shipment", "unit": "kg", "cap": "weight limit",
     "components": ["the base cargo weighs {v} {unit}", "packaging adds {v} {unit}",
                     "restraint hardware adds {v} {unit}"],
     "decoy": "the shipment's declared insured value is ${decoy:,.0f}"},
    {"item": "duty roster", "unit": "hours", "cap": "duty-hour limit",
     "components": ["the scheduled shift runs {v} {unit}", "a mandatory briefing adds {v} {unit}",
                     "carried-over overtime adds {v} {unit}"],
     "decoy": "the crew's total annual leave balance is {decoy:.0f} {unit}"},
    {"item": "expense claim", "unit": "dollars", "cap": "reimbursement cap",
     "components": ["the airfare line item is {v} {unit}", "the lodging line item is {v} {unit}",
                     "the ground-transport line item is {v} {unit}"],
     "decoy": "the employee's annual salary band midpoint is {decoy:,.0f} {unit}"},
]


def _draw_cumulative(rng: random.Random) -> dict:
    domain = rng.choice(CUMULATIVE_DOMAINS)
    n = rng.choice([2, 2, 3])
    chosen = rng.sample(domain["components"], n)
    values = [round(rng.uniform(2, 60), 1) for _ in chosen]
    total = round(sum(values), 1)
    margin = round(total * rng.uniform(-0.3, 0.3), 1)
    threshold = max(1.0, round(total - margin, 1))
    decoy = rng.uniform(20000, 400000)
    return {"domain": domain, "chosen": chosen, "values": values, "total": total,
            "threshold": threshold, "decoy": decoy}


def _twin_cumulative(params: dict, rng: random.Random) -> dict:
    out = dict(params)
    values = list(params["values"])
    idx = rng.randrange(len(values))
    current_total = sum(values)
    margin = current_total - params["threshold"]
    # perturb one component enough to cross the threshold the other way
    shift = abs(margin) + round(rng.uniform(0.5, 5.0), 1)
    values[idx] = max(0.1, round(values[idx] + (shift if margin <= 0 else -shift), 1))
    out["values"] = values
    out["total"] = round(sum(values), 1)
    return out


def make_cumulative(params: dict, rng: random.Random) -> dict:
    domain = params["domain"]
    label = "false" if params["total"] > params["threshold"] else "true"  # true = within
    sentences = [c.format(v=v, unit=domain["unit"]) for c, v in zip(params["chosen"], params["values"])]
    sentences.append(domain["decoy"].format(decoy=params["decoy"], unit=domain["unit"]))
    sentences.append(f"the {domain['cap']} is {params['threshold']} {domain['unit']}")
    rng.shuffle(sentences)
    state = maybe_filler(rng, f"For this {domain['item']}: " + "; ".join(sentences) + ".")
    return {
        "state": state,
        "question": f"Does the total {domain['item']} figure exceed its {domain['cap']}?",
        "options": bool_options(rng,
            f"No, the total is at or under the {domain['cap']}.",
            f"Yes, the total exceeds the {domain['cap']}."),
        "label": label,
        "source": {"kind": "synth_numeric_cumulative"},
    }


# --------------------------------------------------------------------- E ---
# date_order_duration: elapsed days between two events vs a policy window.

DURATION_DOMAINS = [
    ("purchase", "warranty registration", "registered"),
    ("incident", "formal notice", "filed"),
    ("grant date", "vesting claim", "submitted"),
    ("enrollment", "first usage", "recorded"),
]


def _draw_duration(rng: random.Random) -> dict:
    a_name, b_name, verb = rng.choice(DURATION_DOMAINS)
    date_a = rand_date(rng)
    threshold = rng.choice([7, 14, 21, 30, 45, 60, 90, 120])
    margin = rng.choice([rng.randint(1, 5), rng.randint(1, 45)]) * rng.choice([1, -1])
    delta_days = max(1, threshold + margin)
    date_b = date_a + timedelta(days=delta_days)
    return {"a_name": a_name, "b_name": b_name, "verb": verb, "date_a": date_a,
            "threshold": threshold, "delta_days": delta_days, "date_b": date_b}


def _twin_duration(params: dict, rng: random.Random) -> dict:
    out = dict(params)
    margin = params["delta_days"] - params["threshold"]
    new_delta = max(1, params["threshold"] - margin)
    out["delta_days"] = new_delta
    out["date_b"] = params["date_a"] + timedelta(days=new_delta)
    return out


def make_duration(params: dict, rng: random.Random) -> dict:
    label = "true" if params["delta_days"] <= params["threshold"] else "false"
    state = maybe_filler(rng,
        f"{params['a_name'].capitalize()} date: {fmt_date(params['date_a'])}. "
        f"{params['b_name'].capitalize()} {params['verb']}: {fmt_date(params['date_b'])}. "
        f"Policy requires the {params['b_name']} within {params['threshold']} days "
        f"of the {params['a_name']}.")
    return {
        "state": state,
        "question": f"Was the {params['b_name']} completed within the policy's stated window?",
        "options": bool_options(rng,
            "Yes, it was completed within the policy's stated window.",
            "No, it was completed after the policy's stated window."),
        "label": label,
        "source": {"kind": "synth_numeric_duration"},
    }


# ------------------------------------------------------------ choice sets --
# Value-ranking variants. Why these exist: the Noul pairs above teach
# "is this verdict plausible" at a 0.5 threshold. JevBench's temporal_numeric
# items instead put the *naive computation's value* in the option set as a
# decoy (`provenance.surface_answer`), and Von picks that decoy 59-75% of the
# time it is wrong -- 1.5-1.9x what uniform-over-wrong-options predicts, on
# both von-1.2 and the Noul-only continue-train. The trainer already applies
# softmax-CE across an item's options; what was missing is data where gold
# and its own near-miss sit in one option set with identical wording. Every
# choice item below carries the surface value as a mandatory sibling.

CHOICE_IDS = ["a", "b", "c", "d", "e", "f"]


def value_options(rng: random.Random, gold: str, decoys: List[str]) -> tuple:
    """Shuffle gold + unique decoys into id'd options; return (options, gold_id)."""
    vals = [gold]
    for d in decoys:
        if d != gold and d not in vals:
            vals.append(d)
    rng.shuffle(vals)
    opts = [{"id": CHOICE_IDS[i], "description": v} for i, v in enumerate(vals)]
    gold_id = opts[vals.index(gold)]["id"]
    return opts, gold_id


def _fmt_dt(d: datetime) -> str:
    return d.strftime("%d %b %Y %H:%M")


def make_deadline_tz_choice(params: dict, rng: random.Random) -> dict:
    dl, d_off, e_off = params["deadline_local"], params["d_off"], params["e_off"]
    gold_dt = dl - timedelta(hours=d_off[0]) + timedelta(hours=e_off[0])
    surface = dl                                                   # no conversion
    wrong_sign = dl + timedelta(hours=d_off[0]) - timedelta(hours=e_off[0])
    decoys = [_fmt_dt(surface), _fmt_dt(wrong_sign),
              _fmt_dt(gold_dt + timedelta(hours=rng.choice([-1, 1]))),
              _fmt_dt(gold_dt + timedelta(minutes=rng.choice([-30, 30])))]
    opts, gold_id = value_options(rng, _fmt_dt(gold_dt), decoys)
    state = maybe_filler(rng,
        f"Deadline: {_fmt_dt(dl)} {d_off[1]} (UTC{fmt_offset(d_off[0])}). "
        f"The submitter is located in {e_off[1]} (UTC{fmt_offset(e_off[0])}).")
    return {"state": state,
            "question": f"Expressed in the submitter's local time ({e_off[1]}), when is the deadline?",
            "options": opts, "label": gold_id,
            "source": {"kind": "synth_numeric_deadline_tz_choice", "surface": _fmt_dt(surface)}}


def make_month_end_leap_choice(params: dict, rng: random.Random) -> dict:
    start, months, cutoff = params["start"], params["months"], params["cutoff"]
    # naive: same day-of-month, overflowing past month-end (what a shallow read does)
    m0 = start.month - 1 + months
    y, m = start.year + m0 // 12, m0 % 12 + 1
    overflow = start.day - calendar.monthrange(y, m)[1]
    surface = cutoff + timedelta(days=overflow) if overflow > 0 else cutoff - timedelta(days=1)
    feb_swap = cutoff.replace(day=28) if cutoff.month == 2 and cutoff.day == 29 else cutoff + timedelta(days=1)
    decoys = [fmt_date(surface), fmt_date(feb_swap), fmt_date(cutoff - timedelta(days=1)),
              fmt_date(_add_months_clamped(start, months + rng.choice([-1, 1])))]
    opts, gold_id = value_options(rng, fmt_date(cutoff), decoys)
    state = maybe_filler(rng,
        f"Reference date: {fmt_date(start)}. Policy window: {months} months from the "
        f"reference date, extending to the last day of the ending month if that day "
        f"does not exist in the target month.")
    return {"state": state, "question": "On what date does the policy window end?",
            "options": opts, "label": gold_id,
            "source": {"kind": "synth_numeric_month_end_leap_choice", "surface": fmt_date(surface)}}


def make_proration_choice(params: dict, rng: random.Random) -> dict:
    total, period, elapsed, cap = params["total"], params["period_days"], params["elapsed"], params["cap_pct"]
    allowed = params["allowed"]
    uncapped = round(total * (period - elapsed) / period, 2)
    cap_only = round(total * cap / 100.0, 2)
    surface = uncapped if abs(uncapped - allowed) > 0.005 else cap_only
    inverse = round(total * elapsed / period, 2)
    decoys = [f"${surface:.2f}", f"${cap_only:.2f}", f"${inverse:.2f}", f"${total:.2f}"]
    opts, gold_id = value_options(rng, f"${allowed:.2f}", decoys)
    state = maybe_filler(rng,
        f"{params['domain'].capitalize()} total paid: ${total:.2f}. Billing period: {period} days; "
        f"days elapsed before cancellation: {elapsed}. Policy caps any prorated "
        f"{params['kind']} at {cap}% of the total paid.")
    return {"state": state,
            "question": f"What is the maximum {params['kind']} the policy allows?",
            "options": opts, "label": gold_id,
            "source": {"kind": "synth_numeric_proration_choice", "surface": f"${surface:.2f}"}}


def make_cumulative_choice(params: dict, rng: random.Random) -> dict:
    domain, values, total = params["domain"], params["values"], params["total"]
    unit = domain["unit"]
    partial = round(sum(values[:-1]), 1) if len(values) > 2 else round(max(values), 1)
    surface = round(max(values), 1)                                # biggest single line
    decoys = [f"{surface} {unit}", f"{partial} {unit}", f"{params['threshold']} {unit}",
              f"{round(total + rng.choice([-1, 1]) * rng.uniform(0.5, 4.0), 1)} {unit}"]
    opts, gold_id = value_options(rng, f"{total} {unit}", decoys)
    sentences = [c.format(v=v, unit=unit) for c, v in zip(params["chosen"], values)]
    sentences.append(domain["decoy"].format(decoy=params["decoy"], unit=unit))
    sentences.append(f"the {domain['cap']} is {params['threshold']} {unit}")
    rng.shuffle(sentences)
    state = maybe_filler(rng, f"For this {domain['item']}: " + "; ".join(sentences) + ".")
    return {"state": state, "question": f"What is the total {domain['item']} figure?",
            "options": opts, "label": gold_id,
            "source": {"kind": "synth_numeric_cumulative_choice", "surface": f"{surface} {unit}"}}


def make_duration_choice(params: dict, rng: random.Random) -> dict:
    a, b, delta = params["date_a"], params["date_b"], params["delta_days"]
    # naive: day-of-month difference, or whole-months x 30
    naive_dom = abs(b.day - a.day)
    naive_m30 = ((b.year - a.year) * 12 + (b.month - a.month)) * 30
    surface = naive_dom if naive_dom != delta else naive_m30
    decoys = [f"{surface} days", f"{naive_m30} days", f"{delta + rng.choice([-1, 1])} days",
              f"{params['threshold']} days"]
    opts, gold_id = value_options(rng, f"{delta} days", decoys)
    state = maybe_filler(rng,
        f"{params['a_name'].capitalize()} date: {fmt_date(a)}. "
        f"{params['b_name'].capitalize()} {params['verb']}: {fmt_date(b)}. "
        f"Policy requires the {params['b_name']} within {params['threshold']} days of the {params['a_name']}.")
    return {"state": state,
            "question": f"How many days elapsed between the {params['a_name']} and the {params['b_name']}?",
            "options": opts, "label": gold_id,
            "source": {"kind": "synth_numeric_duration_choice", "surface": f"{surface} days"}}


CHOICE_MAKERS = {
    "deadline_tz": make_deadline_tz_choice,
    "month_end_leap": make_month_end_leap_choice,
    "proration_percent": make_proration_choice,
    "cumulative_vs_limit": make_cumulative_choice,
    "date_order_duration": make_duration_choice,
}


# --------------------------------------------------------------------- gen -

FAMILIES = [
    ("deadline_tz", _draw_deadline_tz, _twin_deadline_tz, make_deadline_tz),
    ("month_end_leap", _draw_month_end_leap, _twin_month_end_leap, make_month_end_leap),
    ("proration_percent", _draw_proration, _twin_proration, make_proration),
    ("cumulative_vs_limit", _draw_cumulative, _twin_cumulative, make_cumulative),
    ("date_order_duration", _draw_duration, _twin_duration, make_duration),
]


def generate(n: int, seed: int, dedupe: bool = True, choice_ratio: float = 0.0) -> List[dict]:
    """Generate n records split evenly across 5 families.

    `choice_ratio` is the fraction of each family's budget emitted as value-
    ranking choice sets (gold + surface decoy + near-miss siblings); the rest
    are the original true/false twin pairs. Choice sets are also emitted as
    base+twin, so the same wording appears with the gold at a different value.
    """
    rng = random.Random(seed)
    out: List[dict] = []
    seen = set()
    per_family = max(2, n // len(FAMILIES) // 2 * 2)  # keep pairs together, even count
    n_choice = int(per_family * choice_ratio) // 2 * 2
    n_noul = per_family - n_choice

    def _emit_pairs(family: str, budget: int, render, require_flip: bool) -> int:
        _, draw, twin, _ = next(f for f in FAMILIES if f[0] == family)
        made = stalled = 0
        while made < budget and stalled < 20000:
            params = draw(rng)
            base = render(params, rng)
            tw = render(twin(params, rng), rng)
            if require_flip and tw["label"] == base["label"]:
                stalled += 1
                continue
            key_b, key_t = (base["state"], base["question"]), (tw["state"], tw["question"])
            if dedupe and (key_b in seen or key_t in seen or key_b == key_t):
                stalled += 1
                continue
            seen.add(key_b)
            seen.add(key_t)
            base["family"] = tw["family"] = family
            out.extend((base, tw))
            made += 2
            stalled = 0
        return made

    for family, _, _, render in FAMILIES:
        made = _emit_pairs(family, n_noul, render, require_flip=True)
        if made < n_noul:
            print(f"NOTE: {family} noul generated {made:,} of {n_noul:,} requested.")
        if n_choice:
            made = _emit_pairs(family, n_choice, CHOICE_MAKERS[family], require_flip=False)
            if made < n_choice:
                print(f"NOTE: {family} choice generated {made:,} of {n_choice:,} requested.")
    rng.shuffle(out)
    return out


def audit(records: List[dict]) -> None:
    from training.harden_corpus import gold_is_top, option_id, overlap_scores

    print(f"Total: {len(records):,}")
    fam_counts = Counter(r["family"] for r in records)
    for f, c in fam_counts.most_common():
        print(f"  {f:<22} {c:,}")

    labels = Counter(r["label"] for r in records)
    print(f"\nLabel balance: {dict(labels)} "
          f"({labels['true'] / len(records):.1%} true)")

    measurable = [r for r in records
                  if max(overlap_scores(str(r["state"]), r["options"]), default=0) > 0]
    top = sum(1 for r in measurable if gold_is_top(r))
    print(f"\nGold-is-highest-overlap: {top}/{len(measurable)} = "
          f"{top / max(1, len(measurable)):.1%} of {len(measurable)} measurable "
          f"(of {len(records)} total; target ~32%, numbers should carry the signal, not words)")

    lens = sorted(len(r["state"]) for r in records)
    print(f"\nState length (chars): min {lens[0]} p50 {lens[len(lens)//2]} max {lens[-1]}")


def write_jsonl(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="data_numeric/numeric.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--choice-ratio", type=float, default=0.0,
                    help="fraction of each family emitted as value-ranking choice sets "
                         "(gold + surface decoy + near-miss siblings) instead of true/false pairs")
    args = ap.parse_args()

    records = generate(args.n, args.seed, choice_ratio=args.choice_ratio)
    write_jsonl(records, args.out)
    print(f"Wrote {len(records):,} records to {args.out}")
    if args.audit:
        audit(records)


if __name__ == "__main__":
    main()
