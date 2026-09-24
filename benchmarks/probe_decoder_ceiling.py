"""Zero-shot decoder ceiling probe for the public numeric slice.

Question this answers: is the ModernBERT trunk the bottleneck on
temporal_numeric, or are these items simply not solvable by *any* single-pass
option scorer (no chain-of-thought)? A stock instruct decoder scored with the
same protocol Von uses -- one score per option, argmax, no generation -- is
the discriminator:

  decoder well above chance where Von is at/below chance
      -> trunk number representations are the bottleneck; trunk swap justified.
  decoder also near chance
      -> the items need multi-step reasoning no one-pass scorer does; the
         target is "stop confidently picking the surface answer", not "learn
         to compute", and the encoder stays.

Scoring: for each option, the mean token log-prob of the option's label text
conditioned on a prompt holding the state, the instructions, and the full
option list (so the decoder sees the same information Von's packed sequence
does). Argmax over options. No training, no CoT, CPU only.

Usage:
  uv run --with accelerate python -m benchmarks.probe_decoder_ceiling \\
      --model Qwen/Qwen2.5-1.5B-Instruct --dump benchmarks/data/decoder_ceiling_q15.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.eval_hard_fast import TIER_FILES, build_request, load_rows  # noqa: E402
from benchmarks.stat_gate import is_numeric_item  # noqa: E402


def _prompt(state: str, instructions: str, criteria: Dict[str, str]) -> str:
    opts = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    return (
        "Read the document and answer the question by choosing exactly one of the "
        "listed answers. Reply with the answer key only.\n\n"
        f"DOCUMENT:\n{state}\n\nQUESTION: {instructions}\n\nANSWERS:\n{opts}\n\nANSWER KEY:"
    )


@torch.no_grad()
def _option_logprobs(model, tok, prompt: str, keys: List[str], max_ctx: int) -> List[float]:
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    # Left-truncate the *state* side if the prompt overflows: keep the tail
    # (question + answers) intact, which is where the decision lives.
    budget = max_ctx - 16
    if len(p_ids) > budget:
        p_ids = p_ids[-budget:]
    # Encode the shared prompt once; score each option's tokens off the cached
    # prefix. On a 4-core laptop CPU this is the difference between minutes and
    # hours per hundred items.
    prefix = model(torch.tensor([p_ids]), use_cache=True)
    last_lp = torch.log_softmax(prefix.logits[0, -1].float(), dim=-1)
    out: List[float] = []
    for k in keys:
        k_ids = tok(" " + k, add_special_tokens=False)["input_ids"]
        lps = [last_lp[k_ids[0]].item()]
        if len(k_ids) > 1:
            cache = copy.deepcopy(prefix.past_key_values)
            o = model(torch.tensor([k_ids[:-1]]), past_key_values=cache, use_cache=True)
            lp = torch.log_softmax(o.logits[0].float(), dim=-1)
            for j in range(1, len(k_ids)):
                lps.append(lp[j - 1, k_ids[j]].item())
        out.append(sum(lps) / len(lps))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tiers", nargs="+", default=["hard", "standard"], choices=list(TIER_FILES))
    ap.add_argument("--only-numeric", action="store_true", default=True)
    ap.add_argument("--all", dest="only_numeric", action="store_false", help="score every item, not just the numeric slice")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    rows: List[dict] = []
    for t in args.tiers:
        rows.extend(load_rows(TIER_FILES[t]))
    if args.only_numeric:
        rows = [r for r in rows if is_numeric_item(r)]
    print(f"{len(rows)} items on {args.model} ({'numeric slice' if args.only_numeric else 'all'})", flush=True)

    results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        state, instr, crit = build_request(row, "plain")
        keys = list(crit)
        lps = _option_logprobs(model, tok, _prompt(state, instr, crit), keys, args.max_ctx)
        pick = keys[max(range(len(keys)), key=lambda j: lps[j])]
        gold = str(row["expected"]).strip().lower()
        surface = row.get("provenance", {}).get("surface_answer")
        surface = str(surface).strip().lower() if surface is not None else None
        if row["question"].get("type") == "noul":
            # criteria keys are true/false; JevBench gold and surface_answer are yes/no
            pick = {"true": "yes", "false": "no"}.get(pick, pick)
        results.append({
            "id": row["id"], "family": row["family"], "type": row["question"].get("type", "choice"),
            "n_options": len(keys), "pick": pick, "expected": gold, "hit": pick == gold,
            "surface": surface, "pick_is_surface": surface is not None and pick == surface,
        })
        if i % 10 == 0 or i == len(rows):
            acc = sum(r["hit"] for r in results) / len(results)
            print(f"  [{i}/{len(rows)}] acc so far {acc:.1%}  ({time.time()-t0:.0f}s)", flush=True)

    n = len(results)
    acc = sum(r["hit"] for r in results) / n
    chance = sum(1.0 / r["n_options"] for r in results) / n
    print(f"\n== {args.model} on numeric slice ==  n={n}  acc {acc:.1%}  (chance {chance:.1%})")
    fams = sorted({r["family"] for r in results})
    for f in fams:
        sel = [r for r in results if r["family"] == f]
        h = sum(r["hit"] for r in sel)
        ch = sum(1.0 / r["n_options"] for r in sel) / len(sel)
        line = f"    {f:18s} {h}/{len(sel)} = {h/len(sel):.0%}  (chance {ch:.0%})"
        sur = [r for r in sel if r["surface"] is not None]
        if sur:
            s = sum(r["pick_is_surface"] for r in sur)
            line += f"   P(pick==surface) {s}/{len(sur)}"
        print(line)

    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        json.dump({"model": args.model, "accuracy": acc, "chance": chance, "results": results},
                  open(args.dump, "w", encoding="utf-8"), indent=1)
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
