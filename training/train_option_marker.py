"""Trains Von's Option-Marker Joint Attention Decision Head with RLCD calibration.

Enables single-pass non-autoregressive decision evaluation:
- Pack state and all options into one sequence marked by [MASK] tokens
- Evaluates joint relative competition across all candidate choices in 1 forward pass
- Calibrated with composite Cross-Entropy + Brier score loss
"""

import argparse
import json
import math
import os
import random
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from von.models.option_marker import OptionMarkerModel, split_digits


class OptionMarkerDataset(Dataset):
    def __init__(self, jsonl_path: str, validate: bool = True):
        self.rows = []
        try:
            f = open(jsonl_path, "r", encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Cannot read training corpus {jsonl_path!r}: {exc}") from exc

        with f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{jsonl_path}:{lineno}: malformed JSON in corpus ({exc.msg}). "
                        f"Refusing to train on a partially readable corpus."
                    ) from exc

                if validate:
                    # The collator falls back to target index 0 when the label does
                    # not name an option, which silently trains toward the wrong
                    # answer instead of failing. Catch it at load time instead.
                    opts = row.get("options")
                    if not isinstance(opts, list) or len(opts) < 2:
                        raise ValueError(
                            f"{jsonl_path}:{lineno}: record needs at least 2 options, got {opts!r}"
                        )
                    opt_ids = [o.get("id") for o in opts]
                    if row.get("label") not in opt_ids:
                        raise ValueError(
                            f"{jsonl_path}:{lineno}: label {row.get('label')!r} matches no option id "
                            f"{opt_ids!r}. This would silently train toward option 0."
                        )

                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"{jsonl_path}: corpus is empty.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def estimate_packed_tokens(item: dict) -> int:
    """Cheap character-based estimate of an item's packed sequence length.

    Tokenizing a 290k-row corpus just to bucket it costs more than the training
    step it feeds, so approximate from character count instead.

    The 4.9 chars/token divisor is calibrated against this corpus measured
    through the real tokenizer. An earlier 3.6 divisor over-estimated real
    length by ~1.37x, which silently desynchronised the sampler's long/short
    split from the true token counts and made the timing profile misreport every
    long batch as short. Memory headroom is handled explicitly by
    BATCH_SAFETY_FACTOR rather than by hiding slack in this divisor.
    """
    n = len(item.get("state", "")) + len(item.get("question", ""))
    for opt in item.get("options", []):
        n += len(opt.get("description", "")) + 8  # +8 for the mask/sep scaffolding
    return max(8, n * 10 // 49)  # ~4.9 chars/token, in integer arithmetic


# Estimates are approximate, so leave explicit headroom against the token budget.
BATCH_SAFETY_FACTOR = 1.25


class LengthBucketedBatchSampler(Sampler):
    """Batches by length so long-context examples actually reach the model.

    Two problems are solved together:

    1. *Exposure.* Long examples are a small minority of the corpus, so uniform
       shuffling means the model almost never sees a full-length window. Long
       examples are oversampled until they account for `long_ratio` of batches.

    2. *Memory.* The collator pads to the longest item in the batch, so a single
       3k-token document in a batch of 8 inflates that batch to ~24k tokens and
       OOMs a 16GB card. Batches are therefore built against a token budget:
       long batches automatically get fewer rows.

    DDP safety: every rank derives the identical global batch list from
    (seed, epoch), then takes a strided shard truncated to a common length. Ranks
    that disagree on batch count deadlock at the gradient all-reduce, so the
    truncation is load-bearing, not tidiness.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        max_tokens: int = 8192,
        long_threshold: int = 2048,
        long_ratio: float = 0.30,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.long_threshold = long_threshold
        self.long_ratio = long_ratio
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.long_idx = [i for i, L in enumerate(lengths) if L >= long_threshold]
        self.short_idx = [i for i, L in enumerate(lengths) if L < long_threshold]
        self._cached_len = len(self._build_batches())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _pack(self, indices: List[int]) -> List[List[int]]:
        """Group pre-sorted indices into batches respecting the token budget."""
        batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        for i in indices:
            cand_max = max(cur_max, self.lengths[i])
            # Padded cost is (rows * longest row), which is what actually allocates.
            projected = (len(cur) + 1) * cand_max * BATCH_SAFETY_FACTOR
            if cur and (projected > self.max_tokens or len(cur) >= self.batch_size):
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = cand_max
        if cur and not self.drop_last:
            batches.append(cur)
        elif cur and len(cur) == self.batch_size:
            batches.append(cur)
        return batches

    def _build_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        short = list(self.short_idx)
        rng.shuffle(short)
        short_batches = self._pack(short)

        long_batches: List[List[int]] = []
        if self.long_idx:
            # Target count so long batches are `long_ratio` of the final mix.
            n_short = len(short_batches)
            target_long = round(n_short * self.long_ratio / max(1e-6, 1 - self.long_ratio))

            pool: List[int] = []
            while True:
                chunk = list(self.long_idx)
                rng.shuffle(chunk)
                pool.extend(chunk)
                # Sort within the pool so similar lengths batch together (less padding).
                probe = sorted(pool, key=lambda i: self.lengths[i])
                if len(self._pack(probe)) >= target_long or not target_long:
                    pool = probe
                    break
            long_batches = self._pack(pool)[:target_long] if target_long else []

        if not long_batches:
            batches = short_batches
            rng.shuffle(batches)
        else:
            # Interleave evenly rather than shuffling, so long batches are spread
            # across the epoch instead of clumping into a late memory spike.
            batches = list(short_batches)
            stride = max(1, len(batches) // max(1, len(long_batches)))
            for k, lb in enumerate(long_batches):
                pos = min(len(batches), k * (stride + 1))
                batches.insert(pos, lb)

        # Equal batch count per rank: unequal counts deadlock DDP all-reduce.
        per_rank = len(batches) // self.num_replicas
        if per_rank == 0:
            return batches[self.rank:self.rank + 1]
        return batches[self.rank:per_rank * self.num_replicas:self.num_replicas]

    def __iter__(self):
        return iter(self._build_batches())

    def __len__(self) -> int:
        return self._cached_len


def collate_marker_fn(batch: List[dict], tokenizer, max_length: int = 8192, digit_split: bool = False):
    packed_texts = []
    labels = []
    soft_targets: List[Optional[List[float]]] = []
    mask = tokenizer.mask_token
    sep = tokenizer.sep_token

    for item in batch:
        state = item["state"].strip()
        q = item["question"].strip()
        opts = item["options"]
        target = item["label"]

        opt_ids = [opt["id"] for opt in opts]
        target_idx = opt_ids.index(target) if target in opt_ids else 0
        labels.append(target_idx)

        # Distilled rows carry a full probability distribution over options.
        # One-hot training can only ever teach certainty; the soft target is
        # what the distribution-fidelity half of the Calibration axis scores.
        soft = item.get("target")
        if isinstance(soft, list) and len(soft) == len(opts):
            total_mass = float(sum(soft))
            soft_targets.append([float(x) / total_mass for x in soft] if total_mass > 0 else None)
        else:
            soft_targets.append(None)

        prefix = f"{q} {state}".strip() if q else state
        opts_packed = " ".join(f"{mask} {opt['description'].strip()}" for opt in opts)
        packed = f"{prefix} {sep} {opts_packed}"
        packed_texts.append(split_digits(packed) if digit_split else packed)

    encodings = tokenizer(
        packed_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    batch_mask_positions = []
    mask_id = tokenizer.mask_token_id
    for b in range(len(batch)):
        pos = (encodings["input_ids"][b] == mask_id).nonzero(as_tuple=True)[0].tolist()
        batch_mask_positions.append(pos)

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
        "soft_targets": soft_targets,
        "mask_positions": batch_mask_positions,
    }


def compute_marker_rlcd_loss(
    batch_logits: List[torch.Tensor],
    labels: torch.Tensor,
    brier_weight: float = 0.5,
    soft_targets: Optional[List[Optional[List[float]]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Cross-entropy + Brier against the target distribution.

    Both terms accept a full distribution, so a soft target is a strict
    generalisation of the one-hot case rather than a separate code path.
    Accuracy is still measured against the argmax, so it stays comparable to
    runs trained on hard labels.
    """
    ce_losses = []
    brier_losses = []
    correct = 0
    total = len(labels)

    for i, logits in enumerate(batch_logits):
        target_idx = labels[i].item()
        probs = torch.softmax(logits, dim=-1)

        safe_target = min(target_idx, probs.size(0) - 1)

        soft = soft_targets[i] if soft_targets is not None else None
        if soft is not None and len(soft) == probs.size(0):
            dist = torch.tensor(soft, dtype=probs.dtype, device=probs.device)
        else:
            dist = torch.zeros_like(probs)
            dist[safe_target] = 1.0

        ce = -torch.sum(dist * torch.log(probs + 1e-8))
        ce_losses.append(ce)
        brier_losses.append(torch.sum((probs - dist) ** 2))

        if torch.argmax(probs).item() == safe_target:
            correct += 1

    mean_ce = torch.stack(ce_losses).mean()
    mean_brier = torch.stack(brier_losses).mean()
    total_loss = mean_ce + brier_weight * mean_brier
    accuracy = correct / max(total, 1)

    return total_loss, mean_ce, accuracy


def _write_json(path: str, payload: dict) -> None:
    """Write JSON atomically: temp file in the same directory + os.replace.

    marker_calibration.json is read at serve time by OptionMarkerBackend, and
    written here mid-training on instances that can be killed at any moment
    (the watchdog in launch_universal_training.py did exactly this once --
    see 8fd35a3db9c5). A plain open(path, 'w') truncates the file before
    writing the replacement, so a kill between truncate and flush leaves a
    zero-byte or partial file; a reader (or the next training run resuming
    from it) gets a JSONDecodeError instead of the last-good config. Writing
    to a sibling temp file and renaming over the target is atomic on POSIX
    (same filesystem, same directory) -- readers see either the old file or
    the new one, never a partial one.
    """
    try:
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except OSError as exc:
        raise RuntimeError(f"Cannot write {path!r}: {exc}") from exc


def _env_int(name: str) -> int:
    """Read a required integer env var, naming it if it is missing or malformed."""
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"DDP environment variable {name} is not set.")
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"DDP environment variable {name}={raw!r} is not an integer.") from exc


def _report_step_profile(
    profile: Dict[str, List[float]],
    batches_per_epoch: int,
    epochs: int,
    world_size: int,
    seqlens: Optional[List[Tuple[int, int, float]]] = None,
    long_threshold: int = 2048,
) -> None:
    """Turn a short probe run into a concrete full-run cost estimate."""
    import statistics

    short, long_ = profile["short"], profile["long"]
    # Drop the first few steps: cuDNN autotuning and allocator warmup make them
    # unrepresentative of steady state.
    short, long_ = short[3:] or short, long_[3:] or long_

    print("\n================ STEP TIMING PROFILE ================")
    for name, xs in (("short", short), ("long", long_)):
        if xs:
            print(f"  {name:>5} batches: n={len(xs):4d}  median={statistics.median(xs):.3f}s  "
                  f"mean={statistics.mean(xs):.3f}s")
        else:
            print(f"  {name:>5} batches: none observed")

    if short and long_:
        ratio = statistics.median(long_) / statistics.median(short)
        print(f"  long/short cost ratio: {ratio:.1f}x")

    if seqlens:
        # Report by REAL tokenized length. A binary long/short flag computed from
        # estimates once hid every long batch in the short bucket; the buckets
        # below are measured, so that failure cannot recur silently.
        print("\n  by real padded sequence length:")
        bounds = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 1 << 30)]
        for lo, hi in bounds:
            sel = [(L, rows, dt) for L, rows, dt in seqlens if lo <= L < hi]
            if sel:
                med = statistics.median([dt for _, _, dt in sel])
                rows_med = statistics.median([rows for _, rows, dt in sel])
                label = f"{lo}-{hi}" if hi < (1 << 30) else f"{lo}+"
                print(f"    {label:>10} tok: n={len(sel):4d}  median={med:.3f}s  rows/batch={rows_med:.0f}")
        cheap = [dt for L, _, dt in seqlens if L < long_threshold]
        pricey = [dt for L, _, dt in seqlens if L >= long_threshold]
        if cheap and pricey:
            print(f"    measured cost ratio (>={long_threshold} vs <): "
                  f"{statistics.median(pricey) / statistics.median(cheap):.1f}x")

    observed = short + long_
    if observed:
        frac_long = len(long_) / len(observed)
        mean_step = statistics.mean(observed)
        epoch_s = mean_step * batches_per_epoch
        total_h = epoch_s * epochs / 3600
        print(f"\n  observed long-batch share: {frac_long:.1%}")
        print(f"  batches/epoch (per rank):  {batches_per_epoch:,}")
        print(f"  projected epoch time:      {epoch_s/3600:.2f}h")
        print(f"  projected {epochs}-epoch run:     {total_h:.2f}h")
        for rate, label in ((3.912, "g4dn.12xlarge on-demand"), (5.672, "g5.12xlarge on-demand")):
            print(f"    est. cost @ ${rate}/hr ({label}): ${total_h * rate:.2f}")
    print("=====================================================\n")


def train(
    train_path: str,
    val_path: str,
    base_model_id: str,
    output_dir: str,
    s3_target: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr: float = 3e-5,
    brier_weight: float = 0.5,
    max_position_embeddings: int = 8192,
    max_length: int = 8192,
    length_bucketing: bool = True,
    max_tokens_per_batch: int = 0,
    long_threshold: int = 2048,
    long_ratio: float = 0.30,
    max_steps: int = 0,
    init_checkpoint: Optional[str] = None,
    independent_options: bool = False,
    digit_split: bool = False,
):
    is_ddp = "RANK" in os.environ
    if is_ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = _env_int("RANK")
        local_rank = _env_int("LOCAL_RANK")
        world_size = _env_int("WORLD_SIZE")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = (rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        rank = 0
        world_size = 1

    if is_main:
        print(f"Device: {device} (World Size: {world_size}, DDP: {is_ddp})")
        print(f"Loading OptionMarkerModel with base: {base_model_id}...")

    model = OptionMarkerModel(
        base_model_id=base_model_id,
        max_position_embeddings=max_position_embeddings,
        digit_split=digit_split,
    ).to(device)
    if init_checkpoint:
        ckpt_path = init_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "option_marker.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--init_checkpoint given but no weights found at {ckpt_path}")
        if is_main:
            print(f"Continuing training from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"--init_checkpoint state_dict does not match model architecture "
                f"(missing={missing}, unexpected={unexpected})"
            )
    tokenizer = model.tokenizer

    train_ds = OptionMarkerDataset(train_path)
    val_ds = OptionMarkerDataset(val_path)

    train_sampler = None
    batch_sampler = None
    if length_bucketing and max_tokens_per_batch <= 0:
        # Auto-size from real device memory. A fixed budget calibrated on a 24GB
        # A10G OOMs a 14.5GB T4, and DDP gradient buckets cost more than the
        # single-GPU probe measured, so derive it and stay conservative.
        if device.type == "cuda":
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            max_tokens_per_batch = int(min(8192, max(2048, total_gb * 220)))
        else:
            max_tokens_per_batch = 4096
        if is_main:
            print(f"  -> Auto token budget: {max_tokens_per_batch:,} tok/batch "
                  f"({'%.1f' % total_gb if device.type == 'cuda' else 'cpu'} GB/GPU)")

    if length_bucketing:
        # Bucket by estimated length so long documents are both seen often enough
        # and batched small enough to fit. Falls back to plain shuffling if the
        # corpus turns out to have no long examples at all.
        lengths = [estimate_packed_tokens(r) for r in train_ds.rows]
        n_long = sum(1 for L in lengths if L >= long_threshold)
        if n_long == 0:
            if is_main:
                print(f"  !! No examples >= {long_threshold} tokens; length bucketing disabled.")
            length_bucketing = False
        else:
            batch_sampler = LengthBucketedBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                max_tokens=max_tokens_per_batch,
                long_threshold=long_threshold,
                long_ratio=long_ratio,
                num_replicas=world_size,
                rank=rank,
                seed=42,
            )
            if is_main:
                print(f"  -> Length bucketing: {n_long:,}/{len(lengths):,} rows >= {long_threshold} tok "
                      f"({n_long / len(lengths):.1%}); target {long_ratio:.0%} of batches, "
                      f"budget {max_tokens_per_batch:,} tok/batch")

    if batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            collate_fn=lambda b: collate_marker_fn(b, tokenizer, max_length=max_length, digit_split=digit_split),
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=lambda b: collate_marker_fn(b, tokenizer, max_length=max_length, digit_split=digit_split),
            pin_memory=(device.type == "cuda"),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_marker_fn(b, tokenizer, max_length=max_length, digit_split=digit_split),
        pin_memory=(device.type == "cuda"),
    )

    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    if is_main:
        print(f"\nStarting Option-Marker training:")
        print(f"  -> Train Samples:   {len(train_ds):,}")
        print(f"  -> Val Samples:     {len(val_ds):,}")
        print(f"  -> Batch Size:      {batch_size}")
        print(f"  -> Grad Accum:      {grad_accum_steps} (Effective: {batch_size * grad_accum_steps * world_size})")
        print(f"  -> Total Steps:     {total_steps:,}\n")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr * 0.5},
            {"params": model.scorer.parameters(), "lr": lr * 2.5},
        ],
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 8 // 100,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        elif is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        profile: Dict[str, List[float]] = {"short": [], "long": []}
        profile_seqlens: List[Tuple[int, int, float]] = []

        for step, batch in enumerate(train_loader):
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_t0 = time.time()
            seq_len = batch["input_ids"].shape[1]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask_positions = batch["mask_positions"]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                batch_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_positions=mask_positions,
                    independent_options=independent_options,
                )
                loss, ce_loss, acc = compute_marker_rlcd_loss(
                    batch_logits, labels, brier_weight=brier_weight,
                        soft_targets=batch.get("soft_targets")
                )
                accum_loss = loss / grad_accum_steps

            scaler.scale(accum_loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_dt = time.time() - step_t0
            profile["long" if seq_len >= long_threshold else "short"].append(step_dt)
            profile_seqlens.append((seq_len, batch["input_ids"].shape[0], step_dt))

            if is_main and ((step + 1) % 100 == 0 or (step + 1) == len(train_loader)):
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}) Acc: {acc*100:.1f}% "
                    f"Elapsed: {elapsed:.1f}s"
                )

            if max_steps and (step + 1) >= max_steps:
                if is_main:
                    _report_step_profile(profile, len(train_loader), epochs, world_size,
                                         profile_seqlens, long_threshold)
                    # A time-boxed run (--max_steps) is a real checkpoint request, not
                    # just a timing probe, whenever an init_checkpoint/output_dir is
                    # given: save unconditionally so a short-budget continuation isn't
                    # thrown away for lack of a completed epoch's validation pass.
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(output_dir, "option_marker.pt"))
                        model.encoder.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        _write_json(
                            os.path.join(output_dir, "marker_calibration.json"),
                            {
                                "model_type": "option_marker",
                                "base_model": base_model_id,
                                "init_checkpoint": init_checkpoint,
                                "max_steps_reached": step + 1,
                                "independent_options": independent_options,
                                "digit_split": digit_split,
                                "validated": False,
                                "timestamp": time.time(),
                            },
                        )
                        print(f"Saved max_steps checkpoint ({step + 1} steps, unvalidated) to {output_dir}")
                    except OSError as exc:
                        raise RuntimeError(
                            f"Failed saving max_steps checkpoint to {output_dir!r}: {exc}"
                        ) from exc
                    if s3_target:
                        print(f"Syncing max_steps checkpoint to S3: {s3_target} ...")
                        os.system(f"/usr/bin/aws s3 cp --recursive {output_dir}/ {s3_target}/ || aws s3 cp --recursive {output_dir}/ {s3_target}/")
                return

        # Validation
        if is_main:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_scores_list = []
            val_labels_list = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    mask_positions = batch["mask_positions"]

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        batch_logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            mask_positions=mask_positions,
                            independent_options=independent_options,
                        )
                        loss, _, acc = compute_marker_rlcd_loss(
                            batch_logits, labels, brier_weight=brier_weight,
                        soft_targets=batch.get("soft_targets")
                        )

                    val_loss += loss.item()
                    val_acc += acc

                    for i, logits in enumerate(batch_logits):
                        val_scores_list.append(logits.cpu())
                        val_labels_list.append(labels[i].item())

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)
            print(f"\n--- Epoch {epoch} Validation: Loss = {avg_val_loss:.4f}, Accuracy = {avg_val_acc*100:.2f}% ---\n")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                # A failure here costs the whole run's best checkpoint, so say
                # exactly what broke rather than surfacing a bare OSError.
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(output_dir, "option_marker.pt"))
                    model.encoder.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    _write_json(
                        os.path.join(output_dir, "marker_calibration.json"),
                        {
                            "model_type": "option_marker",
                            "base_model": base_model_id,
                            "best_val_accuracy": round(best_val_acc, 4),
                            "epoch": epoch,
                            "independent_options": independent_options,
                            "digit_split": digit_split,
                            "timestamp": time.time(),
                        },
                    )
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed saving epoch {epoch} checkpoint to {output_dir!r}: {exc}"
                    ) from exc

                if s3_target:
                    print(f"Syncing Epoch {epoch} checkpoint to S3: {s3_target} ...")
                    os.system(f"/usr/bin/aws s3 cp --recursive {output_dir}/ {s3_target}/ || aws s3 cp --recursive {output_dir}/ {s3_target}/")

    if is_main:
        calib_config = {
            "model_type": "option_marker",
            "base_model": base_model_id,
            "best_val_accuracy": round(best_val_acc, 4),
            "independent_options": independent_options,
            "digit_split": digit_split,
            "timestamp": time.time(),
        }
        _write_json(os.path.join(output_dir, "marker_calibration.json"), calib_config)

        if s3_target:
            print(f"Uploading artifacts to S3: {s3_target} ...")
            os.system(f"/usr/bin/aws s3 cp --recursive {output_dir}/ {s3_target}/ || aws s3 cp --recursive {output_dir}/ {s3_target}/")
            print("=== [OPTION-MARKER TRAINING COMPLETE] ===")

    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Option-Marker Joint Attention Model")
    parser.add_argument("--train_data", type=str, default="data_decision/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data_decision/val.jsonl")
    parser.add_argument("--base_model_id", type=str, default="checkpoints/von-modernbert-rlcd")
    parser.add_argument("--output_dir", type=str, default="checkpoints/von-option-marker")
    parser.add_argument("--s3_target", type=str, default="s3://model-weight/von-option-marker")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_position_embeddings", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--brier_weight", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Tokenizer truncation length during training. Must match inference-time "
                             "context or the scorer head never learns long-premise aggregation.")
    parser.add_argument("--no_length_bucketing", action="store_true",
                        help="Disable length-bucketed batching (uniform shuffling instead).")
    parser.add_argument("--max_tokens_per_batch", type=int, default=0,
                        help="Padded token budget per batch (0 = auto-size from GPU memory). "
                             "Long batches get fewer rows so a single long document cannot "
                             "OOM the card.")
    parser.add_argument("--long_threshold", type=int, default=2048,
                        help="Token count at or above which an example counts as long.")
    parser.add_argument("--long_ratio", type=float, default=0.30,
                        help="Target fraction of batches drawn from long examples.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after N steps. If output_dir/s3_target are set this "
                             "also saves an unvalidated checkpoint (time-boxed continuation "
                             "run); otherwise it behaves as a timing probe only.")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Path to an existing option_marker.pt (or its containing dir) "
                             "to continue training from, instead of a fresh randomly-"
                             "initialised scoring head.")
    parser.add_argument("--digit_split", action="store_true",
                        help="Space out every digit in every digit run before packing "
                             "(\"2026\" -> \"2 0 2 6\"), matching the same transform applied "
                             "in OptionMarkerModel.pack_sequence at inference. Fixes "
                             "ModernBERT's leading-digit-dependent BPE merging (see "
                             "split_digits in src/von/models/option_marker.py).")
    parser.add_argument("--independent_options", action="store_true",
                        help="Train with an attention mask + position-id scheme that blocks "
                             "option-to-option attention, making each option's logit a "
                             "provably order-invariant function of (state, that option) alone. "
                             "See build_independent_option_masks in src/von/models/option_marker.py.")
    args = parser.parse_args()

    train(
        train_path=args.train_data,
        val_path=args.val_data,
        base_model_id=args.base_model_id,
        output_dir=args.output_dir,
        s3_target=args.s3_target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        brier_weight=args.brier_weight,
        max_position_embeddings=args.max_position_embeddings,
        max_length=args.max_length,
        length_bucketing=not args.no_length_bucketing,
        max_tokens_per_batch=args.max_tokens_per_batch,
        long_threshold=args.long_threshold,
        long_ratio=args.long_ratio,
        max_steps=args.max_steps,
        init_checkpoint=args.init_checkpoint,
        independent_options=args.independent_options,
        digit_split=args.digit_split,
    )
