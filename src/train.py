from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model import CompactVideoLM, CompactVideoLMConfig


class StreamDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.input_ids = np.load(self.data_dir / "input_ids.npy", mmap_mode="r")
        self.type_ids = np.load(self.data_dir / "type_ids.npy", mmap_mode="r")
        self.source_pos = np.load(self.data_dir / "source_pos.npy", mmap_mode="r")
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text())
        assert len(self.input_ids) == len(self.type_ids) == len(self.source_pos)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.int32),
            "type_ids": torch.tensor(self.type_ids[idx], dtype=torch.int16),
            "source_pos": torch.tensor(self.source_pos[idx], dtype=torch.int32),
        }


def is_distributed(): return int(os.environ.get("WORLD_SIZE", "1")) > 1
def get_rank():       return int(os.environ.get("RANK", "0"))
def get_world_size(): return int(os.environ.get("WORLD_SIZE", "1"))
def is_main():        return get_rank() == 0


def setup_distributed():
    if is_distributed() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sync_pair(value, weight, device):
    if not dist.is_initialized():
        return value, weight
    t = torch.tensor([value, weight], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), float(t[1].item())


def sync_dict(d, device):
    if not dist.is_initialized():
        return d
    keys = sorted(d.keys())
    t = torch.tensor([d[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: float(t[i].item()) for i, k in enumerate(keys)}


def cosine_lr(step, total_steps, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * float(step + 1) / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, device, amp_bf16, max_batches, clamp_fine_to_coarse=False) -> dict:
    model.eval()
    loss_sum = 0.0; token_count = 0.0
    per_type_sum = {"nll_special": 0.0, "nll_coarse": 0.0, "nll_fine": 0.0,
                    "tokens_special": 0.0, "tokens_coarse": 0.0, "tokens_fine": 0.0}
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device=device, dtype=torch.long, non_blocking=True)
        type_ids = batch["type_ids"].to(device=device, dtype=torch.long, non_blocking=True)
        source_pos = batch["source_pos"].to(device=device, dtype=torch.long, non_blocking=True)
        m = model.module if isinstance(model, DDP) else model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=amp_bf16 and device.type == "cuda"):
            _, loss, pt = m(input_ids=input_ids, source_pos=source_pos, type_ids=type_ids,
                            return_per_type_loss=True,
                            clamp_fine_to_coarse=clamp_fine_to_coarse)
        tokens = float(input_ids.shape[0] * max(1, input_ids.shape[1] - 1))
        loss_sum += float(loss.item()) * tokens
        token_count += tokens
        for name in ["special", "coarse", "fine"]:
            per_type_sum[f"nll_{name}"] += pt[f"nll_{name}"] * pt[f"tokens_{name}"]
            per_type_sum[f"tokens_{name}"] += pt[f"tokens_{name}"]

    loss_sum, token_count = sync_pair(loss_sum, token_count, device)
    per_type_sum = sync_dict(per_type_sum, device)

    mean_loss = loss_sum / max(1.0, token_count)
    ppl = math.exp(min(20.0, mean_loss))
    per_type = {}
    for name in ["special", "coarse", "fine"]:
        n = per_type_sum[f"tokens_{name}"]
        per_type[f"nll_{name}"] = per_type_sum[f"nll_{name}"] / max(1.0, n)
        per_type[f"tokens_{name}"] = n
    return {"loss": mean_loss, "ppl": ppl, "tokens": token_count, **per_type}


def build_model(args, manifest) -> CompactVideoLM:
    cfg = CompactVideoLMConfig(
        vocab_size=int(manifest["vocab_size"]),
        source_pos_vocab_size=int(manifest["source_pos_vocab_size"]),
        dim=args.dim, n_layers=args.layers, n_heads=args.heads,
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        pad_token_id=int(manifest["pad_token_id"]),
        tie_embeddings=not args.untied_embeddings,
        max_seq_len=max(2048, int(manifest["seq_len"])),
        loss_head_chunk=int(getattr(args, "loss_head_chunk", 0) or 0),
    )
    return CompactVideoLM(cfg)


def save_checkpoint(path, model, optimizer, step, epoch, best_val_loss,
                    model_cfg, train_manifest, val_manifest):
    module = model.module if isinstance(model, DDP) else model
    torch.save({
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step, "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_config": model_cfg.to_dict(),
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
    }, path)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def parse_args():
    ap = argparse.ArgumentParser(description="Train dense/packed stream LM.")
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--val-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--target-train-tokens", type=float, default=0.0,
                    help="training-token budget")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-max-batches", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--amp-bf16", action="store_true")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--untied-embeddings", action="store_true")
    ap.add_argument("--loss-head-chunk", type=int, default=0,
                    help="CE head chunk (0 = off)")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="checkpoint to resume")
    ap.add_argument("--early-stop-patience", type=int, default=5,
                    help="early-stop patience (0 = off)")
    ap.add_argument("--clamp-fine-to-coarse", action="store_true",
                    help="block fine-to-coarse attention")
    return ap.parse_args()


def main():
    args = parse_args()
    setup_distributed()
    run_name = args.run_name or args.out_dir.name
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    rank = get_rank(); world = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StreamDataset(args.train_dir); val_ds = StreamDataset(args.val_dir)
    train_manifest = train_ds.manifest; val_manifest = val_ds.manifest
    if int(train_manifest["vocab_size"]) != int(val_manifest["vocab_size"]):
        raise ValueError("train/val vocab size mismatch")
    if int(train_manifest["seq_len"]) != int(val_manifest["seq_len"]):
        _msl = max(int(train_manifest["seq_len"]), int(val_manifest["seq_len"]))
        print(f"[train] seq_len mismatch train={train_manifest['seq_len']} val={val_manifest['seq_len']} -> RoPE model max_seq_len={_msl}", flush=True)
        train_manifest["seq_len"] = _msl
        val_manifest["seq_len"] = _msl

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True,
                            drop_last=False, persistent_workers=args.num_workers > 0)

    model = build_model(args, train_manifest).to(device)
    model_cfg_obj = model.cfg
    n_params = model.num_parameters()

    if is_main():
        (args.out_dir / "model_config.json").write_text(json.dumps(model_cfg_obj.to_dict(), indent=2))
        (args.out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str))
        meta = {"run_name": run_name, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_params": n_params, "world_size": world, "host": platform.node(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu_name": (torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"),
                "git_sha": git_sha(),
                "train_manifest": train_manifest, "val_manifest": val_manifest}
        (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[train] run={run_name} params={n_params/1e6:.1f}M seq_len={train_manifest['seq_len']} "
              f"vocab={train_manifest['vocab_size']} bsz={args.batch_size} world={world}", flush=True)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    resume_step = 0
    if getattr(args, "resume_from", None):
        rck = torch.load(args.resume_from, map_location=device, weights_only=False)
        (model.module if isinstance(model, DDP) else model).load_state_dict(rck["model"])
        optimizer.load_state_dict(rck["optimizer"])
        resume_step = int(rck.get("step", 0))
        print(f"[train] RESUMED from {args.resume_from} at optim_step {resume_step}", flush=True)

    seq_len = int(train_manifest["seq_len"])
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    tokens_per_optim_step = args.batch_size * seq_len * world * args.grad_accum
    if args.max_steps > 0:
        total_steps = args.max_steps
    elif args.target_train_tokens > 0:
        total_steps = max(1, math.ceil(args.target_train_tokens / max(1, tokens_per_optim_step)))
    else:
        total_steps = args.epochs * steps_per_epoch

    best_val_loss = float("inf"); patience_counter = 0
    step = 0; epoch = 0; optim_steps = 0
    tokens_seen = float(resume_step) * tokens_per_optim_step
    examples_seen = tokens_seen / max(1, seq_len)
    if resume_step: print(f"[train] resume tokens_seen={tokens_seen:.0f}", flush=True)

    train_log = args.out_dir / "train_log.jsonl"
    val_log = args.out_dir / "val_log.jsonl"
    curve_csv = args.out_dir / "curve.csv"
    per_type_log = args.out_dir / "per_type_nll.jsonl"
    if is_main():
        for p in (train_log, val_log, per_type_log):
            p.write_text("")
        with curve_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["optim_step", "tokens_seen", "val_loss", "val_ppl",
                                    "nll_special", "nll_coarse", "nll_fine", "elapsed_sec"])

    start_time = time.time()
    interval_start = time.time(); interval_tokens = 0.0; interval_examples = 0.0
    stopped = False

    while not stopped:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            lr = cosine_lr(optim_steps, max(1, total_steps), args.warmup_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups: g["lr"] = lr

            input_ids = batch["input_ids"].to(device=device, dtype=torch.long, non_blocking=True)
            type_ids = batch["type_ids"].to(device=device, dtype=torch.long, non_blocking=True)
            source_pos = batch["source_pos"].to(device=device, dtype=torch.long, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.amp_bf16 and device.type == "cuda"):
                _, loss = model(input_ids=input_ids, source_pos=source_pos, type_ids=type_ids,
                                clamp_fine_to_coarse=args.clamp_fine_to_coarse)
                loss = loss / args.grad_accum
            loss.backward()

            local_examples = float(input_ids.shape[0]); local_tokens = float(input_ids.numel())
            examples_seen += local_examples * world; tokens_seen += local_tokens * world
            interval_examples += local_examples * world; interval_tokens += local_tokens * world

            if ((batch_idx + 1) % args.grad_accum) == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                optim_steps += 1

                if is_main() and optim_steps % args.log_every == 0:
                    if device.type == "cuda": torch.cuda.synchronize(device)
                    now = time.time(); dt = max(1e-6, now - interval_start)
                    train_loss = float(loss.item() * args.grad_accum)
                    gpu_mem_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else 0.0
                    rec = {"optim_step": optim_steps, "epoch": epoch, "lr": lr,
                           "train_loss": train_loss, "tokens_seen": tokens_seen,
                           "examples_seen": examples_seen,
                           "interval_tokens_per_sec": interval_tokens / dt,
                           "interval_examples_per_sec": interval_examples / dt,
                           "elapsed_sec": now - start_time, "peak_gpu_mem_gb": gpu_mem_gb}
                    with train_log.open("a") as f: f.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                    interval_start = now; interval_tokens = 0.0; interval_examples = 0.0

                if optim_steps % args.eval_every == 0 or optim_steps == 1:
                    m = evaluate(model, val_loader, device, args.amp_bf16, args.val_max_batches,
                                 clamp_fine_to_coarse=args.clamp_fine_to_coarse)
                    if is_main():
                        rec = {"optim_step": optim_steps, "tokens_seen": tokens_seen,
                               "eval_loss": m["loss"], "eval_ppl": m["ppl"],
                               "nll_special": m["nll_special"], "nll_coarse": m["nll_coarse"],
                               "nll_fine": m["nll_fine"],
                               "tokens_special": m["tokens_special"], "tokens_coarse": m["tokens_coarse"],
                               "tokens_fine": m["tokens_fine"],
                               "elapsed_sec": time.time() - start_time}
                        with val_log.open("a") as f: f.write(json.dumps(rec) + "\n")
                        with per_type_log.open("a") as f: f.write(json.dumps(rec) + "\n")
                        with curve_csv.open("a", newline="") as f:
                            csv.writer(f).writerow([optim_steps, tokens_seen, m["loss"], m["ppl"],
                                                    m["nll_special"], m["nll_coarse"], m["nll_fine"],
                                                    rec["elapsed_sec"]])
                        print(json.dumps(rec), flush=True)

                        improved = m["loss"] < best_val_loss - 1e-4
                        if improved:
                            best_val_loss = m["loss"]; patience_counter = 0
                            save_checkpoint(args.out_dir / "best.pt", model, optimizer,
                                            step=optim_steps, epoch=epoch,
                                            best_val_loss=best_val_loss, model_cfg=model_cfg_obj,
                                            train_manifest=train_manifest, val_manifest=val_manifest)
                        else:
                            patience_counter += 1
                        save_checkpoint(args.out_dir / "latest.pt", model, optimizer,
                                        step=optim_steps, epoch=epoch,
                                        best_val_loss=best_val_loss, model_cfg=model_cfg_obj,
                                        train_manifest=train_manifest, val_manifest=val_manifest)

                        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                            print(f"[early-stop] {patience_counter} evals no improve, stop.", flush=True)
                            stopped = True

                if is_main() and args.save_every > 0 and optim_steps % args.save_every == 0:
                    save_checkpoint(args.out_dir / f"step_{optim_steps}.pt", model, optimizer,
                                    step=optim_steps, epoch=epoch,
                                    best_val_loss=best_val_loss, model_cfg=model_cfg_obj,
                                    train_manifest=train_manifest, val_manifest=val_manifest)

                if is_distributed():
                    st = torch.tensor([1 if stopped else 0], device=device)
                    dist.all_reduce(st, op=dist.ReduceOp.MAX)
                    stopped = bool(st.item())

                if args.max_steps > 0 and optim_steps >= args.max_steps: stopped = True
                if args.target_train_tokens > 0 and tokens_seen >= args.target_train_tokens: stopped = True
                if stopped: break

            step += 1
        epoch += 1
        if args.max_steps <= 0 and args.target_train_tokens <= 0 and epoch >= args.epochs:
            break
        if stopped: break

    m = evaluate(model, val_loader, device, args.amp_bf16, args.val_max_batches,
                 clamp_fine_to_coarse=args.clamp_fine_to_coarse)
    if is_main():
        rec = {"optim_step": optim_steps, "tokens_seen": tokens_seen,
               "eval_loss": m["loss"], "eval_ppl": m["ppl"],
               "nll_special": m["nll_special"], "nll_coarse": m["nll_coarse"], "nll_fine": m["nll_fine"],
               "tokens_special": m["tokens_special"], "tokens_coarse": m["tokens_coarse"],
               "tokens_fine": m["tokens_fine"],
               "elapsed_sec": time.time() - start_time, "final": True}
        with val_log.open("a") as f: f.write(json.dumps(rec) + "\n")
        with per_type_log.open("a") as f: f.write(json.dumps(rec) + "\n")
        with curve_csv.open("a", newline="") as f:
            csv.writer(f).writerow([optim_steps, tokens_seen, m["loss"], m["ppl"],
                                    m["nll_special"], m["nll_coarse"], m["nll_fine"],
                                    rec["elapsed_sec"]])
        print(json.dumps(rec), flush=True)
        save_checkpoint(args.out_dir / "latest.pt", model, optimizer,
                        step=optim_steps, epoch=epoch, best_val_loss=best_val_loss,
                        model_cfg=model_cfg_obj, train_manifest=train_manifest, val_manifest=val_manifest)
        if m["loss"] < best_val_loss:
            save_checkpoint(args.out_dir / "best.pt", model, optimizer,
                            step=optim_steps, epoch=epoch, best_val_loss=m["loss"],
                            model_cfg=model_cfg_obj, train_manifest=train_manifest, val_manifest=val_manifest)
        (args.out_dir / "done.json").write_text(json.dumps({
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_optim_steps": optim_steps, "total_tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss, "elapsed_sec": time.time() - start_time,
        }, indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    main()
