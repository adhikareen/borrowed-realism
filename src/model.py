from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompactVideoLMConfig:
    vocab_size: int
    source_pos_vocab_size: int
    type_vocab_size: int = 3
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    dropout: float = 0.1
    pad_token_id: int = 64000
    tie_embeddings: bool = True
    max_seq_len: int = 2048
    loss_head_chunk: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get(self, start: int, length: int, device, dtype):
        c = self.cos_cached[start:start + length].to(device=device, dtype=dtype)
        s = self.sin_cached[start:start + length].to(device=device, dtype=dtype)
        return c[None, None, :, :], s[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: CompactVideoLMConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base,
                                    max_seq_len=cfg.max_seq_len)

    def forward(self, x, kv_cache=None, start_pos=0, type_ids=None,
                clamp_fine_to_coarse=False, clamp_mode="coarse"):
        b, l, d = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope.get(start_pos, l, x.device, x.dtype)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)

        k_type_full = None
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v
            if clamp_fine_to_coarse and type_ids is not None:
                if "type" in kv_cache:
                    k_type_full = torch.cat([kv_cache["type"], type_ids], dim=1)
                else:
                    k_type_full = type_ids
                kv_cache["type"] = k_type_full
            is_causal = (l > 1)
        else:
            is_causal = True
            if clamp_fine_to_coarse and type_ids is not None:
                k_type_full = type_ids

        if clamp_fine_to_coarse and type_ids is not None:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            lq, lk = scores.shape[-2], scores.shape[-1]
            if is_causal:
                causal = torch.ones(lq, lk, device=scores.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            q_type = type_ids[:, -lq:][:, None, :, None]
            k_type = k_type_full[:, None, None, :]
            if clamp_mode == "coarse":
                fc_mask = (q_type == 2) & (k_type == 1)
            elif clamp_mode == "placebo_fine":
                is_fine_key = (k_type == 1)
                is_fine_key = (k_type_full == 2)[:, None, None, :]
                fine_rank = (k_type_full == 2).cumsum(dim=1) - 1
                early_fine = ((fine_rank < 320) & (k_type_full == 2))[:, None, None, :]
                fc_mask = (q_type == 2) & early_fine
            else:
                raise ValueError(f"unknown clamp_mode {clamp_mode}")
            scores = scores.masked_fill(fc_mask.expand_as(scores), float("-inf"))
            attn = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        y = y.transpose(1, 2).contiguous().view(b, l, d)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: CompactVideoLMConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.up = nn.Linear(cfg.dim, hidden, bias=False)
        self.gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: CompactVideoLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, kv_cache=None, start_pos=0, type_ids=None,
                clamp_fine_to_coarse=False, clamp_mode="coarse"):
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos,
                          type_ids=type_ids, clamp_fine_to_coarse=clamp_fine_to_coarse,
                          clamp_mode=clamp_mode)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CompactVideoLM(nn.Module):
    def __init__(self, cfg: CompactVideoLMConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.source_pos_emb = nn.Embedding(cfg.source_pos_vocab_size, cfg.dim)
        self.type_emb = nn.Embedding(cfg.type_vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, source_pos, type_ids, labels=None,
                kv_caches=None, start_pos: int = 0,
                return_per_type_loss: bool = False,
                clamp_fine_to_coarse: bool = False, clamp_mode: str = "coarse"):
        x = self.token_emb(input_ids) + self.source_pos_emb(source_pos) + self.type_emb(type_ids)
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache, start_pos=start_pos,
                      type_ids=type_ids, clamp_fine_to_coarse=clamp_fine_to_coarse,
                      clamp_mode=clamp_mode)
        if kv_caches is None and (self.cfg.loss_head_chunk or 0) > 0:
            return self._chunked_head_loss(x, input_ids, type_ids, labels,
                                           return_per_type_loss)

        logits = self.lm_head(self.norm(x))

        if kv_caches is not None:
            return logits, None

        if labels is None:
            labels = input_ids
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_types = type_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=self.cfg.pad_token_id,
        )
        if not return_per_type_loss:
            return logits, loss

        per_type = {}
        with torch.no_grad():
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.reshape(-1)
            flat_types = shift_types.reshape(-1)
            valid = flat_labels != self.cfg.pad_token_id
            ce_elem = F.cross_entropy(flat_logits, flat_labels, reduction="none",
                                      ignore_index=self.cfg.pad_token_id)
            for t_id, name in [(0, "special"), (1, "coarse"), (2, "fine")]:
                m = (flat_types == t_id) & valid
                n = int(m.sum().item())
                per_type[f"nll_{name}"] = float(ce_elem[m].sum().item()) / max(1, n)
                per_type[f"tokens_{name}"] = n
        return logits, loss, per_type

    def _chunked_head_loss(self, x, input_ids, type_ids, labels,
                           return_per_type_loss: bool):
        import torch.utils.checkpoint as _ckpt
        C = int(self.cfg.loss_head_chunk)
        if labels is None:
            labels = input_ids
        h = self.norm(x)
        sh = h[:, :-1].reshape(-1, h.size(-1))
        sl = labels[:, 1:].reshape(-1)
        st = type_ids[:, 1:].reshape(-1)
        pad = self.cfg.pad_token_id
        total = sh.shape[0]

        losses = []
        n_valid = 0
        for s in range(0, total, C):
            hc = sh[s:s + C]
            lc = sl[s:s + C]

            def _fn(hc_, lc_=lc):
                return F.cross_entropy(self.lm_head(hc_), lc_,
                                       ignore_index=pad, reduction="sum")

            if torch.is_grad_enabled() and hc.requires_grad:
                lo = _ckpt.checkpoint(_fn, hc, use_reentrant=False)
            else:
                lo = _fn(hc)
            losses.append(lo)
            n_valid += int((lc != pad).sum())
        loss = torch.stack(losses).sum() / max(1, n_valid)

        if not return_per_type_loss:
            return None, loss

        per_type = {}
        with torch.no_grad():
            sums = {0: 0.0, 1: 0.0, 2: 0.0}
            ns = {0: 0, 1: 0, 2: 0}
            for s in range(0, total, C):
                hc = sh[s:s + C]
                lc = sl[s:s + C]
                tc = st[s:s + C]
                ce = F.cross_entropy(self.lm_head(hc), lc,
                                     reduction="none", ignore_index=pad)
                valid = lc != pad
                for t in (0, 1, 2):
                    m = (tc == t) & valid
                    sums[t] += float(ce[m].sum().item())
                    ns[t] += int(m.sum().item())
            for t, name in [(0, "special"), (1, "coarse"), (2, "fine")]:
                per_type[f"nll_{name}"] = sums[t] / max(1, ns[t])
                per_type[f"tokens_{name}"] = ns[t]
        return None, loss, per_type

    @torch.no_grad()
    def generate(self, input_ids, source_pos, type_ids,
                 max_new: int, next_source_pos_fn, next_type_id: int,
                 temperature: float = 0.9, top_k: int = 0,
                 top_p: float = 0.0, repetition_penalty: float = 1.0,
                 rep_pen_vocab_limit: Optional[int] = None,
                 eos_token: Optional[int] = None):
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]
        kv_caches = [{} for _ in range(self.cfg.n_layers)]

        logits, _ = self.forward(input_ids, source_pos, type_ids,
                                 kv_caches=kv_caches, start_pos=0)
        cur_pos = input_ids.shape[1]
        all_ids = [input_ids]; all_src = [source_pos]; all_typ = [type_ids]
        next_logits = logits[:, -1, :]
        use_rep = repetition_penalty is not None and repetition_penalty != 1.0
        if use_rep:
            V = next_logits.size(-1)
            emitted = torch.zeros(B, V, dtype=torch.bool, device=device)

        for step in range(max_new):
            if use_rep and emitted.any():
                pen = torch.where(next_logits > 0,
                                  next_logits / repetition_penalty,
                                  next_logits * repetition_penalty)
                mask = emitted
                if rep_pen_vocab_limit is not None:
                    mask = mask.clone()
                    mask[:, rep_pen_vocab_limit:] = False
                next_logits = torch.where(mask, pen, next_logits)
            next_logits = next_logits / max(temperature, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = torch.where(next_logits < v[:, -1:],
                                          torch.full_like(next_logits, -float("inf")),
                                          next_logits)
            if top_p and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                next_logits = torch.full_like(next_logits, -float("inf")).scatter(
                    1, sorted_idx, sorted_logits)
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            if use_rep:
                emitted.scatter_(1, next_tok, True)

            sp_val = next_source_pos_fn(step)
            next_src = torch.full((B, 1), int(sp_val), dtype=source_pos.dtype, device=device)
            next_typ = torch.full((B, 1), int(next_type_id), dtype=type_ids.dtype, device=device)

            all_ids.append(next_tok); all_src.append(next_src); all_typ.append(next_typ)

            if eos_token is not None and (next_tok == eos_token).all():
                break
            if step == max_new - 1:
                break

            logits, _ = self.forward(next_tok, next_src, next_typ,
                                     kv_caches=kv_caches, start_pos=cur_pos)
            cur_pos += 1
            next_logits = logits[:, -1, :]

        return (torch.cat(all_ids, dim=1),
                torch.cat(all_src, dim=1),
                torch.cat(all_typ, dim=1))
