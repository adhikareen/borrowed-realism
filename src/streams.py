from __future__ import annotations
from dataclasses import dataclass

import numpy as np

LATENT_T = 5
LATENT_H = 16
LATENT_W = 16
DEFAULT_POOL = 2
COARSE_TOKENS = LATENT_T * (LATENT_H // DEFAULT_POOL) * (LATENT_W // DEFAULT_POOL)
FINE_TOKENS = LATENT_T * LATENT_H * LATENT_W
TYPE_SPECIAL = 0
TYPE_COARSE = 1
TYPE_FINE = 2
PAD_SOURCE_POS = 0
BOS_SOURCE_POS = 1
EOS_SOURCE_POS = 2
COARSE_SOURCE_OFFSET = 3
FINE_SOURCE_OFFSET = COARSE_SOURCE_OFFSET + COARSE_TOKENS
SOURCE_POS_VOCAB_SIZE = FINE_SOURCE_OFFSET + FINE_TOKENS


@dataclass(frozen=True)
class Layout:
    pool_size: int = DEFAULT_POOL
    latent_t: int = LATENT_T
    latent_h: int = LATENT_H
    latent_w: int = LATENT_W

    def __post_init__(self):
        if self.latent_h % self.pool_size != 0 or self.latent_w % self.pool_size != 0:
            raise ValueError(
                f"latent ({self.latent_h},{self.latent_w}) not divisible by pool {self.pool_size}"
            )

    @property
    def coarse_h(self) -> int: return self.latent_h // self.pool_size
    @property
    def coarse_w(self) -> int: return self.latent_w // self.pool_size
    @property
    def coarse_tokens(self) -> int: return self.latent_t * self.coarse_h * self.coarse_w
    @property
    def fine_tokens(self) -> int: return self.latent_t * self.latent_h * self.latent_w
    @property
    def coarse_source_offset(self) -> int: return COARSE_SOURCE_OFFSET
    @property
    def fine_source_offset(self) -> int: return self.coarse_source_offset + self.coarse_tokens
    @property
    def source_pos_vocab_size(self) -> int: return self.fine_source_offset + self.fine_tokens

    def k_from_tpf(self, tpf: float, clip_frames: int) -> int:
        total = int(round(tpf * clip_frames))
        return max(0, min(total - self.coarse_tokens, self.fine_tokens))

    def dense_seq_len(self) -> int: return 1 + self.fine_tokens + 1
    def packed_seq_len(self, k: int) -> int: return 1 + self.coarse_tokens + k + 1

    def to_manifest_fields(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "latent_t": self.latent_t,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "coarse_tokens": self.coarse_tokens,
            "fine_tokens": self.fine_tokens,
            "source_pos_vocab_size": self.source_pos_vocab_size,
        }


DEFAULT_LAYOUT = Layout()


@dataclass(frozen=True)
class SpecialIds:
    pad: int
    bos: int
    eos: int


def special_ids(base_vocab_size: int) -> SpecialIds:
    return SpecialIds(pad=base_vocab_size,
                      bos=base_vocab_size + 1,
                      eos=base_vocab_size + 2)


def k_from_tpf(tpf: float, num_frames: int) -> int:
    total = int(round(tpf * num_frames))
    return max(0, min(total - COARSE_TOKENS, FINE_TOKENS))


def dense_seq_len() -> int:
    return 1 + FINE_TOKENS + 1


def packed_seq_len(k: int) -> int:
    return 1 + COARSE_TOKENS + k + 1


def build_dense_sequence(fine_flat: np.ndarray, base_vocab_size: int):
    ids = special_ids(base_vocab_size)
    L = dense_seq_len()
    seq = np.empty(L, dtype=np.int32)
    typ = np.empty(L, dtype=np.uint8)
    pos = np.empty(L, dtype=np.int32)
    seq[0] = ids.bos; seq[-1] = ids.eos
    seq[1:-1] = fine_flat.astype(np.int32, copy=False)
    typ[0] = TYPE_SPECIAL; typ[-1] = TYPE_SPECIAL
    typ[1:-1] = TYPE_FINE
    pos[0] = BOS_SOURCE_POS; pos[-1] = EOS_SOURCE_POS
    pos[1:-1] = FINE_SOURCE_OFFSET + np.arange(FINE_TOKENS, dtype=np.int32)
    return seq, typ, pos


def build_packed_sequence(coarse_flat, fine_flat, selected_pos, base_vocab_size):
    ids = special_ids(base_vocab_size)
    k = int(selected_pos.shape[0])
    L = packed_seq_len(k)
    seq = np.empty(L, dtype=np.int32)
    typ = np.empty(L, dtype=np.uint8)
    pos = np.empty(L, dtype=np.int32)
    seq[0] = ids.bos; seq[-1] = ids.eos
    seq[1:1 + COARSE_TOKENS] = coarse_flat.astype(np.int32, copy=False)
    seq[1 + COARSE_TOKENS:-1] = fine_flat[selected_pos].astype(np.int32, copy=False)
    typ[0] = TYPE_SPECIAL; typ[-1] = TYPE_SPECIAL
    typ[1:1 + COARSE_TOKENS] = TYPE_COARSE
    typ[1 + COARSE_TOKENS:-1] = TYPE_FINE
    pos[0] = BOS_SOURCE_POS; pos[-1] = EOS_SOURCE_POS
    pos[1:1 + COARSE_TOKENS] = COARSE_SOURCE_OFFSET + np.arange(COARSE_TOKENS, dtype=np.int32)
    pos[1 + COARSE_TOKENS:-1] = FINE_SOURCE_OFFSET + selected_pos.astype(np.int32, copy=False)
    return seq, typ, pos


def build_dense_sequence_pool(fine_flat: np.ndarray, base_vocab_size: int, layout: Layout = DEFAULT_LAYOUT):
    ids = special_ids(base_vocab_size)
    L = layout.dense_seq_len()
    fine_n = layout.fine_tokens
    fine_off = layout.fine_source_offset
    seq = np.empty(L, dtype=np.int32)
    typ = np.empty(L, dtype=np.uint8)
    pos = np.empty(L, dtype=np.int32)
    seq[0] = ids.bos; seq[-1] = ids.eos
    seq[1:-1] = fine_flat.astype(np.int32, copy=False)
    typ[0] = TYPE_SPECIAL; typ[-1] = TYPE_SPECIAL
    typ[1:-1] = TYPE_FINE
    pos[0] = BOS_SOURCE_POS; pos[-1] = EOS_SOURCE_POS
    pos[1:-1] = fine_off + np.arange(fine_n, dtype=np.int32)
    return seq, typ, pos


def build_packed_sequence_pool(coarse_flat, fine_flat, selected_pos, base_vocab_size,
                               layout: Layout = DEFAULT_LAYOUT):
    ids = special_ids(base_vocab_size)
    k = int(selected_pos.shape[0])
    coarse_n = layout.coarse_tokens
    fine_off = layout.fine_source_offset
    L = layout.packed_seq_len(k)
    seq = np.empty(L, dtype=np.int32)
    typ = np.empty(L, dtype=np.uint8)
    pos = np.empty(L, dtype=np.int32)
    seq[0] = ids.bos; seq[-1] = ids.eos
    seq[1:1 + coarse_n] = coarse_flat.astype(np.int32, copy=False)
    seq[1 + coarse_n:-1] = fine_flat[selected_pos].astype(np.int32, copy=False)
    typ[0] = TYPE_SPECIAL; typ[-1] = TYPE_SPECIAL
    typ[1:1 + coarse_n] = TYPE_COARSE
    typ[1 + coarse_n:-1] = TYPE_FINE
    pos[0] = BOS_SOURCE_POS; pos[-1] = EOS_SOURCE_POS
    pos[1:1 + coarse_n] = COARSE_SOURCE_OFFSET + np.arange(coarse_n, dtype=np.int32)
    pos[1 + coarse_n:-1] = fine_off + selected_pos.astype(np.int32, copy=False)
    return seq, typ, pos


def assert_packed_manifest_pool(manifest: dict, expected_pool: int, expected_tpf: float | None = None):
    p = int(manifest.get("pool_size", DEFAULT_POOL))
    if p != expected_pool:
        raise ValueError(f"manifest pool_size={p} != expected {expected_pool}")
    latent_t = int(manifest.get("latent_t", manifest.get("latent_frames", 5)))
    latent_h = int(manifest.get("latent_h", 16))
    latent_w = int(manifest.get("latent_w", 16))
    layout = Layout(pool_size=p, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
    expected_coarse = layout.coarse_tokens
    actual_coarse = int(manifest.get("coarse_tokens", -1))
    if actual_coarse != expected_coarse:
        raise ValueError(f"manifest coarse_tokens={actual_coarse} != expected {expected_coarse} for pool={p}")
    actual_fine = int(manifest.get("fine_tokens", -1))
    if actual_fine != layout.fine_tokens:
        raise ValueError(f"manifest fine_tokens={actual_fine} != expected {layout.fine_tokens}")
    if expected_tpf is not None:
        clip_frames = int(manifest.get("clip_frames", 17))
        expected_k = layout.k_from_tpf(expected_tpf, clip_frames)
        actual_k = int(manifest.get("packed_fine_budget", -1))
        if actual_k != expected_k:
            raise ValueError(
                f"manifest packed_fine_budget={actual_k} != expected {expected_k} "
                f"for pool={p} tpf={expected_tpf}"
            )
        expected_seq_len = layout.packed_seq_len(expected_k)
        actual_seq_len = int(manifest.get("seq_len", -1))
        if actual_seq_len != expected_seq_len:
            raise ValueError(
                f"manifest seq_len={actual_seq_len} != expected {expected_seq_len} "
                f"for pool={p} tpf={expected_tpf} K={expected_k}"
            )
