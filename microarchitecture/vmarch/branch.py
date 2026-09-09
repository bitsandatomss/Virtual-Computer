"""Branch prediction zoo + BTB/RAS with a common snapshotable interface.

Predictors: none (always not-taken), bimodal, gshare, tournament, tage_lite.
All update ONLY at commit (non-speculative state); speculative path uses
checkpoints saved at fetch and restored on squash — the standard precise
treatment. Counters are per-PC or indexed by (PC xor history) with tags for
tage_lite. Deterministic; no RNG anywhere.
"""
from __future__ import annotations

import copy


class Predictor:
    name = "base"

    def predict(self, pc: int) -> bool:
        raise NotImplementedError

    def update(self, pc: int, taken: bool) -> None:
        raise NotImplementedError

    def checkpoint(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, ckpt) -> None:
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(ckpt))

    @property
    def predictions(self) -> int:
        return getattr(self, "_pred", 0)

    @property
    def mispredictions(self) -> int:
        return getattr(self, "_mis", 0)

    @property
    def rate(self) -> float:
        return self._mis / self._pred if getattr(self, "_pred", 0) else 0.0


def _sat(counter: int, taken: bool) -> int:
    return min(3, counter + 1) if taken else max(0, counter - 1)


class NonePredictor(Predictor):
    name = "none"

    def __init__(self) -> None:
        self._pred = 0
        self._mis = 0
        self.abstentions = 0

    def predict(self, pc: int) -> bool:
        return False

    def update(self, pc: int, taken: bool) -> None:
        self.abstentions += 1  # records visible resolution without speculation


class Bimodal(Predictor):
    name = "bimodal"

    def __init__(self, entries: int = 512) -> None:
        self.table = [1] * entries
        self.mask = entries - 1
        self._pred = 0
        self._mis = 0

    def predict(self, pc: int) -> bool:
        self._pred += 1
        return self.table[pc & self.mask] >= 2

    def commit(self, pc: int, taken: bool, predicted: bool) -> None:
        if predicted != taken:
            self._mis += 1
        self.table[pc & self.mask] = _sat(self.table[pc & self.mask], taken)

    def update(self, pc: int, taken: bool) -> None:
        self.table[pc & self.mask] = _sat(self.table[pc & self.mask], taken)


class GShare(Predictor):
    name = "gshare"

    def __init__(self, entries: int = 512, hist_bits: int = 8) -> None:
        self.table = [1] * entries
        self.mask = entries - 1
        self.history = 0
        self.hist_mask = (1 << hist_bits) - 1
        self._pred = 0
        self._mis = 0

    def _idx(self, pc: int, hist: int) -> int:
        return (pc ^ hist) & self.mask

    def predict(self, pc: int) -> bool:
        self._pred += 1
        return self.table[self._idx(pc, self.history)] >= 2

    def commit(self, pc: int, taken: bool, predicted: bool, spec_history: int) -> None:
        if predicted != taken:
            self._mis += 1
        self.table[self._idx(pc, spec_history)] = _sat(self.table[self._idx(pc, spec_history)], taken)
        self.history = ((self.history << 1) | int(taken)) & self.hist_mask

    def update(self, pc: int, taken: bool) -> None:
        self.table[self._idx(pc, self.history)] = _sat(self.table[self._idx(pc, self.history)], taken)
        self.history = ((self.history << 1) | int(taken)) & self.hist_mask


class Tournament(Predictor):
    name = "tournament"

    def __init__(self, entries: int = 256, hist_bits: int = 8) -> None:
        self.local = [1] * entries
        self.glob = [1] * entries
        self.choice = [1] * entries
        self.mask = entries - 1
        self.history = 0
        self.hist_mask = (1 << hist_bits) - 1
        self._pred = 0
        self._mis = 0

    def predict(self, pc: int) -> bool:
        self._pred += 1
        i = pc & self.mask
        g = (pc ^ self.history) & self.mask
        use_global = self.choice[i] >= 2
        return (self.glob[g] >= 2) if use_global else (self.local[i] >= 2)

    def commit(self, pc: int, taken: bool, predicted: bool, spec_history: int) -> None:
        if predicted != taken:
            self._mis += 1
        i = pc & self.mask
        g = (pc ^ spec_history) & self.mask
        l_pred = self.local[i] >= 2
        g_pred = self.glob[g] >= 2
        if g_pred != l_pred:  # update chooser only on disagreement
            self.choice[i] = _sat(self.choice[i], g_pred == taken)
        self.local[i] = _sat(self.local[i], taken)
        self.glob[g] = _sat(self.glob[g], taken)
        self.history = ((self.history << 1) | int(taken)) & self.hist_mask

    def update(self, pc: int, taken: bool) -> None:
        i = pc & self.mask
        self.local[i] = _sat(self.local[i], taken)
        self.history = ((self.history << 1) | int(taken)) & self.hist_mask


class TageLite(Predictor):
    """Two tagged tables (T0: short history, T1: long history) + base bimodal.
    A research-honest miniature of TAGE: usefulness counters pick the provider.
    """
    name = "tage_lite"

    def __init__(self, entries: int = 256) -> None:
        self.base = [1] * entries
        self.t0_tag: list[int | None] = [None] * entries
        self.t0_ctr = [0] * entries
        self.t0_use = [0] * entries
        self.t1_tag: list[int | None] = [None] * entries
        self.t1_ctr = [0] * entries
        self.t1_use = [0] * entries
        self.mask = entries - 1
        self.h0 = 0
        self.h1 = 0
        self._pred = 0
        self._mis = 0

    def _idx(self, pc: int, h: int) -> int:
        return (pc ^ (h * 0x9E3779B1)) & self.mask

    def predict(self, pc: int) -> bool:
        self._pred += 1
        i1 = self._idx(pc, self.h1)
        if self.t1_tag[i1] == pc:
            return self.t1_ctr[i1] >= 2
        i0 = self._idx(pc, self.h0)
        if self.t0_tag[i0] == pc:
            return self.t0_ctr[i0] >= 2
        return self.base[pc & self.mask] >= 2

    def commit(self, pc: int, taken: bool, predicted: bool, alt_pred: bool) -> None:
        if predicted != taken:
            self._mis += 1
        i1 = self._idx(pc, self.h1)
        i0 = self._idx(pc, self.h0)
        if self.t1_tag[i1] == pc:
            self.t1_ctr[i1] = _sat(self.t1_ctr[i1], taken)
            self.t1_use[i1] = min(3, self.t1_use[i1] + (1 if predicted == taken else -1))
        elif self.t0_tag[i0] == pc:
            self.t0_ctr[i0] = _sat(self.t0_ctr[i0], taken)
            self.t0_use[i0] = min(3, self.t0_use[i0] + (1 if predicted == taken else -1))
            if predicted != taken and alt_pred == taken:
                # promote: allocate long-history entry
                self.t1_tag[i1] = pc
                self.t1_ctr[i1] = 2 if taken else 1
                self.t1_use[i1] = 0
        else:
            self.base[pc & self.mask] = _sat(self.base[pc & self.mask], taken)
            if predicted != taken:
                self.t0_tag[i0] = pc
                self.t0_ctr[i0] = 2 if taken else 1
                self.t0_use[i0] = 0
        self.h0 = ((self.h0 << 1) | int(taken)) & 0xFF
        self.h1 = ((self.h1 << 1) | int(taken)) & 0xFFFF


def make_predictor(kind: str) -> Predictor:
    return {"none": NonePredictor, "bimodal": Bimodal, "gshare": GShare,
            "tournament": Tournament, "tage_lite": TageLite}[kind]()


class BTB:
    """Branch target buffer: pc -> target. Invalid entries fall through."""

    def __init__(self, entries: int = 64) -> None:
        self.entries = entries
        self.table: dict[int, int] = {}
        self.order: list[int] = []
        self.hits = 0
        self.misses = 0

    def lookup(self, pc: int) -> int | None:
        tgt = self.table.get(pc)
        if tgt is None:
            self.misses += 1
        else:
            self.hits += 1
        return tgt

    def install(self, pc: int, target: int) -> None:
        if pc not in self.table:
            self.order.append(pc)
            if len(self.order) > self.entries:
                self.table.pop(self.order.pop(0), None)
        self.table[pc] = target
