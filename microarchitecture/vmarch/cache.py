"""Set-associative cache hierarchy: L1I/L1D -> L2 -> DRAM row-buffer model.

- LRU replacement within sets, write-back L1D/L2 with dirty tracking,
  write-allocate on stores. Wrong-path fills are allowed (realistic) but
  squashed stores never reach the hierarchy (commit-only writes).
- MSHRs cap outstanding L1D misses; merged MSHR hits counted.
- DRAM: single channel, open-row hit vs conflict/empty latencies.
- Stride-agnostic sequential prefetcher with degree + accuracy accounting.
- All state is plain data (deepcopy-able) for world forking.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

# Audit F6/T3: instruction and data lines share the L2/MSHR/DRAM below the
# split L1s. They MUST NOT share a key space: pc//line_words (0..~50) and
# addr//line_words (0..~256) overlap numerically but are different physical
# regions. Without disambiguation, I-fills masquerade as D-hits (measured:
# 44 phantom L2 hits on stream/irregular) and MSHRs falsely merge across
# spaces. I-lines carry I_BIAS through every shared structure.
I_BIAS = 1 << 20


@dataclass
class CacheSet:
    lines: "OrderedDict[int, bool]" = field(default_factory=OrderedDict)  # tag -> dirty


class SetAssocCache:
    def __init__(self, total_lines: int, assoc: int, line_words: int, name: str = "") -> None:
        assert total_lines % assoc == 0
        self.nsets = total_lines // assoc
        self.assoc = assoc
        self.line_words = line_words
        self.name = name
        self.sets = [CacheSet() for _ in range(self.nsets)]
        self.hits = 0
        self.misses = 0
        self.writebacks = 0
        self.evictions = 0

    def _idx_tag(self, line: int) -> tuple[int, int]:
        return line % self.nsets, line // self.nsets

    def probe(self, line: int) -> bool:
        s, tag = self._idx_tag(line)
        return tag in self.sets[s].lines

    def touch(self, line: int) -> None:
        s, tag = self._idx_tag(line)
        self.sets[s].lines.move_to_end(tag)

    def fill(self, line: int, dirty: bool = False) -> int | None:
        """Insert line; return evicted dirty line number (for writeback) or None."""
        s, tag = self._idx_tag(line)
        st = self.sets[s].lines
        if tag in st:
            st.move_to_end(tag)
            st[tag] = st[tag] or dirty
            return None
        evicted = None
        if len(st) >= self.assoc:
            old_tag, was_dirty = st.popitem(last=False)
            self.evictions += 1
            if was_dirty:
                self.writebacks += 1
                evicted = old_tag * self.nsets + s
        st[tag] = dirty
        return evicted

    def invalidate_all(self) -> None:
        for st in self.sets:
            st.lines.clear()


class DRAM:
    def __init__(self, hit_lat: int = 20, miss_lat: int = 45, row_lines: int = 16) -> None:
        self.hit_lat = hit_lat
        self.miss_lat = miss_lat
        self.row_lines = row_lines
        self.open_row: int | None = None
        self.row_hits = 0
        self.row_misses = 0

    def access(self, line: int) -> int:
        row = line // self.row_lines
        if self.open_row == row:
            self.row_hits += 1
            return self.hit_lat
        self.row_misses += 1
        self.open_row = row
        return self.miss_lat

    @property
    def row_hit_rate(self) -> float:
        t = self.row_hits + self.row_misses
        return self.row_hits / t if t else 0.0


@dataclass
class MemResult:
    latency: int
    energy: float
    l1_hit: bool
    l2_hit: bool
    mshr_merged: bool
    row_hit: bool


class MemoryHierarchy:
    L1_HIT_LAT = 2
    L2_HIT_LAT = 9

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.l1i = SetAssocCache(cfg.l1i_lines, min(cfg.l1_assoc, cfg.l1i_lines), cfg.line_words, "l1i")
        self.l1d = SetAssocCache(cfg.l1d_lines, min(cfg.l1_assoc, cfg.l1d_lines), cfg.line_words, "l1d")
        self.l2 = SetAssocCache(cfg.l2_lines, min(cfg.l2_assoc, cfg.l2_lines), cfg.line_words, "l2")
        self.dram = DRAM()
        self.mshr: dict[int, int] = {}  # line -> cycles remaining
        self.mshr_cap = cfg.mshr_entries
        self.mshr_merges = 0
        self.mshr_stalls = 0
        self.prefetched: set[int] = set()
        self.prefetches_issued = 0
        self.useful_prefetches = 0
        self.accesses = 0
        # Event trace (T3): None = disabled (zero overhead); list = record.
        # Set by Core when cfg.trace is on. cycle is stamped by the core.
        self.trace: list | None = None
        self.cycle: int = 0

    def _log(self, kind: str, line: int, info: str = "") -> None:
        if self.trace is not None:
            self.trace.append((self.cycle, kind, line, info))

    # -- instruction fetch path (lines biased: separate L2/MSHR/DRAM space) --
    def fetch_line(self, line: int) -> tuple[int, bool]:
        line += I_BIAS
        if self.l1i.probe(line):
            self.l1i.hits += 1
            self.l1i.touch(line)
            self._log("i_hit", line)
            return self.L1_HIT_LAT, True
        self.l1i.misses += 1
        lat = self._fill_from_below(line, self.l1i, "i")
        self._log("i_miss", line, f"lat={lat}")
        return lat, False

    # -- data path (called at load-execute / store-commit) --
    def load(self, line: int, prefetch_degree: int, energy_scale: float) -> MemResult:
        from vmarch.config import ENERGY
        self.accesses += 1
        if self.l1d.probe(line):
            self.l1d.hits += 1
            self.l1d.touch(line)
            useful = line in self.prefetched
            if useful:
                self.useful_prefetches += 1
                self.prefetched.discard(line)
            e = ENERGY["l1_hit"] * energy_scale
            e += self._prefetch(line, prefetch_degree)
            self._log("d_hit", line, "useful" if useful else "")
            return MemResult(self.L1_HIT_LAT, e, True, False, False, False)
        if line in self.mshr:
            self.mshr_merges += 1
            e = ENERGY["l1_hit"] * energy_scale
            self._log("mshr_merge", line)
            return MemResult(6, e, False, False, True, False)
        if len(self.mshr) >= self.mshr_cap:
            self.mshr_stalls += 1
            self._log("mshr_stall", line)
            return MemResult(self.L1_HIT_LAT, 0.0, False, False, False, False)  # caller retries
        self.l1d.misses += 1
        lat = self._fill_from_below(line, self.l1d, "d")
        row_hit = self.dram.open_row == line // self.dram.row_lines
        self._log("d_miss", line, f"lat={lat}")
        e = (ENERGY["dram"] if lat > self.L2_HIT_LAT else ENERGY["l2_hit"]) * energy_scale
        e += self._prefetch(line, prefetch_degree)
        return MemResult(lat, e, False, lat <= self.L2_HIT_LAT, False, row_hit)

    def store(self, line: int, energy_scale: float) -> MemResult:
        from vmarch.config import ENERGY
        self.accesses += 1
        if self.l1d.probe(line):
            self.l1d.hits += 1
            self.l1d.touch(line)
            s, tag = self.l1d._idx_tag(line)
            self.l1d.sets[s].lines[tag] = True
            self._log("st_hit", line)
            return MemResult(self.L1_HIT_LAT, ENERGY["l1_hit"] * energy_scale, True, False, False, False)
        self.l1d.misses += 1
        lat = self._fill_from_below(line, self.l1d, "d")  # write-allocate
        self._log("st_miss", line, f"lat={lat}")
        s, tag = self.l1d._idx_tag(line)
        if tag in self.l1d.sets[s].lines:
            self.l1d.sets[s].lines[tag] = True
        return MemResult(lat, ENERGY["dram"] * energy_scale, False, lat <= self.L2_HIT_LAT, False, False)

    def _fill_from_below(self, line: int, upper: SetAssocCache, space: str) -> int:
        if self.l2.probe(line):
            self.l2.hits += 1
            self.l2.touch(line)
            upper.fill(line)
            self._log("l2_hit", line, space)
            return self.L2_HIT_LAT
        self.l2.misses += 1
        lat = self.dram.access(line)
        self._log("dram", line, f"{space} lat={lat} " +
                  ("rowhit" if lat == self.dram.hit_lat else "rowmiss"))
        self.l2.fill(line)
        upper.fill(line)
        return lat

    def _prefetch(self, line: int, degree: int) -> float:
        from vmarch.config import ENERGY
        e = 0.0
        for d in range(1, degree + 1):
            pl = line + d
            if not self.l1d.probe(pl):
                self.prefetches_issued += 1
                self.prefetched.add(pl)
                self.l1d.fill(pl)  # fills from L2/DRAM implicitly; bandwidth cost charged
                self._log("pf_fill", pl)
                e += ENERGY["prefetch"]
        return e

    def tick(self) -> None:
        done = [ln for ln, c in self.mshr.items() if c - 1 <= 0]
        for ln in done:
            del self.mshr[ln]
        for ln in self.mshr:
            self.mshr[ln] -= 1

    @property
    def l1d_miss_rate(self) -> float:
        t = self.l1d.hits + self.l1d.misses
        return self.l1d.misses / t if t else 0.0

    @property
    def l1i_miss_rate(self) -> float:
        t = self.l1i.hits + self.l1i.misses
        return self.l1i.misses / t if t else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        return self.useful_prefetches / self.prefetches_issued if self.prefetches_issued else 0.0
