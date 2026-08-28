"""Precomputes the entire CACTI memory design space `CoreArchitectureExplorationStage`'s NSGA2 search can ever
query, once, up front -- so the actual search (potentially thousands of GA-candidate core builds) never shells
out to CACTI or touches its on-disk result pool (`zigzag.cacti.cacti_parser.CactiParser`, "CactiBatch" branch,
re-reads and linearly scans a shared YAML pool file on every single lookup, even cache hits).

`MemoryInstance.__init__` (`zigzag.hardware.architecture.memory_instance`) is the sole call site of
`CactiParser.get_item`, and passes only three values that actually vary across Stream's generated cores --
`mem_type`, `size`, `r_bw` (the read port's `bw_max`; `bandwidth_min` never reaches CACTI since both ports get
the same `bandwidth_max`, see `core_generator._build_ports`). Port counts (`r_port=1, w_port=1, rw_port=0`),
`bank=1`, and `technology=0.022` are constants Stream never varies. Combined with `CoreGeneLayout`'s discretized
size grid, that means the entire reachable design space is a few hundred `(mem_type, size, r_bw)` triples --
small enough to precompute exhaustively rather than let the GA discover them lazily.
"""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from zigzag.cacti.cacti_parser import CactiParser

from stream.hardware.architecture.core_generator import _align_to_byte
from stream.stages.generation.core_architecture_exploration import CoreParamRanges, _size_grid_values

logger = logging.getLogger(__name__)

CactiConfig = tuple[str, int, int]
CactiResult = tuple[float, float, float]

# Port counts, bank, and technology Stream always uses -- see module docstring. `get_item` is called with
# exactly these values everywhere `MemoryInstance.__init__` triggers `auto_cost_extraction`.
_R_PORT = 1
_W_PORT = 1
_RW_PORT = 0
_BANK = 1
_TECHNOLOGY = 0.022


def enumerate_cacti_configs(ranges: CoreParamRanges) -> set[CactiConfig]:
    """All `(mem_type, size, r_bw)` triples `CoreGeneLayout.decode_node` can ever hand to CACTI for the given
    `ranges`, exactly as `MemoryInstance.__init__` will pass them (i.e. before `CactiParser.get_item`'s own
    internal "rf" -> "sram" remap). `size` is post-`_align_to_byte`, matching what actually reaches
    `MemoryInstance` (the raw grid value alone isn't the true query key -- alignment can shift it)."""
    configs: set[CactiConfig] = set()

    for grid_size in _size_grid_values(ranges.rf_size_range):
        for bandwidth in ranges.rf_bandwidth_choices:
            configs.add(("rf", _align_to_byte(grid_size, min_size=bandwidth), bandwidth))

    for size_range, bandwidth_choices in (
        (ranges.sram_size_range, ranges.sram_bandwidth_choices),
        (ranges.upper_size_range, ranges.upper_bandwidth_choices),
    ):
        for grid_size in _size_grid_values(size_range):
            for bandwidth_max in bandwidth_choices:
                configs.add(("sram", _align_to_byte(grid_size, min_size=bandwidth_max), bandwidth_max))

    return configs


def _run_cacti_config(config: CactiConfig) -> tuple[CactiConfig, CactiResult]:
    """Runs in a worker process: resolves one `(mem_type, size, r_bw)` config via a real (unpatched at this
    point) `CactiParser.get_item` call. Module-level so it can be pickled and sent to worker processes."""
    mem_type, size, r_bw = config
    result = CactiParser().get_item(
        mem_name=f"precompute_{mem_type}_{size}_{r_bw}",
        mem_type=mem_type,
        size=size,
        r_bw=r_bw,
        r_port=_R_PORT,
        w_port=_W_PORT,
        rw_port=_RW_PORT,
        bank=_BANK,
        technology=_TECHNOLOGY,
    )
    return config, result


def precompute_cacti_design_space(
    ranges: CoreParamRanges, max_workers: int | None = None
) -> dict[CactiConfig, CactiResult]:
    """Resolves every config `enumerate_cacti_configs(ranges)` produces via CACTI, in parallel, and returns the
    resulting `{(mem_type, size, r_bw): (r_cost, w_cost, area)}` lookup. Runs once, before any GA work starts,
    so it can safely use up to `max_workers` (defaults to every available core) rather than the low worker
    counts the nested GA executors use to avoid oversubscription.

    `CactiParser`'s "CactiBatch" pool-file writes are exclusive-lock-protected (see module docstring), so
    dispatching the deduplicated config set concurrently is safe."""
    configs = enumerate_cacti_configs(ranges)
    if not configs:
        return {}

    mp_context = multiprocessing.get_context("fork")
    resolved_workers = max_workers if max_workers is not None else os.process_cpu_count()
    with ProcessPoolExecutor(max_workers=resolved_workers, mp_context=mp_context) as executor:
        results = list(executor.map(_run_cacti_config, configs))
    return dict(results)


# Captured once at import time, before any patching -- `install_cacti_monkeypatch` always wraps this true
# original, regardless of how many times it's called.
_original_get_item = CactiParser.get_item


def install_cacti_monkeypatch(design_space: dict[CactiConfig, CactiResult]) -> None:
    """Replaces `CactiParser.get_item` process-wide with a lookup into `design_space`, so every memory Stream
    builds afterward is served from memory -- no subprocess call, no CACTI pool-file read.

    Must be installed before `optimize_tiling`'s `mainstage.run()` (i.e. before `TilingExplorationStage`'s
    first `fork()`): both it and the nested `CoreArchitectureExplorationStage` use a fork `ProcessPoolExecutor`,
    so a patch applied here is inherited copy-on-write by every descendant worker, with no kwargs/pickling
    needed. Permanent for the process -- re-invoking this (e.g. a second `optimize_tiling(...)` call with
    different `core_param_ranges`) simply replaces the closure with one bound to the new `design_space`.

    A config that isn't in `design_space` (should never happen -- `enumerate_cacti_configs` is meant to cover
    every config `CoreGeneLayout.decode_node` can produce for the same `ranges`) falls back to a real,
    unpatched CACTI call and is cached into `design_space` for any further repeats in that same worker process
    (forked workers each hold a private copy of `design_space`, so this doesn't help sibling workers)."""

    def _patched_get_item(self: CactiParser, **kwargs) -> CactiResult:
        key = (kwargs["mem_type"], kwargs["size"], kwargs["r_bw"])
        cached = design_space.get(key)
        if cached is not None:
            return cached
        logger.warning(
            "cacti_precompute: CACTI design-space miss for %s -- falling back to a live CACTI call. This "
            "should not happen if `enumerate_cacti_configs` was run with the same `core_param_ranges`.",
            key,
        )
        result = _original_get_item(self, **kwargs)
        design_space[key] = result
        return result

    CactiParser.get_item = _patched_get_item
