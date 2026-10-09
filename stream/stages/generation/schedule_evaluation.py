"""Building blocks shared by the schedule searches: measuring an accelerator with the real scheduler, reusing
per-`(tile shape, core design)` cost-model results across accelerators, and the plain-data helpers (Pareto
archive, record identities, per-core busy windows) the searches and their reports agree on.

Nothing here is a search. `ScheduleExplorationStage`, `ScheduleRollingStage` and `GraphEvolutionStage` compose
these pieces rather than inheriting them from one another.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from zigzag.utils import pickle_deepcopy

from stream.stages.stage import MainStage, Stage, StageCallable
from stream.utils import CostModelEvaluationLUT
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

OBJECTIVES = ("latency", "energy", "area")
TILE_KEY_T = tuple[int, int]
INTERVAL_T = tuple[int, int]


# --------------------------------------------------------------------------------------- identities -----------


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """True if `a` is at least as good as `b` on every objective and strictly better on at least one."""
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))


def core_dict_key(core_dict: dict[str, Any]) -> str:
    """Content-addressed identity of a core design, ignoring its name: two front designs that decode to the same
    hardware are the same accelerator, and must not be searched (or cost-modelled) twice."""
    without_name = {key: value for key, value in core_dict.items() if key != "name"}
    return hashlib.sha1(repr(_freeze(without_name)).encode()).hexdigest()[:16]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


# Record fields that exist only while the search runs (big, and not plain data), stripped before anything is
# pickled or written.
TRANSIENT_RECORD_KEYS = ("occupancy",)


def strip_unpicklable(record: dict[str, Any]) -> dict[str, Any]:
    """A record without its transient in-memory fields -- what gets persisted."""
    return {key: value for key, value in record.items() if key not in TRANSIENT_RECORD_KEYS}


def format_ranks(ranks: tuple[int, ...]) -> str:
    """Stable text form of a rank tuple, shared by record ids, `schedules.csv` and topology filenames so a row,
    an id and a figure can be matched by eye."""
    return "-".join(str(rank) for rank in ranks)


def tile_key(tile: ComputationNode) -> TILE_KEY_T:
    """Identity of a tile that survives `pickle_deepcopy`, unlike `id(...)`: every combination is measured on
    its own copy of the workload, but `(id, sub_id)` names the same tile in all of them."""
    return (tile.id, tile.sub_id)


def sort_records(records: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    """Records ordered by one objective, or by `edp` (latency x energy). Stable, so ties keep their order."""
    if objective == "edp":
        return sorted(records, key=lambda record: record["latency"] * record["energy"])
    return sorted(records, key=lambda record: record[objective])


# --------------------------------------------------------------------------------------- Pareto archive -------


class ParetoArchive:
    """Non-dominated set of *measured* schedules, serving as both the search's result and its pruning oracle.

    Only measured points are ever stored, which is what makes the pruning sound: a bound dominated by one of
    these can never beat it, whatever the bounded schedule's true cost turns out to be."""

    def __init__(self) -> None:
        self.entries: list[tuple[tuple[float, float, float], dict[str, Any]]] = []

    def is_dominated(self, point: tuple[float, float, float]) -> bool:
        """Whether some stored point dominates `point`."""
        return any(dominates(stored, point) for stored, _ in self.entries)

    def add(self, point: tuple[float, float, float], payload: dict[str, Any]) -> bool:
        """Insert a measured point, dropping every point it dominates. Returns whether it was kept."""
        if self.is_dominated(point):
            return False
        self.entries = [(p, load) for p, load in self.entries if not dominates(point, p)]
        self.entries.append((point, payload))
        return True

    def sorted_by(self, objective: str) -> list[tuple[tuple[float, float, float], dict[str, Any]]]:
        """Entries ordered by one objective, or by `edp` (latency x energy)."""
        if objective == "edp":
            return sorted(self.entries, key=lambda entry: entry[0][0] * entry[0][1])
        index = OBJECTIVES.index(objective)
        return sorted(self.entries, key=lambda entry: entry[0][index])


# ------------------------------------------------------------------------------------- core occupancy ---------


def coalesce(intervals: list[INTERVAL_T]) -> list[INTERVAL_T]:
    """Sorted, non-overlapping cover of `intervals`. Touching intervals are merged: a core that runs one tile
    straight into the next is busy throughout, and splitting that into two windows would only make the overlap
    arithmetic below more expensive without changing any answer."""
    merged: list[INTERVAL_T] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_overlap(a: INTERVAL_T, b: INTERVAL_T) -> int:
    """Length of the intersection of two half-open intervals (0 when disjoint)."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def union_overlap(a: list[INTERVAL_T], b: list[INTERVAL_T]) -> int:
    """Total time two coalesced busy sets are busy at once, by a two-pointer merge -- linear, not quadratic."""
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        total += interval_overlap(a[i], b[j])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


@dataclass
class CoreOccupancy:
    """When each compute core was actually busy in one measured schedule."""

    intervals: dict[int, list[INTERVAL_T]] = field(default_factory=dict)
    load: dict[int, int] = field(default_factory=dict)
    hull: dict[int, INTERVAL_T] = field(default_factory=dict)
    node_groups: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    makespan: int = 0


def core_occupancy(scme: Any, core_ids: list[int]) -> CoreOccupancy:
    """Per-core busy windows of a measured schedule, read off the scheduled tiles.

    `CoalaScheduler` stamps `start`/`end` on the very node objects the `StreamCostModelEvaluation` holds, so
    this needs nothing the cost model doesn't already produce. Cores in `core_ids` that never ran a tile get an
    empty busy set -- they are free to merge with anything.

    Only *compute* occupancy is visible this way: a core is also held up by blocking offchip transfers, so the
    windows are a floor on how busy it was, not a guarantee."""
    offchip_core_id = getattr(scme.accelerator, "offchip_core_id", None)
    raw: dict[int, list[INTERVAL_T]] = {core_id: [] for core_id in core_ids}
    node_groups: dict[int, set[tuple[int, int]]] = {core_id: set() for core_id in core_ids}
    makespan = 0
    for node in scme.workload.node_list:
        if not isinstance(node, ComputationNode):
            continue
        core_id = node.chosen_core_allocation
        if core_id is None or core_id == offchip_core_id or core_id not in raw:
            continue
        start, end = int(node.start), int(node.end)
        if start < 0 or end < start:
            continue
        raw[core_id].append((start, end))
        node_groups[core_id].add((node.id, node.group))
        makespan = max(makespan, end)

    intervals = {core_id: coalesce(windows) for core_id, windows in raw.items()}
    return CoreOccupancy(
        intervals=intervals,
        load={core_id: sum(end - start for start, end in windows) for core_id, windows in intervals.items()},
        hull={
            core_id: (windows[0][0], windows[-1][1]) if windows else (0, 0) for core_id, windows in intervals.items()
        },
        node_groups=node_groups,
        makespan=makespan,
    )


# ---------------------------------------------------------------------------- cost model and measurement -----


class _UnusedLeafStage(Stage):
    """Placeholder sub-stage for stages instantiated only to call one of their methods (e.g.
    `ZigZagCoreMappingEstimationStage.update_cost_lut`). `Stage.__init__` rejects an empty `list_of_callables`
    on a non-leaf stage, and these are never `run()`."""

    def is_leaf(self) -> bool:
        return True

    def run(self):
        yield from ()


@dataclass
class CostCaches:
    """Cost-model results per `(tile shape class, design key)`, plus each design's area.

    A tile's `CostModelEvaluation` depends only on its own core's design (and the shared offchip core), never on
    the rest of the accelerator, so a result learned on one accelerator is valid on every other one that puts
    the same design under the same tile shape. The searches fill this once and assemble every accelerator's
    cost LUT from it instead of re-running ZigZag."""

    latency_attr: str = "latency_total2"
    cme: dict[tuple[int, str], Any] = field(default_factory=dict)
    runtime: dict[tuple[int, str], float] = field(default_factory=dict)
    energy: dict[tuple[int, str], float] = field(default_factory=dict)
    area: dict[str, float] = field(default_factory=dict)

    def add(self, key: tuple[int, str], cme: Any) -> None:
        """Cache one cost-model result (and its runtime/energy summary) under `(shape class, design key)`."""
        self.cme[key] = cme
        self.runtime[key] = float(getattr(cme, self.latency_attr))
        self.energy[key] = float(cme.energy_total)

    def update(self, other: "CostCaches") -> None:
        """Merge another cache in, e.g. one a forked worker filled."""
        self.cme.update(other.cme)
        self.runtime.update(other.runtime)
        self.energy.update(other.energy)
        self.area.update(other.area)

    def harvest(
        self,
        cost_lut: CostModelEvaluationLUT,
        workload: Any,
        accelerator: Any,
        design_key_of_core: dict[int, str],
        shape_class_of_tile: dict[TILE_KEY_T, int],
    ) -> None:
        """Cache one `(tile shape class, design)` cost model result per pair this accelerator covers. Keyed off
        the *core* each tile sits on rather than its node, so it works unchanged when a core hosts several
        layers (the rolled case) or when every core carries the same design (the full-matrix warm-up)."""
        for tile in workload.node_list:
            if not isinstance(tile, ComputationNode):
                continue
            core_id = tile.chosen_core_allocation
            design_key = design_key_of_core[core_id]
            key = (shape_class_of_tile[tile_key(tile)], design_key)
            if key in self.cme:
                continue
            core = accelerator.get_core(core_id)
            equal_node = cost_lut.get_equal_node(tile)
            if equal_node is None or not cost_lut.has_cme(equal_node, core):
                continue
            self.add(key, cost_lut.get_cme(equal_node, core))
            self.area[design_key] = core.get_area()

    def assemble_cost_lut(
        self,
        workload: Any,
        accelerator: Any,
        design_key_of_core: dict[int, str],
        shape_class_of_tile: dict[TILE_KEY_T, int],
    ) -> CostModelEvaluationLUT:
        """Build one accelerator's LUT by copying cached CMEs onto its cores -- the same `core_id` fix-up
        `ZigZagCoreMappingEstimationStage` applies when it finds an equal-performance entry, minus the ZigZag
        run.

        The LUT must be shaped the way `update_cost_lut` shapes it: **one key per unique node, holding every
        core that node can run on**. Keying by tile instead would give each tile its own entry with a single
        core, because `ComputationNode.__hash__` includes `sub_id` while `has_same_performance` excludes it --
        and `SetFixedAllocationPerformanceStage` looks a node up via `get_equal_node` (the first same-shape
        tile) and then asks for *its own* core, which would be missing as soon as two same-shape tiles sit on
        different cores, i.e. whenever `inter_core_tiling > 1`.

        Every core in `tile.possible_core_allocation` is filled, not just the chosen one: that is the shape
        `GeneticAlgorithmAllocationStage` reads back via `cost_lut.get_cores(...)`, and the entries are copies
        from the cache, so the extra fill is cheap."""
        cost_lut = CostModelEvaluationLUT(None, load=False)
        for tile in workload.node_list:
            if not isinstance(tile, ComputationNode):
                continue
            shape_class = shape_class_of_tile[tile_key(tile)]
            representative = cost_lut.get_equal_node(tile) or tile
            for core_id in dict.fromkeys([tile.chosen_core_allocation, *tile.possible_core_allocation]):
                design_key = design_key_of_core.get(core_id)
                if design_key is None:  # not a compute core of this accelerator (e.g. a stale offchip id)
                    continue
                core = accelerator.get_core(core_id)
                if cost_lut.has_cme(representative, core):
                    continue
                cme = pickle_deepcopy(self.cme[(shape_class, design_key)])
                cme.layer.core_allocation = [core_id]
                cme.core_id = core_id
                cost_lut.add_cme(representative, core, cme)
        return cost_lut


class MeasurementPipeline:
    """The stages that turn an accelerator, a pinned workload and a cost LUT into a schedule (typically
    `SetFixedAllocationPerformanceStage`, `SchedulingOrderGenerationStage`, `GeneticAlgorithmAllocationStage`),
    run on demand. A search holds one so it can measure many accelerators without being their parent stage."""

    def __init__(self, list_of_callables: list[StageCallable], kwargs: dict[str, Any]):
        self.list_of_callables = list(list_of_callables)
        self.kwargs = dict(kwargs)

    def measure(self, workload: Any, accelerator: Any, cost_lut: CostModelEvaluationLUT):
        """Schedule `workload` on `accelerator` with `cost_lut` and return the resulting
        `StreamCostModelEvaluation` (the first answer the stages yield)."""
        kwargs = self.kwargs.copy()
        kwargs["workload"] = workload
        kwargs["accelerator"] = accelerator
        kwargs["cost_lut"] = cost_lut
        scme, _ = MainStage(list(self.list_of_callables), **kwargs).run()[0]
        return scme
