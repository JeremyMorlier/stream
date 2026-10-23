"""Joint hardware/schedule optimization over the cost matrix of a schedule search, with OR-Tools CP-SAT.

The rolled search decides the accelerator in two separate steps: a branch and bound over which design each layer's
core gets (`ScheduleExplorationStage`), then a heuristic folding of the resulting schedules onto fewer cores
(`ScheduleRollingStage`). The stages here replace both with one constraint model that decides, at once:

- how many physical cores there are (slots `p`, used or not),
- which design each core gets (`y[p, d]`),
- which layer groups -- the cores of the unrolled topology, one per `(layer, inter-core group)` -- share a core
  (`x[g, p]`),
- and when each piece of work runs,

and returns a (latency, energy, area) Pareto front by an epsilon-constraint sweep. Every distinct solution is then
re-measured with the real scheduler and inserted into the same `ParetoArchive` as the other schedule search
stages, so `ScheduleSelectionStage` and the visualization stages work unchanged. The model's own estimate of each
point is kept next to the measurement (`predicted_latency`, `predicted_energy`), as the evidence for how much the
model can be trusted.

Three levels of detail, as three stages (see `docs/source/rolled_schedule_optimization.md`):

- `GroupScheduleOptimizationStage` (level 1): one task per layer group. Small and fast, but a consumer group only
  starts once its producer groups have finished, so layer fusion -- which Stream exploits by interleaving tiles of
  consecutive layers -- is invisible and latency is over-estimated.
- `ChunkScheduleOptimizationStage` (level 2): each group's tiles are split, in tile order, into
  `chunks_per_group` chunks. A chunk depends on the chunks that hold its tiles' predecessors, so consecutive layers
  pipeline at chunk granularity and fusion is approximated. The hardware decisions are the same; only the time
  model is finer.
- `TileScheduleOptimizationStage` (level 3): one task per tile, with the tile dependency graph (transitively
  reduced) as precedences -- the time model Stream itself schedules, minus what is listed below.

What the model takes from the schedule space (all computed by the full-matrix warm-up the stages request):
per-tile latency (`latency_attr`) and energy for every (tile shape, design) pair, design areas, and the tile
dependency graph. What it leaves out, and leaves to the re-measurement:

- **transfers between cores**, with `link_transfers="none"`: every link is as wide as the schedule space's
  traffic-derived width, and transfers take no time. With `"timed"` (the default), the width of each link is a
  decision among the interconnect model's `width_levels` -- wider costs router and wire area -- and every
  transfer between two groups on different cores is an interval of `ceil(bits / ceil(width / splits))` cycles
  that starts after its producer task and ends before its consumer starts, fetched just in time as Stream does,
  and overlapping compute on both cores, since a link moves data as a DMA does (`_add_link_transfers`). The
  measurement times links at the chosen widths (`InterconnectAreaModel.link_timing="width"`; with `"traffic"`
  every link is as wide as its core pair's whole traffic, and transfers take about a cycle). Contention between
  group pairs that end up on one physical link is left to the measurement. Transfer energy is zero in the
  accelerator templates (`unit_energy_cost: 0`).
- **off-chip traffic and memory capacity** (weight loads over the shared bus, eviction when a core hosting several
  layers runs out of memory). These are the extensions of section 5 of the note above; they are what most often
  makes a measured point slower than predicted. The bus can be modelled (`offchip_bus`). Every tensor Stream loads
  from off-chip (constant operands and graph inputs) crosses the shared bus once per core that uses it:
  - `"aggregate"`: latency is at least their total size over the bus bandwidth. Accurate as a bound, but every
    solution whose compute fits under it is then predicted at the bound, while Stream fetches loads on demand and
    does not overlap them fully with compute. The measured latencies of those tied solutions spread widely (on the
    ResNet bottleneck, 103k to 197k cycles for one predicted 102k), and the ties are broken on energy and area.
  - `"timed"`: each task waits for its own loads, which take the bus one at a time and are fetched just in time,
    as Stream does (`_add_timed_loads`); the aggregate bound is kept as a redundant constraint.
  Bus contention with other transfers, and re-fetches after eviction, are not modelled.

Area is the sum of the chosen designs' areas (offchip excluded as everywhere else in the search) plus the
interconnect's (`ScheduleSpace.area_of`, see `interconnect_area`), and compute energy is exact up to the cost model.
The interconnect follows the placement: a link joins two slots as soon as two groups that exchange data sit on them,
as wide as the widest such pair (`_add_interconnect_area`), each pair's width fixed or chosen (`link_transfers`).
Its routers and wires are exact up to the integer rounding of the wire pitch; the measured area is recomputed
exactly.
"""

import csv
import logging
import math
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import networkx as nx
from ortools.sat.python import cp_model

from stream.hardware.architecture.noc.interconnect_area import (
    ROUTER_EXTRA_PORTS,
    ROUTER_GE_PER_BIT,
    ROUTER_GE_PER_PORT,
    ROUTER_GE_PER_PORT_BIT,
    InterconnectAreaModel,
    fold_link_widths,
)
from stream.stages.generation.schedule_evaluation import ScheduleSearchStage, tile_key
from stream.stages.generation.schedule_space import ScheduleSearchState, ScheduleSpace
from stream.stages.generation.tiled_workload_accelerator_generation import renumber_merged_cores
from stream.stages.stage import StageCallable
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)

# Integer scaling of the objectives for CP-SAT: latency is in whole cycles already; energy (pJ) and area (mm^2) are
# rounded at these resolutions, fine against values of 1e8+ pJ and 1e-1+ mm^2. One area unit is 1e-9 mm^2, so a
# gate equivalent (2e-7 mm^2 in GF22) is a whole number of units.
ENERGY_SCALE = 1.0
AREA_SCALE = 1e9
# Core side lengths in the interconnect wire terms, in units per mm (micrometres).
SIDE_SCALE = 1e3
OBJECTIVES = ("latency", "energy", "area")
# An epsilon grid needs at least its two end points.
MIN_GRID_LEVELS = 2


@dataclass(frozen=True)
class LinkTransfer:
    """The data a consumer task receives from one other group, over the link between the two groups' cores."""

    producer: int  # producer group
    consumer: int  # consumer group
    source: int  # the producer group's last task feeding `target`
    target: int  # consumer task
    bits: int
    splits: int  # inter-core splits of the consumer layer: the link's bandwidth is shared this many ways

    def duration(self, width: int) -> int:
        """Cycles it takes over a link `width` bits wide, at Stream's `ceil(width / splits)` bits per cycle."""
        return math.ceil(self.bits / math.ceil(width / self.splits))


@dataclass
class ModelSolution:
    """One solution of the model: the merge of the unrolled cores and each slot's design, with the model's own
    estimate of its objectives."""

    slot_of_group: dict[int, int]  # unrolled core id -> slot
    design_of_slot: dict[int, str]  # used slot -> design key
    latency: int
    energy: float
    area: float
    # Every tile `(layer id, sub id)` by the model's start time (ties: sub id, then layer id). Used as the measurement's
    # scheduling order with `scheduling_order_source="model"`.
    scheduling_order: tuple[tuple[int, int], ...] = ()
    # Width the model chose for each communicating group pair (unordered), with `link_transfers="timed"`; None when
    # the widths are the traffic-derived ones of the schedule space.
    group_link_widths: dict[tuple[int, int], int] | None = None

    @property
    def rolled_link_widths(self) -> dict[tuple[int, int], int] | None:
        """The chosen widths of the rolled accelerator's links (unordered rolled core pair -> bits): each as wide as
        the widest group pair it carries, as the model's area has it."""
        if self.group_link_widths is None:
            return None
        return fold_link_widths(self.group_link_widths, renumber_merged_cores(self.slot_of_group))

    @property
    def key(self) -> tuple:
        """The accelerator it describes, independent of how the slots are numbered. The order is not part of it: of
        two solutions with the same accelerator, the first found (the anchors come first) keeps its order."""
        rolled_of = renumber_merged_cores(self.slot_of_group)
        slot_of_rolled = {rolled_of[group]: slot for group, slot in self.slot_of_group.items()}
        widths = self.rolled_link_widths
        return (
            tuple(sorted(rolled_of.items())),
            tuple(self.design_of_slot[slot_of_rolled[core]] for core in range(len(slot_of_rolled))),
            tuple(sorted(widths.items())) if widths is not None else (),
        )


class ScheduleOptimizationStage(ScheduleSearchStage):
    """Base of the two levels: builds the CP-SAT model at the granularity `chunks_per_group` gives, sweeps the
    epsilon-constraint grid, measures every distinct solution and adds it to the archive. See the module docstring.

    Takes `schedule_search` from `ScheduleExplorationStage` (which only needs to measure enough to build the space,
    e.g. `max_schedule_evaluations=1`) and hands the extended state down to `ScheduleSelectionStage`."""

    requires_full_cost_matrix = True
    level = 0
    default_chunks_per_group = 1

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        schedule_search: ScheduleSearchState,
        optimization_search_dir: str | None = None,
        chunks_per_group: int | None = None,
        area_levels: int = 5,
        energy_levels: int = 3,
        solve_time_limit_s: float = 20.0,
        solver_workers: int = 8,
        max_optimized_evaluations: int = 200,
        fold_allow_sibling_merge: bool = False,
        scheduling_order_source: Literal["generated", "model"] = "generated",
        offchip_bus: Literal["none", "aggregate", "timed"] = "none",
        link_transfers: Literal["none", "timed"] = "timed",
        **kwargs: Any,
    ):
        """
        Args:
            schedule_search: the unrolled search's state; its archive and records are extended in place.
            optimization_search_dir: where this stage's report goes. Defaults to `schedule_optimization_l<level>/`
                next to the tiled workload.
            chunks_per_group: tasks per layer group (level 2); level 1 always uses 1.
            area_levels, energy_levels: the epsilon-constraint grid, between each objective's minimum and its
                value at the latency-optimal point.
            solve_time_limit_s: CP-SAT time limit per solve (half of it for the tie-break solves). A solve that
                times out still contributes its best solution found.
            solver_workers: CP-SAT search workers.
            max_optimized_evaluations: cap on solutions measured with the real scheduler.
            fold_allow_sibling_merge: let two groups of the same layer share a core. Off by default, as for the
                rolled search: those are the layer's inter-core slices, and the cost model bakes in that they run
                in parallel.
            scheduling_order_source: the tile order each solution is measured in. `"generated"` is Stream's
                generic order (`SchedulingOrderGenerationStage`); `"model"` is the order the model scheduled the
                tiles in, handed to the measurement as `provided_scheduling_order` -- which only takes effect when
                the measurement stages include `ProvidedSchedulingOrderStage`.
            offchip_bus: how the off-chip loads over the shared bus enter the model's latency: `"none"`,
                `"aggregate"` (a lower bound) or `"timed"` (a load interval before each task); see the module
                docstring.
            link_transfers: how core-to-core transfers enter the model: `"none"` (free, every link as wide as the
                schedule space's traffic-derived width) or `"timed"` (each link's width is a decision, and every
                transfer an interval on its link that overlaps compute); see the module docstring. `"timed"` only
                matches the measurement when the interconnect model times links at their width
                (`link_timing="width"`, the default).
        """
        super().__init__(list_of_callables, **kwargs)
        self.schedule_search = schedule_search
        self.space: ScheduleSpace = schedule_search.space
        self.out_dir = optimization_search_dir or os.path.join(
            os.path.dirname(self.space.tiled_workload_path), f"schedule_optimization_l{self.level}"
        )
        self.chunks_per_group = chunks_per_group or self.default_chunks_per_group
        self.area_levels = area_levels
        self.energy_levels = energy_levels
        self.solve_time_limit_s = solve_time_limit_s
        self.solver_workers = solver_workers
        self.max_optimized_evaluations = max_optimized_evaluations
        self.fold_allow_sibling_merge = fold_allow_sibling_merge
        if scheduling_order_source not in ("generated", "model"):
            raise ValueError(f"Unknown scheduling_order_source {scheduling_order_source!r}.")
        self.scheduling_order_source = scheduling_order_source
        if offchip_bus not in ("none", "aggregate", "timed"):
            raise ValueError(f"Unknown offchip_bus {offchip_bus!r}.")
        self.offchip_bus = offchip_bus
        if link_transfers not in ("none", "timed"):
            raise ValueError(f"Unknown link_transfers {link_transfers!r}.")
        self.link_transfers = link_transfers
        self.search_budget = kwargs.get("search_budget")
        self.solve_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------------------- problem data ------

    def _build_problem(self) -> None:  # noqa: PLR0915
        """Everything the model needs, from the schedule space: groups, chunks, per-design durations and energies,
        design areas and the chunk dependency graph."""
        space = self.space
        self.groups: list[int] = list(space.unrolled_core_ids)
        self.layer_of_group: dict[int, int] = {
            core: node_id for node_id, cores in space.core_ids_of_node.items() for core in cores
        }
        needed_shapes = sorted(set(space.shape_class_of_tile.values()))
        self.designs: dict[str, dict[str, Any]] = {
            key: core_dict
            for key, core_dict in space.distinct_designs().items()
            if all((shape, key) in space.costs.runtime for shape in needed_shapes)
        }
        if not self.designs:
            raise ValueError(f"{type(self).__name__}: no design is cost-modelled on every tile shape.")
        self.design_keys: list[str] = sorted(self.designs)
        self.area_of: dict[str, float] = {key: space.costs.area[key] for key in self.design_keys}
        self.interconnect_model: InterconnectAreaModel = space.interconnect_model
        # Unordered group pair -> width of the link it needs when its groups sit on different cores.
        self.group_link_widths: dict[tuple[int, int], int] = space.link_widths() or {}

        # Each group's tiles in tile order, split into contiguous chunks.
        tiles = [tile for tile in space.reference_workload.node_list if isinstance(tile, ComputationNode)]
        tiles_of_group: dict[int, list[Any]] = {group: [] for group in self.groups}
        for tile in tiles:
            tiles_of_group[space.unrolled_core_of_tile[tile_key(tile)]].append(tile)
        self.chunks: list[tuple[int, int]] = []  # (group, index)
        self.chunk_of_tile: dict[tuple[int, int], int] = {}
        chunk_tiles: list[list[Any]] = []
        self.chunk_tiles = chunk_tiles  # tiles of each task, in sub id order
        for group in self.groups:
            ordered = sorted(tiles_of_group[group], key=lambda tile: tile.sub_id)
            nb_chunks = max(1, min(self.chunks_per_group, len(ordered)))
            size = math.ceil(len(ordered) / nb_chunks)
            for index in range(nb_chunks):
                members = ordered[index * size : (index + 1) * size]
                if not members:
                    continue
                for tile in members:
                    self.chunk_of_tile[tile_key(tile)] = len(self.chunks)
                self.chunks.append((group, index))
                chunk_tiles.append(members)

        runtime, energy = space.costs.runtime, space.costs.energy
        shape_of = space.shape_class_of_tile
        self.duration: list[dict[str, int]] = [
            {
                key: max(1, math.ceil(sum(runtime[(shape_of[tile_key(tile)], key)] for tile in members)))
                for key in self.design_keys
            }
            for members in chunk_tiles
        ]
        self.group_energy: dict[int, dict[str, int]] = {
            group: {
                key: round(
                    sum(energy[(shape_of[tile_key(tile)], key)] for tile in tiles_of_group[group]) / ENERGY_SCALE
                )
                for key in self.design_keys
            }
            for group in self.groups
        }
        self._build_offchip_traffic(tiles)

        # Chunk dependencies from the tile edges. Coarse chunks of two groups can depend on each other both ways
        # when their layers interleave (fusion); such mutually dependent chunks may run in any order relative to
        # each other, so the edges inside a strongly connected component are dropped, except each group's own
        # chunk chain.
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(self.chunks)))
        for producer, consumer in space.reference_workload.edges():
            if not (isinstance(producer, ComputationNode) and isinstance(consumer, ComputationNode)):
                continue
            a, b = self.chunk_of_tile[tile_key(producer)], self.chunk_of_tile[tile_key(consumer)]
            if a != b:
                graph.add_edge(a, b)
        component_of = {
            chunk: index
            for index, component in enumerate(nx.strongly_connected_components(graph))
            for chunk in component
        }
        dependencies = [(a, b) for a, b in graph.edges() if component_of[a] != component_of[b]]
        self.dropped_dependencies = graph.number_of_edges() - len(dependencies)
        self._build_link_traffic(dependencies)
        self.chains = [
            (index, index + 1)
            for index in range(len(self.chunks) - 1)
            if self.chunks[index][0] == self.chunks[index + 1][0]
        ]
        # Transitive reduction: every task lasts at least a cycle, so an edge implied by a path of others constrains
        # nothing. Exact, and what keeps the tile level tractable (attention: ~100k tile edges, most of them
        # implied). The timed transfers are built from the unreduced edges (`_build_link_traffic`), so they keep
        # every delay the reduction would hide.
        dag = nx.DiGraph()
        dag.add_nodes_from(range(len(self.chunks)))
        dag.add_edges_from(dependencies + self.chains)
        chain_set = set(self.chains)
        self.dependencies = [edge for edge in nx.transitive_reduction(dag).edges() if edge not in chain_set]
        logger.info(
            f"{type(self).__name__}: {len(self.groups)} group(s), {len(self.chunks)} task(s), "
            f"{len(self.design_keys)} design(s), {len(self.dependencies)} dependency edge(s) after transitive "
            f"reduction of {len(dependencies)} ({self.dropped_dependencies} dropped inside mutually dependent "
            f"chunk sets)."
        )

    def _build_link_traffic(self, dependencies: list[tuple[int, int]]) -> None:
        """The core-to-core transfers of the timed link model, as `self.transfers`: one per (producer group,
        consumer task) pair the kept chunk dependencies `dependencies` join across two groups, carrying the bits of
        every tile edge between them. It can leave once the producer group's last task feeding the consumer has
        ended (`source`; a group's tasks run in chain order). Dependencies dropped inside mutually dependent chunk
        sets get no transfer, as they get no precedence; nor do pairs without a link in the topology (traffic inside
        one layer)."""
        space = self.space
        dependency_set = set(dependencies)
        bits_of_edge: dict[tuple[int, int], int] = {}
        for producer, consumer, data in space.reference_workload.edges(data=True):
            if not (isinstance(producer, ComputationNode) and isinstance(consumer, ComputationNode)):
                continue
            edge = (self.chunk_of_tile[tile_key(producer)], self.chunk_of_tile[tile_key(consumer)])
            bits = data.get("bits", 0)
            if bits > 0 and edge in dependency_set:
                bits_of_edge[edge] = bits_of_edge.get(edge, 0) + bits

        merged: dict[tuple[int, int], list[int]] = {}  # (producer group, consumer task) -> [source, bits]
        unlinked_bits = 0
        for (a, b), bits in bits_of_edge.items():
            g, h = self.chunks[a][0], self.chunks[b][0]
            if g == h:
                continue
            if (min(g, h), max(g, h)) not in self.group_link_widths:
                unlinked_bits += bits
                continue
            entry = merged.setdefault((g, b), [a, 0])
            entry[0] = max(entry[0], a)
            entry[1] += bits
        # Stream gives each of a node's transfers `1 / inter-core splits` of a link's bandwidth
        # (`CoalaScheduler.get_transfer_bandwidth_fraction`), so that many share a link at once.
        self.transfers: list[LinkTransfer] = [
            LinkTransfer(
                producer=g,
                consumer=self.chunks[b][0],
                source=source,
                target=b,
                bits=bits,
                splits=max(1, self.chunk_tiles[b][0].get_total_inter_core_splits()),
            )
            for (g, b), (source, bits) in sorted(merged.items())
        ]
        if unlinked_bits:
            logger.warning(
                f"{type(self).__name__}: {unlinked_bits:.4g} bits between groups without a link are not timed."
            )

    def _build_offchip_traffic(self, tiles: list[ComputationNode]) -> None:
        """The bits every core loads from off-chip, as `self.offchip_bits`: the tensors Stream places off-chip at
        the start (`CoalaScheduler.initialize_offchip_tensors`: constant operands, and the inputs of graph sources),
        each sent once to every core that uses it -- a core keeps a tensor it already holds, recognized by
        `equality_hash` as Stream's memory manager does. Keyed by the groups using a tensor: a core loads the
        tensor once if any of them sits on it."""
        workload = self.space.reference_workload

        def loaded_tensors(tile: ComputationNode) -> list[tuple[int, int]]:
            """(equality hash, bits) of the tensors `tile` reads from off-chip."""
            loaded_ops = tile.constant_operands + tile.partially_constant_operands
            is_source = len(workload.in_edges(tile)) == 0
            return [
                (tensor.equality_hash, tensor.size)
                for op, tensor in tile.operand_tensors.items()
                if op != tile.output_operand and (op in loaded_ops or is_source)
            ]

        size_of: dict[int, int] = {}
        groups_of: dict[int, set[int]] = {}
        for tile in tiles:
            group = self.space.unrolled_core_of_tile[tile_key(tile)]
            for key, size in loaded_tensors(tile):
                size_of[key] = size
                groups_of.setdefault(key, set()).add(group)
        self.offchip_bits: dict[frozenset[int], int] = {}
        for key, users in groups_of.items():
            users_key = frozenset(users)
            self.offchip_bits[users_key] = self.offchip_bits.get(users_key, 0) + size_of[key]
        self.bus_bandwidth = max(1, round(float(self.space.bus_connection["bandwidth"])))

        # For the timed bus: each task loads the tensors its group has not loaded in an earlier task.
        self.load_bits: list[int] = []
        seen_of_group: dict[int, set[int]] = {}
        for (group, _index), members in zip(self.chunks, self.chunk_tiles, strict=True):
            seen = seen_of_group.setdefault(group, set())
            bits = 0
            for tile in members:
                for key, size in loaded_tensors(tile):
                    if key not in seen:
                        seen.add(key)
                        bits += size
            self.load_bits.append(bits)

    def _add_bus_bound(self, model: cp_model.CpModel) -> tuple[Any, int]:
        """The bus time of the off-chip loads (`_build_offchip_traffic`), as a variable `ceil(bits / bandwidth)`,
        and its upper bound.
        A tensor whose users must sit on different cores (siblings of one layer, unless they may merge) is sent to
        each of them, a constant; otherwise the cores it is sent to follow the placement."""
        constant_bits, max_variable_bits = 0, 0
        variable_bits = []
        for users, bits in self.offchip_bits.items():
            if len(users) == 1 or not self.fold_allow_sibling_merge:
                constant_bits += bits * len(users)
                continue
            max_variable_bits += bits * len(users)
            for p in range(len(self.groups)):
                hosted = [self.x[group, p] for group in users if (group, p) in self.x]
                if hosted:
                    loaded = model.NewBoolVar(f"load_{len(variable_bits)}")
                    model.AddMaxEquality(loaded, hosted)
                    variable_bits.append(bits * loaded)
        bandwidth = self.bus_bandwidth
        max_bus_time = math.ceil((constant_bits + max_variable_bits) / bandwidth)
        bus_time = model.NewIntVar(0, max_bus_time, "bus_time")
        bits = constant_bits + sum(variable_bits)
        model.Add(bus_time * bandwidth >= bits)
        model.Add(bus_time * bandwidth <= bits + bandwidth - 1)
        logger.info(
            f"{type(self).__name__}: {constant_bits:.4g} bits of off-chip loads (+ up to {max_variable_bits:.4g} "
            f"depending on placement) over a {bandwidth} bits/cycle bus: latency >= "
            f"{math.ceil(constant_bits / bandwidth)} cycles."
        )
        return bus_time, max_bus_time

    def _add_timed_loads(self, model: cp_model.CpModel, load_durations: list[int], horizon: int) -> None:
        """One load interval per task that loads anything (`self.load_bits`), at the full bus bandwidth, all on one
        `NoOverlap` for the shared bus. A task starts once its load has ended. As in Stream, which starts a node's
        transfers at most their own duration before its core becomes free (`CoalaScheduler.run`, step 2), a load
        cannot end before the previous task of its group has ended: loads are fetched on demand, not prefetched.
        The first task of a group has no such limit, since which task precedes it on a core depends on the
        placement. A tensor shared by sibling groups merged onto one core is loaded by each of them here."""
        previous_of = {b: a for a, b in self.chains}
        self.load_start: dict[int, Any] = {}
        intervals = []
        for c, duration in enumerate(load_durations):
            if duration == 0:
                continue
            start = model.NewIntVar(0, horizon, f"ls_{c}")
            end = model.NewIntVar(0, horizon, f"le_{c}")
            intervals.append(model.NewIntervalVar(start, duration, end, f"load_{c}"))
            model.Add(self.start[c] >= end)
            if c in previous_of:
                model.Add(end >= self.end[previous_of[c]])
            self.load_start[c] = start
        model.AddNoOverlap(intervals)
        logger.info(
            f"{type(self).__name__}: {len(intervals)} timed load(s) on the bus, {sum(load_durations)} cycles in all."
        )

    def _add_link_widths(self, model: cp_model.CpModel) -> None:
        """One width per communicating group pair, among the interconnect model's `width_levels` (`width_choice`,
        one-hot, and `group_width`). The link between two slots is as wide as the widest pair it carries
        (`_add_interconnect_area`), and the pair's transfers take its width (`_add_link_transfers`): a wider link
        costs router and wire area and saves transfer time. Hinted at the schedule space's traffic-derived
        width."""
        levels = self.interconnect_model.width_levels
        for (g, h), heuristic in self.group_link_widths.items():
            choice = {width: model.NewBoolVar(f"z_{g}_{h}_{width}") for width in levels}
            model.AddExactlyOne(choice.values())
            width_var = model.NewIntVar(min(levels), max(levels), f"gw_{g}_{h}")
            model.Add(width_var == sum(width * literal for width, literal in choice.items()))
            hint = min((width for width in levels if width >= heuristic), default=max(levels))
            for width, literal in choice.items():
                model.AddHint(literal, width == hint)
            self.width_choice[g, h] = choice
            self.group_width[g, h] = width_var

    def _same_slot(self, model: cp_model.CpModel, g: int, h: int) -> Any | None:
        """A literal that may only be true when groups `g` and `h` sit on one slot (None if they never can). Only
        bounded from above: it frees their transfers, so the solver sets it whenever it can."""
        together = []
        for p in range(len(self.groups)):
            if (g, p) in self.x and (h, p) in self.x:
                literal = model.NewBoolVar(f"t_{g}_{h}_{p}")
                model.AddImplication(literal, self.x[g, p])
                model.AddImplication(literal, self.x[h, p])
                together.append(literal)
        if not together:
            return None
        same = model.NewBoolVar(f"same_{g}_{h}")
        model.Add(same <= sum(together))
        return same

    def _add_link_transfers(self, model: cp_model.CpModel, horizon: int) -> None:
        """One interval per transfer (`self.transfers`), off the cores: a link moves data the way a DMA does, so a
        transfer overlaps whatever both cores compute meanwhile. It starts once its source task has ended, its
        consumer task starts once it has ended, and it lasts `ceil(bits / ceil(width / splits))` cycles at its group
        pair's chosen width -- none when both groups share a core. As in Stream, which starts a node's transfers
        at most their own duration before its core becomes free (`CoalaScheduler.run`, step 2), it cannot end
        before the previous task of its consumer's group has ended. The transfers of a directed group pair share
        that link, `splits` at a time (Stream's bandwidth fraction).

        Not modelled: contention between different group pairs that end up on the same physical link once their
        groups are merged onto shared cores; the re-measurement accounts for it."""
        previous_of = {b: a for a, b in self.chains}
        levels = self.interconnect_model.width_levels
        same_of: dict[tuple[int, int], Any] = {}
        intervals_of_pair: dict[tuple[int, int], list[Any]] = {}
        capacity_of_pair: dict[tuple[int, int], int] = {}
        for index, transfer in enumerate(self.transfers):
            g, h = transfer.producer, transfer.consumer
            pair = (min(g, h), max(g, h))
            if pair not in same_of:
                same_of[pair] = self._same_slot(model, *pair)
            same = same_of[pair]
            durations = {width: transfer.duration(width) for width in levels}
            duration = model.NewIntVar(0, max(durations.values()), f"td_{index}")
            separated = sum(durations[width] * literal for width, literal in self.width_choice[pair].items())
            if same is None:
                model.Add(duration == separated)
            else:
                model.Add(duration == separated).OnlyEnforceIf(same.Not())
                model.Add(duration == 0).OnlyEnforceIf(same)
            start = model.NewIntVar(0, horizon, f"ts_{index}")
            end = model.NewIntVar(0, horizon, f"te_{index}")
            interval = model.NewIntervalVar(start, duration, end, f"transfer_{index}")
            model.Add(start >= self.end[transfer.source])
            model.Add(self.start[transfer.target] >= end)
            if transfer.target in previous_of:
                model.Add(end >= self.end[previous_of[transfer.target]])
            intervals_of_pair.setdefault((g, h), []).append(interval)
            capacity_of_pair[g, h] = max(capacity_of_pair.get((g, h), 1), transfer.splits)
        for pair, intervals in intervals_of_pair.items():
            if capacity_of_pair[pair] == 1:
                model.AddNoOverlap(intervals)
            else:
                model.AddCumulative(intervals, [1] * len(intervals), capacity_of_pair[pair])
        total_bits = sum(transfer.bits for transfer in self.transfers)
        logger.info(
            f"{type(self).__name__}: {len(self.transfers)} timed link transfer(s), {total_bits:.4g} bits over "
            f"{len(intervals_of_pair)} directed group pair(s), widths among {levels}."
        )
        if self.level >= 3 and len(self.transfers) > 10_000:  # noqa: PLR2004
            logger.warning(
                f"{type(self).__name__}: {len(self.transfers)} transfer intervals at the tile level; the solver may "
                f"need a much longer `solve_time_limit_s` (or `link_transfers='none'`)."
            )

    # ----------------------------------------------------------------------------------------- model ---------

    def _build_model(self) -> None:  # noqa: PLR0912, PLR0915
        """The CP-SAT model of the module docstring. Slots `0..G-1`; group `i` (in unrolled order) may only use
        slots `<= i` and used slots come first, which removes the relabelling symmetry."""
        model = cp_model.CpModel()
        groups, keys = self.groups, self.design_keys
        nb_slots = len(groups)
        slots = range(nb_slots)
        horizon = sum(max(self.duration[c].values()) for c in range(len(self.chunks)))
        if self.offchip_bus == "timed":
            load_durations = [math.ceil(bits / self.bus_bandwidth) for bits in self.load_bits]
            horizon += sum(load_durations)
        timed_links = self.link_transfers == "timed"
        if timed_links:
            narrowest = min(self.interconnect_model.width_levels)
            horizon += sum(transfer.duration(narrowest) for transfer in self.transfers)

        self.y = {(p, key): model.NewBoolVar(f"y_{p}_{key}") for p in slots for key in keys}
        self.used = [model.NewBoolVar(f"u_{p}") for p in slots]
        self.x: dict[tuple[int, int], Any] = {}
        for index, group in enumerate(groups):
            for p in range(index + 1):
                self.x[group, p] = model.NewBoolVar(f"x_{group}_{p}")
            model.AddExactlyOne(self.x[group, p] for p in range(index + 1))
        for p in slots:
            model.Add(sum(self.y[p, key] for key in keys) == self.used[p])
            hosted = [self.x[group, p] for group in groups if (group, p) in self.x]
            model.Add(sum(hosted) >= self.used[p])
            for literal in hosted:
                model.AddImplication(literal, self.used[p])
            if p + 1 < nb_slots:
                model.AddImplication(self.used[p + 1], self.used[p])
        if not self.fold_allow_sibling_merge:
            by_layer: dict[int, list[int]] = {}
            for group in groups:
                by_layer.setdefault(self.layer_of_group[group], []).append(group)
            for siblings in by_layer.values():
                for a, b in combinations(siblings, 2):
                    for p in slots:
                        if (a, p) in self.x and (b, p) in self.x:
                            model.AddBoolOr([self.x[a, p].Not(), self.x[b, p].Not()])

        # Group energy and chunk durations follow the design of the slot the group sits on.
        max_energy = {group: max(self.group_energy[group].values()) for group in groups}
        self.group_energy_var = {group: model.NewIntVar(0, max_energy[group], f"e_{group}") for group in groups}
        for (group, p), literal in self.x.items():
            model.Add(
                self.group_energy_var[group] == sum(self.group_energy[group][key] * self.y[p, key] for key in keys)
            ).OnlyEnforceIf(literal)

        # A task's duration follows the design of its group's slot. Tasks of one group with the same per-design
        # durations (same tile-shape composition) share one duration variable, so the "duration = design's
        # duration" constraints are written once per (group, composition, slot) rather than per task -- at the tile
        # level that is the difference between a few hundred constraints and hundreds of thousands.
        duration_var: dict[tuple[int, tuple], Any] = {}
        for c, (group, _index) in enumerate(self.chunks):
            durations = self.duration[c]
            signature = (group, tuple(durations[key] for key in keys))
            if signature in duration_var:
                continue
            duration = model.NewIntVar(min(durations.values()), max(durations.values()), f"d_{len(duration_var)}")
            duration_var[signature] = duration
            for p in slots:
                if (group, p) in self.x:
                    model.Add(duration == sum(durations[key] * self.y[p, key] for key in keys)).OnlyEnforceIf(
                        self.x[group, p]
                    )

        self.start, self.end = [], []
        intervals_of_slot: dict[int, list[Any]] = {p: [] for p in slots}
        for c, (group, _index) in enumerate(self.chunks):
            duration = duration_var[(group, tuple(self.duration[c][key] for key in keys))]
            start = model.NewIntVar(0, horizon, f"s_{c}")
            end = model.NewIntVar(0, horizon, f"c_{c}")
            self.start.append(start)
            self.end.append(end)
            model.Add(end == start + duration)
            for p in slots:
                if (group, p) in self.x:
                    intervals_of_slot[p].append(
                        model.NewOptionalIntervalVar(start, duration, end, self.x[group, p], f"i_{c}_{p}")
                    )
        for intervals in intervals_of_slot.values():
            model.AddNoOverlap(intervals)
        for a, b in self.dependencies + self.chains:
            model.Add(self.start[b] >= self.end[a])

        if self.offchip_bus == "timed":
            self._add_timed_loads(model, load_durations, horizon)
        self.width_choice: dict[tuple[int, int], dict[int, Any]] = {}
        self.group_width: dict[tuple[int, int], Any] = {}
        if timed_links:
            self._add_link_widths(model)
            self._add_link_transfers(model, horizon)

        finishes = list(self.end)
        if self.offchip_bus != "none":
            # Implied by the timed loads, but stated: CP-SAT's NoOverlap propagation would take long to prove it.
            bus_time, max_bus_time = self._add_bus_bound(model)
            finishes.append(bus_time)
            horizon = max(horizon, max_bus_time)
        self.latency = model.NewIntVar(0, horizon, "latency")
        model.AddMaxEquality(self.latency, finishes)
        self.energy = sum(self.group_energy_var.values())
        self.scaled_area = {key: round(self.area_of[key] * AREA_SCALE) for key in keys}
        self.area = sum(self.scaled_area[key] * self.y[p, key] for p in slots for key in keys)
        model_of_interconnect = getattr(self, "interconnect_model", None)
        if model_of_interconnect is not None and model_of_interconnect.enabled:
            self.area = self.area + self._add_interconnect_area(model, model_of_interconnect)
        self.model = model
        self.objective_expr = {"latency": self.latency, "energy": self.energy, "area": self.area}

    def _add_interconnect_area(self, model: cp_model.CpModel, ic: InterconnectAreaModel) -> Any:  # noqa: PLR0915
        """The interconnect area (`interconnect_area`) in AREA_SCALE units, as a function of the placement.

        A link joins slots `p < q` (`link[p, q]`) as soon as two groups that exchange data sit on them, and is as
        wide as the widest such pair (`width[p, q]`, the `max` of `fold_link_widths`). Both are only bounded from
        below: area is minimized or bounded above, so neither is ever larger than it has to be at an optimum.
        Per used slot, with `k` its links and `W` their total width, the router of `router_ge` (its links plus
        the bus and core ports) expands to a constant, `k`, `W` and the product `k W`; the link wires are
        `pitch * side * (W + 64 k)` (each link runs half along each end) and the bus wires `pitch * wires(bus) *
        side`, `side` the square root of the slot's design area."""
        slots = range(len(self.groups))
        pairs: dict[tuple[int, int], tuple[Any, Any]] = {}
        max_width = ic.link_width_max
        for (g, h), width in self.group_link_widths.items():
            for p in slots:
                if (g, p) not in self.x:
                    continue
                for q in slots:
                    if q == p or (h, q) not in self.x:
                        continue
                    a, b = min(p, q), max(p, q)
                    if (a, b) not in pairs:
                        link = model.NewBoolVar(f"l_{a}_{b}")
                        link_width = model.NewIntVar(0, max_width, f"w_{a}_{b}")
                        model.Add(link_width <= max_width * link)
                        pairs[a, b] = (link, link_width)
                    link, link_width = pairs[a, b]
                    both = self.x[g, p] + self.x[h, q] - 1
                    model.Add(link >= both)
                    if (g, h) in self.group_width:
                        # The pair's chosen width, when both groups sit on these slots.
                        model.Add(link_width >= self.group_width[g, h] - max_width + max_width * both)
                    else:
                        model.Add(link_width >= width * both)
        self.link_vars = pairs

        ge = round(ic.mm2_per_ge * AREA_SCALE)
        wire = round(ic.mm_per_wire * AREA_SCALE / SIDE_SCALE)
        side_of = {key: round(math.sqrt(max(0.0, self.area_of[key])) * SIDE_SCALE) for key in self.design_keys}
        max_side = max(side_of.values(), default=0)
        extra_width = ROUTER_EXTRA_PORTS * ic.bus_width
        # router_ge(k + extra, W + extra_width), with k = W = 0 on an unused slot.
        per_port = ROUTER_GE_PER_PORT + ROUTER_GE_PER_PORT_BIT * extra_width
        per_bit = ROUTER_GE_PER_PORT_BIT * ROUTER_EXTRA_PORTS + ROUTER_GE_PER_BIT
        per_used = (
            ROUTER_GE_PER_PORT * ROUTER_EXTRA_PORTS
            + ROUTER_GE_PER_PORT_BIT * ROUTER_EXTRA_PORTS * extra_width
            + ROUTER_GE_PER_BIT * extra_width
        )
        half_fixed_wires = round(ic.wires(0) / 2)
        terms = []
        for p in slots:
            incident = [pair for pair in pairs if p in pair]
            side = sum(side_of[key] * self.y[p, key] for key in self.design_keys)
            terms.append(ge * per_used * self.used[p] + wire * round(ic.wires(ic.bus_width)) * side)
            if not incident:
                continue
            nb = len(incident)
            ports = model.NewIntVar(0, nb, f"k_{p}")
            model.Add(ports == sum(pairs[pair][0] for pair in incident))
            width = model.NewIntVar(0, nb * max_width, f"W_{p}")
            model.Add(width == sum(pairs[pair][1] for pair in incident))
            port_width = model.NewIntVar(0, nb * nb * max_width, f"kW_{p}")
            model.AddMultiplicationEquality(port_width, [ports, width])
            # Each link runs half along each end: wires(w) / 2 = w + 64 of them along this slot's side.
            half_wires = model.NewIntVar(0, nb * (max_width + half_fixed_wires), f"hw_{p}")
            model.Add(half_wires == width + half_fixed_wires * ports)
            wire_length = model.NewIntVar(0, max_side * nb * (max_width + half_fixed_wires), f"wl_{p}")
            side_var = model.NewIntVar(0, max_side, f"side_{p}")
            model.Add(side_var == side)
            model.AddMultiplicationEquality(wire_length, [side_var, half_wires])
            terms.append(
                ge * (per_port * ports + ROUTER_GE_PER_PORT_BIT * port_width + per_bit * width) + wire * wire_length
            )
        logger.info(
            f"{type(self).__name__}: interconnect area over {len(pairs)} possible link(s) between "
            f"{len(self.group_link_widths)} communicating group pair(s)."
        )
        return sum(terms)

    # ----------------------------------------------------------------------------------------- solving -------

    def _solve(self, objective: str, bounds: dict[str, int], time_limit: float, label: str) -> ModelSolution | None:
        """Minimize `objective` subject to `objective <= bound` for each entry of `bounds`. The bounds are added as
        temporary constraints (removed after the solve), and the previous solution is kept as a hint."""
        model = self.model
        # Epsilon bounds the incremental way: each bound is a constraint guarded by a fresh literal, and only this
        # solve's literals are assumed true. Bounds of earlier solves stay in the model but their literals are free,
        # so they constrain nothing.
        literals = []
        for name, bound in bounds.items():
            literal = model.NewBoolVar(f"eps_{name}_{len(self.solve_log)}")
            model.Add(self.objective_expr[name] <= bound).OnlyEnforceIf(literal)
            literals.append(literal)
        model.ClearAssumptions()
        model.AddAssumptions(literals)
        model.Minimize(self.objective_expr[objective])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = self.solver_workers
        t0 = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - t0
        entry = {
            "label": label,
            "objective": objective,
            **{f"bound_{name}": bounds.get(name, "") for name in OBJECTIVES},
            "status": solver.StatusName(status),
            "time_s": round(elapsed, 2),
        }
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.solve_log.append(entry)
            return None
        solution = self._extract(solver)
        entry.update(latency=solution.latency, energy=solution.energy, area=solution.area)
        self.solve_log.append(entry)
        model.ClearHints()
        for var in self.x.values():
            model.AddHint(var, solver.Value(var))
        for var in self.y.values():
            model.AddHint(var, solver.Value(var))
        for choice in self.width_choice.values():
            for var in choice.values():
                model.AddHint(var, solver.Value(var))
        return solution

    def _extract(self, solver: cp_model.CpSolver) -> ModelSolution:
        slot_of_group = {group: p for (group, p), var in self.x.items() if solver.Value(var)}
        design_of_slot = {p: key for (p, key), var in self.y.items() if solver.Value(var)}
        # A task's tiles all take its start time and keep their sub id order inside it.
        timed = [
            (solver.Value(self.start[task]), tile.sub_id, tile.id)
            for task, members in enumerate(self.chunk_tiles)
            for tile in members
        ]
        return ModelSolution(
            slot_of_group=slot_of_group,
            design_of_slot=design_of_slot,
            latency=solver.Value(self.latency),
            energy=solver.Value(self.energy) * ENERGY_SCALE,
            area=solver.Value(self.area) / AREA_SCALE,
            scheduling_order=tuple((layer, sub_id) for _start, sub_id, layer in sorted(timed)),
            group_link_widths=(
                {
                    pair: next(width for width, var in choice.items() if solver.Value(var))
                    for pair, choice in self.width_choice.items()
                }
                if self.width_choice
                else None
            ),
        )

    def _scaled(self, solution: ModelSolution, name: str) -> int:
        if name == "latency":
            return solution.latency
        if name == "energy":
            return round(solution.energy / ENERGY_SCALE)
        return round(solution.area * AREA_SCALE)

    def _lexicographic(self, order: tuple[str, ...], bounds: dict[str, int], label: str) -> ModelSolution | None:
        """Minimize the objectives in `order` one after the other, each fixed at its optimum for the next (the
        tie-breaks get half the time limit), so the result is non-dominated within the model."""
        bounds = dict(bounds)
        best = None
        for step, objective in enumerate(order):
            limit = self.solve_time_limit_s if step == 0 else self.solve_time_limit_s / 2
            solution = self._solve(objective, bounds, limit, f"{label}:{objective}")
            if solution is None:
                return best
            best = solution
            bounds[objective] = self._scaled(solution, objective)
        return best

    def _sweep(self) -> list[ModelSolution]:
        """Anchors (each objective minimized first), then latency minimized over an (area, energy) budget grid
        between each objective's minimum and its value at the latency-optimal point."""
        solutions: dict[tuple, ModelSolution] = {}

        def keep(solution: ModelSolution | None) -> None:
            if solution is not None:
                solutions.setdefault(solution.key, solution)

        anchors = {}
        for first in OBJECTIVES:
            rest = tuple(o for o in OBJECTIVES if o != first)
            anchors[first] = self._lexicographic((first, *rest), {}, f"anchor-{first}")
            keep(anchors[first])
        fastest = anchors["latency"]
        if fastest is None:
            return list(solutions.values())

        def grid(name: str, levels: int) -> list[int]:
            low = anchors[name] and self._scaled(anchors[name], name)
            high = self._scaled(fastest, name)
            if low is None or levels < MIN_GRID_LEVELS or high <= low:
                return []
            return sorted({round(low + (high - low) * i / (levels - 1)) for i in range(levels - 1)})

        for area_bound in grid("area", self.area_levels):
            for energy_bound in [None, *grid("energy", self.energy_levels)]:
                bounds = {"area": area_bound}
                if energy_bound is not None:
                    bounds["energy"] = energy_bound
                keep(self._lexicographic(("latency", "energy", "area"), bounds, "grid"))
        return list(solutions.values())

    # ------------------------------------------------------------------------------------- measuring -------

    def _measure_solution(self, solution: ModelSolution, archive: Any, records: list[dict[str, Any]]) -> str:
        space = self.space
        merge_of_core = dict(solution.slot_of_group)
        rolled_of = renumber_merged_cores(merge_of_core)
        slot_of_rolled = {rolled_of[group]: slot for group, slot in merge_of_core.items()}
        nb_cores = len(slot_of_rolled)
        design_key_of_core = {core: solution.design_of_slot[slot_of_rolled[core]] for core in range(nb_cores)}
        core_dict_of_core = {core: self.designs[key] for core, key in design_key_of_core.items()}
        link_widths = solution.rolled_link_widths
        area = space.area_of(design_key_of_core, rolled_of, link_widths)
        record_id = f"opt{self.level}-c{nb_cores}-{len(records)}"
        try:
            workload, accelerator, cost_lut, topology = space.assemble_rolled(
                merge_of_core, core_dict_of_core, design_key_of_core, link_widths=link_widths
            )
            extra = (
                {"provided_scheduling_order": list(solution.scheduling_order)}
                if self.scheduling_order_source == "model"
                else {}
            )
            scme = space.measure(workload, accelerator, cost_lut, **extra)
        except Exception:
            logger.warning(f"{type(self).__name__}: solution {record_id} could not be scheduled.", exc_info=True)
            if self.search_budget is not None:
                self.search_budget.record(math.inf, math.inf, area, info=f"{record_id} infeasible")
            return "infeasible"
        point = (float(scme.latency), float(scme.energy), area)
        if self.search_budget is not None:
            self.search_budget.record(*point, info=record_id)
        record = {
            "id": record_id,
            "kind": "rolled",
            "ranks": (),
            "source_id": f"model-l{self.level}",
            "nb_cores": nb_cores,
            "design_keys": tuple(design_key_of_core[core] for core in range(nb_cores)),
            "latency": point[0],
            "energy": point[1],
            "area": point[2],
            "interconnect_area": space.interconnect_of(design_key_of_core, rolled_of, link_widths).total,
            "link_widths": link_widths,
            "predicted_area": solution.area,
            # The schedule search's CSV calls the analytic estimate a bound; here it is the model's prediction.
            "latency_bound": solution.latency,
            "energy_bound": solution.energy,
            "predicted_latency": solution.latency,
            "predicted_energy": solution.energy,
            "fold_tolerance": "",
            "fold_objective": f"cpsat-level{self.level}",
            "fold_overlap": (
                f"chunks={self.chunks_per_group} order={self.scheduling_order_source} bus={self.offchip_bus} "
                f"links={self.link_transfers}"
            ),
            "merge_of_core": merge_of_core,
            "design_key_of_core": design_key_of_core,
            "core_dict_of_core": core_dict_of_core,
            "members_of_core": dict(topology.members_of_core),
            "hosted_node_groups": dict(topology.hosted_node_groups),
            "node_core_ids": dict(topology.node_core_ids),
            "link_connections": list(topology.link_connections),
            "bus_connection": dict(topology.bus_connection),
        }
        if self.scheduling_order_source == "model":
            record["scheduling_order"] = list(solution.scheduling_order)
        records.append(record)
        archive.add(point, record)
        logger.info(
            f"{type(self).__name__}: measured {record_id}: {nb_cores} core(s), latency={point[0]:.6g} "
            f"(predicted {solution.latency:.6g}), energy={point[1]:.6g} (predicted {solution.energy:.6g}), "
            f"area={area:.6g} (predicted {solution.area:.6g}) (front: {len(archive.entries)})."
        )
        return "measured"

    def _save_solve_log(self) -> None:
        path = os.path.join(self.out_dir, "solves.csv")
        columns = ["label", "objective", "bound_latency", "bound_energy", "bound_area", "status", "time_s",
                   "latency", "energy", "area"]  # fmt: skip
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for entry in self.solve_log:
                writer.writerow({column: entry.get(column, "") for column in columns})

    # ------------------------------------------------------------------------------------------- run ---------

    def run(self):
        state = self.schedule_search
        if not self.space.full_matrix:
            raise ValueError(
                f"{type(self).__name__}: needs the full shape x design cost matrix; raise `max_cme_matrix_cells`."
            )
        os.makedirs(self.out_dir, exist_ok=True)
        if self.link_transfers == "timed" and not self.space.interconnect_model.timed_widths:
            logger.warning(
                f"{type(self).__name__}: links are timed in the model but measured at their traffic "
                f"(`link_timing='traffic'`), where transfers take about a cycle: predictions will be pessimistic."
            )
        t0 = time.perf_counter()
        self._build_problem()
        self._build_model()
        t_model = time.perf_counter() - t0
        solutions = self._sweep()
        t_solve = time.perf_counter() - t0 - t_model
        solutions.sort(key=lambda s: (s.latency, s.energy, s.area))
        statuses: dict[str, int] = {}
        for solution in solutions[: self.max_optimized_evaluations]:
            status = self._measure_solution(solution, state.archive, state.records)
            statuses[status] = statuses.get(status, 0) + 1
        t_total = time.perf_counter() - t0
        self._save_solve_log()
        state.stats[f"optimization_l{self.level}"] = {
            "groups": len(self.groups),
            "tasks": len(self.chunks),
            "designs": len(self.design_keys),
            "dependencies": len(self.dependencies),
            "dropped_dependencies": self.dropped_dependencies,
            "link_transfers": len(self.transfers) if self.link_transfers == "timed" else 0,
            "solves": len(self.solve_log),
            "solutions": len(solutions),
            **statuses,
            "model_time_s": round(t_model, 2),
            "solve_time_s": round(t_solve, 2),
            "measure_time_s": round(t_total - t_model - t_solve, 2),
            "total_time_s": round(t_total, 2),
        }
        logger.info(f"{type(self).__name__}: {state.stats[f'optimization_l{self.level}']}.")
        state.report["out_dir"] = self.out_dir
        kwargs = self.kwargs.copy()
        kwargs["schedule_search"] = state
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()


class GroupScheduleOptimizationStage(ScheduleOptimizationStage):
    """Level 1: one task per layer group (see the module docstring)."""

    level = 1
    default_chunks_per_group = 1

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        kwargs.pop("chunks_per_group", None)
        super().__init__(list_of_callables, chunks_per_group=1, **kwargs)


class ChunkScheduleOptimizationStage(ScheduleOptimizationStage):
    """Level 2: `chunks_per_group` tasks per layer group, with chunk-to-chunk dependencies (see the module
    docstring)."""

    level = 2
    default_chunks_per_group = 8


class TileScheduleOptimizationStage(ScheduleOptimizationStage):
    """Level 3: one task per tile, with the tile dependency graph itself (transitively reduced) as precedences.
    The hardware decisions are those of levels 1 and 2; only the time model is exact up to the per-tile costs.
    Much larger than level 2 (every tile is an optional interval on every slot its group may use), so give it a
    longer `solve_time_limit_s`."""

    level = 3

    def __init__(self, list_of_callables: list[StageCallable], **kwargs: Any):
        kwargs.pop("chunks_per_group", None)
        # More chunks than any group has tiles: every tile becomes its own task.
        super().__init__(list_of_callables, chunks_per_group=10**9, **kwargs)
