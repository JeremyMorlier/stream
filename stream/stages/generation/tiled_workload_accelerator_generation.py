import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import yaml
from zigzag.utils import open_yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.core_validator import CoreValidator
from stream.stages.stage import Stage, StageCallable
from stream.utils import get_inter_core_tiling_size
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload, ONNXWorkload

logger = logging.getLogger(__name__)

TPU_CORE_YAML_PATH = "stream/inputs/examples/hardware/cores/tpu_like.yaml"
OFFCHIP_CORE_YAML_PATH = "stream/inputs/examples/hardware/cores/offchip.yaml"
OFFCHIP_BUS_BANDWIDTH = 128.0


def validate_core_yaml(core_yaml_path: str) -> dict[str, Any]:
    validator = CoreValidator(open_yaml(core_yaml_path))
    if not validator.validate():
        raise ValueError(f"Core file {core_yaml_path} failed validation.")
    return validator.normalized_data


def load_offchip_core_data() -> dict[str, Any]:
    return validate_core_yaml(OFFCHIP_CORE_YAML_PATH)


def get_original_nodes(original_workload: ONNXWorkload) -> list[ComputationNode]:
    return [node for node in original_workload.topological_sort() if isinstance(node, ComputationNode)]


def get_tiles_by_original_node(
    workload: ComputationNodeWorkload, original_nodes: list[ComputationNode]
) -> dict[ComputationNode, list[ComputationNode]]:
    tiles = [node for node in workload.node_list if isinstance(node, ComputationNode)]
    return {original_node: [tile for tile in tiles if tile.id == original_node.id] for original_node in original_nodes}


@dataclass
class GroupDedicatedTopology:
    """The shape derived from a tiled workload: one dedicated core id (or block of ids, for inter-core-tiled
    layers) per original workload node, plus one shared offchip core reachable over a bus, and the point-to-
    point links needed between cores that actually exchange data. Shared by `TiledWorkloadAcceleratorGenerationStage`
    (which fills every dedicated core with the same fixed template) and `CoreArchitectureExplorationStage`
    (which fills each original node's dedicated cores with its own searched core design)."""

    node_core_ids: dict[int, list[int]]  # keyed by ORIGINAL node id -- picklable/stable across processes
    offchip_core_id: int
    bus_connection: dict[str, Any]
    link_connections: list[dict[str, Any]]
    original_nodes: list[ComputationNode]


def derive_group_dedicated_topology(
    workload: ComputationNodeWorkload, original_workload: ONNXWorkload
) -> GroupDedicatedTopology:
    """Build one dedicated core (or block of cores, for inter-core-tiled layers) per `(original node, group)`
    pair, i.e. one core per inter-core-tiling slice (tiles sharing a group always share a core, since intra-core
    tiling never changes group), plus one shared offchip core reachable by every dedicated core over a bus (see
    tpu_like_quad_core.yaml). Tile-to-tile edges with bits > 0 become point-to-point links between the
    corresponding cores, bandwidth summed from all tile-edges mapping to that core pair; edges with bits == 0
    (same-core ordering edges) are dropped.

    Since every (node, group) pair now has an exclusive core, each tile's core allocation is pinned directly to
    that core (as a side effect on `workload`) so downstream stages don't try to resolve an allocation against
    the old hardware's core ids."""
    original_nodes = get_original_nodes(original_workload)
    tiles_by_original = get_tiles_by_original_node(workload, original_nodes)
    id_to_original_node = {node.id: node for node in original_nodes}

    node_core_ids: dict[int, list[int]] = {}
    next_core_id = 0
    for original_node in original_nodes:
        k = get_inter_core_tiling_size(original_node)
        node_core_ids[original_node.id] = list(range(next_core_id, next_core_id + k))
        next_core_id += k
    offchip_core_id = next_core_id

    for original_node, tiles in tiles_by_original.items():
        core_ids = node_core_ids[original_node.id]
        for tile in tiles:
            tile.possible_core_allocation = core_ids
            tile.set_chosen_core_allocation(core_ids[tile.group])

    core_pair_bits: dict[tuple[int, int], int] = defaultdict(int)
    for producer_tile, consumer_tile, data in workload.edges(data=True):
        bits = data.get("bits", 0)
        if bits == 0:
            continue
        producer_node = id_to_original_node[producer_tile.id]
        consumer_node = id_to_original_node[consumer_tile.id]
        if producer_node is consumer_node:
            continue
        core_a = node_core_ids[producer_node.id][producer_tile.group]
        core_b = node_core_ids[consumer_node.id][consumer_tile.group]
        core_pair_bits[(core_a, core_b)] += bits

    bus_connection = {
        "type": "bus",
        "cores": list(range(offchip_core_id + 1)),
        "bandwidth": OFFCHIP_BUS_BANDWIDTH,
    }
    link_connections = [
        {"type": "link", "cores": [core_a, core_b], "bandwidth": bits}
        for (core_a, core_b), bits in core_pair_bits.items()
    ]
    logger.info(
        f"Derived group-dedicated topology: {len(original_nodes)} original nodes, "
        f"{offchip_core_id} dedicated cores, 1 offchip core, {len(link_connections)} core-to-core links."
    )

    return GroupDedicatedTopology(
        node_core_ids=node_core_ids,
        offchip_core_id=offchip_core_id,
        bus_connection=bus_connection,
        link_connections=link_connections,
        original_nodes=original_nodes,
    )


@dataclass
class RolledTopology:
    """A `GroupDedicatedTopology` folded onto fewer physical cores: several `(original node, group)` pairs may
    now share one core, so the core count no longer grows with the workload.

    Carries the same fields the unrolled topology does -- so `AcceleratorFactory` input is built the same way --
    plus the provenance a caller needs to explain and draw the fold: which unrolled cores went into each rolled
    core, which `(node, group)` pairs it hosts, and the re-derived core-to-core traffic."""

    node_core_ids: dict[int, list[int]]  # original node id -> rolled core id per group (length unchanged)
    offchip_core_id: int
    bus_connection: dict[str, Any]
    link_connections: list[dict[str, Any]]
    original_nodes: list[ComputationNode]
    members_of_core: dict[int, list[int]]
    hosted_node_groups: dict[int, list[tuple[int, int]]]
    core_pair_bits: dict[tuple[int, int], int]

    @property
    def nb_compute_cores(self) -> int:
        return self.offchip_core_id


def renumber_merged_cores(merge_of_core: dict[int, Any]) -> dict[int, int]:
    """Map each unrolled compute core to its rolled core id.

    Dense from 0 and in first-appearance order over the sorted unrolled ids -- both are required, not
    cosmetic: `AcceleratorFactory.create_core_graph` asserts every core's id equals its index, and the
    deterministic order keeps a fold's ids stable across runs."""
    new_id_of_label: dict[Any, int] = {}
    for core_id in sorted(merge_of_core):
        new_id_of_label.setdefault(merge_of_core[core_id], len(new_id_of_label))
    return {core_id: new_id_of_label[merge_of_core[core_id]] for core_id in sorted(merge_of_core)}


def fold_core_pair_bits(
    unrolled_pair_bits: dict[tuple[int, int], int], rolled_of: dict[int, int]
) -> dict[tuple[int, int], int]:
    """Re-derive core-to-core traffic after a fold.

    Traffic whose two ends landed on the same rolled core is now intra-core and needs no link at all. What
    remains is summed per *unordered* pair, so traffic that used to run over several separate links coalesces
    onto the one link between the merged groups -- and so that the two directions of a pair cannot end up as
    two `"link"` entries, where the second would silently replace the first in the core graph."""
    folded: dict[tuple[int, int], int] = defaultdict(int)
    for (core_a, core_b), bits in unrolled_pair_bits.items():
        rolled_a, rolled_b = rolled_of[core_a], rolled_of[core_b]
        if rolled_a == rolled_b:
            continue
        folded[(min(rolled_a, rolled_b), max(rolled_a, rolled_b))] += bits
    return dict(folded)


def derive_rolled_topology(
    workload: ComputationNodeWorkload,
    original_workload: ONNXWorkload,
    merge_of_core: dict[int, Any],
) -> RolledTopology:
    """Fold the group-dedicated topology by merging every unrolled compute core that shares a label in
    `merge_of_core` (a graph colouring of cores whose busy windows don't collide -- see
    `RolledScheduleExplorationStage`) into one physical core.

    Three things have to be re-derived rather than carried over:

    1. **Core ids.** `AcceleratorFactory.create_core_graph` asserts `core.id == index`, and `create` walks the
       `cores` dict in insertion order, so the merged cores are renumbered densely from 0 in first-appearance
       order over the sorted unrolled ids (deterministic), with offchip appended last.
    2. **Tile pinning.** Every tile is re-pinned to its rolled core, the same side effect
       `derive_group_dedicated_topology` has, so no downstream stage resolves an allocation against the stale
       unrolled ids.
    3. **Links.** Traffic between two cores that ended up merged is now intra-core and free, so those links
       disappear; conversely a rolled core inherits the union of its members' neighbours, so pairs that were
       never adjacent become adjacent and their bits *coalesce onto one link*. That is the connection-adding
       half of rolling, and it matters: a missing point-to-point link does not fail, it silently falls back to
       the shared offchip bus (`Accelerator.find_earliest_time_for_transfer` takes the first shortest path),
       which is orders of magnitude narrower.

    Bits are summed per unordered core pair, matching `derive_group_dedicated_topology`'s summing convention.
    Folding makes the two directions of a pair far more likely to both carry traffic, and one `"link"` entry
    per direction would have the second silently replace the first in the core `DiGraph` -- `AcceleratorFactory`
    already builds both directions from a single entry.
    """
    unrolled = derive_group_dedicated_topology(workload, original_workload)
    original_nodes = unrolled.original_nodes
    tiles_by_original = get_tiles_by_original_node(workload, original_nodes)
    id_to_original_node = {node.id: node for node in original_nodes}

    compute_core_ids = sorted(merge_of_core)
    if compute_core_ids != list(range(unrolled.offchip_core_id)):
        raise ValueError(
            f"derive_rolled_topology: `merge_of_core` must label every compute core exactly once "
            f"({list(range(unrolled.offchip_core_id))}), got {compute_core_ids}."
        )

    rolled_of = renumber_merged_cores(merge_of_core)
    offchip_core_id = len(set(rolled_of.values()))

    members_of_core: dict[int, list[int]] = {rolled_id: [] for rolled_id in range(offchip_core_id)}
    for core_id in compute_core_ids:
        members_of_core[rolled_of[core_id]].append(core_id)

    node_core_ids = {
        node_id: [rolled_of[core_id] for core_id in core_ids] for node_id, core_ids in unrolled.node_core_ids.items()
    }
    hosted_node_groups: dict[int, list[tuple[int, int]]] = {rolled_id: [] for rolled_id in range(offchip_core_id)}
    for node_id, core_ids in unrolled.node_core_ids.items():
        for group, unrolled_id in enumerate(core_ids):
            hosted_node_groups[rolled_of[unrolled_id]].append((node_id, group))

    for original_node, tiles in tiles_by_original.items():
        core_ids = node_core_ids[original_node.id]
        for tile in tiles:
            tile.possible_core_allocation = core_ids
            tile.set_chosen_core_allocation(core_ids[tile.group])

    unrolled_pair_bits: dict[tuple[int, int], int] = defaultdict(int)
    for producer_tile, consumer_tile, data in workload.edges(data=True):
        bits = data.get("bits", 0)
        if bits == 0:
            continue
        producer_node = id_to_original_node[producer_tile.id]
        consumer_node = id_to_original_node[consumer_tile.id]
        if producer_node is consumer_node:
            continue
        core_a = unrolled.node_core_ids[producer_node.id][producer_tile.group]
        core_b = unrolled.node_core_ids[consumer_node.id][consumer_tile.group]
        unrolled_pair_bits[(core_a, core_b)] += bits

    core_pair_bits = fold_core_pair_bits(dict(unrolled_pair_bits), rolled_of)
    bus_connection = {
        "type": "bus",
        "cores": list(range(offchip_core_id + 1)),
        "bandwidth": OFFCHIP_BUS_BANDWIDTH,
    }
    link_connections = [
        {"type": "link", "cores": [core_a, core_b], "bandwidth": bits}
        for (core_a, core_b), bits in core_pair_bits.items()
    ]
    logger.info(
        f"Derived rolled topology: {unrolled.offchip_core_id} -> {offchip_core_id} compute core(s), "
        f"{len(unrolled.link_connections)} -> {len(link_connections)} core-to-core link(s)."
    )

    return RolledTopology(
        node_core_ids=node_core_ids,
        offchip_core_id=offchip_core_id,
        bus_connection=bus_connection,
        link_connections=link_connections,
        original_nodes=original_nodes,
        members_of_core=members_of_core,
        hosted_node_groups=hosted_node_groups,
        core_pair_bits=dict(core_pair_bits),
    )


class TiledWorkloadAcceleratorGenerationStage(Stage):
    """
    Builds an accelerator whose cores are dedicated to the tiled workload groups.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload: ComputationNodeWorkload,
        original_workload: ONNXWorkload,
        accelerator: Accelerator,
        tiled_workload_path: str,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.original_workload = original_workload
        self.accelerator = accelerator
        self.tiled_workload_path = tiled_workload_path

    def run(self):
        kwargs = self.kwargs.copy()
        kwargs["workload"] = self.workload
        kwargs["original_workload"] = self.original_workload
        kwargs["accelerator"] = self.build_group_dedicated_accelerator()
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()

    @staticmethod
    def validate_core_yaml(core_yaml_path: str) -> dict[str, Any]:
        return validate_core_yaml(core_yaml_path)

    def build_group_dedicated_accelerator(self) -> Accelerator:
        """Build a new Accelerator with one dedicated `tpu_like` core per `(original node, group)` pair -- see
        `derive_group_dedicated_topology` for how the topology (core ids, offchip core, links) is derived."""
        topology = derive_group_dedicated_topology(self.workload, self.original_workload)
        offchip_core_id = topology.offchip_core_id

        tpu_core_data = self.validate_core_yaml(TPU_CORE_YAML_PATH)
        offchip_core_data = load_offchip_core_data()

        accelerator_data = {
            "name": f"{self.accelerator.name}_grouped",
            "cores": {i: tpu_core_data for i in range(offchip_core_id)} | {offchip_core_id: offchip_core_data},
            "offchip_core_id": offchip_core_id,
            "unit_energy_cost": 0,
            "core_memory_sharing": [],
            "core_connectivity": [topology.bus_connection, *topology.link_connections],
        }
        self.save_accelerator_yaml(
            offchip_core_id, topology.bus_connection, topology.link_connections, accelerator_data["name"]
        )
        return AcceleratorFactory(accelerator_data).create()

    def save_accelerator_yaml(
        self,
        offchip_core_id: int,
        bus_connection: dict[str, Any],
        link_connections: list[dict[str, Any]],
        accelerator_name: str,
    ) -> None:
        """Save the built accelerator to a human-readable yaml (core filenames, not inlined core definitions,
        matching e.g. tpu_like_quad_core.yaml) in the same directory as the tiled workload (the candidate
        directory when running under TilingExplorationStage's per-candidate sweep)."""
        saveable_accelerator_data = {
            "name": accelerator_name,
            "cores": {i: os.path.basename(TPU_CORE_YAML_PATH) for i in range(offchip_core_id)}
            | {offchip_core_id: os.path.basename(OFFCHIP_CORE_YAML_PATH)},
            "offchip_core_id": offchip_core_id,
            "unit_energy_cost": 0,
            "core_connectivity": [bus_connection, *link_connections],
        }
        accelerator_yaml_path = os.path.join(os.path.dirname(self.tiled_workload_path), "accelerator.yaml")
        with open(accelerator_yaml_path, "w") as f:
            yaml.safe_dump(saveable_accelerator_data, f, sort_keys=False)
        logger.info(f"Saved group-dedicated accelerator to {accelerator_yaml_path}.")
