import logging
import os
from collections import defaultdict
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
        validator = CoreValidator(open_yaml(core_yaml_path))
        if not validator.validate():
            raise ValueError(f"Core file {core_yaml_path} failed validation.")
        return validator.normalized_data

    def build_group_dedicated_accelerator(self) -> Accelerator:
        """Build a new Accelerator with one dedicated `tpu_like` core per `(original node, group)` pair, i.e. one
        core per inter-core-tiling slice (tiles sharing a group always share a core, since intra-core tiling never
        changes group), plus one shared `offchip` core reachable by every dedicated core over a bus (see
        tpu_like_quad_core.yaml). Tile-to-tile edges with bits > 0 become point-to-point links between the
        corresponding cores, bandwidth summed from all tile-edges mapping to that core pair; edges with bits == 0
        (same-core ordering edges) are dropped.

        Since every (node, group) pair now has an exclusive core, each tile's core allocation is pinned directly
        to that core so downstream stages don't try to resolve an allocation against the old hardware's core ids."""
        original_nodes = self.get_original_nodes()
        tiles_by_original = self.get_tiles_by_original_node(original_nodes)
        id_to_original_node = {node.id: node for node in original_nodes}

        node_core_ids: dict[ComputationNode, list[int]] = {}
        next_core_id = 0
        for original_node in original_nodes:
            k = get_inter_core_tiling_size(original_node)
            node_core_ids[original_node] = list(range(next_core_id, next_core_id + k))
            next_core_id += k
        offchip_core_id = next_core_id

        for original_node, tiles in tiles_by_original.items():
            core_ids = node_core_ids[original_node]
            for tile in tiles:
                tile.possible_core_allocation = core_ids
                tile.set_chosen_core_allocation(core_ids[tile.group])

        core_pair_bits: dict[tuple[int, int], int] = defaultdict(int)
        for producer_tile, consumer_tile, data in self.workload.edges(data=True):
            bits = data.get("bits", 0)
            if bits == 0:
                continue
            producer_node = id_to_original_node[producer_tile.id]
            consumer_node = id_to_original_node[consumer_tile.id]
            if producer_node is consumer_node:
                continue
            core_a = node_core_ids[producer_node][producer_tile.group]
            core_b = node_core_ids[consumer_node][consumer_tile.group]
            core_pair_bits[(core_a, core_b)] += bits

        tpu_core_data = self.validate_core_yaml(TPU_CORE_YAML_PATH)
        offchip_core_data = self.validate_core_yaml(OFFCHIP_CORE_YAML_PATH)

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
            f"Building group-dedicated accelerator: {len(original_nodes)} original nodes, "
            f"{offchip_core_id} dedicated cores, 1 offchip core, {len(link_connections)} core-to-core links."
        )

        accelerator_data = {
            "name": f"{self.accelerator.name}_grouped",
            "cores": {i: tpu_core_data for i in range(offchip_core_id)} | {offchip_core_id: offchip_core_data},
            "offchip_core_id": offchip_core_id,
            "unit_energy_cost": 0,
            "core_memory_sharing": [],
            "core_connectivity": [bus_connection, *link_connections],
        }
        self.save_accelerator_yaml(offchip_core_id, bus_connection, link_connections, accelerator_data["name"])
        return AcceleratorFactory(accelerator_data).create()

    def get_original_nodes(self) -> list[ComputationNode]:
        return [node for node in self.original_workload.topological_sort() if isinstance(node, ComputationNode)]

    def get_tiles_by_original_node(
        self, original_nodes: list[ComputationNode]
    ) -> dict[ComputationNode, list[ComputationNode]]:
        tiles = [node for node in self.workload.node_list if isinstance(node, ComputationNode)]
        return {
            original_node: [tile for tile in tiles if tile.id == original_node.id] for original_node in original_nodes
        }

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
