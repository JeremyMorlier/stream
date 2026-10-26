"""Tests for `CoreOptimizationStage`, the core pool covering every case of core specialization.

A wrong pool fails silently: a missing target only makes the front a little worse, and a design added to the wrong
front is still scheduled. So the properties checked here are the target sets and their weights `n_sigma`, the
clustering, the fused chains, and the merge into the per-class fronts. The search itself runs with the per-tile
evaluator stubbed, so no CACTI or ZigZag runs.

Run directly with `python tests/test_core_optimization.py`, or collect with pytest.
"""

import math
import os
import tempfile

from zigzag.datatypes import LayerDim

from stream.stages.generation import core_architecture_exploration as exploration
from stream.stages.generation import core_optimization as module
from stream.stages.generation.core_architecture_exploration import ParetoCoreSearchResult
from stream.stages.generation.core_optimization import (
    CoreOptimizationStage,
    SpecializationTarget,
    dedup_targets,
    fused_chains,
    merge_into_fronts,
    operator_targets,
    shared_targets,
    ward_clusters,
)


class Node:
    """An operator or tile: hashable by identity, as the stage keys `tiles_by_original` by node."""

    def __init__(self, node_id: int, **sizes: int):
        self.id = node_id
        self.layer_dim_sizes = {LayerDim(name): size for name, size in sizes.items()}


def build_search() -> ParetoCoreSearchResult:
    """Three operators. Op 0 is a 1x1 conv with a boundary tile (classes 0 and 1); op 1 is a similar 1x1 conv
    (class 2); op 2 is a 3x3 conv with a much smaller K and C (class 3)."""
    ops = [Node(0), Node(1), Node(2)]
    shapes = [
        dict(K=64, C=64, OX=8, OY=8, FX=1, FY=1),
        dict(K=64, C=64, OX=4, OY=8, FX=1, FY=1),
        dict(K=64, C=32, OX=8, OY=8, FX=1, FY=1),
        dict(K=4, C=4, OX=8, OY=8, FX=3, FY=3),
    ]
    tiles_of_op = {0: [0, 0, 0, 1], 1: [2, 2], 2: [3, 3, 3]}
    unique_tiles = [Node(op, **shape) for op, shape in zip([0, 0, 1, 2], shapes, strict=True)]
    tiles_by_original, tile_to_unique_index = {}, {}
    for op in ops:
        tiles = [Node(op.id, **shapes[sigma]) for sigma in tiles_of_op[op.id]]
        tiles_by_original[op] = tiles
        tile_to_unique_index.update({id(tile): sigma for tile, sigma in zip(tiles, tiles_of_op[op.id], strict=True)})
    return ParetoCoreSearchResult(
        designs_per_tile=[[{"core_dict": {"name": f"tile{s}", "knob": s}, "origin": "tile"}] for s in range(4)],
        unique_tiles=unique_tiles,
        tile_to_unique_index=tile_to_unique_index,
        original_nodes=ops,
        tiles_by_original=tiles_by_original,
    )


def test_operator_targets_cover_only_multi_class_operators_with_their_tile_counts():
    targets = operator_targets(build_search())
    assert [(t.label, t.member_classes, t.counts) for t in targets] == [("op0", [0, 1], [3, 1])]


def test_ward_clusters_split_the_obvious_groups():
    features = [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.0, 5.1], [5.1, 5.0]]
    assert sorted(ward_clusters(features, 2)) == [[0, 1], [2, 3, 4]]
    assert ward_clusters(features, 5) == [[0], [1], [2], [3], [4]]


def test_shared_targets_group_similar_classes_weighted_by_workload_counts():
    targets = shared_targets(build_search(), nb_clusters=2)
    assert [(t.member_classes, t.counts) for t in targets] == [([0, 1, 2], [3, 1, 2])]
    assert shared_targets(build_search(), nb_clusters=0) == []


def test_dedup_drops_single_class_targets_and_repeats():
    targets = [
        SpecializationTarget("operator", "op0", [0, 1], [3, 1]),
        SpecializationTarget("shared", "cluster0", [1, 0], [1, 3]),
        SpecializationTarget("shared", "cluster1", [2], [2]),
        SpecializationTarget("shared", "cluster2", [0, 1], [1, 1]),
    ]
    assert [t.label for t in dedup_targets(targets)] == ["op0", "cluster2"]


class FakeWorkload:
    """A DAG of operators with one non-computation node between ops 1 and 2."""

    def __init__(self, ops, edges):
        self.ops, self.edges = ops, edges

    def successors(self, node):
        return [b for a, b in self.edges if a is node]

    def topological_sort(self):
        return self.ops


def test_fused_chains_follow_the_dag_inside_one_stack():
    ops = [Node(0), Node(1), Node(2), Node(3)]
    reshape = object()
    workload = FakeWorkload(ops, [(ops[0], ops[1]), (ops[1], reshape), (reshape, ops[2]), (ops[2], ops[3])])
    original = module.ComputationNode
    module.ComputationNode = Node
    try:
        chain_ids = lambda stacks, length: [[n.id for n in c] for c in fused_chains(workload, stacks, length)]  # noqa: E731
        assert chain_ids([(0, 1, 2, 3)], 1) == []
        assert chain_ids([(0, 1, 2, 3)], 2) == [[0, 1], [1, 2], [2, 3]]
        assert chain_ids([(0, 1, 2, 3)], 3) == [[0, 1], [0, 1, 2], [1, 2], [1, 2, 3], [2, 3]]
        assert chain_ids([(0, 1), (2, 3)], 3) == [[0, 1], [2, 3]]
    finally:
        module.ComputationNode = original


def test_merge_adds_to_every_member_front_once_and_tags_the_origin():
    search = build_search()
    target = SpecializationTarget("shared", "cluster0", [0, 2], [3, 2])
    designs = [{"core_dict": {"name": "a", "knob": 99}}, {"core_dict": {"name": "copy_of_tile0", "knob": 0}}]
    # The copy of tile 0's design is new to class 2 only.
    assert merge_into_fronts(search, target, designs) == 3
    assert merge_into_fronts(search, target, designs) == 0
    assert [d["core_dict"]["knob"] for d in search.designs_per_tile[0]] == [0, 99]
    assert [(d["core_dict"]["knob"], d["origin"]) for d in search.designs_per_tile[2]] == [
        (2, "tile"),
        (99, "shared"),
        (0, "shared"),
    ]
    assert search.designs_per_tile[1] == build_search().designs_per_tile[1]


def stub_tile_cost(values, gene_layout, tile, *args):
    """A cost that depends on the genes and the tile, so the weighted fronts differ per target."""
    macs = math.prod(tile.layer_dim_sizes.values())
    return (macs / (1 + values[0]), macs * (1 + values[1]), 1 + values[0] + values[1])


def build_stage(directory: str) -> CoreOptimizationStage:
    stage = object.__new__(CoreOptimizationStage)
    stage.kwargs = {"loma_lpf_limit": 6, "temporal_mapping_type": None}
    stage.gene_ranges = exploration.CoreParamRanges()
    stage.nb_core_ga_individuals = 4
    stage.nb_core_ga_generations = 2
    stage.shared_core_pareto_points = 3
    stage.search_budget = None
    stage.core_convergence_patience = None
    stage.core_rel_tol = 0.005
    stage.max_workers = 1
    stage.reuse_pareto_fronts = True
    stage.core_search_dir = directory
    stage.specialized_designs_path = os.path.join(directory, "specialized_designs.pickle")
    return stage


def test_search_targets_returns_decoded_finite_designs_and_reuses_them():
    original = exploration._evaluate_design_on_tile
    exploration._evaluate_design_on_tile = stub_tile_cost
    try:
        with tempfile.TemporaryDirectory() as directory:
            search = build_search()
            targets = operator_targets(search)
            designs = build_stage(directory).search_targets(targets, search)
            assert 0 < len(designs["op0"]) <= 3
            assert all({"core_dict", "fitness", "genes"} <= design.keys() for design in designs["op0"])
            assert all(math.isfinite(v) for design in designs["op0"] for v in design["fitness"])

            exploration._evaluate_design_on_tile = None  # a reused target must not be searched again
            assert build_stage(directory).search_targets(targets, search) == designs
    finally:
        exploration._evaluate_design_on_tile = original


if __name__ == "__main__":
    test_operator_targets_cover_only_multi_class_operators_with_their_tile_counts()
    test_ward_clusters_split_the_obvious_groups()
    test_shared_targets_group_similar_classes_weighted_by_workload_counts()
    test_dedup_drops_single_class_targets_and_repeats()
    test_fused_chains_follow_the_dag_inside_one_stack()
    test_merge_adds_to_every_member_front_once_and_tags_the_origin()
    test_search_targets_returns_decoded_finite_designs_and_reuses_them()
    print("all passed")
