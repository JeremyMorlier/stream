"""Smoke tests for the optimizer-comparison plumbing: the search budget's stop rules (and its per-candidate
sub-budgets), the hypervolume helper, the generated tpu_like grids and the evolving-graph genome operators.

Run: python test_optimizer_comparison.py
"""

import os
import tempfile

from stream.hardware.architecture.grid_generator import tpu_grid_dict, write_tpu_grid
from stream.opt.search_budget import SearchBudget, hypervolume_3d
from stream.stages.generation.graph_evolution import GraphEvolutionStage, GraphGenome
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage


class FakeClock:
    def __init__(self, budget: SearchBudget):
        self.now = 0.0
        budget.elapsed = lambda: self.now  # type: ignore[method-assign]


def test_budget_converges() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        budget = SearchBudget(max_time_s=1000, window_s=100, rel_tol=0.01, trace_path=os.path.join(tmp, "trace.csv"))
        clock = FakeClock(budget)
        budget.record(10, 10, 1)  # EDP 100 at t=0
        clock.now = 50
        budget.record(9, 10, 1)  # EDP 90
        clock.now = 120
        assert not budget.should_stop(), "improved by 10% over the last window"
        budget.record(float("inf"), float("inf"), 1)  # infeasible: counted, never best
        clock.now = 151
        assert budget.should_stop() and budget.stop_reason == "converged"
        summary = budget.summary()
        assert summary["best_edp"] == 90 and summary["n_eval"] == 3
        assert summary["time_to_within_1pct"] == 50
        with open(os.path.join(tmp, "trace.csv")) as f:
            assert len(f.readlines()) == 4


def test_budget_times_out() -> None:
    budget = SearchBudget(max_time_s=10, window_s=100, rel_tol=0.01)
    clock = FakeClock(budget)
    budget.record(1, 1, 1)
    clock.now = 10
    assert budget.should_stop() and budget.stop_reason == "time"
    budget.stop("exhausted")
    assert budget.stop_reason == "time", "the first reason wins"


def test_sub_budget() -> None:
    parent = SearchBudget(max_time_s=100, window_s=1000, rel_tol=0.01)
    clock = FakeClock(parent)
    sub = parent.sub_budget(30)
    sub_clock = FakeClock(sub)  # type: ignore[arg-type]
    sub.record(2, 3, 1)
    assert parent.n_eval == 1 and parent.best_edp == 6, "records go to the parent"
    sub_clock.now = 30
    assert sub.should_stop() and sub.stop_reason == "time"
    assert not parent.should_stop(), "a candidate running out of time does not stop the search"
    other = parent.sub_budget(30)
    clock.now = 100
    assert other.should_stop() and other.stop_reason == "parent"
    capped = SearchBudget(max_time_s=100, window_s=1000, rel_tol=0.01)
    FakeClock(capped).now = 90
    assert capped.sub_budget(30).max_time_s == 10, "a candidate never outlives the search"


def _stage_with_groups(groups_per_layer: dict[int, int]) -> GraphEvolutionStage:
    stage = object.__new__(GraphEvolutionStage)
    stage._context = lambda tiling: type("Context", (), {"groups_per_layer": groups_per_layer})()  # type: ignore
    return stage


def test_graph_genome() -> None:
    genome = GraphGenome(
        nodes={0: [0.0], 1: [1.0], 2: [2.0]},
        links={frozenset((0, 1)), frozenset((1, 2))},
        tiling=(0,),
        alloc={10: [0, 1], 11: [0]},
    )
    genome.prune()
    assert set(genome.nodes) == {0, 1} and genome.links == {frozenset((0, 1))}, "unused node 2 and its link go"

    # Step 2: a new tiling gives layer 10 four groups and layer 11 one; labels stay on the graph's nodes.
    stage = _stage_with_groups({10: 4, 11: 1})
    on_graph = GraphGenome(nodes={5: [0.0], 7: [1.0]}, links=set(), tiling=(1,), alloc={10: [5, 9], 11: []})
    stage._fit_alloc(on_graph)
    assert len(on_graph.alloc[10]) == 4 and on_graph.alloc[10][:2] == [5, 7], "kept, then remapped modulo"
    assert len(on_graph.alloc[11]) == 1 and set(on_graph.alloc[10] + on_graph.alloc[11]) <= {5, 7}

    # Graft: the receiver takes layer 11's allocation, the node hosting it and nothing else from the donor.
    receiver = GraphGenome(nodes={0: [0.0]}, links=set(), tiling=(0,), alloc={10: [0], 11: [0]})
    donor = GraphGenome(nodes={0: [3.0], 1: [4.0]}, links={frozenset((0, 1))}, tiling=(0,), alloc={10: [0], 11: [1]})
    child = GraphEvolutionStage._graft(stage, receiver, donor, {11})
    assert child.alloc[10] == [0] and child.nodes[child.alloc[11][0]] == [4.0] and not child.links


def test_hypervolume() -> None:
    assert hypervolume_3d([(0, 0, 0)], (1, 1, 1)) == 1
    assert hypervolume_3d([(0.5, 0, 0), (0, 0.5, 0.5)], (1, 1, 1)) == 0.625
    assert hypervolume_3d([(2, 0, 0)], (1, 1, 1)) == 0


def test_grid() -> None:
    grid = tpu_grid_dict(4, 4)
    links = [c for c in grid["core_connectivity"] if c["type"] == "link"]
    assert len(links) == 2 * 4 * 3 + 16, "24 mesh links + one link per core to the SIMD core"
    hardware_path, mapping_path = write_tpu_grid(2, 2)
    accelerator = AcceleratorParserStage.parse_accelerator_from_yaml(None, hardware_path)  # type: ignore[arg-type]
    assert len(accelerator.core_list) == 6 and accelerator.offchip_core_id == 5
    assert os.path.exists(mapping_path)


if __name__ == "__main__":
    test_budget_converges()
    test_budget_times_out()
    test_sub_budget()
    test_graph_genome()
    test_hypervolume()
    test_grid()
    print("All optimizer comparison tests passed.")
