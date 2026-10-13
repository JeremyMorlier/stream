"""Smoke tests for the optimizer-comparison plumbing: the search budget's stop rules and front artifacts, the
hypervolume helper, the generated tpu_like grids, the evolving-graph genome operators, the design records and
the report data built from a run that saved little.

Run: python tests/test_optimizer_comparison.py
"""

import csv
import json
import os
import sys
import tempfile

from stream.hardware.architecture.grid_generator import tpu_grid_dict, write_tpu_grid
from stream.opt.search_budget import SearchBudget, hypervolume_3d
from stream.stages.generation.graph_evolution import GraphEvolutionStage, GraphGenome
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.visualization.design_record import (
    core_hierarchy_from_dict,
    format_tiling,
    hierarchy_key,
    rolled_design_record,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_optimizer_comparison  # noqa: E402  -- a script at the repository root


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


def test_budget_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        designs = os.path.join(tmp, "designs")
        budget = SearchBudget(1000, 100, 0.01, os.path.join(tmp, "trace.csv"), artifact_dir=designs)
        built = []

        def artifact(name):
            return lambda: built.append(name) or {"name": name}

        budget.record(10, 10, 10, artifact=artifact("a"))  # 1: front
        budget.record(5, 20, 10, artifact=artifact("b"))  # 2: front (trade-off with a)
        budget.record(11, 11, 11, artifact=artifact("c"))  # 3: dominated by a -- never built
        budget.record(10, 10, 10, artifact=artifact("d"))  # 4: a duplicate of a -- never built
        budget.record(4, 9, 9, artifact=artifact("e"))  # 5: dominates a and b -- their files go
        budget.record(float("inf"), float("inf"), 1, artifact=artifact("f"))  # 6: infeasible
        assert built == ["a", "b", "e"]
        assert sorted(os.listdir(designs)) == ["5.json"]
        with open(os.path.join(designs, "5.json")) as f:
            assert json.load(f) == {"name": "e"}
        with open(os.path.join(tmp, "trace.csv")) as f:
            assert [row["artifact"] for row in csv.DictReader(f)] == ["1", "2", "", "", "5", ""]
        assert budget.is_dominated(5, 10, 10) and not budget.is_dominated(3, 100, 100)


CORE_DICT = {
    "name": "core",
    "memories": {
        "rf_I1": {
            "size": 64,
            "operands": ["I1"],
            "served_dimensions": ["D1"],
            "ports": [{"type": "read", "bandwidth_max": 8}, {"type": "write", "bandwidth_max": 8}],
        },
        "rf_O": {
            "size": 32,
            "operands": ["O"],
            "served_dimensions": ["D2"],
            "ports": [{"type": "read_write", "bandwidth_max": 16}],
        },
        "sram": {
            "size": 8192,
            "operands": ["I2", "I1", "O"],
            "served_dimensions": ["D1", "D2"],
            "ports": [{"type": "read", "bandwidth_max": 64}, {"type": "write", "bandwidth_max": 32}],
        },
    },
    "operational_array": {"dimensions": ["D1", "D2"], "sizes": [4, 8]},
}


def test_design_record_from_dicts() -> None:
    hierarchy = core_hierarchy_from_dict(CORE_DICT)
    assert hierarchy["oa"] == {"D1": 4, "D2": 8} and hierarchy["units"] == 32
    levels = {level["name"]: level for level in hierarchy["levels"]}
    assert levels["rf_I1"]["tier"] == 0 and levels["rf_O"]["tier"] == 0 and levels["sram"]["tier"] == 1
    assert levels["sram"]["operands"] == ["I1", "I2", "O"], "operands in I1, I2, O order"
    assert (levels["sram"]["read_bw"], levels["sram"]["write_bw"]) == (64, 32)
    assert (levels["rf_O"]["read_bw"], levels["rf_O"]["write_bw"]) == (16, 16)
    assert hierarchy_key(hierarchy) == hierarchy_key(core_hierarchy_from_dict({**CORE_DICT, "name": "other"}))

    record = {
        "id": "r0-c2",
        "nb_cores": 2,
        "core_dict_of_core": {0: CORE_DICT, 1: CORE_DICT},
        "node_core_ids": {1: [0], 2: [1]},
        "link_connections": [{"type": "link", "cores": [0, 1], "bandwidth": 64}],
        "bus_connection": {"type": "bus", "cores": [0, 1, 2], "bandwidth": 32},
    }
    design = rolled_design_record(record, {1: "conv", 2: "relu"})
    assert len(design["designs"]) == 1 and set(design["core_design"]) == {"0", "1"}
    assert design["schedule"] is None
    kinds = sorted(core["kind"] for core in design["topology"]["cores"])
    assert kinds == ["bus", "compute", "compute", "offchip"]
    assert format_tiling({3: {"intra_core_tiling": [("OX", 7)], "inter_core_tiling": []}}) == {
        "3": {"intra": "OX7", "inter": "-"}
    }


def _write_run(run_dir: str, rows: list[tuple], artifact: dict | None = None) -> None:
    os.makedirs(os.path.join(run_dir, "designs"))
    with open(os.path.join(run_dir, "trace.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("t", "n_eval", "latency", "energy", "area", "edp", "best_edp", "feasible", "info", "artifact"))
        for n_eval, (latency, energy, area, info, artifact_id) in enumerate(rows, start=1):
            writer.writerow((n_eval, n_eval, latency, energy, area, latency * energy, 1, 1, info, artifact_id))
    if artifact is not None:
        with open(os.path.join(run_dir, "designs", "1.json"), "w") as f:
            json.dump(artifact, f)
    summary = {
        "stop_reason": "time",
        "time_to_stop": 10,
        "n_eval": len(rows),
        "best_edp": 100.0,
        "best_point": None,
        "time_to_within_1pct": 1,
        "max_time_s": 10,
        "window_s": 5,
        "rel_tol": 0.01,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f)


def test_report_data_degrades() -> None:
    artifact = {
        "topology": {"cores": [{"id": 0, "kind": "compute"}], "edges": []},
        "designs": {},
        "core_design": {},
        "schedule": {"rows": ["Core 0"], "tasks": []},
        "energy": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        # "a" saved its front point's record; "b" saved nothing beyond its trace.
        _write_run(os.path.join(tmp, "w", "a"), [(10, 10, 1, "step1 cores=1", "1"), (20, 20, 2, "x", "")], artifact)
        _write_run(os.path.join(tmp, "w", "b"), [(5, 30, 1, "step1 cores=2", ""), (30, 5, 1, "x", "")])
        runs = analyze_optimizer_comparison.load_runs(tmp)
        rows = [
            {"workload": "w", "method": m, "edp_gap_to_best_pct": 0, "pareto_points": 1, "hypervolume": 0.5}
            for m in ("a", "b")
        ]
        data, sidecars = analyze_optimizer_comparison.build_report_data(tmp, runs, rows)
        designs = {d["id"]: d for d in data["by_workload"]["w"]["designs"]}
        assert set(designs) == {"w_a_1", "w_b_1", "w_b_2"}, "only Pareto points, dominated (20, 20, 2) left out"
        assert designs["w_a_1"]["has_schedule"] and designs["w_a_1"]["file"] == "w_a_1.js"
        assert designs["w_b_1"]["source"] == "objectives only" and designs["w_b_1"]["file"] is None
        assert set(sidecars) == {"w_a_1.js"} and "topology" not in sidecars["w_a_1.js"]


if __name__ == "__main__":
    test_budget_converges()
    test_budget_times_out()
    test_graph_genome()
    test_hypervolume()
    test_grid()
    test_budget_artifacts()
    test_design_record_from_dicts()
    test_report_data_degrades()
    print("All optimizer comparison tests passed.")
