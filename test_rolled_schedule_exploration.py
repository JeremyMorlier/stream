"""Tests for `RolledScheduleExplorationStage`'s folding.

Folding is where a wrong answer is easy to miss: a colouring that merges two cores which are in fact busy at
the same time still *runs*, it just quietly serializes, and an area that accidentally counts the offchip core
still *looks* plausible -- while making rolled and unrolled points incomparable on the shared Pareto front.
So the properties checked here are the ones whose violation would be invisible in a plot:

- merged cores really are idle at the same time (at tolerance 0),
- the colouring is not needlessly wasteful, and the exact mode never loses to the hull mode,
- a layer's own inter-core slices are never merged,
- rolled area is measured exactly the way the unrolled bound measures it,
- the re-derived link set drops what became intra-core and coalesces the rest,
- the pre-screen never drops a point the exhaustive fold would have kept.

All synthetic, as in `test_schedule_exploration.py`: the stage is built with `object.__new__` and its
measurement stubbed, so these run in milliseconds instead of the minutes a real workload costs.

Run directly with `python test_rolled_schedule_exploration.py`, or collect with pytest.
"""

import random
from types import SimpleNamespace

from stream.stages.generation.rolled_schedule_exploration import (
    CoreOccupancy,
    RolledScheduleExplorationStage,
    canonical,
    coalesce,
    partition_key,
    union_overlap,
)
from stream.stages.generation.schedule_exploration import ParetoArchive, core_dict_key
from stream.stages.generation.tiled_workload_accelerator_generation import (
    fold_core_pair_bits,
    renumber_merged_cores,
)

# Layer 0 runs on two inter-core slices (cores 0 and 1); layers 1 and 2 get one core each.
CORE_IDS_OF_NODE = {0: [0, 1], 1: [2], 2: [3]}
NB_CORES = 4
NB_DESIGNS = 3


def build_stage(seed: int = 0, **overrides) -> RolledScheduleExplorationStage:
    """A rolled stage with a synthetic plan: 4 unrolled cores over 3 layers, 3 shape classes, 3 designs each."""
    rng = random.Random(seed)
    stage = object.__new__(RolledScheduleExplorationStage)

    stage.decision_classes = [0, 1, 2]
    stage.depth_of_class = {0: 0, 1: 1, 2: 2}
    stage.designs_per_class = [
        [{"core_dict": {"name": f"core_{c}_{r}", "knob": c * 10 + r}} for r in range(NB_DESIGNS)] for c in range(3)
    ]
    stage.design_class_of_node = {0: 0, 1: 1, 2: 2}
    stage.core_ids_of_node = dict(CORE_IDS_OF_NODE)
    stage.unrolled_core_ids = [0, 1, 2, 3]
    stage.design_class_of_core = {0: 0, 1: 0, 2: 1, 3: 2}
    stage.cores_per_class = {0: 2, 1: 1, 2: 1}

    # Two tiles per core, one shape class per layer.
    shape_of_core = {0: 0, 1: 0, 2: 1, 3: 2}
    stage.shape_class_of_tile = {}
    stage.unrolled_core_of_tile = {}
    for core_id in stage.unrolled_core_ids:
        for sub_id in range(2):
            key = (core_id, sub_id)
            stage.shape_class_of_tile[key] = shape_of_core[core_id]
            stage.unrolled_core_of_tile[key] = core_id
    stage.shape_counts_of_core = {core_id: {shape_of_core[core_id]: 2} for core_id in stage.unrolled_core_ids}

    # A chain over the tiles, so the critical-path term of the bound is exercised.
    stage.topo_tile_keys = sorted(stage.shape_class_of_tile)
    stage.topo_predecessors = [[] if i == 0 else [i - 1] for i in range(len(stage.topo_tile_keys))]

    # Full shape x design cost matrix, keyed exactly as the stage keys it -- by `core_dict_key` of the design,
    # which is also what `_design_key_of_core` produces, so the two must not be stubbed apart.
    design_dicts = {
        core_dict_key(stage.designs_per_class[c][r]["core_dict"]): stage.designs_per_class[c][r]["core_dict"]
        for c in range(3)
        for r in range(NB_DESIGNS)
    }
    keys = sorted(design_dicts)
    stage.runtime_cache = {(s, k): rng.uniform(10, 200) for s in range(3) for k in keys}
    stage.energy_cache = {(s, k): rng.uniform(1e3, 1e5) for s in range(3) for k in keys}
    stage.area_cache = {k: rng.uniform(100, 5000) for k in keys}
    stage._distinct_designs = lambda _d=design_dicts: dict(_d)

    stage.fold_overlap = "hull"
    stage.fold_tolerances = (0.0, 0.5, 1.0)
    stage.rolled_core_count_targets = None
    stage.fold_allow_sibling_merge = False
    stage.merged_design_pool = "members"
    stage.merged_design_objectives = ("latency", "energy", "area")
    stage.keep_singleton_designs = True
    stage.max_rolled_variants_per_schedule = 8
    stage.max_rolled_evaluations = 100
    stage.fold_source = "pareto"
    stage.prune_rolled = True
    stage.forbidden_pairs = {canonical(0, 1)}  # layer 0's two slices
    stage._plan_of_record = {}
    stage.fold_log = []
    stage.all_results = []
    for key, value in overrides.items():
        setattr(stage, key, value)
    return stage


def occupancy_from(windows: dict[int, list[tuple[int, int]]]) -> CoreOccupancy:
    intervals = {core: coalesce(w) for core, w in windows.items()}
    return CoreOccupancy(
        intervals=intervals,
        load={c: sum(e - s for s, e in w) for c, w in intervals.items()},
        hull={c: (w[0][0], w[-1][1]) if w else (0, 0) for c, w in intervals.items()},
        node_groups={c: set() for c in intervals},
        makespan=max((w[-1][1] for w in intervals.values() if w), default=0),
    )


def source_record(ranks=(0, 0, 0), **extra) -> dict:
    return {
        "id": "u" + "-".join(map(str, ranks)),
        "kind": "unrolled",
        "ranks": ranks,
        "nb_cores": NB_CORES,
        "latency": 1e6,
        "energy": 1e9,
        "area": 1e4,
        **extra,
    }


def random_occupancy(rng: random.Random, nb_cores: int = NB_CORES) -> CoreOccupancy:
    windows = {}
    for core in range(nb_cores):
        chunks = []
        cursor = rng.randint(0, 40)
        for _ in range(rng.randint(1, 3)):
            start = cursor + rng.randint(0, 30)
            chunks.append((start, start + rng.randint(5, 40)))
            cursor = chunks[-1][1]
        windows[core] = chunks
    return occupancy_from(windows)


# ------------------------------------------------------------------------------------ colouring ---------------


def test_merged_cores_have_disjoint_busy_intervals():
    """The whole premise: at tolerance 0, cores sharing a physical core are never busy at the same time."""
    for seed in range(40):
        rng = random.Random(seed)
        occupancy = random_occupancy(rng)
        for mode in ("hull", "exact"):
            stage = build_stage(fold_overlap=mode, forbidden_pairs=set())
            colour_of = stage._colour(occupancy, stage._conflicts(occupancy, 0.0))
            classes: dict[int, list[int]] = {}
            for core, colour in colour_of.items():
                classes.setdefault(colour, []).append(core)
            for members in classes.values():
                for i, a in enumerate(members):
                    for b in members[i + 1 :]:
                        assert union_overlap(occupancy.intervals[a], occupancy.intervals[b]) == 0, (
                            f"seed {seed}, mode {mode}: cores {a} and {b} merged but overlap"
                        )


def test_hull_colouring_uses_the_minimum_number_of_colours():
    """On hulls the conflict graph is an interval graph, so the greedy must hit the clique number -- the most
    cores live at any one instant."""
    for seed in range(40):
        rng = random.Random(seed)
        occupancy = random_occupancy(rng)
        stage = build_stage(fold_overlap="hull", forbidden_pairs=set())
        colour_of = stage._colour(occupancy, stage._conflicts(occupancy, 0.0))
        nb_colours = len(set(colour_of.values()))

        events = []
        for start, end in occupancy.hull.values():
            events += [(start, 1), (end, -1)]
        depth = best = 0
        for _, delta in sorted(events, key=lambda e: (e[0], -e[1])):
            depth += delta
            best = max(best, depth)
        assert nb_colours == best, f"seed {seed}: {nb_colours} colours for max depth {best}"


def test_exact_mode_never_uses_more_colours_than_hull_mode():
    """The exact conflict graph is a subgraph of the hull one, so it can only need fewer cores."""
    for seed in range(40):
        rng = random.Random(seed)
        occupancy = random_occupancy(rng, nb_cores=6)
        counts = {}
        for mode in ("hull", "exact"):
            stage = build_stage(fold_overlap=mode, forbidden_pairs=set(), unrolled_core_ids=list(range(6)))
            colour_of = stage._colour(occupancy, stage._conflicts(occupancy, 0.0))
            counts[mode] = len(set(colour_of.values()))
        assert counts["exact"] <= counts["hull"], f"seed {seed}: {counts}"


def test_siblings_never_merge():
    """Cores 0 and 1 are one layer's inter-core slices: forbidden even when their windows are disjoint."""
    occupancy = occupancy_from({0: [(0, 10)], 1: [(20, 30)], 2: [(40, 50)], 3: [(60, 70)]})
    for mode in ("hull", "exact"):
        stage = build_stage(fold_overlap=mode)
        colour_of = stage._colour(occupancy, stage._conflicts(occupancy, 1.0))
        assert colour_of[0] != colour_of[1], mode


def test_tolerance_ladder_is_monotone():
    """Raising the tolerance only removes conflicts, so core counts must not go up along the ladder."""
    for seed in range(20):
        rng = random.Random(seed)
        occupancy = random_occupancy(rng, nb_cores=6)
        stage = build_stage(fold_overlap="hull", forbidden_pairs=set(), unrolled_core_ids=list(range(6)))
        counts = [
            len(set(stage._colour(occupancy, stage._conflicts(occupancy, t)).values())) for t in (0.0, 0.25, 0.5, 1.0)
        ]
        assert counts == sorted(counts, reverse=True), f"seed {seed}: {counts}"


# -------------------------------------------------------------------------------------- plans -----------------


def test_core_count_strictly_decreases():
    """A fold that merges nothing is not a fold and must not be measured."""
    occupancy = occupancy_from({0: [(0, 10)], 1: [(0, 10)], 2: [(20, 30)], 3: [(40, 50)]})
    stage = build_stage()
    plans = stage._fold_plans(source_record(occupancy=occupancy))
    assert plans
    assert all(plan.nb_cores < NB_CORES for plan in plans)


def test_no_plans_without_occupancy():
    assert build_stage()._fold_plans(source_record()) == []


def test_node_core_ids_keep_their_length():
    """`possible_core_allocation[node.group]` has to keep working, so a layer's core list keeps its length."""
    occupancy = occupancy_from({0: [(0, 10)], 1: [(0, 10)], 2: [(20, 30)], 3: [(40, 50)]})
    stage = build_stage()
    for plan in stage._fold_plans(source_record(occupancy=occupancy)):
        folded = {node_id: [plan.rolled_of[core] for core in cores] for node_id, cores in CORE_IDS_OF_NODE.items()}
        for node_id, cores in CORE_IDS_OF_NODE.items():
            assert len(folded[node_id]) == len(cores)


def test_rolled_area_never_exceeds_unrolled():
    """Merging can only remove cores, and singletons keep their design, so area cannot grow."""
    occupancy = occupancy_from({0: [(0, 10)], 1: [(0, 10)], 2: [(20, 30)], 3: [(40, 50)]})
    stage = build_stage()
    record = source_record(occupancy=occupancy)
    unrolled_area = sum(
        stage.area_cache[key] for key in stage._design_key_of_core(stage._node_core_dicts(record["ranks"])).values()
    )
    for plan in stage._fold_plans(record):
        assert stage._fold_area(plan) <= unrolled_area + 1e-9


def test_identity_fold_reproduces_the_unrolled_area():
    """The sharpest guard on area comparability: folding nothing must give exactly the unrolled number, i.e.
    compute cores only, never `Accelerator.area`, which also counts the offchip core."""
    stage = build_stage()
    record = source_record()
    design_of_core = stage._design_key_of_core(stage._node_core_dicts(record["ranks"]))
    unrolled_area = sum(stage.area_cache[key] for key in design_of_core.values())

    colour_of = {core: core for core in stage.unrolled_core_ids}  # every core its own class
    rolled_of, design_key_of_core, _dicts = stage._choose_designs(record, colour_of, "area")
    plan = SimpleNamespace(design_key_of_core=design_key_of_core, nb_cores=len(design_key_of_core))
    assert abs(stage._fold_area(plan) - unrolled_area) < 1e-9
    assert all(rolled_of[core] == core for core in stage.unrolled_core_ids)


def test_merged_core_design_is_cost_modelled_for_every_layer_it_hosts():
    """A merged core runs layers that were never meant for its design -- the whole reason the warm-up builds
    the full matrix. Every chosen design must have a cost for every shape it now carries."""
    occupancy = occupancy_from({0: [(0, 10)], 1: [(0, 10)], 2: [(20, 30)], 3: [(40, 50)]})
    stage = build_stage()
    for plan in stage._fold_plans(source_record(occupancy=occupancy)):
        hosted: dict[int, set[int]] = {}
        for tile, core in stage.unrolled_core_of_tile.items():
            hosted.setdefault(plan.rolled_of[core], set()).add(stage.shape_class_of_tile[tile])
        for rolled_id, shapes in hosted.items():
            for shape in shapes:
                assert (shape, plan.design_key_of_core[rolled_id]) in stage.runtime_cache


# ------------------------------------------------------------------------------------- topology ---------------


def test_renumbering_is_contiguous_and_offchip_last():
    """`AcceleratorFactory.create_core_graph` asserts core.id == index, so ids must be dense from 0."""
    rolled_of = renumber_merged_cores({0: "a", 1: "b", 2: "a", 3: "c", 4: "b"})
    assert sorted(set(rolled_of.values())) == [0, 1, 2]
    assert rolled_of[0] == rolled_of[2] and rolled_of[1] == rolled_of[4]
    assert len(set(rolled_of.values())) == 3  # offchip would be id 3


def test_renumbering_is_deterministic():
    labels = {0: "z", 1: "y", 2: "z", 3: "x"}
    assert renumber_merged_cores(labels) == renumber_merged_cores(dict(reversed(list(labels.items()))))


def test_link_rederivation_drops_merged_traffic_and_sums_the_rest():
    """Traffic inside a merged core needs no link; traffic that used to cross several links coalesces onto one."""
    folded = fold_core_pair_bits({(0, 2): 100, (1, 3): 50, (0, 1): 7}, {0: 0, 1: 0, 2: 1, 3: 1})
    assert folded == {(0, 1): 150}, folded  # (0,1) became intra-core; (0,2) and (1,3) merged into one link


def test_link_rederivation_has_no_self_pairs_and_positive_bandwidth():
    folded = fold_core_pair_bits({(0, 1): 5, (1, 2): 9, (2, 0): 4}, {0: 0, 1: 0, 2: 1})
    assert all(a != b for a, b in folded)
    assert all(bits > 0 for bits in folded.values())
    assert folded == {(0, 1): 13}


# --------------------------------------------------------------------------------- bounds and roll ------------


def test_rolled_measurements_respect_their_bounds():
    occupancy = occupancy_from({0: [(0, 10)], 1: [(0, 10)], 2: [(20, 30)], 3: [(40, 50)]})
    stage = build_stage()
    archive, _ = run_roll(stage, occupancy)
    rolled = [r for r in stage.all_results if r["kind"] == "rolled"]
    assert rolled
    for record in rolled:
        assert record["latency"] >= record["latency_bound"] - 1e-6
        assert record["energy"] >= record["energy_bound"] - 1e-6


def test_rolled_prescreen_is_sound():
    """Pruning must not change the front: a bound dominated by a measured point can never beat it."""
    occupancy = occupancy_from({0: [(0, 10)], 1: [(0, 10)], 2: [(20, 30)], 3: [(40, 50)]})
    fronts = []
    for prune in (True, False):
        stage = build_stage(prune_rolled=prune)
        archive, _ = run_roll(stage, occupancy)
        fronts.append({tuple(round(v, 6) for v in point) for point, _ in archive.entries})
    assert fronts[0] == fronts[1], fronts


def test_infeasible_fold_is_recorded_not_raised():
    """A merged core can overflow its memory; that is a property of the fold, not a crash."""
    occupancy = occupancy_from({0: [(0, 10)], 1: [(0, 10)], 2: [(20, 30)], 3: [(40, 50)]})
    stage = build_stage()

    def explode(plan, save_yaml=False):
        raise ValueError("Inserting tensor caused memory overflow.")

    stage._measure_fold = explode
    archive = ParetoArchive()
    archive.add((1e6, 1e9, 1e4), source_record(occupancy=occupancy))
    fold_stats = stage._roll(archive, {})
    assert fold_stats["infeasible"] > 0
    assert fold_stats["measured"] == 0
    assert any(entry["status"] == "infeasible" for entry in stage.fold_log)


def test_combined_archive_drops_dominated_unrolled_points():
    """Rolled and unrolled share one archive, so a better rolled point must evict the unrolled one it beats."""
    archive = ParetoArchive()
    unrolled = source_record()
    archive.add((100.0, 100.0, 100.0), unrolled)
    archive.add((50.0, 50.0, 50.0), {"id": "r0", "kind": "rolled", "ranks": (0, 0, 0), "nb_cores": 1})
    ids = {record["id"] for _point, record in archive.entries}
    assert ids == {"r0"}


def test_partition_key_ignores_colour_numbering():
    assert partition_key({0: 0, 1: 0, 2: 1}) == partition_key({0: 7, 1: 7, 2: 3})


def test_coalesce_merges_touching_windows():
    assert coalesce([(0, 5), (5, 9), (20, 25)]) == [(0, 9), (20, 25)]
    assert union_overlap([(0, 10)], [(5, 20)]) == 5
    assert union_overlap([(0, 10)], [(10, 20)]) == 0


def run_roll(stage: RolledScheduleExplorationStage, occupancy: CoreOccupancy):
    """Roll one source schedule with a stubbed scheduler that is never better than the analytic bound."""

    def fake_measure(plan, save_yaml=False):
        latency, energy, _area = stage._fold_bounds(plan)
        coupling = (hash(plan.key) % 1000) / 1000
        scme = SimpleNamespace(latency=latency * (1 + 0.3 * coupling), energy=energy * (1 + 0.3 * (1 - coupling)))
        topology = SimpleNamespace(
            members_of_core={}, hosted_node_groups={}, node_core_ids={}, link_connections=[], bus_connection={}
        )
        return scme, topology

    stage._measure_fold = fake_measure
    record = source_record(occupancy=occupancy)
    stage.all_results = [record]
    archive = ParetoArchive()
    archive.add((record["latency"], record["energy"], record["area"]), record)
    fold_stats = stage._roll(archive, {})
    return archive, fold_stats


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("All rolled schedule exploration tests passed.")
