"""Tests for the warm-up's tolerance of (tile, core) pairs ZigZag cannot map.

`ZigZagCoreMappingEstimationStage(skip_unmappable=True)` records such a pair instead of raising, and
`ScheduleSpace._drop_unmappable_designs` removes the designs that hit one from the fronts.

Run directly with `python tests/test_warm_up_unmappable.py`, or collect with pytest.
"""

import os
import tempfile
from types import SimpleNamespace

from zigzag.opt.loma.engine import NoValidLoopOrderingFoundException

from stream.stages.estimation.zigzag_core_mapping_estimation import ZigZagCoreMappingEstimationStage
from stream.stages.generation.schedule_evaluation import core_dict_key
from stream.stages.generation.schedule_space import ScheduleSpace
from stream.utils import CostModelEvaluationLUT
from tests.test_memory_hierarchy_violations import conv_tile, core


def stage_on_identical_cores(skip_unmappable: bool, cache_dir: str):
    """An estimation stage over one tile and two hardware-identical cores, bypassing the workload plumbing of
    `__init__`, whose ZigZag run always fails and counts its calls."""
    tile = conv_tile(32, 32, 8, 8)
    cores = {core_id: core(2**18, 256, 2**21) for core_id in (0, 1)}
    for core_id, each in cores.items():
        each.id = core_id
    stage = object.__new__(ZigZagCoreMappingEstimationStage)
    stage.unique_nodes = [tile]
    stage.valid_allocations = {tile: [0, 1]}
    stage.accelerator = SimpleNamespace(get_core=cores.__getitem__)
    stage.cost_lut = CostModelEvaluationLUT(os.path.join(cache_dir, "cost_lut.pickle"))
    stage.layer_stacks = [(0,)]
    stage.skip_unmappable = skip_unmappable
    stage.unmappable = []
    stage.zigzag_calls = 0

    def failing_run_zigzag(*_):
        stage.zigzag_calls += 1
        raise NoValidLoopOrderingFoundException

    stage.run_zigzag = failing_run_zigzag
    return stage, tile


def test_unmappable_pair_is_recorded_once_per_equal_core():
    with tempfile.TemporaryDirectory() as cache_dir:
        stage, tile = stage_on_identical_cores(skip_unmappable=True, cache_dir=cache_dir)
        stage.update_cost_lut()
        assert [each.id for _, each in stage.unmappable] == [0, 1]
        assert stage.zigzag_calls == 1  # the identical second core reuses the failure
        assert not stage.cost_lut.get_cores(tile)


def test_unmappable_pair_still_raises_by_default():
    with tempfile.TemporaryDirectory() as cache_dir:
        stage, _ = stage_on_identical_cores(skip_unmappable=False, cache_dir=cache_dir)
        try:
            stage.update_cost_lut()
        except NoValidLoopOrderingFoundException:
            return
        raise AssertionError("expected NoValidLoopOrderingFoundException")


def test_unmappable_designs_are_dropped_from_the_fronts():
    designs = [{"core_dict": {"name": f"c{size}", "size": size}} for size in (1, 2, 3)]
    space = object.__new__(ScheduleSpace)
    space.decision_classes = [0, 1]
    space.designs_per_class = {0: designs[:2], 1: designs[1:]}
    space.unmappable_designs = {core_dict_key(designs[1]["core_dict"])}
    space._drop_unmappable_designs()
    assert space.designs_per_class == {0: [designs[0]], 1: [designs[2]]}

    space.unmappable_designs.add(core_dict_key(designs[0]["core_dict"]))
    try:
        space._drop_unmappable_designs()
    except ValueError:
        return
    raise AssertionError("expected a ValueError once a front has no design left")


if __name__ == "__main__":
    test_unmappable_pair_is_recorded_once_per_equal_core()
    test_unmappable_pair_still_raises_by_default()
    test_unmappable_designs_are_dropped_from_the_fronts()
    print("all passed")
