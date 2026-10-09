"""End a schedule search: pick its best schedule, measure it once more for real, and report the whole front."""

import logging
from typing import Any

from stream.stages.generation.schedule_evaluation import ScheduleSearchStage, strip_unpicklable
from stream.stages.generation.schedule_space import ScheduleSearchState
from stream.stages.stage import StageCallable

logger = logging.getLogger(__name__)


class ScheduleSelectionStage(ScheduleSearchStage):
    """Takes the archive the schedule searches before it built (`schedule_search`), picks the best schedule by
    `sort_key` (`latency`, `energy`, `area` or `edp`), rebuilds its accelerator -- writing `accelerator.yaml` next
    to the tiled workload -- and runs the stages after it on that accelerator.

    It yields what they yield, with this `extra_info` instead of theirs:
        - `schedule_ranks`, `best_schedule`: the winning record (its ranks alone are ambiguous once folded
          schedules share the front: they inherit their source's ranks but run on a different accelerator);
        - `pareto_schedules`: every Pareto record as plain data, each with what is needed to rebuild it;
        - `all_schedules`: every measured record;
        - `search_stats`: the searches' counters;
        - `schedule_report`: the plain-data context the visualization stages draw from (see
          `stream.stages.visualization`).
    The full Pareto set is reported as records rather than one `StreamCostModelEvaluation` each, since holding
    those does not scale with workload size.
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        schedule_search: ScheduleSearchState,
        sort_key: str = "latency",
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.schedule_search = schedule_search
        self.sort_key = sort_key

    def run(self):
        state = self.schedule_search
        space = state.space
        best_record = state.archive.sorted_by(self.sort_key)[0][1]
        logger.info(
            f"ScheduleSelectionStage: best schedule by {self.sort_key} is {best_record['id']} "
            f"(latency={best_record['latency']:.6g}, energy={best_record['energy']:.6g}, "
            f"area={best_record['area']:.6g}, cores={best_record['nb_cores']})."
        )
        extra_info = {
            "schedule_ranks": best_record["ranks"],
            "best_schedule": strip_unpicklable(best_record),
            "pareto_schedules": [space.payload(record) for _point, record in state.archive.entries],
            "all_schedules": [strip_unpicklable(record) for record in state.records],
            "search_stats": state.stats,
            "schedule_report": {**state.report, "sort_key": self.sort_key, "area_of_design": dict(space.costs.area)},
        }
        # Re-measured once so the winning accelerator is the one written to accelerator.yaml.
        workload, accelerator, cost_lut = space.assemble_record(best_record, save_yaml=True)
        kwargs = self.kwargs.copy()
        kwargs.update(workload=workload, accelerator=accelerator, cost_lut=cost_lut)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        for scme, _sub_extra_info in sub_stage.run():
            yield scme, extra_info
