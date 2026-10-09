"""Plot a schedule search's `(latency, energy, area)` trade-off."""

import os
from typing import Any

from stream.stages.visualization.report_stage import ExtraInfoReportStage
from stream.visualization.schedule_tradeoff import plot_schedule_tradeoffs


class ScheduleTradeoffPlotStage(ExtraInfoReportStage):
    """Draws `schedule_tradeoffs.png` in the search's report directory: the three pairwise projections of every
    measured schedule plus a 3D view, with the Pareto set highlighted (see
    `stream.visualization.schedule_tradeoff`)."""

    def report(self, scme: Any, extra_info: dict[str, Any]) -> None:
        plot_schedule_tradeoffs(
            extra_info["all_schedules"],
            extra_info["pareto_schedules"],
            os.path.join(extra_info["schedule_report"]["out_dir"], "schedule_tradeoffs.png"),
        )
