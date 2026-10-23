"""Write a schedule search's measured schedules and Pareto set to disk."""

import csv
import logging
import os
from typing import Any

from zigzag.utils import pickle_save

from stream.stages.generation.schedule_evaluation import format_ranks
from stream.stages.visualization.report_stage import ExtraInfoReportStage

logger = logging.getLogger(__name__)

SCHEDULE_COLUMNS = (
    "schedule_index",
    "id",
    "kind",
    "ranks",
    "nb_cores",
    "design_keys",
    "latency",
    "energy",
    "area",
    "interconnect_area",
    "latency_bound",
    "energy_bound",
    "pareto",
)
FOLD_COLUMNS = ("fold_tolerance", "fold_objective", "fold_overlap", "source_id")
FOLD_LOG_COLUMNS = ("source_id", "tolerance", "objective", "overlap_mode", "nb_cores", "nb_merges", "status")


class ScheduleResultsWriterStage(ExtraInfoReportStage):
    """Writes, into the search's report directory (`schedule_search/` or `rolled_search/`):

    - `schedules.csv`: every *measured* schedule with its ranks, measured `(latency, energy, area)`, the bounds it
      was admitted on, and whether it ended up on the front. Pruned combinations are not listed (there can be
      arbitrarily many); they are counted in the search stats. After a rolling stage, also the fold parameters
      that produced each rolled schedule.
    - `pareto_schedules.pickle`: `{stats, decision_classes, front_sizes, pareto, all}`, where each pareto entry
      carries what is needed to rebuild that accelerator.
    - `fold_log.csv`, after a rolling stage: every fold considered and what happened to it."""

    def report(self, scme: Any, extra_info: dict[str, Any]) -> None:
        report = extra_info["schedule_report"]
        out_dir = report["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        rolled = report.get("fold_log") is not None
        records = extra_info["all_schedules"]
        pareto = extra_info["pareto_schedules"]
        pareto_ids = {record["id"] for record in pareto}

        csv_path = os.path.join(out_dir, "schedules.csv")
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(SCHEDULE_COLUMNS + (FOLD_COLUMNS if rolled else ()))
            for index, record in enumerate(records):
                writer.writerow(_csv_row(index, record, pareto_ids, rolled))
        summary_path = os.path.join(out_dir, "pareto_schedules.pickle")
        pickle_save(
            {
                "stats": extra_info["search_stats"],
                "decision_classes": report["decision_classes"],
                "front_sizes": report["front_sizes"],
                "pareto": pareto,
                "all": records,
            },
            summary_path,
        )  # type: ignore
        logger.info(
            f"ScheduleResultsWriterStage: saved {len(records)} measured schedule(s) to {csv_path} and "
            f"{len(pareto)} pareto schedule(s) to {summary_path}."
        )
        if rolled:
            self._save_fold_log(report["fold_log"], out_dir)

    @staticmethod
    def _save_fold_log(fold_log: list[dict[str, Any]], out_dir: str) -> None:
        path = os.path.join(out_dir, "fold_log.csv")
        with open(path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(FOLD_LOG_COLUMNS)
            for entry in fold_log:
                writer.writerow([entry[column] for column in FOLD_LOG_COLUMNS])
        logger.info(f"ScheduleResultsWriterStage: saved {len(fold_log)} fold attempt(s) to {path}.")


def _csv_row(index: int, record: dict[str, Any], pareto_ids: set[str], rolled: bool) -> list[Any]:
    row = [
        index,
        record["id"],
        record["kind"],
        format_ranks(record["ranks"]),
        record["nb_cores"],
        "-".join(record["design_keys"]),
        record["latency"],
        record["energy"],
        record["area"],
        record.get("interconnect_area", ""),
        record["latency_bound"],
        record["energy_bound"],
        record["id"] in pareto_ids,
    ]
    if rolled:
        row += [
            record.get("fold_tolerance", ""),
            record.get("fold_objective", ""),
            record.get("fold_overlap", ""),
            record.get("source_id", ""),
        ]
    return row
