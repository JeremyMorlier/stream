"""Draw the hardware graphs behind a schedule search's Pareto front."""

import logging
import os
from typing import Any

from stream.stages.generation.schedule_evaluation import OBJECTIVES, core_dict_key, sort_records
from stream.stages.visualization.report_stage import ExtraInfoReportStage
from stream.visualization.hardware_graph import HardwareTopology, plot_hardware_graph, topology_from_connectivity
from stream.visualization.topology_report import design_payload, write_topology_report

logger = logging.getLogger(__name__)


class TopologyReportStage(ExtraInfoReportStage):
    """Draws the hardware topology of a sample of the Pareto schedules -- `topologies/topology_<k>_<id>.png` --
    and writes `topologies.html` (interactive) and `topologies.json` in the search's report directory.

    Only `max_topology_figures` designs (a kwarg, default 12) get a figure: the per-objective extremes first,
    then a stride through the front. The unrolled topology does not depend on the designs, so past a handful the
    figures of unrolled schedules differ only in their labels. Topologies are built from the connectivity in the
    report rather than by assembling an `Accelerator`, which would construct every `Core` and can invoke CACTI
    just to draw a picture."""

    def report(self, scme: Any, extra_info: dict[str, Any]) -> None:
        report = extra_info["schedule_report"]
        out_dir = report["out_dir"]
        figures_dir = os.path.join(out_dir, "topologies")
        os.makedirs(figures_dir, exist_ok=True)
        pareto = extra_info["pareto_schedules"]
        selected = self._select_for_figures(pareto, report["sort_key"], self.kwargs.get("max_topology_figures", 12))
        payloads: list[dict[str, Any]] = []
        for index, record in enumerate(selected):
            try:
                topology = self._topology_of_record(record, report)
            except Exception:
                logger.warning(f"TopologyReportStage: could not build a topology for {record['id']}.", exc_info=True)
                continue
            plot_hardware_graph(
                topology,
                os.path.join(figures_dir, f"topology_{index}_{record['id']}.png"),
                title=(
                    f"{record['id']}: {record['nb_cores']} core(s), latency={record['latency']:.4g}, "
                    f"energy={record['energy']:.4g}, area={record['area']:.4g}"
                ),
            )
            payloads.append(design_payload(record, topology))
        logger.info(
            f"TopologyReportStage: saved {len(selected)} of {len(pareto)} pareto topology figure(s) to {figures_dir}."
        )
        if payloads:
            write_topology_report(
                payloads,
                os.path.join(out_dir, "topologies.html"),
                json_path=os.path.join(out_dir, "topologies.json"),
                title=f"{report['accelerator_name']}: schedule / topology trade-off",
                measured=extra_info["all_schedules"],
            )

    @staticmethod
    def _select_for_figures(pareto: list[dict[str, Any]], sort_key: str, max_figures: int) -> list[dict[str, Any]]:
        """Which Pareto records get a figure: the per-objective extremes first, then a stride through the front
        so the sample spans it instead of clustering at one end."""
        records = sort_records(pareto, sort_key)
        if len(records) <= max_figures:
            return records
        chosen: dict[str, dict[str, Any]] = {}
        for objective in OBJECTIVES:
            extreme = sort_records(pareto, objective)[0]
            chosen[extreme["id"]] = extreme
        remaining = [record for record in records if record["id"] not in chosen]
        slots = max(0, max_figures - len(chosen))
        if slots and remaining:
            stride = max(1, len(remaining) // slots)
            for record in remaining[::stride][:slots]:
                chosen[record["id"]] = record
        return [record for record in records if record["id"] in chosen]

    @staticmethod
    def _topology_of_record(record: dict[str, Any], report: dict[str, Any]) -> HardwareTopology:
        """The hardware graph a Pareto record describes. A rolled design has its own -- fewer cores, merged
        traffic gone and new links in its place -- carried on the record; an unrolled one runs on the search's
        unrolled topology with the designs its `node_core_dicts` give."""
        area_of_design = report["area_of_design"]
        if record.get("kind") == "rolled":
            design_key_of_core = record["design_key_of_core"]
            node_core_ids = record["node_core_ids"]
            offchip_core_id = record["nb_cores"]
            bus_connection = record["bus_connection"]
            link_connections = record["link_connections"]
        else:
            node_core_ids = report["node_core_ids"]
            design_key_of_core = {
                core_id: core_dict_key(record["node_core_dicts"][node_id])
                for node_id, core_ids in node_core_ids.items()
                for core_id in core_ids
            }
            offchip_core_id = report["offchip_core_id"]
            bus_connection = report["bus_connection"]
            link_connections = report["link_connections"]
        return topology_from_connectivity(
            name=f"{report['accelerator_name']} [{record['id']}]",
            node_core_ids=node_core_ids,
            offchip_core_id=offchip_core_id,
            bus_connection=bus_connection,
            link_connections=link_connections,
            design_keys=design_key_of_core,
            areas={core: area_of_design[key] for core, key in design_key_of_core.items()},
            layer_names=report["layer_names"],
            layer_order=report["layer_order"],
        )
