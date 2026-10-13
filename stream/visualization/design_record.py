"""One measured design as compact, JSON-safe data: its hardware graph, the memory hierarchy of every core design
and the schedule it ran.

The optimizer comparison report shows every Pareto point of every run, and the searches only keep a few full
`StreamCostModelEvaluation`s on disk. This module turns a design into a small record that can be saved at
measurement time (see `SearchBudget.record(..., artifact=...)`) or rebuilt afterwards from what a run saved, so
the report never needs more than plain JSON:

- `topology`: `extract_topology` / `topology_from_connectivity` nodes (laid out, as `design_payload` does) and
  edges;
- `designs`: one memory hierarchy per distinct core design, keyed by a content hash, and `core_design` mapping
  each compute core to its key -- cores often share a design, so each hierarchy is stored once;
- `schedule`: columnar tasks `[row, layer, sub_id, start, end, energy]` per scheduled tile and transfers
  `[row, start, end, layer, energy]` per non-empty link event, `row` indexing `rows` (cores first, then links);
- `energy`: the scme's energy breakdown.

A hierarchy is `{oa, units, area, levels}` with levels innermost first. Each level has a `tier`: its height in
the hierarchy (0 next to the array), the longest chain of levels below it over the operands it serves, so a
level shared by several operands lines up across their columns in a drawing.
"""

import hashlib
import json
import logging
import math
from typing import Any

from stream.visualization.hardware_graph import (
    HardwareTopology,
    core_layers_from_workload,
    extract_topology,
    layout_positions,
    topology_from_connectivity,
)

logger = logging.getLogger(__name__)

OPERAND_ORDER = ("I1", "I2", "O")

# scme attributes of the energy breakdown, as (record key, attribute).
ENERGY_FIELDS = (
    ("cn_onchip", "total_cn_onchip_energy"),
    ("cn_offchip_link", "total_cn_offchip_link_energy"),
    ("cn_offchip_memory", "total_cn_offchip_memory_energy"),
    ("core_to_core_link", "total_core_to_core_link_energy"),
    ("core_to_core_memory", "total_core_to_core_memory_energy"),
    ("eviction_link", "total_eviction_to_offchip_link_energy"),
    ("eviction_memory", "total_eviction_to_offchip_memory_energy"),
    ("sink_output_link", "total_sink_layer_output_offchip_link_energy"),
    ("sink_output_memory", "total_sink_layer_output_offchip_memory_energy"),
)


def _number(value: Any) -> float | int | None:
    """A JSON-safe number: integral values as ints (shorter), non-finite or missing as None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else round(value, 3)


def _key(content: Any) -> str:
    return hashlib.sha1(repr(content).encode()).hexdigest()[:16]


def _assign_tiers(levels: list[dict[str, Any]]) -> None:
    """Set each level's `tier` from the innermost-first order of the levels serving each operand."""
    top_of_operand: dict[str, int] = {}
    for level in levels:
        tier = max((top_of_operand.get(operand, -1) for operand in level["operands"]), default=-1) + 1
        level["tier"] = tier
        for operand in level["operands"]:
            top_of_operand[operand] = tier


def _sorted_operands(operands: Any) -> list[str]:
    names = [str(operand) for operand in operands]
    return sorted(names, key=lambda name: (OPERAND_ORDER.index(name) if name in OPERAND_ORDER else 99, name))


def _port_bandwidths(ports: list[tuple[str, Any]]) -> tuple[Any, Any]:
    """(read, write) bandwidth of a level: the widest port of each direction; `ports` holds (type, bw_max)."""
    read = [bw for kind, bw in ports if "read" in kind and bw is not None]
    write = [bw for kind, bw in ports if "write" in kind and bw is not None]
    return _number(max(read)) if read else None, _number(max(write)) if write else None


def core_hierarchy(core: Any) -> dict[str, Any]:
    """Operational array and memory levels of a built (zigzag) `Core`, innermost level first."""
    oa = core.operational_array
    levels = []
    for level in core.memory_hierarchy.topological_sort():
        instance = level.memory_instance
        port_type = [(str(getattr(port.type, "value", port.type)).lower(), port.bw_max) for port in level.ports]
        read_bw, write_bw = _port_bandwidths(port_type)
        levels.append(
            {
                "name": level.name,
                "size": _number(instance.size),
                "operands": _sorted_operands(level.operands),
                "served": sorted(str(dim) for dim in level.served_dimensions),
                "read_bw": read_bw,
                "write_bw": write_bw,
                "r_cost": _number(getattr(instance, "r_cost", None)),
                "w_cost": _number(getattr(instance, "w_cost", None)),
                "area": _number(getattr(instance, "area", None)),
                "latency": _number(getattr(instance, "latency", None)),
            }
        )
    _assign_tiers(levels)
    area = None
    try:
        area = _number(core.get_area())
    except Exception:  # CACTI-derived; a missing number must not cost the record
        pass
    return {
        "oa": {str(dim): int(size) for dim, size in oa.dimension_sizes.items()},
        "units": int(oa.total_unit_count),
        "area": area,
        "levels": levels,
    }


def core_hierarchy_from_dict(core_dict: dict[str, Any]) -> dict[str, Any]:
    """The same shape read off a core dict (as `core_generator.core_dict_from_params` builds them), without
    constructing the `Core`: CACTI costs and areas are filled in only at construction, so they stay None."""
    levels = []
    for name, memory in core_dict.get("memories", {}).items():
        ports = [(str(port.get("type", "")).lower(), port.get("bandwidth_max")) for port in memory.get("ports", [])]
        read_bw, write_bw = _port_bandwidths(ports)
        levels.append(
            {
                "name": name,
                "size": _number(memory.get("size")),
                "operands": _sorted_operands(memory.get("operands", [])),
                "served": sorted(str(dim) for dim in memory.get("served_dimensions", [])),
                "read_bw": read_bw,
                "write_bw": write_bw,
                "r_cost": _number(memory.get("r_cost")),
                "w_cost": _number(memory.get("w_cost")),
                "area": _number(memory.get("area")),
                "latency": _number(memory.get("latency")),
            }
        )
    _assign_tiers(levels)
    array = core_dict.get("operational_array", {})
    oa = {str(dim): int(size) for dim, size in zip(array.get("dimensions", []), array.get("sizes", []), strict=False)}
    units = 1
    for size in oa.values():
        units *= size
    return {"oa": oa, "units": units, "area": None, "levels": levels}


def hierarchy_key(hierarchy: dict[str, Any]) -> str:
    """Content hash of a hierarchy, for sharing it between the cores that carry the same design."""
    return _key(
        (
            sorted(hierarchy["oa"].items()),
            [
                (lvl["name"], lvl["size"], lvl["operands"], lvl["served"], lvl["read_bw"], lvl["write_bw"])
                for lvl in hierarchy["levels"]
            ],
        )
    )


def topology_payload(topology: HardwareTopology) -> dict[str, Any]:
    """Nodes (with the positions `layout_positions` gives them) and edges, as `design_payload` lays them out."""
    positions = layout_positions(topology)
    cores = []
    for node in topology.nodes:
        x, y = positions.get(node.id, (0.0, 0.0))
        cores.append({**node.to_dict(), "x": float(x), "y": float(y)})
    return {"cores": cores, "edges": [edge.to_dict() for edge in topology.edges]}


def _layer_order(workload: Any) -> list[int]:
    order: list[int] = []
    for node in workload.topological_sort():
        if node.id not in order:
            order.append(node.id)
    return order


def schedule_record(scme: Any) -> dict[str, Any]:
    """The schedule of `scme` in columnar form (see the module docstring)."""
    tasks_raw = [
        node for node in scme.workload.node_list if getattr(node, "start", None) is not None and node.end is not None
    ]
    core_ids = sorted({node.chosen_core_allocation for node in tasks_raw})
    links: dict[int, Any] = {}
    for pairs in scme.accelerator.communication_manager.all_pair_links.values():
        for pair in pairs:
            for link in pair:
                if any(event.end > event.start for event in link.events):
                    links[id(link)] = link
    link_list = sorted(links.values(), key=lambda link: link.get_name_for_schedule_plot())
    rows = [f"Core {core_id}" for core_id in core_ids] + [link.get_name_for_schedule_plot() for link in link_list]
    row_of_core = {core_id: index for index, core_id in enumerate(core_ids)}
    layer_names: dict[int, str] = {}
    tasks = []
    for node in sorted(tasks_raw, key=lambda n: (n.start, n.id, n.sub_id)):
        layer_names.setdefault(node.id, getattr(node, "short_name", f"layer {node.id}"))
        tasks.append(
            [
                row_of_core[node.chosen_core_allocation],
                int(node.id),
                int(node.sub_id),
                _number(node.start),
                _number(node.end),
                _number(getattr(node, "onchip_energy", None)),
            ]
        )
    transfers = []
    for index, link in enumerate(link_list):
        for event in link.events:
            if event.end <= event.start:
                continue
            origin = getattr(event.tensor, "origin", None)
            transfers.append(
                [
                    len(core_ids) + index,
                    _number(event.start),
                    _number(event.end),
                    int(origin.id) if origin is not None else None,
                    _number(event.energy),
                    "b" if event.type == "block" else "t",
                ]
            )
    return {
        "rows": rows,
        "nb_core_rows": len(core_ids),
        "layers": {str(layer): name for layer, name in layer_names.items()},
        "tasks": tasks,
        "transfers": transfers,
        "latency": _number(scme.latency),
    }


def format_tiling(tiling_config: dict[Any, Any] | None) -> dict[str, dict[str, str]] | None:
    """`{layer: {intra_core_tiling, inter_core_tiling}}` as short strings, e.g. `{"3": {"intra": "OX7",
    "inter": "K2"}}`."""
    if not tiling_config:
        return None

    def text(factors: Any) -> str:
        return " ".join(f"{dim}{size}" for dim, size in factors or []) or "-"

    return {
        str(layer): {
            "intra": text(config.get("intra_core_tiling")),
            "inter": text(config.get("inter_core_tiling")),
        }
        for layer, config in tiling_config.items()
        if isinstance(config, dict)
    }


def energy_breakdown(scme: Any) -> dict[str, Any]:
    return {key: _number(getattr(scme, attribute, None)) for key, attribute in ENERGY_FIELDS}


def design_record(scme: Any, **extra: Any) -> dict[str, Any]:
    """Everything the report shows about a measured design, from its `StreamCostModelEvaluation`. `extra` (e.g.
    a tiling configuration) is kept, with anything JSON cannot hold turned into strings."""
    accelerator, workload = scme.accelerator, scme.workload
    layer_names = {node.id: node.short_name for node in workload.node_list if hasattr(node, "short_name")}
    offchip = accelerator.offchip_core_id
    designs: dict[str, Any] = {}
    core_design: dict[str, str] = {}
    for core in accelerator.core_list:
        if core.id == offchip or getattr(core, "type", "compute") == "memory":
            continue
        try:
            hierarchy = core_hierarchy(core)
        except Exception:
            logger.warning(f"design_record: no memory hierarchy for core {core.id}.", exc_info=True)
            continue
        key = hierarchy_key(hierarchy)
        designs.setdefault(key, hierarchy)
        core_design[str(core.id)] = key
    topology = extract_topology(
        accelerator,
        core_layers=core_layers_from_workload(workload),
        layer_names=layer_names,
        design_keys={int(core): key for core, key in core_design.items()},
        layer_order=_layer_order(workload),
    )
    return {
        "topology": topology_payload(topology),
        "designs": designs,
        "core_design": core_design,
        "schedule": schedule_record(scme),
        "energy": energy_breakdown(scme),
        **({"extra": json.loads(json.dumps(extra, default=str))} if extra else {}),
    }


def rolled_design_record(record: dict[str, Any], layer_names: dict[int, str] | None = None) -> dict[str, Any]:
    """The hardware part of a design record from a rolled schedule record (`pareto_schedules.pickle`), which
    carries its core dicts and connectivity but no timing: `schedule` and `energy` are None."""
    designs: dict[str, Any] = {}
    core_design: dict[str, str] = {}
    for core_id, core_dict in record["core_dict_of_core"].items():
        hierarchy = core_hierarchy_from_dict(core_dict)
        key = hierarchy_key(hierarchy)
        designs.setdefault(key, hierarchy)
        core_design[str(core_id)] = key
    topology = topology_from_connectivity(
        name=str(record["id"]),
        node_core_ids=record["node_core_ids"],
        offchip_core_id=record["nb_cores"],
        bus_connection=record.get("bus_connection") or {},
        link_connections=record.get("link_connections") or [],
        design_keys={int(core): key for core, key in core_design.items()},
        layer_names=layer_names,
        layer_order=sorted(record["node_core_ids"]),
    )
    return {
        "topology": topology_payload(topology),
        "designs": designs,
        "core_design": core_design,
        "schedule": None,
        "energy": None,
        "extra": {"fold": {key: record.get(key) for key in ("fold_objective", "fold_tolerance", "fold_overlap")}},
    }
