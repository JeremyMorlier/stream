"""Programmatic generation of Stream/ZigZag hardware "core" definitions.

A core YAML (e.g. ``stream/inputs/examples/hardware/cores/tpu_like.yaml``) is validated
and parsed by exactly the same classes Stream uses when loading a core from disk
(``stream.parser.core_validator.CoreValidator`` and ``zigzag.parser.accelerator_factory.AcceleratorFactory``).
This module builds the equivalent nested dict in Python, runs it through that same
validation/factory pipeline, and returns a ready-to-use ``stream.hardware.architecture.core.Core``.

Generated cores follow the standard three-memory-level pattern seen in ``tpu_like.yaml`` /
``meta_prototype.yaml``: a private register file (RF) per operand serving none or one of the
"array" dimensions (D1/D2), a shared SRAM serving the array dimensions, and a shared "upper"
memory that additionally serves the outer dimension (D3), representing a buffer that feeds a
third spatial unrolling dimension on top of the 2D PE array.
"""

import random
import tempfile
from typing import Any

import yaml
from zigzag.cost_model.cost_model import CostModelEvaluationABC
from zigzag.parser.accelerator_factory import AcceleratorFactory

from stream.hardware.architecture.core import Core
from stream.parser.core_validator import CoreValidator

OPERANDS = ("I1", "I2", "O")


def _direction_allocations(operand: str) -> tuple[list[str], list[str]]:
    """Return (read_directions, write_directions) allocation strings for one operand.

    Output operands flow both ways (tl/th read, fh/fl write) since partial sums are both
    accumulated from the array and refreshed from the level above; inputs only ever flow
    down from above (tl read) and are only ever written from above (fh write).
    """
    if operand == "O":
        return ([f"{operand}, tl", f"{operand}, th"], [f"{operand}, fh", f"{operand}, fl"])
    return ([f"{operand}, tl"], [f"{operand}, fh"])


def _build_ports(
    operands: list[str], read_port: str, write_port: str, bandwidth_min: int, bandwidth_max: int
) -> list[dict[str, Any]]:
    """Build one shared read port + one shared write port covering every operand's directions."""
    read_alloc: list[str] = []
    write_alloc: list[str] = []
    for operand in operands:
        reads, writes = _direction_allocations(operand)
        read_alloc.extend(reads)
        write_alloc.extend(writes)
    return [
        {
            "name": read_port,
            "type": "read",
            "bandwidth_min": bandwidth_min,
            "bandwidth_max": bandwidth_max,
            "allocation": read_alloc,
        },
        {
            "name": write_port,
            "type": "write",
            "bandwidth_min": bandwidth_min,
            "bandwidth_max": bandwidth_max,
            "allocation": write_alloc,
        },
    ]


def _align_to_byte(size: int, min_size: int = 0) -> int:
    """CACTI (invoked via `auto_cost_extraction`) requires memory sizes to be a multiple of 8 bits, and a memory
    can't move more bits per cycle than it can hold in total -- so `min_size` (typically the memory's own
    bandwidth) is also enforced as a floor; CACTI itself rejects some of these degenerate configs (e.g. an
    8-bit memory with 128 bit/cycle bandwidth), but the constraint is a real physical one regardless of
    whether CACTI happens to catch it. Preserves the "0 means omit this memory level" convention -- only a
    nonzero size snaps, floored at 8 (or `min_size`, rounded up to a multiple of 8) so a small nonzero draw
    never rounds away to 0."""
    if size <= 0:
        return 0
    floor = max(8, min_size)
    return max(floor, round(size / 8) * 8)


def build_rf_memory_dict(
    operand: str, size: int, bandwidth: int, served_dimensions: list[str]
) -> dict[str, Any] | None:
    """Deterministic counterpart to `_rf_memory`: builds an RF memory dict from explicit values instead of
    sampling them, so a caller (e.g. a genetic algorithm) can decode its own chosen values into a valid memory
    dict without going through `random.Random`. Returns None if `size` rounds to 0 (no such level).

    `r_cost`/`w_cost`/`area` are intentionally omitted: `auto_cost_extraction` makes ZigZag infer them (and the
    memory's real silicon area) via CACTI when the core is actually built, rather than us guessing plausible
    numbers."""
    size = _align_to_byte(size, min_size=bandwidth)
    if size == 0:
        return None
    return {
        "size": size,
        "latency": 1,
        "mem_type": "rf",
        "auto_cost_extraction": True,
        "operands": [operand],
        "ports": _build_ports([operand], "r_port_1", "w_port_1", bandwidth, bandwidth),
        "served_dimensions": served_dimensions,
    }


def build_shared_memory_dict(
    operands: list[str],
    size: int,
    bandwidth_min: int,
    bandwidth_max: int,
    served_dimensions: list[str],
) -> dict[str, Any] | None:
    """Deterministic counterpart to `_shared_memory`: builds a shared memory dict from explicit values instead
    of sampling them. Returns None if `size` rounds to 0 (no such level). See `build_rf_memory_dict` for why
    `r_cost`/`w_cost`/`area` are omitted in favor of `auto_cost_extraction`, and why `size` is floored at the
    memory's own bandwidth."""
    size = _align_to_byte(size, min_size=max(bandwidth_min, bandwidth_max))
    if size == 0:
        return None
    return {
        "size": size,
        "latency": 1,
        "mem_type": "sram",
        "auto_cost_extraction": True,
        "operands": operands,
        "ports": _build_ports(operands, "r_port_1", "w_port_1", bandwidth_min, bandwidth_max),
        "served_dimensions": served_dimensions,
    }


def _rf_memory(rng: random.Random, operand: str, size_range: tuple[int, int],
               bandwidth_choices: tuple[int, ...], served_dimensions: list[str]) -> dict[str, Any] | None:
    """Build an RF memory dict, or return None if the randomly drawn size is 0 (no such level)."""
    size = rng.randint(*size_range)
    bandwidth = rng.choice(bandwidth_choices)
    return build_rf_memory_dict(operand, size, bandwidth, served_dimensions)


def _shared_memory(rng: random.Random, operands: list[str], size_range: tuple[int, int],
                    bandwidth_choices: tuple[int, ...], served_dimensions: list[str]) -> dict[str, Any] | None:
    """Build a shared memory dict, or return None if the randomly drawn size is 0 (no such level)."""
    size = rng.randint(*size_range)
    if size == 0:
        return None
    bandwidth_min = rng.choice(bandwidth_choices)
    bandwidth_max = rng.choice([b for b in bandwidth_choices if b >= bandwidth_min])
    return build_shared_memory_dict(operands, size, bandwidth_min, bandwidth_max, served_dimensions)


def core_dict_from_params(
    name: str,
    *,
    operands: tuple[str, ...],
    oa_dims: tuple[str, ...],
    oa_sizes: list[int],
    unit_energy: float,
    rf_memories: dict[str, dict[str, Any] | None],
    sram_memory: dict[str, Any] | None,
    upper_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deterministic core-dict assembly shared by `random_core_dict` and any caller (e.g. a genetic algorithm)
    that has already built its own memory dicts via `build_rf_memory_dict`/`build_shared_memory_dict` and just
    needs the final schema-shaped core dict, including the "every operand needs a memory level somewhere"
    check `random_core_dict` also enforces."""
    memories: dict[str, dict[str, Any]] = {}
    for operand in operands:
        rf = rf_memories.get(operand)
        if rf is not None:
            memories[f"rf_{operand}"] = rf
    if sram_memory is not None:
        memories["sram_shared"] = sram_memory
    if upper_memory is not None:
        memories["upper_shared"] = upper_memory

    covered_operands = {operand for memory in memories.values() for operand in memory["operands"]}
    missing_operands = set(operands) - covered_operands
    if missing_operands:
        raise ValueError(
            f"Generated core '{name}' has no non-zero-size memory for operand(s) {sorted(missing_operands)} -- "
            "widen at least one size_range's lower bound above 0 for that operand."
        )

    return {
        "name": name,
        "type": "compute",
        "memories": memories,
        "operational_array": {
            "unit_energy": unit_energy,
            "unit_area": 1,
            "dimensions": list(oa_dims),
            "sizes": oa_sizes,
        },
    }


def random_core_dict(
    rng: random.Random,
    name: str = "generated_core",
    *,
    operands: tuple[str, ...] = OPERANDS,
    oa_dims: tuple[str, ...] = ("D1", "D2", "D3"),
    oa_size_ranges: tuple[tuple[int, int], ...] = ((2, 32), (2, 32), (1, 8)),
    rf_size_range: tuple[int, int] = (8, 1024),
    # Floor of 16 bit/cycle: an RF narrower than the operand's own precision can't even carry
    # one word per access, which makes spatial unrolling on that operand infeasible (unroll = 0).
    rf_bandwidth_choices: tuple[int, ...] = (16, 32, 64, 128),
    sram_size_range: tuple[int, int] = (2**19, 2**24),
    sram_bandwidth_choices: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
    upper_size_range: tuple[int, int] = (2**23, 2**27),
    upper_bandwidth_choices: tuple[int, ...] = (256, 512, 1024, 2048, 4096),
) -> dict[str, Any]:
    """Build a random but schema-valid core dict.

    Layout (bottom to top -- ``MemoryHierarchy.add_memory`` requires memories be added in this
    order, and dict/YAML key order is preserved through parsing): one private RF per operand
    serving a random subset of the "array" dimensions (all ``oa_dims`` but the last one, e.g.
    D1/D2), a shared SRAM serving those same array dimensions, and a shared upper-level memory
    serving *every* ``oa_dims`` entry including the outer one (e.g. D3) -- the buffer that feeds
    the extra spatial unrolling dimension on top of the 2D array.

    Any of ``rf_size_range``/``sram_size_range``/``upper_size_range`` may include 0 (e.g.
    ``(0, 1024)``): whenever a size of 0 is drawn, that memory level is a no-op for this core and
    is simply left out of the generated dict, instead of being emitted as a zero-size memory. This
    lets the generator explore architectures that skip a level entirely (e.g. no private RF, or no
    intermediate SRAM). Each operand must still end up with at least one non-zero memory level
    somewhere in the hierarchy -- a ``ValueError`` is raised otherwise.
    """
    if len(oa_dims) < 2:
        raise ValueError("Need at least 2 operational-array dimensions (array dims + 1 outer dim).")
    if len(oa_dims) != len(oa_size_ranges):
        raise ValueError("oa_dims and oa_size_ranges must have the same length.")

    array_dims = list(oa_dims[:-1])

    rf_memories: dict[str, dict[str, Any] | None] = {}
    for operand in operands:
        served = rng.choice([[], *[[d] for d in array_dims], array_dims])
        rf_memories[operand] = _rf_memory(rng, operand, rf_size_range, rf_bandwidth_choices, served)

    sram_memory = _shared_memory(rng, list(operands), sram_size_range, sram_bandwidth_choices, array_dims)
    upper_memory = _shared_memory(rng, list(operands), upper_size_range, upper_bandwidth_choices, list(oa_dims))

    sizes = [rng.randint(*oa_size_ranges[i]) for i in range(len(oa_dims))]

    return core_dict_from_params(
        name,
        operands=operands,
        oa_dims=oa_dims,
        oa_sizes=sizes,
        unit_energy=round(rng.uniform(0.01, 0.1), 4),
        rf_memories=rf_memories,
        sram_memory=sram_memory,
        upper_memory=upper_memory,
    )


def build_core_from_dict(core_dict: dict[str, Any], core_id: int = 0) -> Core:
    """Validate a core dict with Stream's own ``CoreValidator`` and build a real ``Core`` object from it."""
    validator = CoreValidator(core_dict)
    if not validator.validate():
        raise ValueError(f"Generated core '{core_dict.get('name')}' failed schema validation.")
    factory = AcceleratorFactory(validator.normalized_data)
    zigzag_core = factory.create(core_id=core_id)
    return Core.from_zigzag_core(zigzag_core)


def generate_random_core(
    rng: random.Random, core_id: int = 0, name: str = "generated_core", **kwargs: Any
) -> tuple[Core, dict[str, Any]]:
    """Generate a random core dict, validate + build it, and return both the ``Core`` and its source dict."""
    core_dict = random_core_dict(rng, name=name, **kwargs)
    core = build_core_from_dict(core_dict, core_id=core_id)
    return core, core_dict


def _default_workload_and_mapping(
    oa_dims: list[str], sizes: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A minimal single-layer Conv workload + explicit spatial mapping that fully unrolls the given OA.

    Supports 2D (D1/D2) and 3D (D1/D2/D3) operational arrays: K is unrolled on D1 (output
    channels), C on D2 (reduction/input channels), and -- for the 3D case -- OY on D3 (an output
    spatial dimension), matching the "RF+SRAM serve D1/D2, upper memory also serves D3" layout.
    """
    n = len(oa_dims)
    if n == 2:
        oa_to_layer_dim = {"D1": "K", "D2": "C"}
        loop_dims, loop_sizes = ["B", "K", "C", "OY", "OX"], [1, sizes[0], sizes[1], 4, 4]
    elif n == 3:
        oa_to_layer_dim = {"D1": "K", "D2": "C", "D3": "OY"}
        loop_dims, loop_sizes = ["B", "K", "C", "OY", "OX"], [1, sizes[0], sizes[1], sizes[2], 4]
    else:
        raise ValueError("Default workload/mapping generation only supports 2 or 3 operational-array dimensions.")

    workload = [
        {
            "id": 0,
            "operator_type": "Conv",
            "equation": "O[b][k][oy][ox]+=W[k][c]*I[b][c][oy][ox]",
            "loop_dims": loop_dims,
            "loop_sizes": loop_sizes,
            "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        }
    ]
    spatial_mapping = {oa_dim: [f"{oa_to_layer_dim[oa_dim]}, {sizes[i]}"] for i, oa_dim in enumerate(oa_dims)}
    mapping = [
        {
            "name": "default",
            "spatial_mapping": spatial_mapping,
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        }
    ]
    return workload, mapping


def evaluate_core(
    core_dict: dict[str, Any],
    *,
    workload: list[dict[str, Any]] | None = None,
    mapping: list[dict[str, Any]] | None = None,
    dump_folder: str | None = None,
) -> tuple[float, float, list[tuple[CostModelEvaluationABC, Any]]]:
    """Run the generated core through Stream/ZigZag's actual cost-model evaluation.

    Dumps the core dict to a temp accelerator yaml (zigzag's parser only accepts a yaml path) and
    a matching mapping yaml, then calls ``zigzag.api.get_hardware_performance_zigzag`` -- the same
    entry point ``main_stream_*.py`` scripts build on top of -- on a single-layer workload sized to
    match the core's operational array so it fully exercises every generated dimension.
    """
    from zigzag.api import get_hardware_performance_zigzag  # local import: heavy, only needed here

    oa_dims: list[str] = core_dict["operational_array"]["dimensions"]
    sizes: list[int] = core_dict["operational_array"]["sizes"]
    if workload is None or mapping is None:
        default_workload, default_mapping = _default_workload_and_mapping(oa_dims, sizes)
        workload = workload if workload is not None else default_workload
        mapping = mapping if mapping is not None else default_mapping

    with tempfile.TemporaryDirectory() as tmp_dir:
        accelerator_path = f"{tmp_dir}/{core_dict['name']}.yaml"
        mapping_path = f"{tmp_dir}/mapping.yaml"
        # Plain zigzag's AcceleratorValidator (used by get_hardware_performance_zigzag) has a strict
        # schema with no "type" field -- that field only exists in Stream's CoreValidator subclass.
        zigzag_accelerator_dict = {k: v for k, v in core_dict.items() if k != "type"}
        with open(accelerator_path, "w") as f:
            yaml.safe_dump(zigzag_accelerator_dict, f, sort_keys=False)
        with open(mapping_path, "w") as f:
            yaml.safe_dump(mapping, f, sort_keys=False)

        energy, latency, cmes = get_hardware_performance_zigzag(
            workload=workload,
            accelerator=accelerator_path,
            mapping=mapping_path,
            dump_folder=dump_folder or f"{tmp_dir}/outputs",
            loma_show_progress_bar=False,
        )
    return energy, latency, cmes
