"""Smoke test for `stream.hardware.architecture.core_generator`.

Generates a batch of random 3D cores (RF per operand serving D1/D2, shared SRAM serving
D1/D2, shared upper memory additionally serving D3) plus a smaller batch of 2D cores, checks
that every one of them validates against Stream's own `CoreValidator` schema and builds into a
real `Core` object with a consistent memory hierarchy, and finally runs a handful of them
through Stream/ZigZag's actual hardware cost-model evaluation (`evaluate_core`) to confirm the
generated architectures are not just schema-valid but actually estimable.

Run directly with `python test_generate_core.py`, or collect with `pytest test_generate_core.py`.
"""

import logging
import os
import random

import yaml

from stream.hardware.architecture.core_generator import (
    build_core_from_dict,
    evaluate_core,
    random_core_dict,
)

logging.basicConfig(level=logging.WARNING)

N_SCHEMA_TRIALS = 5
N_EVAL_TRIALS = 2
BASE_SEED = 1234


output_dir = "outputs/test_generate_core/"
os.makedirs(output_dir, exist_ok=True)


def test_random_3d_cores_validate_and_build():
    """N_SCHEMA_TRIALS random 3D (D1/D2/D3) cores must all validate and build without error."""
    failures: list[str] = []
    for i in range(N_SCHEMA_TRIALS):
        rng = random.Random(BASE_SEED + i)
        core_dict = random_core_dict(rng, name=f"gen_3d_{i}")
        try:
            core = build_core_from_dict(core_dict, core_id=i)
            # Export core to yaml
            with open(f"{output_dir}/core_{i}.yaml", "w") as f:
                yaml.dump(core_dict, f, sort_keys=False)
            _check_core_sanity(core, core_dict)
        except Exception as e:  # noqa: BLE001 -- want to collect every failure, not stop at the first
            failures.append(f"trial {i} (seed {BASE_SEED + i}): {e}\n  core_dict={core_dict}")

    if failures:
        raise AssertionError(f"{len(failures)}/{N_SCHEMA_TRIALS} random 3D cores failed:\n" + "\n".join(failures))
    print(f"OK: {N_SCHEMA_TRIALS}/{N_SCHEMA_TRIALS} random 3D cores validated and built successfully.")


def test_random_2d_cores_validate_and_build():
    """Same check, but for a plain 2D (D1/D2) array -- confirms the generator degrades cleanly."""
    failures: list[str] = []
    for i in range(N_SCHEMA_TRIALS):
        rng = random.Random(BASE_SEED + 1000 + i)
        core_dict = random_core_dict(rng, name=f"gen_2d_{i}", oa_dims=("D1", "D2"), oa_size_ranges=((2, 32), (2, 32)))
        try:
            core = build_core_from_dict(core_dict, core_id=i)
            _check_core_sanity(core, core_dict)
        except Exception as e:  # noqa: BLE001
            failures.append(f"trial {i} (seed {BASE_SEED + 1000 + i}): {e}\n  core_dict={core_dict}")

    if failures:
        raise AssertionError(f"{len(failures)}/{N_SCHEMA_TRIALS} random 2D cores failed:\n" + "\n".join(failures))
    print(f"OK: {N_SCHEMA_TRIALS}/{N_SCHEMA_TRIALS} random 2D cores validated and built successfully.")


def _check_core_sanity(core, core_dict) -> None:
    """Structural checks beyond "it didn't throw": memory hierarchy is acyclic and complete."""
    # A cyclic hierarchy would raise NetworkXUnfeasible here.
    core.memory_hierarchy.topological_sort()

    oa_dims = set(core_dict["operational_array"]["dimensions"])
    if "upper_shared" in core_dict["memories"]:
        upper_served = set(core_dict["memories"]["upper_shared"]["served_dimensions"])
        assert upper_served == oa_dims, f"upper_shared must serve every OA dim, got {upper_served} vs {oa_dims}"

    array_dims = oa_dims - {core_dict["operational_array"]["dimensions"][-1]}
    for operand in ("I1", "I2", "O"):
        rf_name = f"rf_{operand}"
        if rf_name not in core_dict["memories"]:
            continue  # a size-0 draw skips this level entirely -- see test_zero_size_memories_are_omitted
        rf_served = set(core_dict["memories"][rf_name]["served_dimensions"])
        assert rf_served <= array_dims, f"{rf_name} must only serve array dims, got {rf_served}"

    # Every operand must resolve to at least one memory level.
    assert len(core.mem_size_dict) == 3, f"expected 3 operands with memory levels, got {core.mem_size_dict.keys()}"


def test_zero_size_memories_are_omitted():
    """A size_range that can draw 0 must skip that memory level instead of emitting a zero-size one."""
    for i in range(N_SCHEMA_TRIALS):
        rng = random.Random(BASE_SEED + 3000 + i)
        core_dict = random_core_dict(
            rng,
            name=f"gen_zero_rf_{i}",
            rf_size_range=(0, 256),  # RF may or may not exist this trial
            sram_size_range=(2**19, 2**20),  # SRAM always present as the fallback level
        )
        for operand in ("I1", "I2", "O"):
            rf_name = f"rf_{operand}"
            assert rf_name not in core_dict["memories"] or core_dict["memories"][rf_name]["size"] > 0

        core = build_core_from_dict(core_dict, core_id=i)
        _check_core_sanity(core, core_dict)
    print(f"OK: {N_SCHEMA_TRIALS}/{N_SCHEMA_TRIALS} zero-size-capable RF draws stayed valid (present or omitted).")

    # RF, SRAM, and upper all forced to 0 for every operand -> no memory left anywhere -> must raise.
    try:
        random_core_dict(
            random.Random(BASE_SEED),
            name="gen_all_zero",
            rf_size_range=(0, 0),
            sram_size_range=(0, 0),
            upper_size_range=(0, 0),
        )
    except ValueError:
        print("OK: an all-zero-size core correctly raised ValueError instead of building an empty hierarchy.")
    else:
        raise AssertionError("expected random_core_dict to raise ValueError when every memory level is size 0")


def test_random_core_hardware_evaluation():
    """A few random cores must actually run through the ZigZag cost model, not just validate."""
    failures: list[str] = []
    results: list[tuple[str, float, float]] = []
    configs = [
        {"oa_dims": ("D1", "D2", "D3"), "oa_size_ranges": ((2, 8), (2, 8), (1, 4))},
        {"oa_dims": ("D1", "D2"), "oa_size_ranges": ((2, 8), (2, 8))},
    ]
    trial = 0
    for config in configs:
        for _ in range(N_EVAL_TRIALS):
            rng = random.Random(BASE_SEED + 2000 + trial)
            name = f"gen_eval_{trial}"
            core_dict = random_core_dict(rng, name=name, **config)
            try:
                energy, latency, _cmes = evaluate_core(core_dict)
                assert energy > 0, "expected positive energy estimate"
                assert latency > 0, "expected positive latency estimate"
                results.append((name, energy, latency))
            except Exception as e:  # noqa: BLE001
                failures.append(f"{name} ({config}, seed {BASE_SEED + 2000 + trial}): {e}\n  core_dict={core_dict}")
            trial += 1

    for name, energy, latency in results:
        print(f"OK: {name} -> energy={energy:.2f}, latency={latency:.2f}")

    if failures:
        raise AssertionError(
            f"{len(failures)}/{trial} random-core hardware evaluations failed:\n" + "\n".join(failures)
        )
    print(f"OK: {trial}/{trial} random-core hardware evaluations completed successfully.")


if __name__ == "__main__":
    test_random_3d_cores_validate_and_build()
    test_random_2d_cores_validate_and_build()
    test_zero_size_memories_are_omitted()
    test_random_core_hardware_evaluation()
