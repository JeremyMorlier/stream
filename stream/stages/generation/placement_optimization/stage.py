"""Stages that solve the placement optimization (`docs/source/schedule_optimization_formulation.md`) with CP-SAT
(`CpSatPlacementOptimizationStage`) or as a Gurobi MILP (`GurobiPlacementOptimizationStage`), the same model in both,
so their fronts, solve times and gaps can be compared.

Each stage builds a `PlacementProblem` from the schedule space, sweeps the epsilon-constraint grid (three
lexicographic anchors, then latency minimized over an (area, energy) budget grid), re-measures every distinct
solution with the real scheduler and adds it to the schedule search's `ParetoArchive`, so `ScheduleSelectionStage`
and the report stages work unchanged. Each solve is logged to `solves.csv` with its status, time and gap.
"""

import csv
import logging
import math
import os
import time
from typing import Any, Literal

from stream.stages.generation.placement_optimization.backend import BackendOptions, PlacementBackend, SolveResult
from stream.stages.generation.placement_optimization.cpsat_backend import CpSatBackend
from stream.stages.generation.placement_optimization.gurobi_backend import GurobiBackend
from stream.stages.generation.placement_optimization.problem import PlacementProblem
from stream.stages.generation.schedule_evaluation import ScheduleSearchStage
from stream.stages.generation.schedule_optimization import (
    AREA_SCALE,
    ENERGY_SCALE,
    MIN_GRID_LEVELS,
    OBJECTIVES,
    ModelSolution,
)
from stream.stages.generation.schedule_space import ScheduleSearchState, ScheduleSpace
from stream.stages.generation.tiled_workload_accelerator_generation import renumber_merged_cores
from stream.stages.stage import StageCallable
from stream.visualization.design_record import design_record

logger = logging.getLogger(__name__)

SOLVE_COLUMNS = ["solver", "label", "objective", "bound_latency", "bound_energy", "bound_area", "status", "time_s",
                 "objective_value", "objective_bound", "gap", "latency", "energy", "area"]  # fmt: skip
# The area grid's first level is the minimum area plus this fraction (see `PlacementSweep.run`).
AREA_SLACK = 0.1


def scaled(solution: ModelSolution, name: str) -> int:
    """An objective of a solution in the model's integer units."""
    if name == "latency":
        return solution.latency
    if name == "energy":
        return round(solution.energy / ENERGY_SCALE)
    return round(solution.area * AREA_SCALE)


class PlacementSweep:
    """The epsilon-constraint sweep over one backend, independent of any stage (the tests drive it directly)."""

    def __init__(
        self,
        backend: PlacementBackend,
        *,
        solve_time_limit_s: float,
        area_levels: int,
        energy_levels: int,
        area_slack: float = AREA_SLACK,
    ):
        self.backend = backend
        self.solve_time_limit_s = solve_time_limit_s
        self.area_levels = area_levels
        self.energy_levels = energy_levels
        self.area_slack = area_slack
        self.solve_log: list[dict[str, Any]] = []

    def solve(self, objective: str, bounds: dict[str, int], time_limit: float, label: str) -> SolveResult:
        result = self.backend.solve(objective, bounds, time_limit)
        entry = {
            "solver": self.backend.name,
            "label": label,
            "objective": objective,
            **{f"bound_{name}": bounds.get(name, "") for name in OBJECTIVES},
            "status": result.status,
            "time_s": round(result.time_s, 3),
            "objective_value": result.objective_value,
            "objective_bound": result.objective_bound,
            "gap": None if result.gap is None else round(result.gap, 6),
        }
        if result.solution is not None:
            entry.update(latency=result.solution.latency, energy=result.solution.energy, area=result.solution.area)
        self.solve_log.append(entry)
        return result

    def lexicographic(self, order: tuple[str, ...], bounds: dict[str, int], label: str) -> ModelSolution | None:
        """Minimize the objectives in `order` one after the other, each fixed at its optimum for the next (the
        tie-breaks get half the time limit), so the result is non-dominated within the model."""
        bounds = dict(bounds)
        best = None
        for step, objective in enumerate(order):
            limit = self.solve_time_limit_s if step == 0 else self.solve_time_limit_s / 2
            result = self.solve(objective, bounds, limit, f"{label}:{objective}")
            if result.solution is None:
                return best
            best = result.solution
            bounds[objective] = scaled(best, objective)
        return best

    def run(self) -> list[ModelSolution]:
        """Anchors (each objective minimized first), then latency minimized over an (area, energy) budget grid
        between each objective's minimum and its value at the latency-optimal point.

        The grid is geometric: the fastest point's area can be hundreds of times the minimum, and a linear grid
        leaves the low-area region empty. The area grid starts at the minimum area plus `area_slack`, not at the
        minimum: MACs cost little area next to memories, so the minimum-area point has a tiny array and is far
        slower than a point with a full array at a few percent more area."""
        solutions: dict[tuple, ModelSolution] = {}

        def keep(solution: ModelSolution | None) -> None:
            if solution is not None:
                solutions.setdefault(solution.key, solution)

        anchors = {}
        for first in OBJECTIVES:
            rest = tuple(o for o in OBJECTIVES if o != first)
            anchors[first] = self.lexicographic((first, *rest), {}, f"anchor-{first}")
            keep(anchors[first])
        fastest = anchors["latency"]
        if fastest is None:
            return list(solutions.values())

        def grid(name: str, levels: int, slack: float = 0.0) -> list[int]:
            if anchors[name] is None or levels < MIN_GRID_LEVELS:
                return []
            low, high = scaled(anchors[name], name) * (1 + slack), scaled(fastest, name)
            if high <= low or low <= 0:
                return []
            return sorted({round(low * (high / low) ** (i / (levels - 1))) for i in range(levels - 1)})

        for area_bound in grid("area", self.area_levels, self.area_slack):
            for energy_bound in [None, *grid("energy", self.energy_levels)]:
                bounds = {"area": area_bound}
                if energy_bound is not None:
                    bounds["energy"] = energy_bound
                keep(self.lexicographic(("latency", "energy", "area"), bounds, "grid"))
        return list(solutions.values())


class PlacementOptimizationStage(ScheduleSearchStage):
    """Base of the two solver stages; `backend_cls` picks the solver. Takes `schedule_search` from
    `ScheduleExplorationStage` (which only needs to build the space, e.g. `max_schedule_evaluations=1`) and hands
    the extended state down to `ScheduleSelectionStage`."""

    requires_full_cost_matrix = True
    backend_cls: type = CpSatBackend

    def __init__(  # noqa: PLR0913
        self,
        list_of_callables: list[StageCallable],
        *,
        schedule_search: ScheduleSearchState,
        optimization_search_dir: str | None = None,
        tasks_per_group: int | None = 1,
        max_cores: int | None = None,
        allow_sibling_merge: bool = True,
        area_levels: int = 5,
        energy_levels: int = 3,
        area_slack: float = AREA_SLACK,
        solve_time_limit_s: float = 20.0,
        solver_workers: int = 8,
        mip_gap: float = 0.0,
        max_optimized_evaluations: int = 200,
        scheduling_order_source: Literal["generated", "model"] = "generated",
        offchip_bus: Literal["none", "aggregate", "timed"] = "none",
        link_transfers: Literal["none", "timed"] = "timed",
        **kwargs: Any,
    ):
        """
        Args:
            schedule_search: the schedule search's state; its archive and records are extended in place.
            optimization_search_dir: where this stage's report goes. Defaults to `placement_<solver>/` next to the
                tiled workload.
            tasks_per_group: tasks per group, contiguous in tile order (`None`: one task per tile).
            max_cores: at most this many cores (slots); `None`: one per group.
            allow_sibling_merge: let two groups of the same layer share a core.
            area_levels, energy_levels: the epsilon-constraint grid.
            area_slack: the area grid's first level, relative to the minimum area (see `PlacementSweep.run`).
            solve_time_limit_s: time limit per solve (half of it for the tie-break solves). A solve that times out
                still contributes its best solution.
            solver_workers: solver threads.
            mip_gap: Gurobi's relative MIP gap (CP-SAT always proves optimality within its time limit).
            max_optimized_evaluations: cap on solutions measured with the real scheduler.
            scheduling_order_source: measure in Stream's generated tile order, or in the model's.
            offchip_bus: `"none"`, `"aggregate"` (bus time bounds latency) or `"timed"` (a load before each task).
            link_transfers: `"none"` (links at their traffic-derived width, transfers free) or `"timed"` (a width
                per communicating group pair, transfers delay their consumers).
        """
        super().__init__(list_of_callables, **kwargs)
        for name, value, allowed in (
            ("scheduling_order_source", scheduling_order_source, ("generated", "model")),
            ("offchip_bus", offchip_bus, ("none", "aggregate", "timed")),
            ("link_transfers", link_transfers, ("none", "timed")),
        ):
            if value not in allowed:
                raise ValueError(f"{type(self).__name__}: unknown {name} {value!r}.")
        self.schedule_search = schedule_search
        self.space: ScheduleSpace = schedule_search.space
        self.solver = self.backend_cls.name
        self.out_dir = optimization_search_dir or os.path.join(
            os.path.dirname(self.space.tiled_workload_path), f"placement_{self.solver}"
        )
        self.tasks_per_group = tasks_per_group
        self.max_cores = max_cores
        self.allow_sibling_merge = allow_sibling_merge
        self.area_levels = area_levels
        self.energy_levels = energy_levels
        self.area_slack = area_slack
        self.solve_time_limit_s = solve_time_limit_s
        self.solver_workers = solver_workers
        self.mip_gap = mip_gap
        self.max_optimized_evaluations = max_optimized_evaluations
        self.scheduling_order_source = scheduling_order_source
        self.offchip_bus = offchip_bus
        self.link_transfers = link_transfers
        self.search_budget = kwargs.get("search_budget")

    # ------------------------------------------------------------------------------------- measuring -------

    def _measure_solution(  # noqa: PLR0914
        self, problem: PlacementProblem, solution: ModelSolution, archive: Any, records: list[dict[str, Any]]
    ) -> str:
        space = self.space
        merge_of_core = dict(solution.slot_of_group)
        rolled_of = renumber_merged_cores(merge_of_core)
        slot_of_rolled = {rolled_of[group]: slot for group, slot in merge_of_core.items()}
        nb_cores = len(slot_of_rolled)
        design_key_of_core = {core: solution.design_of_slot[slot_of_rolled[core]] for core in range(nb_cores)}
        core_dict_of_core = {core: problem.designs[key] for core, key in design_key_of_core.items()}
        link_widths = solution.rolled_link_widths
        area = space.area_of(design_key_of_core, rolled_of, link_widths)
        record_id = f"opt-{self.solver}-c{nb_cores}-{len(records)}"
        try:
            workload, accelerator, cost_lut, topology = space.assemble_rolled(
                merge_of_core, core_dict_of_core, design_key_of_core, link_widths=link_widths
            )
            extra = (
                {"provided_scheduling_order": list(solution.scheduling_order)}
                if self.scheduling_order_source == "model"
                else {}
            )
            scme = space.measure(workload, accelerator, cost_lut, **extra)
        except Exception:
            logger.warning(f"{type(self).__name__}: solution {record_id} could not be scheduled.", exc_info=True)
            if self.search_budget is not None:
                self.search_budget.record(math.inf, math.inf, area, info=f"{record_id} infeasible")
            return "infeasible"
        point = (float(scme.latency), float(scme.energy), area)
        if self.search_budget is not None:
            self.search_budget.record(*point, info=record_id, artifact=lambda: design_record(scme))
        record = {
            "id": record_id,
            "kind": "rolled",
            "ranks": (),
            "source_id": f"placement-{self.solver}",
            "nb_cores": nb_cores,
            "design_keys": tuple(design_key_of_core[core] for core in range(nb_cores)),
            "latency": point[0],
            "energy": point[1],
            "area": point[2],
            "interconnect_area": space.interconnect_of(design_key_of_core, rolled_of, link_widths).total,
            "link_widths": link_widths,
            "predicted_area": solution.area,
            # The schedule search's CSV calls the analytic estimate a bound; here it is the model's prediction.
            "latency_bound": solution.latency,
            "energy_bound": solution.energy,
            "predicted_latency": solution.latency,
            "predicted_energy": solution.energy,
            "fold_tolerance": "",
            "fold_objective": f"{self.solver}-placement",
            "fold_overlap": (
                f"tasks={self.tasks_per_group} cores<={self.max_cores} siblings={self.allow_sibling_merge} "
                f"order={self.scheduling_order_source} bus={self.offchip_bus} links={self.link_transfers}"
            ),
            "merge_of_core": merge_of_core,
            "design_key_of_core": design_key_of_core,
            "core_dict_of_core": core_dict_of_core,
            "members_of_core": dict(topology.members_of_core),
            "hosted_node_groups": dict(topology.hosted_node_groups),
            "node_core_ids": dict(topology.node_core_ids),
            "link_connections": list(topology.link_connections),
            "bus_connection": dict(topology.bus_connection),
        }
        if self.scheduling_order_source == "model":
            record["scheduling_order"] = list(solution.scheduling_order)
        records.append(record)
        archive.add(point, record)
        logger.info(
            f"{type(self).__name__}: measured {record_id}: {nb_cores} core(s), latency={point[0]:.6g} "
            f"(predicted {solution.latency:.6g}), energy={point[1]:.6g} (predicted {solution.energy:.6g}), "
            f"area={area:.6g} (predicted {solution.area:.6g}) (front: {len(archive.entries)})."
        )
        return "measured"

    def _save_solve_log(self, solve_log: list[dict[str, Any]]) -> None:
        with open(os.path.join(self.out_dir, "solves.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SOLVE_COLUMNS)
            writer.writeheader()
            for entry in solve_log:
                writer.writerow({column: entry.get(column, "") for column in SOLVE_COLUMNS})

    # ------------------------------------------------------------------------------------------- run ---------

    def run(self):
        state = self.schedule_search
        if not self.space.full_matrix:
            raise ValueError(
                f"{type(self).__name__}: needs the full shape x design cost matrix; raise `max_cme_matrix_cells`."
            )
        os.makedirs(self.out_dir, exist_ok=True)
        if self.link_transfers == "timed" and not self.space.interconnect_model.timed_widths:
            logger.warning(
                f"{type(self).__name__}: links are timed in the model but measured at their traffic "
                f"(`link_timing='traffic'`), where transfers take about a cycle: predictions will be pessimistic."
            )
        t0 = time.perf_counter()
        problem = PlacementProblem.from_space(
            self.space,
            tasks_per_group=self.tasks_per_group,
            max_cores=self.max_cores,
            allow_sibling_merge=self.allow_sibling_merge,
        )
        problem.save(os.path.join(self.out_dir, "problem.pickle"))
        options = BackendOptions(
            offchip_bus=self.offchip_bus,
            link_transfers=self.link_transfers,
            workers=self.solver_workers,
            mip_gap=self.mip_gap,
            log_path=os.path.join(self.out_dir, f"{self.solver}.log"),
        )
        backend = self.backend_cls(problem, options)
        t_model = time.perf_counter() - t0
        sweep = PlacementSweep(
            backend,
            solve_time_limit_s=self.solve_time_limit_s,
            area_levels=self.area_levels,
            energy_levels=self.energy_levels,
            area_slack=self.area_slack,
        )
        solutions = sweep.run()
        t_solve = time.perf_counter() - t0 - t_model
        solutions.sort(key=lambda s: (s.latency, s.energy, s.area))
        statuses: dict[str, int] = {}
        for solution in solutions[: self.max_optimized_evaluations]:
            status = self._measure_solution(problem, solution, state.archive, state.records)
            statuses[status] = statuses.get(status, 0) + 1
        t_total = time.perf_counter() - t0
        self._save_solve_log(sweep.solve_log)
        solve_statuses: dict[str, int] = {}
        for entry in sweep.solve_log:
            solve_statuses[entry["status"]] = solve_statuses.get(entry["status"], 0) + 1
        state.stats[f"placement_{self.solver}"] = {
            **problem.size_report(),
            **backend.size(),
            "solves": len(sweep.solve_log),
            "solve_statuses": solve_statuses,
            "solutions": len(solutions),
            **statuses,
            "model_time_s": round(t_model, 2),
            "solve_time_s": round(t_solve, 2),
            "measure_time_s": round(t_total - t_model - t_solve, 2),
            "total_time_s": round(t_total, 2),
        }
        logger.info(f"{type(self).__name__}: {state.stats[f'placement_{self.solver}']}.")
        state.report["out_dir"] = self.out_dir
        kwargs = self.kwargs.copy()
        kwargs["schedule_search"] = state
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
        yield from sub_stage.run()


class CpSatPlacementOptimizationStage(PlacementOptimizationStage):
    """The placement optimization solved with OR-Tools CP-SAT."""

    backend_cls = CpSatBackend


class GurobiPlacementOptimizationStage(PlacementOptimizationStage):
    """The placement optimization solved as a Gurobi MILP."""

    backend_cls = GurobiBackend
