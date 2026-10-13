"""Summarize `benchmark_optimizer_comparison.py` runs: convergence curves, Pareto fronts and a comparison table.

Reads `<results-dir>/<workload>/<method>/{trace.csv, summary.json}` and writes into `<results-dir>`:
- `convergence_<workload>.png`: best-so-far EDP against wall time (log-log), one line per method, with a marker
  where each run stopped;
- `pareto_<workload>.png`: every method's non-dominated (latency, energy) points, area as marker size;
- `comparison.csv`: per run -- stop reason, time the stop was declared and real end time (the generation under
  way when the budget stops always finishes), time to reach within 1% of its own final best, best
  EDP (and its gap to the best method on that workload), the best point, evaluation count and the normalized
  (latency, energy, area) hypervolume;
- `comparison.html`: an interactive page in the style of the rolled exploration's `topologies.html` -- per
  workload, the convergence curves, every measured design on a selectable trade-off projection, and for every
  point of each run's Pareto front its details, hardware topology, schedule and core memory hierarchies, see
  `stream.visualization.comparison_report`. The heavy per-point data goes to `comparison_designs/*.js`, loaded
  by the page on demand, so the page itself stays small.

What a Pareto point shows depends on what its run saved (`resolve_point`): runs of the current benchmark save a
design record per front point (`designs/<n_eval>.json`); older runs are rebuilt from the grid candidates'
`result.pickle`, the rolled candidates' `pareto_schedules.pickle` (hardware, no schedule) and the run's
`scme.pickle` (its best design). Anything else is shown with its objectives only.

The hypervolume of each workload is taken on objectives normalized to [ideal, nadir * 1.1] over the union of
all methods' fronts, so methods on one workload share a reference point and the value lies in [0, 1].

Run: python analyze_optimizer_comparison.py [--results-dir outputs/optimizer_comparison]
"""

import argparse
import csv
import json
import math
import os
import sys

from zigzag.utils import pickle_load

from stream.opt.search_budget import hypervolume_3d
from stream.visualization.comparison_report import write_comparison_report
from stream.visualization.design_record import design_record, format_tiling, rolled_design_record
from stream.visualization.optimizer_comparison import plot_convergence, plot_fronts, step2_start

METHOD_ORDER = ("rolled", "graph_ga", "grid_2x2", "grid_4x4")


def read_trace(path: str) -> list[dict[str, float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [
        {
            **{
                key: float(row[key])
                for key in ("t", "n_eval", "latency", "energy", "area", "edp", "best_edp", "feasible")
            },
            "info": row.get("info", ""),
            "artifact": row.get("artifact") or "",
        }
        for row in rows
    ]


def pareto_points(trace: list[dict[str, float]]) -> list[tuple[float, float, float]]:
    return [(row["latency"], row["energy"], row["area"]) for row in pareto_rows(trace)]


def pareto_rows(trace: list[dict[str, float]]) -> list[dict]:
    """The trace rows on the run's (latency, energy, area) front, the first measurement of each point."""
    first: dict[tuple[float, float, float], dict] = {}
    for row in trace:
        if row["feasible"]:
            first.setdefault((row["latency"], row["energy"], row["area"]), row)
    points = sorted(first)
    return [
        first[point]
        for point in points
        if not any(all(o <= p for o, p in zip(other, point, strict=True)) and other != point for other in points)
    ]


def load_runs(results_dir: str) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for workload in sorted(os.listdir(results_dir)):
        workload_dir = os.path.join(results_dir, workload)
        if not os.path.isdir(workload_dir):
            continue
        for method in sorted(os.listdir(workload_dir)):
            run_dir = os.path.join(workload_dir, method)
            trace_path, summary_path = os.path.join(run_dir, "trace.csv"), os.path.join(run_dir, "summary.json")
            if not (os.path.exists(trace_path) and os.path.exists(summary_path)):
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            trace = read_trace(trace_path)
            runs.setdefault(workload, {})[method] = {"summary": summary, "trace": trace, "front": pareto_points(trace)}
    return runs


def normalized_hypervolumes(methods: dict[str, dict]) -> dict[str, float]:
    union = [point for run in methods.values() for point in run["front"]]
    if not union:
        return dict.fromkeys(methods, 0.0)
    ideal = [min(p[i] for p in union) for i in range(3)]
    nadir = [max(p[i] for p in union) * 1.1 for i in range(3)]
    # A fixed-hardware method has one area for every point: give the axis a non-zero span either way.
    span = [(n - i) or 1.0 for i, n in zip(ideal, nadir, strict=True)]

    def normalize(point):
        return tuple((point[i] - ideal[i]) / span[i] for i in range(3))

    return {
        method: hypervolume_3d([normalize(p) for p in run["front"]], (1.0, 1.0, 1.0)) for method, run in methods.items()
    }


def _finite(value: float | None) -> float | None:
    """JSON has no infinity: an infeasible or missing value becomes null."""
    return value if value is not None and math.isfinite(value) else None


def _same_point(latency: float, energy: float, row: dict) -> bool:
    return math.isclose(latency, row["latency"], rel_tol=1e-9) and math.isclose(energy, row["energy"], rel_tol=1e-9)


def _load_pickle(path: str):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    return pickle_load(path)


class PointResolver:
    """Finds what a run saved about each of its Pareto points (see the module docstring), loading every source
    file at most once."""

    def __init__(self, run_dir: str, method: str):
        self.run_dir, self.method = run_dir, method
        self._scme = None
        self._scme_loaded = False
        self._rolled: list[tuple[str, dict]] | None = None

    def run_scme(self):
        if not self._scme_loaded:
            self._scme_loaded = True
            path = os.path.join(self.run_dir, "scme.pickle")
            if os.path.exists(path):
                try:
                    self._scme = _load_pickle(path)
                except Exception as exc:
                    print(f"warning: could not load {path}: {exc!r}")
        return self._scme

    def layer_names(self) -> dict[int, str]:
        scme = self.run_scme()
        if scme is None:
            return {}
        return {node.id: node.short_name for node in scme.workload.node_list if hasattr(node, "short_name")}

    def rolled_records(self) -> list[tuple[str, dict]]:
        """(candidate directory, record) of every rolled tiling candidate's Pareto schedules."""
        if self._rolled is None:
            self._rolled = []
            search_dir = os.path.join(self.run_dir, "tiling_search")
            for candidate in sorted(os.listdir(search_dir)) if os.path.isdir(search_dir) else []:
                path = os.path.join(search_dir, candidate, "rolled_search", "pareto_schedules.pickle")
                if os.path.exists(path):
                    try:
                        self._rolled += [(candidate, r) for r in _load_pickle(path)["pareto"]]
                    except Exception as exc:
                        print(f"warning: could not load {path}: {exc!r}")
        return self._rolled

    def resolve(self, row: dict) -> tuple[dict | None, str]:
        """(design record or None, where it came from) of one Pareto trace row."""
        for source, attempt in (
            ("artifact", self._from_artifact),
            ("best", self._from_run_scme),
            ("tiling candidate", self._from_tiling_candidate),
            ("rolled record", self._from_rolled_record),
        ):
            try:
                record = attempt(row)
            except Exception as exc:  # one unreadable design must not cost the whole report
                print(f"warning: {self.run_dir} #{int(row['n_eval'])} ({source}): {exc!r}")
                continue
            if record is not None:
                return record, source
        return None, "objectives only"

    def _from_artifact(self, row: dict) -> dict | None:
        if not row["artifact"]:
            return None
        with open(os.path.join(self.run_dir, "designs", f"{row['artifact']}.json")) as f:
            return json.load(f)

    def _from_run_scme(self, row: dict) -> dict | None:
        scme = self.run_scme()
        if scme is None or not _same_point(float(scme.latency), float(scme.energy), row):
            return None
        return design_record(scme)

    def _from_tiling_candidate(self, row: dict) -> dict | None:
        kind, _, key = row["info"].partition(" ")
        if kind != "tiling" or not key or " " in key:  # "tiling <key> failed" has no result
            return None
        path = os.path.join(self.run_dir, "tiling_search", f"candidate_{key}", "result.pickle")
        if not os.path.exists(path):
            return None
        scme, extra = _load_pickle(path)
        if not _same_point(float(scme.latency), float(scme.energy), row):
            return None
        return design_record(scme, tiling=format_tiling(extra.get("tiling_config")))

    def _from_rolled_record(self, row: dict) -> dict | None:
        kind, _, plan_id = row["info"].partition(" ")
        if kind != "rolled" or not plan_id or " " in plan_id:  # "rolled <id> infeasible" has no record
            return None
        for candidate, record in self.rolled_records():
            if record["id"] != plan_id or not _same_point(record["latency"], record["energy"], row):
                continue
            # The candidate's own best design was scheduled in full: prefer it when it is this point.
            path = os.path.join(self.run_dir, "tiling_search", candidate, "result.pickle")
            if os.path.exists(path):
                scme, extra = _load_pickle(path)
                if _same_point(float(scme.latency), float(scme.energy), row):
                    return design_record(scme, tiling=format_tiling(extra.get("tiling_config")))
            return rolled_design_record(record, self.layer_names())
        return None


def _point_summary(workload: str, method: str, row: dict, record: dict | None, source: str) -> dict:
    """What the page inlines about a Pareto point; the rest of `record` goes to its side file."""
    n_eval = int(row["n_eval"])
    topology = (record or {}).get("topology") or {"cores": [], "edges": []}
    return {
        "id": f"{workload}_{method}_{n_eval}",
        "method": method,
        "label": f"{method} #{n_eval}",
        "info": row["info"],
        "source": source,
        "t": row["t"],
        "n_eval": n_eval,
        "latency": row["latency"],
        "energy": row["energy"],
        "area": row["area"],
        "nb_cores": sum(1 for core in topology["cores"] if core["kind"] == "compute") or None,
        "nb_links": sum(1 for edge in topology["edges"] if edge["kind"] == "link"),
        "has_hardware": bool(topology["cores"]),
        "has_schedule": bool(record and record.get("schedule")),
        "file": f"{workload}_{method}_{n_eval}.js" if record else None,
        "cores": topology["cores"],
        "edges": topology["edges"],
    }


def build_report_data(results_dir: str, runs: dict[str, dict[str, dict]], rows: list[dict]) -> tuple[dict, dict]:
    """Everything `comparison.html` shows, as plain JSON-safe data, and the per-point side files
    (`{file name: design record minus its topology}`)."""
    row_of = {(row["workload"], row["method"]): row for row in rows}
    by_workload = {}
    budget = {}
    sidecars: dict[str, dict] = {}
    for workload, methods in runs.items():
        order = [m for m in METHOD_ORDER if m in methods] + sorted(m for m in methods if m not in METHOD_ORDER)
        workload_runs, designs = {}, []
        for method in order:
            run, summary, row = methods[method], methods[method]["summary"], row_of[(workload, method)]
            budget = {key: summary[key] for key in ("max_time_s", "window_s", "rel_tol")}
            front = set(run["front"])
            curve, last = [], math.inf
            for entry in run["trace"]:
                if math.isfinite(entry["best_edp"]) and entry["best_edp"] < last:
                    last = entry["best_edp"]
                    curve.append([max(entry["t"], 1e-3), last])
            end_time = max(summary["time_to_stop"], summary.get("total_wall_time") or 0)
            workload_runs[method] = {
                "curve": curve,
                "end_time": end_time,
                "stop_reason": summary["stop_reason"],
                "stop_declared": summary["time_to_stop"] if summary["stop_reason"] in ("time", "converged") else None,
                "step2_start": step2_start(run["trace"]),
                "time_to_within_1pct": summary.get("time_to_within_1pct"),
                "n_eval": summary["n_eval"],
                "best_edp": _finite(summary["best_edp"]),
                "edp_gap_pct": _finite(row["edp_gap_to_best_pct"]),
                "pareto_points": row["pareto_points"],
                "hypervolume": row["hypervolume"],
                "points": [
                    {
                        "latency": entry["latency"],
                        "energy": entry["energy"],
                        "area": entry["area"],
                        "pareto": (entry["latency"], entry["energy"], entry["area"]) in front,
                    }
                    for entry in run["trace"]
                    if entry["feasible"]
                ],
            }
            resolver = PointResolver(os.path.join(results_dir, workload, method), method)
            for trace_row in pareto_rows(run["trace"]):
                record, source = resolver.resolve(trace_row)
                point = _point_summary(workload, method, trace_row, record, source)
                if record is not None:
                    sidecars[point["file"]] = {key: value for key, value in record.items() if key != "topology"}
                designs.append(point)
            sources = [d["source"] for d in designs if d["method"] == method]
            print(
                f"{workload:>9} {method:>9}: {len(sources)} Pareto point(s), "
                + ", ".join(f"{sources.count(source)} {source}" for source in dict.fromkeys(sources))
            )
        best_design = min(range(len(designs)), key=lambda i: designs[i]["latency"] * designs[i]["energy"], default=0)
        by_workload[workload] = {
            "methods": order,
            "runs": workload_runs,
            "designs": designs,
            "best_design": best_design,
        }
    data = {"workloads": sorted(by_workload), "by_workload": by_workload, "budget": budget}
    return data, sidecars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="outputs/optimizer_comparison")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    rows = []
    for workload, methods in runs.items():
        plot_convergence(workload, methods, os.path.join(args.results_dir, f"convergence_{workload}.png"))
        plot_fronts(workload, methods, os.path.join(args.results_dir, f"pareto_{workload}.png"))
        hypervolumes = normalized_hypervolumes(methods)
        best_overall = min(run["summary"]["best_edp"] for run in methods.values())
        for method, run in methods.items():
            summary = run["summary"]
            best = summary.get("best_point") or {}
            rows.append(
                {
                    "workload": workload,
                    "method": method,
                    "stop_reason": summary["stop_reason"],
                    "time_to_stop_s": round(summary["time_to_stop"], 1),
                    "total_wall_time_s": round(summary.get("total_wall_time") or summary["time_to_stop"], 1),
                    "time_to_within_1pct_s": summary["time_to_within_1pct"]
                    and round(summary["time_to_within_1pct"], 1),
                    "n_eval": summary["n_eval"],
                    "best_edp": summary["best_edp"],
                    "edp_gap_to_best_pct": round(100 * (summary["best_edp"] / best_overall - 1), 2)
                    if math.isfinite(summary["best_edp"])
                    else math.inf,
                    "best_latency": best.get("latency"),
                    "best_energy": best.get("energy"),
                    "best_area": best.get("area"),
                    "pareto_points": len(run["front"]),
                    "step2_start_s": step2_start(run["trace"]),
                    "hypervolume": round(hypervolumes[method], 4),
                    "error": summary.get("error"),
                }
            )
    out_path = os.path.join(args.results_dir, "comparison.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(
            f"{row['workload']:>9} {row['method']:>9}: {row['stop_reason']:>9} at {row['time_to_stop_s']:>8}s, "
            f"1% at {row['time_to_within_1pct_s']}s, {row['n_eval']:>5} evals, best EDP {row['best_edp']:.4g} "
            f"(+{row['edp_gap_to_best_pct']}%), HV {row['hypervolume']}"
        )
    print(f"Wrote {out_path}")
    html_path = os.path.join(args.results_dir, "comparison.html")
    data, sidecars = build_report_data(args.results_dir, runs, rows)
    write_comparison_report(data, html_path, sidecars=sidecars)
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
