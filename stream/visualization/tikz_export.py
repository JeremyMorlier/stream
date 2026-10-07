"""Turn one exploration's artifacts into LaTeX figures: the Pareto front as pgfplots, the topologies as TikZ.

`topologies.html` is the interactive twin of this module -- same front, same layout, same colours -- but a
paper cannot embed it, and a screenshot of it (or of the matplotlib PNGs) ages badly next to vector text at the
document's own font size. So the same two views are re-emitted as LaTeX source: fonts, line widths and colours
then come from the paper, and the figures stay editable.

Nothing here re-runs the search. The inputs are the files `ScheduleExplorationStage` already wrote next to each
other -- `schedules.csv` for every measured point and its Pareto flag, `topologies.json` for the cores and wires
of the designs that got a figure -- so this can be pointed at an old output directory.

What comes out of `write_figures`:

- `stream-tikz-defs.tex`: colours and styles, `\\input` once in the paper's preamble. The figure fragments carry
  no styling of their own, so restyling the whole set is a one-file edit.
- `pareto_topologies_<x>_<y>.tex`: the headline figure -- one projection of the front with the accelerators
  behind its lettered points drawn as insets above it, each tied to its point by a leader. This is the whole
  claim in one float: which shape of accelerator sits where on the trade-off.
- `pareto_<x>_<y>.tex`: the same projection on its own, for when the insets belong elsewhere -- dominated
  points greyed behind it, the achievable staircase dashed through it, and the topologies tagged where they sit.
- `topology_<tag>.tex`: one accelerator per tag, drawn at the same coordinates as the PNG and the HTML report.
- `designs.tex`: a booktabs table of the tagged designs, so the figure letters and the numbers stay in sync.
- `figures.tex`: a standalone document `\\input`ing all of the above -- compile it to check the fragments before
  they go anywhere near the paper.

Run: `python -m stream.visualization.tikz_export outputs/<experiment>/rolled_search`
"""

import argparse
import csv
import json
import logging
import math
import os
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- only the colormap is wanted, never a window
import matplotlib.pyplot as plt  # noqa: E402

from stream.visualization.hardware_graph import COMPUTE_CMAP  # noqa: E402
from stream.visualization.schedule_tradeoff import AXES, pareto_staircase  # noqa: E402

logger = logging.getLogger(__name__)

OBJECTIVES = ("latency", "energy", "area")
TAGS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Matches the HTML report's `--accent` / `--accent-2` in light mode, so the two views stay recognisably the
# same figure. The greys are the report's `--dominated` / `--offchip` / `--bus`.
COLORS = {
    "unrolled": (0x3B, 0x6E, 0xA5),
    "rolled": (0xB4, 0x63, 0x2F),
    "dominated": (0xC9, 0xC9, 0xC3),
    "offchip": (0x8D, 0x8D, 0x86),
    "bus": (0xA9, 0xA9, 0xA2),
    "merged": (0xD9, 0xD9, 0xD9),  # a folded core belongs to no single layer, so it gets no layer colour
}


def _tex_escape(text: str) -> str:
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, f"\\{char}" if char != "\\" else r"\textbackslash{}")
    return text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def _tex_number(value: float, digits: int = 2) -> str:
    """`$1.02\\times10^{5}$` -- the objectives span five decades, so plain digits are unreadable in a table."""
    if value == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0**exponent
    if -1 <= exponent <= 3:
        return f"${value:.{max(0, digits - exponent)}f}$"
    return f"${mantissa:.{digits}f}\\times10^{{{exponent}}}$"


def _short_layer(name: str) -> str:
    """`conv1/Conv` -> `conv1`, `/Add` -> `Add`. The operator type is already implied by the layer's name, and
    the full ONNX path does not fit under a node."""
    stem = name.rsplit("/", 1)[0] or name.lstrip("/")
    return _tex_escape(stem or name)


def _core_layer_label(layers: list[dict[str, Any]], max_shown: int = 2) -> str:
    """What a core runs, short enough to sit under it. A folded core can host the whole workload, so past
    `max_shown` the rest is counted rather than listed."""
    if not layers:
        return ""
    names = [_short_layer(layer["name"]) for layer in layers]
    if len(names) <= max_shown:
        return ", ".join(names)
    return ", ".join(names[:max_shown]) + f", +{len(names) - max_shown}"


def read_schedules(csv_path: str) -> list[dict[str, Any]]:
    """Every measured schedule from `schedules.csv`, objectives as floats and `pareto` as a bool."""
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        try:
            for objective in OBJECTIVES:
                record[objective] = float(row[objective])
        except (KeyError, ValueError):
            logger.warning(f"read_schedules: skipping a row of {csv_path} without usable objectives.")
            continue
        record["pareto"] = str(row.get("pareto", "")).strip().lower() == "true"
        record["nb_cores"] = int(row["nb_cores"]) if row.get("nb_cores") else 0
        record["kind"] = row.get("kind") or "unrolled"
        records.append(record)
    return records


def read_designs(json_path: str) -> list[dict[str, Any]]:
    """The drawn designs from `topologies.json` (cores, wires and coordinates included)."""
    with open(json_path) as handle:
        return json.load(handle)["designs"]


def select_designs(designs: list[dict[str, Any]], nb: int) -> list[dict[str, Any]]:
    """Which designs get a topology figure: the per-objective extremes -- the corners of the trade-off a reader
    wants to see -- then a stride through what is left, so the sample spans the front instead of clustering."""
    if nb <= 0 or len(designs) <= nb:
        return list(designs)
    chosen: dict[str, dict[str, Any]] = {}
    for objective in OBJECTIVES:
        extreme = min(designs, key=lambda design, objective=objective: design[objective])
        chosen[extreme["id"]] = extreme
    remaining = [design for design in designs if design["id"] not in chosen]
    slots = max(0, nb - len(chosen))
    if slots and remaining:
        stride = max(1, len(remaining) // slots)
        for design in remaining[::stride][:slots]:
            chosen[design["id"]] = design
    ordered = sorted(chosen.values(), key=lambda design: design["latency"])
    return ordered[:nb]


def layer_colors(designs: list[dict[str, Any]]) -> dict[int, tuple[float, float, float]]:
    """Layer id -> RGB, from the same colormap and the same sort order as `plot_hardware_graph`, so a core is
    the same colour in the PNG, the HTML report and the paper."""
    layer_ids = sorted({layer["id"] for design in designs for core in design["cores"] for layer in core["layers"]})
    cmap = plt.get_cmap(COMPUTE_CMAP)
    return {layer_id: cmap(index % cmap.N)[:3] for index, layer_id in enumerate(layer_ids)}


def _color_definitions(designs: list[dict[str, Any]]) -> str:
    lines = [
        f"\\definecolor{{stream{name}}}{{RGB}}{{{r},{g},{b}}}" for name, (r, g, b) in COLORS.items()
    ]
    for layer_id, (r, g, b) in layer_colors(designs).items():
        lines.append(f"\\definecolor{{streamlayer{layer_id}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}")
    return "\n".join(lines)


def defs_tex(designs: list[dict[str, Any]]) -> str:
    """The preamble fragment: packages, colours, node and line styles. The figures reference these names only,
    so a paper can restyle every one of them without regenerating anything."""
    return (
        "% Generated by stream.visualization.tikz_export -- \\input this in the preamble.\n"
        "\\usepackage{tikz}\n"
        "\\usepackage{pgfplots}\n"
        "\\usepackage{booktabs}\n"
        "\\pgfplotsset{compat=1.16}\n"
        "\\usetikzlibrary{arrows.meta,backgrounds,fit,positioning,shapes.geometric}\n"
        "\n"
        f"{_color_definitions(designs)}\n"
        "\n"
        "\\tikzset{\n"
        "  stream core/.style={circle, draw=black!70, line width=0.5pt, inner sep=0pt, minimum size=17pt,\n"
        "                      font=\\tiny, align=center},\n"
        "  stream offchip/.style={rectangle, rounded corners=1.5pt, draw=black!70, line width=0.5pt,\n"
        "                         fill=streamoffchip, inner sep=2pt, minimum width=26pt, minimum height=13pt,\n"
        "                         font=\\tiny, text=white, align=center},\n"
        "  stream bus/.style={diamond, aspect=1.6, draw=black!70, line width=0.5pt, fill=streambus,\n"
        "                     inner sep=1pt, font=\\tiny, align=center},\n"
        "  % Opaque backing: links leave a core radially, so they run straight through its caption.\n"
        "  stream layer label/.style={font=\\tiny, text=black!65, inner sep=1pt, anchor=north,\n"
        "                             align=center,\n"
        "                             fill=white, fill opacity=0.85, text opacity=1},\n"
        "  stream link/.style={draw=black!65},\n"
        "  stream bus wire/.style={draw=streambus, dash pattern=on 2pt off 1.5pt},\n"
        "  stream tag/.style={circle, draw=black!70, fill=white, line width=0.4pt, inner sep=0.8pt,\n"
        "                     font=\\bfseries\\tiny},\n"
        "  % Inset variants: an inset carries the shape of an accelerator, the legend carries its words.\n"
        "  stream core mini/.style={circle, draw=black!70, line width=0.3pt, inner sep=0pt,\n"
        "                           minimum size=7pt, font=\\fontsize{4}{4}\\selectfont, align=center},\n"
        "  stream offchip mini/.style={rectangle, rounded corners=1pt, draw=black!70, line width=0.3pt,\n"
        "                              fill=streamoffchip, inner sep=0pt, minimum width=11pt,\n"
        "                              minimum height=6pt},\n"
        "  stream bus mini/.style={diamond, aspect=1.6, draw=black!70, line width=0.3pt, fill=streambus,\n"
        "                          inner sep=0pt, minimum width=10pt, minimum height=6pt},\n"
        "  stream inset/.style={rectangle, rounded corners=2pt, draw=black!25, fill=black!2,\n"
        "                       line width=0.4pt, inner sep=0pt},\n"
        "  stream inset title/.style={font=\\tiny, text=black!75, inner sep=0pt, align=left},\n"
        "  % Dotted inside the axis, solid outside it: the rise crosses data, the run crosses nothing.\n"
        "  stream drop/.style={draw=black!45, line width=0.35pt, dotted},\n"
        "  stream leader/.style={draw=black!45, line width=0.35pt},\n"
        "  stream swatch/.style={circle, draw=black!70, line width=0.3pt, inner sep=0pt, minimum size=6pt},\n"
        "  stream legend text/.style={font=\\scriptsize, text=black!75, inner sep=1pt},\n"
        "}\n"
        "\\pgfplotsset{\n"
        "  stream front axis/.style={\n"
        "    width=\\linewidth, height=0.72\\linewidth, scale only axis=false,\n"
        "    tick label style={font=\\scriptsize}, label style={font=\\small},\n"
        "    legend style={font=\\scriptsize, draw=black!30, fill=white, fill opacity=0.85,\n"
        "                  text opacity=1, inner sep=2pt},\n"
        "    legend cell align=left, grid=both, grid style={black!12, line width=0.3pt},\n"
        "    axis line style={black!55}, tick style={black!55},\n"
        "  },\n"
        "}\n"
    )


def _axis_label(key: str) -> str:
    return _tex_escape(AXES.get(key, key.capitalize()))


def _log_colorbar_ticks(vmin: float, vmax: float) -> tuple[list[float], list[str]]:
    """Ticks for a colour bar carrying `log10(objective)`. The third objective spans a decade or more, so the
    bar is mapped in log space; the labels are written back as powers of ten so the reader never sees the log."""
    ticks: list[float] = []
    labels: list[str] = []
    start = math.floor(vmin)
    while start <= math.ceil(vmax):
        for mantissa in (1.0, 2.0, 5.0):
            tick = start + math.log10(mantissa)
            if vmin - 1e-9 <= tick <= vmax + 1e-9:
                ticks.append(tick)
                mantissa_text = "" if mantissa == 1.0 else f"{mantissa:g}\\!\\times\\!"
                labels.append(f"${mantissa_text}10^{{{start}}}$")
        start += 1
    if len(ticks) < 2:
        ticks = [vmin, vmax]
        labels = [_tex_number(10.0**value, 1) for value in ticks]
    return ticks, labels


def _table_rows(records: list[dict[str, Any]], keys: tuple[str, ...]) -> str:
    return "\n".join("        " + " ".join(f"{record[key]:.6g}" for key in keys) for record in records)


def pareto_tikz(  # noqa: PLR0913
    records: list[dict[str, Any]],
    *,
    x_key: str = "latency",
    y_key: str = "area",
    color_key: str = "energy",
    tags: dict[str, str] | None = None,
    color_log: bool = True,
    caption_ready: bool = True,
) -> str:
    """One projection of the front as a pgfplots `axis`.

    Two objectives fit on the axes; the third becomes the marker colour, which is what turns a scatter into a
    visible trade-off (a point sitting off the staircase is one paying for its third objective). Unrolled and
    rolled schedules keep separate markers -- on a rolled search they share the front, and the whole claim is
    which end of it each family occupies.

    Args:
        records: every measured schedule, with a `pareto` flag (as `read_schedules` returns).
        x_key, y_key: the objectives on the axes; `color_key` is the third, on the colour bar.
        tags: design id -> short tag, drawn at that point and matching the topology figures.
        color_log: map the colour bar in log space. The third objective usually spans more than a decade.
        caption_ready: wrap the `tikzpicture` in a `figure` with a caption and label.
    """
    front = [record for record in records if record["pareto"]]
    dominated = [record for record in records if not record["pareto"]]
    if not front:
        raise ValueError("pareto_tikz: no record is flagged as pareto, nothing to draw.")
    tags = tags or {}

    meta = (lambda value: math.log10(value)) if color_log else (lambda value: value)
    metas = [meta(record[color_key]) for record in front]
    vmin, vmax = min(metas), max(metas)
    by_kind = {kind: [record for record in front if record["kind"] == kind] for kind in ("unrolled", "rolled")}
    xs, ys = pareto_staircase([(record[x_key], record[y_key]) for record in front])

    if color_log:
        ticks, labels = _log_colorbar_ticks(vmin, vmax)
        colorbar_style = (
            f"    colorbar style={{font=\\scriptsize, ytick={{{','.join(f'{t:.6g}' for t in ticks)}}},\n"
            f"                     yticklabels={{{','.join(labels)}}},\n"
            f"                     ylabel={{{_axis_label(color_key)}}}, ylabel style={{font=\\scriptsize}}}},\n"
        )
    else:
        colorbar_style = (
            f"    colorbar style={{font=\\scriptsize, ylabel={{{_axis_label(color_key)}}},\n"
            "                     ylabel style={font=\\scriptsize}},\n"
        )

    marks = {"unrolled": "*", "rolled": "triangle*"}
    body = [
        "% Generated by stream.visualization.tikz_export -- needs \\input{stream-tikz-defs} in the preamble.",
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    stream front axis,",
        # The colour bar is drawn outside the axis's `width`, so a full-\linewidth axis overruns the text block.
        "    width=0.84\\linewidth,",
        f"    xlabel={{{_axis_label(x_key)}}}, ylabel={{{_axis_label(y_key)}}},",
        "    xmode=log, ymode=log,",
        "    colormap/viridis, colorbar,",
        f"    point meta min={vmin:.6g}, point meta max={vmax:.6g},",
        colorbar_style.rstrip("\n"),
        "    legend pos=south west,",
        "  ]",
        "",
        f"    % {len(dominated)} measured but dominated schedule(s): the backdrop the front is read against.",
        "    \\addplot[only marks, mark=*, mark size=1.0pt, draw=streamdominated, fill=streamdominated]",
        "      table {",
        "        x y",
        _table_rows(dominated, (x_key, y_key)),
        "      };",
        f"    \\addlegendentry{{dominated ({len(dominated)})}}",
        "",
        "    % Achievable boundary of this projection. A point on the 3D front can sit above it: that is the",
        "    % price it pays on the third objective.",
        "    \\addplot[draw=black!45, line width=0.6pt, dashed, forget plot] coordinates {",
        "      " + " ".join(f"({x:.6g},{y:.6g})" for x, y in zip(xs, ys)),
        "    };",
    ]

    for kind, kind_records in by_kind.items():
        if not kind_records:
            continue
        rows = "\n".join(
            f"        {record[x_key]:.6g} {record[y_key]:.6g} {meta(record[color_key]):.6g}"
            for record in kind_records
        )
        body += [
            "",
            f"    \\addplot[scatter, only marks, scatter src=explicit, mark={marks[kind]}, mark size=1.9pt,",
            "      scatter/use mapped color={draw=black!70, fill=mapped color, line width=0.3pt}]",
            "      table[meta=c] {",
            "        x y c",
            rows,
            "      };",
            f"    \\addlegendentry{{{kind} ({len(kind_records)})}}",
        ]

    tagged = [(tags[record["id"]], record) for record in front if record["id"] in tags]
    if tagged:
        body += ["", "    % The designs drawn as topologies, where they sit on the front."]
        for tag, record in sorted(tagged, key=lambda item: item[0]):
            body.append(
                f"    \\draw[black!55, line width=0.3pt] (axis cs:{record[x_key]:.6g},{record[y_key]:.6g})"
                f" -- ++(9pt,9pt) node[stream tag, anchor=south west] {{{tag}}};"
            )

    body += ["  \\end{axis}", "\\end{tikzpicture}"]
    picture = "\n".join(body)
    if not caption_ready:
        return picture + "\n"
    caption = (
        f"Pareto front of the rolled schedule exploration, projected on {AXES[x_key].split(' [')[0].lower()} "
        f"and {AXES[y_key].split(' [')[0].lower()}, with {AXES[color_key].split(' [')[0].lower()} on the colour "
        f"bar. Circles are unrolled schedules (one core per layer and inter-core group), triangles are the same "
        f"schedules folded onto fewer, reused cores. Letters mark the designs drawn in "
        f"Fig.~\\ref{{fig:stream-topologies}}."
    )
    return (
        "\\begin{figure}[t]\n  \\centering\n"
        + "\n".join("  " + line if line else "" for line in picture.splitlines())
        + f"\n  \\caption{{{caption}}}\n  \\label{{fig:stream-pareto}}\n\\end{{figure}}\n"
    )


def _edge_width(bandwidth: float, max_bandwidth: float) -> float:
    """Line width in pt, log-scaled: an offchip bus (128 bit/cycle) and a link carrying a whole tensor differ by
    orders of magnitude, so linear widths would render the bus invisible. Same shape as `hardware_graph`."""
    if max_bandwidth <= 0:
        return 0.4
    return 0.3 + 1.1 * math.log1p(max(bandwidth, 0.0)) / math.log1p(max_bandwidth)


def _topology_lines(  # noqa: PLR0913
    design: dict[str, Any],
    colors: dict[int, tuple[float, float, float]],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    width: float = 8.0,
    height: float = 4.0,
    variant: Literal["full", "mini"] = "full",
    prefix: str = "",
    show_core_area: bool = True,
) -> list[str]:
    """Draw commands for one accelerator, into the rect of `width` x `height` whose lower-left is `origin`.

    Shared by the standalone topology figure and the insets of the combined figure, so an accelerator is drawn
    the same way wherever it appears. The coordinates are the ones `layout_positions` computed and
    `topologies.json` carries, so a core also sits where the PNG and the HTML report put it: compute cores in
    dataflow columns, the offchip core and the bus hub in a band underneath.

    `mini` shrinks the nodes and drops the per-core captions: an inset has room for the *shape* of an
    accelerator, not for its text, which the inset header and the shared layer legend carry instead.

    Args:
        design: one entry of `topologies.json`.
        colors: layer id -> RGB, from `layer_colors` over every exported design so the palette is shared.
        origin: lower-left corner of the rect to draw into, in picture coordinates.
        prefix: prepended to every node name, so several designs can share one `tikzpicture`.
    """
    cores = design["cores"]
    if not cores:
        raise ValueError(f"_topology_lines: design {design['id']} has no cores.")
    mini = variant == "mini"
    core_style = "stream core mini" if mini else "stream core"
    offchip_style = "stream offchip mini" if mini else "stream offchip"
    bus_style = "stream bus mini" if mini else "stream bus"

    xs = [core["x"] for core in cores]
    ys = [core["y"] for core in cores]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    # An inset is bounded by a visible box, so the extreme cores are pulled in by a node radius to keep their
    # outlines inside it. The standalone figure has no box and keeps its historical full-bleed placement.
    margin = 0.30 if mini else 0.0
    box_width = max(0.1, width - 2 * margin)
    box_height = max(0.1, height - 2 * margin)

    def place(core: dict[str, Any]) -> tuple[float, float]:
        x = 0.5 if xmax == xmin else (core["x"] - xmin) / (xmax - xmin)
        y = 0.5 if ymax == ymin else (core["y"] - ymin) / (ymax - ymin)
        return origin[0] + margin + x * box_width, origin[1] + margin + y * box_height

    positions = {str(core["id"]): place(core) for core in cores}
    lines: list[str] = []
    max_bandwidth = max((edge["bandwidth"] for edge in design["edges"]), default=0.0)
    edge_scale = 0.6 if mini else 1.0
    for kind in ("bus", "link"):  # buses first: they are the backdrop the point-to-point links are read over
        for edge in design["edges"]:
            if edge["kind"] != kind:
                continue
            source, target = positions.get(str(edge["source"])), positions.get(str(edge["target"]))
            if source is None or target is None:
                continue
            style = "stream bus wire" if kind == "bus" else "stream link"
            width_pt = _edge_width(edge["bandwidth"], max_bandwidth) * edge_scale
            lines.append(
                f"  \\draw[{style}, line width={width_pt:.2f}pt] "
                f"({source[0]:.3f},{source[1]:.3f}) -- ({target[0]:.3f},{target[1]:.3f});"
            )

    lines.append("")
    for core in cores:
        x, y = positions[str(core["id"])]
        if core["kind"] == "offchip":
            label = "" if mini else "offchip"
            lines.append(f"  \\node[{offchip_style}] at ({x:.3f},{y:.3f}) {{{label}}};")
            continue
        if core["kind"] == "bus":
            bandwidth = core.get("bandwidth")
            if mini:
                label = ""
            else:
                label = "bus" if bandwidth is None else f"bus\\\\{bandwidth:.0f}"
            lines.append(f"  \\node[{bus_style}] at ({x:.3f},{y:.3f}) {{{label}}};")
            continue
        layers = core["layers"]
        fill = "streammerged" if len(layers) != 1 else f"streamlayer{layers[0]['id']}"
        node = f"{prefix}c{core['id']}"
        text = f"{core['id']}" if mini else f"C{core['id']}"
        lines.append(f"  \\node[{core_style}, fill={fill}] ({node}) at ({x:.3f},{y:.3f}) {{{text}}};")
        if mini:
            continue
        # What the core runs and, optionally, what it costs: two folds of the same schedule can share a wiring
        # and differ only in the per-tile design each reused core got, which area is the only visible trace of.
        caption = [_core_layer_label(layers)]
        if show_core_area and core.get("area") is not None:
            caption.append(f"$A$~{_tex_number(core['area'], 1)}")
        caption_text = "\\\\".join(line for line in caption if line)
        if caption_text:
            lines.append(f"  \\node[stream layer label, yshift=-2pt] at ({node}.south) {{{caption_text}}};")
    return lines


def topology_tikz(  # noqa: PLR0913
    design: dict[str, Any],
    *,
    colors: dict[int, tuple[float, float, float]] | None = None,
    width: float = 8.0,
    row_spacing: float = 1.35,
    tag: str | None = None,
    show_metrics: bool = True,
    show_core_area: bool = True,
) -> str:
    """One accelerator as a standalone `tikzpicture`.

    Args:
        design: one entry of `topologies.json`.
        colors: layer id -> RGB, from `layer_colors` over *all* exported designs so the palette is shared.
        width: picture width in cm; the height follows from the widest column.
        tag: short letter tying this topology to its point on the front.
        show_metrics: print the design's objectives above the drawing.
        show_core_area: print each core's area under it. Keep it on whenever two exported designs share a
            topology, since the per-core design is then the only thing telling them apart.
    """
    cores = design["cores"]
    if not cores:
        raise ValueError(f"topology_tikz: design {design['id']} has no cores.")
    colors = colors if colors is not None else layer_colors([design])

    per_column: dict[Any, int] = {}
    for core in cores:
        per_column[core["x"]] = per_column.get(core["x"], 0) + 1
    height = min(9.0, max(3.0, row_spacing * max(per_column.values())))

    lines = [
        f"% Generated by stream.visualization.tikz_export -- design {design['id']}.",
        "% Needs \\input{stream-tikz-defs} in the preamble.",
        "\\begin{tikzpicture}",
    ]
    lines += _topology_lines(
        design, colors, width=width, height=height, variant="full", show_core_area=show_core_area
    )

    if show_metrics:
        metrics = "; ".join(f"{objective}~{_tex_number(design[objective])}" for objective in OBJECTIVES)
        title = f"{design['nb_cores']} cores; {metrics}"
        if tag:
            title = f"\\textbf{{{tag}}} -- " + title
        lines.append("")
        lines.append(
            f"  \\node[font=\\scriptsize, anchor=south] at ({width / 2:.3f},{height + 0.30:.3f}) {{{title}}};"
        )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def _log_limits(values: list[float], pad: float = 0.06) -> tuple[float, float]:
    """Axis limits with a margin, in log space. The combined figure needs them written out rather than left to
    pgfplots: the leader lines are anchored at `ymax`, so the picture has to know where the axis box ends."""
    positive = [value for value in values if value > 0]
    if not positive:
        raise ValueError("_log_limits: a log axis needs at least one positive value.")
    low, high = min(positive), max(positive)
    span = math.log10(high) - math.log10(low) or 1.0
    return 10 ** (math.log10(low) - pad * span), 10 ** (math.log10(high) + pad * span)


def _layer_names(designs: list[dict[str, Any]]) -> dict[int, str]:
    names: dict[int, str] = {}
    for design in designs:
        for core in design["cores"]:
            for layer in core["layers"]:
                names.setdefault(layer["id"], layer["name"])
    return names


def _legend_lines(designs: list[dict[str, Any]], *, origin: tuple[float, float], width: float) -> list[str]:
    """The shared key for the insets: what a core's colour means, and what the two infrastructure shapes are.

    An inset has no room for per-core captions, so the colour is the only thing saying which layer a core runs;
    without this the insets are pretty and unreadable."""
    # Only the layers a drawn core is actually filled with: a fold can leave every core multi-layer and grey,
    # and a key to seven colours none of which appear is worse than no key at all.
    names = _layer_names(designs)
    shown = {core["layers"][0]["id"] for design in designs for core in design["cores"] if len(core["layers"]) == 1}
    entries: list[tuple[str, str]] = [
        (f"stream swatch, fill=streamlayer{layer_id}", _short_layer(name))
        for layer_id, name in sorted(names.items())
        if layer_id in shown
    ]
    if any(len(core["layers"]) > 1 for design in designs for core in design["cores"]):
        entries.append(("stream swatch, fill=streammerged", "folded (multi-layer)"))
    entries += [("stream offchip mini", "offchip"), ("stream bus mini", "bus")]

    # Fit the columns to the longest label rather than to a guess: layer names come from the ONNX graph, so
    # `conv1` and `layer4.0.downsample.0` are both possible and a fixed column width overruns on one of them.
    widest = max(len(label) for _, label in entries)
    column = min(width, 0.22 + 0.135 * widest + 0.35)
    per_row = max(1, min(len(entries), int(width // column)))
    column = width / per_row
    lines = ["", "  % Key for the insets: a core's fill is the layer it runs."]
    for index, (style, label) in enumerate(entries):
        x = origin[0] + (index % per_row) * column
        y = origin[1] - (index // per_row) * 0.42
        lines.append(f"  \\node[{style}] at ({x:.3f},{y:.3f}) {{}};")
        lines.append(f"  \\node[stream legend text, anchor=west] at ({x + 0.22:.3f},{y:.3f}) {{{label}}};")
    return lines


def pareto_topology_tikz(  # noqa: PLR0913
    records: list[dict[str, Any]],
    designs: list[dict[str, Any]],
    *,
    x_key: str = "latency",
    y_key: str = "area",
    color_key: str | None = "energy",
    colors: dict[int, tuple[float, float, float]] | None = None,
    tags: dict[str, str] | None = None,
    width: float = 16.0,
    axis_height: float = 6.5,
    inset_height: float = 4.0,
    inset_gap: float = 0.35,
    leader_gap: float = 0.5,
    y_label_band: float = 1.9,
    color_log: bool = True,
    show_legend: bool = True,
    float_env: str = "figure",
    caption_ready: bool = True,
) -> str:
    """The front and the accelerators behind it, as one `tikzpicture`.

    `pareto_tikz` and `topology_tikz` say the same things in separate floats, which makes the reader carry a
    letter across a page turn to find out what a point *is*. Here the insets sit in a row above the axis, each
    over its own point and tied to it by a leader, so the shape of the accelerator is read off the front
    directly: many small cores at the fast end, a handful of reused ones at the cheap end.

    The insets are ordered left to right by `x_key`, which is also the order of their points, so no two leaders
    cross. Each leader is drawn in two pieces -- a dotted rise from the point to the top of the axis box, then a
    straight run through the gap to the inset -- because an `axis` clips its own contents, and because a
    diagonal drawn across the data would be read as data.

    Args:
        records: every measured schedule, with a `pareto` flag (as `read_schedules` returns).
        designs: the designs to draw as insets, from `topologies.json`.
        x_key, y_key: the objectives on the axes; `color_key` is the third, on the colour bar (`None` drops it).
        colors: layer id -> RGB, from `layer_colors` over every available design so the palette is shared.
        tags: design id -> short tag. Default: `A`, `B`, ... in left-to-right order.
        width: total width of the picture in cm -- the inset row spans it, and the axis box gets what is left
            once the y-axis labels and the colour bar have taken their bands.
        y_label_band: room to leave left of the axis box for the y tick labels and the y label. Only an
            estimate matters: it decides how much of `width` the axis box keeps, and pgfplots measures the real
            labels itself. Raise it if the picture comes out overfull.
        float_env: `figure` or `figure*` -- the picture is wide, so a two-column paper wants the starred one.
    """
    front = [record for record in records if record["pareto"]]
    dominated = [record for record in records if not record["pareto"]]
    if not front:
        raise ValueError("pareto_topology_tikz: no record is flagged as pareto, nothing to draw.")
    if not designs:
        raise ValueError("pareto_topology_tikz: no design to draw as an inset.")
    colors = colors if colors is not None else layer_colors(designs)
    ordered = sorted(designs, key=lambda design: design[x_key])
    tags = tags or {design["id"]: TAGS[index % len(TAGS)] for index, design in enumerate(ordered)}

    # `scale only axis` sizes the axis *box*, so the y labels (left) and the colour bar (right) hang outside
    # it. Both bands come out of `width` and the box keeps the rest; otherwise the picture is `width` plus two
    # bands wide and runs off the text block. The axis box then starts at x=0 and the picture at -y_label_band.
    bar_band = 1.9 if color_key else 0.0
    axis_width = max(2.0, width - y_label_band - bar_band)
    row_left = -y_label_band
    xmin, xmax = _log_limits([record[x_key] for record in records] + [design[x_key] for design in ordered])
    ymin, ymax = _log_limits([record[y_key] for record in records] + [design[y_key] for design in ordered])
    staircase_x, staircase_y = pareto_staircase([(record[x_key], record[y_key]) for record in front])

    body = [
        "% Generated by stream.visualization.tikz_export -- needs \\input{stream-tikz-defs} in the preamble.",
        "\\begin{tikzpicture}",
        "  \\begin{axis}[",
        "    stream front axis,",
        # The insets are placed against the axis box, so it has to be the box that is sized and anchored --
        # `scale only axis` keeps `width`/`height` off the labels and the colour bar.
        f"    at={{(0cm,0cm)}}, anchor=south west, scale only axis=true,",
        f"    width={axis_width:.3f}cm, height={axis_height:.3f}cm,",
        f"    xlabel={{{_axis_label(x_key)}}}, ylabel={{{_axis_label(y_key)}}},",
        "    xmode=log, ymode=log,",
        f"    xmin={xmin:.6g}, xmax={xmax:.6g}, ymin={ymin:.6g}, ymax={ymax:.6g},",
    ]

    meta = (lambda value: math.log10(value)) if color_log else (lambda value: value)
    if color_key:
        metas = [meta(record[color_key]) for record in front]
        vmin, vmax = min(metas), max(metas)
        body += [
            "    colormap/viridis, colorbar,",
            f"    point meta min={vmin:.6g}, point meta max={vmax:.6g},",
        ]
        if color_log:
            ticks, labels = _log_colorbar_ticks(vmin, vmax)
            body += [
                f"    colorbar style={{font=\\scriptsize, ytick={{{','.join(f'{tick:.6g}' for tick in ticks)}}},",
                f"                     yticklabels={{{','.join(labels)}}},",
                f"                     ylabel={{{_axis_label(color_key)}}}, ylabel style={{font=\\scriptsize}}}},",
            ]
        else:
            body += [
                f"    colorbar style={{font=\\scriptsize, ylabel={{{_axis_label(color_key)}}},",
                "                     ylabel style={font=\\scriptsize}},",
            ]
    body += ["    legend pos=south west,", "  ]"]

    # Leaders first, so every marker and every letter is drawn over them rather than under.
    body += ["", "    % Each inset's point, and the rise that ties it to the inset above."]
    for design in ordered:
        tag = tags[design["id"]]
        x, y = design[x_key], design[y_key]
        body.append(f"    \\draw[stream drop] (axis cs:{x:.6g},{y:.6g}) -- (axis cs:{x:.6g},{ymax:.6g});")
        body.append(f"    \\coordinate (streamtop{tag}) at (axis cs:{x:.6g},{ymax:.6g});")

    body += [
        "",
        f"    % {len(dominated)} measured but dominated schedule(s): the backdrop the front is read against.",
        "    \\addplot[only marks, mark=*, mark size=1.0pt, draw=streamdominated, fill=streamdominated]",
        "      table {",
        "        x y",
        _table_rows(dominated, (x_key, y_key)),
        "      };",
        f"    \\addlegendentry{{dominated ({len(dominated)})}}",
        "",
        "    % Achievable boundary of this projection. A point on the 3D front can sit above it: that is the",
        "    % price it pays on the third objective.",
        "    \\addplot[draw=black!45, line width=0.6pt, dashed, forget plot] coordinates {",
        "      " + " ".join(f"({x:.6g},{y:.6g})" for x, y in zip(staircase_x, staircase_y)),
        "    };",
    ]

    marks = {"unrolled": "*", "rolled": "triangle*"}
    for kind in ("unrolled", "rolled"):
        kind_records = [record for record in front if record["kind"] == kind]
        if not kind_records:
            continue
        if color_key:
            rows = "\n".join(
                f"        {record[x_key]:.6g} {record[y_key]:.6g} {meta(record[color_key]):.6g}"
                for record in kind_records
            )
            body += [
                "",
                f"    \\addplot[scatter, only marks, scatter src=explicit, mark={marks[kind]}, mark size=1.9pt,",
                "      scatter/use mapped color={draw=black!70, fill=mapped color, line width=0.3pt}]",
                "      table[meta=c] {",
                "        x y c",
                rows,
                "      };",
            ]
        else:
            body += [
                "",
                f"    \\addplot[only marks, mark={marks[kind]}, mark size=1.9pt, draw=black!70,",
                f"      fill=stream{kind}] table {{",
                "        x y",
                _table_rows(kind_records, (x_key, y_key)),
                "      };",
            ]
        body.append(f"    \\addlegendentry{{{kind} ({len(kind_records)})}}")

    body += ["", "    % The letter sits below its point, the rise leaves upwards: the two never collide."]
    floor = 10 ** (math.log10(ymin) + 0.12 * (math.log10(ymax) - math.log10(ymin)))
    for design in ordered:
        tag = tags[design["id"]]
        # ...except for a point on the floor of the axis, where below is outside the box and would be clipped.
        # Left of it is still clear: the rise goes up and the front runs away to the right.
        placement = "anchor=east, xshift=-3pt" if design[y_key] < floor else "anchor=north, yshift=-2.5pt"
        body.append(
            f"    \\node[stream tag, {placement}] "
            f"at (axis cs:{design[x_key]:.6g},{design[y_key]:.6g}) {{{tag}}};"
        )
    body.append("  \\end{axis}")

    inset_width = (width - (len(ordered) - 1) * inset_gap) / len(ordered)
    header_height = 0.55
    row_bottom = axis_height + leader_gap
    body.append("")
    body.append("  % The accelerators behind the lettered points, in the order their points appear on the axis.")
    for index, design in enumerate(ordered):
        tag = tags[design["id"]]
        left = row_left + index * (inset_width + inset_gap)
        top = row_bottom + inset_height
        # The second line is set smaller than the first: two objectives in scientific notation do not fit on
        # one \tiny line this narrow, and letting it wrap would push the drawing out of the box.
        header = (
            f"\\textbf{{{tag}}}~ {design['nb_cores']} cores, {design['kind']}\\\\"
            f"{{\\fontsize{{4}}{{4.6}}\\selectfont {x_key}~{_tex_number(design[x_key], 1)}, "
            f"{y_key}~{_tex_number(design[y_key], 1)}}}"
        )
        body += [
            "",
            f"  \\node[stream inset, anchor=south west, minimum width={inset_width:.3f}cm,"
            f" minimum height={inset_height:.3f}cm] (streaminset{tag}) at ({left:.3f},{row_bottom:.3f}) {{}};",
            f"  \\node[stream inset title, anchor=north west, text width={inset_width - 0.3:.3f}cm]"
            f" at ({left + 0.15:.3f},{top - 0.12:.3f}) {{{header}}};",
        ]
        body += _topology_lines(
            design,
            colors,
            origin=(left, row_bottom + 0.1),
            width=inset_width,
            height=inset_height - header_height - 0.2,
            variant="mini",
            prefix=f"{tag}",
        )
        body.append(f"  \\draw[stream leader] (streamtop{tag}) -- (streaminset{tag}.south);")

    if show_legend:
        body += _legend_lines(ordered, origin=(row_left, -1.55), width=width)
    body.append("\\end{tikzpicture}")

    picture = "\n".join(body)
    if not caption_ready:
        return picture + "\n"
    third = f", with {AXES[color_key].split(' [')[0].lower()} on the colour bar" if color_key else ""
    caption = (
        f"The schedule front projected on {AXES[x_key].split(' [')[0].lower()} and "
        f"{AXES[y_key].split(' [')[0].lower()}{third}, and the accelerators behind {len(ordered)} of its points. "
        f"Circles are unrolled schedules (one core per layer and inter-core group), triangles the same schedules "
        f"folded onto fewer, reused cores; grey points are measured but dominated, and the dashed staircase is "
        f"the achievable boundary of this projection. In each inset a circle is a compute core, filled with the "
        f"layer it runs and grey once a fold gave it several, the rounded box is the offchip core and the "
        f"diamond the shared bus; line width is log-scaled bandwidth."
    )
    return (
        f"\\begin{{{float_env}}}[t]\n  \\centering\n"
        + "\n".join("  " + line if line else "" for line in picture.splitlines())
        + f"\n  \\caption{{{caption}}}\n  \\label{{fig:stream-front-topologies}}\n\\end{{{float_env}}}\n"
    )


def designs_table_tex(designs: list[dict[str, Any]], tags: dict[str, str]) -> str:
    """A booktabs table of the tagged designs, so the figure letters and the numbers cannot drift apart."""
    rows = []
    for design in designs:
        rows.append(
            "    "
            + " & ".join(
                [
                    tags.get(design["id"], ""),
                    _tex_escape(design["id"]),
                    design["kind"],
                    str(design["nb_cores"]),
                    _tex_number(design["latency"]),
                    _tex_number(design["energy"]),
                    _tex_number(design["area"]),
                ]
            )
            + r" \\"
        )
    return (
        "% Generated by stream.visualization.tikz_export.\n"
        "\\begin{table}[t]\n  \\centering\n  \\small\n"
        "  \\begin{tabular}{llrrrrr}\n    \\toprule\n"
        "    & Design & Kind & Cores & Latency [cycles] & Energy [pJ] & Area [a.u.] \\\\\n"
        "    \\midrule\n" + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{The designs drawn in Fig.~\\ref{fig:stream-topologies}, marked on the front in "
        "Fig.~\\ref{fig:stream-pareto}.}\n  \\label{tab:stream-designs}\n\\end{table}\n"
    )


def _preview_tex(  # noqa: PLR0913
    pareto_name: str,
    topologies: list[tuple[str, str, dict[str, Any]]],
    table_name: str,
    combined_name: str | None = None,
) -> str:
    """The standalone check document.

    A full-width topology is tall, so the set is split over several floats rather than one -- `\\ContinuedFloat`
    keeps them one numbered figure with running subcaptions, which is also how they want to be laid out in a
    paper once there are more than two of them."""
    caption = (
        "Accelerators behind the lettered points of Fig.~\\ref{fig:stream-pareto}. Compute cores are circles "
        "coloured by the layer they run (grey once a folded core runs several), the rounded box is the offchip "
        "core and the diamond the shared bus; line width is log-scaled bandwidth. Each core is captioned "
        "with the layers it runs and its area $A$, which is what tells apart two folds that landed on "
        "the same wiring with different per-tile designs."
    )
    per_float = 2
    chunks = [topologies[index : index + per_float] for index in range(0, len(topologies), per_float)]
    floats = []
    for chunk_index, chunk in enumerate(chunks):
        subfigures = "".join(
            f"  \\begin{{subfigure}}{{\\linewidth}}\n    \\centering\n"
            f"    \\input{{{name}}}\n"
            f"    \\caption{{{_tex_escape(design['id'])}: {design['nb_cores']} core(s), "
            f"{design['kind']}.}}\n"
            f"  \\end{{subfigure}}\\par\\medskip\n"
            for _, name, design in chunk
        )
        head = "\\begin{figure}[p]\n  \\ContinuedFloat\n" if chunk_index else "\\begin{figure}[p]\n"
        tail = (
            f"  \\caption{{{caption}}}\n  \\label{{fig:stream-topologies}}\n"
            if chunk_index == len(chunks) - 1
            else "  \\caption{(continued below)}\n"
        )
        floats.append(head + "  \\centering\n" + subfigures + tail + "\\end{figure}\n")
    return (
        "% Generated by stream.visualization.tikz_export: compile with pdflatex to check the fragments.\n"
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=2cm,a4paper]{geometry}\n"
        "\\usepackage{subcaption}\n"
        "\\input{stream-tikz-defs}\n"
        "\\begin{document}\n\n"
        + (f"\\input{{{combined_name}}}\n\\clearpage\n\n" if combined_name else "")
        + f"\\input{{{pareto_name}}}\n\n"
        f"\\input{{{table_name}}}\n\n" + "\n".join(floats) + "\n\\end{document}\n"
    )


def write_figures(  # noqa: PLR0913
    search_dir: str,
    out_dir: str | None = None,
    *,
    x_key: str = "latency",
    y_key: str = "area",
    color_key: str = "energy",
    design_ids: list[str] | None = None,
    nb_topologies: int = 4,
    width: float = 8.0,
    color_log: bool = True,
    show_core_area: bool = True,
    combined_width: float = 16.0,
    combined_axis_height: float = 6.5,
    combined_inset_height: float = 4.0,
    combined_float_env: str = "figure",
) -> dict[str, str]:
    """Write the whole figure set for one search directory. Returns `{kind: path}` of what was written.

    Args:
        search_dir: a stage output directory holding `schedules.csv` and `topologies.json`.
        out_dir: where the `.tex` goes (default: `<search_dir>/tikz`).
        design_ids: which designs to draw, by id. Default: `select_designs` picks `nb_topologies` of them.
        combined_width: axis and inset-row width of the combined figure, in cm.
        combined_float_env: `figure` or `figure*` for the combined figure -- it is wide, so a two-column paper
            wants the starred one. The preview document is single-column, so it always gets `figure`.
    """
    out_dir = out_dir or os.path.join(search_dir, "tikz")
    os.makedirs(out_dir, exist_ok=True)
    records = read_schedules(os.path.join(search_dir, "schedules.csv"))
    available = read_designs(os.path.join(search_dir, "topologies.json"))
    by_id = {design["id"]: design for design in available}

    if design_ids:
        missing = [design_id for design_id in design_ids if design_id not in by_id]
        if missing:
            raise KeyError(f"write_figures: {missing} not in topologies.json (have {sorted(by_id)}).")
        designs = sorted((by_id[design_id] for design_id in design_ids), key=lambda design: design["latency"])
    else:
        designs = select_designs(available, nb_topologies)
    tags = {design["id"]: TAGS[index % len(TAGS)] for index, design in enumerate(designs)}
    colors = layer_colors(available)

    written: dict[str, str] = {}

    def write(name: str, content: str, key: str) -> None:
        path = os.path.join(out_dir, name)
        with open(path, "w") as handle:
            handle.write(content)
        written[key] = path

    write("stream-tikz-defs.tex", defs_tex(available), "defs")
    pareto_name = f"pareto_{x_key}_{y_key}.tex"
    write(
        pareto_name,
        pareto_tikz(records, x_key=x_key, y_key=y_key, color_key=color_key, tags=tags, color_log=color_log),
        "pareto",
    )
    topologies: list[tuple[str, str, dict[str, Any]]] = []
    for design in designs:
        tag = tags[design["id"]]
        name = f"topology_{tag}.tex"
        write(
            name,
            topology_tikz(design, colors=colors, width=width, tag=tag, show_core_area=show_core_area),
            f"topology_{tag}",
        )
        topologies.append((tag, name[: -len(".tex")], design))
    write("designs.tex", designs_table_tex(designs, tags), "table")
    combined_name = f"pareto_topologies_{x_key}_{y_key}.tex"
    write(
        combined_name,
        pareto_topology_tikz(
            records,
            designs,
            x_key=x_key,
            y_key=y_key,
            color_key=color_key,
            colors=colors,
            tags=tags,
            width=combined_width,
            axis_height=combined_axis_height,
            inset_height=combined_inset_height,
            color_log=color_log,
            float_env=combined_float_env,
        ),
        "combined",
    )
    write(
        "figures.tex",
        _preview_tex(pareto_name[: -len(".tex")], topologies, "designs", combined_name[: -len(".tex")]),
        "preview",
    )

    logger.info(
        f"write_figures: wrote {len(written)} LaTeX fragment(s) to {out_dir} "
        f"({len(records)} schedule(s), {len(designs)} topology figure(s))."
    )
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("search_dir", help="stage output directory with schedules.csv and topologies.json")
    parser.add_argument("-o", "--out-dir", default=None, help="where to write the .tex (default: <dir>/tikz)")
    parser.add_argument("-x", default="latency", choices=OBJECTIVES, help="objective on the x axis")
    parser.add_argument("-y", default="area", choices=OBJECTIVES, help="objective on the y axis")
    parser.add_argument("-c", "--color", default="energy", choices=OBJECTIVES, help="objective on the colour bar")
    parser.add_argument("--linear-colorbar", action="store_true", help="map the colour bar linearly, not in log")
    parser.add_argument("-d", "--design", action="append", default=None, help="design id to draw (repeatable)")
    parser.add_argument("-n", "--nb-topologies", type=int, default=4, help="how many designs to draw")
    parser.add_argument("--width", type=float, default=8.0, help="topology picture width in cm")
    parser.add_argument("--no-core-area", action="store_true", help="drop the per-core area captions")
    parser.add_argument(
        "--combined-width", type=float, default=16.0, help="combined figure width in cm (axis and inset row)"
    )
    parser.add_argument("--combined-axis-height", type=float, default=6.5, help="combined figure axis height in cm")
    parser.add_argument("--combined-inset-height", type=float, default=4.0, help="combined figure inset height in cm")
    parser.add_argument(
        "--wide", action="store_true", help="wrap the combined figure in figure* (two-column papers)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    written = write_figures(
        args.search_dir,
        args.out_dir,
        x_key=args.x,
        y_key=args.y,
        color_key=args.color,
        design_ids=args.design,
        nb_topologies=args.nb_topologies,
        width=args.width,
        color_log=not args.linear_colorbar,
        show_core_area=not args.no_core_area,
        combined_width=args.combined_width,
        combined_axis_height=args.combined_axis_height,
        combined_inset_height=args.combined_inset_height,
        combined_float_env="figure*" if args.wide else "figure",
    )
    for key, path in written.items():
        print(f"{key:<12} {path}")


if __name__ == "__main__":
    main()
