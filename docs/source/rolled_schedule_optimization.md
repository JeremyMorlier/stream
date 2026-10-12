# Joint hardware/schedule optimization for the rolled search

This note formalizes an alternative to the last two steps of the rolled methodology. Those steps are the unrolled
schedule branch-and-bound (`ScheduleExplorationStage._search`) and the rolling heuristic
(`RolledScheduleExplorationStage._roll`). In their place, a single optimization model decides four things at once:

- how many physical cores the accelerator has;
- which design each core gets;
- which layer groups share a core;
- when each piece of work runs.

The model's Pareto front over **latency, energy and area** is then re-measured with the real Stream scheduler.

The note is organized as follows:

1. What the existing stages already provide.
2. The decisions shared by every level of detail.
3. The three levels of detail: layer group, chunk and tile.
4. How to extract the Pareto front.
5. How to add, as constraints, the behaviour that Stream simulates and the model would otherwise miss.
6. Model sizes, integration and validation.

---

## 1. Inputs available after the rolled warm-up

Everything below is computed before the search starts, by the per-tile core GA and the full-matrix cost-model
warm-up (`RolledScheduleExplorationStage._warm_up_cme_cache`).

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $\mathcal{T}$ | tiles of the tiled workload (computation nodes) | `TiledWorkloadGenerationStage` |
| $E \subseteq \mathcal{T}\times\mathcal{T}$ | tile dependencies, with $b_{tu}$ bits transferred on edge $(t,u)$ | edges of the tiled workload, attribute `bits` |
| $\mathcal{G}$ | layer groups: (original layer, inter-core group) pairs, i.e. the cores of the unrolled topology | `derive_group_dedicated_topology(...).node_core_ids` |
| $\mathcal{T}_g$ | tiles of group $g$ | tile pinning by the group-dedicated topology |
| $s(t)$ | shape class of tile $t$ | `shape_class_of_tile` |
| $\mathcal{D}$ | candidate core designs: the union of the per-tile Pareto fronts | `_distinct_designs()` |
| $\ell_{s,d},\ e_{s,d}$ | latency (`latency_total2`) and energy (`energy_total`) of one tile of shape $s$ on design $d$ | `runtime_cache`, `energy_cache` |
| $a_d$ | area of design $d$ | `area_cache` |
| $M_d^{o}$ | capacity of design $d$'s top memory serving operand $o$ | the design's core dict |
| $B^{\text{link}},\ B^{\text{bus}}$ | point-to-point link and shared-bus bandwidths (bits/cycle) | accelerator template (32 and 128 in `tpu_like_quad_core`) |
| $\epsilon^{\text{link}},\ \epsilon^{\text{bus}},\ \epsilon^{\text{dram}}$ | transfer energy per bit | communication link / off-chip core models |

Derived per-group quantities, since every tile of a group runs on the same design:

$$
L_{g,d} = \sum_{t\in\mathcal{T}_g} \ell_{s(t),d}, \qquad
E_{g,d} = \sum_{t\in\mathcal{T}_g} e_{s(t),d}, \qquad
\text{bits}_{gh} = \sum_{(t,u)\in E,\ t\in\mathcal{T}_g,\ u\in\mathcal{T}_h} b_{tu}.
$$

Only pairs $(g,d)$ whose shapes all pass `lowest_memory_level_violations` are allowed; the set of allowed pairs is
$\mathcal{A}$. The model treats every other pair as forbidden.

---

## 2. Decisions shared by every level

Let $P = |\mathcal{G}|$ be the number of physical core *slots*. A rolled accelerator never has more cores than the
unrolled one, so $P$ slots are enough. Index the slots $p \in \{1,\dots,P\}$.

| Variable | Domain | Meaning |
|---|---|---|
| $u_p$ | $\{0,1\}$ | slot $p$ is a physical core |
| $y_{p,d}$ | $\{0,1\}$ | core $p$ gets design $d$ |
| $x_{g,p}$ | $\{0,1\}$ | group $g$ runs on core $p$ |
| $z_{g,p,d}$ | $\{0,1\}$ | group $g$ runs on core $p$, which has design $d$; linearizes $x_{g,p}\,y_{p,d}$ |

Structural constraints:

$$
\sum_{p} x_{g,p} = 1 \quad \forall g, \qquad
\sum_{d} y_{p,d} = u_p \quad \forall p, \qquad
x_{g,p} \le u_p,
$$

$$
z_{g,p,d} \le x_{g,p},\quad z_{g,p,d} \le y_{p,d},\quad z_{g,p,d} \ge x_{g,p} + y_{p,d} - 1,
\qquad z_{g,p,d} = 0 \ \text{if}\ (g,d)\notin\mathcal{A}.
$$

**Symmetry breaking.** Without it, every relabelling of the slots is a separate solution. Require:

- used slots come first: $u_p \ge u_{p+1}$;
- slots are filled in group order: group $g$ (in a fixed order) may only use slots $p \le g$. This is the standard
  bin-packing symmetry break.

**Area.** This is exact, since area depends only on the chosen designs:

$$
A = \sum_{p}\sum_{d} a_d\, y_{p,d}.
$$

**Compute energy.** This is exact up to the cost model, since a tile's energy depends only on its design:

$$
E^{\text{comp}} = \sum_{g}\sum_{p}\sum_{d} E_{g,d}\, z_{g,p,d}.
$$

The three levels below differ only in **how time is modelled**, i.e. in what a schedulable task is.

---

## 3. Three levels of detail

### Level 1: one task per layer group

**Tasks.** One task per group $g$, with:

- start time $S_g$;
- duration $\lambda_g = \sum_{p,d} L_{g,d}\, z_{g,p,d}$;
- end time $C_g = S_g + \lambda_g$.

**Precedence.** For every pair of groups with traffic ($\text{bits}_{gh} > 0$, $g$ before $h$), $h$ starts after $g$
ends, plus a transfer delay when the two groups sit on different cores:

$$
S_h \ \ge\ C_g + \frac{\text{bits}_{gh}}{B^{\text{link}}}\, \delta_{gh},
\qquad
\delta_{gh} \ge x_{g,p} - x_{h,p}\ \ \forall p.
$$

Here $\delta_{gh} = 1$ iff $g$ and $h$ are on different cores.

**No overlap on a core.** Two groups sharing a core cannot run at the same time. With order variables
$o_{gh}\in\{0,1\}$ and a big constant $H$ (an upper bound on the makespan), for every pair $g, h$ and slot $p$:

$$
S_h \ge C_g - H\,(1 - o_{gh}) - H\,(2 - x_{g,p} - x_{h,p}),\qquad
S_g \ge C_h - H\, o_{gh} - H\,(2 - x_{g,p} - x_{h,p}).
$$

In a constraint-programming solver (OR-Tools CP-SAT) this becomes one `NoOverlap` per slot over *optional*
interval variables `interval[g,p]`, present iff $x_{g,p}=1$. That formulation is much stronger than big-M and is
the recommended way to write it.

**Latency.**

$$
T = \max_g C_g \qquad (T \ge C_g\ \ \forall g).
$$

**What this level misses.** Under layer fusion, Stream interleaves the tiles of consecutive layers: a consumer
layer starts as soon as its first input tiles exist, not when the whole producer layer ends. Level 1 serializes
whole layers, so it **over-estimates latency**. It also cannot let two groups on one core interleave. It is,
however, small and fast to solve (§6). It suits layer-by-layer mode, and works as a quick first front in fused
mode.

### Level 2: chunks of a group

**Tasks.** Split each group's tiles $\mathcal{T}_g$, in Stream's scheduling order (`SchedulingOrderGenerationStage`),
into $K$ contiguous chunks $c = (g,k)$, $k = 1..K$. Each chunk inherits its group's core:

$$
\lambda_{c} = \sum_{p,d} \Big(\sum_{t\in c} \ell_{s(t),d}\Big) z_{g,p,d},
$$

**Dependencies.**

- *Inside a group*, chunks run in order: $S_{(g,k+1)} \ge C_{(g,k)}$.
- *Across groups*, chunk $c'$ of $h$ depends on chunk $c$ of $g$ iff some tile of $c'$ has a predecessor tile in
  $c$:

$$
S_{c'} \ \ge\ C_{c} + \frac{\text{bits}_{cc'}}{B^{\text{link}}}\,\delta_{gh}
\quad \forall (c,c') \text{ with } \text{bits}_{cc'} = \sum_{t\in c,\,u\in c'} b_{tu} > 0 .
$$

  This is what recovers fusion. Consumer chunk 1 only waits for the producer chunks it actually reads, so
  consecutive layers pipeline at chunk granularity.

**No overlap.** One `NoOverlap` per slot over the optional chunk intervals. Chunks of *different* groups on the
same core can now alternate, as Stream's fused schedules do.

**Latency** is $T = \max_c C_c$.

**Choosing $K$.** $K=1$ is level 1, and $K = |\mathcal{T}_g|$ is level 3. A good default is to match the fusion
granularity: make $K$ the number of distinct "rows" of the fused stack's tiling, i.e. the number of tiles of the
smallest-tiled layer in the stack. A larger $K$ only refines accuracy, so $K$ can be increased until the model's
front stops moving relative to the re-measured one.

### Level 3: one task per tile

Each tile $t$ is its own task, with duration $\sum_{p,d} \ell_{s(t),d}\, z_{g(t),p,d}$. It has precedence on every
edge $(t,u)\in E$ (plus the transfer delay when the producer and consumer groups are on different cores) and one
`NoOverlap` per slot.

This is closest to Stream, which schedules tiles. But with about 2,000 tiles and up to 25 slots it means:

- about $5\cdot 10^4$ optional intervals;
- a precedence for every edge of the tiled graph.

Even CP-SAT struggles to prove anything at that size. It is useful only for small workloads, or as a *local*
improvement step: fix the hardware decisions $(u,y,x)$ from a level-2 solution, then solve the tile-level schedule
alone. That is pure scheduling, with no hardware choice left.

---

## 4. Extracting the Pareto front

The three objectives are:

- latency $T$;
- energy $E = E^{\text{comp}} + E^{\text{comm}}$, with $E^{\text{comm}}$ from §5;
- area $A$.

They are minimized with the **ε-constraint method**.

1. **Anchor points.** Minimize each objective alone, breaking ties lexicographically on the other two. This gives
   the three extreme points and the ranges $[T^{\min}, T^{\max}]$, $[E^{\min}, E^{\max}]$, $[A^{\min}, A^{\max}]$.
2. **Grid.** Area takes few distinct values (it is a sum of design areas), and energy is the second objective. So
   for each area budget $\bar A$ on a grid over $[A^{\min}, A^{\max}]$ and each energy budget $\bar E$ on a grid
   over $[E^{\min}, E^{\max}]$, solve
   $$\min T \quad \text{s.t.}\quad A \le \bar A,\ \ E \le \bar E,$$
   then a tie-break $\min E$ at the optimal $T$, and $\min A$ after that, so every point is non-dominated.
3. **Warm starts.** Warm-start each solve from the neighbouring grid point. CP-SAT accepts hints, and Gurobi MIP
   starts.
4. **Re-measure.** Measure every distinct solution with the real scheduler. This is exactly what
   `RolledScheduleExplorationStage._measure_fold` does from a `merge_of_core` and a `core_dict_of_core`: here
   $x$ gives the merge and $y$ the designs. Insert the measured points into the existing `ParetoArchive`. **The
   reported front is the measured one.** The model's own values only decide which accelerators get measured.

Recording the model's predicted $(T, E)$ next to the measured values for every point gives the model error, which
is the evidence for choosing the level of detail.

---

## 5. Adding what Stream simulates as constraints

Stream's scheduler (`stream/cost_model/scheduler.py`) does more than order tasks. It:

- moves tensors over links and the bus with contention;
- keeps tensors in finite core memories, evicting them when space runs out;
- reads and writes off-chip memory.

Each of these can be added to the model, at increasing cost. All of them apply at levels 2 and 3, and the first two
also at level 1.

### 5.1 Transfer latency and energy, with a chosen topology

The rolled search derives links from traffic (`derive_rolled_topology`). In the model, a link between cores $p$
and $q$ exists when two groups that communicate sit on them:

$$
\kappa_{pq} \ \ge\ x_{g,p} + x_{h,q} - 1 \qquad \forall (g,h):\ \text{bits}_{gh} > 0,\ p\ne q .
$$

- **Without** a link, transfers would go over the bus. Make the delay term conditional: use
  $\text{bits}/B^{\text{link}}$ when the link exists, and $\text{bits}/B^{\text{bus}}$ otherwise.
- **If links cost area** (a per-link area $a^{\text{link}}$), add $a^{\text{link}}\sum_{p<q}\kappa_{pq}$ to $A$. The
  model can then trade bandwidth for area, which the current rolled search cannot.

Communication energy:

$$
E^{\text{comm}} = \sum_{(g,h)} \text{bits}_{gh}\,\epsilon^{\text{link}}\,\delta_{gh}
\qquad(\text{plus } \epsilon^{\text{bus}} \text{ for transfers routed over the bus}).
$$

### 5.2 Contention on links and the bus

Transfers can share a link, and every core shares the bus, so transfers must not exceed bandwidth.

- **Explicit transfers.** Make each cross-core transfer $(c\to c')$ an optional interval $\tau_{cc'}$ with a
  bandwidth demand $\beta_{cc'}$ and a length of $\text{bits}_{cc'}/\beta_{cc'}$. It is present iff the producer
  and consumer are on different cores. The consumer then waits for it: $S_{c'} \ge \text{end}(\tau_{cc'})$.
- **Links and the bus.** Stream's `CommunicationLink` lets transfers overlap as long as their summed bandwidth
  stays within the link's: `get_idle_window` searches for a window where the active bandwidth plus the request is
  at most `bandwidth`. So each link (and the bus) is a `Cumulative` resource:
  $$\texttt{Cumulative}\big(\{\tau \text{ routed on the link}\},\ \{\beta_\tau\},\ B^{\text{link}}\big),$$
  with routing decided by the $\kappa$ variables. With a fixed $\beta = B$ per transfer, this reduces to a
  `NoOverlap`, which is a simpler but more pessimistic variant.
- **Broadcasts.** Stream reuses an earlier transfer of the same tensor over the same link
  (`previously_seen_tensors`). So a tensor consumed by several chunks on the same remote core is transferred once:
  at chunk granularity, merge such transfers into one interval that every consumer waits for.

This adds one interval per communicating chunk pair. It is the first refinement worth making at level 2, because
in the profiles the link-window search is the second-largest cost in Stream's own scheduler. Contention matters.

### 5.3 Core memory capacity, eviction and spill

This is the effect that made folded accelerators expensive to schedule. A core hosting several layers must keep
their tensors in a finite memory.

**Tensor lifetimes.** For each tensor $\theta$ that a chunk produces or consumes (inputs, weights, outputs, at
chunk granularity), define on the core holding it:

- a **lifetime interval** from its arrival (production, or the end of its incoming transfer) to its last use;
- a size $\sigma_\theta$ and an operand $o(\theta)$.

**Capacity as a cumulative resource.** For each slot $p$ and operand $o$, the live tensors must fit in the chosen
design's top memory:

$$
\texttt{Cumulative}\Big(\{\text{life}_\theta \text{ on } p\},\ \{\sigma_\theta\},\ \sum_d M^{o}_d\, y_{p,d}\Big).
$$

The capacity depends on the design decision, which CP-SAT supports with a variable capacity. Lifetimes stretch
with the schedule, so this couples memory and time exactly as Stream's `MemoryManager` does.

**Eviction (spill) as a choice.** When a tensor does not fit, Stream evicts it to off-chip memory and fetches it
again before its next use. To model this, give each tensor a binary $\text{spill}_\theta$:

- if $\text{spill}_\theta = 1$, its lifetime on the core is split into two shorter intervals (until it is evicted,
  and from its re-fetch);
- the gap in between costs an off-chip write plus read: $2\sigma_\theta\,\epsilon^{\text{dram}}$ in energy, and
  re-fetch transfers on the bus (§5.2) in time.

The model then *chooses* what to spill, rather than following Stream's priority rule
(`find_best_tensor_combination_to_evict_fast`). It is an optimistic version of the same mechanism.

**Cheaper version.** Without lifetimes, a static bound per core and operand, using the peak working set of the
chunks a core hosts:

$$
\max_{c \text{ on } p} \text{ws}_c^{o} \ \le\ \sum_d M^{o}_d\, y_{p,d},
$$

This rejects only designs that cannot hold even one chunk. It is a feasibility filter, not a model of eviction.

### 5.4 Off-chip traffic

- Each graph input and each weight tensor has to be loaded from DRAM once per core that uses it, through the bus.
- Each graph output is written back.
- When groups that share weights land on one core, the weights are loaded once.

As constraints:

- load transfers are intervals on the bus resource (§5.2) that must end before the first consuming chunk starts;
- their energy is $\sigma\,\epsilon^{\text{dram}}$;
- spills (§5.3) add their own write and read transfers.

This captures the cost of rolling that the current heuristic ignores. A folded core hosting many layers reloads
more weights if its memory is small.

### 5.5 Layer fusion

Fusion is not a separate constraint. It is the **granularity of the dependencies**:

- at level 1 it is absent;
- at level 2 it is approximated by the chunk-to-chunk edges;
- at level 3 it is exact.

### 5.6 What stays in the re-measurement

Some behaviour is deliberately left to the real scheduler, which is why every point is re-measured (§4):

- Stream's greedy priority-based ordering. The model finds an optimal order instead, which can be better or
  worse than what Stream would do.
- The exact eviction choices.
- The per-tile cost-model details behind $\ell$ and $e$: memory stalls already sit inside `latency_total2`, so
  they are not double-counted.
- Any second-order effect of the data layout.

The gap between predicted and measured values is the price of these simplifications, and is reported per point.

---

## 6. Model sizes and integration

Orders of magnitude from the comparison runs. A rolled tiling candidate of the attention block had:

- 25 groups, i.e. unrolled cores, so $P \le 25$ slots (the seed tiling has 11);
- about 100 candidate designs;
- about 2,300 tiles.

| Level | Tasks | Placement booleans $z$ | Precedences | Practical with |
|---|---|---|---|---|
| 1: groups | 25 | ≤ 25·25·100 = 62,500 (sparse after $\mathcal{A}$ and symmetry breaking) | about 25–40 | CP-SAT or MILP, seconds to minutes per ε-point |
| 2: chunks, $K = 8$ | 200 | same $z$ (chunks inherit their group's placement) | a few hundred | CP-SAT with optional intervals, minutes per ε-point |
| 3: tiles | about 2,300 | same $z$ | several thousand | only as a fixed-hardware schedule refinement |

The hardware decisions $(u, y, x, z)$ do not grow with the level of detail. Only the time model does. So level 2
costs much less than its task count suggests.

**Pruning designs.** If 100 designs are too many, keep per group only those on the group's own
latency–energy–area front over $\mathcal{A}$. Designs dominated for every group can never be chosen.

**Where it fits in Stream.** A stage `ScheduleOptimizationStage(RolledScheduleExplorationStage)` would:

1. reuse `_prepare_search`, which provides the per-tile fronts, the full cost matrix and the tile pinning;
2. build the model at the chosen level, with $K$ and the §5 constraints selected by flags;
3. run the ε-constraint sweep (§4);
4. turn each solution into a fold plan (`merge_of_core` from $x$, `core_dict_of_core` from $y$), measure it with
   `_measure_fold`, and insert it into the `ParetoArchive`;
5. report through the existing CSVs and `topologies.html`.

It then drops in as the per-candidate stage of `optimize_rolled_tiling`, in place of
`RolledScheduleExplorationStage`.

**Validation plan.**

1. On the ResNet bottleneck, run level 1 and level 2 ($K \in \{4, 8, 16\}$) against the current rolled search, from
   the same per-tile fronts and cost matrix.
2. Compare:
   - the measured fronts (hypervolume against a shared reference);
   - the model's prediction error;
   - the wall time.
3. Add §5.1–5.2 first, then §5.3–5.4, and keep each only if it reduces the prediction error enough to change which
   accelerators get measured.
