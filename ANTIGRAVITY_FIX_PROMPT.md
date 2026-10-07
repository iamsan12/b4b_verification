# TASK: Fix the PRISM Role-C risk engine and dashboard so every number on screen is derived from real data

You are working in `C:\Users\sanik\OneDrive\Desktop\band4band`. This is the Role C
("Person C") half of a 3-role hardware hackathon project called PRISM
(Power-integrity Risk Identification, Slack-impact ranking, Mitigation).

Role A produces physical-design data (OpenROAD/Yosys). Role B produces an ML IR-drop
predictor. Role C (me) converts IR droop into timing risk, ranks paths, proposes
mitigations, runs design-space exploration, and places telemetry sensors. `app.py`
is a Streamlit dashboard that presents all of it.

An audit found that **most headline numbers on the dashboard are fabricated,
mis-scaled, or mis-attributed**. Your job is to fix all of them. Do not skip any
item. Do not "improve" anything not listed. Do not delete data files.

---

## GROUND RULES — read before writing any code

1. **Never invent, ramp, scale, or synthesize a value that exists in a real input
   file.** If a real column exists, use it. If it is missing or invalid, raise a
   clear exception naming the file and column. Silent fallback is banned.
2. **No magic multipliers.** Any scaling factor must come from a data file or a
   named, documented constant with a physical justification written next to it.
3. **Every number rendered in `app.py` must be read from a CSV**, never hardcoded
   in a caption, subtitle, or `st.write` string.
4. **If a fix makes a result look less impressive, that is the correct outcome.**
   Report the honest number. Do not compensate elsewhere.
5. Preserve all existing output CSV filenames and column names — other people
   consume them.
6. Python 3.14, pandas, numpy, plotly, streamlit, scikit-learn are installed.
   Do not add new dependencies.
7. Work through the files in the order given below. After each numbered item,
   re-run the pipeline and confirm the stated acceptance check passes before
   moving on.

---

## THE FILES

| File | Role |
|---|---|
| `prism_risk_engine.py` | main Role C engine; writes 5 output CSVs |
| `app.py` | Streamlit dashboard |
| `dse_tier3_bo.py` | Tier-3 Bayesian optimisation candidate selection |
| `dse_tier2_nsga2.py` | Tier-2 NSGA-II Pareto front |
| `dse_tier1_sobol.py` | Tier-1 Sobol sampling |
| `telemetry_sensor_placement.py` | greedy submodular sensor placement (B6 bonus) |
| `risk_engine.py` | OLD superseded copy — do not edit, but see item 14 |
| `io_csv.py` | Role B's schema validator |
| `DATA_SCHEMA (1).md` | authoritative CSV contract between roles |

### Real input files (do not modify these)

| File | Contents |
|---|---|
| `paths.csv` | 1601 rows. `path_id, endpoint, clock_domain, slack_ns, delay_ns, inst_ids`. `inst_ids` is semicolon-joined instance ids. Real slack range **2.16 → 11.93 ns, all positive**. `delay_ns` is **negative on every row** (-8.20 → -0.24). Single clock domain `clk_core`. 1491 unique endpoints. |
| `predictions.csv` | Role B contract. First line is a `#` comment — always read with `comment='#'`. 48384 rows = **14 designs (`syn_000`..`syn_013`) × 6 scenarios × 576 tiles**. Columns `design, scenario, partition, tile_id, pred_v, lo_v, hi_v, label_v, coarse_v`. `pred_v` is **Volts**. Scenarios: `idle, seq_read, seq_write, rand_read_4k, gc_compact, ecc_recover`. `tile_id = ty*24 + tx` on a 24×24 coarse grid. |
| `instances.csv` | 71251 rows, `inst_id, inst_name, module, cell_type, x_um, y_um, area_um2, ...`. `inst_id` 0..71250. Coordinates span x 4.275→439.945 µm, y 4.9→438.9 µm. |
| `design_stats.csv` | 1 row. `die_w_um=444.22, die_h_um=444.22, vdd_v=1.1, clock_period_ns=1.0, core_util=0.358, strap_pitch_um=30.0, strap_width_um=1.4, pdn_layers=M1-M4-M7, bump_pitch_um=nan`. design_id `orfs_ssd_ctrl_cfg_small`. |
| `ssd_ctrl_sweep_summary.csv` | Role A's real OpenROAD PnR results. `cfg_small` worst IR drop **1.7985 mV**; `cfg_run_01_p12` worst **0.3051 mV**. |
| `bo_runs.csv` | What Role A actually ran: `cfg_run_01→12 µm, cfg_run_02→24, cfg_run_03→24, cfg_run_04→16`. |
| `bo_proposals.csv` | What Tier-3 BO proposed: strap pitch **8 for all four** — contradicts `bo_runs.csv`. |
| `heatmaps/*.webp` | Real OpenROAD PDNSim renders, IR worst value encoded in filename. |

---

## PART 1 — `prism_risk_engine.py` (six defects, all blocking)

### 1.1 DELETE the fabricated slack ramp

Currently around lines 78–86, inside `load_real_or_mock_data`, after loading the real
`paths.csv` the code does:

```python
n_paths = len(df_paths)
slacks = np.zeros(n_paths)
for i in range(n_paths):
    if i < 12:
        slacks[i] = 0.020 + i * 0.008
    elif i < 30:
        slacks[i] = 0.150 + (i - 12) * 0.02
    else:
        slacks[i] = 0.600 + (i - 30) * 0.005
df_paths['slack_ns'] = slacks
```

This throws away Role A's real STA slack and replaces it with an arithmetic ramp
engineered to manufacture timing violations. **Delete this block entirely** along with
the surrounding "Demo Stress Test Mode" comment. Use `df_paths['slack_ns']` exactly
as it comes from `paths.csv`.

Also delete `df_paths['delay_ns'] = df_paths['delay_ns'].abs()`. See item 1.6 for what
to do instead.

**Acceptance:** `prism_ranked_risk_output.csv` `slack_ns` column must be a permutation
of `paths.csv` `slack_ns` (min 2.16, max 11.93, mean 7.2448). Assert this.

### 1.2 DELETE the 2.2× droop multiplier

Around lines 105–125 the loader applies `scale_factor = 2.2` to the "stress" scenario
and a bare `* 2.2` in three other branches. There is no physical basis for this. It
turns Role B's max prediction of 75.9 mV into a 150 mV droop on the rail — 16.6% IR
drop, which is not a real chip.

**Remove every `2.2` from this file.** Convert Volts to millivolts with `* 1000.0`
and nothing else.

**Acceptance:** `grep -n "2\.2" prism_risk_engine.py` returns nothing.

### 1.3 FIX the instance→tile mapping (currently anti-correlated with physics)

Line ~187 in `run_risk_engine`:

```python
drop = v_drop_map.get(idx, v_drop_map.get(idx % 576, 0.0))
```

and line ~270 in `apply_fix`:

```python
tile_id = idx % 576
```

`idx` is an instance id from `paths.csv` (values up to 28110). `idx % 576` has no
physical meaning. Measured: droop assigned this way correlates **r = −0.26** with the
droop at the tile the cell actually sits in. It is worse than random.

Build a real mapping from `instances.csv` and `design_stats.csv`:

```python
def build_inst_to_tile(instances_csv='instances.csv',
                       stats_csv='design_stats.csv',
                       nx=24, ny=24):
    st = pd.read_csv(stats_csv)
    die_w = float(st['die_w_um'].iloc[0])
    die_h = float(st['die_h_um'].iloc[0])
    if not (die_w > 0 and die_h > 0):
        raise ValueError("design_stats.csv: die_w_um / die_h_um must be > 0")

    inst = pd.read_csv(instances_csv)
    for c in ('inst_id', 'x_um', 'y_um'):
        if c not in inst.columns:
            raise ValueError(f"instances.csv: missing required column '{c}'")

    # DEF database-unit guard (see DATA_SCHEMA "UNIT SANITY")
    if inst['x_um'].max() > die_w * 1.01 or inst['y_um'].max() > die_h * 1.01:
        raise ValueError(
            "instances.csv coordinates exceed the die. Role A likely exported DEF "
            f"database units, not microns. max x={inst['x_um'].max()}, die_w={die_w}")

    tx = np.clip((inst['x_um'] / die_w * nx).astype(int), 0, nx - 1)
    ty = np.clip((inst['y_um'] / die_h * ny).astype(int), 0, ny - 1)
    return dict(zip(inst['inst_id'].astype(int), (ty * nx + tx).astype(int)))
```

Thread this dict through `run_risk_engine`, `apply_fix`, `measured_benefit_ns`,
`verify_joint_mitigation` and `compute_instance_risk_attribution` as an explicit
argument (do not use a module global). Look droop up as
`v_drop_map[inst_to_tile[inst_id]]`. If an `inst_id` from `paths.csv` is absent from
`instances.csv`, **raise** — do not default to 0.0. The current data has a 100%
match rate (2510/2510), so any miss is a real regression.

**Acceptance:** assert every `inst_id` appearing in `paths.csv` resolves to a tile in
`[0, 575]`, and that no lookup silently returns 0.0.

### 1.4 FIX the silent 14→1 design collapse

Line ~111:

```python
v_map_sc = dict(zip(df_sc['tile_id'].astype(int), df_sc['pred_v'] * 1000.0 * scale_factor))
```

`df_sc` holds 8064 rows for one scenario — 14 designs × 576 tiles. `dict(zip(...))` is
last-write-wins, so **13 of 14 designs are silently discarded** and only `syn_013`
survives, purely because of file ordering.

Furthermore `paths.csv` belongs to design `orfs_ssd_ctrl_cfg_small`, which is not any
of `syn_000..syn_013`. Applying an unrelated design's droop map to these timing paths
is unphysical.

Do this:

- Add a `design` parameter to the loader, default `None`.
- If `design` is given, filter `predictions.csv` to that design and assert exactly
  576 unique tile_ids per scenario.
- If `design` is `None`, aggregate across designs explicitly and **label it as such**:
  `df_sc.groupby('tile_id')['pred_v'].mean()` (or `.max()` for a worst-case view —
  pick one, name the variable accordingly, and record the choice in a
  `provenance` field written into the output CSVs).
- Either way, **assert the resulting map has exactly 576 keys covering 0..575**, and
  raise with a clear message if not.
- Emit a printed line stating exactly which designs went into the map and how they
  were combined. The dashboard must surface this too (see item 2.7).

**Acceptance:** the engine prints e.g. `[DROOP MAP] scenario=gc_compact, designs=14
(mean-aggregated), tiles=576/576` and asserts tile coverage.

### 1.5 FIX the row misalignment in `run_scenario_weighted_risk`

Around line 228:

```python
ranked = run_risk_engine(df_paths, sc_data['v_drop_map'])
ranked = ranked.sort_index()
expected_slack_loss += weight * ranked['droop_penalty_ns'].values
```

`run_risk_engine` ends with `.sort_values('effective_slack_ns').reset_index(drop=True)`,
so the subsequent `.sort_index()` is a **no-op** — it cannot undo a sort whose index was
already discarded. Penalties are therefore accumulated in *sorted* order and written
onto paths held in *original* order. Every row of
`prism_scenario_expected_risk.csv` is attributed to the wrong path.

Proof of the defect in the current output: endpoint
`g_chan[0].u_cdc_chr.rbin[0]$_DFF_PN0_` has `expected_penalty_ns = 0.032084` while its
own single-scenario penalty is `0.077566`.

Fix by removing the sort from the computation path. Refactor:

```python
def compute_penalties(df_paths, v_drop_map, inst_to_tile, ...):
    """Returns penalties aligned to df_paths' own row order. Does NOT sort."""
```

and have `run_risk_engine` call `compute_penalties`, then sort only for presentation.
`run_scenario_weighted_risk` must call `compute_penalties` directly.

**Acceptance:** add a regression test. For a single-scenario dict with weight 1.0,
`expected_penalty_ns` must equal `droop_penalty_ns` for every endpoint, matched by
endpoint not by position. Assert exact equality within 1e-12.

### 1.6 FIX the hardcoded electricals and handle the bad `delay_ns`

`calculate_delay_penalty` and `run_risk_engine` hardcode `nominal_v_mv=900,
vth_mv=300`. But `design_stats.csv` says `vdd_v = 1.1`. Read the supply from
`design_stats.csv` and pass it in. Keep `vth_mv` and `alpha` as named constants but
document them (`# nangate45 nominal Vth; alpha-power-law exponent`) and expose them as
function parameters so they can be swept.

`paths.csv` has `delay_ns` **negative on every row** and slack values up to 11.93 ns
against `clock_period_ns = 1.0` — a single-cycle setup path cannot have 11.9 ns of
slack on a 1 ns clock. The current code hides this with `.abs()`.

Do **not** hide it. Add a `validate_paths(df, stats)` function that runs at load and:

- raises if `delay_ns` has any negative values, with the message:
  `paths.csv: delay_ns is negative on N/1601 rows (min=-8.20). Role A must export
  positive path delay. Sign convention or column swap suspected.`
- raises if `max(slack_ns) > clock_period_ns`, with the message:
  `paths.csv: max slack_ns=11.93 exceeds clock_period_ns=1.0 from design_stats.csv.
  A single-cycle setup path cannot have slack greater than the period. Check units,
  SDC, or a slack/delay column swap.`
- warns (does not raise) if `bump_pitch_um` is NaN in `design_stats.csv`.

Then add a single, explicit, top-of-file override flag:

```python
# Role A's paths.csv currently violates the schema (see validate_paths).
# Set this to True ONLY to produce a runnable demo while Role A re-exports.
# When True, every output CSV gains provenance='UNVALIDATED_INPUT' and the
# dashboard renders a persistent warning banner.
ALLOW_UNVALIDATED_PATHS = False
```

When `True`, `abs()` the delay, proceed, and stamp `provenance='UNVALIDATED_INPUT'`
into every emitted CSV as a column. When `False` (the default), the run stops.

**Acceptance:** with the flag `False`, `python prism_risk_engine.py` exits non-zero
with the delay_ns message. With it `True`, it completes and every output CSV carries
the provenance column.

### 1.7 Scenario weights must not be uniform

Line ~112 sets `'weight': 1.0 / len(sc_names)` — six scenarios each at 1/6. The schema
(`DATA_SCHEMA (1).md`) specifies `activity.csv` with a `mission_weight` column,
constant per scenario, summing to 1.0. `idle` and `gc_compact` are not equally likely.

`activity.csv` is **not present** in this directory. So:

- Look for `activity.csv` in the working directory, then in
  `orfs_extracted/orfs_ssd_ctrl_cfg_small/`, then `new_files_extracted/`.
- If found: use `mission_weight`, assert it sums to 1.0 ± 1e-6 across scenarios, and
  assert every scenario in `predictions.csv` appears in it.
- If not found: keep uniform weights BUT set a module-level flag
  `SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = True`, print a loud warning, write a
  `weight_source` column into `prism_scenario_expected_risk.csv` with value
  `uniform_placeholder`, and make the dashboard display that (item 2.7).

---

## PART 2 — `app.py` (dashboard defects)

### 2.1 Delete the hardcoded caption

Line ~165:

```python
st.caption("💡 **Clock-skew staggering** yielded the highest ROI delay recovery (39.91 ps for cost = 10).")
```

`prism_measured_mitigation_catalog.csv` actually says clock-skew staggering yields
**748.82 ps** at cost 10. The 39.91 is stale from an older run.

Compute the best-ROI row from `df_catalog` at render time
(`benefit_ps / cost`, take the argmax) and format the caption from it. Never hardcode.

### 2.2 Fix the mislabelled KPI

Line ~89, KPI 4 "Optimal Pareto Delay Recovery" shows
`df_pareto['verified_true_ps'].max()` = 1328.2 ps. That is a **sum of recovered slack
across all at-risk paths**, but it sits beside KPI 3 "Max Droop Delay Penalty" which is
a **per-path** value. Side by side it reads as if one fix buys 1.3 ns on a single path.

Relabel to `Total Slack Recovered (all at-risk paths)` and add
`st.caption` stating the budget it corresponds to and how many paths are in the
at-risk set. Keep KPI 3 labelled `Max Per-Path Droop Penalty`.

### 2.3 Fix the sensor markers floating off the heatmap

Lines ~280–300. Axis ticks are built as strings rounded to 1 decimal:

```python
x_ticks = [round((i + 0.5) * (444.22 / gw), 1) for i in range(gw)]
...
x=[str(x) for x in x_ticks],
```

but sensor coordinates in `telemetry_sensor_placements.csv` are rounded to 2 decimals
(`263.76`, `347.05`, `97.17`). `'263.76' != '263.8'`, so **zero** sensor coordinates
match an axis category. Plotly silently appends them as new categories and the
diamonds are drawn in the wrong place.

Fix by using **numeric** axes throughout — pass `x=x_ticks` and `y=y_ticks` as floats,
drop every `str()` call, and plot the scatter with the raw float
`real_x_um` / `real_y_um`. Do not round to make strings match.

Also hardcoded `444.22` appears in three places in this block. Read `die_w_um` /
`die_h_um` from `design_stats.csv` once at the top of `app.py` and use that
everywhere, including the tab-4 subtitle text.

### 2.4 Stop plotting the wrong DSE file

Tab 3 plots `bo_proposals.csv` (`PDN_STRAP_PITCH_UM` = 8, 8, 8, 8) directly above a
table from `ssd_ctrl_sweep_summary.csv` showing the pitches Role A actually ran
(12, 24, 24, 16). `bo_runs.csv` confirms 12/24/24/16. The configs disagree too —
proposal rank 1 says `NUM_CH=4, DATA_W=128`; the run was `NUM_CH=2, DATA_W=32`.

Merge `bo_proposals.csv` with `bo_runs.csv` on `proposal_id` and plot **proposed vs
actually-run** strap pitch as a grouped bar chart, so the discrepancy is visible
rather than hidden. Add a caption naming the mismatch explicitly.

Also: the chart colours by `ECC_LANES`, which is `1` for all four proposals, producing
a one-entry legend. Drop the colour encoding or colour by something that varies.

### 2.5 Correct the "provably optimal" claim

Line ~267 (`app.py`) and line ~134 (`telemetry_sensor_placement.py`) both claim
`PROVABLY OPTIMAL (Exceeds 63.2% bound)`.

Greedy submodular maximisation gives a **(1 − 1/e) ≈ 63.2% approximation guarantee** —
the result is guaranteed to be *at least 63.2% as good as the unknown optimum*. It is
not "optimal", and comparing the *measured coverage fraction* against 63.2% is a
category error: those are two different quantities and the comparison is meaningless.

Replace both strings with wording along the lines of:
`Greedy submodular placement — guaranteed within (1−1/e) ≈ 63.2% of the optimal
K-sensor placement. Measured risk coverage: X% of total attributed risk.`

### 2.6 Fix overstated and stale copy

- Tab 3: "Explores **thousands** of floorplan and physical configuration choices" —
  `sobol_samples_tier1.csv` has 512 samples and `tier2_pareto_front.csv` has 5 rows.
  State the real counts, read from the files at render time.
- KPI 1 "Total Timing Paths Analyzed" counts 1601 rows but there are only 1491 unique
  endpoints and 1321 unique instance-lists. Either de-duplicate or label it
  `Timing path records (1491 unique endpoints)`.
- The fallback constants (`total_paths = 5`, `failing_paths = 2`, `max_penalty_ps =
  105.6`, `max_benefit_ps = 72.6`) silently render fake KPIs when a CSV is missing.
  Replace every fallback with `st.error(...)` naming the missing file, and render
  the KPI as `—`.
- Tab 1 chart title says "Real Timing Paths from Person A" — keep it only after item
  1.1 is done and the slack really is Person A's.

### 2.7 Add a provenance / integrity panel

Add a collapsible `st.expander("Data provenance & known caveats")` at the top of the
page, populated from the engine's outputs, showing:

- which design(s) the droop map came from and how they were combined (item 1.4)
- whether scenario weights are real `mission_weight` or a uniform placeholder (1.7)
- whether `ALLOW_UNVALIDATED_PATHS` was on for the run that produced the CSVs (1.6)
- the coarse grid size actually used (see item 3.2)
- the `vdd_v` and `clock_period_ns` read from `design_stats.csv`

If `provenance == 'UNVALIDATED_INPUT'` appears in any loaded CSV, also render a
persistent `st.warning` banner at the top of the page. This must not be dismissible.

---

## PART 3 — `telemetry_sensor_placement.py`

### 3.1 Sum, don't max, when building the tile risk grid

Line ~45:

```python
risk_grid[gy, gx] = max(risk_grid[gy, gx], risk_ps)
```

This keeps only the single worst instance per tile, then `run_greedy_sensor_placement`
sums those values and calls the total "Total Unmitigated Chip Risk". For a coverage
objective the tile value must be the **total** attributed risk in that tile:

```python
risk_grid[gy, gx] += risk_ps
```

### 3.2 Use the contract grid size

`load_grid_risk_map` defaults to `grid_size=(16, 16)`. The frozen contract with Role B
(`DATA_SCHEMA (1).md`) is a **24×24 coarse grid, 576 tiles**, and `tile_id = ty*24+tx`.
Two different grids in one pipeline means the risk map and the droop map do not line
up. Change the default to `(24, 24)` and derive it from a single shared constant
(`NX_COARSE = NY_COARSE = 24`) defined once and imported by both
`prism_risk_engine.py` and `telemetry_sensor_placement.py`.

Note this changes `prism_chip_risk_grid.csv` from 16×16 to 24×24. `app.py` reads that
file with `pd.read_csv` (header row present — that part is correct, leave it) and
derives `gh, gw` from `.shape`, so it will adapt automatically. Verify it does.

### 3.3 Remove the synthetic fallback grid

The `else` branch fabricates a risk grid with hardcoded hotspots
(`risk_grid[2,3] = 450.0` etc.) plus uniform noise when the real files are missing.
Replace with a raise naming the missing file. A demo must never silently show
invented hotspots.

### 3.4 Keep the sensor-count / radius choices honest

`num_sensors=4` and `sensor_radius=2.5` are unexplained. Move them to named constants
with a one-line rationale, and write the chosen values into
`telemetry_sensor_placements.csv` as columns so the dashboard can state them.

---

## PART 4 — `dse_tier3_bo.py` (the BO is fitted on random noise)

Lines 32–34:

```python
np.random.seed(42)
y = np.random.rand(len(X))          # <-- this is the objective
kernel = Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp.fit(X, y)
```

There is no objective function. The GP is trained on uniform random numbers, so
`bo_score` (0.156 / 0.3745 / 0.5986 / 0.732) is a ranking of random draws. Worse, the
acquisition is evaluated only at the training points `X`, so there is no exploration
at all — with a near-interpolating GP the LCB ordering collapses back to the ordering
of the random `y`.

The dashboard currently labels these "Tier 3 Bayesian Optimization Candidates" and
"Ready for handoff to Role A". That is not defensible.

Fix it properly:

1. **Define a real objective.** Use the risk engine as the surrogate target: for each
   candidate configuration, the objective is predicted risk — e.g. total droop-induced
   slack loss (ps) summed over at-risk paths, or worst-case per-path penalty. Import
   the scoring path from `prism_risk_engine.py`; do not duplicate the physics.
   If a candidate's knobs cannot be mapped to a droop map without a full Role A PnR
   run, then use `ssd_ctrl_sweep_summary.csv` (`ir_drop_worst_mv`, `power_mw`,
   `die_area_um2`) as the observed objective for the configurations that *have* been
   run, and fit the GP on those real observations.
2. **Evaluate the acquisition on unobserved candidates**, not on the training set.
   Generate the candidate pool from `sobol_samples_tier1.csv` (512 rows) or the
   Tier-2 Pareto front, exclude anything already measured, and rank those.
3. Keep LCB = `mu - 1.96*sigma` only if the objective is being **minimised**; state
   the direction in a comment and in the printed output.
4. If, after all this, there genuinely is not enough measured data to fit a meaningful
   GP (fewer than ~6 real observations — currently there are 6 rows in
   `ssd_ctrl_sweep_summary.csv`, of which 4 are BO runs), then **say so**: emit
   `bo_proposals.csv` with a `method` column set to `heuristic_ranking` rather than
   `bayesian_optimization`, rank by the measured objective directly, and make the
   dashboard label it "heuristic ranking over measured runs" instead of
   "Bayesian Optimization". An honest heuristic beats a fake GP.
5. Reconcile the strap-pitch discrepancy: the proposals say 8 µm, the runs used
   12/24/24/16. Determine which is correct — check `dse_tier2_nsga2.py` and
   `snap_to_grid` — and make `bo_proposals.csv` and `bo_runs.csv` agree, or document
   in the CSV why they differ.

---

## PART 5 — cleanup and consistency

### 5.1 `risk_engine.py` is a stale duplicate

`risk_engine.py` is an older copy of `prism_risk_engine.py` and carries the same class
of bugs. Nothing should import it. Confirm with grep, then either delete it or rename
to `risk_engine_DEPRECATED.py` with a header comment pointing at
`prism_risk_engine.py`. Do not leave two live engines.

### 5.2 Reconcile the two IR-drop scales

After the fixes, the dashboard will show Role B's predicted droop (tens of mV on
synthetic designs) in Tab 1 and Role A's measured PDNSim worst drop of **1.80 mV** on
the real `ssd_ctrl` design in Tab 3. These differ by roughly 30–80×.

This is a genuine, unresolved discrepancy between two teammates' data, not a coding
bug. Do not paper over it. Add an explicit note in the provenance panel (2.7) stating
both numbers, which design each came from, and that they are not directly comparable
because Role B's predictions are for the `syn_*` corpus and Role A's measurement is
for `orfs_ssd_ctrl_cfg_small`.

For reference, measured with the corrected pipeline:

| droop source | mean per-path penalty | max per-path penalty | violating paths |
|---|---|---|---|
| Role B `gc_compact`, unscaled, spatial tiles | 86.1 ps | 473.3 ps | **0 / 1601** |
| Role A measured 1.80 mV uniform | 5.33 ps | 15.61 ps | **0 / 1601** |

**With the real slack from `paths.csv`, there are ZERO violating paths under any
droop assumption.** The current dashboard's "8 Critical Violating Paths" exists only
because of the fabricated slack ramp. Report zero. Then make the demo about *how much
margin erodes under gc_compact* — which is a stronger and true story.

### 5.3 Honour the schema's feature-leakage ban

`DATA_SCHEMA (1).md` permanently bans `hash, config, design_id, ts_utc, lint_rc` and
any path/filename as model features. If any of the DSE or BO code touches those
columns as inputs, remove them.

---

## PART 6 — verification you must run before declaring done

Create `verify_prism.py` that runs end to end and asserts every item below. It must
exit non-zero on any failure. Paste its full output in your final message.

```
[ ] grep -c "2\.2" prism_risk_engine.py                    == 0
[ ] grep -c "% 576" prism_risk_engine.py                   == 0
[ ] grep -c "39.91" app.py                                 == 0
[ ] grep -c "PROVABLY OPTIMAL" telemetry_sensor_placement.py == 0
[ ] grep -c "np.random.rand(len(X))" dse_tier3_bo.py       == 0
[ ] set(ranked.slack_ns) == set(paths.slack_ns)            # slack is Role A's
[ ] ranked.slack_ns.min() == 2.16 and .max() == 11.93
[ ] (ranked.effective_slack_ns < 0).sum() == 0             # honest: zero violations
[ ] droop map has exactly 576 keys, covering 0..575
[ ] every paths.csv inst_id resolves to a tile in [0,575]; zero silent 0.0 lookups
[ ] scenario_expected_risk matched BY ENDPOINT to ranked_risk:
      single-scenario weight=1.0 -> expected_penalty_ns == droop_penalty_ns (atol 1e-12)
[ ] prism_chip_risk_grid.csv shape == (24, 24)
[ ] every sensor real_x_um in [0, die_w] and real_y_um in [0, die_h]
[ ] marginal_risk_covered_ps is monotonically non-increasing (submodularity)
[ ] app.py contains no numeric literal that also appears in any output CSV
      (i.e. no hardcoded results) — check at minimum: 39.91, 105.6, 72.6, 444.22, 1.80
[ ] nominal supply used by the engine == design_stats.csv vdd_v (1.1 V -> 1100 mV)
[ ] streamlit run app.py starts, all 4 tabs render, no exception in the console
```

Then actually launch it (`streamlit run app.py`), open every tab, and confirm:

- Tab 1: bars show real slack (2.16–11.93 ns range), zero red violations
- Tab 2: catalog caption number matches the CSV
- Tab 3: proposed-vs-run strap pitch both visible; heatmap selector works
- Tab 4: **sensor diamonds sit on top of heatmap cells**, not off to the side
- Provenance expander present and populated
- No `st.metric` shows a fallback constant

---

## WHAT NOT TO DO

- Do not restore any form of slack scaling, ramping, or "demo stress mode" to make
  violations reappear.
- Do not tune constants until the output looks impressive.
- Do not silently swallow a missing file or a failed lookup.
- Do not rewrite `app.py` from scratch — the layout, CSS, tab structure, and heatmap
  selector all work and should be preserved.
- Do not modify `paths.csv`, `predictions.csv`, `instances.csv`, `design_stats.csv`,
  `ssd_ctrl_sweep_summary.csv`, or anything in `heatmaps/`.
- Do not add new pip dependencies.

## FINAL DELIVERABLE

1. All files above patched.
2. `verify_prism.py` created and passing, output pasted.
3. Regenerated: `prism_ranked_risk_output.csv`, `prism_tile_attribution.csv`,
   `prism_scenario_expected_risk.csv`, `prism_measured_mitigation_catalog.csv`,
   `prism_mitigation_pareto_front.csv`, `prism_chip_risk_grid.csv`,
   `telemetry_sensor_placements.csv`, `bo_proposals.csv`.
4. A short `CORRECTIONS.md` listing, per defect: what was wrong, what the number was
   before, what it is now, and why the new one is right.
5. An explicit list of anything you could NOT fix and what input you need from
   Role A or Role B to fix it.
