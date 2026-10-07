# PRISM Role C Audit Corrections Report

This document details every defect audited and corrected across the PRISM Role C risk engine, design space exploration (DSE), telemetry sensor placement, and Streamlit dashboard.

---

## Part 1 — `prism_risk_engine.py`

### Defect 1.1: Fabricated Slack Ramp
- **What was wrong**: The loader previously overwrote real STA timing slacks from `paths.csv` with an artificial arithmetic ramp (0.020 ns to 0.600 ns) engineered to manufacture false timing violations.
- **Before**: Fabricated slacks in `[0.020, 0.600]` ns. Dashboard showed 8 Critical Violating Paths.
- **Now**: Uses `df_paths['slack_ns']` exactly as exported by Role A in `paths.csv`. Min slack: `2.16 ns`, Max slack: `11.93 ns`, Mean: `7.2448 ns`. Violations: `0`.
- **Why right**: Physical integrity requires using Role A's true STA timing slack instead of synthetic stress ramps.

### Defect 1.2: 2.2× Droop Multiplier
- **What was wrong**: A 2.2× multiplier was applied to the "stress" scenario and predictions, inflating a 75.9 mV drop into a 150 mV rail collapse.
- **Before**: 2.2× scaled droop (~150 mV).
- **Now**: Converted Volts to millivolts using `* 1000.0` with no magic multipliers.
- **Why right**: Role B's predicted voltage drops are reported in Volts and must not be scaled arbitrarily.

### Defect 1.3: Instance-to-Tile Mapping Anti-Correlated with Physics (`idx % 576`)
- **What was wrong**: Instance IDs up to 28,110 were mapped using `idx % 576`, which correlated negative (r = −0.26) with physical tile locations.
- **Before**: `idx % 576` modulo hashing.
- **Now**: `build_inst_to_tile()` maps cell `(x_um, y_um)` coordinates from `instances.csv` onto the `24 × 24` coarse grid `[0, 575]` using `die_w_um` and `die_h_um` from `design_stats.csv`.
- **Why right**: Timing paths cross physical floorplan tiles; attribution must reflect actual physical placement.

### Defect 1.4: Silent 14→1 Design Collapse
- **What was wrong**: `dict(zip(df_sc['tile_id'], ...))` was used on `predictions.csv` containing 14 designs, silently overwriting all but `syn_013`.
- **Before**: 13 out of 14 designs silently discarded.
- **Now**: When no single design is specified, `predictions.csv` is mean-aggregated across all 14 designs per tile (`df_sc.groupby('tile_id')['pred_v'].mean()`) with exact 576-tile coverage verified and logged.
- **Why right**: Aggregating across designs explicitly accounts for corpus droop distribution rather than discarding data based on row ordering.

### Defect 1.5: Row Misalignment in `run_scenario_weighted_risk`
- **What was wrong**: `.sort_values('effective_slack_ns').reset_index(drop=True)` inside `run_risk_engine` discarded row indices, causing `run_scenario_weighted_risk` to accumulate scenario penalties onto the wrong endpoints.
- **Before**: Penalties attributed to incorrect endpoints.
- **Now**: Refactored `compute_penalties()` to calculate penalties aligned to `df_paths`' original row order without sorting. `run_scenario_weighted_risk` calls `compute_penalties` directly. Verified exact single-scenario equality within `1e-12`.
- **Why right**: Endpoints must match their own physical path penalties.

### Defect 1.6: Hardcoded Supply Voltage & Schema Validation Flag
- **What was wrong**: Nominal VDD was hardcoded to 900 mV despite `design_stats.csv` specifying `vdd_v = 1.1 V`. Negative `delay_ns` values in `paths.csv` were silently hidden with `.abs()`.
- **Before**: Nominal VDD = 900 mV, silent `.abs()` on negative delays.
- **Now**: Reads `vdd_v = 1.1 V` (1100 mV) directly from `design_stats.csv`. Added `validate_paths()` and an explicit `ALLOW_UNVALIDATED_PATHS` flag. When `False`, run stops on schema violations; when `True`, stamps `provenance='UNVALIDATED_INPUT'` into output CSVs and dashboard.
- **Why right**: Supply voltage must match the PDK/SDC specs, and schema anomalies must be explicitly flagged rather than silently swallowed.

### Defect 1.7: Scenario Weighting
- **What was wrong**: All 6 scenarios were given uniform weights (1/6) without checking for mission weights.
- **Before**: Uniform 0.1667 weights.
- **Now**: Loads real `mission_weight` from `activity.csv` if available (asserting sum = 1.0 ± 1e-6). If missing, sets `SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = True` and records `weight_source='uniform_placeholder'` in output CSVs.
- **Why right**: Operational scenarios have unequal mission probabilities.

---

## Part 2 — `app.py` (Dashboard)

### Defect 2.1: Hardcoded Mitigation ROI Caption
- **What was wrong**: Caption hardcoded `39.91 ps` for clock-skew staggering cost = 10.
- **Before**: "Clock-skew staggering yielded the highest ROI delay recovery (39.91 ps for cost = 10)."
- **Now**: Computed dynamically at render time from `df_catalog` (`benefit_ps / cost`, taking argmax).
- **Why right**: Dashboard copy must reflect the underlying CSV data.

### Defect 2.2: Mislabelled KPI Metric
- **What was wrong**: KPI 4 labelled "Optimal Pareto Delay Recovery" showed 1328.2 ps (total slack across all paths) alongside per-path penalty KPI 3.
- **Before**: Misleading side-by-side comparison of total vs per-path metrics.
- **Now**: Relabelled to "Total Slack Recovered (all at-risk paths)" with an explanatory caption stating budget and path count context.
- **Why right**: Clear labeling prevents misinterpreting total budget recovery as single-path delay gain.

### Defect 2.3: Sensor Markers Floating Off Heatmap
- **What was wrong**: Axis tick labels were converted to rounded strings (`'263.8'`), which failed to match numeric sensor coordinates (`263.76`), causing Plotly to append them as new categorical ticks off the map.
- **Before**: Sensor diamonds drawn off the visual heatmap grid.
- **Now**: Numeric float axes (`x_ticks`, `y_ticks`) and float scatter coordinates used throughout `px.imshow` and `go.Scatter`.
- **Why right**: Numeric coordinates enable exact spatial alignment overlay.

### Defect 2.4: DSE Proposal Discrepancy & Single-Entry Legend
- **What was wrong**: Tab 3 plotted `bo_proposals.csv` (strap pitch 8, 8, 8, 8) directly above Role A's actual run summary (strap pitch 12, 24, 24, 16) with a single-color legend based on `ECC_LANES`.
- **Before**: Hidden mismatch between proposed vs executed strap pitches.
- **Now**: Merged `bo_proposals.csv` with `bo_runs.csv` and plotted proposed vs executed strap pitch as a grouped bar chart with an explicit explanatory caption.
- **Why right**: Discrepancies between DSE proposals and executed PnR runs must be transparently displayed.

### Defect 2.5: Unjustified "Provably Optimal" Claims
- **What was wrong**: Dashboard and script claimed "PROVABLY OPTIMAL (Exceeds 63.2% bound)".
- **Before**: Overstated claim confusing coverage percentage with submodular approximation guarantee.
- **Now**: Replaced with honest wording: "Greedy submodular placement — guaranteed within (1−1/e) ≈ 63.2% of optimal K-sensor placement across total attributed chip risk."
- **Why right**: Submodular optimization guarantees a (1−1/e) approximation, not absolute global optimality.

### Defect 2.6: Stale Copy & Fallback Constants
- **What was wrong**: Copy claimed "thousands of floorplan choices" (when Sobol was 512 and Pareto was 5). Missing CSVs rendered silent fake KPI metrics (`105.6 ps`, `72.6 ps`, `1.80 mV`).
- **Before**: Fake fallback values when CSVs were missing.
- **Now**: Counts read dynamically from files (`512 Sobol samples`, `5 Pareto designs`). Missing files render `st.error(...)` naming the missing file and value `—`.
- **Why right**: Silent fallback constants mask missing input files.

### Defect 2.7: Data Provenance & Integrity Panel
- **What was wrong**: No visibility into data origins, path validation, or cross-teammate scale discrepancies.
- **Before**: No provenance summary.
- **Now**: Added collapsible `st.expander("Data provenance & known caveats")` detailing design sources, scenario weights, validation flags, grid size, nominal VDD/clock, zero-violation timing status, and the scale difference between Role B's `syn_*` droop model and Role A's `ssd_ctrl` 1.80 mV PDNSim measurement. Persistent `st.warning` banner shown when `UNVALIDATED_INPUT` is present.
- **Why right**: Full transparency on data provenance and assumptions.

---

## Part 3 — `telemetry_sensor_placement.py`

### Defect 3.1: Tile Risk Grid Accumulation
- **What was wrong**: Tile risk was accumulated using `max()`, considering only the single worst cell per tile.
- **Before**: `risk_grid[gy, gx] = max(risk_grid[gy, gx], risk_ps)`.
- **Now**: `risk_grid[gy, gx] += risk_ps` to sum all attributed cell risks per tile.
- **Why right**: Coverage objectives require total risk density per spatial tile.

### Defect 3.2: Grid Size Mismatch
- **What was wrong**: Defaulted to `(16, 16)` grid while Role B contract uses `(24, 24)` (576 tiles).
- **Before**: 16×16 grid size.
- **Now**: Uses shared constants `NX_COARSE = 24, NY_COARSE = 24` imported from `prism_risk_engine.py`. `prism_chip_risk_grid.csv` is exported as 24×24.
- **Why right**: Telemetry risk grid must align with Role B's coarse tile contract.

### Defect 3.3: Synthetic Fallback Grid
- **What was wrong**: Fabricated hardcoded risk hotspots (`risk_grid[2,3] = 450.0`) when inputs were missing.
- **Before**: Synthetic hotspot fallback.
- **Now**: Raises `FileNotFoundError` naming missing input files.
- **Why right**: Silent fallback data is banned.

### Defect 3.4: Named Sensor Constants & Parameter Export
- **What was wrong**: Sensor count (4) and radius (2.5) were unexplained magic numbers.
- **Before**: Unannotated numeric literals.
- **Now**: Defined as explicit named constants `NUM_SENSORS = 4` and `SENSOR_RADIUS = 2.5` with physical rationales, and exported into `telemetry_sensor_placements.csv`.
- **Why right**: Physical assumptions must be documented and exported for dashboard consumption.

---

## Part 4 — `dse_tier3_bo.py`

### Defect 4.1: GP Fit on Random Uniform Noise
- **What was wrong**: The Gaussian Process was trained on `y = np.random.rand(len(X))`.
- **Before**: `bo_score` was a ranking of random draws.
- **Now**: Fits GP on real measured observations from `ssd_ctrl_sweep_summary.csv` (`NUM_CH`, `DATA_W`, `ECC_LANES`, `CLK_GATE_EN`, `pdn_strap_pitch_um` → `ir_drop_worst_mv`). Exports `bo_proposals.csv` aligned with actual executed runs and sets `method='heuristic_ranking_over_measured_runs'`.
- **Why right**: Optimization models must be trained on real physical objectives, not uniform random numbers.

---

## Part 5 — Cleanup & Deprecation

### Defect 5.1: Superseded Duplicate `risk_engine.py`
- **What was wrong**: Stale copy `risk_engine.py` existed alongside `prism_risk_engine.py`.
- **Before**: Two live engines.
- **Now**: `risk_engine.py` updated with deprecation header and explicit `ImportError` directing callers to `prism_risk_engine.py`. `risk_engine_DEPRECATED.py` created for archival reference.
- **Why right**: Prevents accidental import or execution of stale engine code.

---

## Unresolved Data Discrepancies & Input Needed from Role A / Role B

1. **Role A `paths.csv` Negative Delay Convention & Slack Range**:
   - `paths.csv` currently contains negative values in `delay_ns` (`-8.20` to `-0.24` ns) and slack values up to `11.93 ns` on a `1.0 ns` clock.
   - *Current Mitigation*: Processed under `ALLOW_UNVALIDATED_PATHS = True` with `provenance='UNVALIDATED_INPUT'`.
   - *Input Needed from Role A*: Re-export `paths.csv` with positive path delays and setup slacks normalized to `clock_period_ns`.

2. **Role B `predictions.csv` vs Role A `ssd_ctrl` Measured Droop Scale**:
   - Role B's predictions for `syn_*` designs predict peak droops up to ~75.9 mV (mean timing penalty under `gc_compact` ~86.1 ps). Role A's PDNSim simulation on `orfs_ssd_ctrl_cfg_small` measures a worst IR drop of 1.80 mV (mean timing penalty ~5.33 ps).
   - *Current Mitigation*: Documented explicitly in the dashboard Data Provenance panel. Both data sets are preserved without artificial scaling.
   - *Input Needed from Role A & B*: Perform cross-validation once Role B trains on Role A's specific `orfs_ssd_ctrl` layout.

---

## Additional Defect Fixes

### Defect 1 — `bo_proposals.csv` Contradictory Ranking & ID Binding
- **What was wrong**: `dse_tier3_bo.py` previously sorted candidates by stale `bo_rank` from `ssd_ctrl_sweep_summary.csv` instead of sorting by `bo_score` (`ir_drop_worst_mv`). This caused rank 2 to have worst IR drop `0.7112 mV` while rank 4 had `0.3758 mV`. Furthermore, `proposal_id` was dynamically formatted as `f"cfg_run_{rank_id:02d}"`, which would silently re-assign proposal IDs when order changed.
- **Before**: 
  - Rank 1: `cfg_run_01`, IR drop = `0.3051 mV`
  - Rank 2: `cfg_run_02`, IR drop = `0.7112 mV` (worst)
  - Rank 3: `cfg_run_03`, IR drop = `0.6859 mV`
  - Rank 4: `cfg_run_04`, IR drop = `0.3758 mV` (2nd best)
- **Now**: 
  - Sorted strictly by measured objective `ir_drop_worst_mv` ascending (minimized), with tie-breaking by `power_mw` ascending then `die_area_um2` ascending.
  - `proposal_id` derived directly from `config` column (`cfg_run_01`, `cfg_run_04`, `cfg_run_03`, `cfg_run_02`) to permanently preserve configuration identity tuples.
  - Added columns `rank_basis = 'measured_ir_drop_worst_mv_ascending'` and `legacy_bo_rank`.
  - Final ranking:
    - Rank 1: `cfg_run_01` (0.3051 mV, legacy_bo_rank 1)
    - Rank 2: `cfg_run_04` (0.3758 mV, legacy_bo_rank 4)
    - Rank 3: `cfg_run_03` (0.6859 mV, legacy_bo_rank 3)
    - Rank 4: `cfg_run_02` (0.7112 mV, legacy_bo_rank 2)
- **Why right**: Candidate ranking must be strictly monotonic with respect to the measured objective (`ir_drop_worst_mv`), while configuration tuples must remain permanently bound to their canonical proposal IDs (`cfg_run_01`..`04`).

### Defect 2 — Duplicate Engine File Deletion (`risk_engine.py`)
- **What was wrong**: `risk_engine.py` was a duplicate file containing deprecated code.
- **Before**: `risk_engine.py` and `risk_engine_DEPRECATED.py` both existed in the root directory.
- **Now**: `risk_engine.py` deleted outright. `risk_engine_DEPRECATED.py` retained with explicit deprecation header pointing to `prism_risk_engine.py`.
- **Why right**: Clean single-engine architecture with no orphan files.

---

## Additional Pipeline Fixes (Fixes 1–5)

### Fix 1 — Dynamic At-Risk Set Thresholding (`prism_risk_engine.py`)
- **What was wrong**: The at-risk threshold was previously hardcoded to `min_effective_slack + 0.5 ns`, which selected only 2 paths out of 1,601 when applied to Person A's real slack values (`min = 2.1568 ns`).
- **Before**: Threshold = `2.6568 ns`, selecting `2` paths (0.12% of total).
- **Now**: Implemented `compute_at_risk_mask()` with configurable modes (`AT_RISK_MODE = 'quantile'`). Quantile mode selects the worst 5% of effective slacks (`AT_RISK_QUANTILE = 0.05`), setting threshold = `4.0186 ns` and selecting **`83` at-risk paths** (5.18% of total). Enforces assertions `20 <= at_risk_count <= 400`. Exported `at_risk_threshold_ns`, `at_risk_mode`, and `at_risk_path_count` to catalog and pareto CSVs.
- **Why right**: Scales dynamically with data distribution and ensures statistically meaningful sample size for mitigation evaluation.

### Fix 2 — Rank Churn Metric & Diagnostic (`prism_rank_churn.csv`)
- **What was wrong**: The system lacked empirical tracking of path reordering caused by droop penalties vs plain STA nominal slack ranking.
- **Before**: No rank churn measurement file.
- **Now**: Added `rank_churn(df)` to `prism_risk_engine.py` and exported `prism_rank_churn.csv`. Measured:
  - Total paths: `1,601`
  - Reordered paths (`n_rank_changes`): **`1,403` of 1,601 paths (87.6%)** changed rank position
  - Max rank shift (`max_rank_shift`): **`203` positions**
  - Spearman rank correlation: **`0.9985`**
  - Displayed headline metric and diagnostic info banner on Tab 1 of `app.py`.
- **Why right**: Demonstrates that droop penalties reorder timing priorities among closely-spaced paths.

### Fix 3 — Tab 1 Visualization (Picosecond Penalty Horizontal Bar Chart)
- **What was wrong**: Tab 1 previously displayed side-by-side vertical bar charts of nominal slack (~3.94 ns) vs effective slack (~3.9385 ns), which looked visually indistinguishable.
- **Before**: Grouped bar chart with uninterpretable ~1.5 ps differences on a ~4 ns scale.
- **Now**: Replaced with a horizontal bar chart displaying `droop_penalty_ns * 1000` (picoseconds lost) directly for top 15 paths, sorted descending. Tooltips display nominal and effective slack. Added secondary chart showing penalty as a percentage of each path's own slack.
- **Why right**: Direct visualization of picoseconds lost makes IR-drop timing impact instantly interpretable.

### Fix 4 — Honest DSE Relabeling & Genuine Candidate Generation (`bo_next_candidates.csv`)
- **What was wrong**: `dse_tier3_bo.py` derived proposals from `ssd_ctrl_sweep_summary.csv`, making proposed vs executed strap pitch identical and claiming "Ready for handoff" on executed runs.
- **Before**: Circular proposal chart and inaccurate handoff assertion on executed runs.
- **Now**: 
  - (a) Relabelled Tab 3 to show executed PnR configurations ranked by measured IR drop. Replaced handoff banner with measured verification result (**5.9× worst IR drop reduction**: `1.7985 mV` → `0.3051 mV`, `cfg_small` → `cfg_run_01`).
  - (b) Trained GP surrogate on the 6 real observations and evaluated LCB acquisition over the unobserved candidate pool from `sobol_samples_tier1.csv` (excluding the 6 executed runs). Generated **4 genuinely new unobserved candidates** saved to `bo_next_candidates.csv` (`cfg_next_01`..`04`). Surface `bo_next_candidates.csv` in Tab 3 as the real handoff to Role A.
- **Why right**: Separates historical PnR sweep analysis from new candidate proposals for Role A handoff.

### Fix 5 — Tab 4 Heatmap Dynamic Range Color Mapping
- **What was wrong**: 474 of 576 tiles had $\le 10$ ps risk penalty while peak tile had $474.9$ ps, rendering a linear color scale visually flat.
- **Before**: Linear color mapping rendering nearly the entire die as one solid color.
- **Now**: Applied `np.log1p` transformation for color mapping (`log1p(Risk Penalty ps)`), while preserving raw unscaled picosecond values in hover tooltips and colorbar annotations.
- **Why right**: Exposes dynamic spatial risk variations across all tiles without distorting quantitative hover data.

---

## Round 3 Ingestion & Pipeline Validation Entry (September 4, 2026)

- **Ingested Archive**: `data_orfs_fixes_round3.zip` (`orfs_ssd_ctrl_cfg_small/paths.csv`, `clock_periods.csv`, `design_stats.csv`)
- **Row Count**: 1,601 rows
- **Multi-Clock Gate Result**: **`ACCEPTED`** via `check_new_paths.py` (Multi-clock per-domain validation: `clk_core` = 5.0 ns, `clk_host` = 10.0 ns, `clk_nand` = 20.0 ns).
- **Engine Override Flag**: `ALLOW_UNVALIDATED_PATHS = False` enforced in `prism_risk_engine.py`.
- **Output Provenance**: `provenance = 'VALIDATED'` stamped across all regenerated CSVs (0 `UNVALIDATED` strings in project outputs).

### STEP 3 Metric Comparison Table

| Metric | Pre-Ingestion (Unvalidated) | Post-Ingestion (Validated Round 3) |
|---|---|---|
| **`paths.csv` Min Slack** | `2.16 ns` | `2.16 ns` |
| **`paths.csv` Max Slack** | `11.93 ns` | `11.93 ns` |
| **Negative Slack Paths (`slack_ns < 0`)** | `0` paths | `0` paths |
| **Critical Violating Paths (`effective_slack_ns < 0`)** | `0` paths | `0` paths |
| **Max Per-Path Droop Penalty** | `13.14 ps` | `13.14 ps` |
| **At-Risk Path Threshold & Count** | `4.0186 ns` (**83 paths** / 5.18%) | `4.0186 ns` (**83 paths** / 5.18%) |
| **Rank Changes (`n_rank_changes`)** | `1,403` / 1,601 (**87.63%**) | `1,403` / 1,601 (**87.63%**) |
| **Max Rank Shift** | `203` positions | `203` positions |
| **Spearman Rank Correlation ($\rho$)** | `0.9985` | `0.9985` |
| **Total Slack Recovered @ Budget 50** | `69.60 ps` (`0.0696 ns`) | `69.60 ps` (`0.0696 ns`) |
| **Top-15 STA vs Droop List Identity** | Changed | Changed |

---

## Audit Fixes: Sets 1–7 Summary

### Set 1: Droop Map Physics & PSM Validation
- **Before**: Hardcoded scaling multipliers and unvalidated droop map ingestion.
- **After**: Directly loads measured power grid CSVs (`measured_droop/*`) with automatic `psm_summary.csv` validation (asserts VDD max = 1.7985 mV and VSS max = 2.5622 mV within 1%).
- **Why**: Ensures physics inputs match OpenROAD PSM signoff simulation exactly without magic scaling.

### Set 2: Mitigation Catalog Physics & At-Risk Path Scoping
- **Before**: Fabricated mitigation recovery percentages and fixed 81-path catalog scoping regardless of recovery checks.
- **After**: Recovery checks toggle controls at-risk set (OFF: n=61 paths, 75.8 ps recovery for 12 µm strap fix; ON: n=81 paths, 97.7 ps recovery). Mitigation benefits computed by re-simulating tile-specific droop reduction.
- **Why**: Prevents double-counting recovery check paths and ties mitigation gains directly to physical tile droop reductions.

### Set 3: Scenario-Weighted Risk & Churn Diagnostics
- **Before**: Equal uniform scenario weights without mission context and discordant pair misclassifications.
- **After**: Strict inequality for discordant pair ranking flips (0 flips under measured droop, 1,402 strict rank shifts) and `activity.csv` mission weights loaded for scenario-weighted expected risk.
- **Why**: Correctly identifies subtle path re-orderings under real measured droop without false violation alarms.

### Set 4: Design Space Exploration (DSE Tier 1, Tier 2, Tier 3)
- **Before**: Hardcoded Sobol indices, min-max normalized hypervolume HV ≈ 1.0, and loose hull classifications listing unmeasured architectures as well-supported.
- **After**: Hypervolume normalized to 1.1× per-objective max (HV ≈ 0.48 for 3 non-dominated PnR runs); Tier 3 candidates strictly classified into well-supported (in-hull), one-step extrapolation (8 µm strap), and exploratory (no training support).
- **Why**: Restricts GP-BO surrogate to valid measured knob bounds and provides honest multi-objective hypervolume stats.

### Set 5: Telemetry Extension & Sensor Placement
- **Before**: Inverted heatmap colorscale rendering zero/low tiles red, clipped edge sensors, and ambiguous kernel labels.
- **After**: Monotone Inferno colorscale with transparent NaN empty tiles, log10(1+value) scaling with real ps colorbar ticks, padded die axes [-10, die+10] µm, and explicit kernel labeling.
- **Why**: Accurately visualizes spatial risk distribution and sensor placement without visual distortion.

### Set 6: Model Accuracy & Provenance Expander
- **Before**: Invented ablation rows, hardcoded Vth = 0.35 V, and static audit status string.
- **After**: Computed holdout metrics from `predictions.csv` (MAE = 1.868 mV, R² = 0.9663), dynamically read Vth = 0.30 V from `prism_risk_engine.py`, rendered per-check results from `audit_result.json`, and fixed conformal coverage badge color.
- **Why**: Replaces hardcoded dashboard text with empirical validation metrics and dynamic provenance assertions.

### Set 7: Model Limitations, Documentation & Lock-In Suite
- **Before**: Missing model limitations documentation, stale numbers in README, and loose verification assertions.
- **After**: Added explicit model limitations block to Provenance Expander, updated README/CORRECTIONS, and extended `verify_prism.py` with 13 comprehensive assertions locking in all 7 set fixes.
- **Why**: Permanently prevents regression and documents all physical assumptions and limitations.



