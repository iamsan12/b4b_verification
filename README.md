# PRISM - Power-Integrity Risk & Slack Mitigation (Role C)

PRISM combines Role A's physical-design timing data, Role B's ML-predicted IR-drop, and measured PnR sweeps into a droop-aware timing-risk engine, a mitigation optimizer, a design-space-exploration (DSE) module, a telemetry sensor-placement engine, and a Streamlit dashboard for the SSD controller design.

This README reflects the **final Role A signoff data** integration.

---

## Architecture

```
 paths.csv (Role A)      predictions.csv (Role B)      instances.csv
 timing paths + slack    24x24 IR-drop (Volts)         cell placements (um)
        |                        |                           |
        +------------------------+---------------------------+
                                 v
                      prism_risk_engine.py
        (droop -> delay penalty -> effective slack, per scenario)
                                 |
        +------------------------+-------------------------+
        v                        v                         v
  mitigation catalog     telemetry_sensor_placement.py   dse_tier3_bo.py
  + Pareto (per scenario) (greedy submodular, K=4)       (GP + LCB)
        \                        |                         /
         +-----------------------+------------------------+
                                 v
                              app.py (Streamlit)
```

Measured Role A power-grid data in `measured_droop/` (12 rail/VDD CSVs, one pair per measured PnR run) feeds the droop map used by the risk engine. Scenario weights come from `activity.csv` (`mission_weight`).

---

## Final Role A signoff inputs

### `paths.csv`
Columns: `path_id, endpoint, clock_domain, slack_ns, delay_ns, inst_ids, check_type`

- 1,601 timing path records (endpoint names are **not** unique: 103 repeated endpoint names, so the dashboard calls them *path records*).
- `check_type` distribution: 1,200 `setup`, 400 `recovery`, 1 `clock_gating`.
- Baseline slack range: 2.10 ns to 11.92 ns.
- `check_type` is read cleanly by the engine and is carried through as a path attribute; no formulas depend on it.
- `compare_paths.py` compares an older `paths.csv` against the new one (endpoint overlap, slack differences).

### Other final inputs
`ssd_ctrl_sweep_summary.csv` (6 measured PnR runs), `psm_summary.csv`, `synth_stats.csv`, `measured_droop/`.

### Design constants (`design_stats.csv`)
Die: **444.22 um x 444.22 um**, VDD = 1.1 V, clock period 5.0 ns, 24 x 24 tile grid (about 18.5 um per tile).

## Droop Sources & Signoff Validation

PRISM Role-C supports seven selectable droop sources evaluated against Role A's signoff data:

| Droop Source | Type | Violations | Mean Penalty | Max Penalty | Min Eff. Slack |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `measured_rail` (**default**) | Tile worst VDD drop + VSS bounce | 0 | 2.65 ps | 9.15 ps | 2.097 ns |
| `measured_vdd` | Tile worst VDD drop | 0 | 1.37 ps | 4.49 ps | 2.098 ns |
| `measured_cell_rail` | Per-instance PSM cell rail | 0 | 2.19 ps | 7.38 ps | 2.098 ns |
| `measured_p12_rail` | 12 µm strap pitch PnR run | 0 | 0.25 ps | 0.94 ps | 2.110 ns |
| `guardband_5pct` | Signoff 5% VDD uniform guard-band | 0 | 126.6 ps | 352.1 ps | 1.974 ns |
| `guardband_10pct` | Signoff 10% VDD uniform guard-band | 0 | 272.7 ps | 758.4 ps | 1.828 ns |
| `roleB_syn_corpus_whatif` | Role B 14 syn_* design average | 0 | 47.5 ps | 155.5 ps | 2.064 ns |

### PSM summary validation
The loader automatically validates loaded measured droop CSVs against `psm_summary.csv` (`max(vdd_drop_v)*1000` = 1.7985 mV and `max(vss_bounce_v)*1000` = 2.5622 mV for `orfs_ssd_ctrl_cfg_small`). If the values deviate by more than 1%, validation raises an error.

### Signoff guard-band what-if (`guardband_5pct`)
A uniform droop equal to a named, cited budget fraction of VDD on **every** tile (`GUARDBAND_FRAC = 0.05`, the standard ±5% supply-tolerance budget; 0.10 is also offered). It is labelled as a hypothetical uniform guard-band, not a measured or predicted scenario, and excluded from N5. At 5%: 0 violations, max penalty ≈352 ps, and ≈1,114 rank reorderings because uniform droop penalizes long-delay paths more. Presenting this as a what-if is the honest version of the "droop-aware ranking differs from STA" argument.

---

## What is scenario-dependent vs global

| Item | Scope |
| :--- | :--- |
| Violations, Max Droop, Slack Recovered, risk ranking, rank churn, mitigation catalog + Pareto, risk grid, telemetry sensor placement | **Scenario-dependent** (loaded from per-scenario CSVs) |
| Path Records (1,601) | Global (same path set for all scenarios) |
| N5 scenario weight vs expected-risk chart | Computed at runtime from `activity.csv` weights and the per-scenario ranked-risk CSVs, cross-checked against `prism_scenario_expected_risk.csv` |
| DSE (Tier 1 Sobol, Tier 2 NSGA-II, Tier 3 BO) | Global / workload-independent |
| Accuracy-tab model metrics (Top-5% hit rate, MAE, R^2, ablation) | Static Role B model-level figures supplied as constants in `app.py`; not recomputed here |

---

## Telemetry sensor placement

`telemetry_sensor_placement.py` places K=4 sensors greedily to maximize covered spatial risk (submodular, (1 - 1/e) approximation guarantee for the greedy algorithm). Sensing radius is 2.5 grid units (about 46 um, 101 tiles).

`telemetry_sensor_placements[_<scenario>].csv` columns: `sensor_id, real_x_um, real_y_um, grid_x, grid_y, marginal_risk_covered_ps, num_sensors, sensor_radius_grid_units`.

Baseline Default result (`measured_rail`): 4 sensors cover 1,797.7 ps of 4,242.4 ps total unmitigated chip risk (42.4%); sensor 1 at (286.89 µm, 249.87 µm) covers 934.6 ps. `marginal_risk_covered_ps` is a spatial sum over all tiles in a sensor's radius, so it is larger than any single-path penalty (this is not a unit mismatch).

---

## Tier 3 BO (DSE)

`dse_tier3_bo.py` fits a Gaussian Process on the 6 measured PnR runs in `ssd_ctrl_sweep_summary.csv` and proposes unobserved candidates by LCB (`mu - 1.96 sigma`) with greedy normalized spacing `d >= 0.25`. Outputs: `bo_proposals.csv` (measured runs ranked by measured worst IR drop) and `bo_next_candidates.csv` (4 GP-LCB proposals). With only 6 training observations the surrogate is a small-sample model; treat proposals as suggestions for the next PnR runs.

---

## Dashboard (`app.py`)

Tabs: Risk, Mitigation, DSE, Telemetry, Accuracy. Notes:
- Accuracy tab shows the PRISM spatial timing-risk grid in **ps** for the selected scenario. It is not a ground-truth PDNSim mV map, and no quantile-interval map is shown.
- Provenance notes (Baseline/idle reuse, scenario consistency) are informational.

---

## Legacy / non-runtime files

These files are kept for history only. They are **not** loaded by `app.py` or by the pipeline scripts:
- `paths_LEGACY_unvalidated.csv`
- `risk_engine_results.csv`
- `risk_engine_DEPRECATED.py`

---

## How to run

```bash
pip install numpy pandas scipy scikit-learn streamlit plotly
python prism_risk_engine.py          # regenerates risk, mitigation, Pareto, churn, grids, expected risk
python telemetry_sensor_placement.py # regenerates telemetry placements
python dse_tier3_bo.py               # regenerates BO outputs
python verify_prism.py               # 6 verification checks
python audit.py                      # data-leakage audit (5 checks)
python -m streamlit run app.py       # dashboard at http://localhost:8501
```

`compare_paths.py <old_paths.csv> <new_paths.csv>` is an optional utility for comparing two `paths.csv` versions.
