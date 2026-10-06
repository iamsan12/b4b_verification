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

---

## Scenarios

Eight selectable views, each backed by its own set of generated CSVs (`prism_ranked_risk_<scenario>.csv`, `prism_measured_mitigation_catalog_<scenario>.csv`, `prism_mitigation_pareto_front_<scenario>.csv`, `prism_rank_churn_<scenario>.csv`, `prism_chip_risk_grid_<scenario>.csv`, `telemetry_sensor_placements_<scenario>.csv`):

| View | Notes |
| :--- | :--- |
| Baseline Default | Files without a scenario suffix. Currently identical to `idle`. |
| `idle`, `seq_read`, `seq_write`, `rand_read_4k`, `gc_compact`, `ecc_recover` | Real workload scenarios with mission weights from `activity.csv`. |
| `gc_compact_stress` | **Stress Burst** scenario (see below). |

Final generated values (from the current CSVs):

| Scenario | Violations | Max Droop Penalty |
| :--- | :---: | :---: |
| Baseline Default / `idle` | 0 | 11.6 ps |
| `seq_read` | 0 | 180.7 ps |
| `seq_write` | 0 | 239.2 ps |
| `rand_read_4k` | 0 | 170.0 ps |
| `gc_compact` | 0 | 206.7 ps |
| `ecc_recover` | 0 | 132.4 ps |
| `gc_compact_stress` | **347** | 13,243.9 ps |

### Stress scenario (`gc_compact_stress`)
The stress scenario intentionally applies peak transient current. Its **347 violations are PRISM droop-aware effective-slack violations** (`effective_slack_ns < 0`, where `effective_slack_ns = slack_ns - droop_penalty_ns`) after predicted droop is converted into a timing penalty. They are **not raw baseline STA failures**: the baseline `paths.csv` has no negative slack. The stress scenario has no mission weight and is excluded from the N5 expected-risk aggregation. It demonstrates PRISM's ability to expose burst-induced power-integrity timing risk and should be read as a synthetic stress test, not a predicted field failure rate.

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

Baseline Default result: 4 sensors cover 2,897.5 ps of 6,020.3 ps total unmitigated risk (48.1%); sensor 1 at (286.89 um, 249.87 um) covers 1,434.2 ps. `marginal_risk_covered_ps` is a spatial sum over all tiles in a sensor's radius, so it is larger than any single-path penalty (this is not a unit mismatch).

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
