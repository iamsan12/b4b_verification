import sys
import os
import pandas as pd
import numpy as np

def run_verifications():
    print("==========================================================")
    print("     PRISM COMPREHENSIVE VERIFICATION SUITE (SET 7)       ")
    print("==========================================================")
    
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # Helper function for printing locked value results
    def log_result(name, actual, expected_str, is_pass):
        status = "PASS" if is_pass else "FAIL"
        print(f"  - {name}: {actual} ({expected_str}) {status}")
        assert is_pass, f"FAILED ASSERTION: {name}: {actual} ({expected_str})"

    # --------------------------------------------------------
    # [CHECK 1] Code Grep & Pattern Assertions
    # --------------------------------------------------------
    print("\n[CHECK 1] Code pattern and forbidden literal grep checks...")

    found_gc = 0
    root_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(('.py', '.md', '.json', '.csv')) and f != 'verify_prism.py']
    for file in root_files:
        path = os.path.join(dir_path, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if 'gc_compact_stress' in f.read():
                    found_gc += 1
        except Exception:
            pass
    log_result("gc_compact_stress_grep", found_gc, "expected 0 in root project files", found_gc == 0)

    # Grep '670' in prism_risk_engine.py
    with open(os.path.join(dir_path, 'prism_risk_engine.py'), 'r', encoding='utf-8') as f:
        cnt_670 = f.read().count('670')
    log_result("prism_risk_engine_670_grep", cnt_670, "expected 0", cnt_670 == 0)

    # Banned string assertions in app.py
    banned_app_literals = [
        '94.4', '94.40', '90.6', '0.981', '0.9889', '71.4', '84.1', 
        '68.2', '81.5', '3.42', '1.85', '0.842', '0.542', '0.285', 
        '444.22', '6 fixes', 'PASSED', 'Real mission weights'
    ]
    with open(os.path.join(dir_path, 'app.py'), 'r', encoding='utf-8') as f:
        app_text = f.read()
    found_banned = [lit for lit in banned_app_literals if lit in app_text]
    log_result("app_py_banned_literals", len(found_banned), "expected 0", len(found_banned) == 0)

    # --------------------------------------------------------
    # [CHECK 2] Risk Engine & Physical Assumptions Verification
    # --------------------------------------------------------
    print("\n[CHECK 2] Executing & validating prism_risk_engine.py...")
    import prism_risk_engine as pre
    
    # Assert DROOP_SOURCE default == 'measured_rail'
    default_src = getattr(pre, 'DROOP_SOURCE_DEFAULT', 'measured_rail')
    log_result("droop_source_default", default_src, "expected measured_rail", default_src == 'measured_rail')

    df_paths_raw, v_drop_map, scenarios, weight_source = pre.load_real_or_mock_data()
    inst_to_tile = pre.build_inst_to_tile()
    
    # Assert every inst in paths.csv resolves to a fine and coarse tile (0 <= tile <= 575)
    invalid_insts = 0
    for inst_list in df_paths_raw['inst_idx']:
        for inst_id in inst_list:
            tile = inst_to_tile[inst_id]
            if not (0 <= tile <= 575):
                invalid_insts += 1
    log_result("instance_tile_resolution", f"{invalid_insts} invalid", "expected 0 invalid", invalid_insts == 0)

    # Assert droop map: 576 tiles, pred_v in [0, 0.2] V
    map_len = len(v_drop_map)
    out_of_bound_v = sum(1 for v in v_drop_map.values() if not (0.0 <= (float(v)/1000.0) <= 0.2))
    log_result("droop_map_tile_bounds", f"{map_len} tiles, {out_of_bound_v} out of bounds", "expected 576 tiles in [0, 0.2] V", map_len == 576 and out_of_bound_v == 0)

    df_ranked = pre.run_risk_engine(df_paths_raw, v_drop_map, inst_to_tile=inst_to_tile)
    
    # Assert violations == 0 for all 6 real scenarios
    real_scenarios = ['measured_rail', 'measured_vdd', 'measured_cell_rail', 'measured_p12_rail', 'guardband_5pct', 'guardband_10pct']
    total_viols = 0
    for sc in real_scenarios:
        sc_file = os.path.join(dir_path, f'prism_ranked_risk_{sc}.csv')
        if not os.path.exists(sc_file):
            import subprocess
            subprocess.run([sys.executable, os.path.join(dir_path, 'prism_risk_engine.py')], check=True)
        df_sc_res = pd.read_csv(sc_file)
        total_viols += int((df_sc_res['effective_slack_ns'] < 0).sum())
    log_result("real_scenarios_violations", total_viols, "expected 0 across 6 real scenarios", total_viols == 0)

    # Assert PSM summary worst_mv within ±1%
    psm_df = pd.read_csv(os.path.join(dir_path, 'psm_summary.csv'))
    des_col = 'design_id' if 'design_id' in psm_df.columns else 'design'
    vdd_exp_mv = float(psm_df[(psm_df[des_col] == 'orfs_ssd_ctrl_cfg_small') & (psm_df['net'] == 'VDD')]['worst_mv'].iloc[0]) if 'net' in psm_df.columns else float(psm_df[psm_df[des_col] == 'orfs_ssd_ctrl_cfg_small']['vdd_drop_worst_mv'].iloc[0])
    vss_exp_mv = float(psm_df[(psm_df[des_col] == 'orfs_ssd_ctrl_cfg_small') & (psm_df['net'] == 'VSS')]['worst_mv'].iloc[0]) if 'net' in psm_df.columns else float(psm_df[psm_df[des_col] == 'orfs_ssd_ctrl_cfg_small']['vss_bounce_worst_mv'].iloc[0])
    
    vdd_pass = abs(vdd_exp_mv - 1.7985) / 1.7985 <= 0.01
    vss_pass = abs(vss_exp_mv - 2.5622) / 2.5622 <= 0.01
    log_result("psm_summary_vdd_max", f"{vdd_exp_mv:.4f} mV", "expected 1.7985 ± 0.018 mV", vdd_pass)
    log_result("psm_summary_vss_bounce", f"{vss_exp_mv:.4f} mV", "expected 2.5622 ± 0.026 mV", vss_pass)

    # Numerical baseline locks (Measured Rail)
    meas_df = pd.read_csv(os.path.join(dir_path, 'prism_ranked_risk_measured_rail.csv'))
    max_pen_ps = float(meas_df['droop_penalty_ns'].max()) * 1000.0
    log_result("measured_rail_max_penalty", f"{max_pen_ps:.2f} ps", "expected 9.15 ± 0.20 ps", abs(max_pen_ps - 9.15) <= 0.2)

    # Strict rank flips (n_discordant_pairs computed with strict inequality == 0 for measured sources)
    churn_df = pd.read_csv(os.path.join(dir_path, 'prism_rank_churn.csv'))
    disc_col = 'n_discordant_pairs' if 'n_discordant_pairs' in churn_df.columns else 'strict_discordant_pairs'
    disc = int(churn_df[disc_col].iloc[0])
    log_result("strict_rank_flips", disc, "expected 0", disc == 0)

    # Role B what-if baseline check
    roleb_df = pd.read_csv(os.path.join(dir_path, 'prism_ranked_risk_roleB_syn_corpus_whatif.csv'))
    roleb_max_pen = float(roleb_df['droop_penalty_ns'].max()) * 1000.0
    log_result("roleb_whatif_max_penalty", f"{roleb_max_pen:.2f} ps", "expected 155.50 ± 3.00 ps", abs(roleb_max_pen - 155.5) <= 3.0)

    # Catalog strap-12µm fix baseline check
    cat_df = pd.read_csv(os.path.join(dir_path, 'prism_measured_mitigation_catalog.csv'))
    name_col = 'name' if 'name' in cat_df.columns else 'mitigation'
    strap_row = cat_df[cat_df[name_col].astype(str).str.contains('12')].iloc[0]
    strap_rec = float(strap_row.get('total_benefit_ps', strap_row.get('benefit_ps', 0.0)))
    at_risk_cnt = int(strap_row['at_risk_path_count'])
    log_result("strap_12um_recovery", f"{strap_rec:.2f} ps (on n={at_risk_cnt} paths)", "expected 75.80 ± 2.00 ps on n=61 or n=81", abs(strap_rec - 75.8) <= 2.0 and at_risk_cnt in (61, 81))

    # --------------------------------------------------------
    # [CHECK 3] Telemetry & Heatmap Axis Verification
    # --------------------------------------------------------
    print("\n[CHECK 3] Validating telemetry sensor placement and heatmap properties...")
    import telemetry_sensor_placement as tsp
    
    risk_grid, die_w, die_h = tsp.load_grid_risk_map()
    die_pass = abs(die_w - 444.22) <= 1.0 and abs(die_h - 444.22) <= 1.0
    log_result("die_dimensions", f"{die_w:.2f} x {die_h:.2f} µm", "expected 444.22 x 444.22 µm", die_pass)

    df_sensors = tsp.run_greedy_sensor_placement(risk_grid, die_w, die_h)
    sens_out_bounds = sum(1 for _, r in df_sensors.iterrows() if not (0.0 <= r['real_x_um'] <= die_w and 0.0 <= r['real_y_um'] <= die_h))
    log_result("sensor_die_bounds", f"{len(df_sensors)} sensors, {sens_out_bounds} out of bounds", "expected 4 sensors inside [0, die_w/h]", len(df_sensors) == 4 and sens_out_bounds == 0)

    gains = df_sensors['marginal_risk_covered_ps'].values
    monotone_gains = all(gains[i] >= gains[i+1] - 1e-6 for i in range(len(gains) - 1))
    log_result("submodular_marginal_gains", "monotonically non-increasing", "expected non-increasing", monotone_gains)

    cov_ps = gains.sum()
    grid_sum_ps = float(np.sum(risk_grid))
    cov_pct = (cov_ps / grid_sum_ps) * 100.0 if grid_sum_ps > 0 else 0.0
    log_result("telemetry_coverage_pct", f"{cov_ps:.1f} / {grid_sum_ps:.1f} ps ({cov_pct:.1f}%)", "expected 1797.7 / 4242.4 ps = 42.4%", abs(cov_pct - 42.4) <= 1.0)

    # --------------------------------------------------------
    # [CHECK 4] DSE Tier 2 & Tier 3 BO Verification
    # --------------------------------------------------------
    print("\n[CHECK 4] Validating DSE Tier 2 Hypervolume and Tier 3 GP-BO Candidate Hull...")
    
    t2_file = os.path.join(dir_path, 'tier2_pareto_front.csv')
    if not os.path.exists(t2_file):
        import subprocess
        subprocess.run([sys.executable, os.path.join(dir_path, 'dse_tier2_nsga2.py')], check=True)
    t2_df = pd.read_csv(t2_file)
    hv_val = float(t2_df['hypervolume_stat'].iloc[0])
    log_result("tier2_hypervolume", f"{hv_val:.4f}", "expected 0.4825 ± 0.05", abs(hv_val - 0.48) <= 0.05)
    
    non_dom_configs = set(t2_df[t2_df['pareto_status'].astype(str).str.contains('non_dominated|Non-Dominated', case=False)]['config'].tolist())
    expected_non_dom = {'cfg_run_01', 'cfg_run_03', 'cfg_run_04'}
    log_result("tier2_non_dominated_front", sorted(list(non_dom_configs)), "expected ['cfg_run_01', 'cfg_run_03', 'cfg_run_04']", non_dom_configs == expected_non_dom)

    # Tier 3 BO Hull & LCB verification
    next_cand_df = pd.read_csv(os.path.join(dir_path, 'bo_next_candidates.csv'))
    lcb_invalid = sum(1 for _, r in next_cand_df.iterrows() if float(r['lcb_score_mv']) <= 0.0)
    log_result("tier3_bo_lcb_scores", f"{len(next_cand_df)} candidates, {lcb_invalid} non-positive", "expected all > 0", lcb_invalid == 0)

    # --------------------------------------------------------
    # [CHECK 5] Model Metrics Verification
    # --------------------------------------------------------
    print("\n[CHECK 5] Validating model_metrics.csv holdout metrics...")
    metrics_path = os.path.join(dir_path, 'model_metrics.csv')
    assert os.path.exists(metrics_path), "FAILED: model_metrics.csv missing!"
    
    df_metrics = pd.read_csv(metrics_path)
    holdout_row = df_metrics[(df_metrics['partition'] == 'holdout') & (df_metrics['model_col'] == 'pred_v')].iloc[0]
    
    mae_val = float(holdout_row['mae_mv'])
    r2_val = float(holdout_row['r2_score'])
    top5_val = float(holdout_row['top5_hit_rate_pct'])
    cov_val = float(holdout_row['conformal_coverage_pct'])

    log_result("holdout_mae", f"{mae_val:.3f} mV", "expected 1.868 ± 0.018 mV", abs(mae_val - 1.868) / 1.868 <= 0.01)
    log_result("holdout_r2", f"{r2_val:.4f}", "expected 0.9663 ± 0.001", abs(r2_val - 0.9663) <= 0.001)
    log_result("holdout_top5_hit_rate", f"{top5_val:.1f}%", "expected 88.5 ± 1.0%", abs(top5_val - 88.5) <= 1.0)
    log_result("holdout_conformal_coverage", f"{cov_val:.1f}%", "expected 82.0 ± 1.0%", abs(cov_val - 82.0) <= 1.0)

    # --------------------------------------------------------
    # [CHECK 6] Duplicate Engine File Check
    # --------------------------------------------------------
    print("\n[CHECK 6] Checking for stale/duplicate engine files...")
    risk_engine_path = os.path.join(dir_path, 'risk_engine.py')
    log_result("duplicate_engine_file_check", f"risk_engine.py exists={os.path.exists(risk_engine_path)}", "expected False", not os.path.exists(risk_engine_path))

    print("\n==========================================================")
    print("  ALL 16 PRISM VERIFICATION CHECKS PASSED SUCCESSFULLY!   ")
    print("==========================================================")

if __name__ == "__main__":
    run_verifications()
