import sys
import os
import pandas as pd
import numpy as np

def run_verifications():
    print("==========================================================")
    print("       PRISM PIPELINE VERIFICATION SUITE (PART 6)        ")
    print("==========================================================")
    
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # 1. Code Text Checks
    print("\n[CHECK 1] Code pattern grep checks...")
    
    def check_file_not_contains(filepath, substring, label):
        path = os.path.join(dir_path, filepath)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        count = content.count(substring)
        print(f"  - {filepath} '{substring}': count = {count}")
        assert count == 0, f"FAILED: '{substring}' found {count} times in {filepath} ({label})"

    check_file_not_contains('prism_risk_engine.py', '2.2', 'multiplier check')
    check_file_not_contains('prism_risk_engine.py', '% 576', 'modulo grid check')
    check_file_not_contains('app.py', '39.91', 'stale hardcoded ROI caption')
    check_file_not_contains('telemetry_sensor_placement.py', 'PROVABLY OPTIMAL', 'unjustified claim check')
    check_file_not_contains('dse_tier3_bo.py', 'np.random.rand(len(X))', 'fake GP target check')
    
    print("  --> ALL CODE PATTERN CHECKS PASSED!")

    # 2. Run Risk Engine Pipeline
    print("\n[CHECK 2] Executing prism_risk_engine.py...")
    import prism_risk_engine as pre
    
    df_paths_raw, v_drop_map, scenarios, weight_source = pre.load_real_or_mock_data()
    inst_to_tile = pre.build_inst_to_tile()
    
    # Assert tile resolution coverage
    for inst_list in df_paths_raw['inst_idx']:
        for inst_id in inst_list:
            tile = inst_to_tile[inst_id]
            assert 0 <= tile <= 575, f"FAILED: inst_id {inst_id} mapped to tile {tile} outside [0, 575]"

    df_ranked = pre.run_risk_engine(
        df_paths_raw, 
        v_drop_map, 
        inst_to_tile=inst_to_tile
    )
    df_attribution = pre.compute_instance_risk_attribution(df_ranked)
    
    paths_orig = pd.read_csv(os.path.join(dir_path, 'paths.csv'))
    
    # Slack preservation check (Item 1.1)
    slack_orig_set = set(paths_orig['slack_ns'])
    slack_ranked_set = set(df_ranked['slack_ns'])
    assert slack_orig_set == slack_ranked_set, "FAILED: ranked slack_ns is not a permutation of paths.csv slack_ns!"
    assert round(df_ranked['slack_ns'].min(), 4) == round(paths_orig['slack_ns'].min(), 4), f"FAILED: min slack = {df_ranked['slack_ns'].min()}"
    assert round(df_ranked['slack_ns'].max(), 4) == round(paths_orig['slack_ns'].max(), 4), f"FAILED: max slack = {df_ranked['slack_ns'].max()}"
    
    # Honest violations check (0 violations)
    failing_count = (df_ranked['effective_slack_ns'] < 0).sum()
    assert failing_count == 0, f"FAILED: expected 0 failing paths, found {failing_count}"

    # Droop map key coverage check
    assert len(v_drop_map) == 576, f"FAILED: droop map has {len(v_drop_map)} keys, expected 576"
    assert set(v_drop_map.keys()) == set(range(576)), "FAILED: droop map keys do not cover [0, 575]"

    # Endpoint/path alignment regression check (Item 1.5)
    single_sc_dict = {'test_sc': {'weight': 1.0, 'v_drop_map': v_drop_map}}
    df_expected = pre.run_scenario_weighted_risk(df_paths_raw, single_sc_dict, inst_to_tile=inst_to_tile)
    
    merge_cols = ['path_id', 'endpoint'] if 'path_id' in df_expected.columns else ['endpoint']
    merged_check = df_expected.merge(df_ranked, on=merge_cols, suffixes=('_exp', '_rnk'))
    diff = (merged_check['expected_penalty_ns'] - merged_check['droop_penalty_ns']).abs().max()
    assert diff < 1e-12, f"FAILED: expected_penalty_ns does not match droop_penalty_ns by path! max diff={diff}"
    
    print("  --> RISK ENGINE VERIFICATIONS PASSED!")

    # 3. Telemetry Placement Verification
    print("\n[CHECK 3] Executing telemetry_sensor_placement.py...")
    import telemetry_sensor_placement as tsp
    
    risk_grid, die_w, die_h = tsp.load_grid_risk_map()
    grid_df = pd.read_csv(os.path.join(dir_path, 'prism_chip_risk_grid.csv'))
    assert grid_df.shape == (24, 24), f"FAILED: prism_chip_risk_grid.csv shape is {grid_df.shape}, expected (24, 24)"
    
    df_sensors = tsp.run_greedy_sensor_placement(risk_grid, die_w, die_h)
    
    for idx, row in df_sensors.iterrows():
        assert 0.0 <= row['real_x_um'] <= die_w, f"FAILED: sensor x={row['real_x_um']} outside die [0, {die_w}]"
        assert 0.0 <= row['real_y_um'] <= die_h, f"FAILED: sensor y={row['real_y_um']} outside die [0, {die_h}]"
        
    gains = df_sensors['marginal_risk_covered_ps'].values
    for i in range(len(gains) - 1):
        assert gains[i] >= gains[i+1] - 1e-6, f"FAILED: marginal risk gains not non-increasing: {gains}"
        
    print("  --> TELEMETRY VERIFICATIONS PASSED!")

    # 4. Tier 3 BO Verification
    print("\n[CHECK 4] Executing dse_tier3_bo.py...")
    import dse_tier3_bo as bo
    df_proposals = bo.select_top_candidates_bo()
    assert len(df_proposals) == 4, f"FAILED: expected 4 BO proposals, got {len(df_proposals)}"
    assert 'executed_strap_pitch_um' in df_proposals.columns or 'PDN_STRAP_PITCH_UM' in df_proposals.columns
    
    # New Assertion 1: bo_score is monotonically non-decreasing with rank
    scores = df_proposals['bo_score'].values
    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i+1] + 1e-6, f"FAILED: bo_score is not monotonically non-decreasing: {scores}"

    # New Assertion 2: configuration tuple for each proposal_id is unchanged
    expected_tuples = {
        'cfg_run_01': (2, 32, 1, 12),
        'cfg_run_02': (2, 32, 1, 24),
        'cfg_run_03': (2, 32, 4, 24),
        'cfg_run_04': (2, 32, 1, 16)
    }
    for idx, row in df_proposals.iterrows():
        prop_id = row['proposal_id']
        tup = (int(row['NUM_CH']), int(row['DATA_W']), int(row['ECC_LANES']), int(row['PDN_STRAP_PITCH_UM']))
        assert prop_id in expected_tuples, f"FAILED: unexpected proposal_id '{prop_id}'"
        assert tup == expected_tuples[prop_id], f"FAILED: configuration tuple for '{prop_id}' changed! Got {tup}, expected {expected_tuples[prop_id]}"

    print("  --> TIER 3 BO VERIFICATIONS PASSED!")

    # 5. Check app.py Hardcoded Literals Check
    print("\n[CHECK 5] app.py numeric literal check...")
    with open(os.path.join(dir_path, 'app.py'), 'r', encoding='utf-8') as f:
        app_text = f.read()
        
    banned_literals = ['39.91', '105.6', '72.6']
    for lit in banned_literals:
        assert lit not in app_text, f"FAILED: banned numeric literal '{lit}' found in app.py"
    
    print("  --> APP.PY HARDCODED LITERALS CHECK PASSED!")

    # FIX 1 & 2 Assertions: At-risk path count bounds & rank churn file matching
    df_cat = pd.read_csv(os.path.join(dir_path, 'prism_measured_mitigation_catalog.csv'))
    at_risk_count = int(df_cat['at_risk_path_count'].iloc[0])
    n_total_paths = len(df_ranked)
    assert 20 <= at_risk_count <= int(0.25 * n_total_paths), f"FAILED: at_risk_path_count={at_risk_count} outside [20, {int(0.25 * n_total_paths)}]"

    churn_path = os.path.join(dir_path, 'prism_rank_churn.csv')
    assert os.path.exists(churn_path), "FAILED: prism_rank_churn.csv missing!"
    df_churn_ver = pd.read_csv(churn_path)
    assert int(df_churn_ver['n_paths'].iloc[0]) == n_total_paths, "FAILED: rank churn n_paths does not match ranked output!"

    # FIX 4 Assertion: bo_next_candidates.csv contains 4 rows, none of which appear in ssd_ctrl_sweep_summary.csv
    next_path = os.path.join(dir_path, 'bo_next_candidates.csv')
    assert os.path.exists(next_path), "FAILED: bo_next_candidates.csv missing!"
    df_next_ver = pd.read_csv(next_path)
    assert len(df_next_ver) == 4, f"FAILED: expected 4 rows in bo_next_candidates.csv, got {len(df_next_ver)}"

    df_sweep_ver = pd.read_csv(os.path.join(dir_path, 'ssd_ctrl_sweep_summary.csv'))
    def parse_pitch_ver(val):
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)
        if isinstance(val, str) and 'platform' in val:
            return 30.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 24.0

    df_sweep_ver['pitch_clean'] = df_sweep_ver['pdn_strap_pitch_um'].apply(parse_pitch_ver)
    executed_tuples_ver = set()
    for _, row in df_sweep_ver.iterrows():
        tup = (int(row['NUM_CH']), int(row['DATA_W']), int(row['ECC_LANES']), int(row['CLK_GATE_EN']), int(bo.snap_to_grid(row['pitch_clean'])))
        executed_tuples_ver.add(tup)

    for idx, row in df_next_ver.iterrows():
        cand_tup = (int(row['NUM_CH']), int(row['DATA_W']), int(row['ECC_LANES']), int(row['CLK_GATE_EN']), int(row['PDN_STRAP_PITCH_UM']))
        assert cand_tup not in executed_tuples_ver, f"FAILED: candidate {cand_tup} in bo_next_candidates.csv already appears in ssd_ctrl_sweep_summary.csv!"

    # FIX 4 (a) Assertion: app.py contains no st.success asserting handoff that bo_next_candidates.csv does not support
    with open(os.path.join(dir_path, 'app.py'), 'r', encoding='utf-8') as f:
        app_code = f.read()
    if 'st.success(' in app_code:
        for line in app_code.splitlines():
            if 'st.success(' in line:
                assert 'bo_next_candidates' in line, f"FAILED: st.success banner '{line.strip()}' asserts handoff without referencing bo_next_candidates.csv!"

    # 6. Check risk_engine.py does not exist
    print("\n[CHECK 6] Duplicate engine file check...")
    risk_engine_path = os.path.join(dir_path, 'risk_engine.py')
    assert not os.path.exists(risk_engine_path), f"FAILED: 'risk_engine.py' still exists at '{risk_engine_path}'!"
    print("  --> DUPLICATE ENGINE FILE CHECK PASSED!")

    print("\n==========================================================")
    print("  ALL PRISM VERIFICATION CHECKS PASSED SUCCESSFULLY!  ")
    print("==========================================================")

if __name__ == "__main__":
    run_verifications()
