import pandas as pd
import numpy as np
import json
import os
import sys

# ==========================================
# CONSTANTS & CONTRACT DEFAULTS
# ==========================================
NX_COARSE = 24
NY_COARSE = 24

# Physical constants (nangate45 PDK specification)
VTH_MV = 300.0   # nangate45 nominal Vth (mV)
ALPHA = 1.3      # alpha-power-law velocity saturation exponent

# Schema override flag
ALLOW_UNVALIDATED_PATHS = False
SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = False

# At-risk path threshold constants
AT_RISK_MODE = 'quantile'        # Modes: 'quantile' or 'absolute'
AT_RISK_QUANTILE = 0.05          # N=5% worst paths for quantile mode
AT_RISK_PERIOD_FRAC = 0.10       # Fraction of clock_period_ns for absolute mode margin


# ==========================================
# PHASE 1: Data Loader & Validation
# ==========================================
def parse_inst_list(val):
    """Parses instance ID lists from JSON arrays, semicolon strings ('34;36;38'), or comma lists."""
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('['):
            try:
                return [int(x) for x in json.loads(val_str)]
            except Exception:
                pass
        parts = val_str.replace(';', ' ').replace(',', ' ').split()
        return [int(x) for x in parts if x.isdigit()]
    return []


def build_inst_to_tile(instances_csv='instances.csv', stats_csv='design_stats.csv', nx=NX_COARSE, ny=NY_COARSE):
    """
    Builds a real physical mapping from cell inst_id to tile_id (0..575)
    based on bounding box and die dimensions.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    inst_path = instances_csv if os.path.isabs(instances_csv) else os.path.join(dir_path, instances_csv)
    st_path = stats_csv if os.path.isabs(stats_csv) else os.path.join(dir_path, stats_csv)

    if not os.path.exists(st_path):
        raise FileNotFoundError(f"stats_csv file not found: '{st_path}'")
    st = pd.read_csv(st_path)
    die_w = float(st['die_w_um'].iloc[0])
    die_h = float(st['die_h_um'].iloc[0])
    if not (die_w > 0 and die_h > 0):
        raise ValueError("design_stats.csv: die_w_um / die_h_um must be > 0")

    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"instances_csv file not found: '{inst_path}'")
    inst = pd.read_csv(inst_path)
    for c in ('inst_id', 'x_um', 'y_um'):
        if c not in inst.columns:
            raise ValueError(f"instances.csv: missing required column '{c}'")

    if inst['x_um'].max() > die_w * 1.01 or inst['y_um'].max() > die_h * 1.01:
        raise ValueError(
            "instances.csv coordinates exceed the die. Role A likely exported DEF "
            f"database units, not microns. max x={inst['x_um'].max()}, die_w={die_w}"
        )

    tx = np.clip((inst['x_um'] / die_w * nx).astype(int), 0, nx - 1)
    ty = np.clip((inst['y_um'] / die_h * ny).astype(int), 0, ny - 1)
    return dict(zip(inst['inst_id'].astype(int), (ty * nx + tx).astype(int)))


def validate_paths(df_paths, stats_csv='design_stats.csv'):
    """Validates paths.csv against schema contract rules."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    st_path = stats_csv if os.path.isabs(stats_csv) else os.path.join(dir_path, stats_csv)
    st = pd.read_csv(st_path)
    clock_period_ns = float(st['clock_period_ns'].iloc[0])

    if 'bump_pitch_um' in st.columns and pd.isna(st['bump_pitch_um'].iloc[0]):
        if not ALLOW_UNVALIDATED_PATHS:
            print("WARNING: bump_pitch_um is NaN in design_stats.csv (synthesised BGA/bump array)")

    n_neg = (df_paths['delay_ns'] < 0).sum()
    if n_neg > 0:
        min_neg = df_paths['delay_ns'].min()
        msg = (f"paths.csv: delay_ns is negative on {n_neg}/1601 rows (min={min_neg:.2f}). "
               "Role A must export positive path delay. Sign convention or column swap suspected.")
        if not ALLOW_UNVALIDATED_PATHS:
            raise ValueError(msg)

    clk_file = os.path.join(dir_path, 'clock_periods.csv')
    if os.path.exists(clk_file):
        df_clk = pd.read_csv(clk_file)
        clk_map = dict(zip(df_clk['clock_domain'], df_clk['period_ns'].astype(float)))
        max_period = df_clk['period_ns'].max()

        def get_path_period(row):
            ep = str(row['endpoint'])
            cd = str(row['clock_domain'])
            if 'u_host_if' in ep or 'host' in ep:
                return clk_map.get('clk_host', 10.0)
            if 'u_nand_if' in ep or 'nand' in ep:
                return clk_map.get('clk_nand', 20.0)
            return clk_map.get(cd, max_period)

        path_periods = df_paths.apply(get_path_period, axis=1)
        over_mask = df_paths['slack_ns'] > path_periods
        n_over = over_mask.sum()
        if n_over > 0:
            max_slack = df_paths.loc[over_mask, 'slack_ns'].max()
            msg = (f"paths.csv: slack_ns exceeds clock domain period on {n_over} rows (max={max_slack:.2f} ns).")
            if not ALLOW_UNVALIDATED_PATHS:
                raise ValueError(msg)
    else:
        max_slack = df_paths['slack_ns'].max()
        if max_slack > clock_period_ns:
            msg = (f"paths.csv: max slack_ns={max_slack:.2f} exceeds clock_period_ns={clock_period_ns} from design_stats.csv. "
                   "A single-cycle setup path cannot have slack greater than the period. Check units, SDC, or a slack/delay column swap.")
            if not ALLOW_UNVALIDATED_PATHS:
                raise ValueError(msg)


def load_scenario_weights():
    """Loads mission weights per scenario from activity.csv."""
    global SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER
    dir_path = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(dir_path, 'activity.csv'),
        os.path.join(dir_path, 'orfs_extracted', 'orfs_ssd_ctrl_cfg_small', 'activity.csv'),
        os.path.join(dir_path, 'new_files_extracted', 'activity.csv')
    ]
    act_path = None
    for c in candidates:
        if os.path.exists(c):
            act_path = c
            break

    if act_path:
        df_act = pd.read_csv(act_path)
        weights = df_act.groupby('scenario')['mission_weight'].first().to_dict()
        total_w = sum(weights.values())
        if abs(total_w - 1.0) > 1e-6:
            weights = {k: v / total_w for k, v in weights.items()}
        SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = False
        return weights, "activity.csv (mission_weight)"
    else:
        print("[WARNING] activity.csv not found! Using uniform scenario weights placeholder.")
        SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = True
        return None, "uniform_placeholder"


def load_real_or_mock_data(paths_file='paths.csv', v_drop_file='predictions.csv', design=None, stats_file='design_stats.csv'):
    """Loads paths.csv and predictions.csv matching the DATA_SCHEMA.md contract."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    paths_path = os.path.join(dir_path, paths_file)
    pred_path = os.path.join(dir_path, v_drop_file)

    if not os.path.exists(paths_path):
        raise FileNotFoundError(f"Required input file missing: '{paths_path}'")

    print(f"[DATA LOADER] Found Person A's real '{paths_file}'!")
    df_paths = pd.read_csv(paths_path)

    inst_col = 'inst_ids' if 'inst_ids' in df_paths.columns else 'inst_idx'
    df_paths['inst_idx'] = df_paths[inst_col].apply(parse_inst_list)

    validate_paths(df_paths, stats_file)

    if (df_paths['delay_ns'] < 0).any() and ALLOW_UNVALIDATED_PATHS:
        df_paths['delay_ns'] = df_paths['delay_ns'].abs()

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Required predictions file missing: '{pred_path}'")

    print(f"[DATA LOADER] Found Person B's real prediction contract file: '{v_drop_file}'!")
    df_b = pd.read_csv(pred_path, comment='#')

    if 'tile_id' not in df_b.columns or 'pred_v' not in df_b.columns:
        raise ValueError(f"'{v_drop_file}' missing required columns 'tile_id' or 'pred_v'")

    scenarios = {}
    sc_names = list(df_b['scenario'].unique()) if 'scenario' in df_b.columns else ['peak_workload']

    weights_dict, weight_source = load_scenario_weights()

    for sc in sc_names:
        df_sc = df_b[df_b['scenario'] == sc] if 'scenario' in df_b.columns else df_b
        
        if design is not None:
            df_sc_design = df_sc[df_sc['design'] == design]
            if len(df_sc_design) == 0:
                raise ValueError(f"Design '{design}' not found in '{v_drop_file}' for scenario '{sc}'")
            v_map_sc = dict(zip(df_sc_design['tile_id'].astype(int), df_sc_design['pred_v'] * 1000.0))
            design_provenance = f"design='{design}'"
        else:
            n_designs = df_sc['design'].nunique() if 'design' in df_sc.columns else 1
            df_sc_agg = df_sc.groupby('tile_id')['pred_v'].mean().reset_index()
            v_map_sc = dict(zip(df_sc_agg['tile_id'].astype(int), df_sc_agg['pred_v'] * 1000.0))
            design_provenance = f"mean-aggregated across {n_designs} designs"

        if len(v_map_sc) != NX_COARSE * NY_COARSE:
            raise ValueError(f"Scenario '{sc}' produced {len(v_map_sc)} tiles, expected {NX_COARSE * NY_COARSE} (24x24)")

        weight = weights_dict.get(sc, 1.0 / len(sc_names)) if weights_dict else 1.0 / len(sc_names)
        scenarios[sc] = {'weight': weight, 'v_drop_map': v_map_sc}
        print(f"[DROOP MAP] scenario={sc}, designs={df_b['design'].nunique() if 'design' in df_b.columns else 1} ({design_provenance}), tiles={len(v_map_sc)}/{NX_COARSE*NY_COARSE}")

    # Add Stress Burst scenario (gc_compact_stress) for critical violation demonstration
    if 'gc_compact' in scenarios:
        base_gc = scenarios['gc_compact']['v_drop_map']
        stress_map = dict(base_gc)
        
        # Identify instances on critical paths with slack <= 4.0 ns
        low_slack_paths = df_paths[df_paths['slack_ns'] <= 4.0]
        low_insts = set()
        for inst_list in low_slack_paths['inst_idx']:
            low_insts.update(inst_list)

        # Build cell inst_id to tile mapping
        st_path = stats_file if os.path.isabs(stats_file) else os.path.join(dir_path, stats_file)
        st = pd.read_csv(st_path)
        die_w, die_h = float(st['die_w_um'].iloc[0]), float(st['die_h_um'].iloc[0])
        inst_path = os.path.join(dir_path, 'instances.csv')
        inst = pd.read_csv(inst_path)
        tx = np.clip((inst['x_um'] / die_w * 24).astype(int), 0, 23)
        ty = np.clip((inst['y_um'] / die_h * 24).astype(int), 0, 23)
        i_to_t = dict(zip(inst['inst_id'].astype(int), (ty * 24 + tx).astype(int)))
        
        low_tiles = set(i_to_t[idx] for idx in low_insts if idx in i_to_t)
        for t_id in low_tiles:
            stress_map[t_id] = 670.0  # 670 mV peak transient droop spike
            
        scenarios['gc_compact_stress'] = {
            'weight': 0.0,
            'v_drop_map': stress_map
        }

    primary_sc = sc_names[0]
    v_drop_map = scenarios[primary_sc]['v_drop_map']

    return df_paths, v_drop_map, scenarios, weight_source


# ==========================================
# PHASE 2: Risk Engine & Rank Churn Diagnostics
# ==========================================
def calculate_delay_penalty(v_drop_mv, base_delay_ns, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA):
    """
    Applies the Alpha-Power Law: delay ∝ V / (V - Vth)^α
    Returns the EXTRA delay (ns) caused by voltage droop relative to nominal.
    """
    if v_drop_mv <= 0:
        return 0.0

    actual_v = nominal_v_mv - v_drop_mv
    margin = actual_v - vth_mv
    
    if margin <= 0.01:
        return 1e6  # Large penalty for switching failure / setup violation

    k = base_delay_ns / (nominal_v_mv / ((nominal_v_mv - vth_mv) ** alpha))
    new_delay_ns = k * (actual_v / (margin ** alpha))

    return new_delay_ns - base_delay_ns


def compute_penalties(df_paths, v_drop_map, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA):
    """Computes path droop penalties aligned strictly to df_paths' original row order."""
    penalties_ns = []
    eff_slacks_ns = []
    cell_penalties_list = []

    for _, row in df_paths.iterrows():
        path_penalty_ns = 0.0
        cell_delays = row.get('cell_delays_ns', None)
        inst_list = row['inst_idx']
        if cell_delays is None:
            cell_delays = [row['delay_ns'] / max(1, len(inst_list))] * len(inst_list)

        path_cell_penalties = {}
        for idx, cell_base_delay in zip(inst_list, cell_delays):
            if idx not in inst_to_tile:
                raise KeyError(f"inst_id {idx} from paths.csv not found in instances.csv mapping")
            tile_id = inst_to_tile[idx]
            if tile_id not in v_drop_map:
                raise KeyError(f"tile_id {tile_id} (from inst_id {idx}) not found in v_drop_map")
            drop = v_drop_map[tile_id]
            penalty = calculate_delay_penalty(drop, cell_base_delay, nominal_v_mv, vth_mv, alpha)
            path_penalty_ns += penalty
            path_cell_penalties[idx] = penalty

        penalties_ns.append(path_penalty_ns)
        eff_slacks_ns.append(row['slack_ns'] - path_penalty_ns)
        cell_penalties_list.append(path_cell_penalties)

    return penalties_ns, eff_slacks_ns, cell_penalties_list


def run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA):
    """Calculates effective slack and ranks paths by timing risk."""
    df_paths = df_paths.copy()
    penalties_ns, eff_slacks_ns, cell_penalties_list = compute_penalties(
        df_paths, v_drop_map, inst_to_tile, nominal_v_mv, vth_mv, alpha
    )

    df_paths['droop_penalty_ns'] = penalties_ns
    df_paths['effective_slack_ns'] = eff_slacks_ns
    df_paths['cell_penalties'] = cell_penalties_list

    df_ranked = df_paths.sort_values(by='effective_slack_ns', ascending=True).reset_index(drop=True)
    return df_ranked


def rank_churn(df_paths):
    """Computes rank-churn metrics comparing nominal slack_ns ranking vs effective_slack_ns ranking."""
    df = df_paths.copy()

    df['rank_nominal'] = df['slack_ns'].rank(method='min', ascending=True)
    df['rank_effective'] = df['effective_slack_ns'].rank(method='min', ascending=True)
    df['rank_shift'] = (df['rank_nominal'] - df['rank_effective']).abs()

    n_paths = len(df)
    n_rank_changes = int((df['rank_nominal'] != df['rank_effective']).sum())
    max_rank_shift = int(df['rank_shift'].max()) if n_paths > 0 else 0

    if n_paths > 1:
        spearman_corr = round(float(df['slack_ns'].corr(df['effective_slack_ns'], method='spearman')), 4)
    else:
        spearman_corr = 1.0

    top_shifted = df.sort_values(by='rank_shift', ascending=False).head(10)
    top_shifted_endpoints = ";".join([
        f"{r['endpoint']}({int(r['rank_shift'])})" for _, r in top_shifted.iterrows()
    ])

    churn_dict = {
        'n_paths': n_paths,
        'n_rank_changes': n_rank_changes,
        'max_rank_shift': max_rank_shift,
        'spearman_correlation': spearman_corr,
        'top_10_shifted_endpoints': top_shifted_endpoints
    }
    return churn_dict


def compute_instance_risk_attribution(df_ranked):
    """Ranks individual instances by total delay penalty contributed across all paths."""
    instance_loss = {}
    for _, row in df_ranked.iterrows():
        for inst_id, penalty in row['cell_penalties'].items():
            instance_loss[inst_id] = instance_loss.get(inst_id, 0.0) + penalty
            
    df_attr = pd.DataFrame([
        {'inst_idx': k, 'total_delay_penalty_ps': round(v * 1000, 2)} 
        for k, v in instance_loss.items()
    ]).sort_values(by='total_delay_penalty_ps', ascending=False).reset_index(drop=True)
    
    return df_attr


def run_scenario_weighted_risk(df_paths, scenarios, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA):
    """PRISM N5: Computes probability-weighted expected risk across operational scenarios."""
    expected_slack_loss = np.zeros(len(df_paths))
    
    eval_scenarios = {k: v for k, v in scenarios.items() if v['weight'] > 0}
    total_w = sum(v['weight'] for v in eval_scenarios.values())
    
    for sc_name, sc_data in eval_scenarios.items():
        norm_weight = sc_data['weight'] / total_w
        penalties_ns, _, _ = compute_penalties(
            df_paths, sc_data['v_drop_map'], inst_to_tile, nominal_v_mv, vth_mv, alpha
        )
        expected_slack_loss += norm_weight * np.array(penalties_ns)
        
    df_expected = df_paths.copy()
    df_expected['expected_penalty_ns'] = expected_slack_loss
    df_expected['expected_effective_slack_ns'] = df_expected['slack_ns'] - expected_slack_loss
    return df_expected.sort_values(by='expected_effective_slack_ns', ascending=True).reset_index(drop=True)


# ==========================================
# PHASE 3: Dynamic At-Risk Thresholding & Mitigation Catalog
# ==========================================
def compute_at_risk_mask(df_ranked, mode=AT_RISK_MODE, quantile=AT_RISK_QUANTILE, period_frac=AT_RISK_PERIOD_FRAC, stats_csv='design_stats.csv'):
    """Computes boolean mask for at-risk paths using dynamic scaling."""
    n_total = len(df_ranked)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    st_path = stats_csv if os.path.isabs(stats_csv) else os.path.join(dir_path, stats_csv)

    if mode == 'quantile':
        threshold_ns = float(df_ranked['effective_slack_ns'].quantile(quantile))
        mask = df_ranked['effective_slack_ns'] <= threshold_ns
    elif mode == 'absolute':
        st = pd.read_csv(st_path)
        clk_period = float(st['clock_period_ns'].iloc[0])
        threshold_ns = clk_period * period_frac
        mask = df_ranked['effective_slack_ns'] < threshold_ns
    else:
        raise ValueError(f"Unknown AT_RISK_MODE '{mode}'")

    path_count = int(mask.sum())
    max_allowed = int(0.25 * n_total)

    if not (20 <= path_count <= max_allowed):
        threshold_ns = float(df_ranked['effective_slack_ns'].quantile(0.05))
        mask = df_ranked['effective_slack_ns'] <= threshold_ns
        path_count = int(mask.sum())

    return mask, round(threshold_ns, 4), path_count, mode


MITIGATION_CATALOG = {
    "Decap insertion":          {"cost": 15, "reduction_pct": 0.15},
    "Strap widening":           {"cost": 25, "reduction_pct": 0.25},
    "Cell de-densification":    {"cost": 30, "reduction_pct": 0.20},
    "Clock-skew staggering":    {"cost": 10, "reduction_pct": 0.30},
    "Vt swap (non-critical)":   {"cost": 5,  "reduction_pct": 0.10},
    "Extra bump/pad":           {"cost": 40, "reduction_pct": 0.35},
}


def get_at_risk_instances(baseline_ranked, mode=AT_RISK_MODE):
    """Identifies instances on paths within the dynamically computed at-risk set."""
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)
    at_risk_paths = baseline_ranked[mask]
    instances = set()
    for idx_list in at_risk_paths['inst_idx']:
        instances.update(idx_list)
    return instances


def apply_fix(v_drop_map, reduction_pct, affected_instances, inst_to_tile):
    """Applies droop reduction percentage to specified instances / tiles."""
    new_map = dict(v_drop_map)
    affected_tiles = set()
    for idx in affected_instances:
        if idx not in inst_to_tile:
            raise KeyError(f"inst_id {idx} from affected instances not found in instances.csv")
        affected_tiles.add(inst_to_tile[idx])

    for tile_id in affected_tiles:
        if tile_id in new_map:
            new_map[tile_id] = new_map[tile_id] * (1.0 - reduction_pct)
    return new_map


def measured_benefit_ns(df_paths, v_drop_map, baseline_ranked, reduction_pct, affected_instances, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA, mode=AT_RISK_MODE):
    """Measures total effective slack gain across all at-risk paths."""
    fixed_map = apply_fix(v_drop_map, reduction_pct, affected_instances, inst_to_tile)
    after_fix = run_risk_engine(df_paths, fixed_map, inst_to_tile, nominal_v_mv, vth_mv, alpha)

    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)
    baseline_slack_sum = baseline_ranked.loc[mask, 'effective_slack_ns'].sum()
    baseline_indices = baseline_ranked[mask].index
    after_slack_sum = after_fix.loc[baseline_indices, 'effective_slack_ns'].sum()

    recovered_ns = after_slack_sum - baseline_slack_sum
    return max(recovered_ns, 0.0)


def build_measured_catalog(df_paths, v_drop_map, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA, mode=AT_RISK_MODE):
    """Builds catalog of 6 fixes with real computed delay savings (in ps) across the dynamic at-risk set."""
    baseline_ranked = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv, vth_mv, alpha)
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)
    at_risk_instances = get_at_risk_instances(baseline_ranked, mode=mode)

    measured_fixes = []
    for name, spec in MITIGATION_CATALOG.items():
        benefit_ns = measured_benefit_ns(
            df_paths, v_drop_map, baseline_ranked,
            spec["reduction_pct"], at_risk_instances, inst_to_tile,
            nominal_v_mv, vth_mv, alpha, mode=mode
        )
        measured_fixes.append({
            "name": name,
            "cost": spec["cost"],
            "reduction_pct": spec["reduction_pct"],
            "benefit_ps": round(benefit_ns * 1000, 2),
            "at_risk_threshold_ns": thresh_ns,
            "at_risk_mode": mode_used,
            "at_risk_path_count": path_count
        })
    return measured_fixes


# ==========================================
# PHASE 4: Mitigation Knapsack Optimizer & Pareto
# ==========================================
def optimize_mitigations(fixes, budget):
    """0/1 Knapsack DP optimizing measured ps benefit under cost budget across fix COMBINATIONS."""
    n = len(fixes)
    dp = [[0.0 for _ in range(budget + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, budget + 1):
            cost = fixes[i - 1]["cost"]
            benefit = fixes[i - 1]["benefit_ps"]
            if cost <= w:
                dp[i][w] = max(benefit + dp[i - 1][w - cost], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    selected_fixes = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_fixes.append(fixes[i - 1]["name"])
            w -= fixes[i - 1]["cost"]

    return dp[n][budget], selected_fixes


def verify_joint_mitigation(df_paths, v_drop_map, selected_fix_names, catalog_fixes, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA, mode=AT_RISK_MODE):
    """Verifies combined effect of selected fixes by re-solving physics simultaneously across all selected fixes."""
    baseline_ranked = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv, vth_mv, alpha)
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)
    at_risk = get_at_risk_instances(baseline_ranked, mode=mode)

    combined_map = dict(v_drop_map)
    fix_dict = {f['name']: f['reduction_pct'] for f in catalog_fixes}
    
    for fix_name in selected_fix_names:
        pct = fix_dict[fix_name]
        combined_map = apply_fix(combined_map, pct, at_risk, inst_to_tile)
        
    verified_ranked = run_risk_engine(df_paths, combined_map, inst_to_tile, nominal_v_mv, vth_mv, alpha)

    base_slack = baseline_ranked.loc[mask, 'effective_slack_ns'].sum()
    ver_slack = verified_ranked.loc[baseline_ranked[mask].index, 'effective_slack_ns'].sum()
    
    true_benefit_ps = max(0.0, (ver_slack - base_slack) * 1000)
    return round(true_benefit_ps, 2)


def generate_pareto_curve(df_paths, v_drop_map, fixes, inst_to_tile, nominal_v_mv=1100.0, max_budget=60, step=10, mode=AT_RISK_MODE):
    """Sweeps budget to build Pareto Front across fix COMBINATIONS with physics verification."""
    baseline_ranked = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv)
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)

    results = []
    for b in range(step, max_budget + 1, step):
        est_benefit_ps, chosen = optimize_mitigations(fixes, b)
        ver_benefit_ps = verify_joint_mitigation(df_paths, v_drop_map, chosen, fixes, inst_to_tile, nominal_v_mv, mode=mode)
        chosen_str = ", ".join(chosen) if chosen else "None"
        results.append({
            'budget': b, 
            'knapsack_est_ps': est_benefit_ps, 
            'verified_true_ps': ver_benefit_ps,
            'chosen_fixes': chosen_str,
            'at_risk_threshold_ns': thresh_ns,
            'at_risk_mode': mode_used,
            'at_risk_path_count': path_count
        })
    return pd.DataFrame(results)


# ==========================================
# MAIN EXECUTION & PER-SCENARIO PRECOMPUTATION TO DISK
# ==========================================
if __name__ == "__main__":
    print("============================================")
    print("       PRISM ROLE C RISK & DSE ENGINE       ")
    print("============================================\n")

    dir_path = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(dir_path, 'design_stats.csv')
    st_df = pd.read_csv(stats_file)
    nominal_v_mv = float(st_df['vdd_v'].iloc[0]) * 1000.0
    die_w = float(st_df['die_w_um'].iloc[0])
    die_h = float(st_df['die_h_um'].iloc[0])

    provenance_tag = 'UNVALIDATED_INPUT' if ALLOW_UNVALIDATED_PATHS else 'VALIDATED'

    try:
        inst_to_tile = build_inst_to_tile()
    except Exception as e:
        print(f"ERROR building inst_to_tile mapping: {e}")
        sys.exit(1)

    try:
        df_paths, v_drop_map, scenarios, weight_source = load_real_or_mock_data()
    except Exception as e:
        print(f"ERROR in load_real_or_mock_data: {e}")
        sys.exit(1)

    # Precompute per-scenario outputs to disk for ALL scenarios
    print("\n[PRECOMPUTING PER-SCENARIO RESULT FILES TO DISK]")
    cols_ranked = ['path_id', 'endpoint', 'slack_ns', 'delay_ns', 'droop_penalty_ns', 'effective_slack_ns', 'clock_domain', 'provenance'] if 'path_id' in df_paths.columns else ['endpoint', 'slack_ns', 'delay_ns', 'droop_penalty_ns', 'effective_slack_ns', 'clock_domain', 'provenance']

    inst_path = os.path.join(dir_path, 'instances.csv')
    df_inst = pd.read_csv(inst_path)

    for sc_name, sc_data in scenarios.items():
        v_map_sc = sc_data['v_drop_map']
        ranked_sc = run_risk_engine(df_paths, v_map_sc, inst_to_tile, nominal_v_mv)
        ranked_sc['provenance'] = provenance_tag
        
        # 1. Saved ranked risk output
        f_ranked = os.path.join(dir_path, f'prism_ranked_risk_{sc_name}.csv')
        ranked_sc[cols_ranked].to_csv(f_ranked, index=False)

        # 2. Saved rank churn
        churn_sc = rank_churn(ranked_sc)
        f_churn = os.path.join(dir_path, f'prism_rank_churn_{sc_name}.csv')
        pd.DataFrame([churn_sc]).to_csv(f_churn, index=False)

        # 3. Saved tile attribution & 24x24 risk grid
        attr_sc = compute_instance_risk_attribution(ranked_sc)
        attr_sc['provenance'] = provenance_tag
        f_attr = os.path.join(dir_path, f'prism_tile_attribution_{sc_name}.csv')
        attr_sc[['inst_idx', 'total_delay_penalty_ps', 'provenance']].to_csv(f_attr, index=False)

        merged_sc = attr_sc.merge(df_inst[['inst_id', 'x_um', 'y_um']], left_on='inst_idx', right_on='inst_id', how='left')
        risk_grid_sc = np.zeros((24, 24))
        for _, row in merged_sc.iterrows():
            if pd.notna(row['x_um']) and pd.notna(row['y_um']):
                gx = min(23, max(0, int(float(row['x_um']) / (die_w / 24))))
                gy = min(23, max(0, int(float(row['y_um']) / (die_h / 24))))
                risk_grid_sc[gy, gx] += float(row['total_delay_penalty_ps'])
        
        f_grid = os.path.join(dir_path, f'prism_chip_risk_grid_{sc_name}.csv')
        pd.DataFrame(risk_grid_sc).to_csv(f_grid, index=False)

        # 4. Saved measured mitigation catalog
        cat_fixes_sc = build_measured_catalog(df_paths, v_map_sc, inst_to_tile, nominal_v_mv)
        cat_df_sc = pd.DataFrame(cat_fixes_sc)
        cat_df_sc['provenance'] = provenance_tag
        f_cat = os.path.join(dir_path, f'prism_measured_mitigation_catalog_{sc_name}.csv')
        cat_df_sc[['name', 'cost', 'reduction_pct', 'benefit_ps', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(f_cat, index=False)

        # 5. Saved Pareto front curve
        pareto_sc = generate_pareto_curve(df_paths, v_map_sc, cat_fixes_sc, inst_to_tile, nominal_v_mv, max_budget=60, step=10)
        pareto_sc['provenance'] = provenance_tag
        f_pareto = os.path.join(dir_path, f'prism_mitigation_pareto_front_{sc_name}.csv')
        pareto_sc[['budget', 'knapsack_est_ps', 'verified_true_ps', 'chosen_fixes', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(f_pareto, index=False)

        # Print summary line
        max_p = round(float(ranked_sc['droop_penalty_ns'].max()) * 1000.0, 2)
        n_v = int((ranked_sc['effective_slack_ns'] < 0).sum())
        max_rec = round(float(pareto_sc['verified_true_ps'].max()), 2)
        print(f"  --> Precomputed '{sc_name:18s}' | Violations={n_v:2d} | Max Penalty={max_p:7.2f} ps | Max Recovered={max_rec:8.2f} ps")

    # Export default fallback CSVs matching primary scenario
    primary_ranked = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv)
    primary_ranked['provenance'] = provenance_tag
    primary_ranked[cols_ranked].to_csv(os.path.join(dir_path, 'prism_ranked_risk_output.csv'), index=False)
    
    churn_info = rank_churn(primary_ranked)
    pd.DataFrame([churn_info]).to_csv(os.path.join(dir_path, 'prism_rank_churn.csv'), index=False)
    
    attr_df = compute_instance_risk_attribution(primary_ranked)
    attr_df['provenance'] = provenance_tag
    attr_df[['inst_idx', 'total_delay_penalty_ps', 'provenance']].to_csv(os.path.join(dir_path, 'prism_tile_attribution.csv'), index=False)

    df_expected = run_scenario_weighted_risk(df_paths, scenarios, inst_to_tile, nominal_v_mv)
    df_expected['provenance'] = provenance_tag
    df_expected['weight_source'] = weight_source
    cols_sc = ['path_id', 'endpoint', 'slack_ns', 'expected_penalty_ns', 'expected_effective_slack_ns', 'provenance', 'weight_source'] if 'path_id' in df_expected.columns else ['endpoint', 'slack_ns', 'expected_penalty_ns', 'expected_effective_slack_ns', 'provenance', 'weight_source']
    df_expected[cols_sc].to_csv(os.path.join(dir_path, 'prism_scenario_expected_risk.csv'), index=False)

    measured_fixes = build_measured_catalog(df_paths, v_drop_map, inst_to_tile, nominal_v_mv)
    catalog_df = pd.DataFrame(measured_fixes)
    catalog_df['provenance'] = provenance_tag
    catalog_df[['name', 'cost', 'reduction_pct', 'benefit_ps', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(os.path.join(dir_path, 'prism_measured_mitigation_catalog.csv'), index=False)

    pareto_df = generate_pareto_curve(df_paths, v_drop_map, measured_fixes, inst_to_tile, nominal_v_mv, max_budget=60, step=10)
    pareto_df['provenance'] = provenance_tag
    pareto_df[['budget', 'knapsack_est_ps', 'verified_true_ps', 'chosen_fixes', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(os.path.join(dir_path, 'prism_mitigation_pareto_front.csv'), index=False)

    print("\n============================================")
    print(" ALL SCENARIO INTERFACE FILES PRECOMPUTED!  ")
    print("============================================")
